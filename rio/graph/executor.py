# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a validated v3 workflow one node at a time.

Semantics follow GaP ``docs/runtime.md`` §7, minus parallel super-steps,
streaming and Send (the arm does one thing at a time):

- A scope starts at the targets of ``START`` and follows static edges and the
  conditional edge of each finished node. A conditional edge back to a node
  that already ran is a loop: the loop body runs again.
- A subgraph binds its declared inputs from the latest subgraph that output the
  same name, runs its scope, and reports an exit value. Any exception inside it
  becomes ``on_error`` when declared.
- Reaching an ``end`` node stops the workflow; its recovery calls run best-effort.

Two things never become ``on_error``: the operator declining a motion and the
operator stopping the run. Both end the workflow at once without recovery,
because recovery calls may move the arm.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tools import MOTION, Tool, ToolRegistry
from .validate import errors, validate_workflow
from .workflow import END, EXIT_KEY, START, Node, Scope, Subgraph, Workflow, resolve

Confirm = Callable[[str, Tool, dict[str, Any]], bool]
"""``confirm(node_path, tool, inputs)``: True lets a motion tool run."""
Event = dict[str, Any]


class Stopped(Exception):
    """The operator declined a motion or stopped the run."""


class WorkflowInvalid(ValueError):
    """The workflow failed validation; nothing ran."""


@dataclass
class Result:
    """How a run ended.

    Attributes:
        status: ``success`` or ``failure`` (an end node), ``stopped`` (operator),
            ``error`` (an exception outside any subgraph), ``limit`` (visit cap).
        end: The end node reached, if any.
        reason: One line for the log.
        events: Every event of the run, in order.
    """

    status: str
    end: str | None = None
    reason: str = ""
    events: list[Event] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "success"


class _Limit(Exception):
    pass


class Executor:
    """Walks one workflow with one tool registry.

    Args:
        workflow: The parsed workflow.
        tools: The cell's tools.
        confirm: Asked before every tool tagged ``motion`` (including recovery);
            False stops the run. None runs motions without asking (tests, dry runs).
        should_stop: Polled before every node; True stops the run (the ``q`` key).
        on_event: Receives every event as it happens (trace, live GUI).
        inputs: Initial inputs, addressable as ``in.<name>`` and bound to subgraphs.
        visit_cap: Upper bound on node visits per scope; a loop that never exits ends there.
    """

    def __init__(
        self,
        workflow: Workflow,
        tools: ToolRegistry,
        *,
        confirm: Confirm | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_event: Callable[[Event], None] | None = None,
        inputs: dict[str, Any] | None = None,
        visit_cap: int = 100,
    ) -> None:
        self.workflow = workflow
        self.tools = tools
        self.confirm = confirm
        self.should_stop = should_stop
        self.on_event = on_event
        self.inputs = dict(inputs or {})
        self.visit_cap = visit_cap
        self.events: list[Event] = []
        self._latest: dict[str, Any] = {}

    def run(self) -> Result:
        """Validate, then walk from ``START`` to an end node.

        Raises:
            WorkflowInvalid: Validation found errors; nothing ran.
        """
        problems = errors(validate_workflow(self.workflow, self.tools))
        if problems:
            raise WorkflowInvalid("\n".join(str(p) for p in problems))
        self._emit("run_start", "", name=self.workflow.name, description=self.workflow.description)
        try:
            end, _ = self._scope(self.workflow, "", dict(self.inputs))
        except Stopped as stop:
            return self._finish(Result("stopped", reason=str(stop)))
        except _Limit as limit:
            return self._finish(Result("limit", reason=str(limit)))
        except Exception as error:
            return self._finish(Result("error", reason=f"{type(error).__name__}: {error}"))
        node = self.workflow.nodes[end]
        reason = self._recover(node)
        return self._finish(Result(str(node.status), end=end, reason=reason))

    # --- scopes ---------------------------------------------------------------

    def _scope(self, scope: Scope, prefix: str, bound: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Walk one scope. Returns the terminal node (top level: the end node) and all outputs."""
        outputs: dict[str, Any] = {}
        done: set[str] = set()
        queue = deque(dst for src, dst in scope.edges if src == START)
        visits = 0
        while queue:
            name = queue.popleft()
            visits += 1
            if visits > self.visit_cap:
                raise _Limit(f"{prefix or 'workflow'}: more than {self.visit_cap} node visits")
            if self.should_stop is not None and self.should_stop():
                raise Stopped(f"operator stopped before {prefix}{name}")
            node = scope.nodes[name]
            path = f"{prefix}{name}"
            if node.type == "end":
                return name, outputs
            outputs[name] = self._node(node, path, outputs, bound)
            done.add(name)
            terminal = False
            for src, dst in scope.edges:
                if src != name:
                    continue
                if dst == END:
                    terminal = True
                elif dst not in done and dst not in queue:
                    queue.append(dst)
            cond = scope.conditional_edges.get(name)
            if cond is not None:
                value = self._route_value(node, outputs[name], cond.router_field)
                if value not in cond.mapping:
                    raise RuntimeError(f"{path}: {cond.router_field}={value!r} has no target")
                dst = cond.mapping[value]
                self._emit("route", path, value=value, to=dst)
                if dst == END:
                    terminal = True
                else:
                    if dst in done:
                        done -= _forward(scope, dst)
                    queue.append(dst)
            if terminal:
                return name, outputs
        raise RuntimeError(f"{prefix or 'workflow'}: ran out of nodes without reaching an end")

    @staticmethod
    def _route_value(node: Node, output: Any, router_field: str | None) -> str:
        key = EXIT_KEY if node.type == "subgraph" else router_field
        if not isinstance(output, dict) or key not in output:
            raise RuntimeError(f"node {node.name!r} output has no {key!r} to route on")
        return str(output[key])

    # --- nodes ----------------------------------------------------------------

    def _node(self, node: Node, path: str, outputs: dict[str, Any], bound: dict[str, Any]) -> Any:
        if node.type == "noop":
            self._emit("node", path, type="noop")
            return {}
        if node.type == "subgraph":
            return self._subgraph(node, path, outputs, bound)
        inputs = resolve(node.inputs or {}, outputs, bound)
        return self._call(path, self.tools.get(str(node.tool)), inputs)

    def _call(self, path: str, tool: Tool, inputs: dict[str, Any]) -> dict[str, Any]:
        if MOTION in tool.tags and self.confirm is not None:
            self._emit("confirm", path, tool=tool.name, inputs=inputs)
            if not self.confirm(path, tool, inputs):
                self._emit("declined", path, tool=tool.name)
                raise Stopped(f"operator declined {tool.name} at {path}")
        self._emit("tool_start", path, tool=tool.name, inputs=inputs)
        started = time.monotonic()
        try:
            out = tool.fn(**inputs)
        except Stopped:
            raise
        except Exception as error:
            self._emit("tool_error", path, tool=tool.name, error=f"{type(error).__name__}: {error}")
            raise
        result = dict(out or {})
        self._emit("tool_end", path, tool=tool.name, outputs=result, elapsed_s=round(time.monotonic() - started, 3))
        return result

    def _subgraph(self, node: Node, path: str, outputs: dict[str, Any], bound: dict[str, Any]) -> dict[str, Any]:
        sg = self.workflow.subgraphs[str(node.ref)]
        given = resolve(node.inputs or {}, outputs, bound)
        inner: dict[str, Any] = {}
        for name in sg.inputs:
            if name in given:
                inner[name] = given[name]
            elif name in self._latest:
                inner[name] = self._latest[name]
            elif name in self.inputs:
                inner[name] = self.inputs[name]
            else:
                raise RuntimeError(f"{path}: no value for input {name!r}")
        self._emit("subgraph_start", path, subgraph=sg.name, skill=sg.skill, inputs=inner)
        started = time.monotonic()
        error_path = False
        try:
            terminal, inner_out = self._scope(sg, f"{path}/", inner)
            exit_value = self._exit_value(sg, terminal, inner_out)
        except (Stopped, _Limit):
            raise
        except Exception as error:
            if sg.on_error is None:
                raise
            self._emit("subgraph_error", path, error=f"{type(error).__name__}: {error}")
            exit_value, inner_out, error_path = sg.on_error, {}, True
        if exit_value not in sg.exit_values():
            raise RuntimeError(f"{path}: exit {exit_value!r} is not in {list(sg.exit_values())}")
        bound_out: dict[str, Any] = {}
        if not error_path:
            for name, binding in sg.outputs.items():
                try:
                    bound_out[name] = resolve(binding, inner_out, inner)
                except (KeyError, IndexError):
                    continue
        self._latest.update(bound_out)
        self._emit(
            "subgraph_end",
            path,
            subgraph=sg.name,
            exit=exit_value,
            outputs=bound_out,
            error_path=error_path,
            elapsed_s=round(time.monotonic() - started, 3),
        )
        return {**bound_out, EXIT_KEY: exit_value}

    @staticmethod
    def _exit_value(sg: Subgraph, terminal: str, outputs: dict[str, Any]) -> str:
        if sg.exit.router_field is None:
            return terminal
        out = outputs.get(terminal)
        if not isinstance(out, dict) or sg.exit.router_field not in out:
            raise RuntimeError(f"{sg.name}: terminal {terminal!r} has no {sg.exit.router_field!r}")
        return str(out[sg.exit.router_field])

    def _recover(self, node: Node) -> str:
        self._emit("end", node.name, status=node.status)
        failed = []
        for i, call in enumerate(node.recovery):
            path = f"{node.name}/recovery[{i}]"
            try:
                self._call(path, self.tools.get(call.tool), dict(call.inputs))
            except Stopped:
                failed.append(f"{call.tool} declined")
                break
            except Exception as error:
                failed.append(f"{call.tool}: {error}")
        reason = f"{node.status} at {node.name}"
        return reason + (f" (recovery: {'; '.join(failed)})" if failed else "")

    # --- events ---------------------------------------------------------------

    def _emit(self, kind: str, path: str, **data: Any) -> None:
        event = {"t": round(time.time(), 3), "event": kind, "node": path, **data}
        self.events.append(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # a broken viewer must not stop the arm
                pass

    def _finish(self, result: Result) -> Result:
        self._emit("run_end", "", status=result.status, end=result.end, reason=result.reason)
        result.events = self.events
        return result


def _forward(scope: Scope, start: str) -> set[str]:
    """``start`` and every node reachable from it: the body a back edge re-runs."""
    seen: set[str] = set()
    queue = deque([start])
    while queue:
        name = queue.popleft()
        if name in seen or name not in scope.nodes:
            continue
        seen.add(name)
        queue.extend(scope.successors(name))
    return seen
