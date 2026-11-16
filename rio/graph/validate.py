# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural checks of a v3 workflow before anything runs.

Rule ids follow GaP ``docs/runtime.md`` §5 (W1-W8, S1-S11). Rules marked ``R``
are this cell's additions: every exit a tool or subgraph can report has a
target, inputs match the tool's declaration, a failure end has a recovery,
and the arm never fans out in parallel.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .tools import ANY, ToolRegistry, types_match
from .workflow import END, EXIT_KEY, IN, START, Scope, Subgraph, Workflow, ref_parts, refs_in

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One finding. ``where`` is a dotted path into the file."""

    severity: str
    rule: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity} {self.rule}] {self.where}: {self.message}"


def errors(issues: list[Issue]) -> list[Issue]:
    """Only the issues that block a run."""
    return [issue for issue in issues if issue.severity == ERROR]


def validate_workflow(workflow: Workflow, tools: ToolRegistry | None = None) -> list[Issue]:
    """Every issue of ``workflow``. With ``tools``, tool names and inputs are checked too.

    Args:
        workflow: A parsed workflow.
        tools: The cell's registry; without it only the structure is checked.

    Returns:
        Issues, errors first.
    """
    out: list[Issue] = []
    _check_scope(workflow, "", out, top=True)
    if not any(node.type == "end" for node in workflow.nodes.values()):
        out.append(Issue(ERROR, "W", "nodes", "needs at least one end node"))
    produced = {name for sg in workflow.subgraphs.values() for name in sg.outputs}
    for name, node in workflow.nodes.items():
        where = f"nodes.{name}"
        if node.type == "subgraph":
            sg = workflow.subgraphs.get(str(node.ref))
            if sg is None:
                out.append(Issue(ERROR, "W7", where, f"ref {node.ref!r} is not a declared subgraph"))
                continue
            for input_name in sg.inputs:
                given = node.inputs is not None and input_name in node.inputs
                if not given and input_name not in produced:
                    out.append(Issue(ERROR, "W8", where, f"input {input_name!r} of {sg.name} is not an output of any subgraph"))
            _check_routes(workflow, name, sg.exit_values(), where, out)
        if node.type == "end" and node.status == "failure" and not node.recovery:
            out.append(Issue(WARNING, "R", where, "failure end without recovery; the gripper stays as it is"))
        if node.type == "end" and tools is not None:
            for i, call in enumerate(node.recovery):
                if call.tool not in tools:
                    out.append(Issue(ERROR, "R", f"{where}.recovery[{i}]", f"unknown tool {call.tool!r}"))
    for sg in workflow.subgraphs.values():
        _check_subgraph(sg, out)
    if tools is not None:
        _check_tools(workflow, workflow, "", tools, out)
        for sg in workflow.subgraphs.values():
            _check_tools(workflow, sg, f"subgraphs.{sg.name}.", tools, out)
    return sorted(out, key=lambda issue: issue.severity != ERROR)


def _check_scope(scope: Scope, prefix: str, out: list[Issue], *, top: bool) -> None:
    w = "W" if top else "S"
    names = set(scope.nodes)
    if not any(src == START for src, _ in scope.edges):
        out.append(Issue(ERROR, f"{w}{2 if top else 1}", f"{prefix}edges", "needs an edge from START"))
    for i, (src, dst) in enumerate(scope.edges):
        where = f"{prefix}edges[{i}]"
        if src not in names and src != START:
            out.append(Issue(ERROR, f"{w}3" if top else "S5", where, f"source {src!r} is not a node"))
        if dst == END and top:
            out.append(Issue(ERROR, "W3", where, "the top level ends on an end node, not END"))
        elif dst not in names and dst != END:
            out.append(Issue(ERROR, f"{w}3" if top else "S5", where, f"target {dst!r} is not a node"))
        if dst == START:
            out.append(Issue(ERROR, f"{w}3", where, "START cannot be a target"))
    for src, cond in scope.conditional_edges.items():
        where = f"{prefix}conditional_edges.{src}"
        if src not in names:
            out.append(Issue(ERROR, "W6", where, f"source {src!r} is not a node"))
            continue
        for value, dst in cond.mapping.items():
            if dst == END and top:
                out.append(Issue(ERROR, "W6", where, f"{value!r} -> END; route to an end node"))
            elif dst not in names and dst != END:
                out.append(Issue(ERROR, "W6", where, f"{value!r} -> {dst!r} is not a node"))
        node = scope.nodes[src]
        if node.type == "subgraph" and cond.router_field != "exit":
            out.append(Issue(ERROR, "S8", where, "a subgraph node routes on router_field 'exit'"))
        elif node.type != "subgraph" and cond.router_field is None:
            out.append(Issue(ERROR, "S8", where, "needs a router_field"))
    for src in [START, *names]:
        fan = [dst for s, dst in scope.edges if s == src]
        if len(fan) > 1:
            out.append(Issue(ERROR, "R", f"{prefix}nodes.{src}", f"parallel fan-out to {fan}; the arm runs one node at a time"))
        if src in scope.conditional_edges and fan:
            out.append(Issue(ERROR, "R", f"{prefix}nodes.{src}", "has both a static and a conditional edge"))
        node = scope.nodes.get(src)
        if node is not None and node.type != "end" and not fan and src not in scope.conditional_edges:
            out.append(Issue(ERROR, f"{w}5" if top else "S", f"{prefix}nodes.{src}", "has no outgoing edge"))
    reached = _reachable(scope)
    for name in sorted(names - reached):
        out.append(Issue(ERROR, f"{w}4" if top else "S2", f"{prefix}nodes.{name}", "is not reachable from START"))


def _reachable(scope: Scope) -> set[str]:
    seen: set[str] = set()
    queue = deque(dst for src, dst in scope.edges if src == START)
    while queue:
        name = queue.popleft()
        if name in seen or name not in scope.nodes:
            continue
        seen.add(name)
        queue.extend(scope.successors(name))
    return seen


def _check_subgraph(sg: Subgraph, out: list[Issue]) -> None:
    prefix = f"subgraphs.{sg.name}."
    _check_scope(sg, prefix, out, top=False)
    names = set(sg.nodes)
    if not sg.exit.success_values:
        out.append(Issue(ERROR, "S7", f"{prefix}exit", "success_values must not be empty"))
    if sg.on_error is not None:
        if sg.on_error in names:
            out.append(Issue(ERROR, "S9", f"{prefix}on_error", f"{sg.on_error!r} is also a node name"))
        if any(sg.on_error in cond.mapping.values() for cond in sg.conditional_edges.values()):
            out.append(Issue(ERROR, "S10", f"{prefix}on_error", "is a conditional-edges target"))
    terminals = {src for src, dst in sg.edges if dst == END}
    terminals |= {src for src, cond in sg.conditional_edges.items() if END in cond.mapping.values()}
    if sg.exit.router_field is None:
        for value in sg.exit.success_values:
            node = sg.nodes.get(value)
            if node is None or node.type != "noop":
                out.append(Issue(ERROR, "S11", f"{prefix}exit", f"success value {value!r} must be a noop node"))
            elif value not in terminals:
                out.append(Issue(ERROR, "S11", f"{prefix}exit", f"success node {value!r} needs an edge to END"))
        for name in sorted(terminals - set(sg.exit.success_values)):
            out.append(Issue(ERROR, "S11", f"{prefix}nodes.{name}", "ends the subgraph but is not a success value"))
    else:
        for value in sg.exit.success_values:
            if value in names:
                out.append(Issue(ERROR, "S11", f"{prefix}exit", f"success value {value!r} collides with a node name"))
    for out_name, binding in sg.outputs.items():
        for ref in refs_in(binding) or [None]:
            if ref is None:
                out.append(Issue(ERROR, "S6", f"{prefix}outputs.{out_name}", "must be a $ref"))
            elif ref_parts(ref)[0] not in names:
                out.append(Issue(ERROR, "S6", f"{prefix}outputs.{out_name}", f"{ref['$ref']!r} names no node"))
    for name, node in sg.nodes.items():
        for ref in refs_in(node.inputs or {}):
            parts = ref_parts(ref)
            if parts[0] == IN:
                if len(parts) < 2 or parts[1] not in sg.inputs:
                    out.append(Issue(ERROR, "S5", f"{prefix}nodes.{name}", f"{ref['$ref']!r} is not a declared input"))
            elif parts[0] not in names:
                out.append(Issue(ERROR, "S5", f"{prefix}nodes.{name}", f"{ref['$ref']!r} names no node in {sg.name}"))


def _check_routes(workflow: Workflow, name: str, values: tuple[str, ...], where: str, out: list[Issue]) -> None:
    cond = workflow.conditional_edges.get(name)
    if cond is not None:
        _check_routes_scope(cond.mapping, values, where, out)


def _check_tools(workflow: Workflow, scope: Scope, prefix: str, tools: ToolRegistry, out: list[Issue]) -> None:
    for name, node in scope.nodes.items():
        where = f"{prefix}nodes.{name}"
        if node.type != "tool":
            continue
        if node.tool not in tools:
            out.append(Issue(ERROR, "R", where, f"unknown tool {node.tool!r}"))
            continue
        tool = tools.get(str(node.tool))
        given = node.inputs or {}
        for key in given:
            if key not in tool.inputs:
                out.append(Issue(ERROR, "R", where, f"{tool.name} has no input {key!r} (inputs: {sorted(tool.inputs)})"))
        for key, param in tool.inputs.items():
            if param.required and key not in given:
                out.append(Issue(ERROR, "R", where, f"{tool.name} needs input {key!r}"))
            if key in given:
                for ref in refs_in(given[key]) if given[key] is not None else []:
                    produced = _ref_type(workflow, scope, ref_parts(ref), tools)
                    if produced is not None and given[key] == ref and not types_match(produced, param.type):
                        out.append(Issue(ERROR, "R", where, f"input {key!r} wants {param.type}, {ref['$ref']!r} is {produced}"))
        cond = scope.conditional_edges.get(name)
        if cond is not None and cond.router_field is not None:
            if tool.outputs and cond.router_field not in tool.outputs:
                out.append(Issue(ERROR, "R", where, f"{tool.name} has no output {cond.router_field!r} to route on"))
            if cond.router_field == "exit" and tool.exits:
                _check_routes_scope(cond.mapping, tool.exits, where, out)


def _check_routes_scope(mapping: dict[str, str], values: tuple[str, ...], where: str, out: list[Issue]) -> None:
    for value in values:
        if value not in mapping:
            out.append(Issue(ERROR, "R", where, f"exit {value!r} has no target in conditional_edges"))
    for value in mapping:
        if value not in values:
            out.append(Issue(ERROR, "R", where, f"routes {value!r}, which it never reports (it reports {list(values)})"))


def _ref_type(workflow: Workflow, scope: Scope, parts: list[str], tools: ToolRegistry) -> str | None:
    """Type name a ref points at, when it can be told without running; else None."""
    if parts[0] == IN:
        if isinstance(scope, Subgraph) and len(parts) == 2:
            return scope.inputs.get(parts[1])
        return None
    node = scope.nodes.get(parts[0])
    if node is None:
        return None
    if node.type == "tool" and node.tool in tools and len(parts) == 2:
        return tools.get(str(node.tool)).outputs.get(parts[1], ANY)
    if node.type == "subgraph" and len(parts) == 2:
        if parts[1] == EXIT_KEY:
            return "str"
        sg = workflow.subgraphs.get(str(node.ref))
        if sg is not None:
            for ref in refs_in(sg.outputs.get(parts[1])):
                return _ref_type(workflow, sg, ref_parts(ref), tools)
    return None
