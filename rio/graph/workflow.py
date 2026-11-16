# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""GaP v3 workflow files: strict loading and ``$ref`` resolution. No robot code here.

The keys follow the GaP runtime schema (github.com/graph-robots/graph-as-policy,
``docs/runtime.md``) so a file written here also loads there. This cell runs a
subset: ``tool``, ``subgraph``, ``noop`` and ``end`` nodes. ``script`` and
``router`` nodes (code written per graph), streaming nodes and Send fan-out are
refused, because generated code must not drive the arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = 3
START = "START"
END = "END"
IN = "in"
EXIT_KEY = "_exit"
"""Reserved output key of a subgraph node; ``router_field: "exit"`` reads it."""
RESERVED = (START, END, IN)
TOP_TYPES = ("tool", "subgraph", "noop", "end")
INNER_TYPES = ("tool", "noop")
END_STATUSES = ("success", "failure")
UNSUPPORTED_TYPES = {
    "script": "script nodes run code written for one graph; on this cell only registered tools run",
    "router": "router nodes run code written for one graph; route on a tool output with conditional_edges",
    "service": "v2 node type; rewrite as type tool",
    "skill": "v2 node type; rewrite as type tool",
    "policy": "v2 node type; rewrite as type tool",
}

_TOP_KEYS = {"version", "meta", "nodes", "edges", "conditional_edges", "subgraphs"}
_NODE_KEYS = {
    "tool": {"type", "tool", "inputs"},
    "subgraph": {"type", "ref", "inputs"},
    "noop": {"type", "inputs"},
    "end": {"type", "status", "recovery", "inputs"},
}
_SUBGRAPH_KEYS = {"skill", "inputs", "outputs", "nodes", "edges", "conditional_edges", "exit", "on_error", "stage"}


class WorkflowError(ValueError):
    """The file does not follow the v3 schema or uses a part this cell does not run."""


@dataclass(frozen=True)
class Recovery:
    """One best-effort tool call made when the workflow lands on an end node."""

    tool: str
    inputs: dict[str, Any]


@dataclass(frozen=True)
class Node:
    """One node. Only the fields of its ``type`` are set."""

    name: str
    type: str
    tool: str | None = None
    ref: str | None = None
    inputs: dict[str, Any] | None = None
    status: str | None = None
    recovery: tuple[Recovery, ...] = ()


@dataclass(frozen=True)
class Conditional:
    """Route on one output field of the source node (``None``: a router's return value)."""

    router_field: str | None
    mapping: dict[str, str]


@dataclass(frozen=True)
class Scope:
    """Nodes and edges of the top level or of one subgraph."""

    nodes: dict[str, Node]
    edges: tuple[tuple[str, str], ...]
    conditional_edges: dict[str, Conditional]

    def successors(self, name: str) -> list[str]:
        """Static and conditional targets of ``name``, in file order."""
        out = [dst for src, dst in self.edges if src == name]
        cond = self.conditional_edges.get(name)
        if cond is not None:
            out += [dst for dst in cond.mapping.values() if dst not in out]
        return out


@dataclass(frozen=True)
class Exit:
    """How a subgraph names its exit: the terminal node's name, or a field of its output."""

    router_field: str | None
    success_values: tuple[str, ...]


@dataclass(frozen=True)
class Subgraph(Scope):
    """A named part of the task owned by one skill, with declared inputs and outputs."""

    name: str
    skill: str
    inputs: dict[str, str]
    outputs: dict[str, Any]
    exit: Exit
    on_error: str | None
    stage: str | None

    def exit_values(self) -> tuple[str, ...]:
        """Every value the subgraph node may route on."""
        return self.exit.success_values + ((self.on_error,) if self.on_error else ())


@dataclass(frozen=True)
class Workflow(Scope):
    """One task graph in GaP v3 form."""

    meta: dict[str, Any]
    subgraphs: dict[str, Subgraph]
    path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or (self.path.stem if self.path else "workflow"))

    @property
    def description(self) -> str:
        return str(self.meta.get("description") or "")


def is_v3(raw: object) -> bool:
    """True when a parsed JSON file declares ``version: 3``."""
    return isinstance(raw, dict) and raw.get("version") == VERSION


def load_workflow(path: Path) -> Workflow:
    """Read and parse one workflow file.

    Raises:
        WorkflowError: The file breaks the schema or uses an unsupported part.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise WorkflowError(f"{path}: not JSON: {error}") from error
    return parse_workflow(raw, path=path)


def parse_workflow(raw: object, *, path: Path | None = None) -> Workflow:
    """Parse a workflow from a JSON object.

    Raises:
        WorkflowError: The object breaks the schema or uses an unsupported part.
    """
    obj = _object(raw, "workflow")
    _keys(obj, _TOP_KEYS, "workflow")
    if obj.get("version") != VERSION:
        raise WorkflowError(f"workflow: version must be {VERSION}")
    meta = _object(obj.get("meta", {}), "meta")
    nodes = _nodes(obj.get("nodes"), "nodes", TOP_TYPES)
    subgraphs = {name: _subgraph(name, sg) for name, sg in _object(obj.get("subgraphs", {}), "subgraphs").items()}
    return Workflow(
        nodes=nodes,
        edges=_edges(obj.get("edges", []), "edges"),
        conditional_edges=_conditionals(obj.get("conditional_edges", {}), "conditional_edges"),
        meta=meta,
        subgraphs=subgraphs,
        path=path,
    )


def is_ref(value: object) -> bool:
    """True for ``{"$ref": "node.field"}``."""
    return isinstance(value, dict) and set(value) == {"$ref"} and isinstance(value["$ref"], str)


def ref_parts(value: dict) -> list[str]:
    """``{"$ref": "a.b.0"}`` as ``["a", "b", "0"]``."""
    return str(value["$ref"]).split(".")


def refs_in(value: object) -> list[dict]:
    """Every ``$ref`` object inside a literal (lists and dicts walked)."""
    if is_ref(value):
        return [value]  # ty: ignore[invalid-return-type]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in refs_in(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in refs_in(item)]
    return []


def resolve(value: object, outputs: dict[str, Any], bound: dict[str, Any]) -> Any:
    """Replace every ``$ref`` in ``value`` by the data it points to.

    Args:
        value: A literal input, possibly holding refs in lists and dicts.
        outputs: Outputs of the nodes of the current scope.
        bound: Bound inputs of the current scope, addressed as ``in.<name>``.

    Raises:
        KeyError: A ref names a node, field, or index that has no value.
    """
    if is_ref(value):
        parts = ref_parts(value)  # ty: ignore[invalid-argument-type]
        head, rest = parts[0], parts[1:]
        if head == IN:
            if not rest:
                raise KeyError("in needs a field")
            current: Any = bound[rest[0]]
            rest = rest[1:]
        else:
            current = outputs[head]
        for part in rest:
            current = _step(current, part)
        return current
    if isinstance(value, dict):
        return {key: resolve(item, outputs, bound) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, outputs, bound) for item in value]
    return value


def _step(current: Any, part: str) -> Any:
    if isinstance(current, (list, tuple)):
        try:
            return current[int(part)]
        except ValueError as error:
            raise KeyError(f"{part!r} is not a list index") from error
    if isinstance(current, dict):
        return current[part]
    if hasattr(current, part):
        return getattr(current, part)
    raise KeyError(part)


# --- parsing helpers -----------------------------------------------------------


def _object(raw: object, where: str) -> dict:
    if not isinstance(raw, dict):
        raise WorkflowError(f"{where}: must be an object")
    return raw


def _keys(obj: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise WorkflowError(f"{where}: unknown keys {unknown} (allowed: {sorted(allowed)})")


def _nodes(raw: object, where: str, types: tuple[str, ...]) -> dict[str, Node]:
    obj = _object(raw, where)
    if not obj:
        raise WorkflowError(f"{where}: must not be empty")
    return {name: _node(name, node, f"{where}.{name}", types) for name, node in obj.items()}


def _node(name: str, raw: object, where: str, types: tuple[str, ...]) -> Node:
    if name in RESERVED:
        raise WorkflowError(f"{where}: {name!r} is a reserved name")
    obj = _object(raw, where)
    kind = obj.get("type")
    if kind in UNSUPPORTED_TYPES:
        raise WorkflowError(f"{where}: type {kind!r} is not run here: {UNSUPPORTED_TYPES[kind]}")
    if "streaming" in obj:
        raise WorkflowError(f"{where}: streaming nodes are not run on this cell")
    if kind not in types:
        raise WorkflowError(f"{where}: type must be one of {list(types)}, got {kind!r}")
    _keys(obj, _NODE_KEYS[kind], where)
    inputs = obj.get("inputs")
    if inputs is not None:
        _object(inputs, f"{where}.inputs")
    if kind == "tool":
        tool = obj.get("tool")
        if not isinstance(tool, str) or not tool:
            raise WorkflowError(f"{where}: tool node needs a tool name")
        return Node(name, kind, tool=tool, inputs=inputs)
    if kind == "subgraph":
        ref = obj.get("ref")
        if not isinstance(ref, str) or not ref:
            raise WorkflowError(f"{where}: subgraph node needs a ref")
        return Node(name, kind, ref=ref, inputs=inputs)
    if kind == "end":
        status = obj.get("status")
        if status not in END_STATUSES:
            raise WorkflowError(f"{where}: end status must be one of {list(END_STATUSES)}")
        raw_recovery = obj.get("recovery", [])
        if not isinstance(raw_recovery, list):
            raise WorkflowError(f"{where}.recovery: must be a list")
        recovery = tuple(_recovery(item, f"{where}.recovery[{i}]") for i, item in enumerate(raw_recovery))
        return Node(name, kind, status=status, recovery=recovery, inputs=inputs)
    return Node(name, kind, inputs=inputs)


def _recovery(raw: object, where: str) -> Recovery:
    obj = _object(raw, where)
    if "service" in obj or "method" in obj:
        raise WorkflowError(f"{where}: v2 recovery; rewrite as {{'tool': ..., 'inputs': ...}}")
    _keys(obj, {"tool", "inputs"}, where)
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool:
        raise WorkflowError(f"{where}: needs a tool name")
    inputs = _object(obj.get("inputs", {}), f"{where}.inputs")
    if refs_in(inputs):
        raise WorkflowError(f"{where}: recovery inputs are literals, no $ref")
    return Recovery(tool, inputs)


def _edges(raw: object, where: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise WorkflowError(f"{where}: must be a list of [src, dst]")
    edges = []
    for i, edge in enumerate(raw):
        if not (isinstance(edge, list) and len(edge) == 2 and all(isinstance(e, str) for e in edge)):
            raise WorkflowError(f"{where}[{i}]: must be [src, dst]")
        edges.append((edge[0], edge[1]))
    return tuple(edges)


def _conditionals(raw: object, where: str) -> dict[str, Conditional]:
    out = {}
    for src, cond in _object(raw, where).items():
        obj = _object(cond, f"{where}.{src}")
        _keys(obj, {"router_field", "mapping"}, f"{where}.{src}")
        field_ = obj.get("router_field")
        if field_ is not None and not isinstance(field_, str):
            raise WorkflowError(f"{where}.{src}.router_field: must be a string or null")
        mapping = _object(obj.get("mapping"), f"{where}.{src}.mapping")
        if not mapping or not all(isinstance(v, str) for v in mapping.values()):
            raise WorkflowError(f"{where}.{src}.mapping: must map values to node names")
        out[src] = Conditional(field_, dict(mapping))
    return out


def _subgraph(name: str, raw: object) -> Subgraph:
    where = f"subgraphs.{name}"
    obj = _object(raw, where)
    if "exit" in obj and isinstance(obj["exit"], dict) and "values" in obj["exit"]:
        raise WorkflowError(f"{where}.exit: 'values' is retired; use success_values and on_error")
    _keys(obj, _SUBGRAPH_KEYS, where)
    skill = obj.get("skill")
    if not isinstance(skill, str) or not skill:
        raise WorkflowError(f"{where}: needs the owning skill")
    inputs = _object(obj.get("inputs", {}), f"{where}.inputs")
    if not all(isinstance(t, str) for t in inputs.values()):
        raise WorkflowError(f"{where}.inputs: map each input to a type name")
    exit_obj = _object(obj.get("exit"), f"{where}.exit")
    _keys(exit_obj, {"router_field", "success_values"}, f"{where}.exit")
    values = exit_obj.get("success_values")
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise WorkflowError(f"{where}.exit.success_values: must be a list of names")
    router_field = exit_obj.get("router_field")
    if router_field is not None and not isinstance(router_field, str):
        raise WorkflowError(f"{where}.exit.router_field: must be a string or null")
    on_error = obj.get("on_error")
    if on_error is not None and not isinstance(on_error, str):
        raise WorkflowError(f"{where}.on_error: must be a string")
    stage = obj.get("stage")
    return Subgraph(
        nodes=_nodes(obj.get("nodes"), f"{where}.nodes", INNER_TYPES),
        edges=_edges(obj.get("edges", []), f"{where}.edges"),
        conditional_edges=_conditionals(obj.get("conditional_edges", {}), f"{where}.conditional_edges"),
        name=name,
        skill=skill,
        inputs=dict(inputs),
        outputs=dict(_object(obj.get("outputs", {}), f"{where}.outputs")),
        exit=Exit(router_field, tuple(values)),
        on_error=on_error,
        stage=str(stage) if stage is not None else None,
    )
