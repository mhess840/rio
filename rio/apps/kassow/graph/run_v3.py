# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""``check`` and ``run`` for GaP v3 workflows on the Kassow cell.

Called from ``run.main`` when the graph file says ``"version": 3``. A run
writes ``workflow.json`` and ``events.jsonl`` (one executor event per line)
into ``experiments/graph/<time>/``.

Dry-run script (``--script``): a JSON object keyed by node path
(``"cube/find"``). Jev nodes take Jev's ``answers`` object, ``operator.ask``
nodes take the answer string.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rio.apps.kassow.experiments import experiments_root
from rio.apps.kassow.skills import DrySkills
from rio.apps.terminal import term
from rio.graph import Executor, Result, Stopped, Tool, ToolRegistry, Workflow, load_workflow, validate_workflow

from .cell import cell_tools
from .vla import dry_run_vla

if TYPE_CHECKING:
    from .run import Args


def run_workflow(args: Args, path: Path) -> str:
    """Check, dry-run or run one v3 workflow. Returns the run status, or ``ok`` for check.

    Raises:
        WorkflowError: The file breaks the schema.
        WorkflowInvalid: Validation found errors (a run only).
    """
    from .run import load_table, scripted_answers

    workflow = load_workflow(path)
    table = load_table(args) if uses_table(workflow) else None
    term(f"workflow: {workflow.name}    {workflow.description}")
    if args.command == "check":
        issues = validate_workflow(workflow, cell_tools(DrySkills(), _no_jev, table=table))
        for issue in issues:
            term(f"  {issue}")
        failed = [issue for issue in issues if issue.severity == "error"]
        term("check failed" if failed else "check ok")
        return "failed" if failed else "ok"
    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text(path.read_text())
    if args.dry_run:
        canned = scripted_answers(args.script)
        tools = cell_tools(
            DrySkills(),
            _scripted_jev(canned),
            table=table,
            run_vla=dry_run_vla,
            ask_operator=_scripted_operator(canned),
        )
        result = execute(workflow, tools, root)
    else:
        from .run import live_cell

        with live_cell(args, table, root) as cell:

            def ask_operator(node: str, question: str, answers: list[str]) -> str:
                reply = cell.choose(question, answers)
                if reply is None:
                    raise Stopped(f"operator stopped at {node}")
                return reply

            tools = cell_tools(cell.skills, cell.ask, table=table, run_vla=cell.run_vla, ask_operator=ask_operator)

            def confirm(node: str, tool: Tool, inputs: dict[str, Any]) -> bool:
                return cell.confirm(f"{node}: {tool.name} {_brief(inputs)}")

            result = execute(
                workflow,
                tools,
                root,
                confirm=confirm if args.confirm else None,
                should_stop=lambda: "q" in cell.keys.pressed(),
            )
    term(f"{'dry-run ' if args.dry_run else ''}{result.status}: {result.reason}")
    term(f"log: {root}")
    return result.status


def execute(workflow: Workflow, tools: ToolRegistry, root: Path, **kwargs: Any) -> Result:
    """Run ``workflow`` and append every event to ``root/events.jsonl`` as it happens."""
    log = root / "events.jsonl"

    def write(event: dict) -> None:
        with log.open("a") as handle:
            handle.write(json.dumps(event, default=jsonable) + "\n")
        if event["event"] in ("tool_start", "subgraph_start", "route", "end", "tool_error", "subgraph_error"):
            term(f"  {event['event']:<15} {event['node']:<24} {_brief(event, skip=('t', 'event', 'node'))}")

    return Executor(workflow, tools, on_event=write, **kwargs).run()


def uses_table(workflow: Workflow) -> bool:
    """True when a node locates an object (the table map is then required)."""
    scopes = [workflow, *workflow.subgraphs.values()]
    return any(node.tool == "jev.locate" for scope in scopes for node in scope.nodes.values())


def jsonable(value: Any) -> Any:
    """JSON stand-in for values tools return (targets, numpy scalars)."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _brief(data: dict, skip: tuple[str, ...] = ()) -> str:
    parts = []
    for key, value in data.items():
        if key in skip:
            continue
        text = json.dumps(value, default=jsonable)
        parts.append(f"{key}={text if len(text) <= 80 else text[:77] + '...'}")
    return " ".join(parts)


def _no_jev(node: str, questions: dict, note: str) -> dict:
    raise RuntimeError("check does not ask Jev")


def _scripted_jev(canned: dict) -> Any:
    def ask(node: str, questions: dict, note: str) -> dict:
        if node not in canned:
            raise RuntimeError(f"dry-run has no scripted Jev answer for {node!r}")
        return canned[node]

    return ask


def _scripted_operator(canned: dict) -> Any:
    def ask(node: str, question: str, answers: list[str]) -> str:
        reply = canned.get(node)
        if not isinstance(reply, str):
            raise RuntimeError(f"dry-run has no scripted operator answer for {node!r}")
        term(f"[dry run] operator asked {question!r} {answers}: {reply}")
        return reply

    return ask
