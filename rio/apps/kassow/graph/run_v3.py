# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""``check`` and ``run`` for GaP v3 workflows on the Kassow cell.

Called from ``run.main`` when the graph file says ``"version": 3``. A run
writes ``workflow.json`` and ``events.jsonl`` (one executor event per line)
into ``experiments/graph/<time>/``.

Dry-run script (``--script``): a JSON object keyed by node path
(``"cube/find"``). Jev nodes take Jev's ``answers`` object, ``sam.locate``
nodes a map of prompt to ``[[x1, y1, x2, y2, score]]`` in 640x480 pixels,
``operator.ask`` nodes the answer string.
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
        dry = DrySkills()
        tools = cell_tools(
            dry,
            _scripted_jev(canned),
            table=table,
            run_vla=dry_run_vla,
            ask_operator=_scripted_operator(canned),
            **_dry_perception(args, canned),
            **_dry_wrist(dry),
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

            from rio.sam import SamClient

            tools = cell_tools(
                cell.skills,
                cell.ask,
                table=table,
                run_vla=cell.run_vla,
                ask_operator=ask_operator,
                photo=cell.photo,
                segment=SamClient().segment,
                handeye=_handeye(),
                views=station_views(cell.env, root),
                wrist=_wrist(),
            )

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
    return any(
        node.tool in ("jev.locate", "sam.locate", "kassow.search", "kassow.look_at")
        for scope in scopes
        for node in scope.nodes.values()
    )


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


def _dry_perception(args: Args, canned: dict) -> dict:
    """``--image``: real SAM on that saved Jev photo (no arm moves); else the scripted answers."""
    if args.image is None:
        return {"photo": lambda node: b"", "segment": scripted_sam(canned)}
    import numpy as np

    from rio.sam import SamClient

    from .cell import Shot

    jpeg = args.image.read_bytes()
    depth_file = args.image.with_suffix(".depth.npz")
    depth = np.load(depth_file)["depth_mm"].astype(np.float32) / 1000.0 if depth_file.is_file() else None
    term(f"[dry run] sam.locate uses SAM on {args.image}" + (" with its depth" if depth is not None else ""))
    shot = Shot(jpeg, depth)
    return {"photo": lambda node: shot, "segment": SamClient().segment, "handeye": _handeye() if depth is not None else None}


def station_views(env: Any, root: Path) -> dict[str, Any]:
    """Photo functions of the overview and wrist cameras the station already streams, with their poses.

    The overview calibration is fixed; the wrist camera's pose is the TCP pose of the photo times its
    mount, so a wrist photo is taken only once the arm stands still (the executor runs one node at a time).
    """
    from rio.apps.kassow.jev_ask import fresh_frame
    from rio.jev import encode_jpeg

    from .cell import Shot
    from .handeye import HandEye, load_handeye

    views: dict[str, Any] = {}
    try:
        overview = load_handeye(camera="overview")
    except (OSError, ValueError, KeyError):
        overview = None
    wrist = _wrist()

    def keep(node: str, camera: str, jpeg: bytes) -> None:
        (root / f"{node.replace('/', '__')}.{camera}.jpg").write_bytes(jpeg)

    if overview is not None and "camera_2" in env.cameras:

        def overview_photo(node: str) -> Shot:
            jpeg = encode_jpeg(fresh_frame(env.cameras["camera_2"], 0.5))
            keep(node, "overview", jpeg)
            return Shot(jpeg, None, overview)

        views["overview"] = overview_photo
    if wrist is not None and "camera_1" in env.cameras:

        def wrist_photo(node: str) -> Shot:
            from rio_hw import time

            time.sleep(0.3)  # let the arm settle after the last move
            jpeg = encode_jpeg(fresh_frame(env.cameras["camera_1"], 0.3))
            eef = env.robot.arm.get_state()["eef_pose"]
            pose = wrist.base_from_camera(eef)
            keep(node, "wrist", jpeg)
            return Shot(jpeg, None, HandEye(wrist.camera_serial, wrist.intrinsics, pose.tolist(), {}, {}, {}))

        views["wrist"] = wrist_photo
    return views


def _dry_wrist(dry: Any) -> dict:
    """Dry runs: the fixed overview camera and a wrist camera posed where the dry arm was last told to look
    (with scripted SAM answers)."""
    wrist = _wrist()
    if wrist is None:
        return {}
    from .cell import Shot
    from .handeye import HandEye, load_handeye

    def photo(node: str) -> Shot:
        pose = getattr(dry, "last_tcp", None)
        if pose is None:
            raise RuntimeError("dry run: the wrist camera has not been moved to a view yet")
        return Shot(
            b"", None, HandEye(wrist.camera_serial, wrist.intrinsics, wrist.base_from_camera(pose).tolist(), {}, {}, {})
        )

    views: dict[str, Any] = {"wrist": photo}
    try:
        overview = load_handeye(camera="overview")
        views["overview"] = lambda node: Shot(b"", None, overview)
    except (OSError, ValueError, KeyError):
        pass
    return {"views": views, "wrist": wrist}


def _wrist() -> Any:
    from .wristeye import load_wristeye

    try:
        return load_wristeye()
    except (OSError, ValueError, KeyError):
        return None


def _handeye() -> Any:
    """The Jev camera's hand-eye calibration, or None (``sam.locate`` then skips depth)."""
    from .handeye import load_handeye

    try:
        return load_handeye()
    except (OSError, ValueError, KeyError) as error:
        term(f"no hand-eye calibration, depth is not measured: {error}")
        return None


def scripted_sam(canned: dict) -> Any:
    """Dry-run SAM: ``{"<node>": {"<prompt>": [[x1, y1, x2, y2, score], ...]}}`` on a 640x480 photo.

    A prompt the script does not list finds nothing.
    """
    from rio.graph import current_node
    from rio.sam import Instance, SamReply

    def segment(jpeg: bytes, prompt: str) -> SamReply:
        node = current_node()
        if node not in canned:
            raise RuntimeError(f"dry-run has no scripted SAM answer for {node!r}")
        boxes = canned[node].get(prompt, [])
        found = [Instance(i, (b[0], b[1], b[2], b[3]), (640, 480), float(b[4])) for i, b in enumerate(boxes)]
        return SamReply(found, "", 0.0)

    return segment


def _scripted_operator(canned: dict) -> Any:
    def ask(node: str, question: str, answers: list[str]) -> str:
        reply = canned.get(node)
        if not isinstance(reply, str):
            raise RuntimeError(f"dry-run has no scripted operator answer for {node!r}")
        term(f"[dry run] operator asked {question!r} {answers}: {reply}")
        return reply

    return ask
