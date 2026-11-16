# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan and walk a graph. The arm stays still unless ``run --no-dry-run`` is set.

``describe`` prints what a graph may use, ``look`` saves one photo per camera,
``ask`` puts Jev questions (or ``--locate``) to a photo, ``check`` validates a
graph, ``run`` walks it, ``view`` serves the live page (``rio.graph.live``).
``describe``, ``look``, and ``ask`` print JSON.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from rio.apps.kassow.experiments import experiments_root
from rio.apps.kassow.hardware import JEV_SERIAL
from rio.apps.kassow.skills import DrySkills
from rio.apps.terminal import make_stdout_blocking, term
from rio.graph import WorkflowError, WorkflowInvalid, is_v3
from rio.jev import JevCfg, JevClient, encode_jpeg

from .check import require_graph
from .model import EXAMPLE_PATH, TARGET_KINDS, Graph, GraphError, load_graph
from .table_map import TableMap, table_map_path
from .vla import dry_run_vla
from .walk import Keys, Skills, WalkLog, walk


@dataclass
class Args:
    """Plan and walk a Graph-as-Policy file. Dry-run is the default."""

    command: Literal["describe", "look", "ask", "check", "run", "view"] = "check"
    """``describe``/``look``/``ask`` help to plan, ``check`` reads a graph, ``run`` walks it."""
    graph: Path | None = None
    """Graph JSON. Unset: the shipped example."""
    dry_run: bool = True
    """Print skills and do not move the arm. Pass ``--no-dry-run`` on the cell."""
    script: Path | None = None
    """Dry-run Jev answers, a JSON object of node name to an ``answers`` object."""
    table: Path | None = None
    """Table map of the Jev camera. Unset: experiments/graph/calibration/table/table_map.json."""
    confirm: bool = True
    """On the cell, wait for space before each move to a located target and each VLA; q ends the walk."""
    image: Path | None = None
    """``ask``: a saved photo. Unset: a fresh Jev photo. Targets are mapped only for ``jev.jpg`` photos.
    ``run`` (dry): ``sam.locate`` asks the real SAM about this Jev photo; the arm does not move."""
    questions: Path | None = None
    """``ask``: JSON file with a Jev question map."""
    locate: str = ""
    """``ask``: describe one object; Jev boxes it and the table map places it."""
    sam: str = ""
    """``ask``: SAM noun phrases separated by ``|`` (e.g. ``"orange plate|plate"``); SAM instead of Jev."""
    device: str = "cuda:1"
    """GPU of the VLA residents."""
    single_slot: bool = False
    """Keep one VLA resident and reload the checkpoint on every experiment switch."""
    gripper_open: float | None = None
    """Jaw opening (0..1) for pick_at, place_at and gripper open. Unset: the ``open`` experiment's opening."""
    jev: JevCfg = field(default_factory=JevCfg)
    mw: str = "Thread"
    mp_method: str = "spawn"
    home_joint_speed: float = 0.51
    """Fastest joint speed of a home moveJ, rad/s."""
    view_host: str = "127.0.0.1"
    """``view``: interface of the live page; 0.0.0.0 shows it on another screen."""
    view_port: int = 8766
    """``view``: port of the live page."""


def graph_path(args: Args) -> Path:
    """The file to check or walk."""
    return args.graph if args.graph is not None else EXAMPLE_PATH


def is_v3_file(path: Path) -> bool:
    """True when ``path`` is a GaP v3 workflow (``"version": 3``) rather than a v1 graph."""
    try:
        return is_v3(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return False


def scripted_answers(path: Path | None) -> dict[str, dict]:
    """Canned Jev answers for a dry run, keyed by node name."""
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("the Jev script must be a JSON object of node name to answers")
    return raw


def load_table(args: Args, graph: Graph | None = None) -> TableMap | None:
    """The Jev camera's table map when the graph locates objects (or no graph is given), else None.

    Raises:
        GraphError: The map is missing, invalid, or belongs to another camera.
    """
    if graph is not None and not graph.needs_table():
        return None
    path = args.table if args.table is not None else table_map_path()
    if not path.is_file():
        raise GraphError(f"table map {path} is missing; record, label and fit it first (graph.calibrate, graph.table_map)")
    try:
        table = TableMap.load(path)
    except ValueError as error:
        raise GraphError(str(error)) from error
    if table.camera != "jev" or table.serial != JEV_SERIAL:
        raise GraphError(f"table map {path} is for {table.camera} {table.serial}, not the Jev camera {JEV_SERIAL}")
    return table


def main(args: Args) -> str:
    """Plan, check, or walk. Returns the stop reason of a run, else ``ok``."""
    make_stdout_blocking()
    if args.command == "view":
        from rio.graph.live import serve

        serve(experiments_root() / "graph", args.view_host, args.view_port)
        return "ok"
    if args.command in ("describe", "look", "ask"):
        print(json.dumps(observe(args), indent=2))
        return "ok"
    path = graph_path(args)
    if is_v3_file(path):
        from .run_v3 import run_workflow

        return run_workflow(args, path)
    graph = load_graph(path)
    require_graph(graph)
    table = load_table(args, graph)
    _print_graph(graph)
    if table is not None:
        loo_mm = table.stats.get("loo_max_m", 0.0) * 1000
        term(f"table map: {table.stats.get('n', '?')} samples, leave-one-out max {loo_mm:.0f} mm")
    if args.command == "check":
        term("check ok")
        return "ok"
    if args.dry_run:
        return _dry(args, graph, table)
    return _live(args, graph, table)


def observe(args: Args) -> dict:
    """``describe``, ``look``, or ``ask``. Photos go to experiments/graph/look/<time>/."""
    from .describe import describe
    from .observe import ask_photo, look

    if args.command == "describe":
        return describe()
    folder = experiments_root() / "graph" / "look" / f"{datetime.now():%Y%m%d_%H%M%S}"
    if args.command == "look":
        return {"photos": look(folder, mw=args.mw)}
    questions = json.loads(args.questions.read_text()) if args.questions is not None else None
    if args.image is None:
        image = Path(look(folder, cameras=("jev",), mw=args.mw)["jev"])
    else:
        image = args.image
    table = None
    if args.locate and image.stem == "jev":
        try:
            table = load_table(args)
        except GraphError as error:
            logger.warning("No target coordinates: {}", error)
    if args.sam:
        return {"image": str(image), "sam": sam_photo(image.read_bytes(), args.sam.split("|"), image.stem == "jev", args)}
    client = JevClient(args.jev)
    result = ask_photo(
        encode_jpeg(image.read_bytes()),
        lambda jpeg, asked: client.ask(jpeg, asked).answers,
        questions=questions,
        locate=args.locate,
        table=table,
    )
    return {"image": str(image), **result}


def sam_photo(jpeg: bytes, prompts: list[str], mapped: bool, args: Args) -> dict:
    """Every SAM instance per prompt with score, box and, on a Jev photo, table ``x, y``."""
    from rio.sam import SamClient

    table = None
    if mapped:
        try:
            table = load_table(args)
        except GraphError as error:
            logger.warning("No target coordinates: {}", error)
    client = SamClient()
    out = {}
    for prompt in (p.strip() for p in prompts):
        if not prompt:
            continue
        reply = client.segment(jpeg, prompt)
        found = []
        for inst in reply.instances:
            item: dict = {"score": inst.score, "box": [round(v) for v in inst.box]}
            if table is not None:
                x, y = table.to_xy(*inst.centre)
                item |= {"x": round(float(x), 4), "y": round(float(y), 4), "inside_area": table.contains(float(x), float(y))}
            found.append(item)
        out[prompt] = {"instances": found, "elapsed_s": round(reply.elapsed_s, 2)}
    return out


def _print_graph(graph: Graph) -> None:
    term(f"task: {graph.task}")
    term(f"start: {graph.start}    max_steps: {graph.max_steps}")
    for name, node in graph.nodes.items():
        kind = node.get("kind")
        extra = ""
        if kind == "move_to":
            extra = f" pose={node.get('pose')}"
        elif kind == "jev":
            extra = f" route={node.get('route')}"
        elif kind == "locate":
            extra = f" {node.get('object')!r} -> {node.get('store')}"
        elif kind in TARGET_KINDS:
            extra = f" target={node.get('target')}"
        elif kind == "vla":
            extra = f" {node.get('experiment')} {node.get('instruction', '')!r}"
        elif kind == "stop":
            extra = f" {node.get('reason', '')}"
        term(f"  {name}: {kind}{extra}")


def _dry(args: Args, graph: Graph, table: TableMap | None) -> str:
    canned = scripted_answers(args.script)

    def ask(node: str, questions: dict, note: str) -> dict:
        if node not in canned:
            raise RuntimeError(f"dry-run has no scripted Jev answer for {node!r}")
        return canned[node]

    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    log = WalkLog(root)
    _keep_graph(root, graph)
    reason = walk(graph, DrySkills(), ask, log, table=table, run_vla=dry_run_vla)
    term(f"dry-run stopped: {reason}")
    term(f"log: {root}")
    return reason


def _keep_graph(root: Path, graph: Graph) -> None:
    payload = {"task": graph.task, "start": graph.start, "max_steps": graph.max_steps, "nodes": graph.nodes}
    (root / "graph.json").write_text(json.dumps(payload, indent=2) + "\n")


def _live(args: Args, graph: Graph, table: TableMap | None) -> str:
    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    log = WalkLog(root)
    _keep_graph(root, graph)
    logger.info("Graph run log: {}", root)
    with live_cell(args, table, root) as cell:
        confirm = cell.confirm if args.confirm else None
        reason = walk(graph, cell.skills, cell.ask, log, keys=cell.keys, table=table, confirm=confirm, run_vla=cell.run_vla)
    term(f"stopped: {reason}")
    term(f"log: {root}")
    return reason


@dataclass
class LiveCell:
    """The started cell: arm skills, Jev on a fresh photo, keys, VLA runner and prompts."""

    skills: Skills
    ask: Callable[[str, dict, str], dict]
    keys: Keys
    run_vla: Callable[[str, str, float], object]
    confirm: Callable[[str], bool]
    choose: Callable[[str, list[str]], str | None]
    photo: Callable[[str], bytes]
    """A fresh Jev photo, saved as ``<node>.jpg`` in the run folder."""


@contextmanager
def live_cell(args: Args, table: TableMap | None, root: Path) -> Generator[LiveCell]:
    """Start the arm, the Jev camera and the keyboard; stop them on exit.

    Args:
        args: Command line.
        table: Table map for the target skills, or None.
        root: Log folder; Jev photos are saved as ``<node>.jpg``.
    """
    from contextlib import ExitStack

    from rio_hw import time
    from rio_hw.middleware import ServerManager

    from rio.apps.kassow.jev_ask import fresh_frame, jev_camera_session
    from rio.apps.kassow.orchestrator.actions import Args as OrchestratorArgs
    from rio.apps.kassow.orchestrator.actions import NodeKeys, station_cfg
    from rio.apps.kassow.skills import KassowSkills
    from rio.envs.env import make_env

    from .vla import make_run_vla

    client = JevClient(args.jev)
    try:
        station = station_cfg(OrchestratorArgs(mw=args.mw))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"arm setup: {error}") from error
    with ExitStack() as stack:
        camera = stack.enter_context(jev_camera_session(mw=args.mw))
        servers, clients, env = make_env(station)
        stack.enter_context(ServerManager(args.mw, list(servers.values())))
        stack.enter_context(env)
        keys = NodeKeys(stack.enter_context(clients["teleop_keyboard"]()))
        skills = KassowSkills(env, home_joint_speed=args.home_joint_speed, gripper_open=args.gripper_open, table=table)
        logger.info("Gripper opens to {:.2f}", skills.gripper_open)

        def photo(node: str) -> bytes:
            jpeg = encode_jpeg(fresh_frame(camera, 0.5))
            (root / f"{node.replace('/', '__')}.jpg").write_bytes(jpeg)
            return jpeg

        def ask(node: str, questions: dict, note: str) -> dict:
            return client.ask(photo(node), questions, note).answers

        def space_or_q(message: str) -> bool:
            term(f"{message}\n  space: go    q: stop")
            while True:
                pressed = keys.pressed()
                if "q" in pressed:
                    return False
                if " " in pressed:
                    return True
                time.sleep(0.05)

        def choose(question: str, answers: list[str]) -> str | None:
            options = "    ".join(f"{i}: {answer}" for i, answer in enumerate(answers, start=1))
            term(f"OPERATOR: {question}\n  {options}    q: stop")
            while True:
                pressed = keys.pressed()
                if "q" in pressed:
                    return None
                for i, answer in enumerate(answers, start=1):
                    if str(i) in pressed:
                        return answer
                time.sleep(0.05)

        run_vla = make_run_vla(env, keys, skills, root / "vla", device=args.device, single_slot=args.single_slot, mw=args.mw)
        yield LiveCell(skills, ask, keys, run_vla, space_or_q, choose, photo)


def cli() -> None:
    """Parse arguments and check or walk a graph."""
    args = tyro.cli(Args)
    if args.gripper_open is not None and not 0.0 < args.gripper_open <= 1.0:
        raise SystemExit(f"--gripper-open must be in (0, 1], got {args.gripper_open}")
    mp.set_start_method(args.mp_method, force=True)
    try:
        status = main(args)
    except (GraphError, WorkflowError, WorkflowInvalid) as error:
        raise SystemExit(f"graph problem:\n{error}") from error
    if status == "failed":
        raise SystemExit(1)
