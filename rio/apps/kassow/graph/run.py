# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan and walk a graph. The arm stays still unless ``run --no-dry-run`` is set.

``describe`` prints what a graph may use, ``look`` saves one photo per camera,
``ask`` puts Jev questions (or ``--locate``) to a photo, ``check`` validates a
graph, ``run`` walks it. ``describe``, ``look``, and ``ask`` print JSON.
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
from rio.jev import JevCfg, JevClient, encode_jpeg

from .check import require_graph
from .model import EXAMPLE_PATH, TARGET_KINDS, Graph, GraphError, load_graph
from .table_map import TableMap, table_map_path
from .vla import dry_run_vla
from .walk import WalkLog, walk


@dataclass
class Args:
    """Plan and walk a Graph-as-Policy file. Dry-run is the default."""

    command: Literal["describe", "look", "ask", "check", "run"] = "check"
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
    """``ask``: a saved photo. Unset: a fresh Jev photo. Targets are mapped only for ``jev.jpg`` photos."""
    questions: Path | None = None
    """``ask``: JSON file with a Jev question map."""
    locate: str = ""
    """``ask``: describe one object; Jev boxes it and the table map places it."""
    device: str = "cuda:1"
    """GPU of the VLA residents."""
    single_slot: bool = False
    """Keep one VLA resident and reload the checkpoint on every experiment switch."""
    jev: JevCfg = field(default_factory=JevCfg)
    mw: str = "Thread"
    mp_method: str = "spawn"
    home_joint_speed: float = 0.51
    """Fastest joint speed of a home moveJ, rad/s."""


def graph_path(args: Args) -> Path:
    """The file to check or walk."""
    return args.graph if args.graph is not None else EXAMPLE_PATH


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
    if args.command in ("describe", "look", "ask"):
        print(json.dumps(observe(args), indent=2))
        return "ok"
    graph = load_graph(graph_path(args))
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
    client = JevClient(args.jev)
    result = ask_photo(
        encode_jpeg(image.read_bytes()),
        lambda jpeg, asked: client.ask(jpeg, asked).answers,
        questions=questions,
        locate=args.locate,
        table=table,
    )
    return {"image": str(image), **result}


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
    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    log = WalkLog(root)
    _keep_graph(root, graph)
    logger.info("Graph run log: {}", root)
    with ExitStack() as stack:
        camera = stack.enter_context(jev_camera_session(mw=args.mw))
        servers, clients, env = make_env(station)
        stack.enter_context(ServerManager(args.mw, list(servers.values())))
        stack.enter_context(env)
        keys = NodeKeys(stack.enter_context(clients["teleop_keyboard"]()))
        skills = KassowSkills(env, home_joint_speed=args.home_joint_speed, table=table)

        def ask(node: str, questions: dict, note: str) -> dict:
            jpeg = encode_jpeg(fresh_frame(camera, 0.5))
            (root / f"{node}.jpg").write_bytes(jpeg)
            reply = client.ask(jpeg, questions, note)
            return reply.answers

        def space_or_q(message: str) -> bool:
            term(f"{message}\n  space: go    q: stop")
            while True:
                pressed = keys.pressed()
                if "q" in pressed:
                    return False
                if " " in pressed:
                    return True
                time.sleep(0.05)

        confirm = space_or_q if args.confirm else None
        run_vla = make_run_vla(env, keys, skills, root / "vla", device=args.device, single_slot=args.single_slot, mw=args.mw)
        reason = walk(graph, skills, ask, log, keys=keys, table=table, confirm=confirm, run_vla=run_vla)
    term(f"stopped: {reason}")
    term(f"log: {root}")
    return reason


def cli() -> None:
    """Parse arguments and check or walk a graph."""
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    try:
        main(args)
    except GraphError as error:
        raise SystemExit(f"graph problem:\n{error}") from error
