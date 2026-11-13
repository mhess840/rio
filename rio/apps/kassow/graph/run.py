# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Check a graph, or walk it. The arm stays still unless ``--no-dry-run`` is set."""

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
from rio.apps.kassow.skills import DrySkills
from rio.apps.terminal import make_stdout_blocking, term
from rio.jev import JevCfg, JevClient, encode_jpeg

from .check import require_graph
from .model import EXAMPLE_PATH, Graph, load_graph
from .walk import WalkLog, walk


@dataclass
class Args:
    """Walk a Graph-as-Policy file. Dry-run is the default."""

    command: Literal["check", "run"] = "check"
    """``check`` only reads the file. ``run`` walks it."""
    graph: Path | None = None
    """Graph JSON. Unset: the shipped example."""
    dry_run: bool = True
    """Print skills and do not move the arm. Pass ``--no-dry-run`` on the cell."""
    script: Path | None = None
    """Dry-run Jev answers, a JSON object of node name to an ``answers`` object."""
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


def main(args: Args) -> str:
    """Check or walk. Returns the stop reason of a run, or ``ok`` after a clean check."""
    make_stdout_blocking()
    graph = load_graph(graph_path(args))
    require_graph(graph)
    _print_graph(graph)
    if args.command == "check":
        term("check ok")
        return "ok"
    if args.dry_run:
        return _dry(args, graph)
    return _live(args, graph)


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
        elif kind == "stop":
            extra = f" {node.get('reason', '')}"
        term(f"  {name}: {kind}{extra}")


def _dry(args: Args, graph: Graph) -> str:
    canned = scripted_answers(args.script)

    def ask(node: str, questions: dict, note: str) -> dict:
        if node not in canned:
            raise RuntimeError(f"dry-run has no scripted Jev answer for {node!r}")
        return canned[node]

    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    reason = walk(graph, DrySkills(), ask, WalkLog(root))
    term(f"dry-run stopped: {reason}")
    term(f"log: {root}")
    return reason


def _live(args: Args, graph: Graph) -> str:
    from contextlib import ExitStack

    from rio_hw.middleware import ServerManager

    from rio.apps.kassow.jev_ask import fresh_frame, jev_camera_session
    from rio.apps.kassow.orchestrator.actions import Args as OrchestratorArgs
    from rio.apps.kassow.orchestrator.actions import NodeKeys, station_cfg
    from rio.apps.kassow.skills import KassowSkills
    from rio.envs.env import make_env

    client = JevClient(args.jev)
    try:
        station = station_cfg(OrchestratorArgs(mw=args.mw))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"arm setup: {error}") from error
    root = experiments_root() / "graph" / f"{datetime.now():%Y%m%d_%H%M%S}"
    log = WalkLog(root)
    logger.info("Graph run log: {}", root)
    with ExitStack() as stack:
        camera = stack.enter_context(jev_camera_session(mw=args.mw))
        servers, clients, env = make_env(station)
        stack.enter_context(ServerManager(args.mw, list(servers.values())))
        stack.enter_context(env)
        keys = NodeKeys(stack.enter_context(clients["teleop_keyboard"]()))
        skills = KassowSkills(env, home_joint_speed=args.home_joint_speed)

        def ask(node: str, questions: dict, note: str) -> dict:
            jpeg = encode_jpeg(fresh_frame(camera, 0.5))
            (root / f"{node}.jpg").write_bytes(jpeg)
            reply = client.ask(jpeg, questions, note)
            return reply.answers

        reason = walk(graph, skills, ask, log, keys=keys)
    term(f"stopped: {reason}")
    term(f"log: {root}")
    return reason


def cli() -> None:
    """Parse arguments and check or walk a graph."""
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
