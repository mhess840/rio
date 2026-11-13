# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Wire the cell (or the dry run) to the state machine and run it."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime

import tyro
from loguru import logger
from rio_hw.middleware import ServerManager

from rio.apps.kassow.experiments import experiments_root, homes_dir, open_spaces, resolve_experiment
from rio.apps.kassow.jev_ask import jev_camera_session
from rio.apps.kassow.skills import DrySkills, KassowSkills
from rio.apps.terminal import make_stdout_blocking, term
from rio.envs.factory import make_node
from rio.jev import JevClient

from .actions import (
    KEYBOARD_ADDR,
    Args,
    NodeKeys,
    dry_run_vla,
    make_ask,
    make_recover,
    make_run_vla,
    preload,
    station_cfg,
)
from .machine import ORGANIZE, CartonCounter, CycleLog, Orchestrator, State, recover_unavailable


def main(args: Args) -> None:
    make_stdout_blocking()
    client = JevClient(args.jev)
    spaces = {number: home.path for number, home in open_spaces().items()}
    organize_home = resolve_experiment(ORGANIZE).home_path
    log = CycleLog(args.runs_dir or experiments_root() / "orchestrator" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}")
    logger.info("Spaces with a home: {}. Log: {}", sorted(spaces), log.root)
    log.write("start", args=asdict(args), spaces={k: v.name for k, v in spaces.items()})

    with ExitStack() as stack:
        camera = stack.enter_context(jev_camera_session(args.jev_serial, args.mw))
        counter = CartonCounter()
        ask = make_ask(args, client, camera, set(spaces), log, counter)
        if args.dry_run:
            server, keyboard_client = make_node(args.mw, "interfaces", "SshKeyboard", {"addr": KEYBOARD_ADDR, "freq": 100})
            stack.enter_context(ServerManager(args.mw, [server]))
            keys = NodeKeys(stack.enter_context(keyboard_client()))
            skills, run_vla = DrySkills(), dry_run_vla
            recover = recover_unavailable
        else:
            from rio.envs.env import make_env

            try:
                preload(args)
                station = station_cfg(args)
            except (FileNotFoundError, ValueError) as error:
                raise SystemExit(f"VLA setup: {error}") from error
            servers, clients, env = make_env(station)
            stack.enter_context(ServerManager(args.mw, list(servers.values())))
            stack.enter_context(env)
            keys = NodeKeys(stack.enter_context(clients["teleop_keyboard"]()))
            skills, run_vla = KassowSkills(env, home_joint_speed=args.home_joint_speed), make_run_vla(args, env, keys, log)
            recover = make_recover(env)
        orchestrator = Orchestrator(
            ask=ask,
            goto=skills.move_to,
            run_vla=run_vla,
            keys=keys,
            recover=recover,
            organize_home=organize_home,
            spaces=spaces,
            log=log,
            max_organize_attempts=args.max_organize_attempts,
            counter=counter,
        )
        orchestrator.lift = skills.lift
        if args.board:
            from .board import DemoBoard

            board = DemoBoard(log.root.parent.parent / "live", args.board_port)
            board.publish(orchestrator)
            board.serve()
            orchestrator.board = board
        from rio.apps.kassow.goto import write_wait_pose

        space1 = spaces.get(1)
        if space1 is not None:
            orchestrator.wait_home = write_wait_pose(homes_dir() / "home_open_wait.json", space1)
        term("\nMoving to the organize home before the first ask.")
        orchestrator.state = State.GOTO
        orchestrator._show()
        if skills.move_to(organize_home) is not True:
            raise SystemExit("Could not reach the organize home before the first ask")
        orchestrator.state = State.IDLE
        orchestrator._show()
        orchestrator.run()
    log.write("quit")


def cli() -> None:
    """Parse ``Args`` from the command line and run the orchestrator."""
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
