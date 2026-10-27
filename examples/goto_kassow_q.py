# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Discrete KORD ``moveJ`` to a saved eval pose. No Vive, no policy, no VelCmd mix.

Mean TCP 1.0 s into the pick demos (approach, still open)::

    uv run python -m examples.goto_kassow_q --pose start

Mean first-close pose (where the cylinder usually sits)::

    uv run python -m examples.goto_kassow_q --pose cylinder
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from examples.cfg.kassow_vive_gripper import KassowViveGripperStation
from examples.teleop_vive_hand import (
    HOME_POST_MOVE_S,
    HOME_SETTLE_TICKS,
    begin_kord_home,
    joints_at_home,
    joints_stopped,
    load_home_state,
)
from rio.cfg import Camera
from rio.envs.env import make_env

PRESETS = {
    "start": Path("data/kassow_vive_gripper/eval_start.json"),
    "cylinder": Path("data/kassow_vive_gripper/eval_cylinder.json"),
}


@dataclass
class Args(KassowViveGripperStation):
    pose: str = "start"
    pose_file: str | None = None
    teleop: str | None = None
    teleop_cfg: None = None
    teleop_keyboard: str | None = None
    teleop_keyboard_cfg: None = None
    teleop_clicker: str | None = None
    teleop_clicker_cfg: None = None
    cameras: dict[str, Camera] = field(default_factory=dict)
    recorder: str | None = None
    visualizer: str | None = None


def resolve_pose_file(args: Args) -> Path:
    if args.pose_file:
        return Path(args.pose_file)
    key = str(args.pose).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"pose must be one of {sorted(PRESETS)} or --pose-file, got {args.pose!r}")
    return PRESETS[key]


def goto_loop(args: Args, env) -> None:
    path = resolve_pose_file(args)
    home = load_home_state(path, float(args.gripper_open))
    q_goal = np.asarray(home["joint_q"], dtype=float)
    xyz = np.asarray(home["eef_pose"], dtype=float)[:3]
    logger.info("Target {}  tcp={}  file={}", args.pose, np.round(xyz, 3).tolist(), path)
    input("Press Enter to moveJ (E-Stop ready)...")

    arm = env.robot.arm
    state = arm.get_state()
    q = np.asarray(state["joint_q"], dtype=float)
    t0 = time.now()
    homing, _, move_until, deadline = begin_kord_home(arm, q, home, args.home_joint_speed, t0)
    if not homing:
        logger.info("Already at target")
        return

    settle = 0
    while True:
        now = time.now()
        state = arm.get_state()
        q = np.asarray(state["joint_q"], dtype=float)
        alarm = int(state.get("alarm_code", 0) or 0)
        landed = joints_at_home(q, q_goal) and now >= move_until and joints_stopped(state)
        settle = settle + 1 if landed else 0
        if alarm:
            logger.warning("Aborted, KORD alarm")
            return
        if settle >= HOME_SETTLE_TICKS or now > deadline:
            eef = np.asarray(state.get("eef_pose", xyz), dtype=float)[:3]
            logger.info("Done tcp={} q_err_max={:.2f} deg", np.round(eef, 3).tolist(),
                        float(np.degrees(np.max(np.abs(q - q_goal)))))
            time.sleep(HOME_POST_MOVE_S)
            if hasattr(env.robot, "move_gripper"):
                env.robot.move_gripper(float(home["gripper"]), now + 0.2)
                time.sleep(0.5)
            return
        time.sleep(0.02)


def main(args: Args) -> None:
    servers, clients, env = make_env(args)
    with ServerManager(args.mw, [*list(servers.values())]):
        with env:
            goto_loop(args, env)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
