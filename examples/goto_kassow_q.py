# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Discrete KORD ``moveJ`` to a saved eval pose. No Vive, no policy, no VelCmd mix.

Home pose of the active experiment (``uv run -m examples.experiment``)::

    uv run python -m examples.goto_kassow_q

Home of a named experiment::

    uv run python -m examples.goto_kassow_q --pose sort

One home of ``open``, by short name or by the task sentence::

    uv run python -m examples.goto_kassow_q --pose space1
    uv run python -m examples.goto_kassow_q --instruction "put the object in space 2"

Pick eval poses (mean TCP 1.0 s into the demos / mean first close), read from
the pick ``raw/`` folder::

    uv run python -m examples.goto_kassow_q --pose start
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

from examples.cfg.kassow_experiments import EXPERIMENTS
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


def pose_presets(home_file: str | None) -> dict[str, Path]:
    """``home`` (this station's experiment), every experiment home, and the pick eval poses.

    An experiment with one home per instruction contributes its short pose names
    (``space1``, ``space2``, ``space3``) instead of a single file.
    """
    pick_raw = EXPERIMENTS["pick"].raw_dir
    presets: dict[str, Path] = {}
    for name, spec in EXPERIMENTS.items():
        if spec.homes:
            for home in spec.homes:
                presets[home.pose] = home.path
        else:
            presets[name] = spec.home_path
    presets["start"] = pick_raw / "eval_start.json"
    presets["cylinder"] = pick_raw / "eval_cylinder.json"
    if home_file:
        presets["home"] = Path(home_file)
    return presets


@dataclass
class Args(KassowViveGripperStation):
    pose: str = "home"
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


def home_owner(path: Path) -> tuple[str | None, str | None]:
    """Experiment and task sentence that own this home file, if it is a preset."""
    for spec in EXPERIMENTS.values():
        for home in spec.homes:
            if home.path == path:
                return spec.name, home.instruction
        if not spec.homes and spec.home_path == path:
            return spec.name, None
    return None, None


def resolve_pose_file(args: Args) -> Path:
    if args.pose_file:
        return Path(args.pose_file)
    presets = pose_presets(args.home_file)
    key = str(args.pose).strip().lower()
    if key not in presets:
        raise ValueError(f"pose must be one of {sorted(presets)} or --pose-file, got {args.pose!r}")
    if not presets[key].is_file():
        raise FileNotFoundError(f"Pose '{key}' has no file at {presets[key]}")
    return presets[key]


def load_pose(path: Path, gripper_open: float, experiment: str = "", instruction: str = "") -> dict:
    """Home snapshot at ``path``, checked against the experiment and sentence that own it."""
    owner, sentence = home_owner(path)
    return load_home_state(
        path,
        float(gripper_open),
        experiment=owner or experiment or None,
        instruction=sentence or instruction or None,
    )


def goto_home(env, home: dict, *, home_joint_speed: float, set_gripper: bool = True) -> bool:
    """KORD ``moveJ`` to ``home`` and wait until the arm has settled. No prompt.

    Args:
        env: Started environment with a Kassow arm.
        home: Snapshot from ``load_pose``.
        home_joint_speed: Fastest joint speed in rad/s; sets the move duration.
        set_gripper: Also drive the gripper to the snapshot opening.

    Returns:
        False if KORD raised an alarm or the move did not land on the target joints.
    """
    q_goal = np.asarray(home["joint_q"], dtype=float)
    xyz = np.asarray(home["eef_pose"], dtype=float)[:3]
    arm = env.robot.arm
    state = arm.get_state()
    q = np.asarray(state["joint_q"], dtype=float)
    t0 = time.now()
    homing, _, move_until, deadline = begin_kord_home(arm, q, home, home_joint_speed, t0)
    if not homing and not joints_at_home(q, q_goal):
        logger.error("moveJ was not queued")
        return False

    settle = 0
    landed_ok = True
    while homing:
        now = time.now()
        state = arm.get_state()
        q = np.asarray(state["joint_q"], dtype=float)
        alarm = int(state.get("alarm_code", 0) or 0)
        landed = joints_at_home(q, q_goal) and now >= move_until and joints_stopped(state)
        settle = settle + 1 if landed else 0
        if alarm:
            logger.warning("Aborted, KORD alarm 0x{:08x}", alarm)
            return False
        if settle >= HOME_SETTLE_TICKS or now > deadline:
            eef = np.asarray(state.get("eef_pose", xyz), dtype=float)[:3]
            landed_ok = joints_at_home(q, q_goal)
            logger.info(
                "Done tcp={} q_err_max={:.2f} deg", np.round(eef, 3).tolist(), float(np.degrees(np.max(np.abs(q - q_goal))))
            )
            time.sleep(HOME_POST_MOVE_S)
            break
        time.sleep(0.02)
    if set_gripper and hasattr(env.robot, "move_gripper"):
        env.robot.move_gripper(float(home["gripper"]), time.now() + 0.2)
        time.sleep(0.5)
    return landed_ok


def goto_loop(args: Args, env) -> None:
    path = resolve_pose_file(args)
    home = load_pose(path, args.gripper_open, args.experiment, str(args.instruction or ""))
    xyz = np.asarray(home["eef_pose"], dtype=float)[:3]
    logger.info("Experiment {}  target {}  tcp={}  file={}", args.experiment, args.pose, np.round(xyz, 3).tolist(), path)
    input("Press Enter to moveJ (E-Stop ready)...")
    if not goto_home(env, home, home_joint_speed=args.home_joint_speed):
        logger.warning("Target not reached")


def main(args: Args) -> None:
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, [*list(servers.values())]):
        with env:
            goto_loop(args, env)


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
