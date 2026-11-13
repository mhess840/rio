# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Discrete KORD ``moveJ`` to a saved eval pose. No Vive, no policy, no VelCmd mix.

Home pose of the active experiment (``uv run -m rio.apps.kassow.experiment_cli``)::

    uv run python -m rio.apps.kassow.goto

Home of a named experiment::

    uv run python -m rio.apps.kassow.goto --pose sort

One home of ``open``, by short name or by the task sentence::

    uv run python -m rio.apps.kassow.goto --pose space1
    uv run python -m rio.apps.kassow.goto --instruction "put the object in space 2"

Pick eval poses (mean TCP 1.0 s into the demos / mean first close), read from
the pick ``raw/`` folder::

    uv run python -m rio.apps.kassow.goto --pose start
    uv run python -m rio.apps.kassow.goto --pose cylinder
"""

from __future__ import annotations

import enum
import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.apps.kassow.experiments import EXPERIMENTS, PACKAGED_HOMES_DIR, homes_dir
from rio.apps.kassow.homing import (
    HOME_POST_MOVE_S,
    HOME_SETTLE_TICKS,
    begin_kord_home,
    joints_at_home,
    joints_stopped,
    load_home_state,
)
from rio.apps.kassow.stations import KassowViveGripperStation
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
    # ``None`` types hide the teleop and camera nodes from this CLI; a moveJ needs only the arm.
    teleop: str | None = None
    teleop_cfg: None = None  # ty: ignore[invalid-mutable-override]
    teleop_keyboard: str | None = None
    teleop_keyboard_cfg: None = None  # ty: ignore[invalid-mutable-override]
    teleop_clicker: str | None = None
    teleop_clicker_cfg: None = None  # ty: ignore[invalid-mutable-override]
    cameras: dict[str, Camera] = field(default_factory=dict)
    recorder: str | None = None
    visualizer: str | None = None


def home_owner(path: Path) -> tuple[str | None, str | None]:
    """Experiment and task sentence that own this home file, if it is a captured or packaged preset."""
    path = Path(path)
    if path.resolve().parent not in {homes_dir().resolve(), PACKAGED_HOMES_DIR}:
        return None, None
    for spec in EXPERIMENTS.values():
        for home in spec.homes:
            if home.filename == path.name:
                return spec.name, home.instruction
        if not spec.homes and spec.home_filename == path.name:
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


class MoveStatus(enum.Enum):
    """How a discrete ``moveJ`` ended."""

    LANDED = "landed"
    MISSED = "missed"
    ALARM = "alarm"


def goto_home(env, home: dict, *, home_joint_speed: float, set_gripper: bool = True) -> MoveStatus:
    """KORD ``moveJ`` to ``home`` and wait until the arm has settled. No prompt.

    Args:
        env: Started environment with a Kassow arm.
        home: Snapshot from ``load_pose``.
        home_joint_speed: Fastest joint speed in rad/s; sets the move duration.
        set_gripper: Also drive the gripper to the snapshot opening.

    Returns:
        ``LANDED`` on the target joints, ``ALARM`` when KORD latches an alarm,
        or ``MISSED`` when the move was not queued or did not settle on target.
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
        return MoveStatus.MISSED

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
            return MoveStatus.ALARM
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
    return MoveStatus.LANDED if landed_ok else MoveStatus.MISSED


def lift_z(env, meters: float, *, home_joint_speed: float) -> MoveStatus:
    """Raise the TCP by ``meters`` along base z, then wait until the joints settle.

    The move is one discrete ``moveJ`` to the IK of the raised pose. The gripper
    command is left alone, so a compartment that is being held open stays open.
    """
    arm = env.robot.arm
    kin = getattr(arm, "_kin", None)
    if kin is None:
        logger.error("Arm has no kinematics; cannot lift")
        return MoveStatus.MISSED
    state = arm.get_state()
    q = np.asarray(state["joint_q"], dtype=float)
    pose = np.asarray(state["eef_pose"], dtype=float).copy()
    pose[2] += float(meters)
    q_goal = np.asarray(kin.ik(q, pose), dtype=float)
    reached = np.asarray(kin.fk(q_goal), dtype=float)
    if abs(float(reached[2] - pose[2])) > 0.02:
        logger.error("Lift of {:.0f} cm is not reachable from this pose", meters * 100)
        return MoveStatus.MISSED
    logger.info("Lifting {:.0f} cm before the next moveJ", meters * 100)
    return goto_home(env, {"joint_q": q_goal, "eef_pose": pose}, home_joint_speed=home_joint_speed, set_gripper=False)


REACH_M = 0.005
REACH_ROT_RAD = 0.02
# An IK answer that turns any joint further than this is a wrap or a flip, not a short move.
MAX_JOINT_TRAVEL_RAD = np.pi


def move_tcp(
    env,
    xyz: np.ndarray,
    rotvec: np.ndarray,
    *,
    home_joint_speed: float,
    seed_q: np.ndarray | None = None,
) -> MoveStatus:
    """``moveJ`` so that the measured TCP lands on ``xyz`` with orientation ``rotvec``.

    IK runs on the URDF, whose TCP differs from the controller's by a few millimetres.
    That difference at the current joints is added to the goal.

    Args:
        env: Started environment with a Kassow arm.
        xyz: Goal in the base frame, as the controller reports ``eef_pose``.
        rotvec: Goal orientation, axis-angle.
        home_joint_speed: Fastest joint speed in rad/s.
        seed_q: IK start. Unset: the current joints.

    Returns:
        ``MISSED`` without moving when IK misses the goal, leaves the joint limits,
        or turns a joint by more than ``MAX_JOINT_TRAVEL_RAD``; otherwise as ``goto_home``.
    """
    from typing import cast

    from scipy.spatial.transform import Rotation

    arm = env.robot.arm
    kin = getattr(arm, "_kin", None)
    if kin is None:
        logger.error("Arm has no kinematics; cannot move the TCP")
        return MoveStatus.MISSED
    state = arm.get_state()
    q = np.asarray(state["joint_q"], dtype=float)
    offset = np.asarray(kin.fk(q), dtype=float)[:3] - np.asarray(state["eef_pose"], dtype=float)[:3]
    goal = np.concatenate([np.asarray(xyz, dtype=float) + offset, np.asarray(rotvec, dtype=float)])
    q_goal = np.asarray(kin.ik(q if seed_q is None else np.asarray(seed_q, dtype=float), goal), dtype=float)
    reached = np.asarray(kin.fk(q_goal), dtype=float)
    pos_err = float(np.linalg.norm(reached[:3] - goal[:3]))
    rot_err = float(cast(Rotation, Rotation.from_rotvec(reached[3:]).inv() * Rotation.from_rotvec(goal[3:])).magnitude())
    if pos_err > REACH_M or rot_err > REACH_ROT_RAD:
        logger.error(
            "TCP {} is not reachable (IK off by {:.0f} mm, {:.1f} deg)",
            np.round(xyz, 3).tolist(),
            pos_err * 1000,
            np.degrees(rot_err),
        )
        return MoveStatus.MISSED
    model = getattr(kin, "model", None)
    lower = getattr(model, "lowerPositionLimit", None)
    upper = getattr(model, "upperPositionLimit", None)
    if lower is not None and upper is not None and (np.any(q_goal < lower) or np.any(q_goal > upper)):
        logger.error("IK answer for {} leaves the joint limits", np.round(xyz, 3).tolist())
        return MoveStatus.MISSED
    travel = float(np.max(np.abs(q_goal - q)))
    if travel > MAX_JOINT_TRAVEL_RAD:
        logger.error("IK answer for {} turns a joint by {:.0f} deg", np.round(xyz, 3).tolist(), np.degrees(travel))
        return MoveStatus.MISSED
    return goto_home(env, {"joint_q": q_goal, "eef_pose": goal}, home_joint_speed=home_joint_speed, set_gripper=False)


WAIT_SHIFT_Y_M = -0.10


def write_wait_pose(path: Path, space1: Path) -> Path:
    """Space-1 home, shifted 10 cm toward the base (negative y), gripper closed.

    The start pose stays the organize home. This file is the closer rest pose
    used once the inlay has been accepted.
    """
    from rio_hw.robots.kassow_kinematics import KassowKinematics

    payload = json.loads(space1.read_text())
    pose = np.asarray(payload["eef_pose"], dtype=float).copy()
    pose[1] += WAIT_SHIFT_Y_M
    q0 = np.asarray(payload["joint_q"], dtype=float)
    kin = KassowKinematics()
    q_goal = np.asarray(kin.ik(q0, pose), dtype=float)
    reached = np.asarray(kin.fk(q_goal), dtype=float)
    if abs(float(reached[1] - pose[1])) > 0.02:
        raise RuntimeError(f"waiting pose is not reachable from {space1.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "open",
                "joint_q": q_goal.tolist(),
                "eef_pose": pose.tolist(),
                "gripper": 0.0,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Waiting pose {} tcp={}", path.name, np.round(pose[:3], 3).tolist())
    return path


def goto_loop(args: Args, env) -> None:
    path = resolve_pose_file(args)
    home = load_pose(path, args.gripper_open, args.experiment, str(args.instruction or ""))
    xyz = np.asarray(home["eef_pose"], dtype=float)[:3]
    logger.info("Experiment {}  target {}  tcp={}  file={}", args.experiment, args.pose, np.round(xyz, 3).tolist(), path)
    input("Press Enter to moveJ (E-Stop ready)...")
    if goto_home(env, home, home_joint_speed=args.home_joint_speed) is not MoveStatus.LANDED:
        logger.warning("Target not reached")


def main(args: Args) -> None:
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, [*list(servers.values())]):
        with env:
            goto_loop(args, env)


def cli() -> None:
    """Parse ``Args`` from the command line and move to the selected pose."""
    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)


if __name__ == "__main__":
    cli()
