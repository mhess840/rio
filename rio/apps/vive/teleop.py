# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate an arm end-effector from a Vive tracker while a Manus glove drives a dexterous hand.

Unlike the spacemouse/gamepad path in teleop_eef.py, the tracker reports an
absolute pose, so motion is retargeted relative to a clutch: pressing the
clutch key snapshots the tracker pose and the current TCP pose together, and
wrist motion is applied relative to that pair. Nothing needs to be calibrated
between SteamVR and the robot.

    STATION=Ur5eViveInspireStation uv run -m examples.teleop_vive_hand
"""

import math
import multiprocessing as mp
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Protocol

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    WorkspaceLimits,
    pose_to_matrix,
    yaw_rotation,
)
from rio_hw.middleware import ServerManager

from rio.apps.terminal import make_stdout_blocking, term
from rio.apps.vive.calibration import resolve_yaw
from rio.apps.vive.controls import (
    apply_live_scale,
    handle_recorder_save,
    handle_recorder_start,
    merge_held_keys,
    sync_scaled_speeds,
)
from rio.envs.env import make_env


class SessionHoming(Protocol):
    """Station-specific return to a saved start pose (e.g. ``rio.apps.kassow.homing.KordHoming``)."""

    on_save: bool
    pending_start: bool
    active: bool

    @property
    def gripper(self) -> float: ...
    def describe(self) -> list[str]: ...
    def begin(self, joint_q: np.ndarray, t: float) -> None: ...
    def update(self, arm_state: dict, eef_pose: np.ndarray, joint_q: np.ndarray, t: float) -> str: ...
    def may_stream(self, t: float) -> bool: ...
    def status(self, joint_q: np.ndarray, recording: bool) -> str: ...


HomingFactory = Callable[[Any, Any, float], SessionHoming]


def read_gripper_position(gripper, timeout: float = 10.0) -> float:
    """Wait until the gripper node has published a position sample."""
    deadline = time.now() + timeout
    last = None
    while time.now() < deadline:
        last = gripper.get_state()
        if isinstance(last, dict) and "gripper_position" in last:
            return float(last["gripper_position"])
        time.sleep(0.05)
    raise RuntimeError(
        "AG gripper has not published state yet (setup still running or Modbus failed). "
        "If the jaws are unhomed, start once with gripper_cfg.calibrate=True."
    )


def teleop_vive_hand(
    args,
    env,
    teleop,
    teleop2=None,
    teleop_keyboard=None,
    teleop_clicker=None,
    visualizer=None,
    yaw_deg=None,
    wait_for_start=True,
    homing_factory: HomingFactory | None = None,
):
    make_stdout_blocking()
    arm = env.robot.arm
    hand = getattr(env.robot, "hand", None)
    gripper = getattr(env.robot, "gripper", None)

    arm_target_pose = np.asarray(arm.get_state()["eef_pose"], dtype=float).copy()
    gripper_open = float(getattr(args, "gripper_open", 1.0))
    gripper_close = float(getattr(args, "gripper_close", 0.0))
    gripper_start = getattr(args, "gripper_start", None)
    if gripper is not None:
        read_gripper_position(gripper)
        gripper_cmd = float(gripper_start) if gripper_start is not None else gripper_open
    else:
        gripper_cmd = 0.0

    homing = None if homing_factory is None else homing_factory(args, arm, gripper_cmd)
    if homing is not None:
        gripper_cmd = homing.gripper

    def homing_active() -> bool:
        return homing is not None and homing.active

    if yaw_deg is None:
        yaw_deg = resolve_yaw(args, teleop)

    base_max_pos_speed = float(args.max_pos_speed)
    base_max_rot_speed = float(args.max_rot_speed)
    base_max_motor_speed = float(getattr(arm, "max_motor_speed", 0.0)) or None
    retargeter = ClutchRetargeter(
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        max_pos_speed=base_max_pos_speed * args.pos_scale,
        max_rot_speed=base_max_rot_speed * args.rot_scale,
        # One rotation about vertical corrects both translation and wrist
        # rotation, since the retargeter applies the map to each.
        axis_map=yaw_rotation(math.radians(yaw_deg)) @ AXIS_MAP_STEAMVR_TO_ROBOT,
        workspace=WorkspaceLimits(
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_z=args.min_z,
        ),
        orientation_enabled=args.orientation_enabled,
    )
    retargeter.hold_pose(arm_target_pose)
    scale_step = float(getattr(args, "scale_step", 1.2))
    min_scale = float(getattr(args, "min_scale", 0.5))
    max_scale = float(getattr(args, "max_scale", 1.5))
    sync_scaled_speeds(
        retargeter,
        base_max_pos_speed=base_max_pos_speed,
        base_max_rot_speed=base_max_rot_speed,
        arm=arm,
        base_max_motor_speed=base_max_motor_speed,
    )

    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
        logger.debug(f"Visualizer: set robot model to {env.robot.urdf_path}")

    print("\nEpisode: n (record) → c (clutch) → demo → s (save + home).")
    print("  Recording is only the window from 'n' to 's'. The robot follows only while clutched.")
    print(f"  '{args.clutch_key}' clutch    '{args.orientation_key}' orientation    'n' start recording    's' save")
    for line in homing.describe() if homing is not None else ():
        print(line)
    print(f"  TCP now: {np.round(arm_target_pose[:3], 3).tolist()}   yaw offset: {yaw_deg:.1f} deg")
    print(
        f"  limits at scale 1.0:  {base_max_pos_speed} m/s, {base_max_rot_speed} rad/s, "
        f"radius {args.min_radius}-{args.max_radius} m, z >= {args.min_z} m"
    )
    print(
        f"  motion scale: pos {retargeter.pos_scale:.2f}  rot {retargeter.rot_scale:.2f}  "
        f"→ {retargeter.max_pos_speed:.2f} m/s, {retargeter.max_rot_speed:.2f} rad/s  "
        f"(mouse back/forward x{scale_step:g}, clamp {min_scale:g}-{max_scale:g})"
    )
    if hand is None:
        print("  No dexterous hand configured.")
    elif teleop2 is None:
        print("  No glove configured; the hand will not be driven.")
    if gripper is not None:
        print(f"  'Space' toggles the gripper ({gripper_close:.2f} closed / {gripper_open:.2f} open).")
    if teleop_clicker is not None:
        if gripper is not None:
            print("  Spare Logitech mouse: left close, right open, back finer, forward coarser.")
        else:
            print("  Spare Logitech mouse: back finer, forward coarser.")
    print("Keep the e-stop within reach.")
    if wait_for_start:
        input(f"Instruction: {args.instruction}\nPress Enter to start")
    time.sleep(getattr(args, "startup_delay", 0.0))

    freq = args.freq
    dt = 1.0 / freq
    command_latency = dt / 2
    t_start = time.now()
    it = 0
    env.set_start_time(t_start)
    env.set_instruction(args.instruction)

    previous_keys: set[str] = set()
    t_last_valid_pose = time.now()
    action = None
    recording = False

    try:
        while True:
            t_cycle_end = t_start + (it + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_cmd_target = t_cycle_end + dt
            now = time.now()
            if t_cmd_target <= now:
                it = int((now - t_start) / dt) + 1
                t_cycle_end = t_start + (it + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_cmd_target = t_cycle_end + dt

            time.precise_wait(t_sample)

            keys = merge_held_keys(
                teleop_keyboard.get_state() if teleop_keyboard else None,
                teleop_clicker.get_state() if teleop_clicker else None,
            )
            if teleop_keyboard or teleop_clicker:
                recording = handle_recorder_start(env, keys, previous_keys, recording)

            tracker_state = teleop.get_state()
            pose_valid = float(tracker_state["pose_valid"]) > 0.5
            arm_state = arm.get_state()
            eef_pose = np.asarray(arm_state["eef_pose"], dtype=float)
            joint_q = np.asarray(arm_state["joint_q"], dtype=float)
            if homing is not None and homing.pending_start:
                homing.pending_start = False
                retargeter.hold_pose(eef_pose)
                homing.begin(joint_q, t_sample)
            if pose_valid:
                t_last_valid_pose = t_sample

            # Clutch and orientation toggles, on key-down only
            pressed = keys - previous_keys
            if args.clutch_key in pressed:
                if homing_active():
                    logger.info("Still homing — wait for Home reached, then clutch")
                elif retargeter.engaged:
                    retargeter.disengage()
                    logger.info("Clutch released")
                elif pose_valid:
                    retargeter.engage(
                        pose_to_matrix(tracker_state["tracker_pose"]),
                        eef_pose[:3],
                        pose_to_matrix(eef_pose)[:3, :3],
                    )
                    logger.info("Clutch engaged")
                else:
                    logger.warning("Cannot engage: no valid tracker pose")
            if args.orientation_key in pressed:
                retargeter.orientation_enabled = not retargeter.orientation_enabled
                logger.info(f"Orientation tracking: {retargeter.orientation_enabled}")
            if ("scale_up" in pressed) != ("scale_down" in pressed):
                pos_s, rot_s = apply_live_scale(
                    retargeter,
                    up="scale_up" in pressed,
                    step=scale_step,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    tracker_pose=pose_to_matrix(tracker_state["tracker_pose"]) if pose_valid else None,
                    pose_valid=pose_valid,
                    base_max_pos_speed=base_max_pos_speed,
                    base_max_rot_speed=base_max_rot_speed,
                    arm=arm,
                    base_max_motor_speed=base_max_motor_speed,
                )
                logger.info(
                    "Motion scale: pos={:.2f}  rot={:.2f}  |  {:.2f} m/s, {:.2f} rad/s",
                    pos_s,
                    rot_s,
                    retargeter.max_pos_speed,
                    retargeter.max_rot_speed,
                )
            if gripper is not None:
                if "pagedown" in pressed:
                    gripper_cmd = gripper_close
                    logger.info("Gripper target: closed ({:.2f})", gripper_cmd)
                elif "pageup" in pressed:
                    gripper_cmd = gripper_open
                    logger.info("Gripper target: open ({:.2f})", gripper_cmd)
                elif " " in pressed:
                    mid = 0.5 * (gripper_open + gripper_close)
                    gripper_cmd = gripper_close if gripper_cmd > mid else gripper_open
                    logger.info("Gripper target: {}", "open" if gripper_cmd > mid else "closed")

            # Walking out of the base stations' view stops motion rather than
            # freezing on the last command.
            if retargeter.engaged and not pose_valid and t_sample - t_last_valid_pose > args.tracking_grace:
                retargeter.disengage()
                logger.warning("Tracking lost, clutch released")

            if retargeter.engaged and pose_valid and not homing_active():
                arm_target_pose = retargeter.target_pose(pose_to_matrix(tracker_state["tracker_pose"]), dt)
                lag = float(np.linalg.norm(arm_target_pose[:3] - eef_pose[:3]))
                if lag > args.max_lag:
                    retargeter.disengage()
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
                    logger.warning(f"Command ran {lag:.3f} m ahead of the arm, clutch released")
            elif homing is not None and homing.active:
                outcome = homing.update(arm_state, eef_pose, joint_q, t_sample)
                if outcome == "done":
                    gripper_cmd = homing.gripper
                if outcome != "aborted":
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
            else:
                # Hold station and keep the reference synced to where the arm
                # actually is, so re-engaging never produces a jump.
                retargeter.hold_pose(eef_pose)
                arm_target_pose = eef_pose.copy()

            arm_streaming = True if homing is None else homing.may_stream(t_sample)

            hand_cmd = None
            if hand is not None and teleop2 is not None:
                hand_cmd = np.asarray(teleop2.get_state()["joint_q"], dtype=float)

            action = env.robot.build_action(arm_target_pose, gripper_cmd=gripper_cmd, hand_cmd=hand_cmd)
            if arm_streaming:
                env.move(action, t_cmd_target + args.arm_latency)
            elif hasattr(env.robot, "move_gripper"):
                env.robot.move_gripper(gripper_cmd, t_cmd_target + args.arm_latency)

            step = env.get_state(action=action)
            if env.recorder and recording:
                env.recorder.record_step(step)
            was_recording = recording
            if teleop_keyboard or teleop_clicker:
                recording = handle_recorder_save(env, keys, previous_keys, recording)
            if homing is not None and homing.on_save and was_recording and not recording:
                if retargeter.engaged:
                    retargeter.disengage()
                    logger.info("Clutch released for home")
                retargeter.hold_pose(eef_pose)
                homing.begin(joint_q, t_sample)
            if visualizer:
                visualizer.log_frame("world/teleop_target", arm_target_pose, axis_length=0.08)
                visualizer.log_env_state("env", step)

            if it % freq == 0:
                if homing is not None and homing.active:
                    mode = homing.status(joint_q, recording)
                elif recording:
                    mode = "REC"
                else:
                    mode = "idle (press 'n')"
                term(
                    f"t: {t_cycle_end - t_start:.1f}s",
                    "|",
                    mode,
                    "|",
                    "ENGAGED" if retargeter.engaged else f"released (press '{args.clutch_key}' to engage)",
                    "|",
                    f"tracking: {'ok' if pose_valid else 'LOST'}",
                    "|",
                    f"tcp: {np.round(eef_pose[:3], 3).tolist()}",
                    "|",
                    f"scale: {retargeter.pos_scale:.2f}/{retargeter.rot_scale:.2f}  {retargeter.max_pos_speed:.2f}m/s",
                    "|",
                    f"hand: {np.round(hand_cmd, 2).tolist() if hand_cmd is not None else None}",
                )

            previous_keys = keys
            time.precise_wait(t_cycle_end)
            it += 1

    except KeyboardInterrupt:
        pass
    finally:
        if env.recorder:
            env.recorder.save(wait=True)


def main(args, homing_factory: HomingFactory | None = None):
    servers, clients, env = make_env(args)
    teleop2_client = clients.get("teleop2")
    keyboard_client = clients.get("teleop_keyboard")
    clicker_client = clients.get("teleop_clicker")
    visualizer_client = clients.get("visualizer")

    with ServerManager(args.mw, list(servers.values())):
        with env, clients["teleop"]() as teleop:
            # Terminal-backed keyboards consume stdin, so finish any interactive
            # Vive calibration and confirmation before starting their listener.
            yaw_deg = resolve_yaw(args, teleop)
            input(f"Instruction: {args.instruction}\nPress Enter to start")
            with (
                teleop2_client() if teleop2_client else nullcontext() as teleop2,
                keyboard_client() if keyboard_client else nullcontext() as teleop_keyboard,
                clicker_client() if clicker_client else nullcontext() as teleop_clicker,
                visualizer_client() if visualizer_client else nullcontext() as visualizer,
            ):
                try:
                    teleop_vive_hand(
                        args,
                        env,
                        teleop,
                        teleop2=teleop2,
                        teleop_keyboard=teleop_keyboard,
                        teleop_clicker=teleop_clicker,
                        visualizer=visualizer,
                        yaw_deg=yaw_deg,
                        wait_for_start=False,
                        homing_factory=homing_factory,
                    )
                except KeyboardInterrupt:
                    pass


def cli(station_cls: type[Any], homing_factory: HomingFactory | None = None) -> None:
    """Parse ``station_cls`` from the command line and teleoperate it."""
    args = tyro.cli(station_cls)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args, homing_factory)
