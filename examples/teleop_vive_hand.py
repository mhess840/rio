"""Teleoperate an arm end-effector from a Vive tracker while a Manus glove
drives a dexterous hand.

Unlike the spacemouse/gamepad path in teleop_eef.py, the tracker reports an
absolute pose, so motion is retargeted relative to a clutch: pressing the
clutch key snapshots the tracker pose and the current TCP pose together, and
wrist motion is applied relative to that pair. Nothing needs to be calibrated
between SteamVR and the robot.

    STATION=Ur5eViveInspireStation uv run -m examples.teleop_vive_hand
"""

import json
import math
import multiprocessing as mp
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    WorkspaceLimits,
    dominant_direction,
    horizontal_angle_between,
    pose_to_matrix,
    yaw_from_motions,
    yaw_rotation,
)
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env

# Directions the operator is asked to sweep, in the robot base frame.
CALIBRATION_SWEEPS = (
    ("+X", np.array([1.0, 0.0, 0.0]), "usually straight away from the robot base"),
    ("+Y", np.array([0.0, 1.0, 0.0]), "90 degrees counter-clockwise from +X, seen from above"),
)


def sweep_problem(points: np.ndarray, min_travel: float) -> str | None:
    """Why a recorded sweep is unusable, or None if it is fine."""
    if len(points) < 10:
        return "Too few valid tracker samples (is tracking dropping out?)."
    net = float(np.linalg.norm(points[-1] - points[0]))
    path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if net < min_travel:
        return f"Only travelled {net * 100:.0f} cm, need {min_travel * 100:.0f} cm."
    # A there-and-back sweep pins the axis but not which way along it points.
    if net < 0.5 * path:
        return "That looked like a round trip; sweep one way only."
    return None


def signal_on_enter(done: threading.Event) -> None:
    input()
    done.set()


def record_sweep(teleop, axis_name, hint, min_travel, sample_rate=50.0, max_attempts=5) -> np.ndarray:
    """Record tracker positions for one guided hand sweep, retrying if unusable."""
    for attempt in range(max_attempts):
        input(
            f"\n  Sweep {axis_name}: move your hand the way you want the TCP to travel "
            f"along robot {axis_name}\n"
            f"  ({hint}).\n"
            f"  Press Enter, move at least {min_travel * 100:.0f} cm in one direction, "
            "then press Enter again: "
        )

        done = threading.Event()
        threading.Thread(target=signal_on_enter, args=(done,), daemon=True).start()

        samples = []
        while not done.is_set():
            state = teleop.get_state()
            if float(state["pose_valid"]) > 0.5:
                samples.append(np.asarray(state["tracker_pose"][:3], dtype=float))
            time.sleep(1.0 / sample_rate)

        points = np.asarray(samples)
        problem = sweep_problem(points, min_travel)
        if problem is None:
            travelled = float(np.linalg.norm(points[-1] - points[0]))
            print(f"  Recorded {len(points)} samples over {travelled * 100:.0f} cm.")
            return points
        remaining = max_attempts - attempt - 1
        print(f"  {problem}" + (f" {remaining} attempts left." if remaining else ""))

    raise RuntimeError(
        f"Could not record a usable {axis_name} sweep. If the tracker never reports a "
        "valid pose, check base station visibility before calibrating."
    )


def calibrate_yaw(args, teleop) -> float:
    """Guided calibration of the yaw between SteamVR's world frame and the robot.

    SteamVR's frame is gravity-aligned, so roll and pitch already agree with the
    robot and a single angle about vertical is the only unknown. Two sweeps are
    recorded rather than one so their measured separation can be checked against
    the 90 degrees it ought to be.
    """
    print("\n=== Vive yaw calibration ===")
    print("Stand where you will actually work, facing the way you will work.")
    print("Wrist orientation does not matter, only the direction you travel.")

    measured, intended = [], []
    for axis_name, axis, hint in CALIBRATION_SWEEPS:
        points = record_sweep(teleop, axis_name, hint, args.min_sweep_travel)
        measured.append(AXIS_MAP_STEAMVR_TO_ROBOT @ dominant_direction(points))
        intended.append(axis)

    separation = math.degrees(horizontal_angle_between(measured[0], measured[1]))
    expected = math.degrees(horizontal_angle_between(intended[0], intended[1]))
    if abs(separation - expected) > args.max_sweep_skew:
        logger.warning(
            f"The two sweeps came out {separation:.0f} degrees apart, expected {expected:.0f}. "
            "Alignment will be rough; re-run with --calibrate-yaw and keep each sweep straight."
        )

    yaw_deg = math.degrees(yaw_from_motions(measured, intended))
    print(f"\n  Yaw offset: {yaw_deg:.1f} degrees   (sweeps {separation:.0f} degrees apart)")
    print("  If motion ends up rotated by roughly 90 degrees, the axis labels were")
    print("  swapped; re-run with --calibrate-yaw and pick the other direction.")
    return yaw_deg


def resolve_yaw(args, teleop) -> float:
    """Yaw offset for this session, calibrating or loading from disk as needed."""
    path = Path(args.yaw_calibration_file) if args.yaw_calibration_file else None

    if args.calibrate_yaw or (path is not None and not path.exists()):
        yaw_deg = calibrate_yaw(args, teleop)
        if path is not None:
            path.write_text(json.dumps({"yaw_offset_deg": yaw_deg}, indent=2))
            print(f"  Saved to {path}. Pass --calibrate-yaw to redo it.")
        return yaw_deg

    if path is not None:
        yaw_deg = float(json.loads(path.read_text())["yaw_offset_deg"])
        logger.info(f"Loaded yaw offset {yaw_deg:.1f} degrees from {path}")
        return yaw_deg

    return args.yaw_offset


# Matches rio.envs.poll Logitech presenter mapping.
_PRESENTER_CLOSE = frozenset({0x4E, 0x4F, 0xFF56, 0xFF53, 109, 106})  # PageDown, Right
_PRESENTER_OPEN = frozenset({0x4B, 0x50, 0xFF55, 0xFF51, 104, 105})  # PageUp, Left


def held_keys(keyboard_state) -> set[str]:
    """Characters currently held down, from the keyboard node's rollover slots."""
    keys = {chr(k) for k in keyboard_state["alphanumeric_state"] if k != 0}
    special = {int(code) for code in keyboard_state["special_state"] if code != 0}
    if 0x2C in special:
        keys.add(" ")
    if special & _PRESENTER_CLOSE:
        keys.add("pagedown")
    if special & _PRESENTER_OPEN:
        keys.add("pageup")
    return keys


def merge_held_keys(*keyboard_states) -> set[str]:
    """Union key sets from one or more keyboard nodes."""
    keys: set[str] = set()
    for state in keyboard_states:
        if state is not None:
            keys |= held_keys(state)
    return keys


def handle_recorder_start(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Start a trajectory on 'n' so this tick is the first recorded frame."""
    if not env.recorder or recording:
        return recording
    if "n" not in keys - previous:
        return recording
    env.recorder.new_trajectory(wait=False)
    print("\n ============================================= ")
    logger.info("Started new trajectory recording")
    print("============================================= \n")
    return True


def handle_recorder_save(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Save on 's' after this tick has been written, so the file ends with 's'."""
    if not env.recorder or not recording:
        return recording
    if "s" not in keys - previous:
        return recording
    env.recorder.save(wait=False)
    logger.info("Saved trajectory in background (press 'n' to start the next one)")
    print("============================================= \n")
    return False


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
):
    arm = env.robot.arm
    hand = getattr(env.robot, "hand", None)
    gripper = getattr(env.robot, "gripper", None)

    arm_target_pose = np.asarray(arm.get_state()["eef_pose"], dtype=float).copy()
    gripper_open = float(getattr(args, "gripper_open", 1.0))
    gripper_close = float(getattr(args, "gripper_close", 0.0))
    if gripper is not None:
        read_gripper_position(gripper)
        gripper_cmd = gripper_open
    else:
        gripper_cmd = 0.0

    if yaw_deg is None:
        yaw_deg = resolve_yaw(args, teleop)

    retargeter = ClutchRetargeter(
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
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

    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
        logger.debug(f"Visualizer: set robot model to {env.robot.urdf_path}")

    print("\nEpisode: Enter (ready) → n (record) → c (clutch on) → … → c (clutch off) → s (save).")
    print("  Recording is only the window from 'n' to 's'. The robot follows only while clutched.")
    print(
        f"  '{args.clutch_key}' clutch    '{args.orientation_key}' orientation    "
        "'n' start recording    's' save"
    )
    print(f"  TCP now: {np.round(arm_target_pose[:3], 3).tolist()}   yaw offset: {yaw_deg:.1f} deg")
    print(f"  limits:  {args.max_pos_speed} m/s, {args.max_rot_speed} rad/s, "
          f"radius {args.min_radius}-{args.max_radius} m, z >= {args.min_z} m")
    if hand is None:
        print("  No dexterous hand configured.")
    elif teleop2 is None:
        print("  No glove configured; the hand will not be driven.")
    if gripper is not None:
        print(f"  'Space' toggles the gripper ({gripper_close:.2f} closed / {gripper_open:.2f} open).")
        print("  Spare Logitech mouse: left click closes, right click opens.")
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

            time.precise_wait(t_sample)

            keys = merge_held_keys(
                teleop_keyboard.get_state() if teleop_keyboard else None,
                teleop_clicker.get_state() if teleop_clicker else None,
            )
            if teleop_keyboard or teleop_clicker:
                recording = handle_recorder_start(env, keys, previous_keys, recording)

            tracker_state = teleop.get_state()
            pose_valid = float(tracker_state["pose_valid"]) > 0.5
            eef_pose = np.asarray(arm.get_state()["eef_pose"], dtype=float)

            if pose_valid:
                t_last_valid_pose = t_sample

            # Clutch and orientation toggles, on key-down only
            pressed = keys - previous_keys
            if args.clutch_key in pressed:
                if retargeter.engaged:
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

            if retargeter.engaged and pose_valid:
                arm_target_pose = retargeter.target_pose(pose_to_matrix(tracker_state["tracker_pose"]), dt)
                lag = float(np.linalg.norm(arm_target_pose[:3] - eef_pose[:3]))
                if lag > args.max_lag:
                    retargeter.disengage()
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
                    logger.warning(f"Command ran {lag:.3f} m ahead of the arm, clutch released")
            else:
                # Hold station and keep the reference synced to where the arm
                # actually is, so re-engaging never produces a jump.
                retargeter.hold_pose(eef_pose)
                arm_target_pose = eef_pose.copy()

            hand_cmd = None
            if hand is not None and teleop2 is not None:
                hand_cmd = np.asarray(teleop2.get_state()["joint_q"], dtype=float)

            action = env.robot.build_action(arm_target_pose, gripper_cmd=gripper_cmd, hand_cmd=hand_cmd)
            env.move(action, t_cmd_target + args.arm_latency)

            step = env.get_state(action=action)
            if env.recorder and recording:
                env.recorder.record_step(step)
            if teleop_keyboard or teleop_clicker:
                recording = handle_recorder_save(env, keys, previous_keys, recording)
            if visualizer:
                visualizer.log_frame("world/teleop_target", arm_target_pose, axis_length=0.08)
                visualizer.log_env_state("env", step)

            if it % freq == 0:
                print(
                    f"t: {t_cycle_end - t_start:.1f}s",
                    "|",
                    "REC" if recording else "idle (press 'n')",
                    "|",
                    "ENGAGED" if retargeter.engaged else f"released (press '{args.clutch_key}' to engage)",
                    "|",
                    f"tracking: {'ok' if pose_valid else 'LOST'}",
                    "|",
                    f"tcp: {np.round(eef_pose[:3], 3).tolist()}",
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


def main(args):
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
                (teleop2_client() if teleop2_client else nullcontext()) as teleop2,
                (keyboard_client() if keyboard_client else nullcontext()) as teleop_keyboard,
                (clicker_client() if clicker_client else nullcontext()) as teleop_clicker,
                (visualizer_client() if visualizer_client else nullcontext()) as visualizer,
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
                    )
                except KeyboardInterrupt:
                    pass


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
