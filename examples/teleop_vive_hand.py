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
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    WorkspaceLimits,
    dominant_direction,
    horizontal_angle_between,
    matrix_to_pose,
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
# Clicker mouse side buttons, plus raw evdev BTN_SIDE / BTN_EXTRA / BACK / FORWARD.
_SCALE_DOWN = frozenset({0x224, 0x113, 0x116})
_SCALE_UP = frozenset({0x225, 0x114, 0x115})


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
    if special & _SCALE_DOWN:
        keys.add("scale_down")
    if special & _SCALE_UP:
        keys.add("scale_up")
    return keys


def merge_held_keys(*keyboard_states) -> set[str]:
    """Union key sets from one or more keyboard nodes."""
    keys: set[str] = set()
    for state in keyboard_states:
        if state is not None:
            keys |= held_keys(state)
    return keys


def step_motion_scale(current: float, *, up: bool, step: float, min_scale: float, max_scale: float) -> float:
    """Multiply or divide a motion scale by ``step``, then clamp."""
    next_value = current * step if up else current / step
    return float(min(max_scale, max(min_scale, next_value)))


def sync_scaled_speeds(
    retargeter: ClutchRetargeter,
    *,
    base_max_pos_speed: float,
    base_max_rot_speed: float,
    arm=None,
    base_max_motor_speed: float | None = None,
) -> None:
    """Set speed ceilings to the unity-scale bases times the current motion scale."""
    retargeter.max_pos_speed = base_max_pos_speed * retargeter.pos_scale
    retargeter.max_rot_speed = base_max_rot_speed * retargeter.rot_scale
    if arm is None:
        return
    if hasattr(arm, "max_pos_speed"):
        arm.max_pos_speed = retargeter.max_pos_speed
    if hasattr(arm, "max_rot_speed"):
        arm.max_rot_speed = retargeter.max_rot_speed
    if base_max_motor_speed is not None and hasattr(arm, "max_motor_speed"):
        arm.max_motor_speed = base_max_motor_speed * retargeter.pos_scale


def apply_live_scale(
    retargeter: ClutchRetargeter,
    *,
    up: bool,
    step: float,
    min_scale: float,
    max_scale: float,
    tracker_pose: np.ndarray | None,
    pose_valid: bool,
    base_max_pos_speed: float | None = None,
    base_max_rot_speed: float | None = None,
    arm=None,
    base_max_motor_speed: float | None = None,
) -> tuple[float, float]:
    """Step translation and rotation scale by the same relative factor.

    Rebases the clutch origin when engaged so the TCP does not jump. Both
    scales share one step so a 20% finer translation is also 20% finer wrist
    motion. Speed ceilings, if bases are given, track the same factor.
    """
    new_pos = step_motion_scale(
        retargeter.pos_scale, up=up, step=step, min_scale=min_scale, max_scale=max_scale
    )
    new_rot = step_motion_scale(
        retargeter.rot_scale, up=up, step=step, min_scale=min_scale, max_scale=max_scale
    )
    changed = new_pos != retargeter.pos_scale or new_rot != retargeter.rot_scale
    retargeter.pos_scale = new_pos
    retargeter.rot_scale = new_rot
    if changed:
        if base_max_pos_speed is not None and base_max_rot_speed is not None:
            sync_scaled_speeds(
                retargeter,
                base_max_pos_speed=base_max_pos_speed,
                base_max_rot_speed=base_max_rot_speed,
                arm=arm,
                base_max_motor_speed=base_max_motor_speed,
            )
        if retargeter.engaged and pose_valid and tracker_pose is not None:
            retargeter.rebase(tracker_pose)
    return new_pos, new_rot


def term(*args, **kwargs) -> None:
    """Print to the terminal; a full non-blocking stdout must not kill teleop."""
    try:
        print(*args, **kwargs)
    except BlockingIOError:
        pass


def make_stdout_blocking() -> None:
    """Undo non-blocking stdout that some terminals leave on stdin/stdout."""
    try:
        sys.stdout.reconfigure(blocking=True)
    except Exception:
        pass


def recorder_ready_for_new(recorder) -> bool:
    """True when the previous file is closed (encode may still be running)."""
    if recorder is None:
        return False
    getter = getattr(recorder, "get_state", None)
    if getter is None:
        return True
    status = getter()
    if not isinstance(status, dict):
        return True
    return bool(status.get("is_closed", False))


def handle_recorder_start(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Start a trajectory on 'n' so this tick is the first recorded frame."""
    if not env.recorder or recording:
        return recording
    if "n" not in keys - previous:
        return recording
    if not recorder_ready_for_new(env.recorder):
        logger.warning("Still recording — press 's' before starting the next trajectory")
        return recording
    env.recorder.new_trajectory(wait=False)
    term("\n ============================================= ")
    logger.info("Started new trajectory recording")
    term("============================================= \n")
    return True


def handle_recorder_save(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Save on 's' after this tick has been written, so the file ends with 's'."""
    if "s" not in keys - previous:
        return recording
    if not env.recorder:
        return recording
    if not recording:
        logger.warning("Not recording — press 'n' first, then 's' to save and home")
        return recording
    env.recorder.save(wait=False)
    logger.info("Saved trajectory in background (press 'n' for the next one)")
    term("============================================= \n")
    return False


HOME_POS_EPS = 0.001
HOME_ROT_EPS = 0.01
HOME_JOINT_EPS = math.radians(1.0)  # official moveJ lands well under 2°
HOME_TIMEOUT_S = 12.0
HOME_SETTLE_TICKS = 20
HOME_MIN_DURATION_S = 2.0
HOME_CANCEL_DURATION_S = 1.0
HOME_POST_MOVE_S = 0.5  # wait after moveJ before VelCmd (mode switch trips ESTOP)
HOME_QD_EPS = 0.05  # rad/s, treat as stopped
HOME_Q_FILENAME = "home_q.json"


def capture_home_state(arm, gripper_cmd: float) -> dict:
    """Record the live joints / TCP as the episode start pose."""
    state = arm.get_state()
    return {
        "joint_q": np.asarray(state["joint_q"], dtype=float).copy(),
        "eef_pose": np.asarray(state["eef_pose"], dtype=float).copy(),
        "gripper": float(gripper_cmd),
    }


def home_q_path(args) -> Path | None:
    """JSON next to the recorder folder, or None if recording is off."""
    recorder_cfg = getattr(args, "recorder_cfg", None)
    base = getattr(recorder_cfg, "path", None) if recorder_cfg is not None else None
    if not base:
        return None
    path = Path(base)
    if path.suffix == ".vla":
        path = path.parent
    return path / HOME_Q_FILENAME


def write_home_state(path: Path, home: dict) -> None:
    """Persist home joints so later eval can reuse the same start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_q": np.asarray(home["joint_q"], dtype=float).tolist(),
        "eef_pose": np.asarray(home["eef_pose"], dtype=float).tolist(),
        "gripper": float(home["gripper"]),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_home_state(path: Path, gripper_fallback: float) -> dict:
    """Load a previously written home snapshot."""
    payload = json.loads(path.read_text())
    return {
        "joint_q": np.asarray(payload["joint_q"], dtype=float).copy(),
        "eef_pose": np.asarray(payload["eef_pose"], dtype=float).copy(),
        "gripper": float(payload.get("gripper", gripper_fallback)),
    }


def step_pose_toward(
    current: np.ndarray,
    goal: np.ndarray,
    *,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
) -> np.ndarray:
    """Move a pose6 ``[xyz, rotvec]`` toward ``goal`` with per-tick speed caps.

    Used for Cartesian creep tests; live Kassow homing uses discrete ``moveJ``.
    """
    out = np.asarray(current, dtype=float).reshape(-1).copy()
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if out.shape[0] < 6 or goal.shape[0] < 6:
        return out
    delta = goal[:3] - out[:3]
    dist = float(np.linalg.norm(delta))
    max_pos = float(max_pos_speed) * float(dt)
    if dist > max_pos > 0.0:
        out[:3] = out[:3] + delta * (max_pos / dist)
    else:
        out[:3] = goal[:3]
    r_cur = st.Rotation.from_rotvec(out[3:6])
    r_goal = st.Rotation.from_rotvec(goal[3:6])
    rotvec = (r_cur.inv() * r_goal).as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    max_rot = float(max_rot_speed) * float(dt)
    if ang > max_rot > 0.0:
        out[3:6] = (r_cur * st.Rotation.from_rotvec(rotvec * (max_rot / ang))).as_rotvec()
    else:
        out[3:6] = goal[3:6]
    return out


def pose_home_error(current: np.ndarray, goal: np.ndarray) -> tuple[float, float]:
    """Return (position L2 in m, rotation angle in rad) from ``current`` to ``goal``."""
    cur = np.asarray(current, dtype=float).reshape(-1)
    tgt = np.asarray(goal, dtype=float).reshape(-1)
    if cur.shape[0] < 6 or tgt.shape[0] < 6:
        return float("inf"), float("inf")
    pos = float(np.linalg.norm(tgt[:3] - cur[:3]))
    rotvec = (st.Rotation.from_rotvec(cur[3:6]).inv() * st.Rotation.from_rotvec(tgt[3:6])).as_rotvec()
    return pos, float(np.linalg.norm(rotvec))


def pose_at_home(
    current: np.ndarray,
    goal: np.ndarray,
    pos_eps: float = HOME_POS_EPS,
    rot_eps: float = HOME_ROT_EPS,
) -> bool:
    """True when TCP xyz and orientation are within the home tolerances."""
    pos, rot = pose_home_error(current, goal)
    return pos <= float(pos_eps) and rot <= float(rot_eps)


def step_q_toward(
    current: np.ndarray,
    goal: np.ndarray,
    *,
    dt: float,
    max_joint_speed: float,
) -> np.ndarray:
    """Move joints toward ``goal`` without any joint exceeding ``max_joint_speed``."""
    out = np.asarray(current, dtype=float).reshape(-1).copy()
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if out.shape != goal.shape:
        return out
    delta = goal - out
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    max_step = float(max_joint_speed) * float(dt)
    if max_abs > max_step > 0.0:
        return out + delta * (max_step / max_abs)
    return goal.copy()


def joint_home_error(current: np.ndarray, goal: np.ndarray) -> float:
    """Return max-abs joint error in rad."""
    cur = np.asarray(current, dtype=float).reshape(-1)
    tgt = np.asarray(goal, dtype=float).reshape(-1)
    if cur.size == 0 or cur.shape != tgt.shape:
        return float("inf")
    return float(np.max(np.abs(tgt - cur)))


def joints_at_home(
    current: np.ndarray,
    goal: np.ndarray,
    eps: float = HOME_JOINT_EPS,
) -> bool:
    """True when every joint is within ``eps`` rad of the saved home q."""
    return joint_home_error(current, goal) <= float(eps)


def home_move_duration(
    current: np.ndarray,
    goal: np.ndarray,
    joint_speed: float,
    min_s: float = HOME_MIN_DURATION_S,
    max_s: float = HOME_TIMEOUT_S,
) -> float:
    """Time-tracking window for KORD ``moveJ`` from ``current`` to ``goal``."""
    err = joint_home_error(current, goal)
    speed = max(float(joint_speed), 1e-3)
    return float(np.clip(err / speed, min_s, max_s))


def queue_discrete_movej(arm, target_q: np.ndarray, duration_s: float) -> bool:
    """Send one official KORD ``moveJ``. False if the arm has no discrete API."""
    fn = getattr(arm, "moveJ_discrete", None)
    if fn is None:
        logger.error("Arm has no moveJ_discrete; cannot home with KORD moveJ")
        return False
    fn(np.asarray(target_q, dtype=float), float(duration_s))
    return True


def begin_kord_home(
    arm,
    joint_q: np.ndarray,
    home: dict,
    home_joint_speed: float,
    t_sample: float,
) -> tuple[bool, bool, float, float]:
    """Queue KORD ``moveJ`` to the saved home q.

    Returns:
        (homing, arm_streaming, move_until, timeout). ``homing`` is False when
        already there or the arm cannot discrete-move. Do not resume VelCmd
        until ``move_until`` — interrupting moveJ with directJControl trips ESTOP.
    """
    q_goal = np.asarray(home["joint_q"], dtype=float)
    q_err = joint_home_error(joint_q, q_goal)
    if joints_at_home(joint_q, q_goal):
        logger.info("Already at home (q_err={:.2f} deg)", math.degrees(q_err))
        return False, True, 0.0, 0.0
    duration = home_move_duration(joint_q, q_goal, home_joint_speed)
    if not queue_discrete_movej(arm, q_goal, duration):
        return False, True, 0.0, 0.0
    move_until = float(t_sample) + duration
    logger.info(
        "Returning to home q={} (q_err={:.2f} deg) via KORD moveJ in {:.1f}s ({:.2f} rad/s)",
        np.round(q_goal, 3).tolist(),
        math.degrees(q_err),
        duration,
        home_joint_speed,
    )
    return True, False, move_until, move_until + 3.0


def joints_stopped(arm_state: dict, eps: float = HOME_QD_EPS) -> bool:
    """True when measured joint speed is below ``eps`` (or qd is missing)."""
    qd = np.asarray(arm_state.get("joint_qd", []), dtype=float).reshape(-1)
    if qd.size == 0:
        return True
    return float(np.max(np.abs(qd))) <= float(eps)


def stop_discrete_home(arm, joint_q: np.ndarray, t_now: float) -> float:
    """Interrupt KORD ``moveJ`` at the current joints. Returns holdoff-until."""
    if queue_discrete_movej(arm, joint_q, HOME_CANCEL_DURATION_S):
        logger.info("Stopping KORD moveJ at current joints ({:.1f}s)", HOME_CANCEL_DURATION_S)
    return float(t_now) + HOME_CANCEL_DURATION_S + 0.2


def joint_path_pose(eef_start: np.ndarray, fk_start: np.ndarray, fk_cmd: np.ndarray) -> np.ndarray:
    """Apply the FK motion ``fk_start → fk_cmd`` onto the measured start TCP."""
    t_start = pose_to_matrix(eef_start)
    t_fk0 = pose_to_matrix(fk_start)
    t_fk = pose_to_matrix(fk_cmd)
    return matrix_to_pose(t_start @ np.linalg.inv(t_fk0) @ t_fk)


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
    make_stdout_blocking()
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

    home_file = home_q_path(args)
    capture_home = bool(getattr(args, "capture_home", False))
    if home_file is not None and home_file.is_file() and not capture_home:
        home = load_home_state(home_file, gripper_cmd)
        logger.info("Loaded home from {}  tcp={}", home_file, np.round(home["eef_pose"][:3], 3).tolist())
    else:
        home = capture_home_state(arm, gripper_cmd)
        if home_file is not None:
            write_home_state(home_file, home)
            logger.info("Home snapshot written to {}", home_file)
    logger.info(
        "Home q={}  tcp={}",
        np.round(home["joint_q"], 3).tolist(),
        np.round(home["eef_pose"][:3], 3).tolist(),
    )
    go_home_on_save = bool(getattr(args, "go_home_on_save", False))
    home_joint_speed = float(getattr(args, "home_joint_speed", 0.15) or 0.15)
    homing = False
    arm_streaming = True
    home_holdoff_until = 0.0
    home_move_until = 0.0
    home_deadline = 0.0
    home_settle = 0
    pending_start_home = bool(go_home_on_save)

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
    if go_home_on_save:
        print(
            f"  's' while clutched is fine: clutch drops, then KORD moveJ to "
            f"{np.round(home['joint_q'], 3).tolist()} "
            f"at {home_joint_speed:.2f} rad/s (accept {math.degrees(HOME_JOINT_EPS):.0f} deg)."
        )
        print("  After Enter the arm homes to home_q.json. Wait for Home reached, then n → c.")
        print("  'n' works immediately after 's' (encode runs in the background). Wait for Home reached, then clutch.")
    print(f"  TCP now: {np.round(arm_target_pose[:3], 3).tolist()}   yaw offset: {yaw_deg:.1f} deg")
    print(
        f"  limits at scale 1.0:  {base_max_pos_speed} m/s, {base_max_rot_speed} rad/s, "
        f"radius {args.min_radius}-{args.max_radius} m, z >= {args.min_z} m"
    )
    print(
        f"  motion scale: pos {retargeter.pos_scale:.2f}  rot {retargeter.rot_scale:.2f}  "
        f"→ {retargeter.max_pos_speed:.2f} m/s, {retargeter.max_rot_speed:.2f} rad/s  "
        f"(mouse back/forward ×{scale_step:g}, clamp {min_scale:g}–{max_scale:g})"
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
            if pending_start_home:
                pending_start_home = False
                retargeter.hold_pose(eef_pose)
                homing, arm_streaming, home_move_until, home_deadline = begin_kord_home(
                    arm, joint_q, home, home_joint_speed, t_sample
                )
                home_settle = 0
            if pose_valid:
                t_last_valid_pose = t_sample

            # Clutch and orientation toggles, on key-down only
            pressed = keys - previous_keys
            if args.clutch_key in pressed:
                if homing:
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

            if retargeter.engaged and pose_valid and not homing:
                arm_target_pose = retargeter.target_pose(pose_to_matrix(tracker_state["tracker_pose"]), dt)
                lag = float(np.linalg.norm(arm_target_pose[:3] - eef_pose[:3]))
                if lag > args.max_lag:
                    retargeter.disengage()
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
                    logger.warning(f"Command ran {lag:.3f} m ahead of the arm, clutch released")
            elif homing:
                q_goal = np.asarray(home["joint_q"], dtype=float)
                q_err = joint_home_error(joint_q, q_goal)
                pos_err, _ = pose_home_error(eef_pose, home["eef_pose"])
                alarm = int(arm_state.get("alarm_code", 0) or 0)
                move_done = t_sample >= home_move_until
                landed = joints_at_home(joint_q, q_goal) and move_done and joints_stopped(arm_state)
                if landed:
                    home_settle += 1
                else:
                    home_settle = 0
                timed_out = t_sample > home_deadline
                if alarm:
                    homing = False
                    arm_streaming = False
                    home_holdoff_until = t_sample + HOME_CANCEL_DURATION_S
                    logger.warning(
                        "Homing aborted, KORD alarm at q_err={:.2f} deg",
                        math.degrees(q_err),
                    )
                elif home_settle >= HOME_SETTLE_TICKS or timed_out:
                    homing = False
                    arm_streaming = False
                    home_holdoff_until = t_sample + HOME_POST_MOVE_S
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
                    if timed_out:
                        logger.warning(
                            "Homing timed out at q_err={:.2f} deg, {:.1f} mm (target q={})",
                            math.degrees(q_err),
                            pos_err * 1e3,
                            np.round(q_goal, 3).tolist(),
                        )
                    else:
                        logger.info(
                            "Home reached (q_err={:.2f} deg, {:.1f} mm, tcp={})",
                            math.degrees(q_err),
                            pos_err * 1e3,
                            np.round(eef_pose[:3], 3).tolist(),
                        )
                else:
                    arm_streaming = False
                    retargeter.hold_pose(eef_pose)
                    arm_target_pose = eef_pose.copy()
            else:
                # Hold station and keep the reference synced to where the arm
                # actually is, so re-engaging never produces a jump.
                retargeter.hold_pose(eef_pose)
                arm_target_pose = eef_pose.copy()

            if (not homing) and t_sample >= home_holdoff_until:
                arm_streaming = True

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
            if go_home_on_save and was_recording and not recording:
                if retargeter.engaged:
                    retargeter.disengage()
                    logger.info("Clutch released for home")
                retargeter.hold_pose(eef_pose)
                homing, arm_streaming, home_move_until, home_deadline = begin_kord_home(
                    arm, joint_q, home, home_joint_speed, t_sample
                )
                home_settle = 0
            if visualizer:
                visualizer.log_frame("world/teleop_target", arm_target_pose, axis_length=0.08)
                visualizer.log_env_state("env", step)

            if it % freq == 0:
                if homing:
                    q_err = joint_home_error(joint_q, home["joint_q"])
                    rec = "+REC" if recording else ""
                    mode = f"HOMING{rec} q={math.degrees(q_err):.1f}deg"
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
