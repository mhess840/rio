import json
import math
import multiprocessing as mp
import sys
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import scipy.spatial.transform as st
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.interpolators import ActionInterpolator
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env
from rio.gripper import GripperEncoding
from rio.policies.pi05 import principal_rotvec, snap_gripper

try:
    import torch
    from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig
except ImportError:
    torch = None
    ActionQueue = None
    LatencyTracker = None
    RTCConfig = None

# Horizon must exceed inference delay (~18). Keep in sync with rio/policies/pi05.py.
# Threshold 22 = delay plus slack so more of each 50-step chunk is executed.
RTC_QUEUE_THRESHOLD = 18
RTC_EXECUTION_HORIZON = 18
RTC_MAX_GUIDANCE_WEIGHT = 10.0
DEFAULT_DIAG_DIR = Path("/tmp/pi05_diag")
# First policy waypoint is often 5–15 cm / ~0.1 rad off the live pose. Closing
# that in the interpolator's 50 ms blend trips Kassow JTORQUE from idle.
STARTUP_JOINT_SPEED = 0.12  # rad/s
STARTUP_POS_SPEED = 0.08  # m/s
STARTUP_ROT_SPEED = 0.25  # rad/s
STARTUP_JOINT_EPS = 0.02  # rad L2
STARTUP_POS_EPS = 0.005  # m

_SUPPORTED_CONTROLLERS = frozenset({"JOINT_VEL", "JOINT_POS", "TASK_POS", "TASK_POS_IK"})


@dataclass
class RolloutSafetyCfg:
    """Fail-closed limits for policy playback, independent of robot firmware limits."""

    max_position_error_m: float = 0.05
    max_rotation_error_rad: float = 0.35
    max_joint_error_rad: float = 0.25
    max_state_age_s: float = 0.25
    inference_timeout_s: float = 2.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def require_arm_state_safe(state: dict, command: np.ndarray, *, now: float, joint_serve: bool, limits: RolloutSafetyCfg):
    """Reject alarms, stale feedback, invalid values, and excessive command tracking error."""
    if not isinstance(state, dict):
        raise RuntimeError("Arm feedback is unavailable")
    try:
        alarm = int(state.get("alarm_code", 0))
        timestamp = float(state["timestamp"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("Arm feedback has no valid alarm/timestamp information") from error
    if alarm:
        raise RuntimeError(f"Arm controller alarm 0x{alarm:08x}")
    age = float(now) - timestamp
    if not np.isfinite(age) or age < -0.01 or age > limits.max_state_age_s:
        raise RuntimeError(f"Arm feedback is stale or has an invalid timestamp (age={age:.3f}s)")
    target = np.asarray(command, dtype=np.float64)
    expected = 8 if joint_serve else 7
    if target.shape != (expected,) or not np.isfinite(target).all():
        raise RuntimeError(f"Invalid {expected}D arm command")
    key = "joint_q" if joint_serve else "eef_pose"
    measured = np.asarray(state.get(key, []), dtype=np.float64).reshape(-1)
    dimensions = 7 if joint_serve else 6
    if measured.shape != (dimensions,) or not np.isfinite(measured).all():
        raise RuntimeError(f"Arm feedback has no valid {key}")
    if joint_serve:
        error = float(np.max(np.abs(target[:7] - measured)))
        if error > limits.max_joint_error_rad:
            raise RuntimeError(f"Joint tracking error {error:.3f} rad exceeds the rollout limit")
    else:
        position_error = float(np.linalg.norm(target[:3] - measured[:3]))
        if position_error > limits.max_position_error_m:
            raise RuntimeError(f"TCP position tracking error {position_error:.3f} m exceeds the rollout limit")
        difference = cast(st.Rotation, st.Rotation.from_rotvec(target[3:6]) * st.Rotation.from_rotvec(measured[3:6]).inv())
        rotation_error = float(difference.magnitude())
        if rotation_error > limits.max_rotation_error_rad:
            raise RuntimeError(f"TCP rotation tracking error {rotation_error:.3f} rad exceeds the rollout limit")


def resolve_control_tick(t_start: float, iteration: int, dt: float, now: float) -> tuple[int, float, float]:
    """Skip missed deadlines without emitting a burst of overdue control commands."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("The control period must be finite and positive")
    cycle_end = t_start + (iteration + 1) * dt
    if cycle_end <= now:
        iteration = max(iteration, int((now - t_start) / dt) + 1)
        cycle_end = t_start + (iteration + 1) * dt
    return iteration, cycle_end, cycle_end + dt


def validate_action_chunk(actions, action_dim: int) -> np.ndarray:
    """Require a non-empty finite action chunk with the configured feature contract."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or not len(chunk) or chunk.shape[1] != action_dim or not np.isfinite(chunk).all():
        raise RuntimeError(f"Policy returned an invalid action chunk; expected finite (T, {action_dim}), got {chunk.shape}")
    return chunk


def stop_arm_motion(arm) -> bool:
    """Request a stream stop without clearing alarms; unsupported drivers must be shut down."""
    stop = getattr(arm, "stop_motion", None)
    if not callable(stop):
        return False
    try:
        stop()
    except Exception:
        logger.exception("Could not stop the arm stream; closing the hardware session")
        return False
    return True


def move_action_checked(env, command, target_time: float, *, joint_serve: bool, limits: RolloutSafetyCfg):
    """Check fresh feedback immediately before dispatching one arm/gripper command."""
    state = env.robot.arm.get_state()
    require_arm_state_safe(state, command, now=time.now(), joint_serve=joint_serve, limits=limits)
    if not np.isfinite(target_time):
        raise RuntimeError("Invalid command deadline")
    env.move(command, t_cmd_target=target_time)


def apply_serve_control(cmd: str, pause: threading.Event, quit_ev: threading.Event) -> None:
    """Map a stdin line to pause/resume/quit. Empty line toggles run vs pause."""
    cmd = cmd.strip().lower()
    if cmd in ("q", "quit"):
        quit_ev.set()
        return
    if cmd in ("p", "pause"):
        pause.set()
        return
    if pause.is_set():
        pause.clear()
    else:
        pause.set()


def _serve_control_watcher(pause: threading.Event, quit_ev: threading.Event) -> None:
    """Enter toggles run/pause; ``q`` quits. Daemon thread so Ctrl+C can still exit."""
    while not quit_ev.is_set():
        line = sys.stdin.readline()
        if line == "":
            quit_ev.set()
            return
        apply_serve_control(line, pause, quit_ev)


def require_arm_controller(arm) -> str:
    """Accept joint-space and Cartesian arm controllers used by policy rollouts."""
    name = getattr(getattr(arm, "robot_controller", None), "name", None)
    if name not in _SUPPORTED_CONTROLLERS:
        raise AssertionError(f"Unsupported arm controller {name!r}; expected one of {sorted(_SUPPORTED_CONTROLLERS)}")
    return name


def chunk_size_from_ckpt(policy_path: str, default: int = 16) -> int:
    """Read ``n_action_steps`` / ``chunk_size`` from a LeRobot checkpoint."""
    cfg_file = Path(policy_path) / "config.json"
    if not cfg_file.is_file():
        return default
    data = json.loads(cfg_file.read_text())
    n = data.get("n_action_steps") or data.get("chunk_size")
    return int(n) if n else default


def resolve_n_execute(policy_path: str, override: int | None, default: int = 16) -> int:
    """How many predicted steps to play before the next request.

    The checkpoint still predicts ``chunk_size`` (often 50). Playing all of them
    lets later steps reverse toward grasp/home. ``override<=0`` means the ckpt
    ``n_action_steps``.
    """
    if override is not None and int(override) > 0:
        return int(override)
    return chunk_size_from_ckpt(policy_path, default=default)


def slice_action_chunk(
    processed: np.ndarray, original: np.ndarray, n_execute: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first ``n_execute`` waypoints of a predicted chunk."""
    n = min(len(processed), len(original))
    processed = processed[:n]
    original = original[:n]
    if n_execute > 0:
        n = min(n, int(n_execute))
        processed = processed[:n]
        original = original[:n]
    return processed, original


def merge_delay_steps(
    latency_s: float,
    time_per_action: float,
    *,
    leftover_present: bool,
    consumed_steps: int | None = None,
) -> int:
    """RTC prefix skip: only drop steps the robot actually played during inference.

    LeRobot still applies ``ceil(latency/dt)`` on an empty queue. The first chunk
    then starts ~1 s into the plan (20 Hz × 1 s ≈ 20 steps) and the first
    waypoint is already several centimetres away.
    """
    if not leftover_present:
        return 0
    if consumed_steps is not None:
        return max(0, int(consumed_steps))
    if latency_s <= 0 or time_per_action <= 0:
        return 0
    return math.ceil(latency_s / time_per_action)


def chunk_hold_pending(now: float, hold_until: float | None) -> bool:
    """True while a between-chunk hold has not elapsed."""
    return hold_until is not None and now < hold_until


def should_send_observation(
    *,
    processing_obs: bool,
    remaining: int,
    use_rtc: bool = False,
    queue_threshold: int = RTC_QUEUE_THRESHOLD,
) -> bool:
    """Match LeRobot ``RTCInferenceEngine``: infer when ``qsize() <= queue_threshold``.

    Without RTC, wait until the queue is empty so a new absolute-TCP chunk is not
    played from index 0 over a live plan.
    """
    if processing_obs:
        return False
    if remaining <= 0:
        return True
    if not use_rtc:
        return False
    return remaining <= int(queue_threshold)


def _as_action_np(
    action,
    action_dim: int,
    *,
    binarize_gripper: bool,
    fold_rotvec: bool = True,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """CPU float32 action; optionally fold rotvec and snap AG95 after interpolation."""
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    x = np.asarray(action, dtype=np.float32).reshape(-1)[:action_dim]
    if fold_rotvec and x.shape[0] >= 6:
        x[3:6] = principal_rotvec(x[3:6][None, :])[0]
    if x.shape[0] >= 7:
        if binarize_gripper:
            x[-1] = snap_gripper(
                np.asarray([x[-1]], dtype=np.float32),
                gripper_close=gripper_close,
                gripper_open=gripper_open,
                gripper_threshold=gripper_threshold,
            )[0]
        else:
            x[-1] = np.clip(x[-1], 0.0, 1.0)
    return x


def _current_gripper_command(
    env, arm_state: dict, *, binarize_gripper: bool, gripper_close: float, gripper_open: float, gripper_threshold: float | None
) -> float:
    gripper = getattr(env.robot, "_gripper", None) or getattr(env.robot, "gripper", None)
    state = arm_state if gripper is None else gripper.get_state()
    value = state.get("gripper_commanded_position")
    if value is None:
        value = state.get("gripper_position")
    if value is None:
        if gripper is None:
            return gripper_close
        raise RuntimeError("The gripper has no valid initial opening or command")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("The gripper has an invalid initial opening")
    if binarize_gripper:
        value = float(
            snap_gripper(
                np.asarray([value]), gripper_close=gripper_close, gripper_open=gripper_open, gripper_threshold=gripper_threshold
            )[0]
        )
    return value


def measured_tcp_action(
    env,
    action_dim: int,
    *,
    binarize_gripper: bool,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Live TCP as a 7D action. Used once to prime the interpolator."""
    state = env.robot.arm.get_state()
    eef = np.asarray(state["eef_pose"], dtype=np.float32).reshape(-1)
    x = np.zeros((action_dim,), dtype=np.float32)
    n = min(6, int(eef.shape[0]), action_dim)
    x[:n] = eef[:n]
    if n >= 6:
        x[3:6] = principal_rotvec(x[3:6][None, :])[0]
    if action_dim > 6:
        x[6] = _current_gripper_command(
            env,
            state,
            binarize_gripper=binarize_gripper,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )
    return x


def measured_joint_action(
    env,
    action_dim: int,
    *,
    binarize_gripper: bool,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Live joints + gripper as an 8D action. Used once to prime the interpolator."""
    state = env.robot.arm.get_state()
    q = np.asarray(state["joint_q"], dtype=np.float32).reshape(-1)
    x = np.zeros((action_dim,), dtype=np.float32)
    n = min(7, int(q.shape[0]), action_dim)
    x[:n] = q[:n]
    if action_dim > 7:
        x[-1] = _current_gripper_command(
            env,
            state,
            binarize_gripper=binarize_gripper,
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )
    return x


def limit_tcp_action_step(
    target: np.ndarray,
    previous: np.ndarray,
    *,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
) -> np.ndarray:
    """Move at most one speed-limited Cartesian step from ``previous`` toward ``target``.

    Gripper (and any extra) slots are copied from ``target``. Without this, the
    interpolator's first 50 ms blend of an 18 cm policy jump is several m/s and
    trips Kassow protective stop.
    """
    out = np.asarray(target, dtype=np.float32).copy()
    prev = np.asarray(previous, dtype=np.float32)
    n = min(3, out.shape[0], prev.shape[0])
    delta = out[:n] - prev[:n]
    dist = float(np.linalg.norm(delta))
    max_pos = float(max_pos_speed) * float(dt)
    if dist > max_pos > 0.0:
        out[:n] = prev[:n] + delta * (max_pos / dist)
    if out.shape[0] >= 6 and prev.shape[0] >= 6:
        r_prev = st.Rotation.from_rotvec(prev[3:6].astype(np.float64))
        r_tgt = st.Rotation.from_rotvec(out[3:6].astype(np.float64))
        dR = cast(st.Rotation, r_tgt * r_prev.inv())
        ang = float(dR.magnitude())
        max_ang = float(max_rot_speed) * float(dt)
        if ang > max_ang > 0.0:
            dR = st.Rotation.from_rotvec(dR.as_rotvec() * (max_ang / ang))
        out[3:6] = principal_rotvec(np.asarray(cast(st.Rotation, dR * r_prev).as_rotvec(), dtype=np.float32)[None, :])[0]
    return out


def limit_joint_action_step(
    target: np.ndarray,
    previous: np.ndarray,
    *,
    dt: float,
    max_joint_speed: float,
    n_joints: int = 7,
) -> np.ndarray:
    """Clip each joint to ``max_joint_speed * dt`` from ``previous``; copy gripper."""
    out = np.asarray(target, dtype=np.float32).copy()
    prev = np.asarray(previous, dtype=np.float32)
    n = min(n_joints, out.shape[0], prev.shape[0])
    max_step = float(max_joint_speed) * float(dt)
    if max_step > 0.0:
        out[:n] = prev[:n] + np.clip(out[:n] - prev[:n], -max_step, max_step)
    return out


def command_tracking_error(current: np.ndarray, target: np.ndarray, *, joint_serve: bool) -> float:
    """L2 error on the commanded joints or TCP xyz."""
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1)
    n = min(7 if joint_serve else 3, cur.shape[0], tgt.shape[0])
    if n <= 0:
        return 0.0
    return float(np.linalg.norm(tgt[:n] - cur[:n]))


def first_waypoint_needs_creep(error: float, *, joint_serve: bool) -> bool:
    """True when the first policy waypoint is too far to blend in 50 ms."""
    eps = STARTUP_JOINT_EPS if joint_serve else STARTUP_POS_EPS
    return float(error) > float(eps)


def is_joint_serve(args) -> bool:
    """True when the station sends ``moveJ`` / 8D joint actions."""
    space = str(getattr(args, "action_space", "") or "").lower()
    policy_space = str(getattr(getattr(args, "policy_cfg", None), "action_space", "") or "").lower()
    return space == "joint_pos" or policy_space == "joints"


# TODO: this should be moved into each individual policy
def create_obs(env, *, gripper_state_source: str = "measured", max_camera_age_s: float | None = None):
    formatted_obs = {}
    state = env.get_state()
    for key in state.observation.cameras:
        camera = state.observation.cameras[key]
        if max_camera_age_s is not None:
            timestamp = (getattr(camera, "meta", None) or {}).get("camera_receive_timestamp")
            if timestamp is None or not np.isfinite(timestamp):
                raise RuntimeError(f"Camera {key} has no valid capture/receive timestamp")
            age = time.now() - float(timestamp)
            if age < -0.01 or age > max_camera_age_s:
                raise RuntimeError(f"Camera {key} is stale (age={age:.3f}s)")
        formatted_obs[key] = camera.rgb
    joints = state.observation.proprio_joints
    gripper = state.observation.gripper_position
    if gripper_state_source == "commanded":
        gripper = getattr(state.observation, "gripper_commanded_position", None)
        if gripper is None:
            raise RuntimeError("The gripper driver does not publish commanded position; update rio-hw or select measured state")
    elif gripper_state_source != "measured":
        raise ValueError("gripper_state_source must be 'measured' or 'commanded'")
    formatted_obs["proprio_joints"] = joints
    formatted_obs["gripper_position"] = gripper
    if joints is not None:
        grip = 0.0 if gripper is None else float(gripper)
        formatted_obs["proprio"] = np.concatenate(
            [np.asarray(joints, dtype=np.float32).reshape(-1), np.asarray([grip], dtype=np.float32)]
        )
    return formatted_obs


def _dump_live_cameras(obs: dict, dest: Path) -> None:
    """Write the first live RGB frames so they can be compared to the train videos."""
    dest.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for key in ("camera_1", "camera_2"):
        img = obs.get(key)
        if img is None:
            logger.warning("Live camera {} is missing from the observation", key)
            continue
        arr = np.asarray(img)
        mean_rgb = np.asarray(arr, dtype=np.float32).mean(axis=(0, 1)) if arr.ndim == 3 else arr.mean()
        logger.info(
            "Live {} shape={} dtype={} min={} max={} meanRGB={}",
            key,
            arr.shape,
            arr.dtype,
            arr.min() if arr.size else None,
            arr.max() if arr.size else None,
            mean_rgb,
        )
        if arr.ndim == 3 and arr.shape[-1] == 3:
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dest / f"live_{key}.png")
    logger.info("Wrote live camera frames to {}", dest)


def create_diagnostic_session(root: Path, metadata: dict) -> Path:
    """Allocate a unique rollout folder without overwriting earlier CSV or chunk files."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="session_", dir=root))
    (session / "session.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return session


def write_chunk_dump(
    dest: Path,
    idx: int,
    leftover: np.ndarray,
    eef: np.ndarray,
    *,
    delay: int,
    qsize: int,
    latency_s: float,
) -> None:
    """Save one merged leftover chunk plus a CSV row (model plan before IK)."""
    dest.mkdir(parents=True, exist_ok=True)
    chunk_dir = dest / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    leftover = np.asarray(leftover, dtype=np.float32)
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    np.save(chunk_dir / f"chunk_{idx:04d}.npy", leftover)
    first = leftover[0, :3] if leftover.ndim == 2 and leftover.shape[0] else np.zeros(3)
    last = leftover[-1, :3] if leftover.ndim == 2 and leftover.shape[0] else first
    span = float(np.linalg.norm(last - first))
    csv_path = dest / "chunks.csv"
    new_file = not csv_path.is_file()
    with csv_path.open("a") as f:
        if new_file:
            f.write("idx,delay,qsize,latency_s,eef_x,eef_y,eef_z,first_x,first_y,first_z,last_x,last_y,last_z,span_m\n")
        f.write(
            f"{idx},{delay},{qsize},{latency_s:.4f},"
            f"{float(eef[0]):.6f},{float(eef[1]):.6f},{float(eef[2]):.6f},"
            f"{float(first[0]):.6f},{float(first[1]):.6f},{float(first[2]):.6f},"
            f"{float(last[0]):.6f},{float(last[1]):.6f},{float(last[2]):.6f},"
            f"{span:.6f}\n"
        )


def append_cmd_eef(
    dest: Path,
    t: float,
    cmd: np.ndarray,
    eef: np.ndarray,
    remaining: int,
    chunk_idx: int,
    *,
    arm_state: dict | None = None,
    joint_serve: bool = False,
) -> None:
    """Append one 100 Hz command vs measured TCP row."""
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "cmd_vs_eef.csv"
    cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    state = arm_state or {}
    alarm = int(state.get("alarm_code", 0) or 0)
    state_age = time.now() - float(state["timestamp"]) if "timestamp" in state else float("nan")
    if joint_serve:
        header = (
            "t,"
            + ",".join(f"cmd_j{index + 1}" for index in range(7))
            + ","
            + ",".join(f"eef_j{index + 1}" for index in range(7))
            + ",err_joint_rad,remaining,chunk_idx,cmd_gripper,alarm_code,state_age_s\n"
        )
        error = float(np.linalg.norm(cmd[:7] - eef[:7]))
        values = [t, *cmd[:7], *eef[:7], error, remaining, chunk_idx, cmd[-1], alarm, state_age]
    else:
        position_error = float(np.linalg.norm(cmd[:3] - eef[:3]))
        difference = cast(
            st.Rotation,
            st.Rotation.from_rotvec(cmd[3:6].astype(float)) * st.Rotation.from_rotvec(eef[3:6].astype(float)).inv(),
        )
        rotation_error = float(difference.magnitude())
        header = (
            "t,cmd_x,cmd_y,cmd_z,eef_x,eef_y,eef_z,err_m,remaining,chunk_idx,"
            "cmd_rx,cmd_ry,cmd_rz,eef_rx,eef_ry,eef_rz,err_rot_rad,cmd_gripper,alarm_code,state_age_s\n"
        )
        values = [
            t,
            *cmd[:3],
            *eef[:3],
            position_error,
            remaining,
            chunk_idx,
            *cmd[3:6],
            *eef[3:6],
            rotation_error,
            cmd[-1],
            alarm,
            state_age,
        ]
    new_file = not csv_path.is_file()
    with csv_path.open("a") as f:
        if new_file:
            f.write(header)
        f.write(",".join(f"{float(value):.6f}" for value in values) + "\n")


def policy_loop(args, env, policy, visualizer=None):
    require_arm_controller(env.robot.arm)
    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
    logger.info("Waiting for the policy to finish loading (9 GB weights onto GPU)...")
    while not policy.policy_loaded:
        time.sleep(0.5)
    logger.info(
        "Policy ready. Enter starts/resumes; Enter or p pauses (weights stay on GPU); q or Ctrl+C quits."
    )
    pause = threading.Event()
    quit_ev = threading.Event()
    pause.set()
    threading.Thread(target=_serve_control_watcher, args=(pause, quit_ev), daemon=True).start()
    freq = args.freq
    if freq <= 0:
        raise ValueError("The control frequency must be positive")
    dt = 1.0 / freq
    arm_cfg = getattr(args, "arm_cfg", None)
    max_pos_speed = float(getattr(arm_cfg, "max_pos_speed", 0.45) or 0.45)
    max_rot_speed = float(getattr(arm_cfg, "max_rot_speed", 1.0) or 1.0)
    max_joint_speed = float(getattr(arm_cfg, "max_motor_speed", 1.0) or 1.0)
    action_hz = float(getattr(args.policy_node_cfg, "action_hz", 20) or 20)
    if not np.isfinite(action_hz) or action_hz <= 0:
        raise ValueError("action_hz must be finite and positive")
    ticks_per_action = max(1, round(freq / action_hz))
    if not math.isclose(ticks_per_action * action_hz, freq):
        raise ValueError("The control frequency must be an integer multiple of action_hz")
    action_dim = args.policy_node_cfg.action_dim
    use_rtc = bool(getattr(args.policy_node_cfg, "use_rtc", False))
    n_execute = resolve_n_execute(
        getattr(args, "policy_path", "") or "",
        getattr(args.policy_node_cfg, "n_execute", None),
    )
    binarize_gripper = bool(getattr(args.policy_cfg, "binarize_gripper", True))
    gripper_encoding: GripperEncoding = {
        "gripper_close": float(getattr(args.policy_cfg, "gripper_close", 0.0)),
        "gripper_open": float(getattr(args.policy_cfg, "gripper_open", 0.7)),
        "gripper_threshold": getattr(args.policy_cfg, "gripper_threshold", None),
    }
    joint_serve = is_joint_serve(args)
    fold_rotvec = not joint_serve
    safety = getattr(args, "safety_cfg", None) or RolloutSafetyCfg()
    tensor_api = torch
    if ActionQueue is None or RTCConfig is None or tensor_api is None:
        raise ImportError("LeRobot ActionQueue is required. Install with: uv sync --group pi")

    execution_horizon = int(
        getattr(args.policy_node_cfg, "rtc_execution_horizon", RTC_EXECUTION_HORIZON) or RTC_EXECUTION_HORIZON
    )
    rtc_cfg = RTCConfig(
        enabled=use_rtc,
        execution_horizon=execution_horizon,
        max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
    )
    queue_threshold = int(getattr(args.policy_node_cfg, "rtc_queue_threshold", RTC_QUEUE_THRESHOLD) or RTC_QUEUE_THRESHOLD)
    chunk_pause_s = max(0.0, float(getattr(args.policy_node_cfg, "chunk_pause_s", 0.0) or 0.0))
    hold_until = None
    time_per_action = 1.0 / action_hz
    diag_root = Path(getattr(args, "diag_dir", None) or DEFAULT_DIAG_DIR)
    recording = False
    started_once = False
    try:
        while not quit_ev.is_set():
            if pause.is_set():
                if started_once:
                    logger.info("Paused. Arm stopped; weights stay on GPU. Enter resumes, q quits.")
                while pause.is_set() and not quit_ev.is_set():
                    time.sleep(0.05)
                if quit_ev.is_set():
                    break
            time.sleep(0.5)
            if hasattr(policy, "reset"):
                policy.reset()
            drained = 0
            while drained < 8:
                pending = policy.get_action_chunk()
                if not pending.get("ready"):
                    break
                drained += 1
            if drained:
                logger.info("Dropped {} stale chunk(s) from before pause", drained)
            diag_dir = create_diagnostic_session(
                diag_root,
                {
                    "started_at": time.now(),
                    "policy_path": str(getattr(args, "policy_path", "")),
                    "instruction": args.instruction,
                    "control_hz": freq,
                    "action_hz": action_hz,
                    "n_execute": n_execute,
                    "use_rtc": use_rtc,
                    "rtc_queue_threshold": queue_threshold,
                    "rtc_execution_horizon": execution_horizon,
                    "binarize_gripper": binarize_gripper,
                    "gripper_state_source": getattr(args.policy_cfg, "gripper_state_source", "measured"),
                    "action_space": "joints" if joint_serve else "tcp",
                    "safety": asdict(safety),
                },
            )
            dumped_cameras = False
            chunk_idx = 0
            started_once = True
            action_queue = ActionQueue(rtc_cfg)
            interpolator = ActionInterpolator(
                multiplier=ticks_per_action, rotvec=not joint_serve, discrete_indices=(-1,) if binarize_gripper else ()
            )
            latency_tracker = LatencyTracker() if LatencyTracker is not None else None
            t_start = time.now()
            it = 0
            if joint_serve:
                action = measured_joint_action(env, action_dim, binarize_gripper=binarize_gripper, **gripper_encoding)
            else:
                action = measured_tcp_action(env, action_dim, binarize_gripper=binarize_gripper, **gripper_encoding)
            has_action = False
            last_sent = action.copy()
            processing_obs = False
            t_request = None
            index_before_inference = 0
            leftover_at_request = False
            startup_goal = None
            env.set_start_time(t_start)
            env.set_instruction(args.instruction)
            logger.info(
                "Serve loop: {} Hz interpolator x{}, action {} Hz, RTC {}, {}",
                freq,
                ticks_per_action,
                action_hz,
                "guided" if use_rtc else "off",
                "joints/moveJ" if joint_serve else "tcp/task_pos_ik",
            )
            logger.info("Execute {} steps of each predicted chunk, then replan", n_execute)
            if use_rtc:
                logger.info(
                    "RTC: request the next chunk when {} steps remain, guide the first {}",
                    queue_threshold,
                    execution_horizon,
                )
            if chunk_pause_s > 0:
                logger.info("Holding the last command {:.1f}s after each chunk before the next image", chunk_pause_s)
            if not use_rtc:
                logger.info("RTC off: those {} steps play out before the next request", n_execute)
            if joint_serve:
                logger.info("Command rate limit: {:.2f} rad/s per joint ({:.1f} Hz)", max_joint_speed, freq)
            else:
                logger.info(
                    "Command rate limit: {:.2f} m/s, {:.2f} rad/s ({:.1f} Hz) — first policy jump is ramped",
                    max_pos_speed,
                    max_rot_speed,
                    freq,
                )
            logger.info("Diag dumps under {}", diag_dir)

            try:
                while not pause.is_set() and not quit_ev.is_set():
                    it, t_cycle_end, t_cmd_target = resolve_control_tick(t_start, it, dt, time.now())
                    require_arm_state_safe(
                        env.robot.arm.get_state(), last_sent, now=time.now(), joint_serve=joint_serve, limits=safety
                    )
                    if processing_obs and t_request is not None and time.now() - t_request > safety.inference_timeout_s:
                        raise RuntimeError("Policy inference exceeded the rollout timeout")

                    policy_loaded = policy.policy_loaded

                    if policy_loaded:
                        remaining = action_queue.qsize()
                        want_obs = should_send_observation(
                            processing_obs=processing_obs,
                            remaining=remaining,
                            use_rtc=use_rtc,
                            queue_threshold=queue_threshold,
                        )
                        if want_obs and chunk_pause_s > 0 and chunk_idx > 0:
                            now = time.now()
                            if hold_until is None:
                                hold_until = now + chunk_pause_s
                                logger.info(
                                    "Chunk {} finished. Holding still {:.1f}s before the next plan",
                                    chunk_idx,
                                    chunk_pause_s,
                                )
                            if chunk_hold_pending(now, hold_until):
                                want_obs = False
                            else:
                                logger.info("Hold done after chunk {}. Taking a new image", chunk_idx)
                                hold_until = None
                        if want_obs:
                            leftover = action_queue.get_left_over() if use_rtc else None
                            leftover_at_request = leftover is not None and leftover.numel() > 0
                            obs = create_obs(
                                env,
                                gripper_state_source=getattr(args.policy_cfg, "gripper_state_source", "measured"),
                                max_camera_age_s=safety.max_state_age_s,
                            )
                            if not dumped_cameras:
                                _dump_live_cameras(obs, diag_dir / "cameras")
                                dumped_cameras = True
                            max_lat = latency_tracker.max() if latency_tracker is not None else 0.0
                            infer_delay = math.ceil(max_lat / time_per_action) if max_lat else 0
                            if leftover_at_request:
                                leftover_np = leftover.detach().cpu().numpy().astype(np.float32)
                                obs["rtc_leftover"] = leftover_np
                                obs["rtc_inference_delay"] = np.int32(infer_delay)
                            logger.warning(
                                "REQUESTING OBS remaining={} leftover={} delay={}",
                                remaining,
                                0 if leftover is None else int(leftover.shape[0]),
                                infer_delay if leftover_at_request else 0,
                            )
                            index_before_inference = action_queue.get_action_index()
                            t_request = time.now()
                            policy.send_observation(obs)
                            processing_obs = True

                        if processing_obs:
                            response = policy.get_action_chunk()
                            if response.get("inference_failed"):
                                raise RuntimeError("Policy inference failed; see the policy worker log")
                            if response["ready"]:
                                processed = validate_action_chunk(response["actions"], action_dim)
                                original = validate_action_chunk(response.get("original_actions", processed), action_dim)
                                if original.shape != processed.shape:
                                    raise RuntimeError("Normalized and physical action chunk shapes do not match")
                                predicted_n = min(len(processed), len(original))
                                processed, original = slice_action_chunk(processed, original, n_execute)
                                if predicted_n != len(processed):
                                    logger.info(
                                        "Playing {}/{} predicted steps",
                                        len(processed),
                                        predicted_n,
                                    )
                                now = time.now()
                                latency_s = 0.0 if t_request is None else float(now - t_request)
                                if latency_tracker is not None and latency_s > 0:
                                    latency_tracker.add(latency_s)
                                merge_delay = merge_delay_steps(
                                    latency_s,
                                    time_per_action,
                                    leftover_present=leftover_at_request,
                                    consumed_steps=action_queue.get_action_index() - index_before_inference,
                                )
                                if use_rtc and not leftover_at_request:
                                    logger.info(
                                        "First chunk: RTC delay skip disabled (queue was empty, latency={:.2f}s)",
                                        latency_s,
                                    )
                                action_queue.merge(
                                    tensor_api.as_tensor(original),
                                    tensor_api.as_tensor(processed),
                                    merge_delay,
                                    action_index_before_inference=index_before_inference,
                                )
                                processing_obs = False
                                t_request = None
                                leftover_at_request = False
                                first = action_queue.get_processed_left_over()
                                if first is not None and len(first):
                                    arm_state = env.robot.arm.get_state()
                                    if joint_serve:
                                        ref = np.asarray(arm_state["joint_q"], dtype=float).reshape(-1)[:7]
                                        vec = first[:, :7].cpu().numpy()
                                        unit = "rad"
                                    else:
                                        ref = np.asarray(arm_state["eef_pose"], dtype=float).reshape(-1)[:3]
                                        vec = first[:, :3].cpu().numpy()
                                        unit = "m"
                                    first_vec = vec[0]
                                    nref = min(len(first_vec), len(ref))
                                    raw_jump = float(np.linalg.norm(first_vec[:nref] - ref[:nref]))
                                    cmd_ref = action[:nref] if has_action else first_vec[:nref]
                                    cmd_jump = float(np.linalg.norm(first_vec[:nref] - cmd_ref))
                                    span = float(np.linalg.norm(vec[-1] - vec[0])) if len(vec) > 1 else 0.0
                                    step = float(np.max(np.linalg.norm(np.diff(vec, axis=0), axis=1))) if len(vec) > 1 else 0.0
                                    logger.info(
                                        "RTC merge delay={} leftover_q={} latency={:.2f}s "
                                        "first {:.3f} {} vs measured, {:.3f} {} vs last cmd, "
                                        "span {:.3f} {}, max step {:.3f} {}",
                                        merge_delay,
                                        action_queue.qsize(),
                                        latency_s,
                                        raw_jump,
                                        unit,
                                        cmd_jump,
                                        unit,
                                        span,
                                        unit,
                                        step,
                                        unit,
                                    )
                                    eef = np.asarray(arm_state.get("eef_pose", ref), dtype=float)
                                    write_chunk_dump(
                                        diag_dir,
                                        chunk_idx,
                                        first.cpu().numpy(),
                                        eef,
                                        delay=merge_delay,
                                        qsize=action_queue.qsize(),
                                        latency_s=latency_s,
                                    )
                                    chunk_idx += 1

                        # Same dispatch as lerobot.rollout.strategies.core.send_next_action.
                        # Prime _prev once so later adds blend. The first waypoint is
                        # crept to at startup speed instead of the interpolator's 50 ms
                        # blend — that blend from idle trips Kassow JTORQUE.
                        if interpolator.needs_new_action() and startup_goal is None:
                            next_act = action_queue.get()
                            if next_act is not None:
                                if not interpolator.has_previous:
                                    if joint_serve:
                                        last_sent = measured_joint_action(
                                            env, action_dim, binarize_gripper=binarize_gripper, **gripper_encoding
                                        )
                                    else:
                                        last_sent = measured_tcp_action(
                                            env, action_dim, binarize_gripper=binarize_gripper, **gripper_encoding
                                        )
                                    raw0 = _as_action_np(
                                        next_act,
                                        action_dim,
                                        binarize_gripper=binarize_gripper,
                                        fold_rotvec=fold_rotvec,
                                        **gripper_encoding,
                                    )
                                    err0 = command_tracking_error(last_sent, raw0, joint_serve=joint_serve)
                                    if first_waypoint_needs_creep(err0, joint_serve=joint_serve):
                                        startup_goal = raw0
                                        logger.info(
                                            "First waypoint {:.3f} {} from live pose; creeping before the chunk",
                                            err0,
                                            "rad" if joint_serve else "m",
                                        )
                                    else:
                                        interpolator.reset(last_sent)
                                        interpolator.add(next_act)
                                else:
                                    interpolator.add(next_act)

                        if startup_goal is not None:
                            if joint_serve:
                                action = limit_joint_action_step(
                                    startup_goal, last_sent, dt=dt, max_joint_speed=STARTUP_JOINT_SPEED
                                )
                            else:
                                action = limit_tcp_action_step(
                                    startup_goal,
                                    last_sent,
                                    dt=dt,
                                    max_pos_speed=STARTUP_POS_SPEED,
                                    max_rot_speed=STARTUP_ROT_SPEED,
                                )
                            last_sent = action
                            move_action_checked(
                                env, action, t_cmd_target + args.arm_latency, joint_serve=joint_serve, limits=safety
                            )
                            has_action = True
                            err = command_tracking_error(last_sent, startup_goal, joint_serve=joint_serve)
                            if not first_waypoint_needs_creep(err, joint_serve=joint_serve):
                                interpolator.reset(last_sent)
                                startup_goal = None
                                logger.info("Startup catch-up done ({:.3f} {})", err, "rad" if joint_serve else "m")
                        else:
                            interp = interpolator.get()
                            if interp is not None:
                                raw = _as_action_np(
                                    interp,
                                    action_dim,
                                    binarize_gripper=binarize_gripper,
                                    fold_rotvec=fold_rotvec,
                                    **gripper_encoding,
                                )
                                if joint_serve:
                                    action = limit_joint_action_step(
                                        raw, last_sent, dt=dt, max_joint_speed=max_joint_speed
                                    )
                                else:
                                    action = limit_tcp_action_step(
                                        raw,
                                        last_sent,
                                        dt=dt,
                                        max_pos_speed=max_pos_speed,
                                        max_rot_speed=max_rot_speed,
                                    )
                                last_sent = action
                                move_action_checked(
                                    env, action, t_cmd_target + args.arm_latency, joint_serve=joint_serve, limits=safety
                                )
                                has_action = True
                            elif has_action:
                                move_action_checked(
                                    env, action, t_cmd_target + args.arm_latency, joint_serve=joint_serve, limits=safety
                                )
                        if has_action:
                            arm_now = env.robot.arm.get_state()
                            measured = (
                                np.asarray(arm_now["joint_q"], dtype=float)
                                if joint_serve
                                else np.asarray(arm_now["eef_pose"], dtype=float)
                            )
                            append_cmd_eef(
                                diag_dir,
                                time.now() - t_start,
                                action,
                                measured,
                                action_queue.qsize(),
                                max(chunk_idx - 1, 0),
                                arm_state=arm_now,
                                joint_serve=joint_serve,
                            )

                        step = env.get_state(action=action)

                        if recording:
                            env.recorder.record_step(step)

                        if visualizer:
                            visualizer.log_env_state("env", step)

                    time.precise_wait(t_cycle_end)
                    it += 1
            except KeyboardInterrupt:
                quit_ev.set()
                logger.info("Quit. Unloading policy.")
                break
            finally:
                action_queue.clear()
                interpolator.reset()
                if not stop_arm_motion(env.robot.arm):
                    quit_ev.set()
                    logger.error("The arm cannot acknowledge a stream stop; ending the hardware session")
                if hasattr(policy, "reset"):
                    policy.reset()
    finally:
        quit_ev.set()
        stop_arm_motion(env.robot.arm)
        if env.recorder:
            env.recorder.save(wait=True)


def main(args):
    from rio.policies.pi05_resident import ensure_resident, shutdown_resident

    device = str(getattr(args.policy_cfg, "device", "cuda:0"))
    if args.unload_policy:
        shutdown_resident(device, policy_path=args.policy_path)
        return

    resident = ensure_resident(
        args.policy_path,
        device,
        use_rtc=bool(getattr(args.policy_cfg, "use_rtc", True)),
        compile=bool(getattr(args.policy_cfg, "compile", False)),
        binarize_gripper=bool(getattr(args.policy_cfg, "binarize_gripper", True)),
        gripper_close=float(getattr(args.policy_cfg, "gripper_close", 0.0)),
        gripper_open=float(getattr(args.policy_cfg, "gripper_open", 0.7)),
        gripper_threshold=getattr(args.policy_cfg, "gripper_threshold", None),
        rtc_max_guidance_weight=float(getattr(args.policy_cfg, "rtc_max_guidance_weight", RTC_MAX_GUIDANCE_WEIGHT)),
        action_space=getattr(args.policy_cfg, "action_space", None),
        rtc_execution_horizon=int(
            getattr(args.policy_node_cfg, "rtc_execution_horizon", RTC_EXECUTION_HORIZON) or RTC_EXECUTION_HORIZON
        ),
        timeout_s=float(getattr(args.policy_node_cfg, "timeout", 180.0) or 180.0),
    )
    logger.info(
        "π₀.₅ stays on {} after this process exits (pid {}). Unload with --unload-policy",
        device,
        resident.pid,
    )

    servers, clients, env = make_env(args)

    with ServerManager(args.mw, [*list(servers.values())]):
        with (
            env,
            clients["policy"]() as policy,
            clients["visualizer"]() if clients["visualizer"] else nullcontext() as visualizer,
        ):
            try:
                policy_loop(args, env, policy, visualizer)
            except KeyboardInterrupt:
                pass
    logger.info("Serve process exiting; π₀.₅ resident pid {} remains on {}", resident.pid, device)


if __name__ == "__main__":
    import examples.policy_cfgs as policy_cfgs
    from examples import get_policy_cfg

    PolicyCfg = get_policy_cfg(policy_cfgs)

    @dataclass
    class Args(PolicyCfg):
        diag_dir: str = str(DEFAULT_DIAG_DIR)
        unload_policy: bool = False
        safety_cfg: RolloutSafetyCfg = field(default_factory=RolloutSafetyCfg)

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
