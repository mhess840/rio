import json
import math
import multiprocessing as mp
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env
from rio.policies.pi05 import principal_rotvec, snap_gripper

try:
    import torch
    from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig
    from lerobot.utils.action_interpolator import ActionInterpolator
except ImportError:
    torch = None
    ActionQueue = None
    LatencyTracker = None
    RTCConfig = None
    ActionInterpolator = None

# Horizon must exceed inference delay (~18). Keep in sync with rio/policies/pi05.py.
# Threshold 22 = delay plus slack so more of each 50-step chunk is executed.
RTC_QUEUE_THRESHOLD = 22
RTC_EXECUTION_HORIZON = 30
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


def merge_delay_steps(latency_s: float, time_per_action: float, *, leftover_present: bool) -> int:
    """RTC prefix skip: only drop steps the robot actually played during inference.

    LeRobot still applies ``ceil(latency/dt)`` on an empty queue. The first chunk
    then starts ~1 s into the plan (20 Hz × 1 s ≈ 20 steps) and the first
    waypoint is already several centimetres away.
    """
    if not leftover_present or latency_s <= 0 or time_per_action <= 0:
        return 0
    return int(math.ceil(latency_s / time_per_action))


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


def _as_action_np(action, action_dim: int, *, binarize_gripper: bool, fold_rotvec: bool = True) -> np.ndarray:
    """CPU float32 action; optionally fold rotvec and snap AG95 after interpolation."""
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    x = np.asarray(action, dtype=np.float32).reshape(-1)[:action_dim]
    if fold_rotvec and x.shape[0] >= 6:
        x[3:6] = principal_rotvec(x[3:6][None, :])[0]
    if x.shape[0] >= 7:
        if binarize_gripper:
            x[-1] = snap_gripper(np.asarray([x[-1]], dtype=np.float32))[0]
        else:
            x[-1] = np.clip(x[-1], 0.0, 1.0)
    return x


def measured_tcp_action(env, action_dim: int, *, binarize_gripper: bool) -> np.ndarray:
    """Live TCP as a 7D action. Used once to prime the interpolator."""
    eef = np.asarray(env.robot.arm.get_state()["eef_pose"], dtype=np.float32).reshape(-1)
    x = np.zeros((action_dim,), dtype=np.float32)
    n = min(6, int(eef.shape[0]), action_dim)
    x[:n] = eef[:n]
    if n >= 6:
        x[3:6] = principal_rotvec(x[3:6][None, :])[0]
    if action_dim > 6:
        grip = 0.7
        x[6] = float(snap_gripper(np.asarray([grip], dtype=np.float32))[0]) if binarize_gripper else 0.7
    return x


def measured_joint_action(env, action_dim: int, *, binarize_gripper: bool) -> np.ndarray:
    """Live joints + gripper as an 8D action. Used once to prime the interpolator."""
    state = env.robot.arm.get_state()
    q = np.asarray(state["joint_q"], dtype=np.float32).reshape(-1)
    x = np.zeros((action_dim,), dtype=np.float32)
    n = min(7, int(q.shape[0]), action_dim)
    x[:n] = q[:n]
    if action_dim > 7:
        grip = 0.7
        x[-1] = float(snap_gripper(np.asarray([grip], dtype=np.float32))[0]) if binarize_gripper else 0.7
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
        dR = r_tgt * r_prev.inv()
        ang = float(dR.magnitude())
        max_ang = float(max_rot_speed) * float(dt)
        if ang > max_ang > 0.0:
            dR = st.Rotation.from_rotvec(dR.as_rotvec() * (max_ang / ang))
        out[3:6] = principal_rotvec(np.asarray((dR * r_prev).as_rotvec(), dtype=np.float32)[None, :])[0]
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
def create_obs(env):
    formatted_obs = {}
    state = env.get_state()
    for key in state.observation.cameras:
        formatted_obs[key] = state.observation.cameras[key].rgb
    joints = state.observation.proprio_joints
    gripper = state.observation.gripper_position
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
) -> None:
    """Append one 100 Hz command vs measured TCP row."""
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "cmd_vs_eef.csv"
    cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    n_err = 7 if cmd.shape[0] >= 7 and eef.shape[0] >= 7 else min(3, cmd.shape[0], eef.shape[0])
    err = float(np.linalg.norm(cmd[:n_err] - eef[:n_err]))
    new_file = not csv_path.is_file()
    with csv_path.open("a") as f:
        if new_file:
            f.write("t,cmd_x,cmd_y,cmd_z,eef_x,eef_y,eef_z,err_m,remaining,chunk_idx\n")
        f.write(
            f"{t:.4f},{float(cmd[0]):.6f},{float(cmd[1]):.6f},{float(cmd[2]):.6f},"
            f"{float(eef[0]):.6f},{float(eef[1]):.6f},{float(eef[2]):.6f},"
            f"{err:.6f},{remaining},{chunk_idx}\n"
        )


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
    dt = 1.0 / freq
    arm_cfg = getattr(args, "arm_cfg", None)
    max_pos_speed = float(getattr(arm_cfg, "max_pos_speed", 0.45) or 0.45)
    max_rot_speed = float(getattr(arm_cfg, "max_rot_speed", 1.0) or 1.0)
    max_joint_speed = float(getattr(arm_cfg, "max_motor_speed", 1.0) or 1.0)
    action_hz = float(getattr(args.policy_node_cfg, "action_hz", 20) or 20)
    ticks_per_action = max(1, round(freq / action_hz))
    action_dim = args.policy_node_cfg.action_dim
    use_rtc = bool(getattr(args.policy_node_cfg, "use_rtc", False))
    n_execute = resolve_n_execute(
        getattr(args, "policy_path", "") or "",
        getattr(args.policy_node_cfg, "n_execute", None),
    )
    binarize_gripper = bool(getattr(args.policy_cfg, "binarize_gripper", True))
    joint_serve = is_joint_serve(args)
    fold_rotvec = not joint_serve
    if ActionQueue is None or ActionInterpolator is None or RTCConfig is None:
        raise ImportError("LeRobot ActionInterpolator / ActionQueue are required. Install with: uv sync --group pi")

    rtc_cfg = RTCConfig(
        enabled=use_rtc,
        execution_horizon=RTC_EXECUTION_HORIZON,
        max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
    )
    queue_threshold = int(getattr(args.policy_node_cfg, "rtc_queue_threshold", RTC_QUEUE_THRESHOLD) or RTC_QUEUE_THRESHOLD)
    chunk_pause_s = max(0.0, float(getattr(args.policy_node_cfg, "chunk_pause_s", 0.0) or 0.0))
    hold_until = None
    time_per_action = 1.0 / action_hz
    diag_dir = Path(getattr(args, "diag_dir", None) or DEFAULT_DIAG_DIR)
    dumped_cameras = False
    chunk_idx = 0
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
            drained = 0
            while drained < 8:
                pending = policy.get_action_chunk()
                if not pending.get("ready"):
                    break
                drained += 1
            if drained:
                logger.info("Dropped {} stale chunk(s) from before pause", drained)
            started_once = True
            action_queue = ActionQueue(rtc_cfg)
            interpolator = ActionInterpolator(multiplier=ticks_per_action)
            latency_tracker = LatencyTracker() if LatencyTracker is not None else None
            t_start = time.now()
            it = 0
            if joint_serve:
                action = measured_joint_action(env, action_dim, binarize_gripper=binarize_gripper)
            else:
                action = measured_tcp_action(env, action_dim, binarize_gripper=binarize_gripper)
            has_action = True
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
                    t_cycle_end = t_start + (it + 1) * dt
                    t_cmd_target = t_cycle_end + dt

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
                            obs = create_obs(env)
                            if not dumped_cameras:
                                _dump_live_cameras(obs, Path("/tmp/pi05_cam_check"))
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
                            if response["ready"]:
                                processed = np.asarray(response["actions"], dtype=np.float32)
                                original = np.asarray(response.get("original_actions", processed), dtype=np.float32)
                                if processed.ndim == 1:
                                    processed = processed[None, :]
                                if original.ndim == 1:
                                    original = original[None, :]
                                processed = processed[:, :action_dim]
                                original = original[:, : min(original.shape[1], processed.shape[1])]
                                if original.shape[1] < processed.shape[1]:
                                    original = processed
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
                                    latency_s, time_per_action, leftover_present=leftover_at_request
                                )
                                if use_rtc and not leftover_at_request:
                                    logger.info(
                                        "First chunk: RTC delay skip disabled (queue was empty, latency={:.2f}s)",
                                        latency_s,
                                    )
                                action_queue.merge(
                                    torch.as_tensor(original),
                                    torch.as_tensor(processed),
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
                                if interpolator._prev is None:
                                    if joint_serve:
                                        last_sent = measured_joint_action(
                                            env, action_dim, binarize_gripper=binarize_gripper
                                        )
                                    else:
                                        last_sent = measured_tcp_action(
                                            env, action_dim, binarize_gripper=binarize_gripper
                                        )
                                    raw0 = _as_action_np(
                                        next_act,
                                        action_dim,
                                        binarize_gripper=binarize_gripper,
                                        fold_rotvec=fold_rotvec,
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
                                        interpolator._prev = torch.as_tensor(last_sent)
                                        interpolator.add(next_act.cpu())
                                else:
                                    interpolator.add(next_act.cpu())

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
                            env.move(action, t_cmd_target=t_cmd_target + args.arm_latency)
                            has_action = True
                            err = command_tracking_error(last_sent, startup_goal, joint_serve=joint_serve)
                            if not first_waypoint_needs_creep(err, joint_serve=joint_serve):
                                interpolator._prev = torch.as_tensor(last_sent)
                                startup_goal = None
                                logger.info("Startup catch-up done ({:.3f} {})", err, "rad" if joint_serve else "m")
                        else:
                            interp = interpolator.get()
                            if interp is not None:
                                raw = _as_action_np(
                                    interp, action_dim, binarize_gripper=binarize_gripper, fold_rotvec=fold_rotvec
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
                                env.move(action, t_cmd_target=t_cmd_target + args.arm_latency)
                                has_action = True
                            elif has_action:
                                env.move(action, t_cmd_target=t_cmd_target + args.arm_latency)
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
            if pause.is_set() and hasattr(policy, "reset"):
                try:
                    policy.reset()
                except Exception:
                    logger.warning("Could not reset in-flight policy request after pause")
    finally:
        quit_ev.set()
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
        action_space=getattr(args.policy_cfg, "action_space", None),
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

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
