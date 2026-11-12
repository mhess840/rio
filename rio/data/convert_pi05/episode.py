# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-episode transforms: resample, unwrap, trim, segment, gripper encoding, and frame mapping."""

from __future__ import annotations

import numpy as np
from loguru import logger

from rio.data.dagger import CONTROL_KEY, dagger_segments
from rio.gripper import GripperBinarizer

from .config import ConvertPi05Cfg
from .schema import ACTION_DIM, ACTION_KEY, CAM1_KEY, CAM2_KEY, GRIPPER_KEY, JOINTS_KEY


def subsample_step(src_fps: int, dst_fps: int) -> int:
    """Integer stride from teleop Hz to dataset Hz."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be positive")
    if src_fps < dst_fps:
        raise ValueError(f"src_fps ({src_fps}) must be >= dst_fps ({dst_fps})")
    step = src_fps // dst_fps
    if src_fps % dst_fps:
        logger.warning("src_fps {} is not a multiple of dst_fps {}; using stride {}", src_fps, dst_fps, step)
    return max(step, 1)


def subsample_dict(data: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
    """Keep every ``step``-th frame, aligned across keys."""
    lengths = [len(v) for v in data.values() if hasattr(v, "__len__")]
    if not lengths:
        raise ValueError("Trajectory has no array features")
    n = min(lengths)
    if len(set(lengths)) > 1:
        logger.warning("Feature lengths differ {}; truncating to {}", lengths, n)
    idx = np.arange(0, n, step)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[0] < n:
            raise ValueError(f"{key} has {arr.shape[0]} frames, expected at least {n}")
        out[key] = arr[idx]
    return out


def unwrap_rotvec_sequence(rotvecs: np.ndarray) -> np.ndarray:
    """Pick the 2π-equivalent rotvec closest to the previous frame.

    SciPy / teleop store the principal value ``|r| ≤ π``, so a continuous
    yaw near ±π jumps by ~2π in R³. Behavior cloning treats that as a huge
    action; IK does not. The encoding stays axis-angle.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    if len(rotvecs) == 0:
        return rotvecs.astype(np.float32)
    out = np.empty_like(rotvecs)
    out[0] = rotvecs[0]
    two_pi = 2.0 * np.pi
    ks = np.arange(-3, 4, dtype=np.float64)
    for t in range(1, len(rotvecs)):
        prev = out[t - 1]
        current = rotvecs[t]
        norm = np.linalg.norm(current)
        if norm < 1e-8:
            out[t] = current
            continue
        axis = current / norm
        candidates = current[None, :] - (two_pi * ks)[:, None] * axis[None, :]
        out[t] = candidates[int(np.argmin(np.linalg.norm(candidates - prev, axis=1)))]
    return out.astype(np.float32)


def idle_start_index(xyz: np.ndarray, idle_xyz_m: float = 0.01) -> int:
    """First frame whose TCP xyz has moved ``idle_xyz_m`` from the start pose.

    Returns 0 when the episode never leaves the deadband so we do not wipe a
    static trajectory.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) == 0 or idle_xyz_m <= 0:
        return 0
    moved = np.flatnonzero(np.linalg.norm(xyz - xyz[0], axis=1) >= idle_xyz_m)
    if moved.size == 0:
        return 0
    return int(moved[0])


def slice_episode(data: dict[str, np.ndarray], start: int, end: int | None = None) -> dict[str, np.ndarray]:
    """Slice every time-major array. Length-1 extras (e.g. a scalar instruction) stay intact."""
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] <= start:
            out[key] = arr
            continue
        out[key] = arr[start:end]
    return out


def prepare_episode(data: dict[str, np.ndarray], cfg: ConvertPi05Cfg) -> tuple[dict[str, np.ndarray], dict]:
    """Subsample, unwrap TCP rotvec, drop the stationary prefix. Gripper is unchanged here."""
    step = subsample_step(cfg.src_fps, cfg.dst_fps)
    data = {key: np.asarray(value) for key, value in data.items()}
    data = subsample_dict(data, step)
    info = {
        "frames_sub": len(data[ACTION_KEY]),
        "idle_start": 0,
        "unwrap_jump_before": 0.0,
        "unwrap_jump_after": 0.0,
    }
    action = np.array(data[ACTION_KEY], dtype=np.float32, copy=True)
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"action must be (T, >=6), got {action.shape}")

    if cfg.unwrap_rotvec and cfg.action_space != "joints":
        rotvec = action[:, 3:6]
        if len(rotvec) > 1:
            info["unwrap_jump_before"] = float(np.max(np.linalg.norm(np.diff(rotvec, axis=0), axis=1)))
        unwrapped = unwrap_rotvec_sequence(rotvec)
        if len(unwrapped) > 1:
            info["unwrap_jump_after"] = float(np.max(np.linalg.norm(np.diff(unwrapped, axis=0), axis=1)))
        action[:, 3:6] = unwrapped
        data[ACTION_KEY] = action

    if cfg.trim_idle:
        xyz = np.asarray(data[ACTION_KEY], dtype=np.float32)[:, :3]
        start = idle_start_index(xyz, cfg.idle_xyz_m)
        remaining = len(xyz) - start
        if start > 0 and remaining < cfg.min_frames:
            logger.warning(
                "Idle trim would leave {} frames (min {}); keeping the prefix",
                remaining,
                cfg.min_frames,
            )
            start = 0
        if start > 0:
            data = slice_episode(data, start)
            info["idle_start"] = start

    info["frames_dst"] = len(np.asarray(data[ACTION_KEY]))
    return data, info


def episode_ranges(data: dict[str, np.ndarray], cfg: ConvertPi05Cfg, name: str) -> list[tuple[int, int]]:
    """Return training episode boundaries without crossing control-source switches."""
    frames = len(np.asarray(data[ACTION_KEY]))
    if cfg.segments == "whole":
        return [(0, frames)]
    if CONTROL_KEY not in data:
        raise KeyError(f"{name} has no {CONTROL_KEY}; --segments {cfg.segments} needs a DAgger rollout")
    ranges = dagger_segments(
        np.asarray(data[CONTROL_KEY])[:frames],
        mode=cfg.segments,
        fps=cfg.dst_fps,
        include_autonomous=cfg.include_autonomous,
        pre_takeover_cut_s=cfg.pre_takeover_cut_s,
        min_segment_s=cfg.min_segment_s,
        action=np.asarray(data[ACTION_KEY])[:frames],
        max_pause_s=cfg.max_pause_s if cfg.trim_operator_pauses else None,
    )
    return [(start, end) for start, end in ranges if end - start >= cfg.min_frames]


def joints_as_action(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build 8D joint actions: next measured ``q`` plus the commanded gripper.

    ``action[t, :7] = q[t+1]`` so the label is not the current state (identity).
    The last frame repeats ``q[-1]``. Gripper stays the teleop command at ``t``.
    """
    joints = np.asarray(data[JOINTS_KEY], dtype=np.float32).reshape(-1, 7)
    raw_action = np.asarray(data[ACTION_KEY], dtype=np.float32)
    if raw_action.ndim != 2 or raw_action.shape[0] != len(joints):
        raise ValueError(f"action {raw_action.shape} must be (T, *) aligned with joints {joints.shape}")
    grip = raw_action[:, -1]
    nxt = np.empty_like(joints)
    if len(joints) > 1:
        nxt[:-1] = joints[1:]
        nxt[-1] = joints[-1]
    else:
        nxt[0] = joints[0]
    return np.concatenate([nxt, grip.reshape(-1, 1)], axis=1).astype(np.float32)


def resolve_gripper_threshold(
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> float:
    """Return the open/close decision boundary.

    ``None`` uses the midpoint between ``gripper_close`` and ``gripper_open``.
    """
    if gripper_threshold is not None:
        return float(gripper_threshold)
    return 0.5 * (gripper_close + gripper_open)


def binarize_action_gripper(
    values: np.ndarray,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Snap commanded gripper values to the two teleop targets.

    Values at or above the threshold become ``gripper_open``. Do not low-pass
    the action — π₀.₅ would imitate the smear.
    """
    return GripperBinarizer(gripper_close, gripper_open, gripper_threshold).action(values).reshape(-1)


def binarize_state_gripper(
    values: np.ndarray,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Hysteresis-snap measured gripper position so the state does not chatter."""
    return GripperBinarizer(gripper_close, gripper_open, gripper_threshold).sequence(values)


def _as_float32_vec(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    if vec.shape[0] < dim:
        raise ValueError(f"{name} has dim {vec.shape[0]}, expected {dim}")
    return vec[:dim]


def build_pi05_frame(
    data: dict[str, np.ndarray],
    frame_idx: int,
    task: str,
    *,
    binarize_gripper: bool = True,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
    action_dim: int | None = None,
) -> dict:
    """Map one robodm frame to LeRobot π₀.₅ keys."""
    joints = _as_float32_vec(data[JOINTS_KEY][frame_idx], 7, JOINTS_KEY)
    gripper_obs = np.float32(data[GRIPPER_KEY][frame_idx]).reshape(-1)[0]
    raw_action = np.asarray(data[ACTION_KEY][frame_idx])
    dim = int(action_dim or np.asarray(data[ACTION_KEY]).shape[-1] or ACTION_DIM)
    action = _as_float32_vec(raw_action, dim, ACTION_KEY)
    if binarize_gripper:
        gripper_obs = binarize_state_gripper(
            np.asarray([gripper_obs], dtype=np.float32),
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )[0]
        action[-1] = binarize_action_gripper(
            np.asarray([action[-1]]),
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )[0]
    cam1 = np.asarray(data[CAM1_KEY][frame_idx])
    cam2 = np.asarray(data[CAM2_KEY][frame_idx])
    if cam1.ndim != 3 or cam1.shape[-1] != 3:
        raise ValueError(f"{CAM1_KEY} expected (H,W,3), got {cam1.shape}")
    if cam2.ndim != 3 or cam2.shape[-1] != 3:
        raise ValueError(f"{CAM2_KEY} expected (H,W,3), got {cam2.shape}")
    return {
        "observation.images.camera_1": cam1,
        "observation.images.camera_2": cam2,
        "observation.state": np.concatenate([joints, np.asarray([gripper_obs], dtype=np.float32)]),
        "action": action,
        "task": task,
    }
