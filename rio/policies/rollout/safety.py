# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed checks around every command a policy rollout sends."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw import time

from rio.arm_resume import ArmAlarm
from rio.schema import ArmState

# Reviewed in code, not on the command line: station speed overrides above these are refused.
MAX_POLICY_POS_SPEED = 0.5  # m/s
MAX_POLICY_ROT_SPEED = 1.0  # rad/s
MAX_POLICY_JOINT_SPEED = 1.0  # rad/s

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


def require_speed_limit(name: str, value: float, ceiling: float) -> float:
    """Accept a rate limit only inside ``(0, ceiling]``.

    Args:
        name: Config field, for the error message.
        value: Requested limit.
        ceiling: Reviewed maximum.

    Raises:
        ValueError: If the limit is not finite, not positive, or above ``ceiling``.
    """
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= ceiling:
        raise ValueError(f"{name}={value} must be in (0, {ceiling}]; the ceiling is a reviewed constant in rollout/safety.py")
    return value


def require_arm_state_safe(
    state: ArmState, command: np.ndarray, *, now: float, joint_serve: bool, limits: RolloutSafetyCfg
) -> None:
    """Reject alarms, stale feedback, invalid values, and excessive command tracking error."""
    if not isinstance(state, dict):
        raise RuntimeError("Arm feedback is unavailable")
    try:
        alarm = int(state.get("alarm_code", 0))
        timestamp = float(state["timestamp"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("Arm feedback has no valid alarm/timestamp information") from error
    if alarm:
        raise ArmAlarm(alarm)
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


def clamp_tcp_lead(command: np.ndarray, measured: np.ndarray, limits: RolloutSafetyCfg) -> np.ndarray:
    """Pull a TCP command back toward the measured pose, inside the tracking limit.

    A new chunk may start several centimetres from the arm. Sending that point
    trips the position check and stops the rollout. Keeping the lead at 80 % of
    each limit lets the arm catch up instead.
    """
    limited = np.array(command, dtype=np.float64, copy=True)
    pose = np.asarray(measured, dtype=np.float64).reshape(-1)
    delta = limited[:3] - pose[:3]
    distance = float(np.linalg.norm(delta))
    position_cap = 0.8 * float(limits.max_position_error_m)
    if distance > position_cap > 0.0:
        limited[:3] = pose[:3] + delta * (position_cap / distance)
    rotation = cast(st.Rotation, st.Rotation.from_rotvec(limited[3:6]) * st.Rotation.from_rotvec(pose[3:6]).inv())
    angle = float(rotation.magnitude())
    rotation_cap = 0.8 * float(limits.max_rotation_error_rad)
    if angle > rotation_cap > 0.0:
        scaled = st.Rotation.from_rotvec(rotation.as_rotvec() * (rotation_cap / angle))
        composed = cast(st.Rotation, scaled * st.Rotation.from_rotvec(pose[3:6]))
        limited[3:6] = composed.as_rotvec()
    return limited.astype(np.float32, copy=False)


def move_action_checked(env, command, target_time: float, *, joint_serve: bool, limits: RolloutSafetyCfg):
    """Check fresh feedback immediately before dispatching one arm/gripper command."""
    state = env.robot.arm.get_state()
    if not joint_serve:
        measured = np.asarray(state.get("eef_pose", []), dtype=np.float64).reshape(-1)
        if measured.shape == (6,) and np.isfinite(measured).all():
            command = clamp_tcp_lead(command, measured, limits)
    require_arm_state_safe(state, command, now=time.now(), joint_serve=joint_serve, limits=limits)
    if not np.isfinite(target_time):
        raise RuntimeError("Invalid command deadline")
    env.move(command, t_cmd_target=target_time)


def require_arm_controller(arm) -> str:
    """Accept joint-space and Cartesian arm controllers used by policy rollouts."""
    name = getattr(getattr(arm, "robot_controller", None), "name", None)
    if name not in _SUPPORTED_CONTROLLERS:
        raise AssertionError(f"Unsupported arm controller {name!r}; expected one of {sorted(_SUPPORTED_CONTROLLERS)}")
    return name


def arm_has_stopped(state: ArmState, max_joint_speed: float = 0.05) -> bool:
    """Require measured, finite joint velocities before declaring a plan complete."""
    velocity = np.asarray(state.get("joint_qd", []), dtype=float).reshape(-1)
    return bool(velocity.size and np.isfinite(velocity).all() and np.max(np.abs(velocity)) <= max_joint_speed)
