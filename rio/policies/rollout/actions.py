# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""From queued policy waypoints to rate-limited control-rate arm commands."""

from typing import cast

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw.interpolators import ActionInterpolator

from rio.gripper import snap_gripper
from rio.rotations import principal_rotvec
from rio.schema import ArmState

from .safety import RolloutSafetyCfg, move_action_checked
from .settings import RolloutSettings

# First policy waypoint is often 5-15 cm / ~0.1 rad off the live pose. Closing
# that in the interpolator's 50 ms blend trips Kassow JTORQUE from idle.
STARTUP_JOINT_SPEED = 0.12  # rad/s
STARTUP_POS_SPEED = 0.08  # m/s
STARTUP_ROT_SPEED = 0.25  # rad/s
STARTUP_JOINT_EPS = 0.02  # rad L2
STARTUP_POS_EPS = 0.005  # m


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
    env,
    arm_state: ArmState,
    *,
    binarize_gripper: bool,
    gripper_close: float,
    gripper_open: float,
    gripper_threshold: float | None,
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


class CommandStreamer:
    """Turns queued policy waypoints into one checked, rate-limited arm command per control tick.

    The first waypoint of a plan is approached at the startup speeds instead of the
    interpolator's 50 ms blend; every later step is limited to the station speeds.

    Args:
        env: Started environment.
        settings: Resolved rollout settings.
    """

    def __init__(self, env, settings: RolloutSettings) -> None:
        self.env = env
        self.settings = settings
        self.interpolator = ActionInterpolator(
            multiplier=settings.ticks_per_action,
            rotvec=not settings.joint_serve,
            discrete_indices=(-1,) if settings.binarize_gripper else (),
        )
        self.action = self.measured()
        self.last_sent = self.action.copy()
        self.has_action = False
        self.startup_goal: np.ndarray | None = None

    def measured(self) -> np.ndarray:
        """Live pose (TCP or joints) plus the current gripper command, as an action."""
        s = self.settings
        measure = measured_joint_action if s.joint_serve else measured_tcp_action
        return measure(self.env, s.action_dim, binarize_gripper=s.binarize_gripper, **s.gripper_encoding)

    def _to_np(self, action) -> np.ndarray:
        s = self.settings
        return _as_action_np(
            action,
            s.action_dim,
            binarize_gripper=s.binarize_gripper,
            fold_rotvec=not s.joint_serve,
            **s.gripper_encoding,
        )

    @property
    def idle(self) -> bool:
        """No startup approach running and the interpolator wants a new waypoint."""
        return self.startup_goal is None and self.interpolator.needs_new_action()

    @property
    def following(self) -> bool:
        """A plan is being played (approach or interpolation)."""
        return self.startup_goal is not None or self.interpolator.has_previous

    def drop_plan(self) -> None:
        """Forget the waypoints being played, e.g. on an operator takeover or hand-back."""
        self.interpolator.reset()
        self.startup_goal = None

    def pull(self, queue) -> None:
        """Take the next queued waypoint once the previous one has been interpolated."""
        if not self.idle:
            return
        next_act = queue.get()
        if next_act is None:
            return
        if self.interpolator.has_previous:
            self.interpolator.add(next_act)
            return
        s = self.settings
        self.last_sent = self.measured()
        first = self._to_np(next_act)
        error = command_tracking_error(self.last_sent, first, joint_serve=s.joint_serve)
        if first_waypoint_needs_creep(error, joint_serve=s.joint_serve):
            self.startup_goal = first
            logger.info(
                "First waypoint {:.3f} {} from live pose; creeping before the chunk", error, "rad" if s.joint_serve else "m"
            )
        else:
            self.interpolator.reset(self.last_sent)
            self.interpolator.add(next_act)

    def _send(self, command: np.ndarray, target_time: float, limits: RolloutSafetyCfg) -> None:
        self.action = command
        self.last_sent = command
        move_action_checked(self.env, command, target_time, joint_serve=self.settings.joint_serve, limits=limits)
        self.has_action = True

    def step(self, target_time: float, limits: RolloutSafetyCfg) -> None:
        """Send this tick's command: startup approach, next interpolated waypoint, or the last one again."""
        s = self.settings
        if self.startup_goal is not None:
            if s.joint_serve:
                command = limit_joint_action_step(
                    self.startup_goal, self.last_sent, dt=s.dt, max_joint_speed=STARTUP_JOINT_SPEED
                )
            else:
                command = limit_tcp_action_step(
                    self.startup_goal, self.last_sent, dt=s.dt, max_pos_speed=STARTUP_POS_SPEED, max_rot_speed=STARTUP_ROT_SPEED
                )
            self._send(command, target_time, limits)
            error = command_tracking_error(self.last_sent, self.startup_goal, joint_serve=s.joint_serve)
            if not first_waypoint_needs_creep(error, joint_serve=s.joint_serve):
                self.interpolator.reset(self.last_sent)
                self.startup_goal = None
                logger.info("Startup catch-up done ({:.3f} {})", error, "rad" if s.joint_serve else "m")
            return
        waypoint = self.interpolator.get()
        if waypoint is not None:
            raw = self._to_np(waypoint)
            if s.joint_serve:
                command = limit_joint_action_step(raw, self.last_sent, dt=s.dt, max_joint_speed=s.max_joint_speed)
            else:
                command = limit_tcp_action_step(
                    raw, self.last_sent, dt=s.dt, max_pos_speed=s.max_pos_speed, max_rot_speed=s.max_rot_speed
                )
            self._send(command, target_time, limits)
        elif self.has_action:
            move_action_checked(self.env, self.action, target_time, joint_serve=s.joint_serve, limits=limits)

    def send_operator(self, command: np.ndarray, target_time: float, limits: RolloutSafetyCfg) -> None:
        """Forward an operator (DAgger) command through the same checks."""
        self._send(command, target_time, limits)
