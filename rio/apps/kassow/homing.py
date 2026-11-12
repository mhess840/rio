# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Start-pose snapshots of a Kassow session and the KORD ``moveJ`` that returns the arm to them."""

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw.interfaces.vive_retarget import matrix_to_pose, pose_to_matrix

from rio.apps.kassow.experiments import resolve_experiment, writable_home

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
    """Named experiment home, explicit ``home_file``, or ``home_q.json`` next to the recorder."""
    explicit = getattr(args, "home_file", None)
    if explicit:
        return Path(explicit)
    name = getattr(args, "experiment", None)
    if name:
        experiment = resolve_experiment(name)
        return experiment.select_home(
            str(getattr(args, "instruction", "") or ""),
            str(getattr(args, "pose", "") or ""),
        )
    recorder_cfg = getattr(args, "recorder_cfg", None)
    base = getattr(recorder_cfg, "path", None) if recorder_cfg is not None else None
    if not base:
        return None
    path = Path(base)
    if path.suffix == ".vla":
        path = path.parent
    return path / HOME_Q_FILENAME


def write_home_state(path: Path, home: dict, experiment: str | None = None, instruction: str | None = None) -> None:
    """Persist home joints so later eval can reuse the same start.

    Args:
        path: JSON file to write.
        home: Snapshot from ``capture_home_state``.
        experiment: Stored as ``name`` so the file cannot be loaded for another experiment.
        instruction: Task sentence this pose belongs to, when an experiment has several homes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_q": np.asarray(home["joint_q"], dtype=float).tolist(),
        "eef_pose": np.asarray(home["eef_pose"], dtype=float).tolist(),
        "gripper": float(home["gripper"]),
    }
    if experiment:
        payload = {"name": experiment, **payload}
    if instruction:
        payload["instruction"] = instruction
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_home_state(
    path: Path,
    gripper_fallback: float,
    experiment: str | None = None,
    instruction: str | None = None,
) -> dict:
    """Load a previously written home snapshot.

    Args:
        path: JSON written by ``write_home_state``.
        gripper_fallback: Gripper value when the file has none.
        experiment: Reject a file whose ``name`` belongs to another experiment.
        instruction: Reject a file captured for a different task sentence.

    Raises:
        ValueError: If the file was captured for a different experiment or sentence.
    """
    payload = json.loads(path.read_text())
    owner = payload.get("name")
    if experiment and owner and owner != experiment:
        raise ValueError(f"{path} is the home of '{owner}', not '{experiment}'")
    filed = payload.get("instruction")
    if instruction and filed and filed != instruction:
        raise ValueError(f"{path} is the home for {filed!r}, not {instruction!r}")
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
    rotvec = cast(st.Rotation, r_cur.inv() * r_goal).as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    max_rot = float(max_rot_speed) * float(dt)
    if ang > max_rot > 0.0:
        out[3:6] = cast(st.Rotation, r_cur * st.Rotation.from_rotvec(rotvec * (max_rot / ang))).as_rotvec()
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
    rotvec = cast(st.Rotation, st.Rotation.from_rotvec(cur[3:6]).inv() * st.Rotation.from_rotvec(tgt[3:6])).as_rotvec()
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


class KordHoming:
    """The session's start pose and, with ``go_home_on_save``, one KORD ``moveJ`` back to it.

    The arm returns home once after the operator starts the session and after each save.
    While it moves, the Vive stream is paused; it resumes only after the arm has settled
    and ``HOME_POST_MOVE_S`` has passed, because interrupting a moveJ trips ESTOP.

    Args:
        arm: Arm client with ``moveJ_discrete``.
        home: Snapshot with ``joint_q``, ``eef_pose``, ``gripper``.
        home_file: Where the snapshot lives, for the operator messages.
        instruction: Task sentence the pose belongs to, for the operator messages.
        joint_speed: Joint speed used to size the moveJ duration, rad/s.
        on_save: Return home at session start and after every save.
    """

    def __init__(
        self, arm, home: dict, *, home_file: Path | None, instruction: str | None, joint_speed: float, on_save: bool
    ) -> None:
        self.arm = arm
        self.home = home
        self.home_file = home_file
        self.instruction = instruction
        self.joint_speed = joint_speed
        self.on_save = on_save
        self.pending_start = on_save
        self.active = False
        self._streaming = True
        self._holdoff_until = 0.0
        self._move_until = 0.0
        self._deadline = 0.0
        self._settle = 0

    @classmethod
    def for_session(cls, args, arm, gripper_cmd: float) -> "KordHoming":
        """Load the session's home snapshot, or capture it from the live pose when asked or when there is none.

        Raises:
            RuntimeError: If an experiment home is missing and ``--capture-home`` was not given.
        """
        home_file = home_q_path(args)
        capture_home = bool(getattr(args, "capture_home", False))
        experiment = getattr(args, "experiment", None) or None
        instruction = str(getattr(args, "instruction", "") or "") or None
        if home_file is not None and home_file.is_file() and not capture_home:
            home = load_home_state(home_file, gripper_cmd, experiment=experiment, instruction=instruction)
            logger.info(
                "Loaded home from {}  instruction={}  tcp={}",
                home_file,
                instruction,
                np.round(home["eef_pose"][:3], 3).tolist(),
            )
        elif home_file is not None and experiment and not capture_home:
            sentence = f" for {instruction!r}" if instruction else ""
            flag = f" --instruction {instruction!r}" if instruction else ""
            raise RuntimeError(
                f"No home pose for experiment '{experiment}'{sentence} ({home_file}). Move the arm to "
                f"that start pose and run once with --capture-home{flag}."
            )
        else:
            home = capture_home_state(arm, gripper_cmd)
            if home_file is not None:
                home_file = writable_home(home_file)
                write_home_state(home_file, home, experiment=experiment, instruction=instruction)
                logger.info("Home snapshot for '{}' written to {}", experiment, home_file)
        logger.info(
            "Home q={}  tcp={}",
            np.round(home["joint_q"], 3).tolist(),
            np.round(home["eef_pose"][:3], 3).tolist(),
        )
        return cls(
            arm,
            home,
            home_file=home_file,
            instruction=instruction,
            joint_speed=float(getattr(args, "home_joint_speed", 0.15) or 0.15),
            on_save=bool(getattr(args, "go_home_on_save", False)),
        )

    @property
    def gripper(self) -> float:
        """Gripper opening stored with the home pose."""
        return float(self.home["gripper"])

    def describe(self) -> list[str]:
        """Operator instructions printed before the session starts."""
        if not self.on_save:
            return []
        sentence = f" ({self.instruction})" if self.instruction else ""
        return [
            f"  's' while clutched is fine: clutch drops, then KORD moveJ to "
            f"{np.round(self.home['joint_q'], 3).tolist()} "
            f"at {self.joint_speed:.2f} rad/s (accept {math.degrees(HOME_JOINT_EPS):.0f} deg).",
            f"  After Enter the arm homes to {self.home_file}{sentence}. Wait for Home reached, then n → c.",
            "  'n' works immediately after 's' (encode runs in the background). Wait for Home reached, then clutch.",
        ]

    def begin(self, joint_q: np.ndarray, t: float) -> None:
        """Queue the moveJ home; the stream stays paused until it has landed."""
        self.active, self._streaming, self._move_until, self._deadline = begin_kord_home(
            self.arm, joint_q, self.home, self.joint_speed, t
        )
        self._settle = 0

    def update(self, arm_state: dict, eef_pose: np.ndarray, joint_q: np.ndarray, t: float) -> str:
        """Advance one tick of an active return: ``moving``, ``done`` (landed or timed out), or ``aborted``."""
        q_goal = np.asarray(self.home["joint_q"], dtype=float)
        q_err = joint_home_error(joint_q, q_goal)
        pos_err, _ = pose_home_error(eef_pose, self.home["eef_pose"])
        alarm = int(arm_state.get("alarm_code", 0) or 0)
        landed = joints_at_home(joint_q, q_goal) and t >= self._move_until and joints_stopped(arm_state)
        self._settle = self._settle + 1 if landed else 0
        timed_out = t > self._deadline
        self._streaming = False
        if alarm:
            self.active = False
            self._holdoff_until = t + HOME_CANCEL_DURATION_S
            logger.warning("Homing aborted, KORD alarm at q_err={:.2f} deg", math.degrees(q_err))
            return "aborted"
        if self._settle >= HOME_SETTLE_TICKS or timed_out:
            self.active = False
            self._holdoff_until = t + HOME_POST_MOVE_S
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
            return "done"
        return "moving"

    def may_stream(self, t: float) -> bool:
        """Whether streamed arm commands are allowed this tick."""
        if not self.active and t >= self._holdoff_until:
            self._streaming = True
        return self._streaming

    def status(self, joint_q: np.ndarray, recording: bool) -> str:
        """One-word mode for the status line while returning home."""
        q_err = joint_home_error(joint_q, self.home["joint_q"])
        return f"HOMING{'+REC' if recording else ''} q={math.degrees(q_err):.1f}deg"
