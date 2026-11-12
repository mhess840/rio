"""Operator takeover during a π₀.₅ rollout (HG-DAgger) on the Kassow station.

The policy drives. ``c`` takes over: the Vive clutch engages at the last
commanded pose and the policy stops. ``c`` again releases the clutch (the arm
holds), and ``r`` with the clutch released hands control back; the arm holds
until the policy's first fresh chunk. Each episode is recorded with
``meta/control_source`` per frame into ``experiments/<exp>/dagger/r<N>/``::

    POLICY=Pi05DaggerCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve \\
        --policy-cfg.device cuda:1 --dagger-round 1 --instruction "put the object in space 2"

Keys: ``n`` start episode, ``c`` takeover/clutch, ``r`` hand back, ``o``
orientation, Space gripper, ``s`` save, ``x`` discard, ``q`` quit. After
``s``/``x`` the arm drives home. Discarded rollouts move to ``discarded/``.
"""

import json
import math
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw import time
from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    WorkspaceLimits,
    pose_to_matrix,
    yaw_rotation,
)

from rio.apps.kassow.homing import (
    HOME_POST_MOVE_S,
    HOME_SETTLE_TICKS,
    begin_kord_home,
    joints_at_home,
    joints_stopped,
    load_home_state,
)
from rio.apps.terminal import make_stdout_blocking, term
from rio.apps.vive.calibration import resolve_yaw
from rio.apps.vive.controls import merge_held_keys
from rio.cleanup import cleanup_step
from rio.data.dagger import SIDECAR_SUFFIX, ControlSource, control_summary

DISCARDED_DIR = "discarded"
# Files the recorder writes beside each ``.vla``; they follow a discarded trajectory.
RECORDER_SIDECARS = ("_meta.json", "_stats.json")
# A save is complete once the recorder has published "not saving" this long after the request.
SAVE_SETTLE_S = 0.1
SAVE_TIMEOUT_S = 120.0


def unique_path(path: Path) -> Path:
    """``path``, or ``<stem>_1<suffix>``, ``<stem>_2<suffix>``, … if it exists."""
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


def next_traj_index(folder: Path) -> int:
    """Index the recorder gives its next file in ``folder`` (max existing + 1)."""
    indices = [int(path.name[-8:-4]) for path in Path(folder).glob("*.vla") if path.name[-8:-4].isdigit()]
    return max(indices) + 1 if indices else 0


class DaggerSession:
    """Who drives, the operator's Vive command, and per-episode recording for one serve run."""

    def __init__(self, args, env, teleop, keyboards: list, *, yaw_deg: float):
        """Bind the station, Vive tracker, and key sources.

        Args:
            args: ``Pi05DaggerCfg``.
            env: Started environment with arm, gripper, cameras, and recorder.
            teleop: Vive tracker client.
            keyboards: Keyboard and clicker clients whose keys are merged.
            yaw_deg: Vive-to-robot yaw from ``resolve_yaw``.
        """
        if env.recorder is None:
            raise RuntimeError("DAgger needs the recorder; Pi05DaggerCfg enables it")
        self.args = args
        self.env = env
        self.teleop = teleop
        self.keyboards = keyboards
        self.round_dir = Path(args.recorder_cfg.path)
        self.home = load_home_state(
            Path(args.home_file),
            float(args.gripper_open),
            experiment=args.experiment,
            instruction=str(getattr(args, "instruction", "") or "") or None,
        )
        self.gripper_open = float(args.gripper_open)
        self.gripper_close = float(args.gripper_close)
        self.retargeter = ClutchRetargeter(
            pos_scale=args.pos_scale,
            rot_scale=args.rot_scale,
            max_pos_speed=args.max_pos_speed * args.pos_scale,
            max_rot_speed=args.max_rot_speed * args.rot_scale,
            axis_map=yaw_rotation(math.radians(yaw_deg)) @ AXIS_MAP_STEAMVR_TO_ROBOT,
            workspace=WorkspaceLimits(min_radius=args.min_radius, max_radius=args.max_radius, min_z=args.min_z),
            orientation_enabled=args.orientation_enabled,
        )
        self.mode = ControlSource.POLICY
        self.recording = False
        self.gripper_cmd = float(self.home["gripper"])
        self._previous_keys: set[str] = set()
        self._t_pose_valid = 0.0
        self._t_status = 0.0
        self._control: list[int] = []
        self._episode: dict = {}
        self._pending: list[tuple[Path, bool, dict, float]] = []
        self._next_index = next_traj_index(self.round_dir)

    @property
    def operator_driving(self) -> bool:
        """True after a takeover until the handback."""
        return self.mode is ControlSource.HUMAN

    @property
    def waiting_for_policy(self) -> bool:
        """True after a handback until the first action of the fresh chunk."""
        return self.mode is ControlSource.HOLD

    def limits(self, safety):
        """Rollout limits; while the operator drives, the clutch lag limit applies instead.

        Teleop lets the TCP lag the command by up to ``max_lag`` before it drops
        the clutch; the 5 cm policy limit would end a fast but normal demo.
        """
        if not self.operator_driving:
            return safety
        return replace(
            safety,
            max_position_error_m=max(safety.max_position_error_m, float(self.args.max_lag) + 0.05),
            max_rotation_error_rad=max(safety.max_rotation_error_rad, 1.0),
        )

    def _pressed(self) -> set[str]:
        keys = merge_held_keys(*(keyboard.get_state() for keyboard in self.keyboards))
        pressed = keys - self._previous_keys
        self._previous_keys = keys
        return pressed

    def _gripper_keys(self, pressed: set[str]) -> bool:
        before = self.gripper_cmd
        if "pagedown" in pressed:
            self.gripper_cmd = self.gripper_close
        elif "pageup" in pressed:
            self.gripper_cmd = self.gripper_open
        elif " " in pressed:
            mid = 0.5 * (self.gripper_open + self.gripper_close)
            self.gripper_cmd = self.gripper_close if self.gripper_cmd > mid else self.gripper_open
        if self.gripper_cmd != before:
            logger.info("Gripper target: {}", "open" if self.gripper_cmd == self.gripper_open else "closed")
            return True
        return False

    def _tracker(self) -> tuple[np.ndarray | None, bool]:
        state = self.teleop.get_state()
        valid = float(state["pose_valid"]) > 0.5
        if valid:
            self._t_pose_valid = time.now()
        return (pose_to_matrix(state["tracker_pose"]) if valid else None), valid

    def _engage(self, pose: np.ndarray) -> bool:
        tracker, valid = self._tracker()
        if not valid or tracker is None:
            logger.warning("Cannot engage: no valid tracker pose")
            return False
        rotation = st.Rotation.from_rotvec(np.asarray(pose[3:6], dtype=float)).as_matrix()
        self.retargeter.engage(tracker, np.asarray(pose[:3], dtype=float), rotation)
        return True

    def poll(self, command: np.ndarray) -> str | None:
        """Read keys once per control tick and switch who drives.

        Args:
            command: The last 7D TCP + gripper command sent to the arm.

        Returns:
            ``takeover``, ``handback``, ``save``, ``discard``, ``quit``, or ``None``.
        """
        pressed = self._pressed()
        if "q" in pressed:
            return "quit"
        if "x" in pressed:
            return "discard"
        if "s" in pressed:
            return "save"
        if self.operator_driving:
            self._gripper_keys(pressed)
            if self.args.orientation_key in pressed:
                self.retargeter.orientation_enabled = not self.retargeter.orientation_enabled
                logger.info("Orientation tracking: {}", self.retargeter.orientation_enabled)
        if self.args.clutch_key in pressed:
            if not self.operator_driving:
                self.mode = ControlSource.HUMAN
                self.gripper_cmd = float(command[6])
                if self._engage(command):
                    logger.info("TAKEOVER: you drive (clutch engaged at the last command)")
                else:
                    self.retargeter.hold_pose(np.asarray(command[:6], dtype=float))
                    logger.warning("TAKEOVER: arm holds; press '{}' to engage", self.args.clutch_key)
                return "takeover"
            if self.retargeter.engaged:
                self.retargeter.disengage()
                logger.info("Clutch released (arm holds). '{}' re-engages, 'r' hands back", self.args.clutch_key)
            else:
                eef = np.asarray(self.env.robot.arm.get_state()["eef_pose"], dtype=float)
                if self._engage(eef):
                    logger.info("Clutch engaged")
        if "r" in pressed:
            if not self.operator_driving:
                logger.info("The policy already drives")
            elif self.retargeter.engaged:
                logger.warning("Release the clutch ('{}') before handing back with 'r'", self.args.clutch_key)
            else:
                self.mode = ControlSource.HOLD
                logger.info("HANDBACK: arm holds until the policy's next chunk")
                return "handback"
        return None

    def human_command(self, eef_pose: np.ndarray, dt: float) -> np.ndarray:
        """7D TCP + gripper target from the Vive clutch; holds the measured TCP while released."""
        eef = np.asarray(eef_pose, dtype=float).reshape(-1)[:6]
        tracker, valid = self._tracker()
        retargeter = self.retargeter
        if retargeter.engaged and not valid and time.now() - self._t_pose_valid > self.args.tracking_grace:
            retargeter.disengage()
            logger.warning("Tracking lost, clutch released")
        if retargeter.engaged and valid and tracker is not None:
            target = retargeter.target_pose(tracker, dt)
            lag = float(np.linalg.norm(target[:3] - eef[:3]))
            if lag > self.args.max_lag:
                retargeter.disengage()
                retargeter.hold_pose(eef)
                target = eef.copy()
                logger.warning("Command ran {:.3f} m ahead of the arm, clutch released", lag)
        else:
            retargeter.hold_pose(eef)
            target = eef.copy()
        return np.concatenate([target[:6], [self.gripper_cmd]]).astype(np.float32)

    def resume_policy(self) -> None:
        """The first action of the fresh chunk was dispatched."""
        self.mode = ControlSource.POLICY
        logger.info("Policy drives again")

    def begin_episode(self) -> None:
        """Open a new ``.vla`` and start labelling frames as policy-driven."""
        self.env.recorder.new_trajectory(wait=True, timeout=10.0)
        index = self._next_index
        self._next_index += 1
        self._episode = {
            "path": self.round_dir / f"traj_{index:04d}.vla",
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self._control = []
        self.mode = ControlSource.POLICY
        self.retargeter.disengage()
        self.recording = True
        term("\n =============================================")
        logger.info("Episode traj_{:04d}: policy drives. c takeover, s save, x discard", index)

    def record(self, step) -> None:
        """Record one control tick with its ``meta/control_source``."""
        if not self.recording:
            return
        step.meta["control_source"] = np.int8(self.mode)
        self.env.recorder.record_step(step)
        self._control.append(int(self.mode))

    def end_episode(self, *, saved: bool, reason: str | None = None) -> None:
        """Close the ``.vla``; a discarded one is moved to ``discarded/`` once encoded."""
        if not self.recording:
            return
        self.recording = False
        self.retargeter.disengage()
        self.env.recorder.save(wait=False)
        summary = control_summary(np.asarray(self._control), fps=float(self.args.freq))
        meta = {
            "experiment": self.args.experiment,
            "round": int(self.args.dagger_round),
            "instruction": self.args.instruction,
            "model": str(self.args.policy_path),
            "started": self._episode.get("started"),
            "outcome": "saved" if saved else "discarded",
            "reason": reason,
            "autonomous": summary["takeovers"] == 0,
            **summary,
        }
        self._pending.append((self._episode["path"], saved, meta, time.now()))
        self.mode = ControlSource.POLICY
        logger.info(
            "{} {}: {:.1f} s, takeovers {}, operator {} / policy {} frames{}",
            "Saved" if saved else "Discarded",
            self._episode["path"].name,
            summary["duration_s"] or 0.0,
            summary["takeovers"],
            summary["frames"]["human"],
            summary["frames"]["policy"],
            f" ({reason})" if reason else "",
        )

    def finalize_pending(self, *, wait: bool = False) -> None:
        """Write sidecars and move discarded files once the recorder finished encoding."""
        deadline = time.now() + SAVE_TIMEOUT_S
        while self._pending:
            status = self.env.recorder.get_state()
            if isinstance(status, dict) and status.get("failed", False):
                raise RuntimeError(f"Recorder failed (error code {status.get('error_code', 0)}); DAgger sidecars not written")
            newest = max(item[3] for item in self._pending)
            done = (
                isinstance(status, dict)
                and not status.get("is_saving", False)
                and float(status.get("timestamp", 0.0)) > newest + SAVE_SETTLE_S
            )
            if done:
                break
            if not wait:
                return
            if time.now() > deadline:
                raise TimeoutError("Recorder did not finish encoding; DAgger sidecars not written")
            time.sleep(0.05)
        for path, saved, meta, _ in self._pending:
            if not path.is_file():
                raise FileNotFoundError(f"Expected {path} after save; DAgger sidecars not written")
            final = path
            if not saved:
                discarded = path.parent / DISCARDED_DIR
                discarded.mkdir(exist_ok=True)
                final = unique_path(discarded / path.name)
                shutil.move(str(path), str(final))
                for suffix in RECORDER_SIDECARS:
                    companion = path.with_name(path.stem + suffix)
                    if companion.is_file():
                        shutil.move(str(companion), str(final.with_name(final.stem + suffix)))
            sidecar = final.with_name(final.stem + SIDECAR_SUFFIX)
            sidecar.write_text(json.dumps(meta, indent=2) + "\n")
        self._pending = []

    def at_home(self) -> bool:
        """True when every joint is within the home tolerance of the experiment home."""
        q = np.asarray(self.env.robot.arm.get_state()["joint_q"], dtype=float)
        return joints_at_home(q, np.asarray(self.home["joint_q"], dtype=float))

    def drive_home(self) -> bool:
        """One KORD ``moveJ`` to the experiment home; the stream must already be stopped.

        Returns:
            Whether the arm ended within the home tolerance.
        """
        arm = self.env.robot.arm
        q_goal = np.asarray(self.home["joint_q"], dtype=float)
        state = arm.get_state()
        homing, _, move_until, deadline = begin_kord_home(
            arm, np.asarray(state["joint_q"], dtype=float), self.home, float(self.args.home_joint_speed), time.now()
        )
        if not homing:
            return self.at_home()
        settle = 0
        while settle < HOME_SETTLE_TICKS:
            now = time.now()
            state = arm.get_state()
            if int(state.get("alarm_code", 0) or 0):
                raise RuntimeError("KORD alarm while driving home")
            q = np.asarray(state["joint_q"], dtype=float)
            landed = joints_at_home(q, q_goal) and now >= move_until and joints_stopped(state)
            settle = settle + 1 if landed else 0
            if now > deadline:
                logger.warning("Home timed out at {:.2f} deg", math.degrees(float(np.max(np.abs(q - q_goal)))))
                break
            time.sleep(0.01)
        time.sleep(HOME_POST_MOVE_S)
        reached = self.at_home()
        if reached:
            logger.info("Home reached")
        return reached

    def _read_gripper(self) -> float:
        gripper = getattr(self.env.robot, "gripper", None)
        state = gripper.get_state() if gripper is not None else {}
        value = state.get("gripper_commanded_position", state.get("gripper_position"))
        if value is None:
            return self.gripper_cmd
        mid = 0.5 * (self.gripper_open + self.gripper_close)
        return self.gripper_open if float(value) > mid else self.gripper_close

    def wait_for_episode(self) -> bool:
        """Between episodes: finish saves, drive home, then wait for ``n`` (True) or ``q`` (False).

        An episode only starts at the home pose: ``n`` re-checks the joints and
        drives home again if the arm was moved (or the last move timed out).
        """
        self.finalize_pending()
        if self.drive_home():
            term(f"\nReady at home. n: start episode | Space: gripper | q: quit   (round folder {self.round_dir})")
        else:
            term("\nNot at home yet. n drives home again and starts only once the arm is there | q: quit")
        self.gripper_cmd = self._read_gripper()
        self._pressed()
        move_gripper = getattr(self.env.robot, "move_gripper", None)
        while True:
            self.finalize_pending()
            pressed = self._pressed()
            if "q" in pressed:
                return False
            if self._gripper_keys(pressed) and move_gripper is not None:
                move_gripper(self.gripper_cmd, time.now() + 0.1)
            if "n" in pressed:
                if not self.at_home():
                    logger.warning("Arm is not at home; driving home before the episode")
                    if not self.drive_home():
                        logger.warning("Home not reached; episode not started. Press n to try again")
                        self._pressed()
                        continue
                status = self.env.recorder.get_state()
                if not isinstance(status, dict) or status.get("is_closed", False):
                    return True
                logger.warning("Recorder still has an open file; press n again in a moment")
            time.sleep(0.02)

    def status(self, t: float) -> None:
        """One status line per second while an episode runs."""
        now = time.now()
        if now - self._t_status < 1.0:
            return
        self._t_status = now
        if self.operator_driving:
            mode = "YOU " + ("(clutch engaged)" if self.retargeter.engaged else "(clutch released: c engage, r hand back)")
        elif self.waiting_for_policy:
            mode = "HOLD (waiting for the policy)"
        else:
            mode = "POLICY (c takeover)"
        takeovers = sum(1 for prev, cur in pairwise(self._control) if cur == ControlSource.HUMAN != prev)
        term(f"t: {t:.1f}s | REC | {mode} | takeovers {takeovers}")

    def close(self) -> None:
        """Discard an open episode and wait for pending saves."""
        if self.recording:
            self.end_episode(saved=False, reason="session ended")
        self.finalize_pending(wait=True)


@contextmanager
def open_dagger(args, env, clients):
    """Vive tracker, keys, and a ``DaggerSession`` for ``policy_loop``.

    Terminal keyboards read stdin, so yaw calibration and the start prompt run
    before their listener starts.
    """
    make_stdout_blocking()
    teleop_client = clients.get("teleop")
    if teleop_client is None:
        raise RuntimeError("DAgger needs the Vive tracker (teleop)")
    keyboard_client = clients.get("teleop_keyboard")
    clicker_client = clients.get("teleop_clicker")
    with teleop_client() as teleop:
        yaw_deg = resolve_yaw(args, teleop)
        input(
            f"Instruction: {args.instruction}\nRecording into {args.recorder_cfg.path}\n"
            "Press Enter: the arm drives home (keep the E-stop within reach)"
        )
        with (
            keyboard_client() if keyboard_client else nullcontext() as keyboard,
            clicker_client() if clicker_client else nullcontext() as clicker,
        ):
            keyboards = [source for source in (keyboard, clicker) if source is not None]
            session = DaggerSession(args, env, teleop, keyboards, yaw_deg=yaw_deg)
            try:
                yield session
            finally:
                with cleanup_step("finish the pending DAgger saves"):
                    session.close()
