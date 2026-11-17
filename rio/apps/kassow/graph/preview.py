# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""XR dry run: walk a graph on the real cell, but play every arm move as a ghost in the headset.

The cameras, Jev, SAM and the table map work as in a live run, so the moves are planned
from what the cell sees now. The arm and the gripper never get a command:

- :class:`PreviewArm` stands in for the arm. It starts at the real joints, answers
  ``get_state`` with its own joints (FK, shifted onto the controller's TCP like
  ``goto.move_tcp``) and turns each ``moveJ_discrete`` into :meth:`MoveView.preview`.
  Everything else a motion path could call is missing on purpose, so an unexpected
  command fails instead of reaching the arm.
- :class:`PreviewGripper` remembers the commanded opening and reports a closed gripper
  as holding, so a graph follows its success edges.
- :func:`preview_run_vla` drives the ghost to a VLA's start pose and stops there.

Photos still come from the real cameras, so a wrist view is taken from where the real
arm stands, not from where the ghost looks.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from rio.apps.terminal import term

HOLD_MARGIN = 0.15
"""How far a closed preview gripper reports open: an object blocks the jaws."""


class PreviewGripper:
    """Remembers the commanded opening; never moves."""

    def __init__(self, opening: float, close: float) -> None:
        self.commanded = float(opening)
        self.close = float(close)

    def get_state(self) -> dict:
        """``gripper_position`` above the close opening after a close, as if an object were held."""
        closing = self.commanded <= self.close + 0.05
        measured = self.commanded + HOLD_MARGIN if closing else self.commanded
        return {"gripper_position": measured, "gripper_commanded_position": self.commanded}


class PreviewArm:
    """A ghost arm with the real arm's kinematics; ``moveJ_discrete`` plays in the headset.

    Args:
        arm: The real arm; only read.
        view: Headset ghost.
        gripper: Opening drawn on the ghost.
        hold_s: Pause at the end of each move, to look at the ghost in the room.
    """

    def __init__(self, arm: Any, view: Any, gripper: PreviewGripper, *, hold_s: float = 0.0) -> None:
        self._arm = arm
        self.hold_s = hold_s
        self._kin = arm._kin
        self.view = view
        self.gripper = gripper
        state = arm.get_state()
        self.real_q = np.asarray(state["joint_q"], dtype=float).copy()
        self.real_eef = np.asarray(state["eef_pose"], dtype=float).copy()
        self.q = self.real_q.copy()
        self.offset = np.asarray(self._kin.fk(self.q), dtype=float)[:3] - self.real_eef[:3]
        self.moves = 0

    def get_state(self) -> dict:
        """The real state with the ghost's joints, TCP, zero joint speed and no alarm."""
        state = dict(self._arm.get_state())
        pose = np.asarray(self._kin.fk(self.q), dtype=float).copy()
        pose[:3] -= self.offset
        state.update(joint_q=self.q.copy(), eef_pose=pose, joint_qd=np.zeros_like(self.q), alarm_code=0)
        return state

    def moveJ_discrete(self, target_q: np.ndarray, duration_s: float) -> None:
        """Play the move with the ghost (blocks for its duration) and take its goal as the new joints."""
        target = np.asarray(target_q, dtype=float).copy()
        self.moves += 1
        turn = float(np.degrees(np.max(np.abs(target - self.q))))
        term(f"[xr preview] move {self.moves}: {duration_s:.1f} s, largest joint turn {turn:.0f} deg")
        self.view.preview(self.q, target, duration_s, q_now=self.real_q, eef_now=self.real_eef, opening=self.gripper.commanded)
        self.q = target
        if self.hold_s > 0:
            time.sleep(self.hold_s)


class PreviewRobot:
    """Robot with :class:`PreviewArm` and :class:`PreviewGripper`; keeps the URDF path for the ghost."""

    def __init__(self, robot: Any, arm: PreviewArm, gripper: PreviewGripper) -> None:
        self.arm = arm
        self.gripper = gripper
        self.urdf_path = getattr(robot, "urdf_path", None)

    def move_gripper(self, opening: float, _t: float | None = None) -> None:
        """Record the opening; the ghost draws it on the next move."""
        self.gripper.commanded = float(opening)
        term(f"[xr preview] gripper -> {opening:.2f}")


class PreviewEnv:
    """The started environment with the arm and gripper replaced; cameras stay real.

    Args:
        env: Started environment.
        view: Headset ghost (``rio.xr.move_view.MoveView``).
        gripper_close: Opening of an empty closed gripper.
        hold_s: Pause after each ghost move.
    """

    def __init__(self, env: Any, view: Any, *, gripper_close: float, hold_s: float = 0.0) -> None:
        state = {}
        gripper = getattr(env.robot, "gripper", None)
        if gripper is not None and callable(getattr(gripper, "get_state", None)):
            state = gripper.get_state() or {}
        self.gripper = PreviewGripper(float(state.get("gripper_position", 1.0)), gripper_close)
        arm = PreviewArm(env.robot.arm, view, self.gripper, hold_s=hold_s)
        self.robot = PreviewRobot(env.robot, arm, self.gripper)
        self.cameras = env.cameras


def open_preview(env: Any, *, gripper_close: float, hold_s: float = 0.0) -> PreviewEnv:
    """:class:`PreviewEnv` for ``env``; needs the XR app and the arm meshes or stick model.

    Raises:
        SystemExit: No XR server answers, or the robot has no URDF.
    """
    from rio.xr.move_view import move_view

    view = move_view(getattr(env.robot, "urdf_path", None))
    if view is None:
        raise SystemExit("XR preview needs the XR app: uv run python -m rio.apps.kassow.xr --headset --no-reach")
    view.clear_preview()
    logger.info("XR preview: arm moves play as a ghost in the headset; the arm and gripper stay still")
    return PreviewEnv(env, view, gripper_close=gripper_close, hold_s=hold_s)


def preview_run_vla(skills: Any) -> Any:
    """``run_vla`` that drives the ghost to the VLA's start pose and reports the VLA as done."""
    from rio.apps.kassow.experiments import resolve_experiment

    def run(experiment: str, instruction: str, timeout_s: float) -> object:
        home = resolve_experiment(experiment).select_home(instruction)
        moved = skills.move_to(home)
        if moved is not True:
            return f"start pose not reached: {moved}"
        term(f"[xr preview] would run the {experiment} VLA {instruction!r} for up to {timeout_s:.0f} s")
        return True

    return run
