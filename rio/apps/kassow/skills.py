# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot skills of the Kassow cell: ``move_to``, ``lift``, ``pick`` and ``place``.

The orchestrator calls these instead of touching the arm or the gripper itself.
Every skill returns ``True`` on success, ``False`` on failure, or ``"alarm"`` /
``"alarm_holding"`` when the arm controller latched an alarm (``alarm_holding``:
an object is in the jaws), the same words the orchestrator already understands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from loguru import logger
from rio_hw import time

from rio.apps.kassow.experiments import resolve_experiment
from rio.apps.kassow.homing import HOME_POST_MOVE_S
from rio.apps.terminal import term
from rio.arm_resume import gripper_is_holding

# Same names as the orchestrator's; imported from there it would be a cycle.
_OPEN = "open"
_ORGANIZE = "organize"

# Gripper feedback arrives at about 2 Hz, so wait for at least one fresh reading.
GRIPPER_SETTLE_S = 1.5
LIFT_M = 0.10

Outcome = bool | str


class Skills(Protocol):
    """What the orchestrator may ask the arm to do."""

    def move_to(self, pose: Path) -> Outcome: ...
    def lift(self, meters: float = LIFT_M) -> Outcome: ...
    def pick(self) -> Outcome: ...
    def place(self) -> Outcome: ...


class KassowSkills:
    """Skills on a started environment with a Kassow arm and an AG95 gripper.

    Args:
        env: Started environment.
        home_joint_speed: Fastest joint speed of a ``moveJ`` in rad/s.
        gripper_open: Opening that releases an object.
        gripper_close: Opening of an empty, closed gripper.
        gripper_settle_s: Wait after a gripper command before the result is read.
    """

    def __init__(
        self,
        env,
        *,
        home_joint_speed: float,
        gripper_open: float | None = None,
        gripper_close: float | None = None,
        gripper_settle_s: float = GRIPPER_SETTLE_S,
    ):
        reference = resolve_experiment(_OPEN)
        self.env = env
        self.home_joint_speed = home_joint_speed
        self.gripper_open = float(reference.gripper_open if gripper_open is None else gripper_open)
        self.gripper_close = float(reference.gripper_close if gripper_close is None else gripper_close)
        self.gripper_settle_s = gripper_settle_s

    def _alarm(self) -> str:
        holding = gripper_is_holding(self.env, close=self.gripper_close)
        return "alarm_holding" if holding else "alarm"

    def move_to(self, pose: Path) -> Outcome:
        """``moveJ`` to a saved joint pose and wait until the arm has settled there."""
        from rio.apps.kassow import goto

        organize = resolve_experiment(_ORGANIZE)
        experiment = organize if pose == organize.home_path else resolve_experiment(_OPEN)
        time.sleep(HOME_POST_MOVE_S)
        home = goto.load_pose(pose, experiment.gripper_open, experiment.name)
        status = goto.goto_home(self.env, home, home_joint_speed=self.home_joint_speed)
        if status is goto.MoveStatus.ALARM:
            return self._alarm()
        return status is goto.MoveStatus.LANDED

    def lift(self, meters: float = LIFT_M) -> Outcome:
        """Raise the TCP by ``meters`` so the next ``moveJ`` clears the crate walls."""
        from rio.apps.kassow import goto

        status = goto.lift_z(self.env, meters, home_joint_speed=self.home_joint_speed)
        if status is goto.MoveStatus.ALARM:
            return self._alarm()
        return status is goto.MoveStatus.LANDED

    def _command_gripper(self, opening: float) -> bool:
        move_gripper = getattr(self.env.robot, "move_gripper", None)
        if not callable(move_gripper):
            logger.error("The station has no gripper")
            return False
        move_gripper(opening, time.now() + 0.2)
        time.sleep(self.gripper_settle_s)
        return True

    def pick(self) -> Outcome:
        """Close the gripper. True only if an object is blocking the jaws afterwards."""
        if not self._command_gripper(self.gripper_close):
            return False
        holding = gripper_is_holding(self.env, close=self.gripper_close)
        if not holding:
            logger.warning("Pick closed on nothing")
        return holding

    def place(self) -> Outcome:
        """Open the gripper and release whatever it holds."""
        return self._command_gripper(self.gripper_open)


class DrySkills:
    """Prints what each skill would do; the arm and the gripper stay idle."""

    def move_to(self, pose: Path) -> Outcome:
        term(f"[dry run] would moveJ to {pose.name}")
        return True

    def lift(self, meters: float = LIFT_M) -> Outcome:
        return True

    def pick(self) -> Outcome:
        term("[dry run] would pick")
        return True

    def place(self) -> Outcome:
        term("[dry run] would place")
        return True
