# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot skills of the Kassow cell: ``move_to``, ``lift``, ``pick``, ``place`` and ``pick_at``.

The orchestrator calls these instead of touching the arm or the gripper itself.
Every skill returns ``True`` on success, ``False`` on failure, or ``"alarm"`` /
``"alarm_holding"`` when the arm controller latched an alarm (``alarm_holding``:
an object is in the jaws), the same words the orchestrator already understands.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from loguru import logger
from rio_hw import time

from rio.apps.kassow.experiments import resolve_experiment
from rio.apps.kassow.homing import HOME_POST_MOVE_S
from rio.apps.terminal import term
from rio.arm_resume import gripper_is_holding

if TYPE_CHECKING:
    from rio.apps.kassow.graph.table_map import TableMap, Target

# Same names as the orchestrator's; imported from there it would be a cycle.
_OPEN = "open"
_ORGANIZE = "organize"

# Gripper feedback arrives at about 2 Hz, so wait for at least one fresh reading.
GRIPPER_SETTLE_S = 1.5
LIFT_M = 0.10
# The measured TCP must land this close to a pick_at waypoint before the next move.
TCP_TOLERANCE_M = 0.01

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
        table: Image-to-table map; ``pick_at`` takes its height, orientation, and area from it.
    """

    def __init__(
        self,
        env,
        *,
        home_joint_speed: float,
        gripper_open: float | None = None,
        gripper_close: float | None = None,
        gripper_settle_s: float = GRIPPER_SETTLE_S,
        table: TableMap | None = None,
    ):
        reference = resolve_experiment(_OPEN)
        self.env = env
        self.home_joint_speed = home_joint_speed
        self.gripper_open = float(reference.gripper_open if gripper_open is None else gripper_open)
        self.gripper_close = float(reference.gripper_close if gripper_close is None else gripper_close)
        self.gripper_settle_s = gripper_settle_s
        self.table = table

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

    def _tcp_near(self, xyz: np.ndarray) -> bool:
        measured = np.asarray(self.env.robot.arm.get_state()["eef_pose"], dtype=float)[:3]
        error = float(np.linalg.norm(measured - xyz))
        if error > TCP_TOLERANCE_M:
            logger.error("TCP landed {:.0f} mm from {}; check the table map and the TCP offset", error * 1000, xyz.round(3))
            return False
        return True

    def _table_point(
        self, target: Target, height_m: float, bounds: tuple[float, float], skill: str
    ) -> tuple[np.ndarray, TableMap] | None:
        """``target`` at the calibrated grasp height plus ``height_m``, or None when refused."""
        table = self.table
        if table is None:
            logger.error("{} needs a table map", skill)
            return None
        if not bounds[0] <= height_m <= bounds[1]:
            logger.error("{} height {} m is outside {}", skill, height_m, bounds)
            return None
        if not table.contains(target.x, target.y):
            logger.error("{}: {} at x={:.3f} y={:.3f} is outside the calibrated area", skill, target.name, target.x, target.y)
            return None
        return np.array([target.x, target.y, table.z_at(target.x, target.y) + height_m]), table

    def _reach(self, goal: np.ndarray, table: TableMap, seed_q: np.ndarray | None) -> Outcome:
        from rio.apps.kassow import goto

        status = goto.move_tcp(self.env, goal, table.rotvec, home_joint_speed=self.home_joint_speed, seed_q=seed_q)
        if status is goto.MoveStatus.ALARM:
            return self._alarm()
        return status is goto.MoveStatus.LANDED and self._tcp_near(goal)

    def _down_and_up(self, point: np.ndarray, table: TableMap, approach_m: float) -> Outcome:
        """``moveJ`` to ``approach_m`` above ``point``, then down onto it."""
        above = point + np.array([0.0, 0.0, approach_m])
        for goal, seed_q in ((above, table.seed_q(point[0], point[1])), (point, None)):
            reached = self._reach(goal, table, seed_q)
            if reached is not True:
                return reached
        return True

    def gripper(self, state: str) -> Outcome:
        """Open or close the jaws without checking for an object."""
        if state not in ("open", "close"):
            logger.error("gripper state must be open or close, got {!r}", state)
            return False
        return self._command_gripper(self.gripper_open if state == "open" else self.gripper_close)

    def move_above(self, target: Target, *, height_m: float) -> Outcome:
        """``moveJ`` to ``height_m`` above the calibrated grasp height at ``target``."""
        from rio.apps.kassow.graph.model import HOVER_M_RANGE

        found = self._table_point(target, height_m, HOVER_M_RANGE, "move_above")
        if found is None:
            return False
        point, table = found
        return self._reach(point, table, table.seed_q(target.x, target.y))

    def pick_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> Outcome:
        """Open, ``moveJ`` above ``target``, descend, close, and lift by ``approach_m``.

        Height, orientation, and the allowed area come from the table map; ``z_offset_m``
        can only raise the calibrated grasp height. True only if the object is still
        in the jaws after the lift.
        """
        from rio.apps.kassow.graph.model import APPROACH_M_RANGE, Z_OFFSET_M_RANGE

        if not APPROACH_M_RANGE[0] <= approach_m <= APPROACH_M_RANGE[1]:
            logger.error("approach {} m is outside {}", approach_m, APPROACH_M_RANGE)
            return False
        found = self._table_point(target, z_offset_m, Z_OFFSET_M_RANGE, "pick_at")
        if found is None:
            return False
        if not self.place():
            return False
        reached = self._down_and_up(*found, approach_m)
        if reached is not True:
            return reached
        holding = self.pick()
        lifted = self.lift(approach_m)
        if lifted is not True:
            return lifted
        return holding is True and gripper_is_holding(self.env, close=self.gripper_close)

    def place_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> Outcome:
        """``moveJ`` above ``target``, descend to the grasp height plus ``z_offset_m``, open, and lift.

        Refuses before moving when the gripper holds nothing.
        """
        from rio.apps.kassow.graph.model import APPROACH_M_RANGE, PLACE_Z_OFFSET_M_RANGE

        if not APPROACH_M_RANGE[0] <= approach_m <= APPROACH_M_RANGE[1]:
            logger.error("approach {} m is outside {}", approach_m, APPROACH_M_RANGE)
            return False
        found = self._table_point(target, z_offset_m, PLACE_Z_OFFSET_M_RANGE, "place_at")
        if found is None:
            return False
        if not gripper_is_holding(self.env, close=self.gripper_close):
            logger.error("place_at: the gripper holds nothing")
            return False
        reached = self._down_and_up(*found, approach_m)
        if reached is not True:
            return reached
        if not self.place():
            return False
        return self.lift(approach_m)


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

    def pick_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> Outcome:
        term(f"[dry run] would pick {target.name} at x={target.x:.3f} y={target.y:.3f} (approach {approach_m:.2f} m)")
        return True

    def place_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> Outcome:
        term(f"[dry run] would place at {target.name} x={target.x:.3f} y={target.y:.3f} (+{z_offset_m:.2f} m)")
        return True

    def move_above(self, target: Target, *, height_m: float) -> Outcome:
        term(f"[dry run] would move {height_m:.2f} m above {target.name} x={target.x:.3f} y={target.y:.3f}")
        return True

    def gripper(self, state: str) -> Outcome:
        term(f"[dry run] would {state} the gripper")
        return True
