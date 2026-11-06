# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Remember an interrupted arm phase and decide how to continue after an alarm.

A script is an ordered list of phases. After the controller alarm is cleared,
an empty gripper repeats the interrupted phase. A gripper that is already
holding skips every remaining grasp, and the approach that only existed to
reach that grasp, and continues at the next phase. The caller starts that
phase from the measured pose. Nothing here talks to a robot.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

# Jaws commanded shut that stay open by more than this (normalized 0-1) are
# treated as holding an object. An empty close lands on ``close`` itself.
HOLD_SLACK = 0.05


class ArmAlarm(RuntimeError):
    """The arm controller latched an alarm and motion must not continue."""

    def __init__(self, alarm: int):
        self.alarm = int(alarm)
        super().__init__(f"Arm controller alarm 0x{self.alarm:08x}")


class PhaseKind(enum.Enum):
    """What a phase does to the object."""

    MOVE = "move"
    GRASP = "grasp"
    RELEASE = "release"


class Resume(enum.Enum):
    """Whether the interrupted phase runs again or the script moves on."""

    RETRY = "retry"
    CONTINUE = "continue"


@dataclass(frozen=True)
class Phase:
    """One step a caller can repeat or skip.

    Args:
        name: Caller-defined label, such as ``movej`` or ``carry``.
        kind: Move, grasp, or release. A grasp is never repeated while holding.
    """

    name: str
    kind: PhaseKind


@dataclass(frozen=True)
class ResumePlan:
    """Phase to run once the alarm word is clear.

    Args:
        action: ``retry`` repeats ``index``. ``continue`` has already skipped
            any grasp that must not run again.
        index: Position in the script. Equal to ``len(phases)`` when holding
            leaves nothing after the last grasp.
    """

    action: Resume
    index: int

    def phase(self, phases: Sequence[Phase]) -> Phase:
        """Return the phase at ``index``.

        Raises:
            IndexError: If the plan has no phase left to run.
        """
        if self.index < 0 or self.index >= len(phases):
            raise IndexError("Resume plan has no phase left to run")
        return phases[self.index]


def gripper_holding(measured: float, commanded: float, *, close: float, slack: float = HOLD_SLACK) -> bool:
    """True when a close was commanded and the jaws stopped short of it.

    Args:
        measured: Reported opening, same units as ``close``.
        commanded: Opening last sent to the gripper.
        close: Opening of an empty, fully closed gripper.
        slack: Extra opening that still counts as closed. A measured opening
            above ``close + slack`` after a close command counts as holding.

    Returns:
        True when the object is blocking the jaws.

    Raises:
        ValueError: If an opening is not finite or ``slack`` is negative.
    """
    if not math.isfinite(measured) or not math.isfinite(commanded) or not math.isfinite(close) or not math.isfinite(slack):
        raise ValueError("Gripper openings and slack must be finite")
    if slack < 0.0:
        raise ValueError("slack must be non-negative")
    return commanded <= close + slack and measured > close + slack


def gripper_is_holding(env, *, close: float, slack: float = HOLD_SLACK, explicit: bool | None = None) -> bool:
    """Read the gripper, or use a caller's explicit hold flag.

    Args:
        env: Object with ``robot.gripper.get_state()`` returning
            ``gripper_position`` and ``gripper_commanded_position``.
        close: Empty-closed opening.
        slack: Passed to :func:`gripper_holding`.
        explicit: When set, this value is the answer and the gripper is not read.

    Returns:
        True when the object should be treated as held. Missing feedback is
        not holding, so the caller repeats the phase instead of skipping it.
    """
    if explicit is not None:
        return explicit
    gripper = getattr(getattr(env, "robot", None), "gripper", None)
    get_state = getattr(gripper, "get_state", None)
    if not callable(get_state):
        return False
    try:
        state = get_state() or {}
        measured = float(state["gripper_position"])
        commanded = float(state.get("gripper_commanded_position", measured))
    except (KeyError, TypeError, ValueError):
        return False
    return gripper_holding(measured, commanded, close=close, slack=slack)


def decide_resume(phases: Sequence[Phase], index: int, *, holding: bool) -> ResumePlan:
    """Choose the phase to run after the alarm has been cleared.

    An empty hand repeats ``phases[index]`` from the measured pose. A held
    object does not repeat a grasp. Approach moves that only lead to a still
    pending grasp are skipped too, and the plan continues at the next phase
    (carry or place). A single move, such as ``moveJ`` to one joint goal,
    stays on that goal: repeating it is not a return to an earlier pose.

    Args:
        phases: Script in execution order.
        index: Phase that was running when the alarm latched.
        holding: True when the gripper already has the object.

    Returns:
        The phase to execute. ``index`` may equal ``len(phases)`` when a held
        object has no phase left after the grasp.

    Raises:
        ValueError: If the script is empty or ``index`` is outside it.
    """
    if not phases:
        raise ValueError("A resume script needs at least one phase")
    if index < 0 or index >= len(phases):
        raise ValueError(f"Phase index {index} is outside a script of {len(phases)}")
    if not holding:
        return ResumePlan(Resume.RETRY, index)

    cursor = index
    while cursor < len(phases):
        if phases[cursor].kind is PhaseKind.GRASP:
            cursor += 1
            continue
        if any(phase.kind is PhaseKind.GRASP for phase in phases[cursor + 1 :]):
            cursor += 1
            continue
        return ResumePlan(Resume.CONTINUE, cursor)
    return ResumePlan(Resume.CONTINUE, len(phases))
