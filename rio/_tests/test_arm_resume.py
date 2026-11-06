# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Resume choice after an arm alarm. No robot and no middleware."""

from types import SimpleNamespace

import pytest

from rio.arm_resume import (
    Phase,
    PhaseKind,
    Resume,
    ResumePlan,
    decide_resume,
    gripper_holding,
    gripper_is_holding,
)

pytestmark = pytest.mark.unit

APPROACH = Phase("approach", PhaseKind.MOVE)
GRASP = Phase("grasp", PhaseKind.GRASP)
CARRY = Phase("carry", PhaseKind.MOVE)
PLACE = Phase("place", PhaseKind.MOVE)
PICK = (APPROACH, GRASP, CARRY, PLACE)


def test_empty_hand_repeats_the_approach_and_the_grasp():
    """The question is which phase runs again when the jaws are empty.

    The script is approach, grasp, carry, place. Nothing is held, so the
    interrupted phase is repeated and earlier phases are not replayed.
    """
    assert decide_resume(PICK, 0, holding=False) == ResumePlan(Resume.RETRY, 0)
    assert decide_resume(PICK, 1, holding=False) == ResumePlan(Resume.RETRY, 1)
    assert decide_resume(PICK, 0, holding=False).phase(PICK) is APPROACH
    assert decide_resume(PICK, 1, holding=False).phase(PICK) is GRASP


def test_holding_skips_the_grasp_and_takes_the_carry():
    """A held object must not approach or grasp again.

    Whether the alarm lands on the approach or on the grasp, the next phase
    is the carry, which the caller starts from the measured pose.
    """
    from_approach = decide_resume(PICK, 0, holding=True)
    from_grasp = decide_resume(PICK, 1, holding=True)
    assert from_approach == ResumePlan(Resume.CONTINUE, 2)
    assert from_grasp == ResumePlan(Resume.CONTINUE, 2)
    assert from_grasp.phase(PICK) is CARRY
    assert from_grasp.phase(PICK).kind is not PhaseKind.GRASP


def test_movej_repeats_the_same_goal():
    """One joint goal is repeated as that goal, not as a return to a previous pose."""
    movej = (Phase("movej", PhaseKind.MOVE),)
    plan = decide_resume(movej, 0, holding=False)
    assert plan == ResumePlan(Resume.RETRY, 0)
    assert plan.phase(movej).name == "movej"


def test_holding_with_nothing_after_the_grasp_has_no_phase():
    plan = decide_resume((APPROACH, GRASP), 1, holding=True)
    assert plan == ResumePlan(Resume.CONTINUE, 2)
    with pytest.raises(IndexError):
        plan.phase((APPROACH, GRASP))


def test_decide_resume_rejects_an_empty_or_unknown_index():
    with pytest.raises(ValueError):
        decide_resume((), 0, holding=False)
    with pytest.raises(ValueError):
        decide_resume(PICK, 4, holding=False)


def test_gripper_holding_needs_a_close_command_that_did_not_finish():
    """Measured opening is the only feedback. A blocked close is a hold.

    ``close`` is 0. An empty gripper that reached 0 is not holding, so the
    phase is repeated. A close command that stalls at 0.2 is holding. An
    open command is not a hold even if the jaws are mid-way.
    """
    assert gripper_holding(0.0, 0.0, close=0.0) is False
    assert gripper_holding(0.04, 0.0, close=0.0) is False
    assert gripper_holding(0.2, 0.0, close=0.0) is True
    assert gripper_holding(0.2, 0.7, close=0.0) is False


def test_explicit_holding_overrides_the_measured_opening():
    env = SimpleNamespace(
        robot=SimpleNamespace(
            gripper=SimpleNamespace(
                get_state=lambda: {"gripper_position": 0.0, "gripper_commanded_position": 0.0},
            )
        )
    )
    assert gripper_is_holding(env, close=0.0) is False
    assert gripper_is_holding(env, close=0.0, explicit=True) is True
    assert gripper_is_holding(SimpleNamespace(robot=SimpleNamespace()), close=0.0) is False


def test_gripper_holding_rejects_a_bad_slack():
    with pytest.raises(ValueError):
        gripper_holding(0.0, 0.0, close=0.0, slack=-0.1)
