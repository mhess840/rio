# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Pick, place, lift and move_to of the Kassow cell on a fake environment."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rio.apps.kassow import goto
from rio.apps.kassow.skills import DrySkills, KassowSkills

pytestmark = pytest.mark.unit


def _env(measured: float):
    """Environment whose gripper reports ``measured`` and remembers the last command."""
    sent: list[float] = []
    state = {"gripper_position": measured, "gripper_commanded_position": 0.0}

    def move_gripper(opening, _target):
        sent.append(opening)
        state["gripper_commanded_position"] = opening

    gripper = SimpleNamespace(get_state=lambda: dict(state))
    return SimpleNamespace(robot=SimpleNamespace(gripper=gripper, move_gripper=move_gripper)), sent


def _skills(env) -> KassowSkills:
    return KassowSkills(env, home_joint_speed=0.5, gripper_open=0.7, gripper_close=0.0, gripper_settle_s=0.0)


def test_pick_closes_and_reports_an_object_in_the_jaws():
    env, sent = _env(measured=0.3)
    assert _skills(env).pick() is True
    assert sent == [0.0]


def test_pick_on_nothing_fails():
    env, sent = _env(measured=0.0)
    assert _skills(env).pick() is False
    assert sent == [0.0]


def test_place_opens_the_gripper():
    env, sent = _env(measured=0.3)
    assert _skills(env).place() is True
    assert sent == [0.7]


def test_pick_and_place_fail_without_a_gripper():
    env = SimpleNamespace(robot=SimpleNamespace())
    assert _skills(env).pick() is False
    assert _skills(env).place() is False


@pytest.mark.parametrize(
    ("status", "measured", "expected"),
    [
        (goto.MoveStatus.LANDED, 0.0, True),
        (goto.MoveStatus.MISSED, 0.0, False),
        (goto.MoveStatus.ALARM, 0.0, "alarm"),
        (goto.MoveStatus.ALARM, 0.3, "alarm_holding"),
    ],
)
def test_lift_reports_the_move_outcome(monkeypatch, status, measured, expected):
    monkeypatch.setattr(goto, "lift_z", lambda env, meters, *, home_joint_speed: status)
    env, _ = _env(measured)
    assert _skills(env).lift() == expected


def test_move_to_reports_an_alarm_while_holding(monkeypatch):
    monkeypatch.setattr("rio.apps.kassow.skills.time.sleep", lambda _s: None)
    monkeypatch.setattr(goto, "load_pose", lambda *args, **kwargs: {})
    monkeypatch.setattr(goto, "goto_home", lambda env, home, *, home_joint_speed: goto.MoveStatus.ALARM)
    env, _ = _env(measured=0.3)
    env.robot.gripper.get_state = lambda: {"gripper_position": 0.3, "gripper_commanded_position": 0.0}
    assert _skills(env).move_to(Path("home_open_space1.json")) == "alarm_holding"


def test_dry_skills_always_succeed():
    dry = DrySkills()
    assert dry.move_to(Path("home.json")) is True
    assert dry.lift() is True
    assert dry.pick() is True
    assert dry.place() is True
