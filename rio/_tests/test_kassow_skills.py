# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Pick, place, lift, move_to and pick_at of the Kassow cell on a fake environment."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rio.apps.kassow import goto
from rio.apps.kassow.graph.table_map import TableMap, Target
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
    cube = Target("cube", 0.1, 0.2, 0.04, 0.04, 0.9)
    assert dry.pick_at(cube, approach_m=0.1, z_offset_m=0.0) is True
    assert dry.place_at(cube, approach_m=0.1, z_offset_m=0.01) is True
    assert dry.move_above(cube, height_m=0.1) is True
    assert dry.gripper("close") is True


def _table() -> TableMap:
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return TableMap(
        camera="jev",
        serial="123",
        homography=np.eye(3),
        plane=np.array([0.0, 0.0, 0.2]),
        rotvec=np.array([np.pi, 0.0, 0.0]),
        hull=square,
        seeds_xy=np.zeros((1, 2)),
        seeds_q=np.full((1, 7), 0.5),
    )


def _pick_env(monkeypatch, measured: float = 0.3, status: goto.MoveStatus = goto.MoveStatus.LANDED, track: bool = True):
    """Fake cell for pick_at: ``move_tcp`` lands exactly on its goal when ``track`` is set."""
    env, sent = _env(measured)
    pose = {"eef_pose": np.zeros(6)}
    env.robot.arm = SimpleNamespace(get_state=lambda: dict(pose))
    moves: list[tuple] = []

    def move_tcp(_env, xyz, rotvec, *, home_joint_speed, seed_q=None):
        moves.append((np.round(xyz, 3).tolist(), None if seed_q is None else float(seed_q[0])))
        if track:
            pose["eef_pose"] = np.concatenate([xyz, rotvec])
        return status

    monkeypatch.setattr(goto, "move_tcp", move_tcp)
    monkeypatch.setattr(goto, "lift_z", lambda env, meters, *, home_joint_speed: goto.MoveStatus.LANDED)
    skills = KassowSkills(env, home_joint_speed=0.5, gripper_open=0.7, gripper_close=0.0, gripper_settle_s=0.0, table=_table())
    return skills, sent, moves


CUBE = Target("cube", 0.1, 0.2, 0.04, 0.04, 0.9)


def test_pick_at_opens_moves_above_descends_closes_and_holds(monkeypatch):
    skills, sent, moves = _pick_env(monkeypatch)
    assert skills.pick_at(CUBE, approach_m=0.1, z_offset_m=0.01) is True
    assert moves == [([0.1, 0.2, 0.31], 0.5), ([0.1, 0.2, 0.21], None)]
    assert sent == [0.7, 0.0]


@pytest.mark.parametrize(
    ("target", "approach_m", "z_offset_m"),
    [
        (Target("far", 2.0, 0.0, 0.04, 0.04, 0.9), 0.1, 0.0),
        (CUBE, 0.1, -0.01),
        (CUBE, 0.5, 0.0),
    ],
)
def test_pick_at_refuses_without_moving(monkeypatch, target, approach_m, z_offset_m):
    skills, sent, moves = _pick_env(monkeypatch)
    assert skills.pick_at(target, approach_m=approach_m, z_offset_m=z_offset_m) is False
    assert moves == [] and sent == []
    skills.table = None
    assert skills.pick_at(CUBE, approach_m=0.1, z_offset_m=0.0) is False


def test_pick_at_stops_on_an_alarm_or_a_tcp_off_target(monkeypatch):
    alarmed, _, moves = _pick_env(monkeypatch, measured=0.0, status=goto.MoveStatus.ALARM)
    assert alarmed.pick_at(CUBE, approach_m=0.1, z_offset_m=0.0) == "alarm"
    assert len(moves) == 1
    off_target, sent, moves = _pick_env(monkeypatch, track=False)
    assert off_target.pick_at(CUBE, approach_m=0.1, z_offset_m=0.0) is False
    assert len(moves) == 1
    assert sent == [0.7]


def test_place_at_descends_opens_and_lifts_only_while_holding(monkeypatch):
    skills, sent, moves = _pick_env(monkeypatch, measured=0.3)
    assert skills.place_at(CUBE, approach_m=0.1, z_offset_m=0.02) is True
    assert moves == [([0.1, 0.2, 0.32], 0.5), ([0.1, 0.2, 0.22], None)]
    assert sent == [0.7]
    empty, sent, moves = _pick_env(monkeypatch, measured=0.0)
    assert empty.place_at(CUBE, approach_m=0.1, z_offset_m=0.02) is False
    assert empty.place_at(CUBE, approach_m=0.1, z_offset_m=0.5) is False
    assert moves == [] and sent == []


def test_move_above_hovers_and_gripper_opens_or_closes(monkeypatch):
    skills, sent, moves = _pick_env(monkeypatch)
    assert skills.move_above(CUBE, height_m=0.15) is True
    assert moves == [([0.1, 0.2, 0.35], 0.5)]
    assert skills.move_above(CUBE, height_m=0.01) is False
    assert len(moves) == 1
    assert skills.gripper("close") is True
    assert skills.gripper("open") is True
    assert skills.gripper("half") is False
    assert sent == [0.0, 0.7]


class _Kin:
    """Pose is the first six joints; the URDF TCP sits a few millimetres off the controller's."""

    OFFSET = np.array([0.003, -0.002, 0.004])
    model: object = None

    def __init__(self, solves: bool = True):
        self.solves = solves

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        return np.concatenate([q[:3] + self.OFFSET, q[3:6]])

    def ik(self, seed, goal):
        q = np.asarray(seed, dtype=float).copy()
        if self.solves:
            q[:3] = goal[:3] - self.OFFSET
            q[3:6] = goal[3:]
        return q


def _tcp_env(kin: _Kin, q: np.ndarray):
    state = {"joint_q": q, "eef_pose": q[:6].copy()}
    return SimpleNamespace(robot=SimpleNamespace(arm=SimpleNamespace(_kin=kin, get_state=lambda: dict(state))))


def test_move_tcp_lands_the_measured_tcp_and_refuses_bad_ik(monkeypatch):
    sent: list[np.ndarray] = []

    def goto_home(_env, home, *, home_joint_speed, set_gripper):
        sent.append(np.asarray(home["joint_q"]))
        return goto.MoveStatus.LANDED

    monkeypatch.setattr(goto, "goto_home", goto_home)
    q = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.15, 0.2, 0.25])
    assert goto.move_tcp(_tcp_env(_Kin(), q), goal, np.zeros(3), home_joint_speed=0.5) is goto.MoveStatus.LANDED
    assert np.allclose(sent[0][:3], goal)

    missed = goto.MoveStatus.MISSED
    assert goto.move_tcp(_tcp_env(_Kin(solves=False), q), goal, np.zeros(3), home_joint_speed=0.5) is missed
    wrapped = q.copy()
    wrapped[6] = 4.0
    assert goto.move_tcp(_tcp_env(_Kin(), q), goal, np.zeros(3), home_joint_speed=0.5, seed_q=wrapped) is missed
    limited = _Kin()
    limited.model = SimpleNamespace(lowerPositionLimit=np.full(7, -0.05), upperPositionLimit=np.full(7, 0.05))
    assert goto.move_tcp(_tcp_env(limited, q), goal, np.zeros(3), home_joint_speed=0.5) is missed
    assert len(sent) == 1
