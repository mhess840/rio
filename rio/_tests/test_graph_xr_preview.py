# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""XR dry run of a graph: moves go to a ghost, never to the arm. No robot, no headset."""

import numpy as np
import pytest

from rio.apps.kassow.graph.preview import PreviewEnv, preview_run_vla

pytestmark = pytest.mark.unit


class _Kin:
    def fk(self, q):
        q = np.asarray(q, dtype=float)
        return np.array([q[0], q[1], q[2], 0.0, 0.0, 0.0])


class _Arm:
    """A real arm that records any call that would move it."""

    def __init__(self) -> None:
        self._kin = _Kin()
        self.commands: list[str] = []

    def get_state(self):
        return {"joint_q": np.full(7, 0.1), "eef_pose": np.array([0.1, 0.1, 0.098, 0, 0, 0]), "alarm_code": 0}

    def moveJ_discrete(self, target_q, duration_s):
        self.commands.append("moveJ")


class _Gripper:
    def get_state(self):
        return {"gripper_position": 0.7, "gripper_commanded_position": 0.7}


class _Robot:
    urdf_path = None

    def __init__(self) -> None:
        self.arm = _Arm()
        self.gripper = _Gripper()

    def move_gripper(self, opening, t):
        self.arm.commands.append("gripper")


class _Env:
    def __init__(self) -> None:
        self.robot = _Robot()
        self.cameras = {"camera_1": object()}


class _View:
    def __init__(self) -> None:
        self.previews: list[tuple[np.ndarray, np.ndarray, float, float]] = []

    def preview(self, q_from, q_to, duration_s, *, q_now, eef_now, opening=None, speed=1.0):
        self.previews.append((np.array(q_from), np.array(q_to), duration_s, opening))

    def show(self, *args, **kwargs):
        pass

    def clear(self):
        pass


def test_a_preview_move_plays_on_the_ghost_and_never_reaches_the_arm():
    from rio.apps.kassow import goto

    env, view = _Env(), _View()
    preview = PreviewEnv(env, view, gripper_close=0.0)
    goal = np.full(7, 0.3)
    status = goto.goto_home(preview, {"joint_q": goal, "eef_pose": np.zeros(6), "gripper": 0.2}, home_joint_speed=1.0)
    assert status is goto.MoveStatus.LANDED
    assert env.robot.arm.commands == []
    (q_from, q_to, duration, _), *_ = view.previews
    assert np.allclose(q_from, 0.1) and np.allclose(q_to, goal) and duration >= 2.0
    state = preview.robot.arm.get_state()
    assert np.allclose(state["joint_q"], goal)
    # The ghost's TCP keeps the controller's offset to the URDF (2 mm in z here).
    assert np.allclose(state["eef_pose"][:3], [0.3, 0.3, 0.298])
    assert preview.robot.gripper.commanded == pytest.approx(0.2)


def test_the_preview_arm_only_offers_reads_and_moveJ():
    preview = PreviewEnv(_Env(), _View(), gripper_close=0.0)
    for name in ("step", "req", "servo", "move_to", "set_joint_velocity"):
        assert not hasattr(preview.robot.arm, name)
        assert not hasattr(preview, name)


def test_a_closed_preview_gripper_reports_a_held_object():
    from rio.arm_resume import gripper_is_holding

    preview = PreviewEnv(_Env(), _View(), gripper_close=0.0)
    preview.robot.move_gripper(0.0)
    assert gripper_is_holding(preview, close=0.0)
    preview.robot.move_gripper(0.7)
    assert not gripper_is_holding(preview, close=0.0)


def test_preview_vla_drives_to_the_start_pose_and_reports_done(monkeypatch):
    class _Skills:
        def __init__(self) -> None:
            self.moves: list[object] = []

        def move_to(self, pose):
            self.moves.append(pose)
            return True

    class _Experiment:
        def select_home(self, instruction):
            return f"home for {instruction}"

    monkeypatch.setattr("rio.apps.kassow.experiments.resolve_experiment", lambda name: _Experiment())
    skills = _Skills()
    assert preview_run_vla(skills)("open", "put the object in space 1", 60.0) is True
    assert skills.moves == ["home for put the object in space 1"]
