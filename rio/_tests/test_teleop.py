# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Tier 1: teleop input -> motion mapping."""

import json

import numpy as np
import pytest

from rio.embodiments.single_arm import SingleArm
from rio.envs.poll import Interface, TeleopMode

pytestmark = pytest.mark.unit


class FakeKeyboard:
    def __init__(self, pressed=""):
        self.pressed = pressed

    def get_state(self):
        return {"alphanumeric_state": [ord(c) for c in self.pressed], "special_state": []}


class FakeStick:
    """Spacemouse or gamepad stand in"""

    def __init__(self, motion=None, buttons=()):
        self._motion = np.zeros(6) if motion is None else np.asarray(motion, float)
        self._buttons = set(buttons)

    def get_motion_state_transformed(self):
        return self._motion

    def is_button_pressed(self, idx):
        return idx in self._buttons


def test_keyboard_translation_keys():
    motion, gripper, _, mode = Interface.poll_keyboard(FakeKeyboard("w"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert motion[0] == 1.0 and not motion[1:].any()
    assert gripper is None

    motion, *_ = Interface.poll_keyboard(FakeKeyboard("s"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert motion[0] == -1.0


def test_keyboard_gripper_and_mode():
    _, gripper, _, _ = Interface.poll_keyboard(FakeKeyboard("["), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0
    _, gripper, _, _ = Interface.poll_keyboard(FakeKeyboard("]"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0
    _, _, _, mode = Interface.poll_keyboard(FakeKeyboard("2"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert mode is TeleopMode.ROTATION


def test_interface_poll_dispatch_matches_direct():
    kb = FakeKeyboard("d")
    via_poll = Interface.poll("Keyboard", kb, 0.0, 0.0, TeleopMode.TRANSLATION)
    direct = Interface.poll_keyboard(kb, 0.0, 0.0, TeleopMode.TRANSLATION)
    np.testing.assert_array_equal(via_poll[0], direct[0])


def test_spacemouse_buttons():
    _, gripper, _, _ = Interface.poll_spacemouse(FakeStick(buttons=[0]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0
    _, gripper, _, _ = Interface.poll_spacemouse(FakeStick(buttons=[1]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0

    _, _, t_change, mode = Interface.poll_spacemouse(FakeStick(buttons=[0, 1]), 5.0, 0.0, TeleopMode.TRANSLATION)
    assert mode is not TeleopMode.TRANSLATION
    assert t_change == 5.0


def test_gamepad_gripper_buttons():
    _, gripper, _, _ = Interface.poll_gamepad(FakeStick(buttons=[0]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0
    _, gripper, _, _ = Interface.poll_gamepad(FakeStick(buttons=[3]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0


def test_eef_cmd_translation_zeroes_rotation():
    target = np.zeros(6)
    delta = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.TRANSLATION, delta, target, 1.0, 1.0)
    np.testing.assert_allclose(out[:3], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(out[3:], 0.0, atol=1e-9)


def test_eef_cmd_2d_drops_z():
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.TRANSLATION_2D, np.ones(6), np.zeros(6), 1.0, 1.0)
    np.testing.assert_allclose(out[:3], [0.1, 0.1, 0.0])


def test_eef_cmd_rotation_zeroes_translation():
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.ROTATION, np.ones(6), np.zeros(6), 1.0, 1.0)
    np.testing.assert_allclose(out[:3], 0.0, atol=1e-9)
    assert np.linalg.norm(out[3:]) > 0


def test_held_keys_maps_mouse_side_buttons_to_scale():
    from examples.teleop_vive_hand import held_keys

    down = held_keys({"alphanumeric_state": [0], "special_state": [0x224, 0]})
    assert "scale_down" in down
    up = held_keys({"alphanumeric_state": [0], "special_state": [0x225, 0]})
    assert "scale_up" in up


def test_step_motion_scale_uses_relative_factor_and_clamps():
    from examples.teleop_vive_hand import step_motion_scale

    assert step_motion_scale(1.0, up=True, step=1.2, min_scale=0.5, max_scale=1.5) == pytest.approx(1.2)
    assert step_motion_scale(1.0, up=False, step=1.2, min_scale=0.5, max_scale=1.5) == pytest.approx(1.0 / 1.2)
    assert step_motion_scale(1.5, up=True, step=1.2, min_scale=0.5, max_scale=1.5) == 1.5
    assert step_motion_scale(0.5, up=False, step=1.2, min_scale=0.5, max_scale=1.5) == 0.5


def test_apply_live_scale_steps_pos_and_rot_together_without_jump():
    from rio_hw.interfaces.vive_retarget import AXIS_MAP_STEAMVR_TO_ROBOT, ClutchRetargeter

    from examples.teleop_vive_hand import apply_live_scale

    retargeter = ClutchRetargeter(pos_scale=1.0, rot_scale=1.0, max_pos_speed=10.0, max_rot_speed=10.0)
    tcp = np.array([0.4, 0.0, 0.3])
    origin = np.eye(4)
    origin[:3, 3] = [0.0, 1.0, 0.0]
    retargeter.engage(origin, tcp, np.eye(3))

    moved = origin.copy()
    moved[:3, 3] = [0.1, 1.0, 0.0]
    before, _ = retargeter.target(moved, dt=1.0)

    pos, rot = apply_live_scale(
        retargeter,
        up=False,
        step=1.2,
        min_scale=0.5,
        max_scale=1.5,
        tracker_pose=moved,
        pose_valid=True,
    )
    assert pos == pytest.approx(1.0 / 1.2)
    assert rot == pytest.approx(1.0 / 1.2)

    after, _ = retargeter.target(moved, dt=1.0)
    np.testing.assert_allclose(after, before, atol=1e-9)

    further = origin.copy()
    further[:3, 3] = [0.2, 1.0, 0.0]
    later, _ = retargeter.target(further, dt=1.0)
    extra = AXIS_MAP_STEAMVR_TO_ROBOT @ np.array([0.1, 0.0, 0.0])
    np.testing.assert_allclose(later, before + pos * extra, atol=1e-9)


def test_apply_live_scale_scales_speed_ceilings_with_motion_scale():
    from rio_hw.interfaces.vive_retarget import ClutchRetargeter

    from examples.teleop_vive_hand import apply_live_scale

    class FakeArm:
        max_pos_speed = 0.45
        max_rot_speed = 1.0
        max_motor_speed = 1.0

    retargeter = ClutchRetargeter(pos_scale=1.0, rot_scale=1.0, max_pos_speed=0.45, max_rot_speed=1.0)
    arm = FakeArm()
    apply_live_scale(
        retargeter,
        up=True,
        step=1.2,
        min_scale=0.5,
        max_scale=1.5,
        tracker_pose=None,
        pose_valid=False,
        base_max_pos_speed=0.45,
        base_max_rot_speed=1.0,
        arm=arm,
        base_max_motor_speed=1.0,
    )
    assert retargeter.pos_scale == pytest.approx(1.2)
    assert retargeter.max_pos_speed == pytest.approx(0.45 * 1.2)
    assert retargeter.max_rot_speed == pytest.approx(1.2)
    assert arm.max_pos_speed == pytest.approx(0.45 * 1.2)
    assert arm.max_rot_speed == pytest.approx(1.2)
    assert arm.max_motor_speed == pytest.approx(1.2)

    apply_live_scale(
        retargeter,
        up=False,
        step=1.2,
        min_scale=0.5,
        max_scale=1.5,
        tracker_pose=None,
        pose_valid=False,
        base_max_pos_speed=0.45,
        base_max_rot_speed=1.0,
        arm=arm,
        base_max_motor_speed=1.0,
    )
    assert retargeter.max_pos_speed == pytest.approx(0.45)
    assert arm.max_motor_speed == pytest.approx(1.0)


def test_step_pose_toward_caps_translation_and_rotation():
    from examples.teleop_vive_hand import pose_at_home, step_pose_toward

    current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = step_pose_toward(current, goal, dt=0.01, max_pos_speed=0.08, max_rot_speed=0.20)
    np.testing.assert_allclose(out[:3], [0.0008, 0.0, 0.0])
    assert abs(out[5]) == pytest.approx(0.002)
    assert not pose_at_home(out, goal)
    assert pose_at_home(goal, goal)


def test_pose_home_error_reports_metres_and_radians():
    from examples.teleop_vive_hand import pose_home_error

    current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.1])
    pos, rot = pose_home_error(current, goal)
    assert pos == pytest.approx(0.01)
    assert rot == pytest.approx(0.1)


def test_joints_stopped_uses_max_abs_qd():
    from examples.teleop_vive_hand import joints_stopped

    assert joints_stopped({"joint_qd": np.zeros(7)})
    assert not joints_stopped({"joint_qd": np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])})
    assert joints_stopped({})


def test_begin_kord_home_skips_when_already_there():
    from examples.teleop_vive_hand import begin_kord_home

    class FakeArm:
        def __init__(self):
            self.calls = []

        def moveJ_discrete(self, q, duration_s):
            self.calls.append((list(q), duration_s))

    q = np.zeros(7)
    home = {"joint_q": q.copy()}
    arm = FakeArm()
    homing, streaming, move_until, timeout = begin_kord_home(arm, q, home, 0.15, 10.0)
    assert homing is False
    assert streaming is True
    assert move_until == 0.0
    assert timeout == 0.0
    assert arm.calls == []


def test_begin_kord_home_queues_movej_when_offset():
    from examples.teleop_vive_hand import HOME_MIN_DURATION_S, begin_kord_home

    class FakeArm:
        def __init__(self):
            self.calls = []

        def moveJ_discrete(self, q, duration_s):
            self.calls.append((list(np.asarray(q, dtype=float)), float(duration_s)))

    current = np.zeros(7)
    goal = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arm = FakeArm()
    homing, streaming, move_until, timeout = begin_kord_home(
        arm, current, {"joint_q": goal}, 0.15, 10.0
    )
    assert homing is True
    assert streaming is False
    assert move_until == pytest.approx(10.0 + HOME_MIN_DURATION_S)
    assert timeout == pytest.approx(10.0 + HOME_MIN_DURATION_S + 3.0)
    assert len(arm.calls) == 1
    np.testing.assert_allclose(arm.calls[0][0], goal)


def test_home_move_duration_clips_to_min_and_max():
    from examples.teleop_vive_hand import HOME_MIN_DURATION_S, HOME_TIMEOUT_S, home_move_duration

    current = np.zeros(7)
    near = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    far = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert home_move_duration(current, near, 0.15) == pytest.approx(HOME_MIN_DURATION_S)
    assert home_move_duration(current, far, 0.15) == pytest.approx(HOME_TIMEOUT_S)
    mid = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert home_move_duration(current, mid, 0.15) == pytest.approx(4.0)


def test_step_q_toward_caps_largest_joint():
    from examples.teleop_vive_hand import joints_at_home, step_q_toward

    current = np.zeros(7)
    goal = np.array([0.4, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = step_q_toward(current, goal, dt=0.01, max_joint_speed=0.20)
    np.testing.assert_allclose(out[0], 0.002)
    np.testing.assert_allclose(out[1], -0.0005)
    assert not joints_at_home(out, goal)
    assert joints_at_home(goal, goal)


def test_joint_path_pose_cancels_fk_offset():
    from examples.teleop_vive_hand import joint_path_pose

    eef_start = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    fk_start = np.array([0.00, 0.0, 0.0, 0.0, 0.0, 0.0])
    fk_cmd = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = joint_path_pose(eef_start, fk_start, fk_cmd)
    np.testing.assert_allclose(out[:3], [0.15, 0.0, 0.0], atol=1e-9)


def test_write_home_state_roundtrip(tmp_path):
    from examples.teleop_vive_hand import capture_home_state, load_home_state, write_home_state

    class FakeArm:
        def get_state(self):
            return {
                "joint_q": np.arange(7, dtype=float),
                "eef_pose": np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0]),
            }

    home = capture_home_state(FakeArm(), 0.7)
    path = tmp_path / "home_q.json"
    write_home_state(path, home)
    payload = json.loads(path.read_text())
    np.testing.assert_allclose(payload["joint_q"], np.arange(7, dtype=float))
    assert payload["gripper"] == pytest.approx(0.7)
    loaded = load_home_state(path, gripper_fallback=0.0)
    np.testing.assert_allclose(loaded["eef_pose"][:3], [0.4, 0.2, 0.3])
    assert loaded["gripper"] == pytest.approx(0.7)


def test_n_starts_and_s_ends_recording():
    from examples.teleop_vive_hand import handle_recorder_save, handle_recorder_start

    class FakeRecorder:
        def __init__(self, closed=True, saving=False):
            self.calls: list[tuple] = []
            self.closed = closed
            self.saving = saving

        def get_state(self):
            return {"is_closed": self.closed, "is_saving": self.saving}

        def save(self, wait=False, start_next=False):
            self.calls.append(("save", wait, start_next))
            self.closed = True
            self.saving = True

        def new_trajectory(self, wait=False):
            self.calls.append(("new_trajectory", wait))
            self.closed = False
            self.saving = False

    class FakeEnv:
        def __init__(self, recorder):
            self.recorder = recorder

    recorder = FakeRecorder()
    env = FakeEnv(recorder)
    recording = handle_recorder_start(env, {"n"}, set(), False)
    assert recording is True
    assert recorder.calls == [("new_trajectory", False)]

    recorder.calls.clear()
    recording = handle_recorder_save(env, {"s"}, set(), recording)
    assert recording is False
    assert recorder.calls == [("save", False, False)]


def test_n_starts_while_previous_encode_runs():
    from examples.teleop_vive_hand import handle_recorder_start

    class FakeRecorder:
        def __init__(self):
            self.calls: list[tuple] = []

        def get_state(self):
            return {"is_closed": True, "is_saving": True}

        def new_trajectory(self, wait=False):
            self.calls.append(("new_trajectory", wait))

    class FakeEnv:
        def __init__(self, recorder):
            self.recorder = recorder

    env = FakeEnv(FakeRecorder())
    assert handle_recorder_start(env, {"n"}, set(), False) is True
    assert env.recorder.calls == [("new_trajectory", False)]


def test_s_warns_when_not_recording():
    from examples.teleop_vive_hand import handle_recorder_save

    class FakeRecorder:
        def __init__(self):
            self.calls: list[tuple] = []

        def save(self, wait=False, start_next=False):
            self.calls.append(("save", wait, start_next))

    class FakeEnv:
        def __init__(self, recorder):
            self.recorder = recorder

    env = FakeEnv(FakeRecorder())
    assert handle_recorder_save(env, {"s"}, set(), False) is False
    assert env.recorder.calls == []


def test_gello_cmd_clips_and_deadbands():
    target = np.zeros(3)
    out = SingleArm.make_gello_joint_cmd(100, np.array([1.0, 1.0, 1.0]), target.copy(), max_joint_delta=0.02)
    np.testing.assert_allclose(out, 0.02)

    out = SingleArm.make_gello_joint_cmd(100, np.array([0.001, 0.001, 0.001]), np.zeros(3), max_joint_delta=0.02, deadband=0.01)
    np.testing.assert_allclose(out, 0.0)
