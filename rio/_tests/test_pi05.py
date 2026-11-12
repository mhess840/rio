# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot π₀.₅ observation mapping and Cartesian inference wiring."""

import threading
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from rio_hw.middlewares._middleware import Client

from examples.policy_inference import (
    RTC_QUEUE_THRESHOLD,
    RolloutSafetyCfg,
    _as_action_np,
    apply_serve_control,
    chunk_hold_pending,
    chunk_size_from_ckpt,
    command_tracking_error,
    create_obs,
    first_waypoint_needs_creep,
    limit_joint_action_step,
    limit_tcp_action_step,
    measured_joint_action,
    measured_tcp_action,
    merge_delay_steps,
    require_arm_controller,
    require_arm_state_safe,
    resolve_control_tick,
    resolve_n_execute,
    should_send_observation,
    slice_action_chunk,
    validate_action_chunk,
)
from rio.embodiments.single_arm import SingleArm
from rio.gripper import GripperBinarizer
from rio.policies.pi05 import ACTION_DIM, _to_action_chunk, format_rio_obs, infer_action_space, principal_rotvec, snap_gripper

pytestmark = pytest.mark.unit


class _Arm:
    def __init__(self, controller_name):
        self.num_joints = 7
        self.robot_controller = SimpleNamespace(name=controller_name)

    def get_state(self):
        return {"eef_pose": np.zeros(6), "joint_q": np.zeros(7), "timestamp": 0.0, "alarm_code": 0}

    def stop_motion(self):
        pass


def test_principal_rotvec_folds_2pi_equivalent():
    r = np.array([[np.pi + 0.2, 0.0, 0.0]], dtype=np.float32)
    out = principal_rotvec(r)
    assert np.linalg.norm(out) <= np.pi + 1e-5
    # Same rotation as going the short way around −π.
    assert out[0, 0] < 0


def test_original_plan_snapshot_cannot_be_modified_by_postprocessing():
    source = np.arange(14, dtype=np.float32).reshape(2, 7)
    original = _to_action_chunk(source)
    processed = _to_action_chunk(source)
    processed[:, -1] = 0.0
    assert original[0, -1] == 6.0
    assert source[0, -1] == 6.0


def test_initial_actions_preserve_a_held_gripper():
    state = {"eef_pose": np.zeros(6), "joint_q": np.zeros(7)}
    gripper = SimpleNamespace(get_state=lambda: {"gripper_position": 0.271, "gripper_commanded_position": 0.0})
    env = SimpleNamespace(robot=SimpleNamespace(arm=SimpleNamespace(get_state=lambda: state), _gripper=gripper))
    assert measured_tcp_action(env, 7, binarize_gripper=True)[-1] == 0.0
    assert measured_joint_action(env, 8, binarize_gripper=True)[-1] == 0.0


def test_action_encoding_respects_custom_gripper_targets():
    command = np.zeros(7, dtype=np.float32)
    command[-1] = 0.8
    result = _as_action_np(command, 7, binarize_gripper=True, gripper_close=0.1, gripper_open=0.9, gripper_threshold=0.6)
    assert result[-1] == pytest.approx(0.9)


def test_replay_drains_the_last_interpolated_waypoint(monkeypatch):
    from examples import replay_pi05_dataset as replay

    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 0] = [0.0, 0.001, 0.002]
    actions[:, -1] = 0.7
    commands = []
    env = SimpleNamespace(
        robot=SimpleNamespace(arm=_Arm("TASK_POS_IK")),
        set_start_time=lambda value: None,
        move=lambda action, **kwargs: commands.append(action.copy()),
    )
    args = SimpleNamespace(
        dataset_root="unused",
        repo_id="local/test",
        episode=0,
        start_frame=0,
        freq=100,
        arm_latency=0.0,
        arm_cfg=SimpleNamespace(max_pos_speed=0.45, max_rot_speed=1.0),
        policy_cfg=SimpleNamespace(binarize_gripper=True),
    )
    monkeypatch.setattr(replay, "_load_dataset", lambda *args: object())
    monkeypatch.setattr(replay, "_episode_bounds", lambda *args: (0, len(actions)))
    monkeypatch.setattr(replay, "_action_window", lambda *args: actions)
    monkeypatch.setattr(replay.time, "now", lambda: 0.0)
    monkeypatch.setattr(replay.time, "precise_wait", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    replay.replay_loop(args, env)
    assert len(commands) == 15
    np.testing.assert_allclose(commands[-1], actions[-1])


def test_snap_gripper_matches_dataset_targets():
    snapped = snap_gripper(np.array([0.0, 0.2, 0.35, 0.51, 0.7, 1.0], dtype=np.float32))
    np.testing.assert_allclose(snapped, [0.0, 0.0, 0.7, 0.7, 0.7, 0.7])


def test_format_rio_obs_keeps_analog_gripper_when_disabled():
    obs = {
        "camera_1": np.zeros((48, 64, 3), dtype=np.uint8),
        "camera_2": np.ones((48, 64, 3), dtype=np.uint8),
        "proprio_joints": np.arange(7, dtype=np.float32),
        "gripper_position": 0.51,
    }
    packed = format_rio_obs(obs, "pick up the cube", binarize_gripper=False)
    assert packed["state"][-1] == pytest.approx(0.51)


def test_format_rio_obs_snaps_analog_gripper_to_open():
    obs = {
        "camera_1": np.zeros((48, 64, 3), dtype=np.uint8),
        "camera_2": np.ones((48, 64, 3), dtype=np.uint8),
        "proprio_joints": np.arange(7, dtype=np.float32),
        "gripper_position": 0.51,
    }
    packed = format_rio_obs(obs, "pick up the cube")
    assert packed["state"][-1] == pytest.approx(0.7)


def test_live_gripper_hysteresis_matches_episode_encoding():
    values = np.array([0.7, 0.4, 0.3, 0.277, 0.1, 0.3, 0.4, 0.5, 0.7], dtype=np.float32)
    expected = GripperBinarizer().sequence(values)
    live = GripperBinarizer()
    obs = {"camera_1": np.zeros((2, 2, 3), dtype=np.uint8), "proprio_joints": np.zeros(7)}
    actual = []
    for value in values:
        obs["gripper_position"] = value
        actual.append(format_rio_obs(obs, "pick up the cylinder", gripper_binarizer=live)["state"][-1])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, [0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.7, 0.7])
    live.reset()
    assert live.update(0.4) == pytest.approx(0.7)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_gripper_encoding_rejects_nonfinite_feedback_and_commands(value):
    encoder = GripperBinarizer()
    with pytest.raises(ValueError, match="finite"):
        encoder.update(value)
    with pytest.raises(ValueError, match="finite"):
        encoder.action([value])


def test_gripper_hysteresis_preserves_the_exporters_float_boundary():
    encoder = GripperBinarizer()
    assert encoder.update(0.0) == 0.0
    immediately_below_open = np.nextafter(np.float32(0.455), np.float32(-np.inf))
    assert encoder.update(immediately_below_open) == 0.0
    assert encoder.update(np.float32(0.455)) == pytest.approx(0.7)


@pytest.mark.parametrize("state", [np.zeros(9), np.full(8, np.nan)])
def test_format_rio_obs_rejects_state_contract_mismatch(state):
    with pytest.raises(ValueError, match="finite state dims"):
        format_rio_obs({"proprio": state, "camera_1": np.zeros((2, 2, 3), dtype=np.uint8)}, "pick up the cylinder")


def test_format_rio_obs_concatenates_joints_and_gripper():
    obs = {
        "camera_1": np.zeros((48, 64, 3), dtype=np.uint8),
        "camera_2": np.ones((48, 64, 3), dtype=np.uint8),
        "proprio_joints": np.arange(7, dtype=np.float32),
        "gripper_position": 0.7,
    }
    packed = format_rio_obs(obs, "pick up the cube")
    np.testing.assert_allclose(packed["state"], np.concatenate([np.arange(7.0), [0.7]]))
    assert packed["state"].shape == (8,)
    assert packed["images"]["camera_1"].shape == (48, 64, 3)
    assert packed["task"] == "pick up the cube"


def test_format_rio_obs_rejects_empty_instruction():
    with pytest.raises(ValueError, match="instruction"):
        format_rio_obs({"camera_1": np.zeros((2, 2, 3)), "proprio_joints": np.zeros(7)}, "")


def test_format_rio_obs_normalizes_rgb_bytes_once():
    image = np.tile(np.array([0, 128, 255], dtype=np.uint8), (4, 6, 1))
    obs = {"camera_1": image, "proprio_joints": np.zeros(7), "gripper_position": 0.7}
    packed = format_rio_obs(obs, "pick up the cylinder")
    normalized = packed["images"]["camera_1"]
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    np.testing.assert_allclose(normalized[0, 0], [0.0, 128.0 / 255.0, 1.0])
    obs["camera_1"] = normalized
    np.testing.assert_array_equal(format_rio_obs(obs, "pick up the cylinder")["images"]["camera_1"], normalized)


@pytest.mark.parametrize("value", [128.0, -0.01, np.nan, np.inf])
def test_format_rio_obs_rejects_invalid_float_pixels(value):
    obs = {"camera_1": np.full((2, 2, 3), value), "proprio_joints": np.zeros(7)}
    with pytest.raises(ValueError, match="normalized"):
        format_rio_obs(obs, "pick up the cylinder")


def test_require_arm_controller_allows_kassow_ik():
    assert require_arm_controller(_Arm("TASK_POS_IK")) == "TASK_POS_IK"
    assert require_arm_controller(_Arm("JOINT_VEL")) == "JOINT_VEL"


def test_require_arm_controller_rejects_unknown():
    with pytest.raises(AssertionError, match="Unsupported"):
        require_arm_controller(_Arm("TORQUE"))


@pytest.mark.parametrize("alarm", [1, 0x10F1])
def test_rollout_rejects_arm_alarm(alarm):
    state = {"timestamp": 10.0, "alarm_code": alarm, "eef_pose": np.zeros(6)}
    with pytest.raises(RuntimeError, match="alarm"):
        require_arm_state_safe(state, np.zeros(7), now=10.0, joint_serve=False, limits=RolloutSafetyCfg())


@pytest.mark.parametrize("timestamp", [9.0, 11.0, np.nan])
def test_rollout_rejects_stale_or_invalid_arm_feedback(timestamp):
    state = {"timestamp": timestamp, "eef_pose": np.zeros(6)}
    with pytest.raises(RuntimeError, match="timestamp"):
        require_arm_state_safe(state, np.zeros(7), now=10.0, joint_serve=False, limits=RolloutSafetyCfg())


@pytest.mark.parametrize("state", [{}, {"timestamp": 10.0}, {"timestamp": 10.0, "eef_pose": np.full(6, np.nan)}])
def test_rollout_rejects_missing_or_nonfinite_arm_feedback(state):
    with pytest.raises(RuntimeError):
        require_arm_state_safe(state, np.zeros(7), now=10.0, joint_serve=False, limits=RolloutSafetyCfg())


def test_rollout_rejects_tracking_error_seen_during_safety_stop():
    state = {"timestamp": 10.0, "eef_pose": np.zeros(6)}
    command = np.zeros(7)
    command[0] = 0.149
    with pytest.raises(RuntimeError, match="position tracking"):
        require_arm_state_safe(state, command, now=10.0, joint_serve=False, limits=RolloutSafetyCfg())


def test_rollout_accepts_equivalent_rotation_vectors_across_pi():
    state = {"timestamp": 10.0, "eef_pose": np.array([0.0, 0.0, 0.0, np.pi - 0.01, 0.0, 0.0])}
    command = np.array([0.0, 0.0, 0.0, -np.pi + 0.01, 0.0, 0.0, 0.7])
    require_arm_state_safe(state, command, now=10.0, joint_serve=False, limits=RolloutSafetyCfg())


def test_rollout_checks_joint_tracking_without_folding_joint_angles():
    state = {"timestamp": 10.0, "joint_q": np.full(7, 4.0)}
    command = np.r_[np.full(7, 4.01), 0.7]
    require_arm_state_safe(state, command, now=10.0, joint_serve=True, limits=RolloutSafetyCfg())
    command[3] = 4.4
    with pytest.raises(RuntimeError, match="Joint tracking"):
        require_arm_state_safe(state, command, now=10.0, joint_serve=True, limits=RolloutSafetyCfg())


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_rollout_safety_limits_cannot_be_disabled_accidentally(value):
    with pytest.raises(ValueError, match="positive"):
        RolloutSafetyCfg(max_position_error_m=value)


def test_control_tick_skips_late_iterations_without_catch_up_burst():
    iteration, cycle_end, command_time = resolve_control_tick(0.0, 0, 0.01, 0.227)
    assert iteration == 23
    assert cycle_end >= 0.227 + 0.01
    assert command_time == pytest.approx(cycle_end + 0.01)
    assert resolve_control_tick(0.0, 0, 0.01, 0.0) == pytest.approx((0, 0.01, 0.02))


@pytest.mark.parametrize("chunk", [np.zeros((2, 6)), np.zeros((0, 7)), np.full((2, 7), np.nan)])
def test_invalid_policy_chunks_are_rejected_before_dispatch(chunk):
    with pytest.raises(RuntimeError, match="action chunk"):
        validate_action_chunk(chunk, 7)


def test_alarm_aborts_the_serve_loop_without_sending_motion(monkeypatch, tmp_path):
    from examples import policy_inference as serve

    stopped = []
    cleared = []
    moved = []
    arm = _Arm("TASK_POS_IK")
    monkeypatch.setattr(arm, "get_state", lambda: {"eef_pose": np.zeros(6), "timestamp": 0.0, "alarm_code": 123})
    monkeypatch.setattr(arm, "stop_motion", lambda: stopped.append(True))
    env = SimpleNamespace(
        robot=SimpleNamespace(arm=arm),
        recorder=None,
        set_start_time=lambda value: None,
        set_instruction=lambda value: None,
        move=lambda *args, **kwargs: moved.append(True),
    )
    policy = SimpleNamespace(
        policy_loaded=True,
        get_action_chunk=lambda: {"ready": False},
        reset=lambda: None,
    )
    args = SimpleNamespace(
        freq=100,
        policy_path="unused",
        policy_node_cfg=SimpleNamespace(action_hz=20, action_dim=7, n_execute=30),
        policy_cfg=SimpleNamespace(binarize_gripper=True),
        instruction="pick up the cylinder",
        diag_dir=str(tmp_path),
    )
    monkeypatch.setattr(serve, "ActionQueue", lambda *args: SimpleNamespace(clear=lambda: cleared.append(True)))
    monkeypatch.setattr(serve, "RTCConfig", SimpleNamespace)
    monkeypatch.setattr(serve, "LatencyTracker", None)
    monkeypatch.setattr(serve, "torch", SimpleNamespace(as_tensor=np.asarray))
    monkeypatch.setattr(
        serve,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            Thread=lambda **kwargs: SimpleNamespace(start=kwargs["args"][0].clear),
        ),
    )
    monkeypatch.setattr(serve.time, "sleep", lambda *args: None)
    monkeypatch.setattr(serve.time, "now", lambda: 0.0)
    with pytest.raises(RuntimeError, match="alarm"):
        serve.policy_loop(args, env, policy)
    assert not moved
    assert cleared
    assert stopped


def test_cpu_rollout_executes_startup_and_interpolation_with_custom_gripper(monkeypatch, tmp_path):
    from examples import policy_inference as serve

    clock = [0.0]
    pose = np.zeros(6, dtype=np.float32)
    grip = [0.9]
    commands = []
    stopped = []
    pause = threading.Event()
    quit_event = threading.Event()
    events = iter([pause, quit_event])

    class Tensor:
        def __init__(self, value):
            self.array = np.asarray(value, dtype=np.float32)

        def __len__(self):
            return len(self.array)

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class Queue:
        def __init__(self, cfg):
            self.buffer = np.empty((0, 7), dtype=np.float32)
            self.index = 0

        def clear(self):
            self.buffer = self.buffer[:0]
            self.index = 0

        def qsize(self):
            return len(self.buffer) - self.index

        def get_action_index(self):
            return self.index

        def merge(self, original, processed, delay, **kwargs):
            self.buffer = processed.array[delay:].copy()
            self.index = 0

        def get_processed_left_over(self):
            return Tensor(self.buffer[self.index :])

        def get(self):
            if not self.qsize():
                return None
            action = Tensor(self.buffer[self.index])
            self.index += 1
            return action

    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 0] = [0.01, 0.011, 0.012]
    actions[:, -1] = 0.1
    pending = [False]

    def response():
        ready = pending[0]
        pending[0] = False
        return {"ready": ready, "actions": actions, "original_actions": actions.copy()}

    policy = SimpleNamespace(
        policy_loaded=True,
        reset=lambda: pending.__setitem__(0, False),
        send_observation=lambda observation: pending.__setitem__(0, True),
        get_action_chunk=response,
    )
    arm = _Arm("TASK_POS_IK")
    monkeypatch.setattr(arm, "get_state", lambda: {"eef_pose": pose.copy(), "timestamp": clock[0], "alarm_code": 0})
    monkeypatch.setattr(arm, "stop_motion", lambda: stopped.append(True))
    gripper = SimpleNamespace(get_state=lambda: {"gripper_position": grip[0], "gripper_commanded_position": grip[0]})

    def state(**kwargs):
        return SimpleNamespace(
            observation=SimpleNamespace(
                cameras={
                    "camera_1": SimpleNamespace(
                        rgb=np.zeros((4, 4, 3), dtype=np.uint8), meta={"camera_receive_timestamp": clock[0]}
                    )
                },
                proprio_joints=np.zeros(7),
                gripper_position=grip[0],
            )
        )

    def move(action, **kwargs):
        commands.append(action.copy())
        pose[:] = action[:6]
        grip[0] = float(action[-1])
        if len(commands) >= 12:
            quit_event.set()

    env = SimpleNamespace(
        robot=SimpleNamespace(arm=arm, _gripper=gripper),
        recorder=None,
        set_start_time=lambda value: None,
        set_instruction=lambda value: None,
        get_state=state,
        move=move,
    )
    args = SimpleNamespace(
        freq=100,
        policy_path="unused",
        policy_node_cfg=SimpleNamespace(action_hz=20, action_dim=7, n_execute=30),
        policy_cfg=SimpleNamespace(binarize_gripper=True, gripper_close=0.1, gripper_open=0.9, gripper_threshold=0.6),
        instruction="test",
        arm_latency=0.0,
        diag_dir=str(tmp_path),
    )
    monkeypatch.setattr(serve, "ActionQueue", Queue)
    monkeypatch.setattr(serve, "RTCConfig", SimpleNamespace)
    monkeypatch.setattr(serve, "LatencyTracker", None)
    monkeypatch.setattr(serve, "torch", SimpleNamespace(as_tensor=Tensor))
    monkeypatch.setattr(serve, "_dump_live_cameras", lambda *args: None)
    monkeypatch.setattr(
        serve,
        "threading",
        SimpleNamespace(Event=lambda: next(events), Thread=lambda **kwargs: SimpleNamespace(start=pause.clear)),
    )
    monkeypatch.setattr(serve.time, "sleep", lambda duration: None)
    monkeypatch.setattr(serve.time, "now", lambda: clock[0])
    monkeypatch.setattr(serve.time, "precise_wait", lambda deadline: clock.__setitem__(0, deadline))
    serve.policy_loop(args, env, policy)
    assert len(commands) == 12
    np.testing.assert_allclose([command[-1] for command in commands], 0.1)
    assert commands[-1][0] > 0.01
    assert stopped


def test_task_pos_action_is_eef_plus_gripper():
    arm = SingleArm(arm=cast(Client, _Arm("TASK_POS_IK")), action_space="TASK_POS")
    assert arm.arm_dim == 6
    action = np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0, 0.7], dtype=np.float32)
    parsed = arm.parse_action(action)
    np.testing.assert_array_equal(parsed["arm_cmd"], action[:6])
    assert parsed["gripper_cmd"] == pytest.approx(0.7)
    assert ACTION_DIM == 7


def test_task_pos_move_dispatches_movel_and_gripper():
    class _RecArm(_Arm):
        def __init__(self):
            super().__init__("TASK_POS_IK")
            self.calls = []

        def moveL(self, pose, target_time):
            self.calls.append(("moveL", list(pose), target_time))

    class _RecGripper:
        def __init__(self):
            self.calls = []

        def moveG(self, target_pos, target_time):
            self.calls.append(("moveG", list(target_pos), target_time))

    arm = _RecArm()
    gripper = _RecGripper()
    robot = SingleArm(arm=cast(Client, arm), gripper=cast(Client, gripper), action_space="TASK_POS")
    action = np.array([0.4, 0.1, 0.3, 0.0, 0.2, 0.0, 0.7], dtype=np.float32)
    robot.move(action, t_cmd_target=1.5)
    assert arm.calls[0][0] == "moveL"
    np.testing.assert_allclose(arm.calls[0][1], action[:6])
    assert gripper.calls[0][0] == "moveG"


def test_create_obs_builds_8d_proprio():
    class _Obs:
        cameras = {"camera_1": SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8))}
        proprio_joints = np.ones(7, dtype=np.float32)
        gripper_position = 0.4

    class _Env:
        def get_state(self):
            return SimpleNamespace(observation=_Obs())

    obs = create_obs(_Env())
    assert obs["proprio"].shape == (8,)
    np.testing.assert_allclose(obs["proprio"][-1], 0.4)


def test_create_obs_keeps_measured_and_commanded_gripper_semantics_distinct():
    observation = SimpleNamespace(
        cameras={},
        proprio_joints=np.zeros(7),
        gripper_position=0.271,
        gripper_commanded_position=0.0,
    )
    env = SimpleNamespace(get_state=lambda: SimpleNamespace(observation=observation))
    assert create_obs(env)["proprio"][-1] == pytest.approx(0.271)
    assert create_obs(env, gripper_state_source="commanded")["proprio"][-1] == 0.0
    observation.gripper_commanded_position = None
    with pytest.raises(RuntimeError, match="commanded position"):
        create_obs(env, gripper_state_source="commanded")


def test_rollout_rejects_stale_camera_frames(monkeypatch):
    from examples import policy_inference

    camera = SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8), meta={"camera_receive_timestamp": 9.0})
    observation = SimpleNamespace(cameras={"camera_1": camera}, proprio_joints=np.zeros(7), gripper_position=0.7)
    env = SimpleNamespace(get_state=lambda: SimpleNamespace(observation=observation))
    monkeypatch.setattr(policy_inference.time, "now", lambda: 10.0)
    with pytest.raises(RuntimeError, match=r"Camera.*stale"):
        create_obs(env, max_camera_age_s=0.25)
    camera.meta["camera_receive_timestamp"] = 9.99
    assert create_obs(env, max_camera_age_s=0.25)["camera_1"].shape == (4, 4, 3)


def test_single_arm_exposes_real_gripper_measurement_and_last_command():
    arm = _Arm("TASK_POS_IK")
    gripper = SimpleNamespace(get_state=lambda: {"gripper_position": 0.271, "gripper_commanded_position": 0.0})
    robot = SingleArm(arm=cast(Client, arm), gripper=cast(Client, gripper), action_space="TASK_POS")
    observation = robot.get_obs({})
    assert observation.gripper_position == pytest.approx(0.271)
    assert observation.gripper_commanded_position == 0.0


def test_pi05_inference_slices_padding_and_clips_gripper():
    pytest.importorskip("torch")
    from rio.policies import pi05 as pi05_mod

    policy = object.__new__(pi05_mod.Pi05)
    policy.device = "cpu"
    policy.instruction = "do the task"
    policy.binarize_gripper = True
    policy.gripper_close = 0.0
    policy.gripper_open = 0.7
    policy.gripper_threshold = None
    policy.preprocessor = lambda batch: batch
    policy.postprocessor = lambda actions: actions
    policy.rtc_execution_horizon = pi05_mod.RTC_EXECUTION_HORIZON

    class _FakeNet:
        def __init__(self):
            self.kwargs = None

        def predict_action_chunk(self, batch, **kwargs):
            self.kwargs = kwargs
            assert batch["observation.state"].shape[-1] == 8
            image = batch["observation.images.camera_1"]
            assert image.shape[0] == 3
            padded = np.full((16, 32), 2.0, dtype=np.float32)
            padded[:, 0] = 0.1
            return padded

    policy.policy = _FakeNet()
    obs = {
        "camera_1": np.zeros((48, 64, 3), dtype=np.uint8),
        "camera_2": np.zeros((48, 64, 3), dtype=np.uint8),
        "proprio_joints": np.zeros(7, dtype=np.float32),
        "gripper_position": 0.7,
    }
    result = policy.inference(obs)
    chunk = result["actions"]
    assert chunk.shape == (16, 7)
    assert chunk.dtype == np.float32
    np.testing.assert_allclose(chunk[:, 0], 0.1)
    np.testing.assert_allclose(chunk[:, -1], 0.7)
    np.testing.assert_allclose(result["original_actions"][:, 0], 0.1)
    assert policy.policy.kwargs == {}

    if pi05_mod.torch is None:
        return
    leftover = np.linspace(0, 1, 12 * 7, dtype=np.float32).reshape(12, 7)
    obs["rtc_leftover"] = leftover
    obs["rtc_inference_delay"] = np.int32(16)
    result = policy.inference(obs)
    assert policy.policy.kwargs["inference_delay"] == 16
    assert policy.policy.kwargs["execution_horizon"] == len(leftover)
    prefix = policy.policy.kwargs["prev_chunk_left_over"].detach().cpu().numpy()
    assert prefix.shape[0] == pi05_mod.RTC_EXECUTION_HORIZON
    np.testing.assert_allclose(prefix[: leftover.shape[0]], leftover)
    np.testing.assert_allclose(prefix[leftover.shape[0] :], 0.0)
    assert result["actions"].shape == (16, 7)


def test_make_policy_resolves_pi05_class():
    from rio.envs.factory import make_policy
    from rio.policies.pi05 import IMPORT_ERROR, Pi05

    if IMPORT_ERROR is not None:
        with pytest.raises(ImportError):
            make_policy("Pi05", {"policy_path": "ckpts/pi05", "device": "cpu"})
        return
    policy = make_policy("Pi05", {"policy_path": "ckpts/pi05", "device": "cpu", "compile": False})
    assert isinstance(policy, Pi05)
    assert policy.policy_path == "ckpts/pi05"


def test_pi05_lerobot_cfg_is_cartesian_without_vive():
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    cfg = Pi05LerobotCfg(policy_path="ckpts/pi05", instruction="do the task")
    assert cfg.policy == "Pi05"
    assert cfg.action_space == "task_pos"
    assert cfg.arm_cfg.robot_controller == "task_pos_ik"
    assert cfg.teleop is None
    assert cfg.teleop_clicker is None
    assert cfg.policy_node_cfg.action_dim == 7
    assert cfg.policy_node_cfg.proprio_dim == 8
    assert cfg.policy_cfg.binarize_gripper is True
    assert cfg.policy_cfg.gripper_open == pytest.approx(0.7)
    assert cfg.gripper_cfg.startup_position is None
    assert cfg.freq == 100
    assert cfg.policy_node_cfg.action_hz == 20
    assert cfg.policy_node_cfg.freq == 20
    assert cfg.policy_node_cfg.use_rtc is True
    assert cfg.policy_node_cfg.n_execute == 30
    assert cfg.policy_node_cfg.rtc_queue_threshold == 18
    assert cfg.policy_node_cfg.rtc_execution_horizon == 18
    assert cfg.policy_node_cfg.chunk_pause_s == pytest.approx(0.0)
    assert cfg.policy_cfg.use_rtc is True
    assert cfg.arm_cfg.cmd_freq == 100
    assert cfg.arm_cfg.max_pos_speed == pytest.approx(0.45)
    assert cfg.arm_cfg.max_rot_speed == pytest.approx(1.0)
    assert list(cfg.cameras) == ["camera_1", "camera_2"]


def test_chunk_hold_pending_blocks_until_deadline():
    assert not chunk_hold_pending(10.0, None)
    assert chunk_hold_pending(10.0, 15.0)
    assert not chunk_hold_pending(15.0, 15.0)


def test_should_send_observation_without_rtc_waits_until_idle():
    kwargs = {"processing_obs": False, "use_rtc": False}
    assert not should_send_observation(remaining=16, **kwargs)
    assert should_send_observation(remaining=0, **kwargs)


def test_first_waypoint_needs_creep_on_large_joint_jump():
    live = np.zeros(8, dtype=np.float32)
    far = np.array([0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    err = command_tracking_error(live, far, joint_serve=True)
    assert first_waypoint_needs_creep(err, joint_serve=True)
    near = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    err_near = command_tracking_error(live, near, joint_serve=True)
    assert not first_waypoint_needs_creep(err_near, joint_serve=True)


def test_first_waypoint_needs_creep_on_large_tcp_jump():
    live = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    far = np.array([0.45, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    err = command_tracking_error(live, far, joint_serve=False)
    assert first_waypoint_needs_creep(err, joint_serve=False)


def test_merge_delay_skips_nothing_on_empty_queue():
    assert merge_delay_steps(1.01, 0.05, leftover_present=False) == 0
    assert merge_delay_steps(1.01, 0.05, leftover_present=True) == 21
    assert merge_delay_steps(0.0, 0.05, leftover_present=True) == 0


def test_merge_delay_uses_consumed_actions_when_queue_ran_dry():
    assert merge_delay_steps(1.3, 0.05, leftover_present=True, consumed_steps=18) == 18
    assert merge_delay_steps(1.3, 0.05, leftover_present=False, consumed_steps=18) == 0


def test_rtc_padding_does_not_extend_guidance_into_nonexistent_actions(monkeypatch):
    from rio.policies import pi05

    monkeypatch.setattr(pi05, "torch", SimpleNamespace(float32=np.float32, as_tensor=lambda value, **kwargs: value))
    monkeypatch.setattr(
        pi05,
        "_normalize_prev_actions_length",
        lambda value, target_steps: np.pad(value[:target_steps], ((0, max(0, target_steps - len(value))), (0, 0))),
    )
    observation = {"rtc_leftover": np.ones((18, 7), dtype=np.float32), "rtc_leftover_length": 3}
    kwargs = pi05._rtc_predict_kwargs(observation, "cpu", execution_horizon=18)
    assert kwargs["execution_horizon"] == 3
    assert kwargs["prev_chunk_left_over"].shape == (18, 7)
    np.testing.assert_array_equal(kwargs["prev_chunk_left_over"][3:], 0.0)
    observation["rtc_leftover_length"] = 0
    assert pi05._rtc_predict_kwargs(observation, "cpu") == {}


def test_short_checkpoint_limits_rtc_prefix_to_available_queue(tmp_path):
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    (tmp_path / "config.json").write_text('{"chunk_size":16,"n_action_steps":16}')
    cfg = Pi05LerobotCfg(policy_path=str(tmp_path), instruction="pick up the cylinder")
    assert cfg.policy_node_cfg.chunk_size == 16
    assert cfg.policy_node_cfg.rtc_queue_threshold == 15
    assert cfg.policy_node_cfg.rtc_execution_horizon == 15


def test_checkpoint_transport_size_uses_prediction_not_execution_horizon(tmp_path):
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    (tmp_path / "config.json").write_text('{"chunk_size":50,"n_action_steps":16}')
    cfg = Pi05LerobotCfg(policy_path=str(tmp_path), instruction="pick up the cylinder")
    assert cfg.policy_node_cfg.chunk_size == 50


def test_should_send_observation_rtc_requests_during_lead():
    kwargs = {"processing_obs": False, "use_rtc": True, "queue_threshold": RTC_QUEUE_THRESHOLD}
    assert not should_send_observation(remaining=40, **kwargs)
    assert not should_send_observation(remaining=19, **kwargs)
    assert should_send_observation(remaining=18, **kwargs)
    assert should_send_observation(remaining=0, **kwargs)
    assert not should_send_observation(remaining=10, processing_obs=True, use_rtc=True)


def test_chunk_size_from_ckpt_reads_n_action_steps(tmp_path):
    (tmp_path / "config.json").write_text('{"n_action_steps": 50, "chunk_size": 50}')
    assert chunk_size_from_ckpt(str(tmp_path)) == 50
    assert chunk_size_from_ckpt(str(tmp_path / "missing"), default=16) == 16


def test_resolve_n_execute_prefers_override_then_ckpt(tmp_path):
    (tmp_path / "config.json").write_text('{"n_action_steps": 25, "chunk_size": 50}')
    assert resolve_n_execute(str(tmp_path), 16) == 16
    assert resolve_n_execute(str(tmp_path), None) == 25
    assert resolve_n_execute(str(tmp_path), 0) == 25


def test_slice_action_chunk_keeps_prefix():
    processed = np.arange(50 * 7, dtype=np.float32).reshape(50, 7)
    original = processed + 1
    out_p, out_o = slice_action_chunk(processed, original, 16)
    assert out_p.shape == (16, 7)
    np.testing.assert_array_equal(out_p, processed[:16])
    np.testing.assert_array_equal(out_o, original[:16])


def test_limit_tcp_action_step_caps_first_jump():
    prev = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    target = np.array([0.58, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = limit_tcp_action_step(target, prev, dt=0.01, max_pos_speed=0.45, max_rot_speed=1.0)
    np.testing.assert_allclose(out[:3], [0.4045, 0.0, 0.3], atol=1e-6)
    assert out[-1] == pytest.approx(0.0)


def test_limit_tcp_action_step_caps_rotation():
    prev = np.zeros(7, dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7], dtype=np.float32)
    out = limit_tcp_action_step(target, prev, dt=0.01, max_pos_speed=0.45, max_rot_speed=1.0)
    assert float(np.linalg.norm(out[3:6])) == pytest.approx(0.01, abs=1e-5)


def test_limit_joint_action_step_caps_per_joint():
    prev = np.zeros(8, dtype=np.float32)
    target = np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    out = limit_joint_action_step(target, prev, dt=0.01, max_joint_speed=1.0)
    np.testing.assert_allclose(out[:2], [0.01, -0.01], atol=1e-6)
    assert out[-1] == pytest.approx(0.7)


def test_as_action_np_snaps_interpolated_gripper():
    x = _as_action_np(np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2], dtype=np.float32), 7, binarize_gripper=True)
    assert x[-1] == pytest.approx(0.0)
    x_open = _as_action_np(np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4], dtype=np.float32), 7, binarize_gripper=True)
    assert x_open[-1] == pytest.approx(0.7)


def test_as_action_np_skips_rotvec_fold_for_joints():
    raw = np.array([0.1, 0.2, 0.3, 4.0, 0.0, 0.0, 0.1, 0.7], dtype=np.float32)
    x = _as_action_np(raw, 8, binarize_gripper=True, fold_rotvec=False)
    assert x[3] == pytest.approx(4.0)
    assert x[-1] == pytest.approx(0.7)


def test_infer_action_space_rejects_tcp_ckpt_in_joint_mode():
    cfg = SimpleNamespace(action_feature_names=["x", "y", "z", "rx", "ry", "rz", "gripper"])
    assert infer_action_space(cfg) == "tcp"
    with pytest.raises(ValueError, match="TCP"):
        infer_action_space(cfg, "joints")
    joint_cfg = SimpleNamespace(action_feature_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"])
    assert infer_action_space(joint_cfg, "joints") == "joints"


def test_pi05_lerobot_joint_cfg_sets_movej():
    from examples.policy_cfgs.pi05_lerobot_joint import Pi05LerobotJointCfg

    cfg = Pi05LerobotJointCfg(policy_path="ckpts/pi05_joints", instruction="do the task")
    assert cfg.action_space == "joint_pos"
    assert cfg.arm_cfg.robot_controller == "joint_pos"
    assert cfg.arm_cfg.stream_j_speed == pytest.approx(0.15)
    assert cfg.policy_node_cfg.action_dim == 8
    assert cfg.policy_cfg.action_space == "joints"


def test_pi05_lerobot_joint_cfg_rejects_tcp_checkpoint(tmp_path):
    from examples.policy_cfgs.pi05_lerobot_joint import Pi05LerobotJointCfg

    (tmp_path / "config.json").write_text(
        '{"action_feature_names": ["x", "y", "z", "rx", "ry", "rz", "gripper"], "chunk_size": 50}'
    )
    with pytest.raises(ValueError, match="TCP checkpoint"):
        Pi05LerobotJointCfg(policy_path=str(tmp_path), instruction="do the task")


def test_lerobot_interpolator_multiplier_five():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.utils.action_interpolator import ActionInterpolator

    interp = ActionInterpolator(multiplier=5)
    interp.add(torch.zeros(7))
    first = interp.get()
    assert first is not None
    assert interp.needs_new_action()
    interp.add(torch.ones(7))
    vals = [float(interp.get()[0]) for _ in range(5)]
    np.testing.assert_allclose(vals, [0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-5)


def test_action_queue_merge_skips_inference_delay():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.rtc import ActionQueue, RTCConfig

    queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0))
    first = torch.arange(50, dtype=torch.float32)[:, None].repeat(1, 7)
    queue.merge(first, first, 0)
    for _ in range(16):
        queue.get()
    leftover = queue.get_left_over()
    assert leftover.shape[0] == 34
    second = torch.arange(50, dtype=torch.float32)[:, None].repeat(1, 7) + 100
    queue.merge(second, second, 16, action_index_before_inference=0)
    played = queue.get()
    np.testing.assert_allclose(played[0].item(), 116.0)


def test_rtc_prefix_weights_cover_splice_when_horizon_exceeds_delay():
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.rtc import RTCConfig, RTCProcessor

    processor = RTCProcessor(RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0))
    short = processor.get_prefix_weights(18, 10, 50)
    assert float(short[18]) == pytest.approx(0.0)
    long = processor.get_prefix_weights(18, 30, 50)
    assert float(long[17]) == pytest.approx(1.0)
    assert float(long[18]) > 0.5
    assert float(long[29]) >= 0.0
    assert float(long[30]) == pytest.approx(0.0)


def test_serve_control_enter_toggles_pause_and_q_quits():
    pause = threading.Event()
    quit_ev = threading.Event()
    pause.set()
    apply_serve_control("", pause, quit_ev)
    assert not pause.is_set()
    assert not quit_ev.is_set()
    apply_serve_control("", pause, quit_ev)
    assert pause.is_set()
    apply_serve_control("p", pause, quit_ev)
    assert pause.is_set()
    pause.clear()
    apply_serve_control("pause", pause, quit_ev)
    assert pause.is_set()
    apply_serve_control("q", pause, quit_ev)
    assert quit_ev.is_set()
