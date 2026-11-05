# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""DAgger takeover: frame labels, segment selection, session control, serve loop, conversion."""

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import robodm

from examples import dagger_kassow
from examples import policy_inference as serve
from examples.cfg.kassow_experiments import EXPERIMENTS, dagger_rounds, dataset_tasks, load_state
from examples.dagger_kassow import DaggerSession, next_traj_index, unique_path
from examples.data.convert_to_lerobot_pi05 import ConvertExperimentCfg, main
from examples.experiment import dagger_round_summary
from examples.policy_cfgs.pi05_dagger import Pi05DaggerCfg
from examples.policy_inference import RolloutSafetyCfg
from examples.teleop_vive_hand import write_home_state
from rio.data.convert_pi05 import ConvertPi05Cfg, convert
from rio.data.dagger import CONTROL_KEY, ControlSource, control_summary, dagger_segments

from .test_kassow_experiment import _register_model

P, H, W = int(ControlSource.POLICY), int(ControlSource.HUMAN), int(ControlSource.HOLD)


def _control(*runs: tuple[int, int]) -> np.ndarray:
    return np.concatenate([np.full(length, value, dtype=np.int8) for value, length in runs])


# --- segment selection -------------------------------------------------------------------------


@pytest.mark.unit
def test_human_mode_keeps_only_operator_runs():
    control = _control((P, 60), (H, 40), (W, 10), (P, 30), (H, 25))
    assert dagger_segments(control, mode="human", fps=20) == [(60, 100), (140, 165)]


@pytest.mark.unit
def test_all_mode_cuts_before_each_takeover_and_drops_hold():
    control = _control((P, 60), (H, 40), (W, 10), (P, 30), (H, 25))
    # 1 s = 20 frames before each takeover is the policy going wrong; hold frames never train.
    assert dagger_segments(control, mode="all", fps=20, pre_takeover_cut_s=1.0) == [(0, 40), (60, 100), (140, 165)]


@pytest.mark.unit
def test_short_runs_and_autonomous_rollouts():
    tap = _control((P, 50), (H, 5), (P, 50))
    assert dagger_segments(tap, mode="human", fps=20, min_segment_s=1.0) == []
    autonomous = _control((P, 80))
    assert dagger_segments(autonomous, mode="human", fps=20) == []
    assert dagger_segments(autonomous, mode="human", fps=20, include_autonomous=True) == [(0, 80)]
    assert dagger_segments(autonomous, mode="whole", fps=20) == [(0, 80)]


def _action(still: list[tuple[int, int]], n: int, switch_at: int | None = None) -> np.ndarray:
    """Commanded TCP moving 4 cm/s at 20 Hz, except standing still from frame s up to frame e."""
    step = np.full(n, 0.002)
    for s, e in still:
        step[s:e] = 0.0
    action = np.zeros((n, 7))
    action[:, 0] = np.concatenate([[0.4], 0.4 + np.cumsum(step[:-1])])
    action[:, 6] = 0.7
    if switch_at is not None:
        action[switch_at:, 6] = 0.0
    return action


@pytest.mark.unit
def test_handover_stillness_is_cut_from_operator_runs():
    control = _control((P, 20), (H, 40), (P, 20))
    # 0.5 s reaction after the takeover, then a 0.5 s pause (clutch released to re-grip).
    action = _action([(20, 30), (45, 55)], 80)
    trimmed = dagger_segments(control, mode="human", fps=20, min_segment_s=0.2, action=action, max_pause_s=0.3)
    assert trimmed == [(30, 45), (55, 60)]
    untrimmed = dagger_segments(control, mode="human", fps=20, min_segment_s=0.2, action=action, max_pause_s=None)
    assert untrimmed == [(20, 60)]


@pytest.mark.unit
def test_grasp_pauses_and_the_episode_end_are_kept():
    control = _control((P, 20), (H, 40), (P, 20))
    grasp = _action([(20, 30), (45, 55)], 80, switch_at=50)
    assert dagger_segments(control, mode="human", fps=20, min_segment_s=0.2, action=grasp, max_pause_s=0.3) == [(30, 60)]
    ending = _control((P, 20), (H, 40))
    still_end = _action([(20, 30), (50, 60)], 60)
    assert dagger_segments(ending, mode="human", fps=20, min_segment_s=0.2, action=still_end, max_pause_s=0.3) == [(30, 60)]


@pytest.mark.unit
def test_unknown_segment_mode_is_rejected():
    with pytest.raises(ValueError, match="segments"):
        dagger_segments(_control((P, 5)), mode="operator", fps=20)


@pytest.mark.unit
def test_control_summary_counts_takeovers_and_handbacks():
    summary = control_summary(_control((P, 100), (H, 50), (W, 10), (P, 40), (H, 20)), fps=100)
    assert summary["frames"] == {"policy": 140, "human": 70, "hold": 10}
    assert summary["takeovers"] == 2
    assert summary["handbacks"] == 1
    assert summary["duration_s"] == pytest.approx(2.2)


@pytest.mark.unit
def test_rollout_file_helpers(tmp_path):
    assert next_traj_index(tmp_path) == 0
    (tmp_path / "traj_0004.vla").write_bytes(b"")
    (tmp_path / "traj_0001.vla").write_bytes(b"")
    assert next_traj_index(tmp_path) == 5
    assert unique_path(tmp_path / "new.vla") == tmp_path / "new.vla"
    assert unique_path(tmp_path / "traj_0004.vla") == tmp_path / "traj_0004_1.vla"


# --- session control with fake hardware --------------------------------------------------------


class _Keys:
    def __init__(self):
        self.held: set[str] = set()

    def get_state(self):
        alpha = np.zeros(6, dtype=np.int64)
        special = np.zeros(6, dtype=np.int64)
        for i, key in enumerate(sorted(k for k in self.held if len(k) == 1 and k != " ")):
            alpha[i] = ord(key)
        if " " in self.held:
            special[0] = 0x2C
        return {"alphanumeric_state": alpha, "special_state": special}


class _Recorder:
    def __init__(self, folder: Path, clock: list[float]):
        self.folder = folder
        self.clock = clock
        self.closed = True
        self.index = next_traj_index(folder) - 1
        self.path: Path | None = None
        self.labels: list[int] = []

    def new_trajectory(self, wait=True, timeout=None):
        self.index += 1
        self.path = self.folder / f"traj_{self.index:04d}.vla"
        self.closed = False

    def record_step(self, step):
        self.labels.append(int(step.meta["control_source"]))

    def save(self, wait=False):
        assert self.path is not None
        self.path.write_bytes(b"vla")
        self.closed = True

    def get_state(self):
        return {"is_closed": self.closed, "is_saving": False, "timestamp": self.clock[0]}


def _session(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(dagger_kassow.time, "now", lambda: clock[0])
    round_dir = tmp_path / "r1"
    round_dir.mkdir()
    home_file = tmp_path / "home_sort.json"
    write_home_state(home_file, {"joint_q": np.zeros(7), "eef_pose": np.zeros(6), "gripper": 0.7}, experiment="sort")
    eef = np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0])
    arm = SimpleNamespace(get_state=lambda: {"eef_pose": eef.copy(), "joint_q": np.zeros(7), "timestamp": clock[0]})
    recorder = _Recorder(round_dir, clock)
    env = SimpleNamespace(robot=SimpleNamespace(arm=arm), recorder=recorder)
    tracker = {"pose_valid": 1.0, "tracker_pose": np.array([1.0, 2.0, 1.0, 0.0, 0.0, 0.0])}
    teleop = SimpleNamespace(get_state=lambda: tracker)
    keys = _Keys()
    args = SimpleNamespace(
        recorder_cfg=SimpleNamespace(path=str(round_dir)),
        home_file=str(home_file),
        experiment="sort",
        gripper_open=0.7,
        gripper_close=0.0,
        pos_scale=1.0,
        rot_scale=1.0,
        max_pos_speed=0.45,
        max_rot_speed=1.0,
        min_radius=0.25,
        max_radius=1.4,
        min_z=0.05,
        orientation_enabled=True,
        clutch_key="c",
        orientation_key="o",
        tracking_grace=0.25,
        max_lag=0.25,
        freq=100,
        dagger_round=1,
        instruction="put the object in space 2",
        policy_path="/models/run1",
        home_joint_speed=0.15,
    )
    session = DaggerSession(args, env, teleop, [keys], yaw_deg=0.0)
    return session, keys, clock, recorder


def _press(session, keys, key, command):
    keys.held = {key}
    event = session.poll(command)
    keys.held = set()
    session.poll(command)
    return event


@pytest.mark.unit
def test_takeover_engages_at_the_last_command_without_a_jump(tmp_path, monkeypatch):
    session, keys, _, _ = _session(tmp_path, monkeypatch)
    command = np.array([0.42, 0.12, 0.31, 0.0, 0.0, 0.1, 0.0], dtype=np.float32)
    assert _press(session, keys, "c", command) == "takeover"
    assert session.operator_driving
    assert session.retargeter.engaged
    # Tracker unmoved since engage: the target is the last policy command, gripper kept closed.
    target = session.human_command(np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0]), 0.01)
    np.testing.assert_allclose(target, command, atol=1e-6)


@pytest.mark.unit
def test_handback_needs_a_released_clutch(tmp_path, monkeypatch):
    session, keys, _, _ = _session(tmp_path, monkeypatch)
    command = np.zeros(7, dtype=np.float32)
    command[:3] = [0.4, 0.1, 0.3]
    _press(session, keys, "c", command)
    assert _press(session, keys, "r", command) is None
    assert session.operator_driving
    assert _press(session, keys, "c", command) is None
    assert not session.retargeter.engaged
    assert _press(session, keys, "r", command) == "handback"
    assert session.waiting_for_policy
    session.resume_policy()
    assert not session.operator_driving and not session.waiting_for_policy


@pytest.mark.unit
def test_operator_limits_follow_the_clutch_lag_only_while_driving(tmp_path, monkeypatch):
    session, keys, _, _ = _session(tmp_path, monkeypatch)
    safety = RolloutSafetyCfg()
    assert session.limits(safety) is safety
    _press(session, keys, "c", np.r_[0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 0.7])
    assert session.limits(safety).max_position_error_m == pytest.approx(0.30)
    assert session.limits(safety).max_state_age_s == safety.max_state_age_s


@pytest.mark.unit
def test_episodes_are_labelled_and_discarded_files_are_kept(tmp_path, monkeypatch):
    session, keys, clock, recorder = _session(tmp_path, monkeypatch)
    command = np.r_[0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 0.7]

    def ticks(n):
        for _ in range(n):
            session.record(SimpleNamespace(meta={}))

    session.begin_episode()
    ticks(30)
    _press(session, keys, "c", command)
    ticks(20)
    _press(session, keys, "c", command)
    _press(session, keys, "r", command)
    ticks(5)
    session.resume_policy()
    ticks(10)
    session.end_episode(saved=True)
    session.begin_episode()
    ticks(15)
    session.end_episode(saved=False, reason="quit")
    assert recorder.labels == [P] * 30 + [H] * 20 + [W] * 5 + [P] * 10 + [P] * 15
    session.finalize_pending()
    assert session._pending, "must wait until the recorder published after the save"
    clock[0] += 1.0
    session.finalize_pending()
    saved = json.loads((tmp_path / "r1" / "traj_0000.dagger.json").read_text())
    assert saved["outcome"] == "saved" and saved["takeovers"] == 1 and saved["handbacks"] == 1
    assert saved["frames"] == {"policy": 40, "human": 20, "hold": 5}
    assert saved["instruction"] == "put the object in space 2"
    assert not (tmp_path / "r1" / "traj_0001.vla").exists()
    discarded = json.loads((tmp_path / "r1" / "discarded" / "traj_0001.dagger.json").read_text())
    assert discarded["outcome"] == "discarded" and discarded["reason"] == "quit" and discarded["autonomous"]
    assert (tmp_path / "r1" / "discarded" / "traj_0001.vla").is_file()
    summary = dagger_round_summary(tmp_path / "r1")
    assert summary.startswith("1 saved (0 without takeover), 1 discarded")
    assert "1.00 takeovers/episode" in summary and "operator 33% of frames" in summary


class _KeyScript(_Keys):
    """One key set per ``get_state`` call; an optional hook runs when that step is read."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)

    def get_state(self):
        keys, hook = self.steps.pop(0) if self.steps else (set(), None)
        if hook is not None:
            hook()
        self.held = keys
        return super().get_state()


def _homing_arm(session, clock, monkeypatch, *, q0, reaches: bool):
    q = [np.asarray(q0, dtype=float)]
    moves = []

    def move_j(target, duration):
        moves.append(float(duration))
        if reaches:
            q[0] = np.asarray(target, dtype=float).copy()

    monkeypatch.setattr(dagger_kassow.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    session.env.robot.arm = SimpleNamespace(
        get_state=lambda: {"eef_pose": np.zeros(6), "joint_q": q[0].copy(), "timestamp": clock[0], "alarm_code": 0},
        moveJ_discrete=move_j,
    )
    return q, moves


@pytest.mark.unit
def test_n_drives_home_again_if_the_arm_was_moved_before_the_start(tmp_path, monkeypatch):
    session, _, clock, recorder = _session(tmp_path, monkeypatch)
    q, moves = _homing_arm(session, clock, monkeypatch, q0=np.zeros(7), reaches=True)
    pushed = lambda: q.__setitem__(0, np.full(7, 0.2))
    session.keyboards = [_KeyScript([(set(), None), (set(), None), ({"n"}, pushed)])]
    assert session.wait_for_episode()
    assert len(moves) == 1
    assert session.at_home()
    assert recorder.closed


@pytest.mark.unit
def test_episode_does_not_start_away_from_home(tmp_path, monkeypatch):
    session, _, clock, recorder = _session(tmp_path, monkeypatch)
    _, moves = _homing_arm(session, clock, monkeypatch, q0=np.full(7, 0.2), reaches=False)
    session.keyboards = [_KeyScript([(set(), None), ({"n"}, None), (set(), None), ({"q"}, None)])]
    assert not session.wait_for_episode()
    # Once after the previous episode, once more on n; neither reached home, so nothing started.
    assert len(moves) == 2
    assert recorder.path is None


@pytest.mark.unit
def test_dagger_cfg_records_into_the_round_folder_not_raw(tmp_path):
    _register_model("sort", tmp_path)
    state = load_state(EXPERIMENTS["sort"])
    state["frozen"] = True
    (EXPERIMENTS["sort"].state_path).write_text(json.dumps(state))
    cfg = Pi05DaggerCfg(experiment="sort", dagger_round=2)
    assert cfg.recorder == "Recorder"
    assert cfg.recorder_cfg.path == str(EXPERIMENTS["sort"].root / "dagger" / "r2")
    assert cfg.teleop == "ViveTracker" and cfg.teleop_keyboard == "SshKeyboard"
    assert cfg.instruction == "put the object in space 1"
    assert cfg.policy_node_cfg.use_rtc
    with pytest.raises(ValueError, match="dagger-round"):
        Pi05DaggerCfg(experiment="sort", dagger_round=0)


# --- serve loop: takeover and handback with RTC on ----------------------------------------------


class _Tensor:
    def __init__(self, value):
        self.array = np.asarray(value, dtype=np.float32)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return _Tensor(self.array[key])

    @property
    def shape(self):
        return self.array.shape

    def numel(self):
        return int(self.array.size)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _Queue:
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

    def get_left_over(self):
        return _Tensor(self.buffer[self.index :])

    def merge(self, original, processed, delay, **kwargs):
        self.buffer = processed.array[delay:].copy()
        self.index = 0

    def get_processed_left_over(self):
        return _Tensor(self.buffer[self.index :])

    def get(self):
        if not self.qsize():
            return None
        action = _Tensor(self.buffer[self.index])
        self.index += 1
        return action


class _ScriptedDagger:
    """Stands in for ``DaggerSession``: events by control tick, a fixed operator pose."""

    def __init__(self, script: dict[int, str], human_pose: np.ndarray):
        self.script = script
        self.human_pose = human_pose
        self.mode = ControlSource.POLICY
        self.tick = 0
        self.recording = False
        self.labels: list[int] = []
        self.ended: list[tuple[bool, str | None]] = []
        self.resumed_at: list[int] = []
        self.episodes = 0

    @property
    def operator_driving(self):
        return self.mode is ControlSource.HUMAN

    @property
    def waiting_for_policy(self):
        return self.mode is ControlSource.HOLD

    def limits(self, safety):
        return safety

    def wait_for_episode(self):
        self.episodes += 1
        return self.episodes == 1

    def begin_episode(self):
        self.recording = True

    def poll(self, command):
        event = self.script.get(self.tick)
        self.tick += 1
        if event == "takeover":
            self.mode = ControlSource.HUMAN
        elif event == "handback":
            self.mode = ControlSource.HOLD
        return event

    def human_command(self, eef_pose, dt):
        return np.r_[self.human_pose, 0.0].astype(np.float32)

    def resume_policy(self):
        self.mode = ControlSource.POLICY
        self.resumed_at.append(self.tick)

    def record(self, step):
        self.labels.append(int(self.mode))

    def status(self, t):
        pass

    def end_episode(self, *, saved, reason=None):
        self.recording = False
        self.ended.append((saved, reason))


@pytest.mark.unit
def test_serve_loop_drops_the_stale_chunk_and_restarts_after_handback(monkeypatch, tmp_path):
    clock = [0.0]
    pose = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    grip = [0.7]
    commands: list[np.ndarray] = []
    calls: list[str] = []
    polls_until_ready = [None]

    stale = np.tile(np.r_[0.9, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7].astype(np.float32), (30, 1))
    fresh = np.tile(np.r_[0.405, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7].astype(np.float32), (30, 1))
    chunks = iter([stale] + [fresh] * 10)
    current = [stale]

    def send_observation(observation):
        calls.append("send")
        current[0] = next(chunks)
        polls_until_ready[0] = 3

    def get_action_chunk():
        if polls_until_ready[0] is None:
            return {"ready": False}
        polls_until_ready[0] -= 1
        if polls_until_ready[0] > 0:
            return {"ready": False}
        polls_until_ready[0] = None
        return {"ready": True, "actions": current[0], "original_actions": current[0].copy()}

    policy = SimpleNamespace(
        policy_loaded=True,
        reset=lambda: calls.append("reset"),
        send_observation=send_observation,
        get_action_chunk=get_action_chunk,
    )
    arm = SimpleNamespace(
        robot_controller=SimpleNamespace(name="TASK_POS_IK"),
        get_state=lambda: {"eef_pose": pose.copy(), "timestamp": clock[0], "alarm_code": 0},
        stop_motion=lambda: calls.append("stop"),
    )
    gripper = SimpleNamespace(get_state=lambda: {"gripper_position": grip[0], "gripper_commanded_position": grip[0]})

    def state(**kwargs):
        camera = SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8), meta={"camera_receive_timestamp": clock[0]})
        return SimpleNamespace(
            observation=SimpleNamespace(cameras={"camera_1": camera}, proprio_joints=np.zeros(7), gripper_position=grip[0]),
            meta={},
        )

    def move(action, **kwargs):
        commands.append(np.asarray(action).copy())
        pose[:] = action[:6]
        grip[0] = float(action[-1])

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
        policy_node_cfg=SimpleNamespace(action_hz=20, action_dim=7, n_execute=30, use_rtc=True, rtc_queue_threshold=18),
        policy_cfg=SimpleNamespace(binarize_gripper=True, gripper_close=0.0, gripper_open=0.7),
        instruction="put the object in space 2",
        arm_latency=0.0,
        diag_dir=str(tmp_path),
    )
    human_pose = np.array([0.43, 0.02, 0.3, 0.0, 0.0, 0.0])
    dagger = _ScriptedDagger({1: "takeover", 20: "handback", 120: "save"}, human_pose)
    monkeypatch.setattr(serve, "ActionQueue", _Queue)
    monkeypatch.setattr(serve, "RTCConfig", SimpleNamespace)
    monkeypatch.setattr(serve, "LatencyTracker", None)
    monkeypatch.setattr(serve, "torch", SimpleNamespace(as_tensor=_Tensor))
    monkeypatch.setattr(serve, "_dump_live_cameras", lambda *args: None)
    monkeypatch.setattr(serve.time, "sleep", lambda duration: None)
    monkeypatch.setattr(serve.time, "now", lambda: clock[0])
    monkeypatch.setattr(serve.time, "precise_wait", lambda deadline: clock.__setitem__(0, deadline))
    monkeypatch.setattr(serve, "threading", SimpleNamespace(Event=threading.Event, Thread=None))

    serve.policy_loop(args, env, policy, dagger=dagger)

    xs = np.array([command[0] for command in commands])
    # The chunk requested before the takeover (x=0.9) is never played.
    assert xs.max() < 0.5
    # The operator pose is sent while driving; after the handback the fresh plan takes over.
    assert np.isclose(xs, 0.43).sum() >= 18
    assert dagger.resumed_at and dagger.resumed_at[0] > 20
    assert xs[-1] == pytest.approx(0.405, abs=1e-3)
    # The stale chunk arrives while the operator drives; reset and the new request follow the handback.
    sends = [i for i, call in enumerate(calls) if call == "send"]
    assert "reset" in calls[sends[0] : sends[1]]
    # RTC resumes normally afterwards: the next request overlaps the running fresh chunk.
    assert len(sends) >= 3
    assert dagger.ended == [(True, None)]
    assert set(dagger.labels) == {P, H, W}


# --- conversion of rollouts and merge with the original demos ----------------------------------


def _write_rollout(path: Path, control: np.ndarray, instruction: str | None) -> None:
    rng = np.random.default_rng(int(control.sum()))
    traj = robodm.Trajectory(path=str(path), mode="w", video_codec="rawvideo")
    for i, value in enumerate(control):
        t = i * 50
        for camera in ("camera_1", "camera_2"):
            traj.add(
                f"observation/cameras/{camera}/rgb",
                rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
                timestamp=t,
                time_unit="ms",
            )
        traj.add("observation/proprio_joints", np.full(7, i * 0.01, dtype=np.float32), timestamp=t, time_unit="ms")
        traj.add("observation/gripper_position", np.float32(0.7), timestamp=t, time_unit="ms")
        traj.add("observation/gripper_commanded_position", np.float32(0.7), timestamp=t, time_unit="ms")
        action = np.array([0.4 + i * 0.001, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
        traj.add("action", action, timestamp=t, time_unit="ms")
        traj.add(CONTROL_KEY, np.int8(value), timestamp=t, time_unit="ms")
        if instruction is not None:
            traj.add("instruction", instruction, timestamp=t, time_unit="ms")
    traj.close()


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.integration
def test_rollouts_become_operator_episodes_appended_to_an_untouched_base(tmp_path):
    pytest.importorskip("lerobot")
    sort = EXPERIMENTS["sort"]
    sort.raw_dir.mkdir(parents=True)
    _write_rollout(sort.raw_dir / "traj_0000.vla", _control((P, 30)), None)
    (sort.raw_dir / "tasks.yaml").write_text("0: sort the packages\n")
    shared: dict[str, Any] = {
        "src_fps": 20,
        "dst_fps": 20,
        "min_frames": 4,
        "trim_idle": False,
        "vcodec": "h264",
        "image_writer_threads": 1,
    }
    base = sort.lerobot_dir / "base"
    convert(ConvertPi05Cfg(input=sort.raw_dir, output=base, tasks=sort.raw_dir / "tasks.yaml", repo_id="local/b", **shared))
    before = _tree_digest(base)

    round_dir = sort.dagger_round_dir(1)
    round_dir.mkdir(parents=True)
    _write_rollout(
        round_dir / "traj_0000.vla", _control((P, 20), (H, 12), (W, 3), (P, 10), (H, 8)), "put the object in space 2"
    )
    _write_rollout(round_dir / "traj_0001.vla", _control((P, 25)), "put the object in space 2")
    assert dagger_rounds(sort) == [1]

    result = main(
        ConvertExperimentCfg(
            experiment="sort",
            variant="dagger_r1",
            base_dataset="base",
            dagger_rounds=(1,),
            min_segment_s=0.3,
            gripper_state_source="measured",
            **shared,
        )
    )
    # One base episode plus the two operator runs (12 and 8 frames); the autonomous rollout adds none.
    assert result["kept"] == 3
    assert result["dagger_episodes"] == {1: 2}
    info = json.loads((sort.lerobot_dir / "dagger_r1" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3 and info["total_frames"] == 30 + 12 + 8
    assert sorted(dataset_tasks(sort.lerobot_dir / "dagger_r1")) == ["put the object in space 2", "sort the packages"]
    assert _tree_digest(base) == before
    notes = load_state(sort)["datasets"]["dagger_r1"]
    assert notes["base_dataset"] == "base" and notes["dagger_rounds"] == [1] and notes["segments"] == "human"

    with pytest.raises(FileExistsError):
        main(ConvertExperimentCfg(experiment="sort", variant="dagger_r1", base_dataset="base", dagger_rounds=(1,), **shared))
    with pytest.raises(ValueError, match="base-dataset"):
        main(ConvertExperimentCfg(experiment="sort", variant="d2", dagger_rounds=(1,), **shared))
