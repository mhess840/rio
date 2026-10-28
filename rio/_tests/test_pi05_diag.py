# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for π₀.₅ model-vs-control diagnostics (no GPU, no robot)."""

import csv
import json

import numpy as np
import pytest

from examples.diagnose_pi05_offline import (
    episode_frac_index,
    episode_mid_index,
    episode_start_index,
    gripper_match,
    joint_ade_rad,
    joint_span_rad,
    parse_sample_at,
    resolve_instruction,
    rio_obs_from_frame,
    to_hwc_float32,
    to_hwc_uint8,
    xyz_ade_m,
    xyz_span_m,
)
from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg
from examples.policy_inference import append_cmd_eef, create_diagnostic_session, write_chunk_dump

pytestmark = pytest.mark.unit


def test_episode_mid_index_fits_horizon():
    assert episode_mid_index(0, 200, 50) == 100
    assert episode_mid_index(0, 80, 50) == 30
    with pytest.raises(ValueError, match="horizon"):
        episode_mid_index(0, 40, 50)


def test_episode_start_index_is_first_frame():
    assert episode_start_index(10, 200, 25) == 10
    with pytest.raises(ValueError, match="horizon"):
        episode_start_index(0, 20, 25)


def test_episode_frac_and_parse_sample_at():
    assert episode_frac_index(0, 200, 50, 0.0) == 0
    assert episode_frac_index(0, 200, 50, 1.0) == 150
    assert parse_sample_at("start") == [0.0]
    assert parse_sample_at("mid") == [0.5]
    assert parse_sample_at("0,0.25,0.5") == [0.0, 0.25, 0.5]


def test_xyz_ade_and_span():
    gt = np.zeros((10, 7), dtype=np.float32)
    gt[:, 0] = np.linspace(0.0, 0.20, 10)
    pred = gt.copy()
    pred[:, 0] += 0.01
    ade, mx = xyz_ade_m(pred, gt)
    assert ade == pytest.approx(0.01, abs=1e-5)
    assert mx == pytest.approx(0.01, abs=1e-5)
    assert xyz_span_m(gt) == pytest.approx(0.20, abs=1e-5)


def test_joint_ade_and_span():
    gt = np.zeros((10, 8), dtype=np.float32)
    gt[:, 0] = np.linspace(0.0, 0.40, 10)
    pred = gt.copy()
    pred[:, 0] += 0.02
    ade, mx = joint_ade_rad(pred, gt)
    assert ade == pytest.approx(0.02, abs=1e-5)
    assert mx == pytest.approx(0.02, abs=1e-5)
    assert joint_span_rad(gt) == pytest.approx(0.40, abs=1e-5)


def test_gripper_match_snaps():
    pred = np.zeros((4, 7), dtype=np.float32)
    gt = np.zeros((4, 7), dtype=np.float32)
    pred[:, -1] = [0.0, 0.2, 0.6, 0.7]
    gt[:, -1] = [0.0, 0.0, 0.7, 0.7]
    assert gripper_match(pred, gt) == pytest.approx(1.0)


def test_to_hwc_uint8_from_chw_float():
    chw = np.zeros((3, 4, 5), dtype=np.float32)
    chw[0] = 1.0
    out = to_hwc_uint8(chw)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 255


def test_rio_obs_from_frame_matches_serve_keys():
    frame = {
        "observation.state": np.arange(8, dtype=np.float32),
        "observation.images.camera_1": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation.images.camera_2": np.ones((2, 3, 3), dtype=np.uint8),
    }
    obs = rio_obs_from_frame(frame, "pick up the cube")
    assert obs["proprio"].shape == (8,)
    assert obs["camera_1"].shape == (2, 3, 3)
    assert obs["gripper_position"] == pytest.approx(7.0)


def test_offline_input_preserves_float_pixels_without_byte_quantization():
    image = np.full((3, 4, 5), 0.1234567, dtype=np.float32)
    result = to_hwc_float32(image)
    assert result.dtype == np.float32
    assert result.shape == (4, 5, 3)
    np.testing.assert_array_equal(result, np.transpose(image, (1, 2, 0)))


def test_offline_instruction_uses_the_actual_dataset_task():
    assert resolve_instruction({"task": ["pick up the blue object"]}) == "pick up the blue object"
    assert resolve_instruction({"task": "place the object"}, "explicit test") == "explicit test"
    with pytest.raises(ValueError, match="task label"):
        resolve_instruction({})


def test_camera_capture_timestamps_are_retained_in_observations():
    from types import SimpleNamespace

    from rio.envs.env import Env

    env = object.__new__(Env)
    env.cameras = {
        "camera_1": SimpleNamespace(
            get_state=lambda: {
                "color": np.zeros((4, 4, 3), dtype=np.uint8),
                "camera_capture_timestamp": 9.9,
                "camera_receive_timestamp": 10.0,
            }
        )
    }
    camera = env.get_cameras()["camera_1"]
    assert camera.meta["camera_capture_timestamp"] == 9.9
    assert camera.meta["camera_receive_timestamp"] == 10.0


def test_write_chunk_dump_and_cmd_eef(tmp_path):
    leftover = np.zeros((8, 7), dtype=np.float32)
    leftover[-1, :3] = [0.05, 0.0, 0.0]
    eef = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    write_chunk_dump(tmp_path, 0, leftover, eef, delay=18, qsize=32, latency_s=0.9)
    assert (tmp_path / "chunks" / "chunk_0000.npy").is_file()
    text = (tmp_path / "chunks.csv").read_text()
    assert "span_m" in text
    assert "0.050000" in text
    cmd = np.array([0.11, 0.2, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    append_cmd_eef(tmp_path, 1.5, cmd, eef, remaining=20, chunk_idx=0)
    cmd_csv = (tmp_path / "cmd_vs_eef.csv").read_text()
    assert "err_m" in cmd_csv


def test_pi05_lerobot_cfg_rtc_off_syncs_policy_cfg():
    cfg = Pi05LerobotCfg(policy_path="ckpts/pi05", instruction="do the task")
    cfg.policy_node_cfg.use_rtc = False
    cfg.policy_cfg.use_rtc = True
    Pi05LerobotCfg.__post_init__(cfg)
    assert cfg.policy_node_cfg.use_rtc is False
    assert cfg.policy_cfg.use_rtc is False


def test_diagnostic_sessions_never_overwrite_an_earlier_run(tmp_path):
    first = create_diagnostic_session(tmp_path, {"n_execute": 30})
    second = create_diagnostic_session(tmp_path, {"n_execute": 25})
    assert first != second
    assert json.loads((first / "session.json").read_text())["n_execute"] == 30
    assert json.loads((second / "session.json").read_text())["n_execute"] == 25
    actions = np.zeros((2, 7), dtype=np.float32)
    write_chunk_dump(first, 0, actions, np.zeros(6), delay=0, qsize=2, latency_s=0.1)
    write_chunk_dump(second, 0, actions + 1, np.zeros(6), delay=0, qsize=2, latency_s=0.1)
    np.testing.assert_array_equal(np.load(first / "chunks/chunk_0000.npy", allow_pickle=False), actions)


def test_diagnostics_record_rotation_alarm_and_feedback_age(tmp_path, monkeypatch):
    from examples import policy_inference

    monkeypatch.setattr(policy_inference.time, "now", lambda: 10.0)
    cmd = np.array([0.11, 0.2, 0.3, 0.1, 0.0, 0.0, 0.7])
    eef = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    append_cmd_eef(tmp_path, 1.0, cmd, eef, 20, 0, arm_state={"alarm_code": 123, "timestamp": 9.9})
    with (tmp_path / "cmd_vs_eef.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["err_m"]) == pytest.approx(0.01)
    assert float(row["err_rot_rad"]) == pytest.approx(0.1)
    assert float(row["alarm_code"]) == 123
    assert float(row["state_age_s"]) == pytest.approx(0.1)
    assert float(row["cmd_gripper"]) == pytest.approx(0.7)


def test_joint_diagnostics_do_not_label_radians_as_metres(tmp_path):
    append_cmd_eef(tmp_path, 1.0, np.zeros(8), np.ones(7) * 0.01, 20, 0, joint_serve=True)
    with (tmp_path / "cmd_vs_eef.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert "err_m" not in row
    assert "cmd_j7" in row
    assert float(row["err_joint_rad"]) == pytest.approx(np.sqrt(7) * 0.01, abs=1e-6)
