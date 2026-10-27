# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for π₀.₅ model-vs-control diagnostics (no GPU, no robot)."""

import numpy as np
import pytest

from examples.diagnose_pi05_offline import (
    episode_frac_index,
    episode_mid_index,
    episode_start_index,
    gripper_match,
    parse_sample_at,
    joint_ade_rad,
    joint_span_rad,
    rio_obs_from_frame,
    to_hwc_uint8,
    xyz_ade_m,
    xyz_span_m,
)
from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg
from examples.policy_inference import append_cmd_eef, write_chunk_dump

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
