# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kassow → LeRobot π₀.₅ conversion."""

import json
from pathlib import Path

import numpy as np
import pytest
import robodm

from rio.data.convert_pi05 import (
    ACTION_DIM,
    JOINT_ACTION_DIM,
    STATE_DIM,
    ConvertPi05Cfg,
    binarize_action_gripper,
    binarize_state_gripper,
    build_pi05_frame,
    convert,
    idle_start_index,
    joints_as_action,
    parse_task_map,
    prepare_episode,
    slice_episode,
    subsample_dict,
    subsample_step,
    traj_index,
    unwrap_rotvec_sequence,
    write_task_template,
)

rng = np.random.default_rng(0)


def test_traj_index_from_filename():
    assert traj_index(Path("traj_0007.vla")) == 7
    assert traj_index(Path("/data/traj_0042.vla")) == 42


def test_subsample_step_100_to_20():
    assert subsample_step(100, 20) == 5


def test_subsample_dict_keeps_alignment():
    n = 11
    data = {
        "action": np.arange(n, dtype=np.float32),
        "other": np.arange(n, dtype=np.float32) * 2,
    }
    out = subsample_dict(data, step=5)
    np.testing.assert_array_equal(out["action"], [0, 5, 10])
    np.testing.assert_array_equal(out["other"], [0, 10, 20])


def test_binarize_action_maps_one_to_open():
    values = np.array([0.0, 0.2, 0.7, 1.0], dtype=np.float32)
    snapped = binarize_action_gripper(values, gripper_close=0.0, gripper_open=0.7)
    np.testing.assert_allclose(snapped, [0.0, 0.0, 0.7, 0.7])


def test_binarize_action_uses_explicit_threshold():
    values = np.array([0.39, 0.40, 0.41], dtype=np.float32)
    snapped = binarize_action_gripper(values, gripper_close=0.0, gripper_open=0.7, gripper_threshold=0.4)
    np.testing.assert_allclose(snapped, [0.0, 0.7, 0.7])


def test_unwrap_rotvec_removes_2pi_jump():
    r0 = np.array([np.pi - 0.05, 0.1, 0.0], dtype=np.float32)
    r1 = np.array([-np.pi + 0.05, 0.1, 0.0], dtype=np.float32)
    out = unwrap_rotvec_sequence(np.stack([r0, r1]))
    assert np.linalg.norm(out[1] - out[0]) < 0.5
    assert np.linalg.norm(out[1] - out[0]) < np.linalg.norm(r1 - r0) / 2


def test_idle_start_skips_stationary_prefix():
    xyz = np.zeros((10, 3), dtype=np.float32)
    xyz[5:] = [0.05, 0.0, 0.0]
    assert idle_start_index(xyz, 0.01) == 5


def test_idle_start_keeps_all_if_never_moves():
    assert idle_start_index(np.zeros((8, 3), dtype=np.float32), 0.01) == 0


def test_prepare_episode_trims_and_unwraps():
    n = 12
    action = np.zeros((n, 7), dtype=np.float32)
    action[:, 3] = np.pi - 0.05
    action[6, 3] = -np.pi + 0.05
    action[6:, 0] = 0.05
    data = {
        "action": action,
        "observation/proprio_joints": np.zeros((n, 7), dtype=np.float32),
        "other": np.arange(n, dtype=np.float32),
    }
    cfg = ConvertPi05Cfg(input=Path("."), src_fps=20, dst_fps=20, min_frames=4, idle_xyz_m=0.01)
    prepared, info = prepare_episode(data, cfg)
    assert info["idle_start"] == 6
    assert len(prepared["action"]) == 6
    np.testing.assert_array_equal(prepared["other"], np.arange(6, 12))
    assert info["unwrap_jump_before"] > np.pi
    assert info["unwrap_jump_after"] < 0.5


def test_joints_as_action_uses_next_q():
    data = {
        "observation/proprio_joints": np.stack([np.full(7, float(i), dtype=np.float32) for i in range(4)]),
        "action": np.array([[0, 0, 0, 0, 0, 0, 0.7] for _ in range(4)], dtype=np.float32),
    }
    out = joints_as_action(data)
    assert out.shape == (4, JOINT_ACTION_DIM)
    np.testing.assert_allclose(out[0, :7], np.full(7, 1.0))
    np.testing.assert_allclose(out[-1, :7], np.full(7, 3.0))
    np.testing.assert_allclose(out[:, -1], 0.7)


def test_convert_refuses_joint_overwrite_of_tcp_folder(tmp_path: Path):
    vla_dir = tmp_path / "vla"
    vla_dir.mkdir()
    _write_sample_vla(vla_dir / "traj_0000.vla", n=8, fps=20)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("0: pick up the cube\n")
    with pytest.raises(ValueError, match="Refusing"):
        convert(
            ConvertPi05Cfg(
                input=vla_dir,
                output=tmp_path / "kassow_pi05",
                tasks=tasks,
                action_space="joints",
                src_fps=20,
                dst_fps=20,
                min_frames=4,
            )
        )


def test_slice_episode_keeps_length_one_instruction():
    data = {"action": np.zeros((5, 7), dtype=np.float32), "instruction": np.array(["do it"])}
    out = slice_episode(data, 2)
    assert len(out["action"]) == 3
    assert out["instruction"][0] == "do it"


def test_binarize_state_hysteresis_ignores_mid_noise():
    # Start open, dip into the deadband, then close.
    values = np.array([0.7, 0.51, 0.30, 0.277, 0.10, 0.0], dtype=np.float32)
    snapped = binarize_state_gripper(values, gripper_close=0.0, gripper_open=0.7)
    np.testing.assert_allclose(snapped, [0.7, 0.7, 0.7, 0.7, 0.0, 0.0])


def test_parse_task_map_ranges(tmp_path: Path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "0-2: pick up the blue cylinder",
                '3: "put it in the bin"',
                "4-5:",
            ]
        )
        + "\n"
    )
    mapping = parse_task_map(path)
    assert mapping[0] == mapping[2] == "pick up the blue cylinder"
    assert mapping[3] == "put it in the bin"
    assert 4 not in mapping
    assert 5 not in mapping


def test_parse_task_map_json(tmp_path: Path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"10-11": "open the drawer"}))
    mapping = parse_task_map(path)
    assert mapping == {10: "open the drawer", 11: "open the drawer"}


def test_convert_without_tasks_writes_template(tmp_path: Path):
    vla = tmp_path / "traj_0000.vla"
    vla.write_bytes(b"")
    with pytest.raises(ValueError, match="no language label"):
        convert(ConvertPi05Cfg(input=vla, output=tmp_path / "out"))
    assert (tmp_path / "tasks.yaml").exists()


def test_write_task_template(tmp_path: Path):
    files = [tmp_path / "traj_0000.vla", tmp_path / "traj_0002.vla"]
    for path in files:
        path.touch()
    out = write_task_template(files, tmp_path / "tasks.yaml")
    text = out.read_text()
    assert "0-2:" in text


def test_build_pi05_frame_schema_and_no_euler():
    rotvec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    action = np.concatenate([[1.0, 2.0, 3.0], rotvec, [1.0]]).astype(np.float32)
    data = {
        "observation/cameras/camera_1/rgb": rng.integers(0, 255, size=(1, 8, 12, 3), dtype=np.uint8),
        "observation/cameras/camera_2/rgb": rng.integers(0, 255, size=(1, 8, 12, 3), dtype=np.uint8),
        "observation/proprio_joints": np.arange(7, dtype=np.float32)[None, :],
        "observation/gripper_position": np.array([0.7], dtype=np.float32),
        "action": action[None, :],
    }
    frame = build_pi05_frame(data, 0, "pick up the cube")
    assert frame["task"] == "pick up the cube"
    assert frame["observation.state"].shape == (STATE_DIM,)
    assert frame["action"].shape == (ACTION_DIM,)
    np.testing.assert_allclose(frame["action"][:6], [1.0, 2.0, 3.0, 0.1, -0.2, 0.3])
    assert frame["action"][-1] == pytest.approx(0.7)
    np.testing.assert_allclose(frame["observation.state"][:7], np.arange(7, dtype=np.float32))
    assert frame["observation.images.camera_1"].shape == (8, 12, 3)
    assert frame["observation.images.camera_2"].dtype == np.uint8


def test_build_pi05_frame_keeps_raw_gripper_when_disabled():
    action = np.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3, 1.0], dtype=np.float32)
    data = {
        "observation/cameras/camera_1/rgb": rng.integers(0, 255, size=(1, 8, 12, 3), dtype=np.uint8),
        "observation/cameras/camera_2/rgb": rng.integers(0, 255, size=(1, 8, 12, 3), dtype=np.uint8),
        "observation/proprio_joints": np.arange(7, dtype=np.float32)[None, :],
        "observation/gripper_position": np.array([0.51], dtype=np.float32),
        "action": action[None, :],
    }
    frame = build_pi05_frame(data, 0, "pick up the cube", binarize_gripper=False)
    assert frame["action"][-1] == pytest.approx(1.0)
    assert frame["observation.state"][-1] == pytest.approx(0.51)


def _write_sample_vla(path: Path, n: int = 20, fps: int = 20) -> None:
    traj = robodm.Trajectory(path=str(path), mode="w", video_codec="rawvideo")
    dt_ms = 1000.0 / fps
    for i in range(n):
        gripper_cmd = 1.0 if i >= n // 2 else 0.0
        gripper_obs = 0.51 if i >= n // 2 else 0.3
        action = np.array([0.1, 0.2, 0.3, 0.01, -0.02, 0.03, gripper_cmd], dtype=np.float32)
        traj.add(
            "observation/cameras/camera_1/rgb",
            rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            timestamp=i * dt_ms,
            time_unit="ms",
        )
        traj.add(
            "observation/cameras/camera_2/rgb",
            rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            timestamp=i * dt_ms,
            time_unit="ms",
        )
        traj.add("observation/proprio_joints", np.zeros(7, dtype=np.float32), timestamp=i * dt_ms, time_unit="ms")
        traj.add("observation/gripper_position", np.float32(gripper_obs), timestamp=i * dt_ms, time_unit="ms")
        traj.add("action", action, timestamp=i * dt_ms, time_unit="ms")
    traj.close()


@pytest.mark.integration
def test_convert_writes_lerobot_v3(tmp_path: Path):
    pytest.importorskip("lerobot")

    vla_dir = tmp_path / "vla"
    vla_dir.mkdir()
    _write_sample_vla(vla_dir / "traj_0000.vla", n=20, fps=20)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("0: pick up the cube\n")
    output = tmp_path / "lerobot"
    summary = convert(
        ConvertPi05Cfg(
            input=vla_dir,
            output=output,
            tasks=tasks,
            src_fps=20,
            dst_fps=20,
            min_frames=4,
            repo_id="local/test_pi05",
            vcodec="h264",
            image_writer_threads=1,
        )
    )
    assert summary["kept"] == 1
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v3")
    assert info["fps"] == 20
    assert info["total_episodes"] == 1
    features = info["features"]
    assert features["observation.state"]["shape"] == [STATE_DIM]
    assert features["action"]["shape"] == [ACTION_DIM]
    assert "observation.images.camera_1" in features
    assert "observation.images.camera_2" in features


@pytest.mark.integration
def test_convert_joints_writes_8d_action(tmp_path: Path):
    pytest.importorskip("lerobot")

    vla_dir = tmp_path / "vla"
    vla_dir.mkdir()
    _write_sample_vla(vla_dir / "traj_0000.vla", n=20, fps=20)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("0: pick up the cube\n")
    output = tmp_path / "kassow_pi05_joints"
    summary = convert(
        ConvertPi05Cfg(
            input=vla_dir,
            output=output,
            tasks=tasks,
            action_space="joints",
            src_fps=20,
            dst_fps=20,
            min_frames=4,
            repo_id="local/test_pi05_joints",
            vcodec="h264",
            image_writer_threads=1,
        )
    )
    assert summary["kept"] == 1
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [JOINT_ACTION_DIM]
    assert info["features"]["action"]["names"][:7] == ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]
