# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing as mp
import os
import tempfile
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import robodm
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager
from tqdm import tqdm

from rio.data.loader import dict_to_step
from rio.data.recorder import Recorder
from rio.envs.factory import make_node
from rio.schema import Camera, Observation, Step

mp.set_start_method("spawn", force=True)

_INSTRUCTION = "Move the arm to the target position"


def create_dummy_step(timestep: int, rng: np.random.Generator | None = None) -> Step:
    if rng is None:
        rng = np.random.default_rng()

    # Create dummy camera data with random values
    rgb_image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    depth_image = rng.integers(0, 5, size=(480, 640), dtype=np.uint16)

    camera = Camera(rgb=rgb_image, depth=depth_image, meta={"camera_name": "wrist_camera", "timestep": timestep})

    # Create observation with random values
    observation = Observation(
        proprio=rng.standard_normal(7, dtype=np.float32),
        cameras={"wrist": camera},
    )

    # Create action with random values
    action = rng.standard_normal(7, dtype=np.float32)
    instruction = _INSTRUCTION
    # Create step
    step = Step(
        timestep=timestep,
        observation=observation,
        action=action,
        instruction=instruction,
        meta={"episode": 1, "step_in_episode": timestep},
    )

    return step


def compare_steps(loaded: Step, original: Step, timestep: int):
    """Compare two Step objects for equality."""
    # Verify timestep
    assert abs(loaded.timestep - original.timestep) < 1e-10, f"Timestep mismatch at index {timestep}"
    assert loaded.instruction == original.instruction, f"Instruction mismatch at index {timestep}"

    # Verify action
    np.testing.assert_array_equal(loaded.action.shape, original.action.shape)
    np.testing.assert_array_almost_equal(loaded.action, original.action, decimal=5)

    # Verify observation fields
    obs_loaded = loaded.observation
    obs_orig = original.observation

    # Check all proprio fields
    for field in ["proprio", "proprio_eef", "proprio_joints", "hand_pose", "hand_joints"]:
        val_loaded = getattr(obs_loaded, field, None)
        val_orig = getattr(obs_orig, field, None)
        if val_loaded is not None and val_orig is not None:
            np.testing.assert_array_equal(
                val_loaded.shape, val_orig.shape, err_msg=f"{field} shape mismatch at timestep {timestep}", verbose=False
            )
            np.testing.assert_array_almost_equal(
                val_loaded, val_orig, decimal=5, err_msg=f"{field} value mismatch at timestep {timestep}", verbose=False
            )

    # Check gripper position
    if obs_loaded.gripper_position is not None and obs_orig.gripper_position is not None:
        assert np.isclose(obs_loaded.gripper_position, obs_orig.gripper_position, rtol=1e-5)

    # Check cameras
    assert obs_loaded.cameras.keys() == obs_orig.cameras.keys()
    for cam_name in obs_orig.cameras:
        # breakpoint()
        cam_loaded = obs_loaded.cameras[cam_name]
        cam_orig = obs_orig.cameras[cam_name]

        # Display images side by side for visual inspection (optional)

        if cam_loaded.rgb is not None and cam_orig.rgb is not None:
            np.testing.assert_array_equal(
                cam_loaded.rgb.shape,
                cam_orig.rgb.shape,
                err_msg=f"RGB image shape mismatch at timestep {timestep}",
                verbose=False,
            )
            np.testing.assert_array_equal(
                cam_loaded.rgb, cam_orig.rgb, err_msg=f"RGB image value mismatch at timestep {timestep}", verbose=False
            )

        if cam_loaded.depth is not None and cam_orig.depth is not None:
            np.testing.assert_array_equal(
                cam_loaded.depth.shape,
                cam_orig.depth.shape,
                err_msg=f"Depth image shape mismatch at timestep {timestep}",
                verbose=False,
            )
            np.testing.assert_array_almost_equal(
                cam_loaded.depth,
                cam_orig.depth,
                decimal=5,
                err_msg=f"Depth image value mismatch at timestep {timestep}",
                verbose=False,
            )


@pytest.fixture
def temp_trajectory_path():
    """Create a temporary path for trajectory file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "test_trajectory.vla"
        yield str(trajectory_path)


@pytest.fixture
def test_trajectory_path():
    """Create a test/data in home directory for trajectory file."""
    home_dir = Path.home()
    data_dir = home_dir / "recontrol_test_data"
    data_dir.mkdir(exist_ok=True)
    trajectory_path = data_dir / "test_trajectory.vla"
    yield str(trajectory_path)


@pytest.fixture
def middleware():
    mw = os.environ.get("MIDDLEWARE", "Thread")
    yield mw


@pytest.mark.integration
@pytest.mark.parametrize(
    "freq,duration,description",
    [
        (10, 2, "Low frequency"),
        (100, 2, "Medium frequency"),
        (250, 2, "Medium frequency"),
        # (500, 2, "High frequency"),
    ],
)
def test_recorder_loader_roundtrip(temp_trajectory_path, middleware, freq, duration, description):
    """
    Test recorder/loader roundtrip with data integrity verification.

    Records data at various frequencies for different durations, then verifies:
    - Correct number of timesteps
    - All timestep values and sequence
    - All array shapes, sizes, and values match original
    """
    dt = 1.0 / freq
    rng = np.random.default_rng(seed=42)

    logger.info(f"{description}: Recording at {freq} Hz for {duration}s (expected ~{int(freq * duration)} steps)...")

    # Store original steps for comparison
    original_steps = []

    # Create recorder configuration
    # Adjust trajectory path to include test parameters
    test_path = f"{temp_trajectory_path.rsplit('.', 1)[0]}_{freq}hz_{duration}s.vla"
    logger.info(f"Trajectory path: {test_path}")

    recorder_cfg = {
        "path": test_path,
        "freq": freq,
        "max_queue_size": 10 * freq,
        "video_codec": "rawvideo",
        "rawvideo_codec": "rawvideo_pyarrow",
    }

    # Phase 1: Record dummy data at specified frequency
    recorder_server, recorder_client = make_node(middleware, "data", "Recorder", recorder_cfg, package="rio")

    with ServerManager(middleware, [recorder_server]):
        with recorder_client() as recorder:
            steps = [create_dummy_step(timestep=i, rng=rng) for i in range(int(freq * duration))]
            t_start = time.now()
            it = 0
            max_iterations = int(freq * duration)

            with tqdm(total=max_iterations, desc=f"{description} - Recording", unit="steps") as pbar:
                while it < max_iterations:
                    t_cycle_end = t_start + (it + 1) * dt

                    step = steps[it]
                    # Create and record step
                    original_steps.append(step)
                    recorder.record_step(step)

                    # Precise timing
                    time.precise_wait(t_cycle_end)
                    it += 1
                    pbar.update(1)

                    # Safety check: don't run forever if timing breaks
                    if time.now() - t_start > duration * 1.5:
                        logger.warning(f"Safety timeout reached at iteration {it}")
                        break

            num_steps = len(original_steps)
            actual_duration = time.now() - t_start
            actual_freq = num_steps / actual_duration

            recorder.save(wait=True, timeout=None)

    logger.info(f"Recorded {num_steps} steps in {actual_duration:.2f}s (target: {freq} Hz, actual: {actual_freq:.1f} Hz)")

    # Verify the file was created
    assert Path(test_path).exists(), "Trajectory file was not created"

    # Phase 2: Load and verify data
    _loaded = robodm.Trajectory(path=test_path, mode="r")
    loaded_data = _loaded.load()

    timesteps = loaded_data.get("timestep", [])
    assert len(timesteps) == num_steps, f"Expected {num_steps} timesteps, got {len(timesteps)}"

    for idx in range(num_steps):
        loaded_step_dict = {feature: loaded_data[feature][idx] for feature in loaded_data}
        loaded_step = dict_to_step(loaded_step_dict)
        original_step = original_steps[idx]

        compare_steps(loaded_step, original_step, idx)
    logger.info("All steps verified successfully!")

    # TODO: Implement loader verification logic


def _recorder_stub():
    published = []
    recorder = object.__new__(Recorder)
    recorder._close_lock = threading.Lock()
    recorder._close_queue = Queue()
    recorder._pending_closes = 0
    recorder._completed_save_id = 0
    recorder.error_code = 0
    recorder.is_closed = True
    recorder.is_saving = False
    recorder.freq = 100
    recorder.request_queue = Queue()
    recorder.ring_buffer = SimpleNamespace(
        put=lambda value, **kwargs: published.append(value),
        get=lambda: published[-1] if published else {},
    )
    return recorder, published


@pytest.mark.unit
def test_recorder_writes_schema_version_and_manifest_after_a_successful_save(tmp_path):
    from rio.schema import RECORDING_SCHEMA_VERSION

    recorder, published = _recorder_stub()
    recorder._manifest = {"rio": {"git_commit": "abc"}}
    recorder._pending_closes = 1
    recorder._close_queue.put((SimpleNamespace(close=lambda: None), str(tmp_path / "traj_0003.vla"), 7))
    recorder._close_queue.put(None)
    recorder._close_worker()
    meta = json.loads((tmp_path / "traj_0003_meta.json").read_text())
    assert meta["schema_version"] == RECORDING_SCHEMA_VERSION
    assert meta["trajectory"] == "traj_0003.vla"
    assert meta["manifest"] == {"rio": {"git_commit": "abc"}}
    assert not published[-1]["failed"] and published[-1]["save_id"] == 7


@pytest.mark.unit
def test_recorder_close_failure_is_latched_and_blocks_further_recording():
    recorder, published = _recorder_stub()

    def fail():
        raise OSError("disk failure")

    recorder._pending_closes = 1
    recorder.is_saving = True
    recorder._close_queue.put((SimpleNamespace(close=fail), "failed.vla", 123))
    recorder._close_queue.put(None)
    recorder._close_worker()
    assert published[-1]["failed"]
    assert published[-1]["error_code"] == 2
    assert published[-1]["save_id"] == 123
    assert not published[-1]["is_saving"]
    with pytest.raises(RuntimeError, match="Recorder failed"):
        recorder.record_dict({"action": [0.0]})
    with pytest.raises(RuntimeError, match="Recorder failed"):
        recorder.new_trajectory(wait=False)


@pytest.mark.unit
def test_recorder_frame_failure_is_not_silently_discarded():
    recorder, published = _recorder_stub()
    recorder.is_closed = False

    def fail(*args, **kwargs):
        raise OSError("recording failed")

    recorder.datamanager = SimpleNamespace(add_by_dict=fail)
    recorder._record({"action": [0.0]})
    assert published[-1]["failed"]
    assert published[-1]["error_code"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(("unit", "timestamp"), [("s", 0.125), ("ms", 125.0), ("us", 125000.0)])
def test_recorder_preserves_subsecond_timestamps_as_integer_nanoseconds(unit, timestamp):
    recorder, _ = _recorder_stub()
    calls = []
    recorder.is_closed = False
    recorder.time_unit = unit
    recorder.datamanager = SimpleNamespace(add_by_dict=lambda *args, **kwargs: calls.append(kwargs))
    recorder._record({"action": [0.0]}, timestamp)
    assert calls == [{"timestamp": 125000000, "time_unit": "ns"}]
    assert recorder.error_code == 0


@pytest.mark.unit
def test_recorder_save_waits_for_the_requested_save_id(monkeypatch):
    recorder, _ = _recorder_stub()
    monkeypatch.setattr("rio.data.recorder.time.now_ns", lambda: 123)
    monkeypatch.setattr("rio.data.recorder.time.Rate", lambda frequency: SimpleNamespace(sleep=lambda: None))
    reads = []

    def status():
        reads.append(True)
        return {"is_closed": True, "is_saving": False, "save_id": 0 if len(reads) == 1 else 123}

    recorder.ring_buffer.get = status
    recorder.save()
    assert len(reads) == 2
    assert recorder.request_queue.get()["save_id"] == 123


@pytest.mark.unit
def test_recorder_timeout_also_applies_when_no_status_arrives(monkeypatch):
    recorder, _ = _recorder_stub()
    clock = iter([0.0, 0.2])
    monkeypatch.setattr("rio.data.recorder.time.now", lambda: next(clock))
    monkeypatch.setattr("rio.data.recorder.time.Rate", lambda frequency: SimpleNamespace(sleep=lambda: None))
    recorder.ring_buffer.get = lambda: None
    with pytest.raises(TimeoutError):
        recorder.save(timeout=0.1)
