# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
from loguru import logger
from rio_hw.middleware import ServerManager

from rio.envs.factory import make_policy
from rio.policies import __policies__
from rio.policies.policy_interface import PolicyInterfaceClient, PolicyInterfaceServer

# Determine which policies to test
POLICY_ENV = os.environ.get("RECONTROL_TEST_POLICY", None)
if POLICY_ENV is None:
    logger.info("RECONTROL_TEST_POLICY not set, testing all policies")
    POLICIES_TO_TEST = __policies__
else:
    logger.info(f"RECONTROL_TEST_POLICY set to {POLICY_ENV}")
    POLICIES_TO_TEST = [POLICY_ENV]


class _BlockingPolicy:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.resets = 0

    def set_instruction(self, instruction):
        self.instruction = instruction

    def construct_policy(self):
        pass

    def reset(self):
        self.resets += 1

    def inference(self, observation):
        value = int(observation["proprio"][0])
        if value == 1:
            self.started.set()
            if not self.release.wait(timeout=2):
                raise TimeoutError("Test did not release the inference")
        if value < 0:
            raise ValueError("Invalid test observation")
        return np.full((2, 7), value, dtype=np.float32)


def _worker_observation(value):
    proprio = np.zeros(8, dtype=np.float32)
    proprio[0] = value
    return {"proprio": proprio}


def _thread_policy_client(policy, *, with_camera=False):
    return PolicyInterfaceClient(
        "Thread",
        policy=policy,
        instruction="test",
        resolutions=[(4, 4)] if with_camera else [],
        action_dim=7,
        proprio_dim=8,
        chunk_size=2,
        freq=100,
    )


@pytest.mark.unit
def test_reset_discards_in_flight_policy_response(monkeypatch):
    policy = _BlockingPolicy()
    client = _thread_policy_client(policy)
    published = []
    ready = threading.Event()
    original_put = client.ring_buffer.put

    def put(data, **kwargs):
        original_put(data, **kwargs)
        if data.get("ready"):
            published.append(data["actions"].copy())
            ready.set()

    monkeypatch.setattr(client.ring_buffer, "put", put)
    try:
        with client:
            client.send_observation(_worker_observation(1))
            assert policy.started.wait(timeout=2)
            client.reset()
            client.send_observation(_worker_observation(2))
            policy.release.set()
            assert ready.wait(timeout=2)
            result = client.get_action_chunk()
            assert result["ready"]
            np.testing.assert_array_equal(result["actions"], np.full((2, 7), 2))
    finally:
        policy.release.set()
    assert len(published) == 1
    assert policy.resets >= 2


@pytest.mark.unit
def test_policy_failure_is_reported_without_losing_loaded_state(monkeypatch):
    client = _thread_policy_client(_BlockingPolicy())
    failed = threading.Event()
    original_put = client.ring_buffer.put

    def put(data, **kwargs):
        original_put(data, **kwargs)
        if data.get("inference_failed"):
            failed.set()

    monkeypatch.setattr(client.ring_buffer, "put", put)
    with client:
        client.send_observation(_worker_observation(-1))
        assert failed.wait(timeout=2)
        result = client.get_action_chunk()
        assert result["inference_failed"]
        assert not result["ready"]
        assert client.policy_loaded


@pytest.mark.unit
def test_policy_request_schema_preserves_rgb_bytes_and_default_camera_names():
    client = _thread_policy_client(_BlockingPolicy(), with_camera=True)
    assert client.camera_keys == ["camera_1"]
    assert client.example_request["camera_1"].dtype == np.uint8
    assert "_request_generation" in client.example_request


def _advance_request_generation(generation):
    with generation.get_lock():
        generation.value += 1


@pytest.mark.unit
def test_request_generation_is_shared_with_spawned_worker():
    client = _thread_policy_client(_BlockingPolicy())
    process = mp.get_context("spawn").Process(target=_advance_request_generation, args=(client._request_generation,))
    try:
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0
        assert client._request_generation.value == 1
        client.send_observation(_worker_observation(2))
        assert int(client.request_queue.get()["_request_generation"]) == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


@pytest.mark.unit
def test_policy_observation_transport_preserves_rtc_length_and_image_range():
    client = _thread_policy_client(_BlockingPolicy(), with_camera=True)
    observation = _worker_observation(2)
    observation.update(
        camera_1=np.full((4, 4, 3), 0.5, dtype=np.float32),
        proprio_joints=np.zeros(7),
        rtc_leftover=np.ones((3, 7), dtype=np.float32),
        rtc_inference_delay=2,
    )
    client.send_observation(observation)
    request = client.request_queue.get()
    assert set(request) == set(client.example_request)
    assert request["camera_1"].dtype == np.uint8
    np.testing.assert_array_equal(request["camera_1"], 128)
    assert request["rtc_leftover_length"] == 3
    assert request["rtc_leftover"].shape == (18, 7)
    np.testing.assert_array_equal(request["rtc_leftover"][:3], 1.0)
    np.testing.assert_array_equal(request["rtc_leftover"][3:], 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("value", [-0.1, 128.0, np.nan, np.inf])
def test_policy_transport_rejects_invalid_float_camera_pixels(value):
    client = _thread_policy_client(_BlockingPolicy(), with_camera=True)
    observation = _worker_observation(2)
    observation["camera_1"] = np.full((4, 4, 3), value)
    with pytest.raises(ValueError, match="RGB"):
        client.send_observation(observation)
    assert client.request_queue.empty()


@dataclass
class PolicyConfig:
    policy_path: str = "lerobot/smolvla_base"
    compile: bool = False
    device: str = "cuda:0"


@dataclass
class PolicyInterfaceConfig:
    instruction: str = "Move the robot arm to the left."
    resolutions: list[tuple[int, int]] | None = None
    action_dim: int = 6
    chunk_size: int = 50
    use_rtc: bool = False
    freq: int = 100
    max_buffer_size: int = 30

    def __post_init__(self):
        if self.resolutions is None:
            self.resolutions = [(256, 256), (256, 256), (256, 256)]


@dataclass
class TestConfig:
    mw: str = "Shm"
    mp_method: str | None = "spawn"  # "fork" or "spawn"
    freq: int = 30

    num_cams: int = 3
    cam_resolution: tuple = (256, 256, 3)
    action_dim: int = 6


@pytest.fixture
def policy_cfg():
    return PolicyConfig()


@pytest.fixture
def policy_interface_cfg():
    return PolicyInterfaceConfig()


@pytest.fixture
def test_cfg():
    return TestConfig()


@pytest.fixture
def make_dummy_observation(test_cfg):
    """Create dummy observation matching policy schema"""
    rng = np.random.default_rng()
    obs = {
        "proprio": rng.standard_normal(test_cfg.action_dim).astype(np.float32),
    }
    for i in range(test_cfg.num_cams):
        obs[f"camera{i + 1}"] = rng.standard_normal(test_cfg.cam_resolution).astype(np.float32)
    return obs


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("policy_name", POLICIES_TO_TEST)
def test_policy_interface(policy_name, policy_cfg, policy_interface_cfg, test_cfg, make_dummy_observation):
    # 1. Instantiate policy wrapper
    policy = make_policy(policy_name, vars(policy_cfg))

    # 2. Instantiate policy node (server and client factories)
    policy_interface_kwargs = {
        "policy": policy,
        "instruction": policy_interface_cfg.instruction,
        "resolutions": policy_interface_cfg.resolutions,
        "action_dim": policy_interface_cfg.action_dim,
        "chunk_size": policy_interface_cfg.chunk_size,
        "use_rtc": policy_interface_cfg.use_rtc,
        "freq": policy_interface_cfg.freq,
        "max_buffer_size": policy_interface_cfg.max_buffer_size,
    }

    server = lambda: PolicyInterfaceServer(test_cfg.mw, **policy_interface_kwargs)
    client = lambda: PolicyInterfaceClient(test_cfg.mw, **policy_interface_kwargs)

    # Start server and client
    with ServerManager(test_cfg.mw, [server]):
        with client() as policy_client:
            # Wait for policy to initialize
            time.sleep(2)

            # 3. Put dummy obs in the request queue
            obs = make_dummy_observation
            policy_client.send_observation(obs)

            # Give the node time to process
            time.sleep(0.5)

            # 4. Read action from action buffer
            t0 = time.time()
            action_data = policy_client.get_action_chunk()
            t1 = time.time()

            # Verify action data
            assert action_data is not None, "No action data received"
            assert "actions" in action_data, "Action data missing 'actions' key"
            assert "timestamp" in action_data, "Action data missing 'timestamp' key"
            assert "ready" in action_data, "Action data missing 'ready' key"

            if action_data["ready"]:
                assert action_data["actions"].shape == (
                    policy_interface_cfg.chunk_size,
                    policy_interface_cfg.action_dim,
                ), f"Unexpected action shape: {action_data['actions'].shape}"
                logger.info(
                    f"Policy: {policy_name} - Interface test successful - "
                    f"Action chunk shape: {action_data['actions'].shape}, "
                    f"Retrieval time: {t1 - t0:.4f} seconds"
                )
            else:
                logger.warning(f"Policy: {policy_name} - Action not ready, may need more processing time")
