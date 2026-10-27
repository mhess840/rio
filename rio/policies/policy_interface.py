# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node
from threadpoolctl import threadpool_limits

from ._policy import Policy


class PolicyInterface(Node):
    __api__ = [
        "send_observation",
        "get_action_chunk",
        "reset",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        policy: Policy,
        instruction: str,
        resolutions: list[tuple[int, int]],
        action_dim: int,
        proprio_dim: int,
        chunk_size: int,
        use_rtc: bool = False,
        freq: int = 100,
        action_hz: int = 20,
        infer_lead_s: float = 1.0,
        rtc_queue_threshold: int = 30,
        max_buffer_size: int = 30,
        chunk_request_threshold: float = 0.75,
        camera_keys: list[str] | None = None,
        timeout: float = 180.0,
        n_execute: int | None = None,
        chunk_pause_s: float = 0.0,
        rtc_execution_horizon: int = 18,
    ):
        self.policy = policy
        self.instruction = instruction
        self.resolutions = resolutions
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.use_rtc = use_rtc
        self.freq = freq
        self.action_hz = action_hz
        self.infer_lead_s = infer_lead_s
        self.rtc_queue_threshold = rtc_queue_threshold
        self.chunk_size = chunk_size
        self.max_buffer_size = max_buffer_size
        self.last_timestamp = time.now()
        self.chunk_request_threshold = chunk_request_threshold
        self.camera_keys = camera_keys
        self.n_execute = n_execute
        self.chunk_pause_s = chunk_pause_s
        self.rtc_execution_horizon = rtc_execution_horizon

        super().__init__(freq=freq, max_buffer_size=max_buffer_size, timeout=timeout)

    # NOTE: Defines request/pub schemas in post_init following other constructors
    def __post_init__(self):
        if len(self.camera_keys) != len(self.resolutions):
            logger.error("Length of camera_keys must match length of resolutions")
            raise ValueError("Length of camera_keys must match length of resolutions")
        if self.camera_keys is None:
            logger.warning("camera_keys is None, defaulting to camera_1, camera_2, ...")
            self.camera_keys = [f"camera_{i + 1}" for i in range(len(self.resolutions))]

        obs_schema = {"proprio": np.zeros(shape=(self.proprio_dim,), dtype=np.float32)}
        for i, resolution in enumerate(self.resolutions):
            obs_schema[self.camera_keys[i]] = np.zeros(shape=(*resolution, 3), dtype=np.float32)

        action_chunk = np.zeros(shape=(self.chunk_size, self.action_dim), dtype=np.float32)
        action_schema = {"actions": action_chunk, "original_actions": action_chunk.copy()}

        self.example_request = {**obs_schema}
        self.example_data = {**action_schema, "timestamp": time.now(), "ready": False, "policy_loaded": False}

        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def pubreq(self):
        """
        Handles policy inference requests and pushes action chunks onto the ring buffer
        """
        threadpool_limits(1)

        # initialize model on device
        self.policy.set_instruction(self.instruction)
        self.policy.construct_policy()
        logger.info("PolicyInterface: Policy constructed.")
        empty_chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        self.ring_buffer.put(
            {
                "actions": empty_chunk,
                "original_actions": empty_chunk.copy(),
                "timestamp": time.now(),
                "ready": False,
                "policy_loaded": True,
            },
            wait=False,
        )
        logger.debug("PolicyInterface: Policy constructed and instruction set.")
        try:
            # Main loop
            rate = time.Rate(self.freq)
            if self.req_ready_event is not None:
                self.req_ready_event.set()
            self.pub_ready_event.set()
            while not self.exit_event.is_set():
                # Pop from request queue
                if not self.request_queue.empty():
                    obs = self.request_queue.get()
                    receive_time = time.now()
                    result = self.policy.inference(obs)
                    if isinstance(result, dict):
                        action_chunk = result["actions"]
                        original_chunk = result.get("original_actions", action_chunk)
                    else:
                        action_chunk = result
                        original_chunk = action_chunk
                    data = {
                        "actions": np.asarray(action_chunk, dtype=np.float32),
                        "original_actions": np.asarray(original_chunk, dtype=np.float32),
                        "timestamp": receive_time,
                        "ready": True,
                        "policy_loaded": True,
                    }
                    self.ring_buffer.put(data, wait=False)

                rate.precise_sleep()
        except KeyboardInterrupt:
            pass

    def send_observation(self, raw_obs):
        self.request_queue.put(raw_obs)

    def get_action_chunk(self):
        """
        Retrieves an action chunk from the ring buffer if available.
        Otherwise, sends an empty array.
        """
        data = self.ring_buffer.get()
        # handle stale action chunks
        # TODO: is there a better way to mark stale action chunks?

        if not isinstance(data, dict) or data.get("timestamp") is None or data["timestamp"] <= self.last_timestamp:
            empty_chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            empty_data = {
                "actions": empty_chunk,
                "original_actions": empty_chunk.copy(),
                "timestamp": time.now(),
                "ready": False,
                "policy_loaded": False,
            }
            return empty_data
        else:
            self.last_timestamp = data["timestamp"]
            return data

    def get_action_chunk_blocking(self, obs):
        result = self.policy.inference(obs)
        if isinstance(result, dict):
            return {
                "actions": result["actions"],
                "original_actions": result.get("original_actions", result["actions"]),
            }
        return {"actions": result, "original_actions": result}

    def reset(self):
        # TODO: addpt to use the middleware reset method
        """
        Resets the policy interface state, clearing buffers and resetting timestamps.
        """
        # Clear the request queue
        while not self.request_queue.empty():
            try:
                self.request_queue.clear()
            except Exception:
                break

        # Reset the policy if it has a reset method
        if hasattr(self.policy, "reset") and callable(self.policy.reset):
            self.policy.reset()
            logger.info("PolicyInterface: Policy reset.")

        # Clear ring buffer and restore the loaded marker so serve does not
        # think the 9 GB weights vanished.
        self.ring_buffer.clear()
        empty_chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        self.ring_buffer.put(
            {
                "actions": empty_chunk,
                "original_actions": empty_chunk.copy(),
                "timestamp": time.now(),
                "ready": False,
                "policy_loaded": True,
            },
            wait=False,
        )

        logger.debug("PolicyInterface: Reset complete.")

    @property
    def policy_loaded(self):
        data = self.ring_buffer.get()
        if not isinstance(data, dict):
            return False
        policy_loaded = data.get("policy_loaded", False)
        return policy_loaded


def PolicyInterfaceServer(mw, *args, **kwargs):
    return ServerFactory(mw, PolicyInterface, *args, **kwargs)


def PolicyInterfaceClient(mw, *args, **kwargs):
    return ClientFactory(mw, PolicyInterface, *args, **kwargs)
