# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

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
        self.camera_keys = (
            camera_keys if camera_keys is not None else [f"camera_{index + 1}" for index in range(len(resolutions))]
        )
        self.n_execute = n_execute
        self.chunk_pause_s = chunk_pause_s
        self.rtc_execution_horizon = rtc_execution_horizon
        if self.rtc_execution_horizon < 1:
            raise ValueError("rtc_execution_horizon must be positive")
        self._request_generation = mp.Value("q", 0)

        super().__init__(freq=freq, max_buffer_size=max_buffer_size, timeout=timeout)

    # NOTE: Defines request/pub schemas in post_init following other constructors
    def __post_init__(self):
        if self.camera_keys is None:
            logger.warning("camera_keys is None, defaulting to camera_1, camera_2, ...")
            self.camera_keys = [f"camera_{i + 1}" for i in range(len(self.resolutions))]
        if len(self.camera_keys) != len(self.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")

        obs_schema = {
            "proprio": np.zeros(shape=(self.proprio_dim,), dtype=np.float32),
            "_request_generation": 0,
            "rtc_leftover": np.zeros((self.rtc_execution_horizon, self.action_dim), dtype=np.float32),
            "rtc_leftover_length": 0,
            "rtc_inference_delay": 0,
        }
        for i, resolution in enumerate(self.resolutions):
            obs_schema[self.camera_keys[i]] = np.zeros(shape=(*resolution, 3), dtype=np.uint8)

        action_chunk = np.zeros(shape=(self.chunk_size, self.action_dim), dtype=np.float32)
        action_schema = {"actions": action_chunk, "original_actions": action_chunk.copy()}

        self.example_request = {**obs_schema}
        self.example_data = {
            **action_schema,
            "timestamp": time.now(),
            "ready": False,
            "policy_loaded": False,
            "inference_failed": False,
            "generation": 0,
        }

        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def _empty_result(self, *, loaded: bool = True, failed: bool = False) -> dict:
        empty = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        return {
            "actions": empty,
            "original_actions": empty.copy(),
            "timestamp": time.now(),
            "ready": False,
            "policy_loaded": loaded,
            "inference_failed": failed,
            "generation": int(self._request_generation.value),
        }

    def pubreq(self):
        """
        Handles policy inference requests and pushes action chunks onto the ring buffer
        """
        threadpool_limits(1)

        # initialize model on device
        self.policy.set_instruction(self.instruction)
        self.policy.construct_policy()
        logger.info("PolicyInterface: Policy constructed.")
        self.ring_buffer.put(self._empty_result(), wait=False)
        logger.debug("PolicyInterface: Policy constructed and instruction set.")
        try:
            # Main loop
            rate = time.Rate(self.freq)
            if self.req_ready_event is not None:
                self.req_ready_event.set()
            self.pub_ready_event.set()
            previous_generation = -1
            while not self.exit_event.is_set():
                with self._request_generation.get_lock():
                    obs = None if self.request_queue.empty() else self.request_queue.get()
                    generation = int(self._request_generation.value)
                if obs is not None:
                    request_generation = int(obs.pop("_request_generation", -1))
                    if request_generation != generation:
                        continue
                    receive_time = time.now()
                    try:
                        if previous_generation != generation:
                            reset = getattr(self.policy, "reset", None)
                            if callable(reset):
                                reset()
                            previous_generation = generation
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
                            "inference_failed": False,
                            "generation": generation,
                        }
                        with self._request_generation.get_lock():
                            if generation == self._request_generation.value and not self.exit_event.is_set():
                                self.ring_buffer.put(data, wait=False)
                    except Exception:
                        logger.exception("Policy inference failed")
                        with self._request_generation.get_lock():
                            if generation == self._request_generation.value:
                                self.ring_buffer.put(self._empty_result(failed=True), wait=False)

                rate.precise_sleep()
        except KeyboardInterrupt:
            pass

    def send_observation(self, raw_obs):
        proprio = np.asarray(raw_obs.get("proprio"), dtype=np.float32)
        if proprio.shape != (self.proprio_dim,) or not np.isfinite(proprio).all():
            raise ValueError(f"Expected a finite {self.proprio_dim}D proprio vector")
        request = {"proprio": proprio, "_request_generation": int(self._request_generation.value)}
        for key, resolution in zip(self.camera_keys, self.resolutions, strict=True):
            image = np.asarray(raw_obs.get(key))
            if image.shape != (*resolution, 3):
                raise ValueError(f"Missing or incorrectly shaped RGB image {key}")
            if image.dtype != np.uint8:
                if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
                    raise ValueError(f"Invalid RGB pixels in {key}")
                if np.min(image) < 0.0 or np.max(image) > 1.0:
                    raise ValueError(f"Floating-point RGB pixels in {key} must be normalized to [0, 1]")
                image = np.rint(image * 255.0).astype(np.uint8)
            request[key] = image
        leftover = np.asarray(raw_obs.get("rtc_leftover", []), dtype=np.float32)
        prefix = np.zeros((self.rtc_execution_horizon, self.action_dim), dtype=np.float32)
        length = 0
        if leftover.size:
            if leftover.ndim != 2 or leftover.shape[1] != self.action_dim or not np.isfinite(leftover).all():
                raise ValueError("Invalid RTC leftover action matrix")
            length = min(len(leftover), self.rtc_execution_horizon)
            prefix[:length] = leftover[:length]
        delay = int(np.asarray(raw_obs.get("rtc_inference_delay", 0)).reshape(-1)[0])
        if delay < 0:
            raise ValueError("RTC inference delay must not be negative")
        request.update(rtc_leftover=prefix, rtc_leftover_length=length, rtc_inference_delay=delay)
        self.request_queue.put(request)

    def get_action_chunk(self):
        """
        Retrieves an action chunk from the ring buffer if available.
        Otherwise, sends an empty array.
        """
        data = self.ring_buffer.get()
        # handle stale action chunks
        # TODO: is there a better way to mark stale action chunks?

        if (
            not isinstance(data, dict)
            or data.get("timestamp") is None
            or data["timestamp"] <= self.last_timestamp
            or int(data.get("generation", -1)) != self._request_generation.value
        ):
            return self._empty_result(loaded=bool(isinstance(data, dict) and data.get("policy_loaded")))
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
        """Invalidate queued and in-flight work without racing the model's inference thread."""
        with self._request_generation.get_lock():
            self._request_generation.value += 1
            while not self.request_queue.empty():
                self.request_queue.get()
            self.last_timestamp = time.now()
            self.ring_buffer.clear()
            self.ring_buffer.put(self._empty_result(), wait=False)
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
