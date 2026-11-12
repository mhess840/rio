# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import math
import os
import threading
from queue import Queue
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node

from ..manifest import run_manifest
from ..schema import RECORDING_SCHEMA_VERSION, Step

try:
    import robodm
except ImportError:
    robodm = None
    logger.warning("robodm is not installed. Recorder node will not function.")

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil is not installed. System stats will not be collected.")

try:
    import pynvml

    pynvml.nvmlInit()
    _gpu_available = True
except Exception:
    if TYPE_CHECKING:
        raise
    pynvml = None
    _gpu_available = False


def step_to_dict(step: Step) -> dict[str, Any]:
    """
    Convert a Step object into a flat dictionary suitable for recording.

    """
    state = {}
    if hasattr(step, "__dataclass_fields__"):
        items = step.__dict__.items()
    elif isinstance(step, dict):
        items = step.items()

    for k, v in items:
        # if is dataclass, flatten
        if hasattr(v, "__dataclass_fields__") or isinstance(v, dict):
            items = step_to_dict(v).items()
            for k2, v2 in items:
                if v2 is not None:
                    state[f"{k}/{k2}"] = v2
        else:
            if v is not None:
                state[k] = v

    return state


class Recorder(Node):
    __api__ = [
        "record_state",
        "record_dict",
        "record_step",
        "save",
        "get_state",
        "new_trajectory",
        "ready",
        "set_reward",
        "set_final_reward",
    ]
    __pub__ = True  # Publish recording status
    __req__ = True

    def __init__(
        self,
        path: str,
        video_codec: str = "libx264",
        codec_options: dict[str, Any] | None = None,
        raw_codec: str = "rawvideo_pyarrow",
        start_recording: bool = True,
        log_stats: bool = False,
        system_stats_freq: int = 1,
        *,
        freq: int = 100,
        max_buffer_size: int = 100,
        max_queue_size: int | None = None,  # Auto set from freq if None
        **kwargs,
    ):
        """
        Initialize the Recorder node.
        """
        self.base_path = path
        self.video_codec = video_codec
        self.codec_options = codec_options or {}
        self.raw_codec = raw_codec
        self.datamanager = None
        self.traj_prefix = None
        self.dataset_path = None
        self.time_unit = kwargs.get("time_unit", "s")
        self.start_recording = start_recording
        self.log_stats = log_stats
        self.system_stats_freq = system_stats_freq
        self.last_stats_time = 0.0

        if max_queue_size is None:
            max_queue_size = freq * 10  # Default to 10 seconds of queue

        self.path = None
        # Closed until `_new_trajectory` opens the first file (including
        # `start_recording=True` during `__post_init__`).
        self.is_closed = True
        self.is_saving = False
        self._close_queue: Queue = Queue()
        self._close_thread: threading.Thread | None = None
        self._pending_closes = 0
        self._close_lock = threading.Lock()
        self.error_code = 0
        self._completed_save_id = 0
        self._manifest: dict | None = None

        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)
        logger.info(f"Recorder initialized with path: {self.base_path} and freq: {self.freq} Hz")

        if not self.verbose:
            logger.disable("rio.data.recorder")

    def __post_init__(self):
        """Initialize the robodm Trajectory and set up the node."""
        # Initialize robodm Trajectory in write mode
        # Try to use raw_codec if supported, otherwise fall back to basic initialization
        self._init_dataset()

        self.example_request = {
            "type": "record",
            "state": {},
            "timestamp": None,
        }
        self.example_data = {
            "is_closed": False,
            "is_saving": False,
            "failed": False,
            "error_code": 0,
            "save_id": 0,
            "timestamp": 0.0,
        }

        self.worker = None
        self.run = self.pubreq

        super().__post_init__()
        self._configure_logger()

    def _create_datamanager(self):
        """Create the robodm Trajectory datamanager."""
        if robodm is None:
            raise ImportError("robodm is required for recording")
        if self.path is None:
            raise RuntimeError("Recorder trajectory path is not initialized")
        options: dict[str, Any] = {
            "path": self.path,
            "mode": "w",
            "video_codec": self.video_codec,
            "codec_options": self.codec_options,
        }
        if "raw_codec" in inspect.signature(robodm.Trajectory).parameters:
            options["raw_codec"] = self.raw_codec
        self.datamanager = robodm.Trajectory(**options)

    def _init_stats(self):
        self.stats: dict[str, Any] = {
            "file_name": self.path,
            "start_time": time.now(),
            "steps_recorded": 0,
            "num_steps": 0,
            "cpu_usage": [],
            "ram_mem": [],
            "gpus_usage": [],
            "gpus_mem": [],
            "gpus_mem_percentage": [],
            "rewards": [],
            "final_reward": None,
        }

    def _end_stats(self):
        self.stats["end_time"] = time.now()
        self.stats["total_time"] = self.stats["end_time"] - self.stats["start_time"]
        self.stats["avg_cpu_usage"] = np.mean(self.stats["cpu_usage"]) if self.stats["cpu_usage"] else 0
        self.stats["avg_ram_mem"] = np.mean(self.stats["ram_mem"]) if self.stats["ram_mem"] else 0
        self.stats["avg_gpus_usage"] = np.mean(self.stats["gpus_usage"]) if self.stats["gpus_usage"] else 0
        self.stats["avg_gpus_mem"] = np.mean(self.stats["gpus_mem"]) if self.stats["gpus_mem"] else 0
        self.stats["avg_gpus_mem_percentage"] = (
            np.mean(self.stats["gpus_mem_percentage"]) if self.stats["gpus_mem_percentage"] else 0
        )
        self.stats["total_reward"] = sum(self.stats["rewards"]) if self.stats["rewards"] else 0
        self.stats["avg_reward"] = np.mean(self.stats["rewards"]) if self.stats["rewards"] else 0

        logger.info(f"Recording stats: {self.stats}")
        self._save_stats()

    def _save_stats(self):
        if self.path is None:
            logger.error("Cannot save stats, trajectory path is None")
            return

        stats_path = self.path[:-4] + "_stats.json"
        if os.path.exists(stats_path):
            logger.warning(f"Overwriting existing stats file at {stats_path}")
            os.remove(stats_path)
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=4)

    def _capture_system_stats(self):
        """Capture current system stats (CPU, RAM, GPU usage/memory)."""
        if psutil is None:
            return

        # Capture CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent

        self.stats["cpu_usage"].append(cpu_percent)
        self.stats["ram_mem"].append(ram_percent)

        # Capture GPU stats if available
        if _gpu_available and pynvml is not None:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                all_gpus_util = []
                all_gpus_mem = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    # Get GPU utilization (compute load in %)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    # Get memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    all_gpus_util.append(utilization.gpu)
                    all_gpus_mem.append(mem_info)

                self.stats["gpus_usage"].append(all_gpus_util)
                self.stats["gpus_mem"].append([mem.used for mem in all_gpus_mem])
                self.stats["gpus_mem_percentage"].append([mem.used / mem.total * 100 for mem in all_gpus_mem])

            except Exception as e:
                logger.debug(f"Error capturing GPU stats: {e}")

    def _init_dataset(self) -> None:
        """
        parse and validate the trajectory path and define the trajectory index.
        """
        self.traj_index = -1
        # check if base_path already exists and if it's a directory, find the next available traj_index
        if os.path.exists(self.base_path):
            logger.info(f"Adding new trajectories to existing dataset at {self.base_path}")

            # find the last traj_index used assuming a format anyname_####.vla
            if os.path.isdir(self.base_path):
                self.dataset_path = self.base_path
                self.traj_prefix = "traj_"

            elif self.base_path.endswith(".vla"):
                # if it's a file, check if it ends with .vla and increment index accordingly
                base_no_ext = self.base_path[:-4]
                self.traj_prefix = os.path.basename(base_no_ext)
                self.dataset_path = os.path.dirname(self.base_path)
            else:
                os.makedirs(self.base_path, exist_ok=True)
                self.dataset_path = self.base_path
                self.traj_prefix = "traj_"
                # raise ValueError(f"Invalid path: {self.base_path}. Must be a directory or a .vla file.")

            existing_trajs = [fname for fname in os.listdir(self.dataset_path) if fname.endswith(".vla")]
            existing_indices = [int(fname[-8:-4]) for fname in existing_trajs if fname[-8:-4].isdigit()]
            if existing_indices:
                self.traj_index = max(existing_indices)
        else:
            if self.base_path.endswith(".vla"):
                self.dataset_path = os.path.dirname(self.base_path)
                self.traj_prefix = os.path.basename(self.base_path)[:-4]
            else:
                os.makedirs(self.base_path, exist_ok=True)
                self.dataset_path = self.base_path
                self.traj_prefix = "traj_"

        if self.start_recording:
            logger.info("Starting first trajectory immediately upon initialization.")
            self._new_trajectory(self.dataset_path)
        else:
            logger.warning("Recorder initialized with start_recording=False. Call new_trajectory() to start recording.")
            self.is_closed = True

    def _configure_logger(self):
        """
        Override this method in subclasses to set up custom logging configuration.
        """
        pass

    def pubreq(self):
        """Combined publish and request processing loop."""
        try:
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            self.pub_ready_event.set()

            logger.info("Recorder started, listening for requests...")
            self._close_thread = threading.Thread(target=self._close_worker, name="recorder-encode", daemon=True)
            self._close_thread.start()

            while not self.exit_event.is_set():
                while not self.request_queue.empty():
                    req = self.request_queue.get()
                    if req is None:
                        break

                    req_type = req.get("type")
                    if req_type == "record":
                        self._record(req.get("state", {}), req.get("timestamp"))
                    elif req_type == "save":
                        self._save(save_id=int(req.get("save_id", 0)))
                        if req.get("start_next") and self.is_closed:
                            self._new_trajectory(req.get("path"))
                    elif req_type == "set_reward":
                        if self.log_stats:
                            self.stats["rewards"].append(req.get("reward", 0.0))
                        else:
                            logger.debug("log_stats=False, ignoring set_reward")
                    elif req_type == "set_final_reward":
                        if self.log_stats:
                            self.stats["final_reward"] = req.get("reward", 0.0)
                            self._save_stats()
                        else:
                            logger.debug("log_stats=False, ignoring set_final_reward")
                    elif req_type == "new_trajectory":
                        if self.is_closed:
                            self._new_trajectory(req.get("path"))
                        else:
                            logger.warning("Cannot start new trajectory before saving the current one.")
                    else:
                        raise ValueError(f"Unknown request type: {req_type}")

                # Capture system stats periodically
                if self.log_stats:
                    current_time = time.now()
                    if not self.is_closed and (current_time - self.last_stats_time) >= (1.0 / self.system_stats_freq):
                        self._capture_system_stats()
                        self.last_stats_time = current_time

                self._put_status()

                rate.precise_sleep()

        except KeyboardInterrupt:
            pass
        finally:
            if not self.is_closed:
                logger.info("Recorder exiting, saving trajectory...")
                self._save()
            self._close_queue.put(None)
            if self._close_thread is not None:
                self._close_thread.join()

    def _put_status(self):
        with self._close_lock:
            status = {
                "is_closed": self.is_closed,
                "is_saving": self.is_saving,
                "failed": self.error_code != 0,
                "error_code": self.error_code,
                "save_id": self._completed_save_id,
                "timestamp": time.now(),
            }
        self.ring_buffer.put(status, wait=False)

    def _fail(self, code: int) -> None:
        with self._close_lock:
            self.error_code = self.error_code or code
        self._put_status()

    @staticmethod
    def _require_healthy(status) -> None:
        if not isinstance(status, dict):
            raise RuntimeError("Recorder status is unavailable")
        if status.get("failed", False):
            raise RuntimeError(f"Recorder failed (error code {status.get('error_code', 0)}); inspect the recorder log")

    def get_state(self):
        """Get recorder status."""
        return self.ring_buffer.get()

    def _record(self, state: dict[str, Any], timestamp: float | None = None):
        """Internal method to record a state dictionary synchronously."""
        if self.is_closed or self.datamanager is None:
            return

        try:
            if timestamp is None:
                self.datamanager.add_by_dict(state, timestamp=None, time_unit=self.time_unit)
            else:
                scales = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
                if self.time_unit not in scales or not math.isfinite(timestamp):
                    raise ValueError("Recorder timestamp or time unit is invalid")
                timestamp_ns = round(timestamp * scales[self.time_unit])
                self.datamanager.add_by_dict(state, timestamp=timestamp_ns, time_unit="ns")
        except Exception as e:
            logger.error(f"Error recording dict: {e}")
            self._fail(1)

    def _close_worker(self):
        """Encode previously recorded trajectories without blocking teleop."""
        while True:
            item = self._close_queue.get()
            if item is None:
                return
            datamanager, path, save_id = item
            try:
                if datamanager is not None:
                    logger.warning("Recorder saving {} in the background...", path)
                    datamanager.close()
                    self._write_meta(path)
                    logger.info("Recorder saved trajectory to {}", path)
            except Exception as e:
                logger.error("Error saving recorder {}: {}", path, e)
                self._fail(2)
            finally:
                with self._close_lock:
                    self._pending_closes = max(0, self._pending_closes - 1)
                    self.is_saving = self._pending_closes > 0
                    self._completed_save_id = max(self._completed_save_id, save_id)
                self._put_status()

    def _write_meta(self, path: str) -> None:
        """``<traj>_meta.json``: recording schema version and the code/library versions that wrote it."""
        if self._manifest is None:
            self._manifest = run_manifest()
        meta = {"schema_version": RECORDING_SCHEMA_VERSION, "trajectory": os.path.basename(path), "manifest": self._manifest}
        with open(path[:-4] + "_meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, default=str)

    def _save(self, save_id: int = 0):
        """Hand the current trajectory to the encode worker and mark it closed."""
        if self.is_closed:
            with self._close_lock:
                self._pending_closes += 1
                self.is_saving = True
            self._close_queue.put((None, None, save_id))
            self._put_status()
            return

        if self.log_stats:
            try:
                self._end_stats()
            except Exception:
                logger.exception("Could not save recorder statistics")
                self._fail(3)

        datamanager = self.datamanager
        path = getattr(self, "path", None)
        self.datamanager = None
        self.is_closed = True
        if datamanager is None or path is None:
            self._put_status()
            return

        with self._close_lock:
            self._pending_closes += 1
            self.is_saving = True
        self._put_status()
        self._close_queue.put((datamanager, path, save_id))

    def _new_trajectory(self, path: str | None = None):
        """
        Internal method to start a new trajectory in the same recorder instance.
        """
        if self.error_code:
            self._put_status()
            return
        if not self.is_closed:
            logger.warning("Cannot start new trajectory before saving the current one.")
            return

        if self.dataset_path is None or self.traj_prefix is None:
            raise ValueError("Dataset path is not set. Make sure the dataset was initialized correctly.")

        self.traj_index += 1

        if path is None:
            path = self.base_path
        if os.path.isdir(self.base_path):
            traj_filename = f"{self.traj_prefix}{self.traj_index:04d}.vla"
            self.path = os.path.join(self.dataset_path, traj_filename)
        else:
            self.path = self.base_path

        logger.info("Starting new trajectory...")
        self._create_datamanager()
        if self.log_stats:
            self._init_stats()
        logger.info(f"Started new trajectory at {self.path}")

        self.is_closed = False

    # Client API methods (called via middleware)
    def record_step(self, step: Step):
        """
        Record a Step object.

        Args:
            step: Step object containing observation, action, and metadata
        """
        state = step_to_dict(step)

        # Record the flattened state with timestamp
        self.record_dict(state, timestamp=step.timestep)

    def record_state(self, state: dict[str, Any], timestamp: float | None = None):
        """
        Record state with optional timestamp.

        Args:
            state: Dictionary containing feature names and their values
            timestamp: Optional timestamp in milliseconds. If None, uses current time.
        """
        self._require_healthy(self.ring_buffer.get())
        self.request_queue.put({"type": "record", "state": state, "timestamp": timestamp})

    def record_dict(self, state: dict[str, Any], timestamp: float | None = None):
        """
        Record a nested state dictionary (supports hierarchical feature names).

        Args:
            state: Dictionary containing feature names and their values.
                   Can be nested (e.g., {"camera": {"rgb": image, "depth": depth_map}})
            timestamp: Optional timestamp in milliseconds.
        """
        self._require_healthy(self.ring_buffer.get())
        self.request_queue.put({"type": "record", "state": state, "timestamp": timestamp})

    def save(self, wait: bool = True, timeout: float | None = 120.0, start_next: bool = False):
        """
        Save the trajectory file and close the recorder.

        Encoding runs in a background thread. If ``start_next`` is True, a new
        trajectory is opened immediately so teleop can continue.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0.0):
            raise ValueError("Recorder save timeout must be finite and non-negative")
        req = {
            "type": "save",
            "start_next": start_next,
            "save_id": time.now_ns(),
        }
        self.request_queue.put(req)

        if wait:
            start_time = time.now()
            rate = time.Rate(self.freq)

            while True:
                if timeout is not None and time.now() - start_time > timeout:
                    raise TimeoutError("Recorder did not confirm the requested save before the timeout")
                status = self.ring_buffer.get()
                if not isinstance(status, dict):
                    rate.sleep()
                    continue
                self._require_healthy(status)
                completed = int(status.get("save_id", 0)) >= req["save_id"]
                correct_state = not status.get("is_closed", True) if start_next else status.get("is_closed", False)
                if completed and correct_state and not status.get("is_saving", False):
                    break
                rate.sleep()

            logger.info("Recorder finished saving trajectory.")

    def new_trajectory(self, path: str | None = None, wait: bool = True, timeout: float | None = 600):
        """
        Start a new trajectory in the same recorder instance.
        """
        status = self.ring_buffer.get()
        self._require_healthy(status)
        if not status.get("is_closed", False):
            raise RuntimeError("Cannot start a new trajectory before saving the current one.")

        req = {
            "type": "new_trajectory",
            "path": path,
        }
        self.request_queue.put(req)
        logger.info(f"Started new trajectory at {self.path}")

        if wait:
            start_time = time.now()
            rate = time.Rate(self.freq)
            while self.ring_buffer.get().get("is_closed", False):
                self._require_healthy(self.ring_buffer.get())
                if timeout is not None:
                    if time.now() - start_time > timeout:
                        raise TimeoutError("Recorder did not start a new trajectory before the timeout")
                rate.sleep()

    def ready(self):
        status = self.ring_buffer.get()
        return isinstance(status, dict) and not status.get("failed", False) and not status.get("is_closed", True)

    def set_reward(self, reward: float):
        """
        Add reward for the current step (if applicable).
        """
        self.request_queue.put(
            {
                "type": "set_reward",
                "reward": reward,
            }
        )

    def set_final_reward(self, reward: float):
        """
        Add final reward for the trajectory.
        """
        self.request_queue.put(
            {
                "type": "set_final_reward",
                "reward": reward,
            }
        )


def RecorderServer(mw, *args, **kwargs):
    """Create a Recorder server instance."""
    return ServerFactory(mw, Recorder, *args, **kwargs)


def RecorderClient(mw, *args, **kwargs):
    """Create a Recorder client instance."""
    return ClientFactory(mw, Recorder, *args, **kwargs)
