# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep a π₀.₅ checkpoint resident in GPU memory across rollout restarts.

CUDA allocations die with the process, so a VRAM occupancy check cannot reuse
another job's weights. This daemon holds the tensors; later serve processes
attach over an authenticated local socket instead of calling ``from_pretrained`` again.

Modules: ``paths`` (owner-only runtime directory and metadata), ``identity``
(what must match for reuse), ``host`` (PID liveness, GPU occupancy), ``client``
(RPC), ``process`` (attach/spawn/stop under the GPU lock), ``server`` (the daemon).
"""

from .client import PING_RETRIES, ResidentClient, try_connect
from .host import gpu_compute_apps, gpu_index, log_gpu_occupancy, pid_alive
from .identity import checkpoint_fingerprint, identity_matches, resident_identity, runtime_fingerprint
from .paths import clear_lock, device_tag, flock_path, lock_path, log_path, read_lock, socket_path, write_lock
from .process import CONNECT_TIMEOUT_S, ensure_resident, shutdown_resident, spawn_resident, start_resident

__all__ = [
    "CONNECT_TIMEOUT_S",
    "PING_RETRIES",
    "ResidentClient",
    "checkpoint_fingerprint",
    "clear_lock",
    "device_tag",
    "ensure_resident",
    "flock_path",
    "gpu_compute_apps",
    "gpu_index",
    "identity_matches",
    "lock_path",
    "log_gpu_occupancy",
    "log_path",
    "pid_alive",
    "read_lock",
    "resident_identity",
    "runtime_fingerprint",
    "shutdown_resident",
    "socket_path",
    "spawn_resident",
    "start_resident",
    "try_connect",
    "write_lock",
]
