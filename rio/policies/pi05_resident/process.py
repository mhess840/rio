# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Resident lifecycle: attach, spawn under the GPU-wide lock, and stop only authenticated residents."""

from __future__ import annotations

import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from rio.policies.rtc import RTC_EXECUTION_HORIZON, RTC_MAX_GUIDANCE_WEIGHT

from .client import ResidentClient, try_connect
from .host import log_gpu_occupancy, pid_alive
from .identity import resident_identity
from .paths import (
    authkey_from_meta,
    clear_lock,
    ensure_runtime_directory,
    flock_path,
    log_path,
    read_lock,
    resolved_policy_path,
    socket_path,
)

try:
    import fcntl
except ImportError:
    fcntl = None

CONNECT_TIMEOUT_S = 180.0
# Directory that contains the ``rio`` package; the daemon starts there.
_PACKAGE_PARENT = Path(__file__).resolve().parents[3]


def spawn_resident(identity: dict, extra: dict | None = None) -> subprocess.Popen:
    """Start a detached daemon. Caller holds the device flock."""
    extra = extra or {}
    ensure_runtime_directory()
    slot = identity.get("slot", "")
    sock = socket_path(identity["device"], slot)
    if sock.exists():
        sock.unlink()
    cmd = [
        sys.executable,
        "-m",
        "rio.policies.pi05_resident",
        "--policy-path",
        identity["policy_path"],
        "--device",
        identity["device"],
    ]
    if slot:
        cmd.extend(["--slot", slot])
    cmd.append("--use-rtc" if identity["use_rtc"] else "--no-use-rtc")
    if identity["compile"]:
        cmd.append("--compile")
    if identity["binarize_gripper"] is False:
        cmd.append("--no-binarize-gripper")
    if extra.get("action_space"):
        cmd.extend(["--action-space", str(extra["action_space"])])
    if extra.get("rtc_execution_horizon"):
        cmd.extend(["--rtc-execution-horizon", str(int(extra["rtc_execution_horizon"]))])
    for name in ("gripper_close", "gripper_open", "gripper_threshold", "rtc_max_guidance_weight"):
        if identity[name] is not None:
            cmd.extend(["--" + name.replace("_", "-"), str(identity[name])])
    log_file = log_path(identity["device"], slot)
    logger.info("Starting π₀.₅ resident on {} (log {})", identity["device"], log_file)
    with open(log_file, "ab", buffering=0) as handle:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(_PACKAGE_PARENT),
            close_fds=True,
        )


@contextmanager
def _device_lock(device: str):
    if fcntl is None:
        raise RuntimeError("The GPU resident requires POSIX file locking; use in-process inference on this platform")
    acquire = getattr(fcntl, "flock", None)
    exclusive = getattr(fcntl, "LOCK_EX", None)
    if not callable(acquire) or exclusive is None:
        raise RuntimeError("POSIX file locking is unavailable")
    ensure_runtime_directory()
    flock = flock_path(device)
    with open(flock, "a") as lock_fd:
        acquire(lock_fd.fileno(), exclusive)
        yield


def _shutdown_metadata(meta: dict, timeout_s: float = 10.0) -> None:
    """Stop only a resident that authenticates and confirms its PID and checkpoint."""
    device = meta["device"]
    slot = meta.get("slot", "")
    pid = int(meta.get("pid") or 0)
    if not pid_alive(pid):
        clear_lock(device, slot=slot, expected_pid=pid)
        return
    client = ResidentClient(
        address=meta.get("address") or str(socket_path(device, slot)), authkey=authkey_from_meta(meta), meta=meta
    )
    try:
        reply = client.ping()
        if int(reply.get("pid") or 0) != pid or reply.get("policy_path") != meta.get("policy_path"):
            raise RuntimeError("Resident identity changed; refusing to stop or replace it")
        if reply.get("slot", "") != slot:
            raise RuntimeError("Resident slot changed; refusing to stop or replace it")
        client.call({"op": "shutdown"})
    except (OSError, EOFError, ValueError, ConnectionError) as error:
        raise RuntimeError("Existing resident cannot be authenticated; refusing to start a second GPU process") from error
    finally:
        client.close()
    deadline = time.monotonic() + timeout_s
    while pid_alive(pid):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Resident pid {pid} did not exit; no replacement was started")
        time.sleep(0.05)
    clear_lock(device, slot=slot, expected_pid=pid)


def ensure_resident(
    policy_path: str | Path,
    device: str,
    *,
    slot: str = "",
    use_rtc: bool = True,
    compile: bool = False,
    timeout_s: float = CONNECT_TIMEOUT_S,
    **extra,
) -> ResidentClient:
    """Attach to a matching daemon, or spawn one and wait until weights are on GPU."""
    identity = resident_identity(policy_path, device, slot=slot, use_rtc=use_rtc, compile=compile, **extra)
    log_gpu_occupancy(device)
    existing = try_connect(identity)
    if existing is not None:
        logger.info(
            "Reusing π₀.₅ already on {} (pid {}, {})",
            device,
            existing.pid,
            identity["policy_path"],
        )
        return existing

    with _device_lock(device):
        existing = try_connect(identity)
        if existing is not None:
            logger.info("Reusing π₀.₅ already on {} (pid {})", device, existing.pid)
            return existing
        previous = read_lock(device, slot)
        if previous is not None:
            _shutdown_metadata(previous)
        proc = spawn_resident(identity, extra)
        logger.info("Spawned π₀.₅ resident pid {} — loading 9 GB onto {}", proc.pid, device)

        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while time.monotonic() < deadline:
            client = try_connect(identity)
            if client is not None:
                logger.info("π₀.₅ resident ready on {} (pid {})", device, client.pid)
                return client
            if proc.poll() is not None:
                tail = ""
                log_file = log_path(device, slot)
                if log_file.is_file():
                    tail = log_file.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"π₀.₅ resident exited {proc.returncode}. Log:\n{tail}")
            now = time.monotonic()
            if now - last_log > 5.0:
                logger.info("Waiting for π₀.₅ weights on {} ({:.0f}s left)", device, deadline - now)
                last_log = now
            time.sleep(0.5)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("The newly spawned resident did not terminate after startup timeout") from error
        clear_lock(device, slot=slot, expected_pid=proc.pid)
        raise TimeoutError(f"π₀.₅ resident did not become ready on {device} within {timeout_s:.0f}s")


def shutdown_resident(device: str, *, policy_path: str | Path | None = None, slot: str = "") -> bool:
    """Ask our daemon to unload. Never signals a PID we did not spawn."""
    with _device_lock(device):
        meta = read_lock(device, slot)
        if meta is None:
            logger.info("No π₀.₅ resident lockfile for {}", device)
            return False
        if policy_path is not None and meta.get("policy_path") != resolved_policy_path(policy_path):
            logger.warning("Refusing to unload a different checkpoint on {}", device)
            return False
        _shutdown_metadata(meta)
        return True


def start_resident(cfg) -> ResidentClient:
    """Attach to or start the resident selected by a π₀.₅ station config (``policy_cfg`` + ``policy_node_cfg``)."""
    policy = cfg.policy_cfg
    node = cfg.policy_node_cfg
    slot = str(policy.resident_slot or "")
    return ensure_resident(
        cfg.policy_path,
        str(policy.device),
        use_rtc=bool(policy.use_rtc),
        compile=bool(policy.compile),
        binarize_gripper=bool(policy.binarize_gripper),
        gripper_close=float(policy.gripper_close),
        gripper_open=float(policy.gripper_open),
        gripper_threshold=policy.gripper_threshold,
        rtc_max_guidance_weight=float(policy.rtc_max_guidance_weight or RTC_MAX_GUIDANCE_WEIGHT),
        action_space=policy.action_space,
        rtc_execution_horizon=int(node.rtc_execution_horizon or RTC_EXECUTION_HORIZON),
        timeout_s=float(node.timeout or CONNECT_TIMEOUT_S),
        **({"slot": slot} if slot else {}),
    )
