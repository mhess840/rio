# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep a π₀.₅ checkpoint resident in GPU memory across policy_inference restarts.

CUDA allocations die with the process, so a VRAM occupancy check cannot reuse
another job's weights. This daemon holds the tensors; later serve processes
attach over a Unix socket instead of calling ``from_pretrained`` again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Protocol

from loguru import logger

try:
    import fcntl
except ImportError:
    fcntl = None

LEGACY_LOCK_DIR = Path(tempfile.gettempdir())
LOCK_DIR = LEGACY_LOCK_DIR / f"rio_pi05_{os.getuid() if hasattr(os, 'getuid') else 'user'}"
CONNECT_TIMEOUT_S = 180.0
PING_RETRIES = 3


def gpu_index(device: str) -> int | None:
    """Return the CUDA index for ``cuda:N``, or None for CPU."""
    text = str(device).strip().lower()
    if text == "cpu" or text.startswith("cpu:"):
        return None
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text == "cuda":
        return 0
    return None


def device_tag(device: str) -> str:
    """Filesystem-safe device id, e.g. ``cuda-1``."""
    text = str(device).strip().lower()
    if not re.fullmatch(r"(?:cuda|cpu)(?::\d+)?", text):
        raise ValueError(f"Unsupported resident device {device!r}")
    return text.replace(":", "-")


def _ensure_runtime_directory() -> None:
    if LOCK_DIR.is_symlink():
        raise PermissionError("Resident runtime directory must not be a symbolic link")
    LOCK_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        if LOCK_DIR.stat().st_uid != os.getuid():
            raise PermissionError("Resident runtime directory belongs to another user")
        LOCK_DIR.chmod(0o700)


def _metadata_paths(device: str) -> list[Path]:
    return list(dict.fromkeys([lock_path(device), LEGACY_LOCK_DIR / f"rio_pi05_{device_tag(device)}.json"]))


def lock_path(device: str) -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device)}.json"


def socket_path(device: str) -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device)}.sock"


def flock_path(device: str) -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device)}.lock"


def log_path(device: str) -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device)}.log"


def resolved_policy_path(policy_path: str | Path) -> str:
    return str(Path(policy_path).expanduser().resolve())


def checkpoint_fingerprint(policy_path: str | Path) -> str:
    """Detect checkpoint replacement without repeatedly reading multi-gigabyte weights."""
    root = Path(policy_path).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(root.glob("*")):
        if not path.is_file() or path.suffix not in {".json", ".safetensors", ".bin", ".model"}:
            continue
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)).encode())
        if path.suffix == ".json":
            digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_fingerprint() -> str:
    """Identify the code and library versions actually loaded by the resident."""
    digest = hashlib.sha256(sys.version.encode())
    for path in (Path(__file__), Path(__file__).with_name("pi05.py"), Path(__file__).parents[1] / "gripper.py"):
        digest.update(path.read_bytes())
    for package in ("lerobot", "torch", "transformers"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not-installed"
        digest.update(f"{package}:{installed}".encode())
    return digest.hexdigest()


def resident_identity(
    policy_path: str | Path,
    device: str,
    *,
    use_rtc: bool,
    compile: bool,
    binarize_gripper: bool = True,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
    rtc_execution_horizon: int = 18,
    rtc_max_guidance_weight: float = 10.0,
    action_space: str | None = None,
) -> dict:
    """Fields that must match for a live daemon to be reused."""
    return {
        "policy_path": resolved_policy_path(policy_path),
        "device": str(device).strip(),
        "use_rtc": bool(use_rtc),
        "compile": bool(compile),
        "binarize_gripper": bool(binarize_gripper),
        "gripper_close": float(gripper_close),
        "gripper_open": float(gripper_open),
        "gripper_threshold": None if gripper_threshold is None else float(gripper_threshold),
        "rtc_execution_horizon": int(rtc_execution_horizon),
        "rtc_max_guidance_weight": float(rtc_max_guidance_weight),
        "requested_action_space": action_space,
        "checkpoint_fingerprint": checkpoint_fingerprint(policy_path),
        "runtime_fingerprint": runtime_fingerprint(),
    }


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) and exit_code.value == 259)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_compute_apps(device: str) -> list[dict]:
    """PIDs currently using the given GPU. Empty if nvidia-smi is missing."""
    idx = gpu_index(device)
    if idx is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        apps.append(
            {
                "pid": pid,
                "name": parts[1] if len(parts) > 1 else "",
                "memory_mib": parts[2] if len(parts) > 2 else "",
            }
        )
    return apps


def read_lock(device: str) -> dict | None:
    for path in _metadata_paths(device):
        try:
            if path.is_symlink():
                raise PermissionError("Resident metadata must not be a symbolic link")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            continue
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or (os.name == "posix" and info.st_uid != os.getuid()):
                raise PermissionError("Resident metadata is not an owned regular file")
            if os.name == "posix" and path.parent == LOCK_DIR and info.st_mode & 0o077:
                raise PermissionError("Resident metadata exposes its IPC authentication key")
            try:
                data = json.load(handle)
            except json.JSONDecodeError as error:
                raise RuntimeError("Invalid resident metadata; refusing to create a duplicate GPU process") from error
        if not isinstance(data, dict) or data.get("device") != device or not isinstance(data.get("pid"), int):
            raise RuntimeError("Resident metadata has an invalid device or PID")
        return data
    return None


def write_lock(meta: dict) -> None:
    _ensure_runtime_directory()
    path = lock_path(meta["device"])
    descriptor, temporary = tempfile.mkstemp(prefix=".rio_pi05_", dir=LOCK_DIR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def clear_lock(device: str, *, expected_pid: int | None = None) -> None:
    current = read_lock(device)
    if current is not None and expected_pid is not None and current.get("pid") != expected_pid:
        return
    for path in [*_metadata_paths(device), socket_path(device), LEGACY_LOCK_DIR / f"rio_pi05_{device_tag(device)}.sock"]:
        if path.is_symlink():
            raise PermissionError("Refusing to remove a symbolic link in resident metadata")
        if path.exists():
            if os.name == "posix" and path.stat().st_uid != os.getuid():
                raise PermissionError("Refusing to remove another user's resident file")
            path.unlink()


def _authkey_from_meta(meta: dict) -> bytes:
    raw = meta.get("authkey", "")
    if isinstance(raw, str):
        key = bytes.fromhex(raw)
    else:
        key = bytes(raw)
    if len(key) < 16:
        raise ValueError("Resident IPC authentication key is missing or too short")
    return key


def identity_matches(meta: dict, identity: dict) -> bool:
    required = {"policy_path", "device", "use_rtc", "compile"}
    return required.issubset(identity) and all(meta.get(key) == value for key, value in identity.items())


@dataclass
class ResidentClient:
    """RPC client for a live π₀.₅ daemon."""

    class Connection(Protocol):
        def send(self, value: dict, /) -> None: ...
        def poll(self, timeout: float) -> bool: ...
        def recv(self) -> object: ...
        def close(self) -> None: ...

    address: str
    authkey: bytes
    meta: dict
    timeout_s: float = 5.0
    _conn: Connection | None = None

    @property
    def pid(self) -> int:
        return int(self.meta.get("pid") or 0)

    def close(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def _connect(self):
        self.close()
        self._conn = Client(self.address, authkey=self.authkey)

    def call(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(PING_RETRIES):
            try:
                connection = self._conn
                if connection is None:
                    self._connect()
                    connection = self._conn
                if connection is None:
                    raise ConnectionError("Resident connection could not be established")
                connection.send(payload)
                if not connection.poll(self.timeout_s):
                    raise TimeoutError("Resident RPC timed out")
                reply = connection.recv()
                if not isinstance(reply, dict):
                    raise RuntimeError(f"resident returned {type(reply)}")
                if reply.get("error"):
                    raise RuntimeError(str(reply["error"]))
                return reply
            except AuthenticationError as error:
                self.close()
                raise ConnectionError("Resident IPC authentication failed") from error
            except (OSError, EOFError, ConnectionError) as exc:
                last_error = exc
                self.close()
                if payload.get("op") != "ping":
                    raise ConnectionError("Resident request failed; stale work will not be replayed") from exc
                time.sleep(0.2 * (attempt + 1))
        raise ConnectionError(f"π₀.₅ resident at {self.address} is not reachable") from last_error

    def ping(self) -> dict:
        return self.call({"op": "ping"})

    def set_instruction(self, instruction: str | None) -> None:
        self.call({"op": "set_instruction", "instruction": instruction})

    def reset(self) -> None:
        self.call({"op": "reset"})

    def inference(self, observation: dict):
        reply = self.call({"op": "infer", "obs": observation})
        return {
            "actions": reply["actions"],
            "original_actions": reply.get("original_actions", reply["actions"]),
        }

    def shutdown(self) -> None:
        try:
            self.call({"op": "shutdown"})
        except (OSError, ConnectionError, RuntimeError):
            pass
        finally:
            self.close()


def try_connect(identity: dict) -> ResidentClient | None:
    """Return a live matching daemon, or None."""
    meta = read_lock(identity["device"])
    if meta is None:
        return None
    pid = int(meta.get("pid") or 0)
    if not pid_alive(pid):
        logger.info("Stale π₀.₅ lockfile for {} (pid {} dead)", identity["device"], pid)
        clear_lock(identity["device"], expected_pid=pid)
        return None
    if not identity_matches(meta, identity):
        logger.warning(
            "π₀.₅ resident pid {} on {} holds {} rtc={} compile={}; wanted {} rtc={} compile={}",
            pid,
            meta.get("device"),
            meta.get("policy_path"),
            meta.get("use_rtc"),
            meta.get("compile"),
            identity["policy_path"],
            identity["use_rtc"],
            identity["compile"],
        )
        return None
    address = meta.get("address") or str(socket_path(identity["device"]))
    try:
        client = ResidentClient(address=address, authkey=_authkey_from_meta(meta), meta=meta)
        reply = client.ping()
        if reply.get("pid") != pid or not identity_matches(reply, identity):
            client.close()
            raise RuntimeError("Connected resident does not match the expected identity")
    except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
        logger.info("π₀.₅ resident pid {} not answering ({})", pid, exc)
        return None
    client.meta.update({k: reply[k] for k in reply if k not in {"ok", "error"}})
    return client


def log_gpu_occupancy(device: str) -> None:
    apps = gpu_compute_apps(device)
    if not apps:
        logger.info("GPU {} has no listed compute apps (or nvidia-smi missing)", device)
        return
    parts = [f"pid={a['pid']} {a['name']} {a['memory_mib']} MiB" for a in apps]
    logger.info("GPU {} occupancy: {}", device, "; ".join(parts))


def spawn_resident(identity: dict, extra: dict | None = None) -> subprocess.Popen:
    """Start a detached daemon. Caller holds the device flock."""
    extra = extra or {}
    _ensure_runtime_directory()
    sock = socket_path(identity["device"])
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
    log_file = log_path(identity["device"])
    logger.info("Starting π₀.₅ resident on {} (log {})", identity["device"], log_file)
    with open(log_file, "ab", buffering=0) as handle:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[2]),
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
    _ensure_runtime_directory()
    flock = flock_path(device)
    with open(flock, "a") as lock_fd:
        acquire(lock_fd.fileno(), exclusive)
        yield


def _shutdown_metadata(meta: dict, timeout_s: float = 10.0) -> None:
    """Stop only a resident that authenticates and confirms its PID and checkpoint."""
    device = meta["device"]
    pid = int(meta.get("pid") or 0)
    if not pid_alive(pid):
        clear_lock(device, expected_pid=pid)
        return
    client = ResidentClient(
        address=meta.get("address") or str(socket_path(device)), authkey=_authkey_from_meta(meta), meta=meta
    )
    try:
        reply = client.ping()
        if int(reply.get("pid") or 0) != pid or reply.get("policy_path") != meta.get("policy_path"):
            raise RuntimeError("Resident identity changed; refusing to stop or replace it")
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
    clear_lock(device, expected_pid=pid)


def ensure_resident(
    policy_path: str | Path,
    device: str,
    *,
    use_rtc: bool = True,
    compile: bool = False,
    timeout_s: float = CONNECT_TIMEOUT_S,
    **extra,
) -> ResidentClient:
    """Attach to a matching daemon, or spawn one and wait until weights are on GPU."""
    identity = resident_identity(policy_path, device, use_rtc=use_rtc, compile=compile, **extra)
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
        previous = read_lock(device)
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
                log_file = log_path(device)
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
        clear_lock(device, expected_pid=proc.pid)
        raise TimeoutError(f"π₀.₅ resident did not become ready on {device} within {timeout_s:.0f}s")


def shutdown_resident(device: str, *, policy_path: str | Path | None = None) -> bool:
    """Ask our daemon to unload. Never signals a PID we did not spawn."""
    with _device_lock(device):
        meta = read_lock(device)
        if meta is None:
            logger.info("No π₀.₅ resident lockfile for {}", device)
            return False
        if policy_path is not None and meta.get("policy_path") != resolved_policy_path(policy_path):
            logger.warning("Refusing to unload a different checkpoint on {}", device)
            return False
        _shutdown_metadata(meta)
        return True


def _serve_forever(cfg) -> None:
    from rio.policies.pi05 import Pi05

    options = {
        "use_rtc": cfg.use_rtc,
        "compile": cfg.compile,
        "binarize_gripper": cfg.binarize_gripper,
        "gripper_close": getattr(cfg, "gripper_close", 0.0),
        "gripper_open": getattr(cfg, "gripper_open", 0.7),
        "gripper_threshold": getattr(cfg, "gripper_threshold", None),
        "rtc_execution_horizon": int(getattr(cfg, "rtc_execution_horizon", 18)),
        "rtc_max_guidance_weight": float(getattr(cfg, "rtc_max_guidance_weight", 10.0)),
        "action_space": cfg.action_space,
    }
    identity = resident_identity(cfg.policy_path, cfg.device, **options)
    _ensure_runtime_directory()
    address = str(socket_path(cfg.device))
    authkey = os.urandom(16)
    sock = Path(address)
    if sock.exists():
        sock.unlink()

    policy = Pi05(
        policy_path=identity["policy_path"],
        device=cfg.device,
        **options,
    )
    policy.construct_policy(in_process=True)
    if identity["checkpoint_fingerprint"] != checkpoint_fingerprint(cfg.policy_path):
        raise RuntimeError("Checkpoint changed while loading; copy it atomically before retrying")
    listener = Listener(address, authkey=authkey)
    if os.name == "posix":
        Path(address).chmod(0o600)
    write_lock(
        {
            **identity,
            "pid": os.getpid(),
            "address": address,
            "authkey": authkey.hex(),
            "action_space": policy.action_space,
            "action_dim": policy.action_dim,
        }
    )
    logger.info("π₀.₅ resident loaded on {} pid {}", cfg.device, os.getpid())
    stopping = threading.Event()
    infer_lock = threading.Lock()

    def _on_term(_signum, _frame):
        stopping.set()
        try:
            listener.close()
        except OSError:
            pass

    def _handle(conn) -> None:
        try:
            while not stopping.is_set():
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                op = msg.get("op") if isinstance(msg, dict) else None
                try:
                    with infer_lock:
                        if op == "ping":
                            conn.send(
                                {
                                    "ok": True,
                                    "pid": os.getpid(),
                                    **identity,
                                    "action_space": policy.action_space,
                                    "action_dim": policy.action_dim,
                                }
                            )
                        elif op == "set_instruction":
                            policy.set_instruction(msg.get("instruction"))
                            conn.send({"ok": True})
                        elif op == "reset":
                            policy.reset()
                            conn.send({"ok": True})
                        elif op == "infer":
                            conn.send(policy.inference(msg["obs"]))
                        elif op == "shutdown":
                            conn.send({"ok": True})
                            stopping.set()
                            try:
                                listener.close()
                            except OSError:
                                pass
                            break
                        else:
                            conn.send({"error": f"unknown op {op!r}"})
                except Exception as exc:
                    logger.exception("resident op {} failed", op)
                    try:
                        conn.send({"error": str(exc)})
                    except OSError:
                        break
        finally:
            conn.close()

    def _accept_connections() -> None:
        while not stopping.is_set():
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError):
                logger.warning("Rejected an unauthenticated resident connection")
                continue
            except OSError:
                stopping.set()
                break
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _on_term)
    threading.Thread(target=_accept_connections, daemon=True).start()
    try:
        stopping.wait()
    finally:
        try:
            listener.close()
        except OSError:
            pass
        clear_lock(cfg.device, expected_pid=os.getpid())
        logger.info("π₀.₅ resident on {} exiting", cfg.device)


def main(cfg) -> None:
    _serve_forever(cfg)


if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    import tyro

    @_dc
    class ResidentCfg:
        policy_path: str
        device: str = "cuda:0"
        use_rtc: bool = True
        compile: bool = False
        binarize_gripper: bool = True
        gripper_close: float = 0.0
        gripper_open: float = 0.7
        gripper_threshold: float | None = None
        rtc_execution_horizon: int = 18
        rtc_max_guidance_weight: float = 10.0
        action_space: str | None = None

    main(tyro.cli(ResidentCfg))
