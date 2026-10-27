# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep a π₀.₅ checkpoint resident in GPU memory across policy_inference restarts.

CUDA allocations die with the process, so a VRAM occupancy check cannot reuse
another job's weights. This daemon holds the tensors; later serve processes
attach over a Unix socket instead of calling ``from_pretrained`` again.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from pathlib import Path

from loguru import logger

LOCK_DIR = Path("/tmp")
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
    return str(device).strip().lower().replace(":", "-")


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


def resident_identity(policy_path: str | Path, device: str, *, use_rtc: bool, compile: bool) -> dict:
    """Fields that must match for a live daemon to be reused."""
    return {
        "policy_path": resolved_policy_path(policy_path),
        "device": str(device).strip(),
        "use_rtc": bool(use_rtc),
        "compile": bool(compile),
    }


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
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
    path = lock_path(device)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_lock(meta: dict) -> None:
    path = lock_path(meta["device"])
    path.write_text(json.dumps(meta, indent=2) + "\n")


def clear_lock(device: str) -> None:
    lock_path(device).unlink(missing_ok=True)
    sock = socket_path(device)
    if sock.exists():
        sock.unlink()


def _authkey_from_meta(meta: dict) -> bytes:
    raw = meta.get("authkey", "")
    if isinstance(raw, str):
        return bytes.fromhex(raw)
    return bytes(raw)


def identity_matches(meta: dict, identity: dict) -> bool:
    for key in ("policy_path", "device", "use_rtc", "compile"):
        if meta.get(key) != identity.get(key):
            return False
    return True


@dataclass
class ResidentClient:
    """RPC client for a live π₀.₅ daemon."""

    address: str
    authkey: bytes
    meta: dict
    _conn: object | None = None

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
                if self._conn is None:
                    self._connect()
                self._conn.send(payload)
                reply = self._conn.recv()
                if not isinstance(reply, dict):
                    raise RuntimeError(f"resident returned {type(reply)}")
                if reply.get("error"):
                    raise RuntimeError(str(reply["error"]))
                return reply
            except (OSError, EOFError, ConnectionError) as exc:
                last_error = exc
                self.close()
                time.sleep(0.2 * (attempt + 1))
        raise ConnectionError(f"π₀.₅ resident at {self.address} is not reachable") from last_error

    def ping(self) -> dict:
        return self.call({"op": "ping"})

    def set_instruction(self, instruction: str | None) -> None:
        self.call({"op": "set_instruction", "instruction": instruction})

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
        clear_lock(identity["device"])
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
    except (OSError, ConnectionError, RuntimeError) as exc:
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
    if extra.get("binarize_gripper") is False:
        cmd.append("--no-binarize-gripper")
    if extra.get("action_space"):
        cmd.extend(["--action-space", str(extra["action_space"])])
    log_file = log_path(identity["device"])
    handle = open(log_file, "ab", buffering=0)
    logger.info("Starting π₀.₅ resident on {} (log {})", identity["device"], log_file)
    return subprocess.Popen(
        cmd,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[2]),
        close_fds=True,
    )


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
    identity = resident_identity(policy_path, device, use_rtc=use_rtc, compile=compile)
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

    flock = flock_path(device)
    flock.parent.mkdir(parents=True, exist_ok=True)
    with open(flock, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        existing = try_connect(identity)
        if existing is not None:
            logger.info("Reusing π₀.₅ already on {} (pid {})", device, existing.pid)
            return existing
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
    raise TimeoutError(f"π₀.₅ resident did not become ready on {device} within {timeout_s:.0f}s")


def shutdown_resident(device: str, *, policy_path: str | Path | None = None) -> bool:
    """Ask our daemon to unload. Never signals a PID we did not spawn."""
    meta = read_lock(device)
    if meta is None:
        logger.info("No π₀.₅ resident lockfile for {}", device)
        return False
    pid = int(meta.get("pid") or 0)
    if policy_path is not None and meta.get("policy_path") != resolved_policy_path(policy_path):
        logger.warning("Refusing to unload pid {}: lockfile path {} != {}", pid, meta.get("policy_path"), policy_path)
        return False
    identity = {
        "policy_path": meta.get("policy_path"),
        "device": meta.get("device", device),
        "use_rtc": bool(meta.get("use_rtc")),
        "compile": bool(meta.get("compile")),
    }
    client = try_connect(identity)
    if client is not None:
        logger.info("Unloading π₀.₅ resident pid {} on {}", pid, device)
        client.shutdown()
    elif pid_alive(pid):
        logger.info("Sending SIGTERM to π₀.₅ resident pid {}", pid)
        os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not pid_alive(pid):
            break
        time.sleep(0.1)
    clear_lock(device)
    return True


def _serve_forever(cfg) -> None:
    from rio.policies.pi05 import Pi05

    identity = resident_identity(cfg.policy_path, cfg.device, use_rtc=cfg.use_rtc, compile=cfg.compile)
    address = str(socket_path(cfg.device))
    authkey = os.urandom(16)
    sock = Path(address)
    if sock.exists():
        sock.unlink()

    policy = Pi05(
        policy_path=identity["policy_path"],
        device=cfg.device,
        compile=cfg.compile,
        binarize_gripper=cfg.binarize_gripper,
        use_rtc=cfg.use_rtc,
        action_space=cfg.action_space,
    )
    policy.construct_policy(in_process=True)
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

    listener = Listener(address, authkey=authkey)
    stopping = False
    infer_lock = threading.Lock()

    def _on_term(_signum, _frame):
        nonlocal stopping
        stopping = True
        try:
            listener.close()
        except OSError:
            pass

    def _handle(conn) -> None:
        nonlocal stopping
        try:
            while not stopping:
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
                        elif op == "infer":
                            conn.send(policy.inference(msg["obs"]))
                        elif op == "shutdown":
                            conn.send({"ok": True})
                            stopping = True
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

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _on_term)

    try:
        while not stopping:
            try:
                conn = listener.accept()
            except OSError:
                break
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()
    finally:
        try:
            listener.close()
        except OSError:
            pass
        clear_lock(cfg.device)
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
        action_space: str | None = None

    main(tyro.cli(ResidentCfg))
