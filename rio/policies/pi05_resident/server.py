# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The resident process: load π₀.₅ once, then answer authenticated RPCs until shut down."""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import TypedDict

from loguru import logger

from rio.policies.rtc import RTC_EXECUTION_HORIZON, RTC_MAX_GUIDANCE_WEIGHT

from .identity import checkpoint_fingerprint, resident_identity
from .paths import clear_lock, ensure_runtime_directory, socket_path, write_lock


@dataclass
class ResidentCfg:
    """Command line of ``python -m rio.policies.pi05_resident`` (started by ``spawn_resident``)."""

    policy_path: str
    device: str = "cuda:0"
    slot: str = ""
    use_rtc: bool = True
    compile: bool = False
    binarize_gripper: bool = True
    gripper_close: float = 0.0
    gripper_open: float = 0.7
    gripper_threshold: float | None = None
    rtc_execution_horizon: int = RTC_EXECUTION_HORIZON
    rtc_max_guidance_weight: float = RTC_MAX_GUIDANCE_WEIGHT
    action_space: str | None = None


def _serve_forever(cfg: ResidentCfg) -> None:
    from rio.policies.pi05 import Pi05

    class InferenceOptions(TypedDict):
        use_rtc: bool
        compile: bool
        binarize_gripper: bool
        gripper_close: float
        gripper_open: float
        gripper_threshold: float | None
        rtc_execution_horizon: int
        rtc_max_guidance_weight: float
        action_space: str | None

    options: InferenceOptions = {
        "use_rtc": cfg.use_rtc,
        "compile": cfg.compile,
        "binarize_gripper": cfg.binarize_gripper,
        "gripper_close": cfg.gripper_close,
        "gripper_open": cfg.gripper_open,
        "gripper_threshold": cfg.gripper_threshold,
        "rtc_execution_horizon": int(cfg.rtc_execution_horizon),
        "rtc_max_guidance_weight": float(cfg.rtc_max_guidance_weight),
        "action_space": cfg.action_space,
    }
    slot = cfg.slot
    identity = resident_identity(cfg.policy_path, cfg.device, slot=slot, **options)
    ensure_runtime_directory()
    address = str(socket_path(cfg.device, slot))
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
        clear_lock(cfg.device, slot=slot, expected_pid=os.getpid())
        logger.info("π₀.₅ resident on {} exiting", cfg.device)


def main(cfg) -> None:
    _serve_forever(cfg)


def cli() -> None:
    """Parse ``ResidentCfg`` and serve."""
    import tyro

    main(tyro.cli(ResidentCfg))
