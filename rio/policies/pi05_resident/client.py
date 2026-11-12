# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Authenticated RPC client of a live π₀.₅ resident."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from typing import Protocol

from loguru import logger

from .host import pid_alive
from .identity import identity_matches
from .paths import authkey_from_meta, clear_lock, read_lock, socket_path

PING_RETRIES = 3


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
    slot = identity.get("slot", "")
    meta = read_lock(identity["device"], slot)
    if meta is None:
        return None
    pid = int(meta.get("pid") or 0)
    if not pid_alive(pid):
        logger.info("Stale π₀.₅ lockfile for {} (pid {} dead)", identity["device"], pid)
        clear_lock(identity["device"], slot=slot, expected_pid=pid)
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
    address = meta.get("address") or str(socket_path(identity["device"], slot))
    try:
        client = ResidentClient(address=address, authkey=authkey_from_meta(meta), meta=meta)
        reply = client.ping()
        if reply.get("pid") != pid or not identity_matches(reply, identity):
            client.close()
            raise RuntimeError("Connected resident does not match the expected identity")
    except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
        logger.info("π₀.₅ resident pid {} not answering ({})", pid, exc)
        return None
    client.meta.update({k: reply[k] for k in reply if k not in {"ok", "error"}})
    return client
