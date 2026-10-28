# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the π₀.₅ GPU-resident daemon (no checkpoint load)."""

import json
import os
import threading
import uuid
from contextlib import nullcontext
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from rio.policies import pi05_resident as resident

pytestmark = pytest.mark.unit


def test_gpu_index_and_device_tag():
    assert resident.gpu_index("cuda:1") == 1
    assert resident.gpu_index("cuda") == 0
    assert resident.gpu_index("cpu") is None
    assert resident.device_tag("cuda:1") == "cuda-1"


def test_identity_matches_ckpt_and_device(tmp_path: Path):
    wanted = resident.resident_identity(tmp_path, "cuda:1", use_rtc=False, compile=False)
    other = dict(wanted)
    assert resident.identity_matches(other, wanted)
    other["use_rtc"] = True
    assert not resident.identity_matches(other, wanted)


def test_identity_changes_when_weights_are_replaced_in_place(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first checkpoint")
    previous = resident.resident_identity(tmp_path, "cuda:1", use_rtc=True, compile=False)
    weights.write_bytes(b"second checkpoint")
    current = resident.resident_identity(tmp_path, "cuda:1", use_rtc=True, compile=False)
    assert previous["checkpoint_fingerprint"] != current["checkpoint_fingerprint"]
    assert not resident.identity_matches(previous, current)


@pytest.mark.parametrize("setting", [{"rtc_execution_horizon": 30}, {"binarize_gripper": False}, {"gripper_open": 1.0}])
def test_identity_includes_inference_encoding_and_rtc_settings(tmp_path, setting):
    previous = resident.resident_identity(tmp_path, "cuda:1", use_rtc=True, compile=False)
    current = resident.resident_identity(tmp_path, "cuda:1", use_rtc=True, compile=False, **setting)
    assert not resident.identity_matches(previous, current)


def test_process_liveness_probe_does_not_signal_the_current_process():
    assert resident.pid_alive(os.getpid())
    assert not resident.pid_alive(-1)


def test_try_connect_clears_stale_lock(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(resident, "LOCK_DIR", tmp_path)
    identity = resident.resident_identity(tmp_path / "ckpt", "cuda:1", use_rtc=False, compile=False)
    meta = {**identity, "pid": 99999999, "address": str(tmp_path / "missing.sock"), "authkey": "ab"}
    resident.write_lock(meta)
    assert resident.lock_path("cuda:1").is_file()
    assert resident.try_connect(identity) is None
    assert not resident.lock_path("cuda:1").is_file()


def test_resident_client_ping_and_infer(tmp_path: Path):
    address = rf"\\.\pipe\rio_pi05_test_{uuid.uuid4().hex}" if os.name == "nt" else str(tmp_path / "pi05.sock")
    authkey = b"unit-test-authkey"
    ready = threading.Event()
    shutdown = threading.Event()

    def serve():
        listener = Listener(address, authkey=authkey)
        ready.set()
        conn = listener.accept()
        try:
            while not shutdown.is_set():
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                if msg["op"] == "ping":
                    conn.send({"ok": True, "pid": os.getpid(), "action_space": "tcp", "action_dim": 7})
                elif msg["op"] == "infer":
                    conn.send(
                        {
                            "actions": np.zeros((2, 7), dtype=np.float32),
                            "original_actions": np.ones((2, 7), dtype=np.float32),
                        }
                    )
                elif msg["op"] == "shutdown":
                    conn.send({"ok": True})
                    break
        finally:
            conn.close()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    client = resident.ResidentClient(address=address, authkey=authkey, meta={"pid": os.getpid()})
    ping = client.ping()
    assert ping["ok"] is True
    out = client.inference({"camera_1": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert out["actions"].shape == (2, 7)
    np.testing.assert_array_equal(out["original_actions"][0], 1.0)
    client.shutdown()
    shutdown.set()
    thread.join(timeout=2)


def test_gpu_compute_apps_parses_nvidia_smi(monkeypatch):
    def fake_check_output(cmd, **kwargs):
        assert "-i" in cmd and "1" in cmd
        return "12345, python, 9312\n67890, python, 400\n"

    monkeypatch.setattr(resident.subprocess, "check_output", fake_check_output)
    apps = resident.gpu_compute_apps("cuda:1")
    assert [a["pid"] for a in apps] == [12345, 67890]
    assert apps[0]["memory_mib"] == "9312"


def test_mismatched_resident_exits_before_replacement_is_spawned(tmp_path, monkeypatch):
    events = []
    client = SimpleNamespace(pid=456)
    connects = iter([None, None, client])
    monkeypatch.setattr(resident, "try_connect", lambda identity: next(connects))
    monkeypatch.setattr(resident, "_device_lock", nullcontext)
    monkeypatch.setattr(resident, "log_gpu_occupancy", lambda device: None)
    monkeypatch.setattr(resident, "read_lock", lambda device: {"device": device, "pid": 123})
    monkeypatch.setattr(resident, "_shutdown_metadata", lambda meta: events.append("old_exited"))
    monkeypatch.setattr(resident, "spawn_resident", lambda *args: events.append("spawn") or SimpleNamespace(pid=456))
    assert resident.ensure_resident(tmp_path, "cuda:1", use_rtc=True, rtc_execution_horizon=30) is client
    assert events == ["old_exited", "spawn"]


def test_resident_that_cannot_stop_is_not_replaced(tmp_path, monkeypatch):
    spawned = []
    monkeypatch.setattr(resident, "try_connect", lambda identity: None)
    monkeypatch.setattr(resident, "_device_lock", nullcontext)
    monkeypatch.setattr(resident, "log_gpu_occupancy", lambda device: None)
    monkeypatch.setattr(resident, "read_lock", lambda device: {"device": device, "pid": 123})
    monkeypatch.setattr(
        resident, "_shutdown_metadata", lambda meta: (_ for _ in ()).throw(TimeoutError("old resident still running"))
    )
    monkeypatch.setattr(resident, "spawn_resident", lambda *args: spawned.append(True))
    with pytest.raises(TimeoutError, match="still running"):
        resident.ensure_resident(tmp_path, "cuda:1")
    assert not spawned


def test_shutdown_rejects_reused_or_mismatched_pid_without_sending_a_signal(monkeypatch):
    calls = []
    client = SimpleNamespace(
        ping=lambda: {"pid": 999, "policy_path": "/other"},
        call=calls.append,
        close=lambda: None,
    )
    monkeypatch.setattr(resident, "pid_alive", lambda pid: True)
    monkeypatch.setattr(resident, "ResidentClient", lambda **kwargs: client)
    monkeypatch.setattr(resident.os, "kill", lambda *args: calls.append("signal"))
    with pytest.raises(RuntimeError, match="identity changed"):
        resident._shutdown_metadata({"pid": 123, "device": "cuda:1", "policy_path": "/checkpoint", "authkey": "ab" * 16})
    assert not calls


def test_write_lock_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(resident, "LOCK_DIR", tmp_path)
    meta = {
        "policy_path": "/ckpt",
        "device": "cuda:1",
        "use_rtc": False,
        "compile": False,
        "pid": os.getpid(),
        "authkey": "aa",
    }
    resident.write_lock(meta)
    loaded = resident.read_lock("cuda:1")
    assert loaded is not None
    assert loaded["pid"] == os.getpid()
    assert json.loads(resident.lock_path("cuda:1").read_text())["device"] == "cuda:1"


@pytest.mark.parametrize("device", ["../escape", "cuda:1/../../escape", "not-a-device"])
def test_resident_device_cannot_escape_runtime_directory(device):
    with pytest.raises(ValueError, match="device"):
        resident.lock_path(device)


def test_rpc_timeout_closes_the_connection_without_hanging(monkeypatch):
    closed = []
    client = resident.ResidentClient(address="unused", authkey=b"a" * 16, meta={}, timeout_s=0.01)
    connection = SimpleNamespace(send=lambda payload: None, poll=lambda timeout: False, close=lambda: closed.append(True))
    monkeypatch.setattr(client, "_connect", lambda: setattr(client, "_conn", connection))
    monkeypatch.setattr(resident.time, "sleep", lambda duration: None)
    with pytest.raises(ConnectionError, match="not reachable"):
        client.ping()
    assert closed


def test_clear_lock_does_not_remove_a_newer_resident(tmp_path, monkeypatch):
    monkeypatch.setattr(resident, "LOCK_DIR", tmp_path)
    resident.write_lock({"pid": 222, "device": "cuda:1", "authkey": "ab" * 16})
    resident.clear_lock("cuda:1", expected_pid=111)
    loaded = resident.read_lock("cuda:1")
    assert loaded is not None
    assert loaded["pid"] == 222


def test_failed_inference_is_not_replayed_after_a_timeout():
    sent = []
    connection = SimpleNamespace(send=sent.append, poll=lambda timeout: False, close=lambda: None)
    client = resident.ResidentClient(
        address="unused", authkey=b"a" * 16, meta={}, _conn=cast(resident.ResidentClient.Connection, connection)
    )
    with pytest.raises(ConnectionError, match="not be replayed"):
        client.inference({"proprio": np.zeros(8)})
    assert len(sent) == 1


def test_server_recovers_from_bad_authentication_and_exits_after_shutdown(tmp_path, monkeypatch):
    from rio.policies import pi05

    calls = []
    messages = iter([{"op": "reset"}, {"op": "shutdown"}])
    replies = []
    connection = SimpleNamespace(recv=lambda: next(messages), send=replies.append, close=lambda: None)

    class FakePolicy:
        action_space = "tcp"
        action_dim = 7

        def __init__(self, **kwargs):
            pass

        def construct_policy(self, **kwargs):
            calls.append("loaded")

        def reset(self):
            calls.append("reset")

    attempts = []

    def accept():
        attempts.append(True)
        if len(attempts) == 1:
            raise resident.AuthenticationError("stale test key")
        return connection

    def listener(address, **kwargs):
        Path(address).touch()
        return SimpleNamespace(accept=accept, close=lambda: None)

    def inline_thread(*, target, args=(), **kwargs):
        return SimpleNamespace(start=lambda: target(*args))

    monkeypatch.setattr(resident, "LOCK_DIR", tmp_path / "runtime")
    monkeypatch.setattr(resident, "LEGACY_LOCK_DIR", tmp_path / "legacy")
    monkeypatch.setattr(resident, "Listener", listener)
    monkeypatch.setattr(resident.threading, "Thread", inline_thread)
    monkeypatch.setattr(resident.signal, "signal", lambda *args: None)
    monkeypatch.setattr(pi05, "Pi05", FakePolicy)
    cfg = SimpleNamespace(
        policy_path=str(tmp_path), device="cpu", use_rtc=False, compile=False, binarize_gripper=True, action_space="tcp"
    )
    resident._serve_forever(cfg)
    assert attempts == [True, True]
    assert calls == ["loaded", "reset"]
    assert replies == [{"ok": True}, {"ok": True}]
    assert not resident.lock_path("cpu").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX credential permissions")
def test_resident_credentials_are_private(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(resident, "LOCK_DIR", runtime)
    resident.write_lock({"pid": os.getpid(), "device": "cuda:1", "authkey": "ab" * 16})
    assert runtime.stat().st_mode & 0o777 == 0o700
    assert resident.lock_path("cuda:1").stat().st_mode & 0o777 == 0o600
