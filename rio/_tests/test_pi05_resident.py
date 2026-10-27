# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the π₀.₅ GPU-resident daemon (no checkpoint load)."""

import json
import os
import threading
from multiprocessing.connection import Listener
from pathlib import Path

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


def test_try_connect_clears_stale_lock(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(resident, "LOCK_DIR", tmp_path)
    identity = resident.resident_identity(tmp_path / "ckpt", "cuda:1", use_rtc=False, compile=False)
    meta = {**identity, "pid": 99999999, "address": str(tmp_path / "missing.sock"), "authkey": "ab"}
    resident.write_lock(meta)
    assert resident.lock_path("cuda:1").is_file()
    assert resident.try_connect(identity) is None
    assert not resident.lock_path("cuda:1").is_file()


def test_resident_client_ping_and_infer(tmp_path: Path):
    address = str(tmp_path / "pi05.sock")
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
    assert loaded["pid"] == os.getpid()
    assert json.loads(resident.lock_path("cuda:1").read_text())["device"] == "cuda:1"
