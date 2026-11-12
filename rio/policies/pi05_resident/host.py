# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Process liveness and GPU occupancy as seen from the host."""

from __future__ import annotations

import os
import subprocess
import sys

from loguru import logger


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


def log_gpu_occupancy(device: str) -> None:
    apps = gpu_compute_apps(device)
    if not apps:
        logger.info("GPU {} has no listed compute apps (or nvidia-smi missing)", device)
        return
    parts = [f"pid={a['pid']} {a['name']} {a['memory_mib']} MiB" for a in apps]
    logger.info("GPU {} occupancy: {}", device, "; ".join(parts))
