# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Owner-only runtime directory, per-device/slot file names, and the resident metadata file."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path

LEGACY_LOCK_DIR = Path(tempfile.gettempdir())
LOCK_DIR = LEGACY_LOCK_DIR / f"rio_pi05_{os.getuid() if hasattr(os, 'getuid') else 'user'}"


def device_tag(device: str, slot: str = "") -> str:
    """Filesystem-safe device id, e.g. ``cuda-1``."""
    text = str(device).strip().lower()
    if not re.fullmatch(r"(?:cuda|cpu)(?::\d+)?", text):
        raise ValueError(f"Unsupported resident device {device!r}")
    if slot and not re.fullmatch(r"[a-z0-9_]{1,32}", slot):
        raise ValueError("Resident slot must contain 1-32 lowercase letters, digits, or underscores")
    return text.replace(":", "-") + (f"-{slot}" if slot else "")


def ensure_runtime_directory() -> None:
    if LOCK_DIR.is_symlink():
        raise PermissionError("Resident runtime directory must not be a symbolic link")
    LOCK_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        if LOCK_DIR.stat().st_uid != os.getuid():
            raise PermissionError("Resident runtime directory belongs to another user")
        LOCK_DIR.chmod(0o700)


def metadata_paths(device: str, slot: str = "") -> list[Path]:
    if slot:
        return [lock_path(device, slot)]
    return list(dict.fromkeys([lock_path(device), LEGACY_LOCK_DIR / f"rio_pi05_{device_tag(device)}.json"]))


def lock_path(device: str, slot: str = "") -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device, slot)}.json"


def socket_path(device: str, slot: str = "") -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device, slot)}.sock"


def flock_path(device: str) -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device)}.lock"


def log_path(device: str, slot: str = "") -> Path:
    return LOCK_DIR / f"rio_pi05_{device_tag(device, slot)}.log"


def resolved_policy_path(policy_path: str | Path) -> str:
    return str(Path(policy_path).expanduser().resolve())


def read_lock(device: str, slot: str = "") -> dict | None:
    for path in metadata_paths(device, slot):
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
        if data.get("slot", "") != slot:
            raise RuntimeError("Resident metadata belongs to another slot")
        return data
    return None


def write_lock(meta: dict) -> None:
    ensure_runtime_directory()
    path = lock_path(meta["device"], meta.get("slot", ""))
    descriptor, temporary = tempfile.mkstemp(prefix=".rio_pi05_", dir=LOCK_DIR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def clear_lock(device: str, *, slot: str = "", expected_pid: int | None = None) -> None:
    current = read_lock(device, slot)
    if current is not None and expected_pid is not None and current.get("pid") != expected_pid:
        return
    paths = [*metadata_paths(device, slot), socket_path(device, slot)]
    if not slot:
        paths.append(LEGACY_LOCK_DIR / f"rio_pi05_{device_tag(device)}.sock")
    for path in paths:
        if path.is_symlink():
            raise PermissionError("Refusing to remove a symbolic link in resident metadata")
        if path.exists():
            if os.name == "posix" and path.stat().st_uid != os.getuid():
                raise PermissionError("Refusing to remove another user's resident file")
            path.unlink()


def authkey_from_meta(meta: dict) -> bytes:
    raw = meta.get("authkey", "")
    if isinstance(raw, str):
        key = bytes.fromhex(raw)
    else:
        key = bytes(raw)
    if len(key) < 16:
        raise ValueError("Resident IPC authentication key is missing or too short")
    return key
