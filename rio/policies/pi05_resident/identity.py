# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""What a resident must match before it is reused: checkpoint, code, libraries, and inference options."""

from __future__ import annotations

import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .paths import device_tag, resolved_policy_path

_PACKAGE = Path(__file__).resolve().parent
_RIO = _PACKAGE.parents[1]
# Every source file the resident process executes besides third-party libraries.
_RUNTIME_SOURCES = (
    *sorted(_PACKAGE.glob("*.py")),
    _RIO / "policies" / "pi05.py",
    _RIO / "policies" / "rtc.py",
    _RIO / "gripper.py",
    _RIO / "rotations.py",
)


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
    for path in _RUNTIME_SOURCES:
        digest.update(path.name.encode())
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
    slot: str = "",
    binarize_gripper: bool = True,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
    rtc_execution_horizon: int = 18,
    rtc_max_guidance_weight: float = 10.0,
    action_space: str | None = None,
) -> dict:
    """Fields that must match for a live daemon to be reused."""
    device_tag(device, slot)
    return {
        "policy_path": resolved_policy_path(policy_path),
        "device": str(device).strip(),
        "slot": slot,
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


def identity_matches(meta: dict, identity: dict) -> bool:
    required = {"policy_path", "device", "use_rtc", "compile"}
    return required.issubset(identity) and all(
        meta.get(key, "" if key == "slot" else None) == value for key, value in identity.items()
    )
