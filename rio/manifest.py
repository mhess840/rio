# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Versions that identify a run, written next to recordings and rollout diagnostics.

Kept outside ``rio.data`` so that building an environment does not import the recorder and formatters.
"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from loguru import logger

LIBRARIES = ("numpy", "torch", "lerobot", "robodm", "transformers")
_PINNED_COMMIT = re.compile(r"@\s*git\+\S+@([0-9a-fA-F]{40})\b")


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git(folder: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, timeout=5.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def package_source(module: str, distribution: str | None = None) -> dict | None:
    """Version, location, and (for a git checkout) commit and dirty flag of one package.

    Args:
        module: Import name, e.g. ``"rio_hw"``.
        distribution: Distribution name when it differs from ``module``.
    """
    spec = find_spec(module)
    if spec is None or spec.origin is None:
        return None
    folder = Path(spec.origin).resolve().parent
    commit = _git(folder, "rev-parse", "HEAD")
    status = _git(folder, "status", "--porcelain", "--untracked-files=no") if commit else None
    return {
        "version": _version(distribution or module),
        "path": str(folder),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def file_fingerprint(path: Path) -> dict | None:
    """Size, modification time, and SHA-256 of one file (e.g. the native bridge)."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest.hexdigest()}


def pinned_commit(distribution: str, dependency: str) -> str | None:
    """Git commit ``distribution`` pins ``dependency`` to in its ``Requires-Dist`` (``dep @ git+URL@SHA``).

    Args:
        distribution: Installed distribution that declares the pin, e.g. ``"rio"``.
        dependency: Name of the pinned dependency, e.g. ``"rio_hw"``.

    Returns:
        The 40-hex commit, or ``None`` when the distribution or the pin is not found.
    """
    try:
        requirements = metadata.requires(distribution) or []
    except metadata.PackageNotFoundError:
        return None
    wanted = dependency.replace("-", "_").lower()
    for requirement in requirements:
        name = re.split(r"[\s@\[;<>=!~]", requirement, maxsplit=1)[0]
        match = _PINNED_COMMIT.search(requirement)
        if name.replace("-", "_").lower() == wanted and match:
            return match.group(1).lower()
    return None


def installed_commit(module: str, distribution: str | None = None) -> tuple[str | None, bool | None]:
    """Commit and dirty flag of the installed package: its git checkout, else pip's ``direct_url.json``.

    Args:
        module: Import name, e.g. ``"rio_hw"``.
        distribution: Distribution name when it differs from ``module``.
    """
    source = package_source(module, distribution)
    if source is not None and source["git_commit"]:
        return source["git_commit"].lower(), source["git_dirty"]
    try:
        direct_url = metadata.distribution(distribution or module).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None, None
    if not direct_url:
        return None, None
    try:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (ValueError, AttributeError):
        return None, None
    return (commit.lower() if isinstance(commit, str) else None), False


def check_rio_hw_pin() -> str | None:
    """Warn when the running rio_hw is not the commit RIO pins (or has local edits).

    A station may run a deliberate local fix of rio_hw, so this warns instead of refusing.

    Returns:
        The warning text, or ``None`` when the commit matches or cannot be determined.
    """
    pin = pinned_commit("rio", "rio_hw")
    commit, dirty = installed_commit("rio_hw")
    if pin is None or commit is None:
        return None
    if commit != pin:
        message = f"rio_hw is at commit {commit[:12]}, but RIO pins {pin[:12]}; behaviour may differ from the tested pair"
    elif dirty:
        message = f"rio_hw is at the pinned commit {pin[:12]} but has uncommitted changes"
    else:
        return None
    logger.warning(message)
    return message


def run_manifest(**extra) -> dict:
    """RIO, rio_hw, native KORD bridge, key libraries, interpreter, and host of this process.

    Args:
        **extra: Run-specific fields (checkpoint, instruction, ...) merged into the result.
    """
    bridge = find_spec("_kord_bridge")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "rio": package_source("rio"),
        "rio_hw": package_source("rio_hw"),
        "rio_hw_pin": pinned_commit("rio", "rio_hw"),
        "kord_bridge": None if bridge is None or bridge.origin is None else file_fingerprint(Path(bridge.origin)),
        "libraries": {name: _version(name) for name in LIBRARIES},
        **extra,
    }
