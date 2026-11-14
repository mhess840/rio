# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Versions that identify a run, written next to recordings and rollout diagnostics.

Kept outside ``rio.data`` so that building an environment does not import the recorder and formatters.
"""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from loguru import logger

LIBRARIES = ("numpy", "torch", "lerobot", "robodm", "transformers")


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
    """Version, location, and (for a git checkout) commit, branch, dirty flag, and lag behind upstream.

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
    upstream = _git(folder, "rev-parse", "--abbrev-ref", "@{upstream}") if commit else None
    behind = _git(folder, "rev-list", "--count", "HEAD..@{upstream}") if upstream else None
    return {
        "version": _version(distribution or module),
        "path": str(folder),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_branch": _git(folder, "rev-parse", "--abbrev-ref", "HEAD") if commit else None,
        "git_upstream": upstream,
        "git_behind": int(behind) if behind and behind.isdigit() else None,
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


def check_rio_hw() -> str | None:
    """Warn when the local rio_hw checkout has uncommitted changes or is behind its upstream branch.

    RIO always runs the sibling ``../rio-hw`` checkout as it is, so the newest commit
    of that branch is the intended version. ``behind`` uses the last ``git fetch``;
    nothing is fetched here.

    Returns:
        The warning text, or ``None`` when the checkout is clean and up to date or unknown.
    """
    source = package_source("rio_hw")
    if source is None or not source["git_commit"]:
        return None
    problems = []
    if source.get("git_behind"):
        problems.append(f"{source['git_behind']} commit(s) behind {source.get('git_upstream')} (git -C ../rio-hw pull)")
    if source["git_dirty"]:
        problems.append("uncommitted changes (recordings name a commit that does not match the code)")
    if not problems:
        return None
    message = f"rio_hw at {source['git_commit'][:12]} on {source.get('git_branch')}: " + "; ".join(problems)
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
        "kord_bridge": None if bridge is None or bridge.origin is None else file_fingerprint(Path(bridge.origin)),
        "libraries": {name: _version(name) for name in LIBRARIES},
        **extra,
    }
