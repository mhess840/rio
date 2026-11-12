# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Trajectory files and their language labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from loguru import logger

from .schema import INSTRUCTION_KEY

TRAJ_NAME_RE = re.compile(r"traj_(\d+)\.vla$", re.IGNORECASE)
RANGE_RE = re.compile(r"^(\d+)(?:\s*-\s*(\d+))?$")


def list_vla_files(path: Path) -> list[Path]:
    """Return sorted ``.vla`` files from a file or directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".vla":
            raise ValueError(f"Expected a .vla file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    files = sorted(path.glob("*.vla"))
    if not files:
        raise FileNotFoundError(f"No .vla files in {path}")
    return files


def traj_index(path: Path) -> int:
    """Parse ``traj_0012.vla`` → 12. Fallback: stem digits or -1."""
    match = TRAJ_NAME_RE.search(path.name)
    if match:
        return int(match.group(1))
    digits = re.search(r"(\d+)$", path.stem)
    if digits:
        return int(digits.group(1))
    raise ValueError(f"Cannot parse trajectory index from {path.name}")


def _parse_range_key(key: str) -> tuple[int, int]:
    match = RANGE_RE.match(str(key).strip())
    if not match:
        raise ValueError(f"Invalid task-map key {key!r}; use '12' or '0-19'")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    if end < start:
        raise ValueError(f"Invalid range {key!r}")
    return start, end


def _strip_task_value(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return text.strip()


def parse_task_map(path: Path) -> dict[int, str]:
    """Load ``{index: instruction}`` from JSON or ``start-end: text`` lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Task JSON must be an object of range → instruction")
        items = payload.items()
    else:
        items = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Task line must be 'range: instruction', got {line!r}")
            key, value = stripped.split(":", 1)
            items.append((key, value))

    mapping: dict[int, str] = {}
    for key, value in items:
        task = _strip_task_value(str(value))
        if not task:
            continue
        start, end = _parse_range_key(str(key))
        for idx in range(start, end + 1):
            mapping[idx] = task
    return mapping


def write_task_template(vla_files: list[Path], path: Path) -> Path:
    """Write a fill-in YAML listing every trajectory index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [traj_index(f) for f in vla_files]
    lo, hi = min(indices), max(indices)
    lines = [
        "# Language labels for Kassow π₀.₅ conversion.",
        "# Index matches traj_XXXX.vla (traj_0000 → 0).",
        "# Fill ranges, then run the converter with --tasks <this file>.",
        "#",
        "# Example:",
        "# 0-19: pick up the blue cylinder",
        "# 20-34: put the cylinder in the bin",
        "",
        f"{lo}-{hi}:",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote task template to {} ({} trajectories, indices {}-{})", path, len(vla_files), lo, hi)
    return path


def resolve_task(
    idx: int,
    data: dict[str, np.ndarray],
    task_map: dict[int, str],
    *,
    allow_missing_tasks: bool,
    missing_task: str,
) -> str | None:
    """Label of one trajectory: task map first, then the recorded instruction, then the fallback."""
    if idx in task_map:
        return task_map[idx]
    if INSTRUCTION_KEY in data:
        raw = data[INSTRUCTION_KEY][0]
        text = str(raw).strip() if raw is not None else ""
        if text:
            return text
    if allow_missing_tasks:
        return missing_task
    return None
