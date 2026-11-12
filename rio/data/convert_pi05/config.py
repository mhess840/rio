# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Conversion settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rio.data.dagger import SEGMENT_MODES


@dataclass
class ConvertPi05Cfg:
    """CLI/config for Kassow ``.vla`` → LeRobot v3 conversion."""

    input: Path | None = None
    output: Path | None = None
    tasks: Path | None = None
    repo_id: str = "local/kassow_pi05"
    # ``tcp``: action = xyz+rotvec+gripper (7). ``joints``: next-frame q + gripper (8).
    action_space: str = "tcp"
    src_fps: int = 100
    dst_fps: int = 20
    robot_type: str = "kassow"
    binarize_gripper: bool = True
    gripper_close: float = 0.0
    gripper_open: float = 0.7
    gripper_threshold: float | None = None
    gripper_state_source: str = "measured"
    min_frames: int = 16
    unwrap_rotvec: bool = True
    trim_idle: bool = True
    idle_xyz_m: float = 0.01
    clean: bool = False
    dry_run: bool = False
    write_tasks_template: Path | None = None
    allow_missing_tasks: bool = False
    missing_task: str = "unlabeled"
    vcodec: str = "h264"
    image_writer_threads: int = 4
    # Parallel episode encodes. Same codec settings as workers=1; only the schedule changes.
    workers: int = 1
    segments: str = "whole"
    include_autonomous: bool = False
    pre_takeover_cut_s: float = 1.0
    min_segment_s: float = 1.0
    trim_operator_pauses: bool = True
    max_pause_s: float = 0.3

    def __post_init__(self):
        space = str(self.action_space).strip().lower()
        if space not in {"tcp", "joints"}:
            raise ValueError(f"action_space must be 'tcp' or 'joints', got {self.action_space!r}")
        self.action_space = space
        if self.gripper_state_source not in {"measured", "commanded"}:
            raise ValueError("gripper_state_source must be 'measured' or 'commanded'")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.segments not in SEGMENT_MODES:
            raise ValueError(f"segments must be one of {SEGMENT_MODES}")
        for name in ("pre_takeover_cut_s", "min_segment_s", "max_pause_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
