# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Folder layout and writers of the per-rollout diagnostics (plans, commands, first frames)."""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw import time

from rio.data.diagnostics import AsyncDiagnostics
from rio.schema import ArmState
from rio.workspace import experiments_root


def default_diag_dir() -> Path:
    """``<experiments workspace>/pi05_diag``."""
    return experiments_root() / "pi05_diag"


def dump_live_cameras(obs: dict, dest: Path) -> None:
    """Write the first live RGB frames so they can be compared to the train videos."""
    dest.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for key in ("camera_1", "camera_2"):
        img = obs.get(key)
        if img is None:
            logger.warning("Live camera {} is missing from the observation", key)
            continue
        arr = np.asarray(img)
        mean_rgb = np.asarray(arr, dtype=np.float32).mean(axis=(0, 1)) if arr.ndim == 3 else arr.mean()
        logger.info(
            "Live {} shape={} dtype={} min={} max={} meanRGB={}",
            key,
            arr.shape,
            arr.dtype,
            arr.min() if arr.size else None,
            arr.max() if arr.size else None,
            mean_rgb,
        )
        if arr.ndim == 3 and arr.shape[-1] == 3:
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dest / f"live_{key}.png")
    logger.info("Wrote live camera frames to {}", dest)


def create_diagnostic_session(root: Path, metadata: dict) -> Path:
    """Allocate a unique rollout folder without overwriting earlier CSV or chunk files."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="session_", dir=root))
    (session / "session.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return session


def write_chunk_dump(
    dest: Path,
    idx: int,
    leftover: np.ndarray,
    eef: np.ndarray,
    *,
    delay: int,
    qsize: int,
    latency_s: float,
) -> None:
    """Save one merged leftover chunk plus a CSV row (model plan before IK)."""
    dest.mkdir(parents=True, exist_ok=True)
    chunk_dir = dest / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    leftover = np.asarray(leftover, dtype=np.float32)
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    np.save(chunk_dir / f"chunk_{idx:04d}.npy", leftover)
    first = leftover[0, :3] if leftover.ndim == 2 and leftover.shape[0] else np.zeros(3)
    last = leftover[-1, :3] if leftover.ndim == 2 and leftover.shape[0] else first
    span = float(np.linalg.norm(last - first))
    csv_path = dest / "chunks.csv"
    new_file = not csv_path.is_file()
    with csv_path.open("a") as f:
        if new_file:
            f.write("idx,delay,qsize,latency_s,eef_x,eef_y,eef_z,first_x,first_y,first_z,last_x,last_y,last_z,span_m\n")
        f.write(
            f"{idx},{delay},{qsize},{latency_s:.4f},"
            f"{float(eef[0]):.6f},{float(eef[1]):.6f},{float(eef[2]):.6f},"
            f"{float(first[0]):.6f},{float(first[1]):.6f},{float(first[2]):.6f},"
            f"{float(last[0]):.6f},{float(last[1]):.6f},{float(last[2]):.6f},"
            f"{span:.6f}\n"
        )


def append_cmd_eef(
    dest: Path,
    t: float,
    cmd: np.ndarray,
    eef: np.ndarray,
    remaining: int,
    chunk_idx: int,
    *,
    arm_state: ArmState | None = None,
    joint_serve: bool = False,
) -> None:
    """Append one 100 Hz command vs measured TCP row."""
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "cmd_vs_eef.csv"
    cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    state = arm_state or {}
    alarm = int(state.get("alarm_code", 0) or 0)
    state_age = time.now() - float(state["timestamp"]) if "timestamp" in state else float("nan")
    if joint_serve:
        header = (
            "t,"
            + ",".join(f"cmd_j{index + 1}" for index in range(7))
            + ","
            + ",".join(f"eef_j{index + 1}" for index in range(7))
            + ",err_joint_rad,remaining,chunk_idx,cmd_gripper,alarm_code,state_age_s\n"
        )
        error = float(np.linalg.norm(cmd[:7] - eef[:7]))
        values = [t, *cmd[:7], *eef[:7], error, remaining, chunk_idx, cmd[-1], alarm, state_age]
    else:
        position_error = float(np.linalg.norm(cmd[:3] - eef[:3]))
        difference = cast(
            st.Rotation,
            st.Rotation.from_rotvec(cmd[3:6].astype(float)) * st.Rotation.from_rotvec(eef[3:6].astype(float)).inv(),
        )
        rotation_error = float(difference.magnitude())
        header = (
            "t,cmd_x,cmd_y,cmd_z,eef_x,eef_y,eef_z,err_m,remaining,chunk_idx,"
            "cmd_rx,cmd_ry,cmd_rz,eef_rx,eef_ry,eef_rz,err_rot_rad,cmd_gripper,alarm_code,state_age_s\n"
        )
        values = [
            t,
            *cmd[:3],
            *eef[:3],
            position_error,
            remaining,
            chunk_idx,
            *cmd[3:6],
            *eef[3:6],
            rotation_error,
            cmd[-1],
            alarm,
            state_age,
        ]
    new_file = not csv_path.is_file()
    with csv_path.open("a") as f:
        if new_file:
            f.write(header)
        f.write(",".join(f"{float(value):.6f}" for value in values) + "\n")


class RolloutDiagnostics:
    """One session folder per rollout; every write runs on the ``AsyncDiagnostics`` worker.

    Args:
        writer: Background writer shared by all rollouts of one ``policy_loop``.
        root: Parent folder of the session folders.
        metadata: Written to ``session.json``.
        dump_cameras: Writer of the first camera frames.
    """

    def __init__(
        self, writer: AsyncDiagnostics, root: Path, metadata: dict, dump_cameras: Callable[[dict, Path], None]
    ) -> None:
        self.writer = writer
        self.dir = create_diagnostic_session(root, metadata)
        self._dump_cameras = dump_cameras
        self._cameras_dumped = False

    def cameras(self, obs: dict) -> None:
        """Keep the first observation's frames of this rollout."""
        if not self._cameras_dumped:
            self.writer.submit(self._dump_cameras, obs, self.dir / "cameras")
            self._cameras_dumped = True

    def chunk(self, idx: int, leftover: np.ndarray, eef: np.ndarray, *, delay: int, qsize: int, latency_s: float) -> None:
        """Queue one merged plan for ``chunks/`` and ``chunks.csv``."""
        self.writer.submit(write_chunk_dump, self.dir, idx, leftover, eef, delay=delay, qsize=qsize, latency_s=latency_s)

    def command(
        self,
        t: float,
        cmd: np.ndarray,
        measured: np.ndarray,
        remaining: int,
        chunk_idx: int,
        arm_state: ArmState,
        joint_serve: bool,
    ) -> None:
        """Queue one ``cmd_vs_eef.csv`` row."""
        self.writer.submit(
            append_cmd_eef, self.dir, t, cmd, measured, remaining, chunk_idx, arm_state=arm_state, joint_serve=joint_serve
        )
