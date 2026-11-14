# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve the Kassow workspace to a headset. Display only: nothing here moves the arm.

    uv run python -m rio.apps.kassow.xr

Layers: robot base axes, saved-pose TCPs, teleop limits, the calibrated table area
(if a table map exists), and the reachable area with the gripper pointing down.
Other processes add layers with :class:`rio.xr.XrPublisher`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from rio.apps.kassow.experiments import home_file
from rio.apps.kassow.graph.table_map import TableMap, table_map_path
from rio.apps.kassow.stations.kassow_vive import KassowViveStation
from rio.workspace import experiments_root
from rio.xr import XrHub, XrServer
from rio.xr.auth import token_path, write_token

from .layers import base_layer, poses_layer, reach_layer, saved_pose_tcps, table_layer, teleop_limits_layer
from .reach import reach_slices


@dataclass
class Args:
    """Show the Kassow workspace in a headset over WebXR. Nothing here moves the arm."""

    port: int = 8770
    """Local port; the headset reaches it through ``adb reverse``."""
    pose: str = "home_pick.json"
    """Saved pose whose orientation and joints seed the reach slices."""
    heights_m: tuple[float, ...] = ()
    """Base-frame heights of the reach slices. Empty: table grasp height +0/+0.1/+0.2 m, else pose height -0.1/0/+0.1 m."""
    step_m: float = 0.03
    """Grid spacing of the reach slices."""
    reach: bool = True
    """Compute the reach slices (needs pinocchio; cached after the first run)."""
    table: Path | None = None
    """Table map. Unset: experiments/graph/calibration/table/table_map.json, skipped if missing."""
    min_radius: float = KassowViveStation.min_radius
    max_radius: float = KassowViveStation.max_radius
    min_z: float = KassowViveStation.min_z


def load_table(path: Path) -> TableMap | None:
    if not path.is_file():
        return None
    try:
        return TableMap.load(path)
    except ValueError as error:
        logger.warning("Table map not shown: {}", error)
        return None


def slice_heights(args: Args, table: TableMap | None, pose: np.ndarray) -> tuple[float, ...]:
    if args.heights_m:
        return tuple(args.heights_m)
    if table is not None:
        base = table.z_at(float(pose[0]), float(pose[1]))
        return (base, base + 0.10, base + 0.20)
    return (float(pose[2]) - 0.10, float(pose[2]), float(pose[2]) + 0.10)


def add_reach(hub: XrHub, args: Args, table: TableMap | None) -> None:
    """Compute the reach layer and publish it; the page shows it as soon as it is ready."""
    try:
        from rio_hw.robots.kassow_kinematics import KassowKinematics

        kin = KassowKinematics()
    except ImportError as error:
        logger.warning("Reach layer skipped: {}", error)
        return
    snapshot = json.loads(home_file(args.pose).read_text())
    pose = np.asarray(snapshot["eef_pose"], dtype=float)
    rotvec = table.rotvec if table is not None else pose[3:6]
    slices = reach_slices(
        kin,
        slice_heights(args, table, pose),
        rotvec,
        seed_xy=pose[:2],
        seed_q=np.asarray(snapshot["joint_q"], dtype=float),
        step_m=args.step_m,
        max_radius=args.max_radius,
        cache_dir=experiments_root() / "xr",
        model_id=str(kin.urdf_path),
    )
    hub.put(reach_layer(slices, args.step_m))


def main(args: Args, stop: threading.Event | None = None) -> XrServer:
    """Build the layers, serve them, and block until Ctrl-C or ``stop``."""
    hub = XrHub()
    tcps = saved_pose_tcps()
    hub.set_anchors(tcps)
    hub.put(base_layer())
    hub.put(poses_layer(tcps))
    hub.put(teleop_limits_layer(args.min_radius, args.max_radius, args.min_z))
    table = load_table(args.table or table_map_path())
    if table is not None:
        hub.put(table_layer(table))
    server = XrServer(hub, port=args.port, token=write_token(token_path()))
    server.start()
    print(f"Desktop preview: {server.url}")
    print(f"Quest over USB:  adb reverse tcp:{server.port} tcp:{server.port}, then open http://localhost:{server.port}")
    print("Display only, not a safety function. Ctrl-C stops.")
    if args.reach:
        add_reach(hub, args, table)
    waiter = stop or threading.Event()
    try:
        waiter.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return server


def cli() -> None:
    main(tyro.cli(Args))
