# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Layers of the Kassow cell for the headset. Each function returns one :class:`rio.xr.Layer`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rio.apps.kassow.experiments import PACKAGED_HOMES_DIR, homes_dir
from rio.apps.kassow.graph.table_map import TableMap
from rio.xr import Layer, Lines, Mesh, Points
from rio.xr.scene import axes, circle, fan, segments

TELEOP_MAX = (1.0, 0.6, 0.1, 0.6)
TELEOP_MIN = (1.0, 0.25, 0.25, 0.8)
TABLE = (0.2, 0.8, 1.0, 0.25)
TABLE_EDGE = (0.2, 0.8, 1.0, 1.0)
SLICE_COLORS = ((0.2, 1.0, 0.4, 0.9), (0.3, 0.7, 1.0, 0.9), (0.9, 0.5, 1.0, 0.9), (1.0, 0.9, 0.3, 0.9))


def saved_pose_tcps() -> dict[str, np.ndarray]:
    """TCP position of every saved pose file, captured ones over packaged ones of the same name."""
    files = {path.name: path for path in sorted(PACKAGED_HOMES_DIR.glob("*.json"))}
    if homes_dir().is_dir():
        files.update({path.name: path for path in sorted(homes_dir().glob("*.json"))})
    tcps = {}
    for name, path in files.items():
        try:
            pose = np.asarray(json.loads(Path(path).read_text())["eef_pose"], dtype=float).reshape(-1)
        except (KeyError, ValueError, TypeError):
            continue
        if pose.size >= 3 and np.all(np.isfinite(pose[:3])):
            tcps[Path(name).stem] = pose[:3]
    return tcps


def base_layer() -> Layer:
    return Layer("base", "Robot base axes", tuple(axes(0.25)))


def poses_layer(tcps: dict[str, np.ndarray]) -> Layer:
    """Saved-pose TCPs; with the arm in one of them, its dot should sit on the gripper."""
    points = np.array(list(tcps.values()), dtype=float).reshape(-1, 3)
    return Layer("poses", "Saved poses (TCP)", (Points(points, (1.0, 1.0, 1.0, 1.0), 0.02),))


def teleop_limits_layer(min_radius: float, max_radius: float, min_z: float) -> Layer:
    """Shells the Vive teleop clamps the TCP to: radius from the base origin and the z floor."""
    items: list[Lines] = []
    for radius, color in ((max_radius, TELEOP_MAX), (min_radius, TELEOP_MIN)):
        if radius <= min_z:
            continue
        lines = []
        for z in np.linspace(min_z, radius * 0.95, 6):
            lines.append(segments(circle(np.array([0.0, 0.0, z]), math.sqrt(radius**2 - z**2)), closed=True))
        start = math.asin(min_z / radius)
        elevation = np.linspace(start, math.pi / 2, 16)
        for azimuth in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
            arc = np.stack(
                [
                    radius * np.cos(elevation) * math.cos(azimuth),
                    radius * np.cos(elevation) * math.sin(azimuth),
                    radius * np.sin(elevation),
                ],
                axis=1,
            )
            lines.append(segments(arc))
        items.append(Lines(np.concatenate(lines), color))
    return Layer("teleop_limits", "Teleop limits (radius, z floor)", tuple(items), visible=False)


def table_layer(table: TableMap) -> Layer:
    """The calibrated area where locate and the target skills are allowed, at grasp height."""
    corners = np.array([[x, y, table.z_at(x, y)] for x, y in table.hull], dtype=float)
    return Layer(
        "table",
        "Calibrated table area (grasp height)",
        (Mesh(corners, fan(corners), TABLE), Lines(segments(corners, closed=True), TABLE_EDGE)),
    )


def reach_layer(slices: dict[float, np.ndarray], step_m: float) -> Layer:
    """One dot per reachable grid cell, one color per height."""
    items = tuple(
        Points(np.asarray(points, dtype=float).reshape(-1, 3), SLICE_COLORS[i % len(SLICE_COLORS)], 0.6 * step_m)
        for i, (_, points) in enumerate(sorted(slices.items()))
        if len(points)
    )
    heights = ", ".join(f"{z:.2f}" for z in sorted(slices))
    return Layer("reach", f"Reachable, gripper down (z {heights} m)"[:80], items)
