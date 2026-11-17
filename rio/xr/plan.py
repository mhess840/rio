# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The gripper-tip polyline a rollout still has to execute, as one replaceable scene layer.

Actions are absolute flange poses ``[x, y, z, rx, ry, rz, gripper]`` (rotation vector),
the same frame the Kassow reports as ``eef_pose``. The fingertips sit on the flange z
axis, :data:`GRIPPER_TIP_Z_M` away, closer when the AG95 opens.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from .publisher import XrPublisher
from .scene import Layer, Lines, Points

LAYER_NAME = "vla_plan"
OPEN = (0.3, 0.85, 1.0, 1.0)
CLOSED = (1.0, 0.45, 0.15, 1.0)
GRIPPER_CLOSED_BELOW = 0.35
GOAL = (1.0, 1.0, 1.0, 1.0)
# AG95 datasheet, flange to fingertip: 200 mm fully closed, 185 mm fully open.
GRIPPER_TIP_Z_M = (0.200, 0.185)


def gripper_tips(poses: np.ndarray, *, opening: float | None = None) -> np.ndarray:
    """Fingertip positions for flange poses, shape ``(N, 3)``.

    Args:
        poses: ``(N, D)`` flange poses; columns 3 to 5 are the rotation vector, column 6
            the gripper opening (0 closed, 1 open) when present.
        opening: Overrides column 6 for every row, e.g. when only the arm is read.

    Returns:
        Flange position plus the flange z axis times the tip distance for that opening.
        Rows without a rotation are returned unchanged.
    """
    poses = np.asarray(poses, dtype=float).reshape(len(poses), -1)
    if poses.shape[1] < 6:
        return poses[:, :3].copy()
    if opening is not None:
        grip = np.full(len(poses), float(opening))
    elif poses.shape[1] >= 7:
        grip = poses[:, 6]
    else:
        grip = np.zeros(len(poses))
    closed, opened = GRIPPER_TIP_Z_M
    reach = closed + (opened - closed) * np.clip(grip, 0.0, 1.0)
    z_axis = Rotation.from_rotvec(poses[:, 3:6]).as_matrix()[:, :, 2]
    return poses[:, :3] + z_axis * reach[:, None]


class _Sender:
    """One background publisher, created on the first plan and skipped when no server is up."""

    publisher: XrPublisher | None = None
    unavailable = False


def vla_plan_layer(actions: np.ndarray, *, skip: int = 0) -> Layer | None:
    """Remaining fingertip waypoints as a line. Gripper colours each segment.

    Args:
        actions: ``(N, D)`` with position in the first three columns. Column 6, when
            present, is the gripper; values below :data:`GRIPPER_CLOSED_BELOW` draw orange.
        skip: Leading rows already executed.

    Returns:
        The ``vla_plan`` layer, or None when fewer than two finite positions remain.
    """
    chunk = np.asarray(actions, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] < 3 or skip < 0:
        return None
    chunk = chunk[int(skip) :]
    if len(chunk) < 2 or not np.all(np.isfinite(chunk[:, : min(chunk.shape[1], 6)])):
        return None
    xyz = gripper_tips(chunk)
    grip = chunk[:, 6] if chunk.shape[1] >= 7 else np.ones(len(chunk))
    open_rows: list[np.ndarray] = []
    closed_rows: list[np.ndarray] = []
    for index in range(len(xyz) - 1):
        (closed_rows if float(grip[index]) < GRIPPER_CLOSED_BELOW else open_rows).append(xyz[index : index + 2])
    items: list[Lines | Points] = []
    if open_rows:
        items.append(Lines(np.concatenate(open_rows), OPEN))
    if closed_rows:
        items.append(Lines(np.concatenate(closed_rows), CLOSED))
    items.append(Points(xyz[-1:], GOAL, 0.02))
    return Layer(LAYER_NAME, "VLA plan (remaining fingertip path)", tuple(items))


def publish_vla_plan(actions: np.ndarray) -> None:
    """Send :func:`vla_plan_layer` to the local XR server. Missing server: log once and skip."""
    if _Sender.unavailable:
        return
    layer = vla_plan_layer(actions)
    if layer is None:
        return
    try:
        if _Sender.publisher is None:
            _Sender.publisher = XrPublisher()
        _Sender.publisher.publish(layer)
    except (FileNotFoundError, OSError, ValueError) as error:
        _Sender.unavailable = True
        logger.info("VLA plan stays off the headset: {}", error)
