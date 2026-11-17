# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The 30 mm ArUco marker glued to the gripper, in the controller TCP frame.

The mount is the mean of the Jev hand-eye samples from 13 November 2026
(marker id 1, ``DICT_4X4_50``). The sticker normal points along the TCP's
negative x axis. The centre sat about 8 mm, 1 mm and 95 mm from the reported
TCP, with a sample scatter of roughly 6 mm. Corners are in ArUco order:
top-left, top-right, bottom-right, bottom-left, as you look at the black pattern.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

MARKER_ID = 1
MARKER_SIZE_M = 0.030
CORNER_NAMES = ("marker_tl", "marker_tr", "marker_br", "marker_bl")
# tcp_from_marker: marker axes expressed in the TCP frame, then the centre.
TCP_FROM_MARKER = np.array(
    [
        [-0.019307, -0.007291, -0.999787, -0.008323],
        [0.015795, -0.999851, 0.006987, -0.000892],
        [-0.999689, -0.015656, 0.019420, 0.094644],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def marker_corners() -> np.ndarray:
    """Four corners in the marker frame, ArUco order, shape ``(4, 3)``."""
    half = MARKER_SIZE_M / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=float,
    )


def corners_in_base(eef_pose6: np.ndarray) -> dict[str, np.ndarray]:
    """Corner positions in the robot base for a controller TCP pose6.

    The arm has to be standing at ``eef_pose6``. The sticker is assumed to still
    sit where the November fit measured it.
    """
    pose = np.asarray(eef_pose6, dtype=float).reshape(6)
    base_from_tcp = np.eye(4)
    base_from_tcp[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    base_from_tcp[:3, 3] = pose[:3]
    base_from_marker = base_from_tcp @ TCP_FROM_MARKER
    local = np.concatenate([marker_corners(), np.ones((4, 1))], axis=1)
    corners = (base_from_marker @ local.T).T[:, :3]
    return {name: corners[i] for i, name in enumerate(CORNER_NAMES)}
