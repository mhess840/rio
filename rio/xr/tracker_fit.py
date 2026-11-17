# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit the headset frame to the robot base from a controller the gripper holds.

At every pose ``i`` the robot reports ``base_from_tcp`` (``B_i``) and the headset
reports ``xr_from_controller`` (``A_i``). With the controller rigid in the gripper,

    A_i = X @ B_i @ Y,   X = xr_from_base,   Y = tcp_from_controller.

Neither the grip nor the base placement has to be known in advance. Rotations
of the gripper between poses fix ``X``; spreading the poses fixes it well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MIN_POSES = 4
# Relative motions that turn less than this carry no rotation-axis information.
MIN_PAIR_ANGLE_RAD = np.radians(5.0)
# How a rotation residual weighs against a position residual: 1 rad counts as 0.1 m.
ROTATION_WEIGHT_M = 0.1


@dataclass
class TrackerFit:
    """Result of :func:`fit_tracker`.

    Attributes:
        xr_from_base: Robot base to the headset reference space.
        tcp_from_controller: Where the controller sits in the gripper.
        position_errors_m: Per pose, predicted vs tracked controller position.
        rotation_errors_rad: Per pose, predicted vs tracked controller orientation.
    """

    xr_from_base: np.ndarray
    tcp_from_controller: np.ndarray
    position_errors_m: np.ndarray
    rotation_errors_rad: np.ndarray

    @property
    def rms_m(self) -> float:
        return float(np.sqrt(np.mean(self.position_errors_m**2)))

    @property
    def max_m(self) -> float:
        return float(np.max(self.position_errors_m))


def _matrix(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = translation
    return out


def pose6_matrix(pose6: np.ndarray) -> np.ndarray:
    """A ``[x, y, z, rx, ry, rz]`` rotation-vector pose as a 4 x 4 matrix."""
    pose = np.asarray(pose6, dtype=float).reshape(6)
    return _matrix(pose[3:], pose[:3])


def _invert(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return out


def _initial_x(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Closed-form ``X`` from relative motions: ``A_i A_j^-1 = X (B_i B_j^-1) X^-1``."""
    axes_a, axes_b, motions = [], [], []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            ma = a[i] @ _invert(a[j])
            mb = b[i] @ _invert(b[j])
            va = Rotation.from_matrix(ma[:3, :3]).as_rotvec()
            vb = Rotation.from_matrix(mb[:3, :3]).as_rotvec()
            if min(np.linalg.norm(va), np.linalg.norm(vb)) < MIN_PAIR_ANGLE_RAD:
                continue
            axes_a.append(va)
            axes_b.append(vb)
            motions.append((ma, mb))
    if len(axes_a) < 2:
        raise ValueError("the poses need rotations of at least 5 deg between them")
    axes_b_arr = np.asarray(axes_b)
    unit = axes_b_arr / np.linalg.norm(axes_b_arr, axis=1, keepdims=True)
    spread = np.linalg.svd(unit, compute_uv=False)
    if spread[1] < 0.2 * spread[0]:
        raise ValueError("the gripper turned about one axis only; add tilts about another axis")
    u, _, vt = np.linalg.svd(axes_b_arr.T @ np.asarray(axes_a))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ flip @ u.T
    # (R_a - I) t_X = R_X t_b - t_a
    lhs = np.concatenate([ma[:3, :3] - np.eye(3) for ma, _ in motions])
    rhs = np.concatenate([rotation @ mb[:3, 3] - ma[:3, 3] for ma, mb in motions])
    translation = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _initial_y(a: list[np.ndarray], b: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    x_inv = _invert(x)
    guesses = [_invert(bi) @ x_inv @ ai for ai, bi in zip(a, b, strict=True)]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(np.stack([g[:3, :3] for g in guesses])).mean().as_matrix()
    out[:3, 3] = np.mean([g[:3, 3] for g in guesses], axis=0)
    return out


def _errors(a: list[np.ndarray], b: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = [], []
    for ai, bi in zip(a, b, strict=True):
        predicted = x @ bi @ y
        position.append(np.linalg.norm(predicted[:3, 3] - ai[:3, 3]))
        rotation.append(Rotation.from_matrix(predicted[:3, :3].T @ ai[:3, :3]).magnitude())
    return np.asarray(position), np.asarray(rotation)


def fit_tracker(base_from_tcp: list[np.ndarray], xr_from_controller: list[np.ndarray]) -> TrackerFit:
    """Solve ``A_i = X B_i Y`` for ``X = xr_from_base`` and ``Y = tcp_from_controller``.

    A closed-form start from the relative motions is refined by least squares
    on the controller position and orientation at every pose.

    Args:
        base_from_tcp: 4 x 4 gripper poses reported by the robot.
        xr_from_controller: 4 x 4 controller poses tracked by the headset, same order.

    Returns:
        The fit and its per-pose residuals.

    Raises:
        ValueError: Fewer than :data:`MIN_POSES` pairs, or rotations about a single axis only.
    """
    b = [np.asarray(m, dtype=float).reshape(4, 4) for m in base_from_tcp]
    a = [np.asarray(m, dtype=float).reshape(4, 4) for m in xr_from_controller]
    if len(a) != len(b) or len(a) < MIN_POSES:
        raise ValueError(f"the fit needs at least {MIN_POSES} pose pairs")
    x0 = _initial_x(a, b)
    y0 = _initial_y(a, b, x0)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _matrix(params[0:3], params[3:6]), _matrix(params[6:9], params[9:12])

    def residuals(params: np.ndarray) -> np.ndarray:
        x, y = unpack(params)
        out = []
        for ai, bi in zip(a, b, strict=True):
            predicted = x @ bi @ y
            out.append(predicted[:3, 3] - ai[:3, 3])
            out.append(Rotation.from_matrix(predicted[:3, :3].T @ ai[:3, :3]).as_rotvec() * ROTATION_WEIGHT_M)
        return np.concatenate(out)

    start = np.concatenate(
        [
            Rotation.from_matrix(x0[:3, :3]).as_rotvec(),
            x0[:3, 3],
            Rotation.from_matrix(y0[:3, :3]).as_rotvec(),
            y0[:3, 3],
        ]
    )
    solved = least_squares(residuals, start, loss="soft_l1", f_scale=0.005)
    x, y = unpack(solved.x)
    position, rotation = _errors(a, b, x, y)
    return TrackerFit(x, y, position, rotation)
