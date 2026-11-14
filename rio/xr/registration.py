# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid transform from the robot base frame to the headset frame, from touched points."""

from __future__ import annotations

import numpy as np

MIN_POINTS = 3
# Smallest spread (second singular value of the centred points) that still fixes a rotation.
MIN_SPREAD_M = 0.05


def fit_rigid(base: np.ndarray, xr: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch fit of ``xr ~ R @ base + t`` without scale.

    Args:
        base: ``N x 3`` points in the robot base frame.
        xr: The same points as touched in the headset frame.

    Returns:
        The 4 x 4 ``xr_from_base`` matrix and the RMS residual in metres.

    Raises:
        ValueError: Fewer than three pairs, mismatched shapes, or points too close to a line.
    """
    base = np.asarray(base, dtype=float).reshape(-1, 3)
    xr = np.asarray(xr, dtype=float).reshape(-1, 3)
    if len(base) != len(xr) or len(base) < MIN_POINTS:
        raise ValueError(f"registration needs at least {MIN_POINTS} point pairs")
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(xr))):
        raise ValueError("registration points must be finite")
    base_mean, xr_mean = base.mean(axis=0), xr.mean(axis=0)
    a, b = base - base_mean, xr - xr_mean
    spread = np.linalg.svd(a, compute_uv=False)
    if spread[1] < MIN_SPREAD_M:
        raise ValueError("registration points lie (nearly) on a line; spread them out")
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ flip @ u.T
    translation = xr_mean - rotation @ base_mean
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    residual = xr - (base @ rotation.T + translation)
    return matrix, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
