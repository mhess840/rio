# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Objects on the table from the Jev camera's depth: centre, height and yaw.

The depth image is aligned to the colour image, so a SAM box in colour pixels
selects the same depth pixels. With the hand-eye pose of the camera every pixel
becomes a point in the robot base frame. Around the box lies bare table: its
median height is the local table height. Inside the box, points clearly above
it belong to the object. No mask is needed for an object standing alone; a
mask (later) only sharpens the selection in clutter.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .handeye import HandEye

MIN_HEIGHT_M = 0.008
"""Points this far above the local table belong to the object (D435i noise at 0.86 m is a few mm)."""
MIN_POINTS = 25
GROW = 0.6
"""The table ring extends this fraction of the box size beyond the box on every side."""
ROUND_FILL = 0.87
"""Hull area / rectangle area below this means round (circle ≈ 0.79, square ≈ 1.0): no yaw."""
SQUARE_RATIO = 0.15
"""Sides within this relative difference: a square, yaw repeats every 90°."""


@dataclass(frozen=True)
class Measured:
    """One object measured in depth, base frame, metres and degrees.

    Attributes:
        x: Centre of the top surface.
        y: Centre of the top surface.
        top_z: Top of the object (90th percentile of its points).
        table_z: Local table height around it.
        height_m: ``top_z - table_z``.
        length_m: Long side of the footprint.
        width_m: Short side of the footprint.
        yaw_deg: Direction of the long side about base z, in [0, period); None for round objects.
        period_deg: 180, or 90 for a square footprint.
        n_points: Object points used.
    """

    x: float
    y: float
    top_z: float
    table_z: float
    height_m: float
    length_m: float
    width_m: float
    yaw_deg: float | None
    period_deg: float
    n_points: int

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def back_project(depth_m: np.ndarray, us: np.ndarray, vs: np.ndarray, handeye: HandEye) -> np.ndarray:
    """Base-frame points (N x 3) of pixels ``(us, vs)`` with their depth (camera z, metres)."""
    intr = handeye.intrinsics
    pixels = np.stack([us, vs], axis=1).astype(np.float64).reshape(-1, 1, 2)
    normalised = cv2.undistortPoints(pixels, intr.matrix, np.asarray(intr.dist, dtype=np.float64)).reshape(-1, 2)
    z = depth_m[vs, us].astype(np.float64)
    camera = np.column_stack([normalised * z[:, None], z])
    transform = handeye.transform
    return camera @ transform[:3, :3].T + transform[:3, 3]


def measure(
    depth_m: np.ndarray,
    box: tuple[float, float, float, float],
    handeye: HandEye,
    *,
    min_height_m: float = MIN_HEIGHT_M,
    min_points: int = MIN_POINTS,
) -> Measured | None:
    """Measure the object inside a colour-pixel box ``x1, y1, x2, y2`` (inclusive).

    Returns:
        The measurement, or None when the ring has too little table or the box
        too few points above it (e.g. depth holes, a flat object).
    """
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in box)
    gx, gy = max(3, int((x2 - x1 + 1) * GROW)), max(3, int((y2 - y1 + 1) * GROW))
    rx1, ry1, rx2, ry2 = max(0, x1 - gx), max(0, y1 - gy), min(w - 1, x2 + gx), min(h - 1, y2 + gy)
    vs, us = np.mgrid[ry1 : ry2 + 1, rx1 : rx2 + 1]
    us, vs = us.ravel(), vs.ravel()
    valid = depth_m[vs, us] > 0
    us, vs = us[valid], vs[valid]
    if us.size == 0:
        return None
    points = back_project(depth_m, us, vs, handeye)
    mx, my = max(2, (x2 - x1) // 10), max(2, (y2 - y1) // 10)
    inside = (us >= x1) & (us <= x2) & (vs >= y1) & (vs <= y2)
    near = (us >= x1 - mx) & (us <= x2 + mx) & (vs >= y1 - my) & (vs <= y2 + my)
    ring = ~near
    if ring.sum() < min_points:
        return None
    table_z = float(np.median(points[ring, 2]))
    above = inside & (points[:, 2] > table_z + min_height_m)
    if above.sum() < min_points:
        return None
    obj = points[above]
    top_z = float(np.percentile(obj[:, 2], 90))
    height = top_z - table_z
    top = obj[obj[:, 2] > table_z + 0.5 * height]
    if len(top) < min_points // 2:
        top = obj
    x, y = (float(v) for v in top[:, :2].mean(axis=0))
    # The top face is clean; side faces and edge pixels blur the outline.
    length, width, yaw, period = footprint(top[:, :2])
    return Measured(x, y, top_z, table_z, height, length, width, yaw, period, len(obj))


def footprint(xy: np.ndarray) -> tuple[float, float, float | None, float]:
    """Long side, short side, yaw of the long side (None if round) and the yaw period of 2-D points."""
    pts = (xy * 1000.0).astype(np.float32)
    (_, _), (a, b), angle = cv2.minAreaRect(pts)
    length, width = max(a, b) / 1000.0, min(a, b) / 1000.0
    if length <= 0:
        return 0.0, 0.0, None, 180.0
    rect_area = a * b
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    if rect_area <= 0 or hull_area / rect_area < ROUND_FILL:
        return length, width, None, 180.0
    long_angle = angle if a >= b else angle + 90.0
    square = (length - width) / length < SQUARE_RATIO
    period = 90.0 if square else 180.0
    return length, width, long_angle % period, period


TOP_MIN_SATURATION = 90
"""Coloured object pixels; the table and its holes are grey."""
SQUARE_SIDES = 0.7
"""Top-face rectangle sides at least this similar: square (yaw period 90)."""


def top_face_yaw(
    image_bgr: np.ndarray,
    box: tuple[float, float, float, float],
    handeye: HandEye,
    top_z: float,
    *,
    margin_px: int = 6,
) -> dict | None:
    """Yaw of a coloured object's top face from the colour image, in the base frame.

    Depth is too smeared at the edges of a 4 cm cube to give its yaw. In colour the
    top face is the brighter part of the object (the side faces are in shade):
    coloured pixels in the box, split by Otsu on brightness, largest bright blob,
    minimum-area rectangle, corners projected onto the plane ``z = top_z``. A side
    face that leaks in makes the rectangle longer but does not turn it.

    Returns:
        ``yaw_deg`` (in [0, period); None for a round face), ``period_deg``, ``sides_m``,
        ``fill`` (blob / rectangle) and ``pixels``; or None when the box holds no coloured blob.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in box)
    cx1, cy1 = max(0, x1 - margin_px), max(0, y1 - margin_px)
    crop = image_bgr[cy1 : min(h, y2 + margin_px + 1), cx1 : min(w, x2 + margin_px + 1)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    coloured = hsv[..., 1] > TOP_MIN_SATURATION
    if coloured.sum() < 20:
        return None
    value = hsv[..., 2][coloured].reshape(-1, 1).astype(np.uint8)
    threshold, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (coloured & (hsv[..., 2] > threshold)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if count < 2:
        return None
    blob = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == blob)
    if len(xs) < 20:
        return None
    rect = cv2.minAreaRect(np.column_stack([xs + cx1, ys + cy1]).astype(np.float32))
    fill = len(xs) / max(1.0, rect[1][0] * rect[1][1])
    corners = cv2.boxPoints(rect)
    hits = [handeye.pixel_to_plane(float(u), float(v), top_z) for u, v in corners]
    if any(hit is None for hit in hits):
        return None
    xy = np.array([hit[:2] for hit in hits if hit is not None])
    a, b = xy[1] - xy[0], xy[2] - xy[1]
    sides = sorted((float(np.linalg.norm(a)), float(np.linalg.norm(b))))
    period = 90.0 if sides[0] >= SQUARE_SIDES * sides[1] else 180.0
    long_edge = a if np.linalg.norm(a) >= np.linalg.norm(b) else b
    yaw: float | None = math.degrees(math.atan2(long_edge[1], long_edge[0])) % period
    if fill < ROUND_FILL:  # a disc fills about 79 % of its rectangle
        yaw, period = None, 180.0
    return {
        "yaw_deg": yaw,
        "period_deg": period,
        "sides_m": [round(s, 4) for s in sides],
        "fill": round(fill, 3),
        "pixels": len(xs),
    }


def fit_table_plane(depth_m: np.ndarray, handeye: HandEye, *, stride: int = 4, rounds: int = 4) -> dict:
    """Plane ``z = a x + b y + c`` through the table seen by the camera, robust to objects on it.

    Returns:
        ``a``, ``b``, ``c``, ``tilt_deg``, ``rms_mm`` and ``n`` (points kept).
    """
    vs, us = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    us, vs = us.ravel(), vs.ravel()
    keep = depth_m[vs, us] > 0
    points = back_project(depth_m, us[keep], vs[keep], handeye)
    points = points[points[:, 2] < np.median(points[:, 2]) + 0.03]
    coeffs = np.zeros(3)
    for _ in range(rounds):
        design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
        coeffs, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        residual = points[:, 2] - design @ coeffs
        cut = max(0.003, 2.5 * float(np.std(residual)))
        points = points[np.abs(residual) < cut]
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    residual = points[:, 2] - design @ coeffs
    a, b, c = (float(v) for v in coeffs)
    return {
        "a": a,
        "b": b,
        "c": c,
        "tilt_deg": math.degrees(math.atan(math.hypot(a, b))),
        "rms_mm": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "n": len(points),
    }
