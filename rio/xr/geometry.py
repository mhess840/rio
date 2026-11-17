# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn scene layers into coloured vertices in the OpenXR local frame.

The robot base is z-up. OpenXR is y-up, x to the right, and -z forward.
:data:`YUP_FROM_BASE` is that axis change. Before registration the cell is
shifted in front of the head; after it, the headset uses the fitted matrix
instead, which already includes whatever rotation the touched points implied.
"""

from __future__ import annotations

import numpy as np

from .scene import Layer, Lines, Mesh, Points, SceneError

YUP_FROM_BASE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
PREVIEW_FORWARD_M = 1.2
# Light for ``Mesh.lit``, in the robot base frame: from above, a little to the side.
LIGHT_FROM_BASE = np.array([0.3, -0.4, 0.87]) / np.linalg.norm([0.3, -0.4, 0.87])
LIT_AMBIENT = 0.35
PREVIEW_DOWN_M = 0.4
Vertex = np.ndarray


def preview_transform() -> np.ndarray:
    """Place the robot origin in front of and below the head, in the OpenXR frame."""
    shift = np.eye(4)
    shift[1, 3] = -PREVIEW_DOWN_M
    shift[2, 3] = -PREVIEW_FORWARD_M
    return shift @ YUP_FROM_BASE


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4 x 4 matrix to ``N x 3`` points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((0, 3))
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (np.asarray(matrix, dtype=float) @ homo.T).T[:, :3]


def _paint(points: np.ndarray, color: tuple[float, float, float, float]) -> np.ndarray:
    painted = np.empty((len(points), 7), dtype=np.float32)
    painted[:, :3] = points
    painted[:, 3:] = np.asarray(color, dtype=np.float32)
    return painted


def _crosses(points: np.ndarray, matrix: np.ndarray, size_m: float, color: tuple[float, float, float, float]) -> np.ndarray:
    """Three segments through each point, ``size_m`` tip to tip, along the transformed axes."""
    centers = transform_points(points, matrix)
    if len(centers) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    half = np.asarray(matrix, dtype=float)[:3, :3] * (float(size_m) / 2.0)
    rows = np.empty((len(centers) * 6, 3), dtype=float)
    for axis in range(3):
        offset = half[:, axis]
        rows[axis * 2 :: 6] = centers - offset
        rows[axis * 2 + 1 :: 6] = centers + offset
    return _paint(rows, color)


def _triangles(mesh: Mesh, matrix: np.ndarray) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=int).reshape(-1, 3)
    if len(triangles) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    vertices = transform_points(mesh.vertices, matrix)
    painted = _paint(vertices[triangles.reshape(-1)], mesh.color)
    if mesh.lit:
        corners = np.asarray(mesh.vertices, dtype=float)[triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        # Two-sided: the facing of a triangle soup is not reliable.
        shade = LIT_AMBIENT + (1.0 - LIT_AMBIENT) * np.abs(normals @ LIGHT_FROM_BASE)
        painted[:, 3:6] *= np.repeat(shade, 3)[:, None].astype(np.float32)
    return painted


def layer_vertices(layers: tuple[Layer, ...] | list[Layer], matrix: np.ndarray) -> tuple[Vertex, Vertex]:
    """Line vertices and triangle vertices, each ``N x 7`` as xyz rgba.

    Hidden layers are left out. Points become small crosses so they stay a
    fixed size in metres. An empty scene is two arrays of shape ``(0, 7)``.
    """
    lines: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    for layer in layers:
        if not layer.visible:
            continue
        for item in layer.items:
            if isinstance(item, Points):
                lines.append(_crosses(item.positions, matrix, item.size_m, item.color))
            elif isinstance(item, Lines):
                lines.append(_paint(transform_points(item.positions, matrix), item.color))
            elif isinstance(item, Mesh):
                triangles.append(_triangles(item, matrix))
    line_vertices = np.concatenate(lines) if lines else np.zeros((0, 7), dtype=np.float32)
    triangle_vertices = np.concatenate(triangles) if triangles else np.zeros((0, 7), dtype=np.float32)
    return np.ascontiguousarray(line_vertices, dtype=np.float32), np.ascontiguousarray(triangle_vertices, dtype=np.float32)


def layers_from_scene(payloads: list[dict]) -> list[Layer]:
    """Parse a hub scene, dropping a payload the viewer cannot draw."""
    layers = []
    for payload in payloads:
        try:
            layers.append(Layer.from_json(payload))
        except SceneError:
            continue
    return layers


def pose_axes(position: np.ndarray, rotation: np.ndarray, length: float) -> np.ndarray:
    """Three coloured segments of ``length`` from ``position`` along the columns of ``rotation``."""
    origin = np.asarray(position, dtype=float).reshape(3)
    basis = np.asarray(rotation, dtype=float).reshape(3, 3)
    rows = []
    for axis, channel in enumerate(((1.0, 0.2, 0.2, 1.0), (0.3, 1.0, 0.4, 1.0), (0.4, 0.6, 1.0, 1.0))):
        end = origin + basis[:, axis] * length
        rows.append([*origin, *channel])
        rows.append([*end, *channel])
    return np.asarray(rows, dtype=np.float32)


def rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Column-vector rotation for an OpenXR quaternion."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
