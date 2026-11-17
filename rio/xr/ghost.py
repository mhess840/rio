# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""A see-through stick model of the arm: joint centres from the URDF, tubes between them.

The URDF only needs kinematics (the Kassow one has no geometry). :class:`GhostArm` solves
where the joints will be for a flange pose with a damped least-squares IK seeded at the
current joints, and :func:`ghost_items` turns the joint centres into tube and sphere meshes.
"""

from __future__ import annotations

import numpy as np

from .scene import Mesh

GHOST = (0.75, 0.9, 1.0, 0.28)
GHOST_JOINT = (0.85, 0.95, 1.0, 0.45)
ARM_RADIUS_M = 0.045
JOINT_RADIUS_M = 0.06
# AG95: body along the flange z axis, then two fingers that open along the flange x axis.
GRIPPER_BODY_M = 0.13
GRIPPER_FINGER_M = 0.065
GRIPPER_STROKE_M = 0.095


def tube(a: np.ndarray, b: np.ndarray, radius: float, sides: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Open cylinder from ``a`` to ``b``: vertices ``(2 * sides, 3)`` and triangles ``(2 * sides, 3)``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ring = radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    vertices = np.concatenate([a + ring, b + ring])
    i = np.arange(sides)
    j = (i + 1) % sides
    triangles = np.concatenate([np.stack([i, j, i + sides], 1), np.stack([j, j + sides, i + sides], 1)])
    return vertices, triangles


def sphere(centre: np.ndarray, radius: float, rings: int = 5, sides: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Low-poly UV sphere: poles plus ``rings - 1`` latitude rings of ``sides`` vertices."""
    centre = np.asarray(centre, dtype=float)
    vertices = [centre + np.array([0.0, 0.0, radius])]
    for k in range(1, rings):
        theta = np.pi * k / rings
        phi = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
        ring = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.full(sides, np.cos(theta))], 1)
        vertices.extend(centre + radius * ring)
    vertices.append(centre + np.array([0.0, 0.0, -radius]))
    triangles = []
    bottom = len(vertices) - 1
    for s in range(sides):
        n = (s + 1) % sides
        triangles.append([0, 1 + s, 1 + n])
        last = 1 + (rings - 2) * sides
        triangles.append([bottom, last + n, last + s])
    for k in range(rings - 2):
        top, low = 1 + k * sides, 1 + (k + 1) * sides
        for s in range(sides):
            n = (s + 1) % sides
            triangles.append([top + s, low + s, low + n])
            triangles.append([top + s, low + n, top + n])
    return np.asarray(vertices), np.asarray(triangles)


def merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate meshes, offsetting each part's triangle indices."""
    vertices, triangles, offset = [], [], 0
    for v, t in parts:
        vertices.append(v)
        triangles.append(np.asarray(t) + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(triangles)


def ghost_items(joints: np.ndarray, flange: np.ndarray, opening: float) -> tuple[Mesh, Mesh]:
    """Tubes between joint centres plus an AG95 sketch, and spheres at the joints.

    Args:
        joints: ``(K, 3)`` joint centres from the base outwards, flange last.
        flange: ``4 x 4`` flange pose in the same frame.
        opening: Gripper opening, 0 closed to 1 open.

    Returns:
        The arm tubes and the joint spheres.
    """
    joints = np.asarray(joints, dtype=float)
    tubes = [tube(joints[k], joints[k + 1], ARM_RADIUS_M) for k in range(len(joints) - 1)]
    origin, x_axis, z_axis = flange[:3, 3], flange[:3, 0], flange[:3, 2]
    palm = origin + z_axis * GRIPPER_BODY_M
    tubes.append(tube(origin, palm, ARM_RADIUS_M * 1.2))
    half = 0.5 * GRIPPER_STROKE_M * float(np.clip(opening, 0.0, 1.0)) + 0.008
    for side in (-1.0, 1.0):
        base = palm + side * half * x_axis
        tubes.append(tube(base, base + z_axis * GRIPPER_FINGER_M, 0.009, sides=6))
    spheres = [sphere(point, JOINT_RADIUS_M) for point in joints[:-1]]
    tube_v, tube_t = merge(tubes)
    sphere_v, sphere_t = merge(spheres)
    return Mesh(tube_v, tube_t, GHOST), Mesh(sphere_v, sphere_t, GHOST_JOINT)


class GhostArm:
    """Joint centres and IK for one URDF, through Pinocchio.

    Args:
        urdf_path: Kinematic URDF of the arm.
        ee_frame: Flange frame name; the last frame of the model when empty.

    Raises:
        ImportError: Pinocchio is not installed.
    """

    def __init__(self, urdf_path: str, ee_frame: str = "end_effector") -> None:
        # Optional: only the ghost needs Pinocchio, and it ships without type information.
        import pinocchio as pin  # noqa: PLC0415  # ty: ignore[unresolved-import]

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        name = ee_frame if ee_frame and self.model.existFrame(ee_frame) else self.model.frames[-1].name
        self.ee_id = self.model.getFrameId(name)

    def joints(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint centres ``(njoints - 1 + 1, 3)`` with the flange appended, and the flange ``4 x 4``."""
        pin = self.pin
        q = np.asarray(q, dtype=float).reshape(-1)[: self.model.nq]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        centres = [np.array(self.data.oMi[i].translation) for i in range(1, self.model.njoints)]
        flange = np.array(self.data.oMf[self.ee_id].homogeneous)
        centres.append(flange[:3, 3])
        return np.asarray(centres), flange

    def solve(self, goal: np.ndarray, seed_q: np.ndarray, *, iterations: int = 80) -> np.ndarray | None:
        """Joints that put the flange at ``goal`` (``xyz`` plus rotation vector), or None if IK misses by > 1 cm.

        Args:
            goal: Flange pose in the URDF base frame.
            seed_q: Start of the search, normally the current joints.
            iterations: Damped least-squares steps.
        """
        pin = self.pin
        goal = np.asarray(goal, dtype=float).reshape(-1)
        target = pin.SE3(pin.exp3(goal[3:6]), goal[:3])
        q = np.asarray(seed_q, dtype=float).reshape(-1)[: self.model.nq].copy()
        damping = 1e-4
        for _ in range(iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            error = pin.log6(self.data.oMf[self.ee_id].actInv(target)).vector
            if float(np.linalg.norm(error)) < 1e-4:
                break
            jacobian = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.LOCAL)
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            q = pin.integrate(self.model, q, step)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        if float(np.linalg.norm(self.data.oMf[self.ee_id].translation - goal[:3])) > 0.01:
            return None
        return q
