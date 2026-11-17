# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""A see-through model of the arm: real link shapes when converted, else tubes between joint centres.

The URDF only needs kinematics (the Kassow one has no geometry). :class:`GhostArm` solves
where the joints will be for a flange pose with a damped least-squares IK seeded at the
current joints, and :func:`ghost_items` turns the joint centres into tube and sphere meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from .robodk import default_mesh_path, load_link_meshes
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


def gripper_parts(flange: np.ndarray, opening: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """AG95 sketch at ``flange``: a body along flange z and two fingers that open along flange x."""
    origin, x_axis, z_axis = flange[:3, 3], flange[:3, 0], flange[:3, 2]
    palm = origin + z_axis * GRIPPER_BODY_M
    parts = [tube(origin, palm, ARM_RADIUS_M * 1.2)]
    half = 0.5 * GRIPPER_STROKE_M * float(np.clip(opening, 0.0, 1.0)) + 0.008
    for side in (-1.0, 1.0):
        base = palm + side * half * x_axis
        parts.append(tube(base, base + z_axis * GRIPPER_FINGER_M, 0.009, sides=6))
    return parts


def ghost_items(
    joints: np.ndarray,
    flange: np.ndarray,
    opening: float,
    *,
    color: tuple[float, float, float, float] = GHOST,
    joint_color: tuple[float, float, float, float] = GHOST_JOINT,
) -> tuple[Mesh, Mesh]:
    """Tubes between joint centres plus an AG95 sketch, and spheres at the joints.

    Args:
        joints: ``(K, 3)`` joint centres from the base outwards, flange last.
        flange: ``4 x 4`` flange pose in the same frame.
        opening: Gripper opening, 0 closed to 1 open.
        color: RGBA of the tubes.
        joint_color: RGBA of the joint spheres.

    Returns:
        The arm tubes and the joint spheres.
    """
    joints = np.asarray(joints, dtype=float)
    tubes = [tube(joints[k], joints[k + 1], ARM_RADIUS_M) for k in range(len(joints) - 1)]
    tubes += gripper_parts(flange, opening)
    spheres = [sphere(point, JOINT_RADIUS_M) for point in joints[:-1]]
    tube_v, tube_t = merge(tubes)
    sphere_v, sphere_t = merge(spheres)
    return Mesh(tube_v, tube_t, color), Mesh(sphere_v, sphere_t, joint_color)


class GhostArm:
    """Joint centres and IK for one URDF, through Pinocchio.

    With link meshes (``rio.xr.robodk``, one per URDF joint plus the base) the ghost has
    the real shapes; without, it is a stick model.

    Args:
        urdf_path: Kinematic URDF of the arm.
        ee_frame: Flange frame name; the last frame of the model when empty.
        mesh_path: Link meshes. Unset: ``experiments/xr/arm_mesh.npz`` if it exists.

    Raises:
        ImportError: Pinocchio is not installed.
    """

    def __init__(self, urdf_path: str, ee_frame: str = "end_effector", mesh_path: Path | None = None) -> None:
        # Optional: only the ghost needs Pinocchio, and it ships without type information.
        import pinocchio as pin  # noqa: PLC0415  # ty: ignore[unresolved-import]

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        name = ee_frame if ee_frame and self.model.existFrame(ee_frame) else self.model.frames[-1].name
        self.ee_id = self.model.getFrameId(name)
        self.links: list[tuple[np.ndarray, np.ndarray]] | None = None
        path = mesh_path if mesh_path is not None else default_mesh_path()
        if path.exists():
            links = load_link_meshes(path)
            if len(links) == self.model.njoints:
                self.links = links
            else:
                logger.warning("{} has {} link meshes, the URDF {} joints; stick ghost", path, len(links), self.model.njoints)

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

    def items_at(self, q: np.ndarray, q_now: np.ndarray, eef_now: np.ndarray, opening: float, **colors) -> tuple[Mesh, Mesh]:
        """Ghost meshes for joints ``q``, shifted onto the controller's ``eef_pose`` frame.

        The URDF flange and the controller's ``eef_pose`` differ by a few millimetres; the
        difference at the current joints ``q_now`` is removed, like ``goto.move_tcp`` adds it.

        Returns the arm and the gripper (real shapes), or tubes and joint spheres (stick model).

        Args:
            q: Joints to draw.
            q_now: Current joints.
            eef_now: Current measured flange pose; position first.
            opening: Gripper opening, 0 closed to 1 open.
            **colors: ``color`` and ``joint_color`` for :func:`ghost_items`.
        """
        _, flange_now = self.joints(q_now)
        offset = flange_now[:3, 3] - np.asarray(eef_now, dtype=float)[:3]
        centres, flange = self.joints(q)
        flange = flange.copy()
        flange[:3, 3] -= offset
        if self.links is None:
            return ghost_items(centres - offset, flange, opening, **colors)
        color = colors.get("color", GHOST)
        parts = []
        for index, (vertices, triangles) in enumerate(self.links):
            placement = np.array(self.data.oMi[index].homogeneous)  # still the FK of ``q``
            parts.append((vertices @ placement[:3, :3].T + placement[:3, 3] - offset, triangles))
        arm_v, arm_t = merge(parts)
        grip_v, grip_t = merge(gripper_parts(flange, opening))
        return Mesh(arm_v, arm_t, color, lit=True), Mesh(grip_v, grip_t, color, lit=True)

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
