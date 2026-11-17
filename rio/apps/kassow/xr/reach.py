# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Where the gripper reaches a horizontal slice with the grasp orientation.

Each slice is grown cell by cell from a seed: a cell counts when IK from its
neighbour's joints succeeds under the same rules as ``goto.move_tcp``. Results
are cached, because a few thousand IK calls take a while.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from loguru import logger

from rio.apps.kassow.goto import solve_ik

MAX_CELLS = 20_000
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def reach_slice(
    kin,
    z: float,
    rotvec: np.ndarray,
    *,
    seed_xy: np.ndarray,
    seed_q: np.ndarray,
    step_m: float,
    max_radius: float,
) -> np.ndarray:
    """Reachable grid points (``N x 3``) at height ``z``, connected to ``seed_xy``.

    Args:
        kin: Kinematics with ``fk`` and ``ik`` (URDF pose6, rotvec).
        z: Base-frame height of the slice.
        rotvec: TCP orientation for every cell.
        seed_xy: Grid origin; the fill starts here.
        seed_q: Joints the first IK starts from.
        step_m: Grid spacing.
        max_radius: Cells farther from the base axis are not tried.
    """
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    sx, sy = (float(value) for value in np.asarray(seed_xy, dtype=float).reshape(2))

    def goal(cell: tuple[int, int]) -> np.ndarray:
        return np.array([sx + cell[0] * step_m, sy + cell[1] * step_m, z, *rotvec])

    first, _ = solve_ik(kin, goal((0, 0)), seed_q=seed_q, from_q=seed_q)
    if first is None:
        return np.zeros((0, 3))
    reached = {(0, 0): first}
    refused: set[tuple[int, int]] = set()
    queue = deque([(0, 0)])
    while queue and len(reached) < MAX_CELLS:
        cell = queue.popleft()
        q = reached[cell]
        for di, dj in NEIGHBOURS:
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt in reached or nxt in refused:
                continue
            target = goal(nxt)
            if math.hypot(target[0], target[1]) > max_radius:
                refused.add(nxt)
                continue
            q_next, _ = solve_ik(kin, target, seed_q=q, from_q=q)
            if q_next is None:
                refused.add(nxt)
            else:
                reached[nxt] = q_next
                queue.append(nxt)
    return np.array([goal(cell)[:3] for cell in reached])


def reach_slices(
    kin,
    heights: tuple[float, ...],
    rotvec: np.ndarray,
    *,
    seed_xy: np.ndarray,
    seed_q: np.ndarray,
    step_m: float,
    max_radius: float,
    cache_dir: Path | None = None,
    model_id: str = "",
) -> dict[float, np.ndarray]:
    """:func:`reach_slice` for each height, read from or written to ``cache_dir``.

    ``model_id`` names the kinematic model in the cache key, e.g. the URDF path.
    """
    params = {
        "heights": [round(float(z), 4) for z in heights],
        "rotvec": np.round(np.asarray(rotvec, dtype=float), 4).tolist(),
        "seed_xy": np.round(np.asarray(seed_xy, dtype=float), 4).tolist(),
        "seed_q": np.round(np.asarray(seed_q, dtype=float), 4).tolist(),
        "step_m": round(float(step_m), 4),
        "max_radius": round(float(max_radius), 4),
        "model": model_id,
    }
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    path = None if cache_dir is None else cache_dir / f"reach_{key}.json"
    if path is not None and path.is_file():
        cached = json.loads(path.read_text())
        return {float(z): np.asarray(points, dtype=float).reshape(-1, 3) for z, points in cached["slices"].items()}
    slices = {}
    for z in heights:
        logger.info("Reach slice z={:.3f} m ...", z)
        slices[float(z)] = reach_slice(
            kin, float(z), rotvec, seed_xy=seed_xy, seed_q=seed_q, step_m=step_m, max_radius=max_radius
        )
        logger.info("Reach slice z={:.3f} m: {} cells", z, len(slices[float(z)]))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": params, "slices": {str(z): np.round(p, 4).tolist() for z, p in slices.items()}}
        path.write_text(json.dumps(payload) + "\n")
    return slices
