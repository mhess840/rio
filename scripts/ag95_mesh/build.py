# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""AG95 part soups -> ``gripper_mesh.npz`` for ``rio.xr.ghost.GripperModel``. See README.md.

Usage: build.py <folder with parts.npz and housing.npy> <cell mm> <out.npz>
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

from rio.xr.robodk import cluster

# STEP frame of the AG95 file (mm): mount face centre, and the x of the gripper's middle.
CX, Y0, CZ = 20.5, -25.72, 93.0
# Right finger, read off the CAD: proximal phalanx pivot on the body, its joint with the
# distal phalanx, and the parallel rod's joint with the distal phalanx.
PIVOT = np.array([37.0, 85.0, CZ])
JOINT = np.array([75.0, 122.5, CZ])
ROD_JOINT = np.array([86.5, 118.0, CZ])
TIP_PARTS = ("FINGERTIP", "DISTAL_PHALANX", "M3X5", "к")  # "к" names the joint pins
STROKE_M = 0.095


def to_flange(points: np.ndarray) -> np.ndarray:
    """STEP mm to flange metres: x stays (opening), y becomes z (approach), z becomes -y."""
    p = np.asarray(points, dtype=float)
    return np.stack([p[..., 0] - CX, -(p[..., 2] - CZ), p[..., 1] - Y0], -1) / 1000


def mirror(point: np.ndarray) -> np.ndarray:
    return np.array([2 * CX - point[0], point[1], point[2]])


def ends(soup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper end of an elongated part (centroids of the last 4 mm along its long axis)."""
    flat = soup[:, :2] - soup[:, :2].mean(0)
    along = flat @ np.linalg.svd(flat, full_matrices=False)[2][0]
    first, last = soup[along < along.min() + 4.0].mean(0), soup[along > along.max() - 4.0].mean(0)
    low, high = sorted((first, last), key=lambda point: point[1])
    return low, high


def main(folder: str, cell_mm: float, out_path: str) -> None:
    parts = np.load(f"{folder}/parts.npz")
    housing = np.load(f"{folder}/housing.npy").reshape(-1, 3, 3)
    # The housing surface model also carries copies of the finger parts; keep what no solid covers.
    distance, _ = cKDTree(np.concatenate([parts[key] for key in parts.files])).query(housing.mean(1))
    static = [housing[distance > 2.0].reshape(-1, 3)]
    sides = {side: {"tip": [], "turns": [], "pairs": []} for side in ("left", "right")}
    for key in parts.files:
        name, soup = key.split("|")[-1], parts[key]
        side = "left" if soup[:, 0].mean() < CX else "right"
        place = (lambda p: p) if side == "right" else mirror
        arm = place(JOINT) - place(PIVOT)
        if any(tag in name for tag in TIP_PARTS):
            sides[side]["tip"].append(soup)
        elif "PROXIMAL_PHALANX" in name or "CRANK" in name:
            sides[side]["turns"].append((soup, place(PIVOT)))
        elif "PARALLEL_ROB" in name:
            sides[side]["turns"].append((soup, place(ROD_JOINT) - arm))
        elif "ADAPTIVE" in name or not key.split(":")[1]:
            sides[side]["pairs"].append((soup, *ends(soup)))
        else:
            static.append(soup)
    out: dict[str, np.ndarray] = {"stroke_m": np.array(STROKE_M)}

    def put(prefix: str, soup: np.ndarray) -> None:
        vertices, triangles = cluster(to_flange(soup), cell_mm / 1000)
        out[f"{prefix}_vertices"] = vertices.astype(np.float32)
        out[f"{prefix}_triangles"] = triangles.astype(np.int32)
        print(prefix, len(triangles))

    put("static", np.concatenate(static))
    for side, groups in sides.items():
        place = (lambda p: p) if side == "right" else mirror
        put(f"{side}_tip", np.concatenate(groups["tip"]))
        for k, (soup, pivot) in enumerate(groups["turns"]):
            put(f"{side}_turn{k}", soup)
            out[f"{side}_turn{k}_pivot"] = to_flange(pivot)
        for k, (soup, low, high) in enumerate(groups["pairs"]):
            put(f"{side}_pair{k}", soup)
            out[f"{side}_pair{k}_low"], out[f"{side}_pair{k}_high"] = to_flange(low), to_flange(high)
        out[f"{side}_turns"], out[f"{side}_pairs"] = np.array(len(groups["turns"])), np.array(len(groups["pairs"]))
        out[f"{side}_pivot"], out[f"{side}_joint"] = to_flange(place(PIVOT)), to_flange(place(JOINT))
    np.savez_compressed(out_path, **out)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
