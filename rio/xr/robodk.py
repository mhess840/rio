# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Arm link meshes from a RoboDK ``.robot`` library file, placed on the URDF joints.

The Kassow URDF only has kinematics; RoboDK's free robot library has the shapes. A
``.robot`` file is a 4-byte length plus a zlib stream. Inside are modified DH rows
(``alpha, a, theta, d`` in millimetres) and, per link, colored triangle blocks in that
link's DH frame: vertex records of three float32 plus two bytes of packed normal.

:func:`convert` checks that the DH chain and the URDF agree (each link's DH frame must
sit at a fixed transform from the URDF joint frame for random joints), moves every
link's triangles into its URDF joint frame in metres, thins them by vertex clustering,
and saves an ``.npz`` that :class:`rio.xr.ghost.GhostArm` draws. Laid out for the
RoboDK library's Kassow KR1018 file; other files are refused when the check fails.

    uv run python -m rio.xr.robodk ~/Downloads/Kassow-Robots-KR1018.robot
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from rio.workspace import experiments_root

DH_TABLE_AT = 0x5E0
DH_ROW_BYTES = 0xA0
DH_COLUMNS = slice(12, 16)
MESHES_AT = 0x7D08
VERTEX_BYTES = 14
# Each link entry is followed by two empty ones in the files seen so far.
ENTRY_STRIDE = 3
MAX_FRAME_DEVIATION = 1e-6


def default_mesh_path() -> Path:
    """Where the converted arm meshes live: ``experiments/xr/arm_mesh.npz``."""
    return experiments_root() / "xr" / "arm_mesh.npz"


@dataclass(frozen=True)
class RobodkRobot:
    """What :func:`read_robodk` found.

    Attributes:
        dh: ``(joints, 4)`` modified DH rows ``alpha [rad], a [mm], theta [rad], d [mm]``, one per joint.
        links: Per entry (base first), triangle soups ``(T * 3, 3)`` in millimetres in that frame.
    """

    dh: np.ndarray
    links: list[np.ndarray]


def read_robodk(path: Path, joints: int) -> RobodkRobot:
    """Decode the DH rows and link meshes of a RoboDK ``.robot`` file.

    Args:
        path: The ``.robot`` file.
        joints: Revolute joints of the arm; the file then holds ``joints`` DH rows and ``joints + 1`` meshes.

    Raises:
        ValueError: The data does not have the expected layout.
    """
    blob = Path(path).read_bytes()
    size = int.from_bytes(blob[:4], "big")
    raw = zlib.decompress(blob[4:])
    if len(raw) != size:
        raise ValueError(f"{path}: decompressed {len(raw)} bytes, header says {size}")
    table = np.frombuffer(raw[: DH_TABLE_AT + joints * DH_ROW_BYTES], dtype="<f8")
    dh = np.array([table[(DH_TABLE_AT + row * DH_ROW_BYTES) // 8 :][DH_COLUMNS] for row in range(joints)], dtype=float)
    entries: list[list[np.ndarray]] = []
    offset = MESHES_AT
    while offset < len(raw) - 8 and len(entries) < (joints + 1) * ENTRY_STRIDE:
        (blocks,) = struct.unpack_from("<q", raw, offset)
        offset += 8
        if not 0 <= blocks <= 64:
            raise ValueError(f"{path}: implausible block count {blocks} at byte {offset - 8}")
        soups = []
        for _ in range(blocks):
            (triangles,) = struct.unpack_from("<q", raw, offset)
            offset += 8 + 16  # count, then an RGBA color
            end = offset + triangles * 3 * VERTEX_BYTES
            if triangles < 0 or end > len(raw):
                raise ValueError(f"{path}: block of {triangles} triangles runs past the data")
            records = np.frombuffer(raw[offset:end], dtype=np.uint8).reshape(-1, VERTEX_BYTES)
            soups.append(records[:, :12].copy().view("<f4").reshape(-1, 3).astype(float))
            offset = end
        entries.append(soups)
    links = [np.concatenate(soups) if soups else np.zeros((0, 3)) for soups in entries[::ENTRY_STRIDE]]
    if len(links) != joints + 1 or any(len(link) == 0 for link in links):
        raise ValueError(f"{path}: expected {joints + 1} link meshes, found {sum(len(link) > 0 for link in links)}")
    return RobodkRobot(dh, links)


def dh_frames(dh: np.ndarray, q: np.ndarray) -> list[np.ndarray]:
    """Modified DH frames in metres: the base (identity), then one frame per joint.

    Args:
        dh: Rows from :class:`RobodkRobot`.
        q: Joint angles, one per row.
    """

    def rot_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])

    def rot_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])

    def shift(x: float, z: float) -> np.ndarray:
        matrix = np.eye(4)
        matrix[0, 3], matrix[2, 3] = x, z
        return matrix

    frames, frame = [np.eye(4)], np.eye(4)
    for (alpha, a, theta, d), angle in zip(dh, np.asarray(q, dtype=float), strict=True):
        frame = frame @ rot_x(alpha) @ shift(a / 1000, 0) @ rot_z(theta + angle) @ shift(0, d / 1000)
        frames.append(frame.copy())
    return frames


def cluster(soup: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Indexed mesh from a triangle soup, merging vertices that share a ``cell_m`` grid cell.

    Triangles that collapse or repeat are dropped. Returns vertices ``(V, 3)`` and triangles ``(T, 3)``.
    """
    keys = np.floor(np.asarray(soup, dtype=float) / cell_m).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(first), 3))
    np.add.at(sums, inverse, soup)
    vertices = sums / np.bincount(inverse, minlength=len(first))[:, None]
    triangles = inverse.reshape(-1, 3)
    keep = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (triangles[:, 0] != triangles[:, 2])
    triangles = triangles[keep]
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(unique)]
    used = np.unique(triangles)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[triangles]


def link_frames(urdf_path: str | Path, dh: np.ndarray) -> list[np.ndarray]:
    """Fixed transforms from each URDF joint frame to the DH frame of the same link (base first).

    Raises:
        ValueError: The URDF and the DH chain disagree for random joints.
    """
    import pinocchio as pin  # noqa: PLC0415  # ty: ignore[unresolved-import]

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    if model.nq != len(dh):
        raise ValueError(f"URDF has {model.nq} joints, the RoboDK file {len(dh)}")

    def urdf(q: np.ndarray) -> list[np.ndarray]:
        pin.forwardKinematics(model, data, q)
        return [np.array(data.oMi[i].homogeneous) for i in range(model.njoints)]

    def relative(q: np.ndarray) -> list[np.ndarray]:
        return [np.linalg.inv(u) @ r for u, r in zip(urdf(q), dh_frames(dh, q), strict=True)]

    fixed = relative(np.zeros(model.nq))
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(model.lowerPositionLimit.clip(-2.5), model.upperPositionLimit.clip(max=2.5))
        deviation = max(float(np.abs(a - b).max()) for a, b in zip(relative(q), fixed, strict=True))
        if deviation > MAX_FRAME_DEVIATION:
            raise ValueError(f"RoboDK DH chain and URDF disagree by {deviation:.2e} for q={np.round(q, 2)}")
    return fixed


def convert(source: Path, urdf_path: str | Path, out: Path, *, cell_m: float = 0.004) -> Path:
    """Write per-link meshes in URDF joint frames (metres) to ``out``.

    Args:
        source: RoboDK ``.robot`` file.
        urdf_path: Kinematic URDF of the same arm.
        out: ``.npz`` to write: ``vertices_<i>`` and ``triangles_<i>`` for link ``i`` (0 = base).
        cell_m: Vertex clustering cell; larger is coarser.
    """
    import pinocchio as pin  # noqa: PLC0415  # ty: ignore[unresolved-import]

    joints = pin.buildModelFromUrdf(str(urdf_path)).nq
    robot = read_robodk(source, joints)
    fixed = link_frames(urdf_path, robot.dh)
    arrays: dict[str, np.ndarray] = {"links": np.array(len(robot.links), dtype=np.int64)}
    for index, (soup, frame) in enumerate(zip(robot.links, fixed, strict=True)):
        points = soup / 1000.0
        in_urdf = points @ frame[:3, :3].T + frame[:3, 3]
        vertices, triangles = cluster(in_urdf, cell_m)
        arrays[f"vertices_{index}"] = vertices.astype(np.float32)
        arrays[f"triangles_{index}"] = triangles.astype(np.int32)
        logger.info("Link {}: {} triangles -> {}", index, len(soup) // 3, len(triangles))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, allow_pickle=False, **arrays)
    logger.info("Wrote {}", out)
    return out


def load_link_meshes(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-link ``(vertices, triangles)`` written by :func:`convert`, base first."""
    with np.load(path) as data:
        return [(data[f"vertices_{i}"].astype(float), data[f"triangles_{i}"].astype(int)) for i in range(int(data["links"]))]


@dataclass
class Args:
    source: tyro.conf.Positional[Path]
    """RoboDK ``.robot`` file of the arm."""
    out: Path | None = None
    """Output. Unset: experiments/xr/arm_mesh.npz."""
    cell_mm: float = 6.0
    """Vertex clustering cell; larger is coarser and faster to draw."""


def main() -> None:
    from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH  # noqa: PLC0415

    args = tyro.cli(Args)
    convert(args.source.expanduser(), DEFAULT_URDF_PATH, args.out or default_mesh_path(), cell_m=args.cell_mm / 1000)


if __name__ == "__main__":
    main()
