# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Map a point of a fixed camera's photo to the robot table, at grasp height.

The camera looks down on a flat table, so a homography from image fractions to
base ``x, y`` is enough. Each calibration sample is one object resting on the
table: the TCP that set it down (``pose.json``), and photos taken after the arm
moved away. ``label`` asks Jev to box the object in every photo; ``fit`` turns
the box centres and TCPs into the map and refuses it when leave-one-out errors
are too large.

    uv run python -m rio.apps.kassow.graph.calibrate            # record samples
    uv run python -m rio.apps.kassow.graph.table_map --command label --target "the red cube"
    uv run python -m rio.apps.kassow.graph.table_map --command fit

With a hand-eye calibration of the camera (``graph.handeye``) the homography is
computed instead of fitted: the ray through a pixel meets the plane of the grasped
object, and the tool offset turns that point into the TCP. The samples then only
give the grasp height, orientation and IK seeds; no Jev labels are needed.

    uv run python -m rio.apps.kassow.graph.table_map --command handeye
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from rio.apps.kassow.experiments import experiments_root
from rio.apps.kassow.hardware import JEV_SERIAL, OVERVIEW_SERIAL
from rio.jev.client import JevCfg, JevClient, encode_jpeg

from .check import BOX_YES, NOUL_YES
from .locate import Box, locate_questions, read_box, read_noul

if TYPE_CHECKING:
    from .handeye import HandEye

POSE_NAME = "pose.json"
TABLE_MAP_NAME = "table_map.json"
CAMERAS = ("overview", "jev")
MIN_FIT_SAMPLES = 6
# A target this far outside the calibrated area is still trusted.
HULL_MARGIN_M = 0.02
MAX_LOO_ERROR_M = 0.02
MAX_Z_ERROR_M = 0.01
# Reported TCP to the centre of an object held at the fingertips, along the tool axis (AG95, 2026-11-13).
TOOL_OFFSET_M = 0.233
ROT_SPREAD_WARN_DEG = 10.0


def calibration_dir() -> Path:
    """Folder of the table samples, outside the orchestrator logs."""
    return experiments_root() / "graph" / "calibration" / "table"


def table_map_path() -> Path:
    """Where ``fit`` writes the map and the graph runner reads it."""
    return calibration_dir() / TABLE_MAP_NAME


def labels_path(root: Path, camera: str) -> Path:
    return root / f"labels_{camera}.json"


def _homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def _normalizer(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(points - centre, axis=1)))
    if not spread > 0.0:
        raise ValueError("calibration points all coincide")
    scale = math.sqrt(2.0) / spread
    return np.array([[scale, 0.0, -scale * centre[0]], [0.0, scale, -scale * centre[1]], [0.0, 0.0, 1.0]])


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT: the 3 x 3 ``H`` with ``dst ~ H @ src`` from at least four point pairs.

    Raises:
        ValueError: Fewer than four pairs, or the points are (nearly) collinear.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("a homography needs at least four point pairs")
    t_src, t_dst = _normalizer(src), _normalizer(dst)
    a = _homogeneous(src) @ t_src.T
    b = _homogeneous(dst) @ t_dst.T
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[7] < 1e-8 * singular[0]:
        raise ValueError("calibration points are collinear; spread them over the table")
    homography = np.linalg.inv(t_dst) @ vt[-1].reshape(3, 3) @ t_src
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("homography is degenerate")
    return homography / homography[2, 2]


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``N x 2`` points.

    Raises:
        ValueError: A point maps to infinity.
    """
    mapped = _homogeneous(np.asarray(points, dtype=float).reshape(-1, 2)) @ np.asarray(homography).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("point maps to infinity")
    return mapped[:, :2] / mapped[:, 2:3]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise corners of the convex hull of 2D points."""
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float).reshape(-1, 2)})
    if len(unique) < 3:
        return np.asarray(unique, dtype=float).reshape(-1, 2)

    def turn(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def inside_hull(hull: np.ndarray, point: np.ndarray, margin: float = 0.0) -> bool:
    """True when ``point`` is inside the counter-clockwise ``hull`` grown by ``margin``."""
    if len(hull) < 3:
        return False
    px, py = (float(value) for value in np.asarray(point, dtype=float).reshape(2))
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        ex, ey = b - a
        if (ex * (py - a[1]) - ey * (px - a[0])) / math.hypot(ex, ey) < -margin:
            return False
    return True


@dataclass(frozen=True)
class Target:
    """An object located on the table, in the robot base frame (metres)."""

    name: str
    x: float
    y: float
    width_m: float
    depth_m: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width_m": round(self.width_m, 4),
            "depth_m": round(self.depth_m, 4),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class TableMap:
    """Image fractions of one fixed camera to base ``x, y``, plus the calibrated grasp.

    Attributes:
        camera: ``jev`` or ``overview``.
        serial: RealSense serial of that camera when the map was fitted.
        homography: Image ``(u, v)`` fractions to base ``(x, y)`` in metres.
        plane: ``z = a x + b y + c`` through the TCPs of the calibration grasps.
        rotvec: Mean TCP orientation of the calibration grasps.
        hull: Counter-clockwise corners of the calibrated area.
        seeds_xy: TCP ``x, y`` of each sample.
        seeds_q: Joint angles of each sample; IK starts from the nearest one.
        margin_m: How far outside ``hull`` a target is still trusted.
        stats: Fit quality written by ``fit``.
    """

    camera: str
    serial: str
    homography: np.ndarray
    plane: np.ndarray
    rotvec: np.ndarray
    hull: np.ndarray
    seeds_xy: np.ndarray
    seeds_q: np.ndarray
    margin_m: float = HULL_MARGIN_M
    stats: dict = field(default_factory=dict)

    def to_xy(self, u: float, v: float) -> np.ndarray:
        return apply_homography(self.homography, np.array([[u, v]]))[0]

    def z_at(self, x: float, y: float) -> float:
        """Calibrated grasp height of the TCP at ``x, y``."""
        return float(self.plane @ np.array([x, y, 1.0]))

    def contains(self, x: float, y: float) -> bool:
        return inside_hull(self.hull, np.array([x, y]), self.margin_m)

    def seed_q(self, x: float, y: float) -> np.ndarray:
        index = int(np.argmin(np.linalg.norm(self.seeds_xy - np.array([x, y]), axis=1)))
        return self.seeds_q[index].copy()

    def target(self, box: Box, name: str) -> Target:
        """Centre and footprint of a Jev box on the table."""
        u, v = box.centre
        x, y = self.to_xy(u, v)
        width = float(np.linalg.norm(self.to_xy(box.right, v) - self.to_xy(box.left, v)))
        depth = float(np.linalg.norm(self.to_xy(u, box.bottom) - self.to_xy(u, box.top)))
        return Target(name, float(x), float(y), width, depth, box.confidence)

    def save(self, path: Path) -> None:
        payload = {
            "camera": self.camera,
            "serial": self.serial,
            "homography": self.homography.tolist(),
            "plane": self.plane.tolist(),
            "rotvec": self.rotvec.tolist(),
            "hull": self.hull.tolist(),
            "seeds_xy": self.seeds_xy.tolist(),
            "seeds_q": self.seeds_q.tolist(),
            "margin_m": self.margin_m,
            "stats": self.stats,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> TableMap:
        """Read a map written by :meth:`save`.

        Raises:
            ValueError: A field is missing, has the wrong shape, or is not finite.
        """
        raw = json.loads(Path(path).read_text())
        try:
            table = cls(
                camera=str(raw["camera"]),
                serial=str(raw["serial"]),
                homography=np.asarray(raw["homography"], dtype=float),
                plane=np.asarray(raw["plane"], dtype=float),
                rotvec=np.asarray(raw["rotvec"], dtype=float),
                hull=np.asarray(raw["hull"], dtype=float),
                seeds_xy=np.asarray(raw["seeds_xy"], dtype=float),
                seeds_q=np.asarray(raw["seeds_q"], dtype=float),
                margin_m=float(raw["margin_m"]),
                stats=dict(raw.get("stats") or {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"table map {path} is invalid: {error}") from error
        shapes_ok = (
            table.homography.shape == (3, 3)
            and table.plane.shape == (3,)
            and table.rotvec.shape == (3,)
            and table.hull.ndim == 2
            and table.hull.shape[0] >= 3
            and table.hull.shape[1] == 2
            and table.seeds_xy.ndim == 2
            and table.seeds_xy.shape[1] == 2
            and table.seeds_q.ndim == 2
            and len(table.seeds_q) == len(table.seeds_xy) > 0
        )
        arrays = (table.homography, table.plane, table.rotvec, table.hull, table.seeds_xy, table.seeds_q)
        if not shapes_ok or not all(np.all(np.isfinite(a)) for a in arrays) or not 0.0 <= table.margin_m <= 0.1:
            raise ValueError(f"table map {path} has wrong shapes or values")
        return table


@dataclass(frozen=True)
class Sample:
    """One recorded grasp on the table."""

    name: str
    folder: Path
    xyz: np.ndarray
    rotvec: np.ndarray
    joint_q: np.ndarray


def sample_folders(root: Path) -> list[Path]:
    """Numbered sample folders that hold a ``pose.json``, in order."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (p / POSE_NAME).is_file())


def load_samples(root: Path) -> list[Sample]:
    """Samples that have both the TCP and every photo."""
    samples = []
    for folder in sample_folders(root):
        if not all((folder / f"{camera}.jpg").is_file() for camera in CAMERAS):
            continue
        record = json.loads((folder / POSE_NAME).read_text())
        eef = np.asarray(record["eef_pose"], dtype=float).reshape(6)
        joint_q = np.asarray(record["joint_q"], dtype=float).reshape(-1)
        samples.append(Sample(folder.name, folder, eef[:3], eef[3:], joint_q))
    return samples


def label_samples(
    root: Path,
    ask: Callable[[bytes, dict], dict],
    target: str,
    *,
    camera: str = "jev",
    repeats: int = 1,
    relabel: bool = False,
) -> dict:
    """Ask Jev to box ``target`` in each photo and store the box centre in ``labels_<camera>.json``.

    A label is usable (``ok``) when the median visibility and box confidence pass
    the graph defaults. ``spread`` is the largest distance of one repeat's centre
    from the median, as a fraction of the image.

    Args:
        root: Sample folder.
        ask: ``ask(jpeg, questions)`` returns the ``answers`` object.
        target: What to box, e.g. ``the red cube``.
        camera: Which photo of each sample.
        repeats: Questions per photo.
        relabel: Ask again for samples that already have a label.
    """
    path = labels_path(root, camera)
    labels: dict = json.loads(path.read_text()) if path.is_file() else {}
    questions = locate_questions(target)
    for sample in load_samples(root):
        if sample.name in labels and not relabel:
            continue
        jpeg = encode_jpeg((sample.folder / f"{camera}.jpg").read_bytes())
        visible, confidence, centres = [], [], []
        for _ in range(max(1, repeats)):
            answers = ask(jpeg, questions)
            box = read_box(answers)
            visible.append(read_noul(answers))
            confidence.append(box.confidence)
            centres.append(box.centre)
        uv = np.median(np.asarray(centres), axis=0)
        record = {
            "target": target,
            "u": float(uv[0]),
            "v": float(uv[1]),
            "visible": float(np.median(visible)),
            "confidence": float(np.median(confidence)),
            "spread": float(np.max(np.linalg.norm(np.asarray(centres) - uv, axis=1))),
        }
        record["ok"] = record["visible"] >= NOUL_YES and record["confidence"] >= BOX_YES
        labels[sample.name] = record
        path.write_text(json.dumps(labels, indent=2) + "\n")
    return labels


def fit_table_map(
    samples: list[Sample],
    labels: dict,
    *,
    camera: str,
    serial: str,
    margin_m: float = HULL_MARGIN_M,
) -> TableMap:
    """Fit the map from every sample with an ``ok`` label.

    ``stats`` holds the fit and leave-one-out errors per sample, the plane residual,
    and the orientation spread. The caller decides whether they are good enough.

    Raises:
        ValueError: Fewer than ``MIN_FIT_SAMPLES`` usable samples, or collinear samples.
    """
    used = [s for s in samples if isinstance(labels.get(s.name), dict) and labels[s.name].get("ok")]
    if len(used) < MIN_FIT_SAMPLES:
        raise ValueError(f"{len(used)} usable samples; fit needs at least {MIN_FIT_SAMPLES} so leave-one-out can check it")
    uv = np.array([[labels[s.name]["u"], labels[s.name]["v"]] for s in used], dtype=float)
    xyz = np.array([s.xyz for s in used], dtype=float)
    xy = xyz[:, :2]
    homography = fit_homography(uv, xy)
    fit_err = np.linalg.norm(apply_homography(homography, uv) - xy, axis=1)
    loo_err = np.empty(len(used))
    for i in range(len(used)):
        rest = np.arange(len(used)) != i
        loo_err[i] = np.linalg.norm(apply_homography(fit_homography(uv[rest], xy[rest]), uv[i : i + 1])[0] - xy[i])
    plane, *_ = np.linalg.lstsq(_homogeneous(xy), xyz[:, 2], rcond=None)
    z_err = np.abs(_homogeneous(xy) @ plane - xyz[:, 2])
    rotations = Rotation.from_rotvec(np.array([s.rotvec for s in used]))
    mean = rotations.mean()
    rot_spread = float(np.max(cast(Rotation, mean.inv() * rotations).magnitude()))
    stats = {
        "fitted": datetime.now().isoformat(timespec="seconds"),
        "n": len(used),
        "fit_rms_m": float(np.sqrt(np.mean(fit_err**2))),
        "loo_rms_m": float(np.sqrt(np.mean(loo_err**2))),
        "loo_max_m": float(np.max(loo_err)),
        "z_max_error_m": float(np.max(z_err)),
        "rot_spread_deg": math.degrees(rot_spread),
        "samples": {
            s.name: {"fit_m": float(f), "loo_m": float(lo), "z_m": float(z)}
            for s, f, lo, z in zip(used, fit_err, loo_err, z_err, strict=True)
        },
    }
    return TableMap(
        camera=camera,
        serial=serial,
        homography=homography,
        plane=np.asarray(plane, dtype=float),
        rotvec=mean.as_rotvec(),
        hull=convex_hull(xy),
        seeds_xy=xy,
        seeds_q=np.array([s.joint_q for s in used], dtype=float),
        margin_m=margin_m,
        stats=stats,
    )


def table_map_from_handeye(
    handeye: HandEye,
    samples: list[Sample],
    *,
    camera: str = "jev",
    tool_offset_m: float = TOOL_OFFSET_M,
    margin_m: float = HULL_MARGIN_M,
    grid: int = 9,
) -> TableMap:
    """Map computed from the camera pose: pixel ray, object plane, then back along the tool to the TCP.

    The samples are grasps of the object on the table. They give the grasp orientation
    (mean), the TCP height (plane through the grasps), the height of the object centre
    (``tool_offset_m`` along each grasp's tool axis), the calibrated area and the IK seeds.

    Raises:
        ValueError: Fewer than three samples.
    """
    if len(samples) < 3:
        raise ValueError(f"{len(samples)} samples; the grasp plane needs at least 3")
    xyz = np.array([s.xyz for s in samples], dtype=float)
    rotations = Rotation.from_rotvec(np.array([s.rotvec for s in samples], dtype=float))
    mean = rotations.mean()
    tool = np.array([0.0, 0.0, tool_offset_m])
    object_z = float(np.mean(xyz[:, 2] + rotations.apply(tool)[:, 2]))
    tool_xy = mean.apply(tool)[:2]
    width, height = handeye.intrinsics.width, handeye.intrinsics.height
    fractions = np.array([[u, v] for u in np.linspace(0.02, 0.98, grid) for v in np.linspace(0.02, 0.98, grid)])
    src, dst = [], []
    for u, v in fractions:
        hit = handeye.pixel_to_plane(u * width, v * height, object_z)
        if hit is not None:
            src.append((u, v))
            dst.append(hit[:2] - tool_xy)
    src_a, dst_a = np.asarray(src), np.asarray(dst)
    homography = fit_homography(src_a, dst_a)
    exact = float(np.max(np.linalg.norm(apply_homography(homography, src_a) - dst_a, axis=1)))
    xy = xyz[:, :2]
    plane, *_ = np.linalg.lstsq(_homogeneous(xy), xyz[:, 2], rcond=None)
    z_err = np.abs(_homogeneous(xy) @ plane - xyz[:, 2])
    rot_spread = float(np.max(cast(Rotation, mean.inv() * rotations).magnitude()))
    stats = {
        "fitted": datetime.now().isoformat(timespec="seconds"),
        "source": "handeye",
        "n": len(samples),
        # Same key as a fitted map, so describe and run report it: the hand-eye error at the marker.
        "loo_max_m": handeye.stats["loo_max_mm"] / 1000.0,
        "tool_offset_m": tool_offset_m,
        "object_z_m": object_z,
        "homography_max_error_m": exact,
        "z_max_error_m": float(np.max(z_err)),
        "rot_spread_deg": math.degrees(rot_spread),
        "handeye": {k: handeye.stats[k] for k in ("fitted", "n_samples", "loo_rms_mm", "loo_max_mm", "camera_xyz_m")},
    }
    return TableMap(
        camera=camera,
        serial=handeye.camera_serial,
        homography=homography,
        plane=np.asarray(plane, dtype=float),
        rotvec=mean.as_rotvec(),
        hull=convex_hull(xy),
        seeds_xy=xy,
        seeds_q=np.array([s.joint_q for s in samples], dtype=float),
        margin_m=margin_m,
        stats=stats,
    )


def map_problems(table: TableMap, *, max_error_m: float = MAX_LOO_ERROR_M, max_z_error_m: float = MAX_Z_ERROR_M) -> list[str]:
    """Reasons to refuse a fitted map. Empty when it may be written."""
    problems = []
    if table.stats["loo_max_m"] > max_error_m:
        problems.append(f"leave-one-out error {table.stats['loo_max_m'] * 1000:.0f} mm exceeds {max_error_m * 1000:.0f} mm")
    if table.stats["z_max_error_m"] > max_z_error_m:
        problems.append(
            f"grasp heights leave the fitted plane by {table.stats['z_max_error_m'] * 1000:.0f} mm "
            f"(limit {max_z_error_m * 1000:.0f} mm); hold every sample at the same height above the table"
        )
    return problems


@dataclass
class Args:
    """Label the table samples with Jev, or fit the image-to-table map."""

    command: Literal["label", "fit", "handeye"] = "fit"
    root: Path | None = None
    """Sample folder. Unset: experiments/graph/calibration/table."""
    camera: Literal["jev", "overview"] = "jev"
    target: str = "the cube"
    """What Jev boxes in every photo (``label``)."""
    repeats: int = 1
    """Questions per photo (``label``). Three or more show how steady the box is."""
    relabel: bool = False
    """Ask again for samples that already have a label."""
    max_error_m: float = MAX_LOO_ERROR_M
    """``fit`` refuses the map above this leave-one-out error."""
    max_z_error_m: float = MAX_Z_ERROR_M
    """``fit`` refuses the map when a grasp height leaves the fitted plane by more."""
    margin_m: float = HULL_MARGIN_M
    """How far outside the calibrated area a target is still trusted."""
    tool_offset_m: float = TOOL_OFFSET_M
    """``handeye``: reported TCP to the centre of the grasped object, along the tool axis."""
    jev: JevCfg = field(default_factory=JevCfg)


def main(args: Args) -> Path:
    """Label or fit. Returns the labels file or the map file."""
    root = args.root or calibration_dir()
    if args.command == "label":
        client = JevClient(args.jev)
        labels = label_samples(
            root,
            lambda jpeg, questions: client.ask(jpeg, questions).answers,
            args.target,
            camera=args.camera,
            repeats=args.repeats,
            relabel=args.relabel,
        )
        for name, label in sorted(labels.items()):
            print(
                f"{name}  u={label['u']:.3f} v={label['v']:.3f}  visible={label['visible']:.2f}  "
                f"conf={label['confidence']:.2f}  spread={label['spread']:.4f}  {'ok' if label['ok'] else 'SKIP'}"
            )
        return labels_path(root, args.camera)
    if args.command == "handeye":
        from .handeye import load_handeye, result_path

        if args.camera != "jev":
            raise SystemExit("handeye: only the Jev camera has a hand-eye calibration")
        table = table_map_from_handeye(
            load_handeye(result_path()),
            load_samples(root),
            camera=args.camera,
            tool_offset_m=args.tool_offset_m,
            margin_m=args.margin_m,
        )
        s = table.stats
        print(
            f"n={s['n']}  object z {s['object_z_m'] * 1000:.0f} mm  grasp plane max {s['z_max_error_m'] * 1000:.0f} mm  "
            f"orientation spread {s['rot_spread_deg']:.1f} deg  "
            f"homography exact to {s['homography_max_error_m'] * 1e6:.1f} um  hand-eye loo {s['handeye']['loo_rms_mm']:.1f} mm"
        )
        path = root / TABLE_MAP_NAME
        table.save(path)
        print(f"wrote {path}")
        return path
    labels_file = labels_path(root, args.camera)
    if not labels_file.is_file():
        raise SystemExit(f"{labels_file} is missing; run label first")
    serial = JEV_SERIAL if args.camera == "jev" else OVERVIEW_SERIAL
    table = fit_table_map(
        load_samples(root), json.loads(labels_file.read_text()), camera=args.camera, serial=serial, margin_m=args.margin_m
    )
    for name, row in table.stats["samples"].items():
        print(
            f"{name}  fit {row['fit_m'] * 1000:5.1f} mm  "
            f"leave-one-out {row['loo_m'] * 1000:5.1f} mm  z {row['z_m'] * 1000:4.1f} mm"
        )
    print(
        f"n={table.stats['n']}  loo rms {table.stats['loo_rms_m'] * 1000:.1f} mm  "
        f"max {table.stats['loo_max_m'] * 1000:.1f} mm  orientation spread {table.stats['rot_spread_deg']:.1f} deg"
    )
    if table.stats["rot_spread_deg"] > ROT_SPREAD_WARN_DEG:
        print(f"warning: grasp orientations differ by up to {table.stats['rot_spread_deg']:.0f} deg; pick_at uses their mean")
    problems = map_problems(table, max_error_m=args.max_error_m, max_z_error_m=args.max_z_error_m)
    if problems:
        raise SystemExit("map not written:\n  " + "\n  ".join(problems))
    path = root / TABLE_MAP_NAME
    table.save(path)
    print(f"wrote {path}")
    return path


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
