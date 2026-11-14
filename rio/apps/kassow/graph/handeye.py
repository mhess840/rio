# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Hand-eye calibration of a fixed camera against the Kassow base with ArUco markers on the arm.

The arm drives itself through poses around the pose it is in at the start. At each pose the
measured TCP, the joints, and the pixel centres of the visible markers are stored:

- the gripper marker (default id 1, 30 mm) rides on the TCP,
- the back marker (default id 2, 40 mm) rides on a wrist link that does not turn with the
  gripper; which link is decided by the fit (``link6`` or ``link5``).

A least-squares fit finds where the camera sits in the base frame and where each marker sits
on its link. Only marker centres are used, so small markers are enough. A pose counts when
either marker is seen.

Check without moving (default)::

    uv run python -m rio.apps.kassow.graph.handeye

Drive the poses and fit (E-stop in reach)::

    uv run python -m rio.apps.kassow.graph.handeye --no-dry-run

Fit a recorded session again::

    uv run python -m rio.apps.kassow.graph.handeye --command fit --session experiments/graph/calibration/handeye/jev/<stamp>
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rio.apps.kassow.experiments import experiments_root, jev_camera
from rio.apps.kassow.goto import Args as GotoArgs
from rio.apps.kassow.goto import MoveStatus, goto_home, move_tcp
from rio.apps.kassow.hardware import JEV_SERIAL
from rio.cfg import Camera
from rio.envs.env import make_env

CAMERA_KEY = "camera_jev"
TCP = "tcp"
MARKER_SIZES_M = {1: 0.030, 2: 0.040, 3: 0.050}
# Links the back marker may sit on; the fit keeps the one that explains the pixels best.
BACK_LINKS = ("link6", "link5")
FRAMES_PER_POSE = 3
SETTLE_S = 0.6
# Pose grid around the start pose (base frame offsets, tool-frame rotations).
GRID_XY_M = 0.08
GRID_Z_M = 0.04
TILT_DEG = 12.0
SPIN_DEG = 20.0
# Guards checked before anything moves.
MIN_TCP_Z_M = 0.30
MAX_DROP_M = 0.05
MAX_JOINT_DELTA_RAD = np.radians(60.0)
REACH_M = 0.005
# A fit is stored only below these leave-one-out errors at the markers.
MAX_LOO_RMS_MM = 3.0
MAX_LOO_MAX_MM = 6.0
MAX_MARKER_OFFSET_M = 0.25
MIN_SAMPLES = 8


def handeye_root() -> Path:
    """Folder of the hand-eye sessions and results."""
    return experiments_root() / "graph" / "calibration" / "handeye"


def result_path(camera: str = "jev") -> Path:
    """Where the accepted calibration of ``camera`` is stored."""
    return handeye_root() / f"{camera}.json"


# --- marker detection ---------------------------------------------------------------


def _detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def marker_centre(corners: np.ndarray) -> np.ndarray:
    """Pixel where the diagonals of the four corners cross (exact under perspective)."""
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return c.mean(axis=0)
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / denom
    return p + t * r


def detect_markers(image: np.ndarray) -> dict[int, np.ndarray]:
    """Corners (4 x 2) of every DICT_4X4_50 marker in an RGB or grey image, by id."""
    grey = image if image.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector().detectMarkers(grey)
    if ids is None:
        return {}
    return {int(i): np.asarray(c, dtype=float).reshape(4, 2) for c, i in zip(corners, ids.ravel(), strict=True)}


def detect_marker(image: np.ndarray, marker_id: int) -> np.ndarray | None:
    """Corners (4 x 2) of ``marker_id``, or ``None``."""
    return detect_markers(image).get(marker_id)


# --- camera model -------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Pinhole model of one colour stream; ``dist`` in OpenCV order (k1, k2, p1, p2, k3)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def realsense_intrinsics(serial: str, width: int = 640, height: int = 480) -> Intrinsics:
    """Factory intrinsics of the RGB stream of one RealSense, read without streaming.

    Raises:
        RuntimeError: If the camera or a ``width`` x ``height`` colour profile is not found.
    """
    rs: Any = importlib.import_module("pyrealsense2")  # compiled module without type stubs
    for device in rs.context().query_devices():
        if device.get_info(rs.camera_info.serial_number) != serial:
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                video = profile.as_video_stream_profile()
                if (video.width(), video.height()) == (width, height):
                    i = video.get_intrinsics()
                    return Intrinsics(width, height, i.fx, i.fy, i.ppx, i.ppy, [float(c) for c in i.coeffs])
    raise RuntimeError(f"RealSense {serial} with a {width}x{height} colour profile not found")


def project(points_cam: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Pixel coordinates of camera-frame points (N x 3)."""
    uv, _ = cv2.projectPoints(
        np.asarray(points_cam, dtype=float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intr.matrix, np.asarray(intr.dist)
    )
    return uv.reshape(-1, 2)


# --- samples and fit ----------------------------------------------------------------


@dataclass
class Sample:
    """One pose.

    ``frames`` holds pose6 (base frame) of every frame a marker may ride on: ``tcp`` is the
    measured TCP, links come from the URDF at ``joint_q``. ``markers`` maps the marker id
    (as a string, for JSON) to its corners.
    """

    name: str
    joint_q: list[float]
    frames: dict[str, list[float]]
    markers: dict[str, list[list[float]]]


@dataclass
class HandEye:
    """Camera pose in the base frame, marker offsets on their frames, and fit statistics."""

    camera_serial: str
    intrinsics: Intrinsics
    base_from_camera: list[list[float]]
    mounts: dict[str, str]
    """Marker id -> frame it rides on."""
    marker_offsets_m: dict[str, list[float]]
    """Marker id -> centre in its frame."""
    stats: dict

    @property
    def transform(self) -> np.ndarray:
        return np.asarray(self.base_from_camera, dtype=float)

    def pixel_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction (base frame) of the ray through pixel ``(u, v)``."""
        undistorted = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=float), self.intrinsics.matrix, np.asarray(self.intrinsics.dist)
        ).reshape(2)
        direction = self.transform[:3, :3] @ np.array([undistorted[0], undistorted[1], 1.0])
        return self.transform[:3, 3].copy(), direction / np.linalg.norm(direction)

    def pixel_to_plane(self, u: float, v: float, z: float) -> np.ndarray | None:
        """Base-frame point where the ray through ``(u, v)`` meets the plane ``z``; ``None`` if parallel or behind."""
        origin, direction = self.pixel_ray(u, v)
        if abs(direction[2]) < 1e-6:
            return None
        s = (z - origin[2]) / direction[2]
        return None if s <= 0 else origin + s * direction


@dataclass
class _Observation:
    sample: int
    marker: int
    rot: np.ndarray
    pos: np.ndarray
    uv: np.ndarray
    corners: np.ndarray


def _observations(samples: list[Sample], mounts: dict[int, str]) -> list[_Observation]:
    out = []
    for i, s in enumerate(samples):
        for key, corners in s.markers.items():
            marker = int(key)
            frame = mounts.get(marker)
            if frame is None or frame not in s.frames:
                continue
            pose = np.asarray(s.frames[frame], dtype=float)
            c = np.asarray(corners, dtype=float)
            out.append(_Observation(i, marker, Rotation.from_rotvec(pose[3:6]).as_matrix(), pose[:3], marker_centre(c), c))
    return out


def _camera_points(params: np.ndarray, obs: list[_Observation], ids: list[int]) -> np.ndarray:
    """Marker centres in the camera frame for ``[rotvec_cb, t_cb, offset per id]`` (camera <- base)."""
    r_cb = Rotation.from_rotvec(params[:3]).as_matrix()
    offsets = {m: params[6 + 3 * k : 9 + 3 * k] for k, m in enumerate(ids)}
    base = np.array([o.rot @ offsets[o.marker] + o.pos for o in obs])
    return base @ r_cb.T + params[3:6]


def _residuals(params, obs, ids, intr) -> np.ndarray:
    points = _camera_points(params, obs, ids)
    if np.any(points[:, 2] <= 1e-3):
        return np.full(2 * len(obs), 1e3)
    return (project(points, intr) - np.array([o.uv for o in obs])).ravel()


def _initial_guess(obs: list[_Observation], ids: list[int], intr: Intrinsics) -> np.ndarray:
    """Kabsch between frame origins and per-observation PnP marker positions, offsets 0."""
    cam, base = [], []
    for o in obs:
        half = MARKER_SIZES_M[o.marker] / 2.0
        square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=float)
        ok, _rvec, tvec = cv2.solvePnP(square, o.corners, intr.matrix, np.asarray(intr.dist), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            cam.append(tvec.ravel())
            base.append(o.pos)
    cam_a, base_a = np.array(cam), np.array(base)
    cb, cc = base_a.mean(axis=0), cam_a.mean(axis=0)
    u, _s, vt = np.linalg.svd((base_a - cb).T @ (cam_a - cc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r_cb = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return np.concatenate([Rotation.from_matrix(r_cb).as_rotvec(), cc - r_cb @ cb, np.zeros(3 * len(ids))])


def _solve(obs: list[_Observation], ids: list[int], intr: Intrinsics, x0: np.ndarray) -> np.ndarray:
    return least_squares(_residuals, x0, args=(obs, ids, intr), method="lm", x_scale="jac").x


def fit_handeye(samples: list[Sample], intr: Intrinsics, *, mounts: dict[int, str], camera_serial: str) -> HandEye:
    """Fit the camera pose and one offset per marker.

    ``stats`` holds, per sample and marker, the fit error and the leave-one-out error (the
    sample left out of the fit) in pixels and in millimetres at the marker's depth.

    Raises:
        ValueError: Fewer than ``MIN_SAMPLES`` samples with a mounted marker, or an unknown marker id.
    """
    unknown = sorted(set(mounts) - set(MARKER_SIZES_M))
    if unknown:
        raise ValueError(f"unknown marker ids {unknown}; known: {sorted(MARKER_SIZES_M)}")
    obs = _observations(samples, mounts)
    used = sorted({o.sample for o in obs})
    if len(used) < MIN_SAMPLES:
        raise ValueError(f"{len(used)} samples see a marker; the fit needs at least {MIN_SAMPLES}")
    ids = sorted({o.marker for o in obs})
    x = _solve(obs, ids, intr, _initial_guess(obs, ids, intr))
    points = _camera_points(x, obs, ids)
    uv = np.array([o.uv for o in obs])
    err_px = np.linalg.norm(project(points, intr) - uv, axis=1)
    mm_per_px = points[:, 2] / intr.fx * 1000.0
    loo_px = np.empty(len(obs))
    for i in used:
        rest = [o for o in obs if o.sample != i]
        mine = [k for k, o in enumerate(obs) if o.sample == i]
        xi = _solve(rest, ids, intr, x) if {o.marker for o in rest} == set(ids) else x
        pts = _camera_points(xi, [obs[k] for k in mine], ids)
        loo_px[mine] = np.linalg.norm(project(pts, intr) - uv[mine], axis=1)
    loo_mm = loo_px * mm_per_px
    r_cb = Rotation.from_rotvec(x[:3]).as_matrix()
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = r_cb.T
    base_from_camera[:3, 3] = -r_cb.T @ x[3:6]
    per_sample: dict[str, dict] = {}
    for k, o in enumerate(obs):
        per_sample.setdefault(samples[o.sample].name, {})[str(o.marker)] = {
            "fit_px": float(err_px[k]),
            "loo_px": float(loo_px[k]),
            "loo_mm": float(loo_mm[k]),
        }
    stats = {
        "fitted": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(used),
        "n_observations": {str(m): sum(o.marker == m for o in obs) for m in ids},
        "rms_px": float(np.sqrt(np.mean(err_px**2))),
        "loo_rms_px": float(np.sqrt(np.mean(loo_px**2))),
        "loo_rms_mm": float(np.sqrt(np.mean(loo_mm**2))),
        "loo_max_mm": float(np.max(loo_mm)),
        "camera_xyz_m": base_from_camera[:3, 3].round(4).tolist(),
        "samples": per_sample,
    }
    offsets = {str(m): x[6 + 3 * k : 9 + 3 * k].tolist() for k, m in enumerate(ids)}
    return HandEye(camera_serial, intr, base_from_camera.tolist(), {str(m): mounts[m] for m in ids}, offsets, stats)


def fit_best_back_link(
    samples: list[Sample],
    intr: Intrinsics,
    *,
    gripper_marker: int,
    back_marker: int | None,
    camera_serial: str,
    links: tuple[str, ...] = BACK_LINKS,
) -> HandEye:
    """``fit_handeye`` for each link the back marker may sit on; the fit with the lowest pixel rms wins."""
    if back_marker is None or not any(str(back_marker) in s.markers for s in samples):
        return fit_handeye(samples, intr, mounts={gripper_marker: TCP}, camera_serial=camera_serial)
    fits = []
    for link in links:
        try:
            fits.append(
                fit_handeye(samples, intr, mounts={gripper_marker: TCP, back_marker: link}, camera_serial=camera_serial)
            )
        except ValueError as error:
            logger.warning("back marker on {}: {}", link, error)
    if not fits:
        raise ValueError("no fit succeeded")
    for fit in fits:
        logger.info("back marker on {}: rms {:.2f} px", fit.mounts.get(str(back_marker)), fit.stats["rms_px"])
    return min(fits, key=lambda fit: fit.stats["rms_px"])


def fit_problems(fit: HandEye) -> list[str]:
    """Reasons not to store ``fit``; empty when it is good enough."""
    problems = []
    if fit.stats["loo_rms_mm"] > MAX_LOO_RMS_MM:
        problems.append(f"leave-one-out rms {fit.stats['loo_rms_mm']:.1f} mm exceeds {MAX_LOO_RMS_MM:.0f} mm")
    if fit.stats["loo_max_mm"] > MAX_LOO_MAX_MM:
        problems.append(f"leave-one-out max {fit.stats['loo_max_mm']:.1f} mm exceeds {MAX_LOO_MAX_MM:.0f} mm")
    for marker, offset in fit.marker_offsets_m.items():
        if np.linalg.norm(offset) > MAX_MARKER_OFFSET_M:
            problems.append(f"marker {marker} {np.linalg.norm(offset) * 100:.0f} cm from {fit.mounts[marker]} is implausible")
    return problems


def save_handeye(fit: HandEye, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(fit), indent=2) + "\n")
    return path


def load_handeye(path: Path | None = None) -> HandEye:
    """Stored calibration of the Jev camera (or ``path``).

    Raises:
        FileNotFoundError: If no calibration was accepted yet.
    """
    data = json.loads((path or result_path()).read_text())
    data["intrinsics"] = Intrinsics(**data["intrinsics"])
    return HandEye(**data)


# --- poses --------------------------------------------------------------------------


def calibration_poses(anchor: np.ndarray) -> list[np.ndarray]:
    """TCP goals (pose6) around ``anchor``: a 3 x 3 grid at two heights, each with its own tilt and spin.

    Every goal changes the orientation as well as the position, so the marker offsets are
    observable. The path is a serpentine and the lower layer runs it backwards, so every
    move is one grid step.
    """
    anchor = np.asarray(anchor, dtype=float)
    r0 = Rotation.from_rotvec(anchor[3:6])
    tilts = [(TILT_DEG, 0.0), (0.0, TILT_DEG), (-TILT_DEG, 0.0), (0.0, -TILT_DEG)]
    steps = (-GRID_XY_M, 0.0, GRID_XY_M)
    layer = [(dx, dy) for row, dy in enumerate(steps) for dx in (steps if row % 2 == 0 else steps[::-1])]
    path = [(dx, dy, GRID_Z_M) for dx, dy in layer] + [(dx, dy, -GRID_Z_M) for dx, dy in reversed(layer)]
    poses = []
    for k, (dx, dy, dz) in enumerate(path):
        ax, ay = tilts[k % len(tilts)]
        spin = SPIN_DEG if k % 2 == 0 else -SPIN_DEG
        delta = Rotation.from_euler("xyz", [ax, ay, spin], degrees=True)
        xyz = anchor[:3] + np.array([dx, dy, dz])
        poses.append(np.concatenate([xyz, cast(Rotation, r0 * delta).as_rotvec()]))
    return poses


def check_poses(kin, q_anchor: np.ndarray, anchor_tcp: np.ndarray, poses: list[np.ndarray]) -> list[tuple[np.ndarray, str]]:
    """``(pose, problem)`` for every goal; ``problem`` is empty when the goal may be driven.

    Uses the URDF IK from the start joints; nothing moves.
    """
    model = getattr(kin, "model", None)
    lower = getattr(model, "lowerPositionLimit", None)
    upper = getattr(model, "upperPositionLimit", None)
    offset = np.asarray(kin.fk(q_anchor), dtype=float)[:3] - np.asarray(anchor_tcp, dtype=float)[:3]
    floor = max(MIN_TCP_Z_M, float(anchor_tcp[2]) - MAX_DROP_M)
    out = []
    for pose in poses:
        if pose[2] < floor:
            out.append((pose, f"z {pose[2]:.3f} m below the floor {floor:.3f} m"))
            continue
        goal = np.concatenate([pose[:3] + offset, pose[3:6]])
        q = np.asarray(kin.ik(q_anchor, goal), dtype=float)
        reached = np.asarray(kin.fk(q), dtype=float)
        if float(np.linalg.norm(reached[:3] - goal[:3])) > REACH_M:
            out.append((pose, "not reachable"))
        elif lower is not None and upper is not None and (np.any(q < lower) or np.any(q > upper)):
            out.append((pose, "outside the joint limits"))
        elif float(np.max(np.abs(q - q_anchor))) > MAX_JOINT_DELTA_RAD:
            out.append((pose, f"turns a joint by {np.degrees(np.max(np.abs(q - q_anchor))):.0f} deg"))
        else:
            out.append((pose, ""))
    return out


def link_poses(kin, joint_q: np.ndarray, links: tuple[str, ...] = BACK_LINKS) -> dict[str, list[float]]:
    """Pose6 (base frame) of each URDF link at ``joint_q``."""
    pin: Any = importlib.import_module("pinocchio")
    q = np.asarray(joint_q, dtype=float)
    pin.forwardKinematics(kin.model, kin.data, q)
    pin.updateFramePlacements(kin.model, kin.data)
    out = {}
    for name in links:
        placement = kin.data.oMf[kin.model.getFrameId(name)]
        rotvec = Rotation.from_matrix(np.asarray(placement.rotation)).as_rotvec()
        out[name] = [*np.asarray(placement.translation, dtype=float).tolist(), *rotvec.tolist()]
    return out


# --- station run --------------------------------------------------------------------


@dataclass
class Args(GotoArgs):
    """Hand-eye calibration of the Jev camera. Only the arm, the gripper and the Jev camera are started."""

    command: Literal["run", "fit"] = "run"
    dry_run: bool = True
    """Check the markers and every pose without moving."""
    gripper_marker: int = 1
    """ArUco id on the gripper (1 = 30 mm, 2 = 40 mm, 3 = 50 mm)."""
    back_marker: int | None = 2
    """ArUco id on the wrist link behind the gripper rotation; None when there is none."""
    session: str | None = None
    """``fit``: session folder to fit again."""
    cameras: dict[str, Camera] = field(default_factory=lambda: {CAMERA_KEY: jev_camera()})


def _capture(env, wanted: set[int], after: float) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    """Newest frames after ``after``: the last image, the arm state, and the corners of every wanted marker
    that was seen in all ``FRAMES_PER_POSE`` frames (averaged)."""
    camera = env.cameras[CAMERA_KEY]
    seen: dict[int, list[np.ndarray]] = {m: [] for m in wanted}
    image, state, frames = None, {}, 0
    deadline = time.now() + 5.0
    last_stamp = after
    while frames < FRAMES_PER_POSE and time.now() < deadline:
        frame = camera.get_state()
        stamp = float(frame.get("camera_receive_timestamp") or frame.get("timestamp") or 0.0)
        if stamp <= last_stamp or frame.get("color") is None:
            time.sleep(0.03)
            continue
        last_stamp = stamp
        image = np.asarray(frame["color"]).copy()
        state = env.robot.arm.get_state()
        for marker, corners in detect_markers(image).items():
            if marker in seen:
                seen[marker].append(corners)
        frames += 1
    if image is None:
        raise RuntimeError("no fresh Jev camera frame")
    found = {m: np.mean(c, axis=0) for m, c in seen.items() if len(c) == frames == FRAMES_PER_POSE}
    return image, found, state


def load_session(folder: Path) -> tuple[list[Sample], Intrinsics, dict]:
    meta = json.loads((folder / "session.json").read_text())
    samples = [Sample(**json.loads(line)) for line in (folder / "samples.jsonl").read_text().splitlines() if line.strip()]
    return samples, Intrinsics(**meta["intrinsics"]), meta


def report_and_store(fit: HandEye) -> bool:
    """Log the fit; store it as the Jev calibration when it passes ``fit_problems``."""
    s = fit.stats
    for name, markers in s["samples"].items():
        cells = "  ".join(f"m{m}: loo {row['loo_mm']:.1f} mm" for m, row in markers.items())
        logger.info("{}  {}", name, cells)
    offsets = {m: np.round(np.asarray(o) * 1000, 1).tolist() for m, o in fit.marker_offsets_m.items()}
    logger.info(
        "samples {} (markers {}) | rms {:.2f} px | leave-one-out rms {:.1f} mm, max {:.1f} mm | camera at {} | "
        "mounts {} | offsets mm {}",
        s["n_samples"],
        s["n_observations"],
        s["rms_px"],
        s["loo_rms_mm"],
        s["loo_max_mm"],
        s["camera_xyz_m"],
        fit.mounts,
        offsets,
    )
    problems = fit_problems(fit)
    if problems:
        path = save_handeye(fit, result_path().with_suffix(".rejected.json"))
        logger.error("Calibration not stored: {} (kept for inspection in {})", "; ".join(problems), path)
        return False
    logger.success("Stored {}", save_handeye(fit, result_path()))
    return True


def run(args: Args) -> None:
    serial = str(args.cameras[CAMERA_KEY].serial or JEV_SERIAL)
    intr = realsense_intrinsics(serial)
    wanted = {args.gripper_marker} | ({args.back_marker} if args.back_marker is not None else set())
    logger.info("Jev camera {}: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}", serial, intr.fx, intr.fy, intr.cx, intr.cy)
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())), env:
        arm = env.robot.arm
        kin = getattr(arm, "_kin", None)
        if kin is None:
            raise RuntimeError("the arm has no kinematics")
        state = arm.get_state()
        q0 = np.asarray(state["joint_q"], dtype=float)
        anchor = np.asarray(state["eef_pose"], dtype=float)
        logger.info("Start pose tcp={} (the arm returns here)", np.round(anchor[:3], 3).tolist())
        _image, found, _ = _capture(env, wanted, time.now())
        for marker in sorted(wanted):
            if marker in found:
                logger.info("marker {} seen at {} px", marker, np.round(marker_centre(found[marker]), 1).tolist())
            else:
                logger.warning("marker {} not seen from the start pose", marker)
        if not found:
            raise RuntimeError("no marker visible from the start pose; turn the gripper marker toward the Jev camera")
        checked = check_poses(kin, q0, anchor, calibration_poses(anchor))
        goals = [pose for pose, problem in checked if not problem]
        for i, (pose, problem) in enumerate(checked):
            logger.info("pose {:02d} xyz={} {}", i, np.round(pose[:3], 3).tolist(), problem or "ok")
        if len(goals) < MIN_SAMPLES:
            raise RuntimeError(f"only {len(goals)} poses pass the checks; move the start pose toward the table centre")
        if args.dry_run:
            logger.info("Dry run: {} of {} poses would be driven. Add --no-dry-run to drive them.", len(goals), len(checked))
            return
        folder = handeye_root() / "jev" / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(parents=True)
        meta = {
            "camera_serial": serial,
            "gripper_marker": args.gripper_marker,
            "back_marker": args.back_marker,
            "intrinsics": asdict(intr),
        }
        (folder / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
        input(f"Drive {len(goals)} poses at {args.home_joint_speed:.2f} rad/s? E-stop in reach. Enter = start ")
        samples: list[Sample] = []
        try:
            for i, pose in enumerate(goals):
                status = move_tcp(env, pose[:3], pose[3:6], home_joint_speed=args.home_joint_speed)
                if status is MoveStatus.ALARM:
                    logger.error("Arm alarm at pose {:02d}; stopping", i)
                    break
                if status is not MoveStatus.LANDED:
                    logger.warning("pose {:02d} not reached; skipped", i)
                    continue
                time.sleep(SETTLE_S)
                image, found, st = _capture(env, wanted, time.now())
                name = f"{i:02d}"
                cv2.imwrite(str(folder / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not found:
                    logger.warning("pose {} no marker in every frame; skipped", name)
                    continue
                q = np.asarray(st["joint_q"], dtype=float)
                sample = Sample(
                    name,
                    q.tolist(),
                    {TCP: np.asarray(st["eef_pose"], dtype=float)[:6].tolist(), **link_poses(kin, q)},
                    {str(m): c.tolist() for m, c in found.items()},
                )
                with (folder / "samples.jsonl").open("a") as handle:
                    handle.write(json.dumps(asdict(sample)) + "\n")
                samples.append(sample)
                logger.info("pose {} markers {} ({} samples)", name, sorted(found), len(samples))
        finally:
            logger.info("Returning to the start pose")
            goto_home(env, {"joint_q": q0, "eef_pose": anchor}, home_joint_speed=args.home_joint_speed, set_gripper=False)
        logger.info("Session {}", folder)
        if len(samples) < MIN_SAMPLES:
            logger.error("{} samples are too few to fit", len(samples))
            return
        fit = fit_best_back_link(
            samples, intr, gripper_marker=args.gripper_marker, back_marker=args.back_marker, camera_serial=serial
        )
        report_and_store(fit)


def main(args: Args) -> None:
    if args.command == "fit":
        if not args.session:
            raise SystemExit("--session is required for fit")
        samples, intr, meta = load_session(Path(args.session))
        fit = fit_best_back_link(
            samples,
            intr,
            gripper_marker=int(meta["gripper_marker"]),
            back_marker=meta.get("back_marker"),
            camera_serial=str(meta["camera_serial"]),
        )
        report_and_store(fit)
        return
    run(args)


def cli() -> None:
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)


if __name__ == "__main__":
    cli()
