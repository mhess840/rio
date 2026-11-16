# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Eye-in-hand calibration of the wrist camera: where it sits on the TCP.

The wrist camera moves with the arm but never relative to the TCP. With that
fixed transform, every wrist photo taken while the arm stands still maps into
the base frame: ``base_from_camera = base_from_tcp(eef_pose) @ tcp_from_camera``.

One ArUco marker (DICT_4X4_50, ids 1-3 of ``calibration/print``) lies flat on the
table where the wrist camera sees it from the start pose. The arm drives the
same 3 x 3 x 2 pattern as the Jev calibration around its start pose (tilt and
spin per pose, every pose checked with the URDF IK, ``--dry-run`` by default),
records the TCP pose and the marker corners, and returns. The fit starts from
a closed-form AX = XB solution (Park & Martin) and refines the camera mount and
the marker pose together on the reprojection error.

    uv run python -m rio.apps.kassow.graph.wristeye                       # check marker and poses
    uv run python -m rio.apps.kassow.graph.wristeye --no-dry-run          # drive and fit
    uv run python -m rio.apps.kassow.graph.wristeye --command fit --session <folder>
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro
from loguru import logger
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rio.apps.kassow.goto import Args as GotoArgs
from rio.apps.kassow.goto import MoveStatus, goto_home, move_tcp
from rio.apps.kassow.hardware import WRIST_SERIAL
from rio.cfg import Camera

from .handeye import (
    MARKER_SIZES_M,
    SETTLE_S,
    Intrinsics,
    calibration_poses,
    check_poses,
    detect_markers,
    handeye_root,
    realsense_intrinsics,
)

CAMERA_KEY = "camera_wrist"
FRAMES_PER_POSE = 3
MIN_SAMPLES = 8
MAX_RMS_PX = 2.0
MAX_LOO_RMS_MM = 3.0
MAX_LOO_MAX_MM = 6.0
"""Leave-one-out: fit without one view, predict where the marker appears in it, as distance on the marker plane."""


def wrist_camera() -> Camera:
    """The wrist D435i as the policy station streams it (colour 640 x 480)."""
    return Camera(
        addr="127.0.0.1:5130",
        cam_type="Realsense",
        serial=WRIST_SERIAL,
        model="D400",
        enable_depth=False,
        hardware_reset=False,
        resolution=(480, 640),
        resolution_depth=(480, 640),
    )


def result_path() -> Path:
    return handeye_root() / "wrist.json"


def pose_matrix(pose6) -> np.ndarray:
    """4 x 4 transform of ``x, y, z, rx, ry, rz`` (rotation vector)."""
    pose = np.asarray(pose6, dtype=float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    out[:3, 3] = pose[:3]
    return out


def marker_points(size_m: float) -> np.ndarray:
    """Marker corners in its own frame, in ArUco order (top-left, top-right, bottom-right, bottom-left)."""
    h = size_m / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


@dataclass
class Sample:
    name: str
    joint_q: list[float]
    tcp: list[float]
    """TCP pose6 as the controller reports it."""
    corners: list[list[float]]
    """Marker corners (4 x 2) in the wrist image, averaged over the frames."""


@dataclass
class WristEye:
    """Mount of the wrist camera on the TCP, the marker pose used, and fit statistics."""

    camera_serial: str
    intrinsics: Intrinsics
    tcp_from_camera: list[list[float]]
    marker: int
    base_from_marker: list[list[float]]
    stats: dict = field(default_factory=dict)

    def base_from_camera(self, eef_pose6) -> np.ndarray:
        """Camera pose in the base frame for a TCP pose (take photos only while the arm stands still)."""
        return pose_matrix(eef_pose6) @ np.asarray(self.tcp_from_camera, dtype=float)

    def pixel_to_plane(self, eef_pose6, u: float, v: float, z: float) -> np.ndarray | None:
        """Base-frame point where the ray through pixel ``(u, v)`` meets the plane ``z``."""
        transform = self.base_from_camera(eef_pose6)
        intr = self.intrinsics
        n = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), intr.matrix, np.asarray(intr.dist)).reshape(2)
        direction = transform[:3, :3] @ np.array([n[0], n[1], 1.0])
        origin = transform[:3, 3]
        if abs(direction[2]) < 1e-6:
            return None
        s = (z - origin[2]) / direction[2]
        return None if s <= 0 else origin + s * direction


def save_wristeye(fit: WristEye, path: Path | None = None) -> Path:
    file = path or result_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(asdict(fit), indent=2) + "\n")
    return file


def load_wristeye(path: Path | None = None) -> WristEye:
    """Stored wrist calibration.

    Raises:
        FileNotFoundError: Not calibrated yet.
    """
    data = json.loads((path or result_path()).read_text())
    data["intrinsics"] = Intrinsics(**data["intrinsics"])
    return WristEye(**data)


# --- fit -----------------------------------------------------------------------------


def _pnp(corners: np.ndarray, size_m: float, intr: Intrinsics) -> np.ndarray:
    ok, rvec, tvec = cv2.solvePnP(
        marker_points(size_m),
        np.asarray(corners, dtype=np.float64),
        intr.matrix,
        np.asarray(intr.dist),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    out = np.eye(4)
    out[:3, :3] = cv2.Rodrigues(rvec)[0]
    out[:3, 3] = tvec.reshape(3)
    return out


def _params(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]])


def _matrix(params: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    out[:3, 3] = params[3:6]
    return out


def solve_ax_xb(base_from_tcp: list[np.ndarray], camera_from_marker: list[np.ndarray]) -> np.ndarray:
    """Closed-form ``tcp_from_camera`` (Park & Martin) from all pairs of views of a fixed marker.

    For views i, j: ``A X = X B`` with ``A = inv(T_j) T_i`` (TCP motion) and
    ``B = C_j inv(C_i)`` (camera motion seen through the marker).
    """
    pairs = []
    for i in range(len(base_from_tcp)):
        for j in range(i + 1, len(base_from_tcp)):
            a = np.linalg.inv(base_from_tcp[j]) @ base_from_tcp[i]
            b = camera_from_marker[j] @ np.linalg.inv(camera_from_marker[i])
            pairs.append((a, b))
    m = np.zeros((3, 3))
    for a, b in pairs:
        alpha = Rotation.from_matrix(a[:3, :3]).as_rotvec()
        beta = Rotation.from_matrix(b[:3, :3]).as_rotvec()
        m += np.outer(beta, alpha)
    w, v = np.linalg.eigh(m.T @ m)
    rotation = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ v.T @ m.T
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([rotation @ b[:3, 3] - a[:3, 3] for a, b in pairs])
    translation, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, translation
    return out


def _residuals(params: np.ndarray, samples: list[Sample], points: np.ndarray, intr: Intrinsics) -> np.ndarray:
    tcp_from_camera, base_from_marker = _matrix(params[:6]), _matrix(params[6:])
    out = []
    for s in samples:
        camera_from_marker = np.linalg.inv(pose_matrix(s.tcp) @ tcp_from_camera) @ base_from_marker
        projected, _ = cv2.projectPoints(
            points, cv2.Rodrigues(camera_from_marker[:3, :3])[0], camera_from_marker[:3, 3], intr.matrix, np.asarray(intr.dist)
        )
        out.append((projected.reshape(-1, 2) - np.asarray(s.corners)).ravel())
    return np.concatenate(out)


def fit_wristeye(
    samples: list[Sample], intr: Intrinsics, *, marker: int, camera_serial: str, size_m: float | None = None, loo: bool = True
) -> WristEye:
    """Camera mount and marker pose from the samples.

    Raises:
        ValueError: Fewer than ``MIN_SAMPLES`` samples or an unknown marker.
    """
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"{len(samples)} samples; at least {MIN_SAMPLES} are needed")
    if size_m is None and marker not in MARKER_SIZES_M:
        raise ValueError(f"marker {marker} has no known size; known: {sorted(MARKER_SIZES_M)}")
    size = float(size_m or MARKER_SIZES_M[marker])
    points = marker_points(size)
    base_from_tcp = [pose_matrix(s.tcp) for s in samples]
    camera_from_marker = [_pnp(np.asarray(s.corners), size, intr) for s in samples]
    start = solve_ax_xb(base_from_tcp, camera_from_marker)
    guesses = [b @ start @ c for b, c in zip(base_from_tcp, camera_from_marker, strict=True)]
    marker_guess = guesses[0].copy()
    marker_guess[:3, 3] = np.median([g[:3, 3] for g in guesses], axis=0)
    solution = least_squares(
        _residuals, np.concatenate([_params(start), _params(marker_guess)]), args=(samples, points, intr), method="lm"
    )
    tcp_from_camera, base_from_marker = _matrix(solution.x[:6]), _matrix(solution.x[6:])
    residual = solution.fun.reshape(-1, 2)
    per_sample = np.linalg.norm(residual, axis=1).reshape(len(samples), 4).mean(axis=1)
    centres = []
    for b, c in zip(base_from_tcp, camera_from_marker, strict=True):
        centres.append((b @ tcp_from_camera @ c)[:3, 3])
    spread = np.linalg.norm(np.asarray(centres) - base_from_marker[:3, 3], axis=1) * 1000.0
    loo_mm = []
    if loo:
        for i, held_out in enumerate(samples):
            refit = least_squares(_residuals, solution.x, args=(samples[:i] + samples[i + 1 :], points, intr), method="lm")
            error = _residuals(refit.x, [held_out], points, intr).reshape(-1, 2)
            camera = pose_matrix(held_out.tcp) @ _matrix(refit.x[:6])
            distance = float(np.linalg.norm(_matrix(refit.x[6:])[:3, 3] - camera[:3, 3]))
            loo_mm.append(float(np.sqrt(np.mean(np.sum(error**2, axis=1)))) * distance / intr.fx * 1000.0)
    stats = {
        "fitted": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "marker_size_m": size,
        "loo_rms_mm": float(np.sqrt(np.mean(np.square(loo_mm)))) if loo_mm else None,
        "loo_max_mm": float(np.max(loo_mm)) if loo_mm else None,
        "rms_px": float(np.sqrt(np.mean(residual**2))),
        "per_sample_px": {s.name: round(float(p), 2) for s, p in zip(samples, per_sample, strict=True)},
        "marker_spread_rms_mm": float(np.sqrt(np.mean(spread**2))),
        "marker_spread_max_mm": float(np.max(spread)),
        "camera_offset_mm": np.round(tcp_from_camera[:3, 3] * 1000.0, 1).tolist(),
        "camera_axis_in_tcp": np.round(tcp_from_camera[:3, 2], 3).tolist(),
        "marker_xyz_m": np.round(base_from_marker[:3, 3], 4).tolist(),
    }
    return WristEye(camera_serial, intr, tcp_from_camera.tolist(), marker, base_from_marker.tolist(), stats)


def fit_problems(fit: WristEye) -> list[str]:
    s, problems = fit.stats, []
    if s["rms_px"] > MAX_RMS_PX:
        problems.append(f"reprojection rms {s['rms_px']:.2f} px > {MAX_RMS_PX}")
    if s.get("loo_rms_mm") is None:
        problems.append("no leave-one-out check")
    elif s["loo_rms_mm"] > MAX_LOO_RMS_MM or s["loo_max_mm"] > MAX_LOO_MAX_MM:
        problems.append(
            f"leave-one-out {s['loo_rms_mm']:.1f} mm rms / {s['loo_max_mm']:.1f} mm max > {MAX_LOO_RMS_MM} / {MAX_LOO_MAX_MM}"
        )
    if np.linalg.norm(s["camera_offset_mm"]) > 250:
        problems.append(f"camera {np.linalg.norm(s['camera_offset_mm']):.0f} mm from the TCP")
    return problems


# --- station run ---------------------------------------------------------------------


@dataclass
class Args(GotoArgs):
    """Eye-in-hand calibration of the wrist camera. Only the arm, the gripper and the wrist camera run."""

    command: Literal["run", "fit"] = "run"
    dry_run: bool = True
    """Check the marker and every pose without moving."""
    marker: int | None = None
    """ArUco id on the table; unset: the largest known id the wrist camera sees."""
    marker_size_m: float | None = None
    """Measured side of the printed marker (black square); unset: the nominal size of its id."""
    session: str | None = None
    """``fit``: session folder to fit again."""
    cameras: dict[str, Camera] = field(default_factory=lambda: {CAMERA_KEY: wrist_camera()})


def _capture(env, marker: int | None) -> tuple[np.ndarray | None, int | None, np.ndarray | None, dict]:
    """Last image, the marker id, its corners averaged over ``FRAMES_PER_POSE`` fresh frames, and the arm state."""
    from rio_hw import time

    camera = env.cameras[CAMERA_KEY]
    seen: dict[int, list[np.ndarray]] = {}
    image, state, frames, last = None, {}, 0, time.now()
    deadline = time.now() + 5.0
    while frames < FRAMES_PER_POSE and time.now() < deadline:
        frame = camera.get_state()
        stamp = float(frame.get("camera_receive_timestamp") or frame.get("timestamp") or 0.0)
        if stamp <= last or frame.get("color") is None:
            time.sleep(0.03)
            continue
        last, image, state = stamp, np.asarray(frame["color"]).copy(), env.robot.arm.get_state()
        for found, corners in detect_markers(image).items():
            if found in MARKER_SIZES_M and (marker is None or found == marker):
                seen.setdefault(found, []).append(corners)
        frames += 1
    complete = {m: c for m, c in seen.items() if len(c) == frames == FRAMES_PER_POSE}
    if not complete:
        return image, None, None, state
    chosen = marker if marker in complete else max(complete, key=lambda m: MARKER_SIZES_M[m])
    return image, chosen, np.mean(complete[chosen], axis=0), state


def run(args: Args) -> WristEye | None:
    from rio_hw import time
    from rio_hw.middleware import ServerManager

    from rio.envs.env import make_env

    intr = realsense_intrinsics(WRIST_SERIAL)
    logger.info("Wrist camera {}: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}", WRIST_SERIAL, intr.fx, intr.fy, intr.cx, intr.cy)
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())), env:
        arm = env.robot.arm
        kin = getattr(arm, "_kin", None)
        if kin is None:
            raise RuntimeError("the arm has no kinematics")
        time.sleep(1.0)
        state = arm.get_state()
        q0 = np.asarray(state["joint_q"], dtype=float)
        anchor = np.asarray(state["eef_pose"], dtype=float)
        image, marker, corners, _ = _capture(env, args.marker)
        folder = handeye_root() / "wrist" / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(parents=True)
        if image is not None:
            cv2.imwrite(str(folder / "start.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if marker is None or corners is None:
            raise RuntimeError(f"no ArUco marker (ids {sorted(MARKER_SIZES_M)}) in the wrist view; see {folder / 'start.jpg'}")
        logger.info("marker {} seen at {} px from the start pose", marker, np.round(corners.mean(axis=0), 1).tolist())
        checked = check_poses(kin, q0, anchor, calibration_poses(anchor))
        goals = [pose for pose, problem in checked if not problem]
        for i, (pose, problem) in enumerate(checked):
            logger.info("pose {:02d} xyz={} {}", i, np.round(pose[:3], 3).tolist(), problem or "ok")
        if len(goals) < MIN_SAMPLES:
            raise RuntimeError(f"only {len(goals)} poses pass the checks")
        if args.dry_run:
            logger.info("Dry run: {} of {} poses would be driven. Add --no-dry-run to drive them.", len(goals), len(checked))
            return None
        (folder / "session.json").write_text(
            json.dumps({"camera_serial": WRIST_SERIAL, "marker": marker, "intrinsics": asdict(intr)}, indent=2) + "\n"
        )
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
                image, seen, corners, st = _capture(env, marker)
                name = f"{i:02d}"
                if image is not None:
                    cv2.imwrite(str(folder / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if seen is None or corners is None:
                    logger.warning("pose {} marker not in every frame; skipped", name)
                    continue
                sample = Sample(
                    name,
                    np.asarray(st["joint_q"], dtype=float).tolist(),
                    np.asarray(st["eef_pose"], dtype=float)[:6].tolist(),
                    corners.tolist(),
                )
                with (folder / "samples.jsonl").open("a") as handle:
                    handle.write(json.dumps(asdict(sample)) + "\n")
                samples.append(sample)
                logger.info("pose {} ok ({} samples)", name, len(samples))
        finally:
            logger.info("Returning to the start pose")
            goto_home(env, {"joint_q": q0, "eef_pose": anchor}, home_joint_speed=args.home_joint_speed, set_gripper=False)
        logger.info("Session {}", folder)
        return fit_and_store(samples, intr, marker, args.marker_size_m)


def fit_and_store(samples: list[Sample], intr: Intrinsics, marker: int, size_m: float | None = None) -> WristEye | None:
    if len(samples) < MIN_SAMPLES:
        logger.error("{} samples are too few to fit", len(samples))
        return None
    fit = fit_wristeye(samples, intr, marker=marker, camera_serial=WRIST_SERIAL, size_m=size_m)
    s = fit.stats
    logger.info(
        "samples {} | rms {:.2f} px | leave-one-out {:.1f} mm rms, {:.1f} mm max | single-view marker spread rms {:.1f} mm, "
        "max {:.1f} mm | camera offset {} mm, axis {} | marker at {}",
        s["n_samples"],
        s["rms_px"],
        s["loo_rms_mm"],
        s["loo_max_mm"],
        s["marker_spread_rms_mm"],
        s["marker_spread_max_mm"],
        s["camera_offset_mm"],
        s["camera_axis_in_tcp"],
        s["marker_xyz_m"],
    )
    problems = fit_problems(fit)
    if problems:
        path = save_wristeye(fit, result_path().with_suffix(".rejected.json"))
        logger.error("Calibration not stored: {} (kept in {})", "; ".join(problems), path)
        return None
    logger.success("Stored {}", save_wristeye(fit))
    return fit


def main(args: Args) -> WristEye | None:
    if args.command == "fit":
        if not args.session:
            raise SystemExit("--session is required for fit")
        folder = Path(args.session)
        meta = json.loads((folder / "session.json").read_text())
        samples = [Sample(**json.loads(line)) for line in (folder / "samples.jsonl").read_text().splitlines() if line.strip()]
        return fit_and_store(samples, Intrinsics(**meta["intrinsics"]), int(meta["marker"]), args.marker_size_m)
    return run(args)


def cli() -> None:
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)


if __name__ == "__main__":
    cli()
