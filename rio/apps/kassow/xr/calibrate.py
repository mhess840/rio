# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the headset to the robot with a Touch controller held in the gripper.

The headset view must already run (``uv run python -m rio.apps.kassow.xr --headset``).
Put the controller in the closed gripper, park the arm above the table where the
headset sees the controller, then:

    uv run python -m rio.apps.kassow.xr.calibrate --dry-run   # plan only, no motion
    uv run python -m rio.apps.kassow.xr.calibrate              # Enter before every moveJ

Only the arm is connected. The gripper node is left out on purpose: its start-up
would open the AG95 and drop the controller. The arm only goes up from the start
pose and tilts about the TCP. The last pose is the start pose again; a different
controller reading there means the controller slipped or the headset jumped.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time as wall
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager
from scipy.spatial.transform import Rotation

from rio.apps.kassow.goto import Args as GotoArgs
from rio.apps.kassow.goto import MoveStatus, move_tcp, solve_ik
from rio.envs.env import make_env
from rio.xr.auth import read_token, token_path
from rio.xr.publisher import DEFAULT_URL, XrPublisher
from rio.xr.scene import Layer, Lines, Points
from rio.xr.tracker_fit import TrackerFit, fit_tracker

from .cli import calibration_path

# Base-frame shift (m) and tilt about the TCP's own axes (deg, xyz), before scaling.
# Every target is at or above the start height. The last one is the start again.
TARGETS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.00, 0.00, 0.00), (0.0, 0.0, 0.0)),
    ((0.10, 0.00, 0.05), (0.0, 0.0, 0.0)),
    ((-0.10, 0.00, 0.05), (0.0, 0.0, 0.0)),
    ((0.00, 0.10, 0.10), (0.0, 0.0, 0.0)),
    ((0.00, -0.10, 0.10), (0.0, 0.0, 0.0)),
    ((0.00, 0.00, 0.15), (0.0, 0.0, 1.2)),
    ((0.00, 0.00, 0.15), (0.0, 0.0, -1.2)),
    ((0.05, 0.05, 0.10), (1.0, 0.0, 0.0)),
    ((-0.05, -0.05, 0.10), (-1.0, 0.0, 0.0)),
    ((0.05, -0.05, 0.10), (0.0, 1.0, 0.0)),
    ((-0.05, 0.05, 0.10), (0.0, -1.0, 0.0)),
    ((0.00, 0.00, 0.00), (0.0, 0.0, 0.0)),
)
MAX_AGE_S = 0.15
MIN_SAMPLES = 10
MAX_SAMPLE_STD_M = 0.0015
SLIP_M = 0.003
# Controller orientation scatters by about 1-1.5 deg per pose; 1 deg rejected a good run.
SLIP_RAD = np.radians(3.0)
ACCEPT_RMS_M = 0.005


@dataclass
class Args(GotoArgs):
    """Calibrate the headset with the controller in the gripper. Enter before every move."""

    device: str = "right"
    """Controller in the gripper: ``right`` or ``left``."""
    dry_run: bool = False
    """Read the arm and the controller, plan and IK-check every target, move nothing."""
    shift_scale: float = 1.0
    """Scales the base-frame shifts of :data:`TARGETS` (1.0: up to 10 cm sideways, 15 cm up)."""
    tilt_deg: float = 25.0
    """Tilt about the TCP for a unit entry in :data:`TARGETS`."""
    settle_s: float = 1.0
    """Wait after a move before sampling."""
    sample_s: float = 1.0
    """Averaging window for the controller and the TCP."""
    wait_s: float = 120.0
    """How long to wait at a pose for the controller to be tracked before skipping it."""
    accept_any: bool = False
    """Apply and save the fit even when it fails the slip or residual check."""
    url: str = DEFAULT_URL
    # The gripper stays disconnected: its start-up would open the AG95 and drop the controller.
    gripper: str | None = None
    gripper_cfg: None = None  # ty: ignore[invalid-attribute-override]


def _http(url: str, payload: dict | None = None) -> object:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        headers["Authorization"] = f"Bearer {read_token(token_path())}"
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2.0) as response:
        return json.loads(response.read())


def read_device(url: str, device: str) -> tuple[np.ndarray | None, str, str]:
    """One controller reading: ``(xr_from_controller or None, reason, space)``."""
    tracked = _http(f"{url}/tracked")
    assert isinstance(tracked, dict)
    entry = tracked.get("devices", {}).get(device)
    space = str(tracked.get("space", ""))
    if entry is None:
        return None, f"headset reports no {device} controller", space
    if not entry["tracked"]:
        return None, f"{device} controller is not tracked (out of view or asleep)", space
    if entry["age_s"] > MAX_AGE_S:
        return None, f"{device} controller reading is {entry['age_s']:.1f} s old", space
    return np.asarray(entry["matrix"], dtype=float), "", space


def wait_for_device(url: str, device: str, timeout_s: float) -> None:
    """Block until the headset tracks ``device``, telling the operator why it waits.

    Raises:
        RuntimeError: Still not tracked after ``timeout_s``.
    """
    end = wall.monotonic() + timeout_s
    last = ""
    while True:
        pose, reason, _ = read_device(url, device)
        if pose is not None:
            if last:
                logger.info("{} controller tracked again", device)
            return
        if reason.split(" reading is")[0] != last.split(" reading is")[0]:
            logger.warning("Waiting for the headset: {}. Headset on, no menu open.", reason)
            last = reason
        if wall.monotonic() > end:
            raise RuntimeError(reason)
        wall.sleep(0.2)


def sample_device(url: str, device: str, seconds: float) -> np.ndarray:
    """Average controller pose over ``seconds``.

    Raises:
        RuntimeError: Too few tracked samples, or the controller moved while sampling.
    """
    poses = []
    reasons = set()
    end = wall.monotonic() + seconds
    while wall.monotonic() < end:
        pose, reason, _ = read_device(url, device)
        if pose is None:
            reasons.add(reason)
        else:
            poses.append(pose)
        wall.sleep(1.0 / 30.0)
    if len(poses) < MIN_SAMPLES:
        raise RuntimeError(f"only {len(poses)} tracked samples: {'; '.join(sorted(reasons)) or 'no reason given'}")
    positions = np.stack([pose[:3, 3] for pose in poses])
    spread = float(np.max(np.std(positions, axis=0)))
    if spread > MAX_SAMPLE_STD_M:
        raise RuntimeError(f"controller moved while sampling ({spread * 1000:.1f} mm std)")
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in poses])).mean().as_matrix()
    out[:3, 3] = positions.mean(axis=0)
    return out


def sample_tcp(env, seconds: float) -> np.ndarray:
    """Mean controller TCP as a 4 x 4 ``base_from_tcp``."""
    poses = []
    end = time.now() + seconds
    while time.now() < end:
        poses.append(np.asarray(env.robot.arm.get_state()["eef_pose"], dtype=float).reshape(6))
        time.sleep(0.02)
    stacked = np.stack(poses)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(stacked[:, 3:]).mean().as_matrix()
    out[:3, 3] = stacked[:, :3].mean(axis=0)
    return out


def plan_targets(start: np.ndarray, args: Args) -> list[np.ndarray]:
    """Target TCP pose6 for every entry of :data:`TARGETS`, relative to ``start``."""
    base_from_start = Rotation.from_rotvec(start[3:])
    out = []
    for shift, tilt in TARGETS:
        xyz = start[:3] + np.asarray(shift) * args.shift_scale
        turn = Rotation.from_rotvec(np.radians(np.asarray(tilt) * args.tilt_deg))
        target = cast(Rotation, base_from_start * turn)
        out.append(np.concatenate([xyz, target.as_rotvec()]))
    return out


def check_plan(kin, start_q: np.ndarray, targets: list[np.ndarray]) -> list[str]:
    """IK reasons per target, empty strings where the target is reachable from the previous one."""
    reasons = []
    q = start_q
    for target in targets:
        solved, reason = solve_ik(kin, target, seed_q=q, from_q=q)
        reasons.append(reason)
        if solved is not None:
            q = solved
    return reasons


def report(fit: TrackerFit) -> None:
    for index, (position, rotation) in enumerate(zip(fit.position_errors_m, fit.rotation_errors_rad, strict=True)):
        logger.info("  pose {:2d}: {:5.1f} mm  {:4.2f} deg", index, position * 1000, np.degrees(rotation))
    logger.info("Residual rms {:.1f} mm, max {:.1f} mm", fit.rms_m * 1000, fit.max_m * 1000)
    grip = fit.tcp_from_controller
    logger.info(
        "Controller in the gripper: {} mm from the TCP",
        np.round(grip[:3, 3] * 1000, 1).tolist(),
    )


def check_layer(base_from_tcp: list[np.ndarray]) -> Layer:
    """Recorded TCP positions, so the overlay can be compared with the arm right after the fit."""
    points = np.stack([pose[:3, 3] for pose in base_from_tcp])
    return Layer(
        "calibration_poses",
        "Calibration TCPs",
        (Points(points, (1.0, 0.2, 0.8, 1.0), 0.03), Lines(np.repeat(points, 2, axis=0)[1:-1], (1.0, 0.2, 0.8, 0.6))),
    )


def run(args: Args, env) -> None:
    arm = env.robot.arm
    kin = getattr(arm, "_kin", None)
    if kin is None:
        raise SystemExit("the arm has no kinematics; cannot plan the calibration poses")
    state = arm.get_state()
    start = np.asarray(state["eef_pose"], dtype=float).reshape(6)
    start_q = np.asarray(state["joint_q"], dtype=float)
    targets = plan_targets(start, args)
    reasons = check_plan(kin, start_q, targets)
    logger.info("Start TCP {}", np.round(start[:3], 3).tolist())
    for index, (target, reason) in enumerate(zip(targets, reasons, strict=True)):
        delta = (target[:3] - start[:3]) * 100
        tilt = np.degrees(cast(Rotation, Rotation.from_rotvec(start[3:]).inv() * Rotation.from_rotvec(target[3:])).magnitude())
        logger.info(
            "  {:2d}: dx {:+5.1f} dy {:+5.1f} dz {:+5.1f} cm, tilt {:4.1f} deg  {}",
            index,
            *delta,
            tilt,
            reason or "ok",
        )
    if args.dry_run:
        logger.info("Dry run: nothing moved")
        return
    if any(reasons):
        raise SystemExit("some targets are unreachable; lower --shift-scale / --tilt-deg or move the start pose")

    base_from_tcp: list[np.ndarray] = []
    xr_from_controller: list[np.ndarray] = []
    for index, target in enumerate(targets):
        answer = input(f"Enter: moveJ to pose {index}/{len(targets) - 1}, q: stop (E-Stop ready) ").strip().lower()
        if answer == "q":
            raise SystemExit("stopped by the operator; nothing saved")
        status = move_tcp(env, target[:3], target[3:], home_joint_speed=args.home_joint_speed)
        if status is MoveStatus.ALARM:
            raise SystemExit("KORD alarm; nothing saved")
        if status is not MoveStatus.LANDED:
            logger.warning("Pose {} not reached; skipped", index)
            continue
        time.sleep(args.settle_s)
        try:
            wait_for_device(args.url, args.device, args.wait_s)
            controller = sample_device(args.url, args.device, args.sample_s)
        except RuntimeError as error:
            logger.warning("Pose {} skipped: {}", index, error)
            continue
        base_from_tcp.append(sample_tcp(env, 0.3))
        xr_from_controller.append(controller)
        logger.info("Pose {} recorded ({} so far)", index, len(base_from_tcp))

    fit = fit_tracker(base_from_tcp, xr_from_controller)
    report(fit)
    first, last = xr_from_controller[0], xr_from_controller[-1]
    slip_m = float(np.linalg.norm(first[:3, 3] - last[:3, 3]))
    slip_rad = float(Rotation.from_matrix(first[:3, :3].T @ last[:3, :3]).magnitude())
    logger.info("Start vs end: {:.1f} mm, {:.2f} deg", slip_m * 1000, np.degrees(slip_rad))
    problems = []
    if slip_m > SLIP_M or slip_rad > SLIP_RAD:
        problems.append("the controller reading at the start pose changed: slipped in the gripper or headset jumped")
    if fit.rms_m > ACCEPT_RMS_M:
        problems.append(f"residual rms {fit.rms_m * 1000:.1f} mm is above {ACCEPT_RMS_M * 1000:.0f} mm")
    _, _, space = read_device(args.url, args.device)
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "space": space,
        "device": args.device,
        "xr_from_base": fit.xr_from_base.tolist(),
        "tcp_from_controller": fit.tcp_from_controller.tolist(),
        "rms_m": fit.rms_m,
        "max_m": fit.max_m,
        "slip_m": slip_m,
        "slip_rad": slip_rad,
        "poses": len(base_from_tcp),
        "base_from_tcp": [pose.tolist() for pose in base_from_tcp],
        "xr_from_controller": [pose.tolist() for pose in xr_from_controller],
    }
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if problems and not args.accept_any:
        rejected = path.with_name("calibration_rejected.json")
        rejected.write_text(json.dumps(result, indent=2) + "\n")
        for problem in problems:
            logger.error("Not applied: {}", problem)
        logger.info("Raw data kept in {}", rejected)
        return
    path.write_text(json.dumps(result, indent=2) + "\n")
    _http(f"{args.url}/calibration", result)
    publisher = XrPublisher(token=read_token(token_path()))
    try:
        publisher.publish(check_layer(base_from_tcp))
        publisher.flush()
    finally:
        publisher.close()
    logger.info("Calibration applied and saved to {} ({} space)", path, space)


def main(args: Args) -> None:
    try:
        wait_for_device(args.url, args.device, args.wait_s)
    except RuntimeError as error:
        raise SystemExit(f"headset not ready: {error}") from error
    _, _, space = read_device(args.url, args.device)
    logger.info("Headset sees the {} controller in the {} space", args.device, space)
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with env:
            run(args, env)


def cli() -> None:
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)


if __name__ == "__main__":
    cli()
