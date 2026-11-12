# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Guided yaw calibration between SteamVR's world frame and the robot base."""

import json
import math
import threading
from pathlib import Path

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    dominant_direction,
    horizontal_angle_between,
    yaw_from_motions,
)

# Directions the operator is asked to sweep, in the robot base frame.
CALIBRATION_SWEEPS = (
    ("+X", np.array([1.0, 0.0, 0.0]), "usually straight away from the robot base"),
    ("+Y", np.array([0.0, 1.0, 0.0]), "90 degrees counter-clockwise from +X, seen from above"),
)


def sweep_problem(points: np.ndarray, min_travel: float) -> str | None:
    """Why a recorded sweep is unusable, or None if it is fine."""
    if len(points) < 10:
        return "Too few valid tracker samples (is tracking dropping out?)."
    net = float(np.linalg.norm(points[-1] - points[0]))
    path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if net < min_travel:
        return f"Only travelled {net * 100:.0f} cm, need {min_travel * 100:.0f} cm."
    # A there-and-back sweep pins the axis but not which way along it points.
    if net < 0.5 * path:
        return "That looked like a round trip; sweep one way only."
    return None


def signal_on_enter(done: threading.Event) -> None:
    input()
    done.set()


def record_sweep(teleop, axis_name, hint, min_travel, sample_rate=50.0, max_attempts=5) -> np.ndarray:
    """Record tracker positions for one guided hand sweep, retrying if unusable."""
    for attempt in range(max_attempts):
        input(
            f"\n  Sweep {axis_name}: move your hand the way you want the TCP to travel "
            f"along robot {axis_name}\n"
            f"  ({hint}).\n"
            f"  Press Enter, move at least {min_travel * 100:.0f} cm in one direction, "
            "then press Enter again: "
        )

        done = threading.Event()
        threading.Thread(target=signal_on_enter, args=(done,), daemon=True).start()

        samples = []
        while not done.is_set():
            state = teleop.get_state()
            if float(state["pose_valid"]) > 0.5:
                samples.append(np.asarray(state["tracker_pose"][:3], dtype=float))
            time.sleep(1.0 / sample_rate)

        points = np.asarray(samples)
        problem = sweep_problem(points, min_travel)
        if problem is None:
            travelled = float(np.linalg.norm(points[-1] - points[0]))
            print(f"  Recorded {len(points)} samples over {travelled * 100:.0f} cm.")
            return points
        remaining = max_attempts - attempt - 1
        print(f"  {problem}" + (f" {remaining} attempts left." if remaining else ""))

    raise RuntimeError(
        f"Could not record a usable {axis_name} sweep. If the tracker never reports a "
        "valid pose, check base station visibility before calibrating."
    )


def calibrate_yaw(args, teleop) -> float:
    """Guided calibration of the yaw between SteamVR's world frame and the robot.

    SteamVR's frame is gravity-aligned, so roll and pitch already agree with the
    robot and a single angle about vertical is the only unknown. Two sweeps are
    recorded rather than one so their measured separation can be checked against
    the 90 degrees it ought to be.
    """
    print("\n=== Vive yaw calibration ===")
    print("Stand where you will actually work, facing the way you will work.")
    print("Wrist orientation does not matter, only the direction you travel.")

    measured, intended = [], []
    for axis_name, axis, hint in CALIBRATION_SWEEPS:
        points = record_sweep(teleop, axis_name, hint, args.min_sweep_travel)
        measured.append(AXIS_MAP_STEAMVR_TO_ROBOT @ dominant_direction(points))
        intended.append(axis)

    separation = math.degrees(horizontal_angle_between(measured[0], measured[1]))
    expected = math.degrees(horizontal_angle_between(intended[0], intended[1]))
    if abs(separation - expected) > args.max_sweep_skew:
        logger.warning(
            f"The two sweeps came out {separation:.0f} degrees apart, expected {expected:.0f}. "
            "Alignment will be rough; re-run with --calibrate-yaw and keep each sweep straight."
        )

    yaw_deg = math.degrees(yaw_from_motions(measured, intended))
    print(f"\n  Yaw offset: {yaw_deg:.1f} degrees   (sweeps {separation:.0f} degrees apart)")
    print("  If motion ends up rotated by roughly 90 degrees, the axis labels were")
    print("  swapped; re-run with --calibrate-yaw and pick the other direction.")
    return yaw_deg


def resolve_yaw(args, teleop) -> float:
    """Yaw offset for this session, calibrating or loading from disk as needed."""
    path = Path(args.yaw_calibration_file) if args.yaw_calibration_file else None

    if args.calibrate_yaw or (path is not None and not path.exists()):
        yaw_deg = calibrate_yaw(args, teleop)
        if path is not None:
            path.write_text(json.dumps({"yaw_offset_deg": yaw_deg}, indent=2))
            print(f"  Saved to {path}. Pass --calibrate-yaw to redo it.")
        return yaw_deg

    if path is not None:
        yaw_deg = float(json.loads(path.read_text())["yaw_offset_deg"])
        logger.info(f"Loaded yaw offset {yaw_deg:.1f} degrees from {path}")
        return yaw_deg

    return args.yaw_offset
