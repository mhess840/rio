# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Record table points for the image-to-robot map. The arm is not moved.

Drive with the Vive clutch, hold the gripper tip at grasp height, release the
clutch, then press ``k``. Each press stores the measured TCP and one photo from
the overview camera and the Jev camera. Ctrl-C ends the session.

    uv run python -m rio.apps.kassow.graph.calibrate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from rio.apps.kassow.experiments import experiments_root, jev_camera
from rio.apps.kassow.jev_ask import fresh_frame
from rio.apps.kassow.stations.kassow_vive_gripper import KassowViveGripperStation
from rio.apps.kassow.teleop import cli
from rio.jev import encode_jpeg

CAMERA_KEYS = (("camera_2", "overview"), ("camera_jev", "jev"))


def calibration_dir() -> Path:
    """Folder of the table samples, outside the orchestrator logs."""
    return experiments_root() / "graph" / "calibration" / "table"


def save_sample(env, eef_pose: np.ndarray, joint_q: np.ndarray, root: Path | None = None) -> Path:
    """Write one folder of two JPEGs and one pose line. Returns the folder.

    The files are ``overview.jpg`` and ``jev.jpg``.

    Raises:
        RuntimeError: If either overview camera has no fresh frame.
        KeyError: If the environment is missing one of the cameras.
    """
    folder = calibration_dir() if root is None else root
    folder.mkdir(parents=True, exist_ok=True)
    poses = folder / "poses.jsonl"
    index = sum(1 for _ in poses.open()) + 1 if poses.is_file() else 1
    sample = folder / f"{index:03d}"
    sample.mkdir()
    images: dict[str, str] = {}
    for key, name in CAMERA_KEYS:
        jpeg = encode_jpeg(fresh_frame(env.cameras[key], max_age_s=0.5))
        path = sample / f"{name}.jpg"
        path.write_bytes(jpeg)
        images[name] = f"{sample.name}/{path.name}"
    pose = np.asarray(eef_pose, dtype=float).reshape(-1)
    record = {
        "t": datetime.now().isoformat(timespec="seconds"),
        "images": images,
        "xyz": [round(float(value), 6) for value in pose[:3]],
        "eef_pose": [round(float(value), 6) for value in pose[:6]],
        "joint_q": [round(float(value), 6) for value in np.asarray(joint_q, dtype=float).reshape(-1)],
    }
    with poses.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    logger.info("Saved {}  xyz={}", sample.name, record["xyz"])
    return sample


@dataclass
class CalibrateStation(KassowViveGripperStation):
    """Vive drive with the overview camera and the Jev camera. ``k`` stores both."""

    calibration_key: str = "k"

    def __post_init__(self) -> None:
        super().__post_init__()
        cameras = dict(self.cameras)
        cameras.pop("camera_1", None)
        cameras["camera_jev"] = jev_camera()
        self.cameras = cameras


def main() -> None:
    """Drive with the clutch and store a sample on ``k``."""
    cli(CalibrateStation)


if __name__ == "__main__":
    main()
