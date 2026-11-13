# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Record table samples for the image-to-robot map. The arm is not moved.

One sample, two presses of ``k``:

1. Drive with the Vive clutch, set the object (e.g. the cube) down on the table
   and keep it in the closed gripper. Release the clutch and press ``k``: the TCP is stored.
2. Open the gripper, move the arm out of the camera view, press ``k`` again:
   one photo from the overview camera and one from the Jev camera are stored.

Grasp every sample at the same height and orientation, and spread the samples
over the whole area the arm should pick from. Ctrl-C ends the session. Then
``table_map label`` and ``table_map fit``.

    uv run python -m rio.apps.kassow.graph.calibrate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from rio.apps.kassow.experiments import jev_camera
from rio.apps.kassow.jev_ask import fresh_frame
from rio.apps.kassow.stations.kassow_vive_gripper import KassowViveGripperStation
from rio.apps.kassow.teleop import cli
from rio.jev import encode_jpeg

from .table_map import POSE_NAME, calibration_dir, sample_folders

# Photo names must match table_map.CAMERAS.
CAMERA_KEYS = (("camera_2", "overview"), ("camera_jev", "jev"))


def pending_sample(root: Path) -> Path | None:
    """The newest sample that has a TCP but no photos yet."""
    folders = sample_folders(root)
    if folders and not all((folders[-1] / f"{name}.jpg").is_file() for _key, name in CAMERA_KEYS):
        return folders[-1]
    return None


def save_sample(env, eef_pose: np.ndarray, joint_q: np.ndarray, root: Path | None = None) -> Path:
    """First press stores the TCP in a new folder, the next press the photos of that folder.

    Returns:
        The sample folder.

    Raises:
        RuntimeError: If either camera has no fresh frame.
        KeyError: If the environment is missing one of the cameras.
    """
    folder = calibration_dir() if root is None else root
    folder.mkdir(parents=True, exist_ok=True)
    pending = pending_sample(folder)
    if pending is None:
        folders = sample_folders(folder)
        sample = folder / f"{(int(folders[-1].name) if folders else 0) + 1:03d}"
        sample.mkdir()
        pose = np.asarray(eef_pose, dtype=float).reshape(-1)
        record = {
            "t": datetime.now().isoformat(timespec="seconds"),
            "xyz": [round(float(value), 6) for value in pose[:3]],
            "eef_pose": [round(float(value), 6) for value in pose[:6]],
            "joint_q": [round(float(value), 6) for value in np.asarray(joint_q, dtype=float).reshape(-1)],
        }
        (sample / POSE_NAME).write_text(json.dumps(record) + "\n")
        logger.info("Sample {} TCP xyz={}. Open, move the arm out of view, press k for the photos", sample.name, record["xyz"])
        return sample
    jpegs = {name: encode_jpeg(fresh_frame(env.cameras[key], max_age_s=0.5)) for key, name in CAMERA_KEYS}
    for name, jpeg in jpegs.items():
        (pending / f"{name}.jpg").write_bytes(jpeg)
    logger.info("Sample {} photos stored", pending.name)
    return pending


@dataclass
class CalibrateStation(KassowViveGripperStation):
    """Vive drive with the overview camera and the Jev camera. ``k`` stores the TCP, then the photos."""

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
