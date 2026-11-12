# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Recording keys read from Kassow ``.vla`` files and the π₀.₅ dataset layout written from them."""

CAM1_KEY = "observation/cameras/camera_1/rgb"
CAM2_KEY = "observation/cameras/camera_2/rgb"
JOINTS_KEY = "observation/proprio_joints"
GRIPPER_KEY = "observation/gripper_position"
GRIPPER_COMMANDED_KEY = "observation/gripper_commanded_position"
ACTION_KEY = "action"
INSTRUCTION_KEY = "instruction"

REQUIRED_KEYS = (CAM1_KEY, CAM2_KEY, JOINTS_KEY, GRIPPER_KEY, ACTION_KEY)

STATE_DIM = 8
ACTION_DIM = 7
JOINT_ACTION_DIM = 8
STATE_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
JOINT_ACTION_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
TCP_DATASET_DIR_NAMES = frozenset({"kassow_pi05", "kassow_pi05_alignment"})


def validate_keys(data: dict) -> None:
    """Raise if a trajectory lacks a key the conversion needs."""
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Trajectory missing required keys {missing}; have {sorted(data)}")
