# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert Kassow robodm ``.vla`` trajectories to a LeRobot v3 π₀.₅ dataset.

``schema`` holds the recording keys and dataset layout, ``config`` the settings,
``tasks`` the files and language labels, ``episode`` the per-episode transforms,
``lerobot_io`` the LeRobot version shims and merging, and ``pipeline`` the run.
"""

from .config import ConvertPi05Cfg
from .episode import (
    binarize_action_gripper,
    binarize_state_gripper,
    build_pi05_frame,
    episode_ranges,
    idle_start_index,
    joints_as_action,
    prepare_episode,
    resolve_gripper_threshold,
    slice_episode,
    subsample_dict,
    subsample_step,
    unwrap_rotvec_sequence,
)
from .lerobot_io import create_lerobot_dataset, merge_lerobot_datasets, patch_lerobot_pyav, pi05_features
from .pipeline import convert
from .schema import (
    ACTION_DIM,
    ACTION_KEY,
    ACTION_NAMES,
    CAM1_KEY,
    CAM2_KEY,
    GRIPPER_COMMANDED_KEY,
    GRIPPER_KEY,
    INSTRUCTION_KEY,
    JOINT_ACTION_DIM,
    JOINT_ACTION_NAMES,
    JOINTS_KEY,
    REQUIRED_KEYS,
    STATE_DIM,
    STATE_NAMES,
    TCP_DATASET_DIR_NAMES,
    validate_keys,
)
from .tasks import list_vla_files, parse_task_map, resolve_task, traj_index, write_task_template

__all__ = [
    "ACTION_DIM",
    "ACTION_KEY",
    "ACTION_NAMES",
    "CAM1_KEY",
    "CAM2_KEY",
    "GRIPPER_COMMANDED_KEY",
    "GRIPPER_KEY",
    "INSTRUCTION_KEY",
    "JOINTS_KEY",
    "JOINT_ACTION_DIM",
    "JOINT_ACTION_NAMES",
    "REQUIRED_KEYS",
    "STATE_DIM",
    "STATE_NAMES",
    "TCP_DATASET_DIR_NAMES",
    "ConvertPi05Cfg",
    "binarize_action_gripper",
    "binarize_state_gripper",
    "build_pi05_frame",
    "convert",
    "create_lerobot_dataset",
    "episode_ranges",
    "idle_start_index",
    "joints_as_action",
    "list_vla_files",
    "merge_lerobot_datasets",
    "parse_task_map",
    "patch_lerobot_pyav",
    "pi05_features",
    "prepare_episode",
    "resolve_gripper_threshold",
    "resolve_task",
    "slice_episode",
    "subsample_dict",
    "subsample_step",
    "traj_index",
    "unwrap_rotvec_sequence",
    "validate_keys",
    "write_task_template",
]
