# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, TypedDict

import numpy as np


class ActionSpace(Enum):
    JOINT_POS = auto()
    TASK_POS = auto()
    JOINT_VEL = auto()
    TASK_VEL = auto()
    JOINT_TORQUE = auto()


# Written to ``<traj>_meta.json`` by the recorder; bump when a recorded key changes meaning.
# 1: SingleArm ``observation/proprio`` = measured joints + gripper; timestamps stored in ns.
# 2: Bimanual ``observation/proprio`` = per arm measured joints + gripper (was TCP pose in TASK_POS).
RECORDING_SCHEMA_VERSION = 2


class ArmState(TypedDict, total=False):
    """Keys of a ``rio_hw`` arm state that RIO reads; drivers may publish more."""

    timestamp: float
    alarm_code: int
    joint_q: np.ndarray
    joint_qd: np.ndarray
    eef_pose: np.ndarray
    gripper_position: float
    gripper_commanded_position: float


# Keys of ``ArmState``; a typo in a required-field list is a type error, not a silent runtime miss.
ArmField = Literal[
    "timestamp",
    "alarm_code",
    "joint_q",
    "joint_qd",
    "eef_pose",
    "gripper_position",
    "gripper_commanded_position",
]


def require_state_fields(state: Mapping, fields: Iterable[ArmField], *, source: str = "arm") -> None:
    """Fail at startup when a driver does not publish fields a caller depends on.

    Args:
        state: One published state dict.
        fields: Keys that must be present and not ``None``.
        source: Name used in the error message.

    Raises:
        RuntimeError: If any field is missing.
    """
    if not isinstance(state, Mapping):
        raise RuntimeError(f"The {source} state is unavailable")
    missing = sorted(name for name in fields if state.get(name) is None)
    if missing:
        raise RuntimeError(f"The {source} driver does not publish {', '.join(missing)}; this rollout cannot run safely")


@dataclass
class Camera:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Observation:
    proprio: np.ndarray | None  # SingleArm: measured joints + gripper, in every action space
    cameras: dict[str, Camera] = field(default_factory=dict)


@dataclass
class Step:
    timestep: float | None
    observation: Observation
    instruction: str | None
    action: np.ndarray | None
    meta: dict | None = field(default_factory=dict)
