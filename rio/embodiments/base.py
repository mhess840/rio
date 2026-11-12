# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum, auto
from numbers import Integral
from typing import Generic, Protocol, TypeVar

import numpy as np
from rio_hw import time

from ..schema import ActionSpace, Observation

# Type variables for generic typing
ObsType = TypeVar("ObsType", bound="Observation")


class StateClient(Protocol):
    def get_state(self) -> dict: ...


class JointClient(StateClient, Protocol):
    def moveJ(self, target: list[float], t_cmd_target: float) -> None: ...


class ArmClient(JointClient, Protocol):
    def moveL(self, target: list[float], t_cmd_target: float) -> None: ...
    def speedJ(self, target: list[float], t_cmd_target: float) -> None: ...


class GripperClient(StateClient, Protocol):
    def moveG(self, target: list[float], t_cmd_target: float) -> None: ...


class EmbodimentType(Enum):
    SINGLE_ARM = auto()
    BIMANUAL = auto()
    HUMANOID = auto()


class BaseEmbodiment(ABC, Generic[ObsType]):
    # Set by each concrete embodiment; read by the shared dispatch helpers.
    action_space: ActionSpace

    @abstractmethod
    def get_state(self) -> dict:
        raise NotImplementedError("Should be implemented for each embodiment.")

    @abstractmethod
    def get_obs(self, cams: dict) -> ObsType:
        raise NotImplementedError("Should be implemented for each embodiment.")

    @abstractmethod
    def move(self, action: np.ndarray, t_cmd_target: float, /) -> None:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @abstractmethod
    def build_action(self, arm_cmd: np.ndarray, gripper_cmd: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @abstractmethod
    def parse_action(self, action: np.ndarray) -> dict:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @staticmethod
    def _get_num_joints(client: StateClient) -> int:
        count = getattr(client, "num_joints", None)
        if isinstance(count, Integral) and not isinstance(count, bool) and count > 0:
            return int(count)
        state = client.get_state()
        joints = state.get("joint_q", state.get("q"))
        if joints is not None:
            shape = np.asarray(joints).shape
            if len(shape) == 1 and shape[0] > 0:
                return shape[0]
        raise ValueError("Cannot determine a positive joint count from the client or measured state")

    @staticmethod
    def joint_gripper_proprio(joint_q, gripper_pos, num_joints: int) -> np.ndarray | None:
        """Measured joints + gripper of one arm, the recorded ``observation/proprio`` layout in every action space.

        Args:
            joint_q: Measured joint positions.
            gripper_pos: Measured gripper opening.
            num_joints: Expected joint count of the arm.

        Returns:
            ``(num_joints + 1,)`` array, or ``None`` while the state is incomplete or non-finite.
        """
        joints = None if joint_q is None else np.asarray(joint_q, dtype=np.float64).reshape(-1)
        grip = None if gripper_pos is None else np.asarray(gripper_pos, dtype=np.float64).reshape(-1)
        if (
            joints is not None
            and grip is not None
            and joints.shape == (num_joints,)
            and grip.shape == (1,)
            and np.isfinite(joints).all()
            and np.isfinite(grip).all()
        ):
            return np.concatenate([joints, grip])
        return None

    @staticmethod
    def _resolve_gripper(arm, gripper=None):
        """Resolve the node that owns the gripper channel.

        An explicit gripper client always wins; otherwise the arm itself drives
        its gripper when it exposes ``moveG`` (integrated gripper).

        Args:
            arm: Arm client.
            gripper: Optional dedicated gripper client.

        Returns:
            The gripper controller, or ``None`` if no gripper is available.
        """
        if gripper is not None:
            return gripper
        if hasattr(arm, "moveG"):
            return arm
        return None

    def _dispatch_arm(self, arm, arm_cmd, t_cmd_target):
        """Send an arm command via the method matching ``self.action_space``."""
        arm_cmd = np.asarray(arm_cmd).tolist()
        if self.action_space == ActionSpace.TASK_POS:
            arm.moveL(arm_cmd, t_cmd_target)
        elif self.action_space == ActionSpace.JOINT_POS:
            arm.moveJ(arm_cmd, t_cmd_target)
        elif self.action_space == ActionSpace.JOINT_VEL:
            arm.speedJ(arm_cmd, t_cmd_target)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not implemented")

    @staticmethod
    def _dispatch_gripper(gripper, gripper_cmd, t_cmd_target, binarize=False):
        """Send a gripper command via ``moveG``; no-op if either is missing."""
        if gripper is None or gripper_cmd is None:
            return
        if binarize:
            gripper_cmd = gripper_cmd if gripper_cmd > 0.5 else gripper_cmd - 0.1
        now = time.now()
        if t_cmd_target <= now:
            # Background encode or a hitch can make the 100 Hz stamp late;
            # AG95 moveG asserts on past times and would kill teleop.
            t_cmd_target = now + 0.01
        gripper.moveG([float(gripper_cmd)], t_cmd_target)
