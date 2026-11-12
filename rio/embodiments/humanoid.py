# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from ..schema import ActionSpace, Observation
from .base import BaseEmbodiment, JointClient


@dataclass
class HumanoidObs(Observation):
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    quaternion: np.ndarray | None = None
    gyroscope: np.ndarray | None = None


class Humanoid(BaseEmbodiment):
    def __init__(
        self,
        humanoid: JointClient,
        action_space: str = "JOINT_POS",
        **kwargs,
    ):
        self.humanoid = humanoid
        self.action_space = ActionSpace[action_space]
        self.urdf_path = kwargs.get("urdf_path", "")
        self.__obs_schema__ = HumanoidObs

        # Get number of joints dynamically
        self.num_joints = self._get_num_joints(self.humanoid)

    def move(
        self,
        action: np.ndarray,
        t_cmd_target: float,
        **kwargs,
    ):
        if self.action_space == ActionSpace.JOINT_POS:
            self.humanoid.moveJ(action.tolist(), t_cmd_target)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not implemented for Humanoid")

    def parse_action(self, action: np.ndarray) -> dict:
        return {"q": action}

    def build_action(self, arm_cmd: np.ndarray, gripper_cmd: np.ndarray | None = None) -> np.ndarray:
        if gripper_cmd is not None:
            raise ValueError("Humanoid actions do not have a gripper channel")
        return arm_cmd

    def get_state(self):
        return self.humanoid.get_state()

    def get_obs(self, cams: dict) -> HumanoidObs:
        """Build humanoid observation."""
        state = self.humanoid.get_state()

        q = state.get("joint_q", None)
        dq = state.get("joint_qd", None)
        quaternion = state.get("imu_quat", None)
        gyroscope = state.get("imu_gyro", None)

        obs = HumanoidObs(
            proprio=q,
            q=q,
            dq=dq,
            quaternion=quaternion,
            gyroscope=gyroscope,
        )
        return obs
