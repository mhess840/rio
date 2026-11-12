# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..schema import ActionSpace, Observation
from .base import ArmClient, BaseEmbodiment, GripperClient, JointClient


@dataclass
class BimanualObs(Observation):
    # Left arm (arm1)
    arm1_proprio_eef: np.ndarray | None = None
    arm1_proprio_joints: np.ndarray | None = None
    gripper1_position: float | None = None
    hand1_pose: np.ndarray | None = None
    hand1_joints: np.ndarray | None = None

    # Right arm (arm2)
    arm2_proprio_eef: np.ndarray | None = None
    arm2_proprio_joints: np.ndarray | None = None
    gripper2_position: float | None = None
    hand2_pose: np.ndarray | None = None
    hand2_joints: np.ndarray | None = None


class Bimanual(BaseEmbodiment):
    def __init__(
        self,
        arm1: ArmClient,
        arm2: ArmClient,
        gripper1: GripperClient | None = None,
        gripper2: GripperClient | None = None,
        hand1: JointClient | None = None,
        hand2: JointClient | None = None,
        action_space: str = "TASK_POS",
        **kwargs,
    ):
        self.arm1 = arm1
        self.arm2 = arm2
        self.gripper1 = gripper1
        self.gripper2 = gripper2
        self.hand1 = hand1
        self.hand2 = hand2
        self.action_space = ActionSpace[action_space]

        # Resolve the node that owns each gripper channel (explicit client, else the
        # arm itself when it exposes moveG). Commanded via moveG in all cases.
        self._gripper1 = self._resolve_gripper(self.arm1, self.gripper1)
        self._gripper2 = self._resolve_gripper(self.arm2, self.gripper2)

        # Get number of joints for each arm dynamically
        self.arm1_num_joints = self._get_num_joints(self.arm1)
        self.arm2_num_joints = self._get_num_joints(self.arm2)

        if self.action_space == ActionSpace.TASK_POS:
            self.arm1_dim = self.arm2_dim = 6
        else:  # JOINT_POS or JOINT_VEL
            self.arm1_dim = self.arm1_num_joints
            self.arm2_dim = self.arm2_num_joints

        logger.info(f"Arm1 has {self.arm1_num_joints} joints, Arm2 has {self.arm2_num_joints} joints")

        self.urdf_path = kwargs.get("urdf_path", "")
        self.__obs_schema__ = BimanualObs

    def move(
        self,
        action: np.ndarray,
        t_cmd_target: float,
        binarize_gripper: bool = False,
    ):
        """
        Execute bimanual action (concatenation of arm1 and arm2 commands).

        Args:
            action: Concatenated action [arm1_cmd, gripper1, arm2_cmd, gripper2]
            t_cmd_target: Target time for command
            binarize_gripper: Whether to binarize gripper commands
        """
        # Parse action into components
        parsed = self.parse_action(action)

        arm1_cmd = parsed["arm1_cmd"]
        gripper1_cmd = parsed.get("gripper1_cmd")
        hand1_cmd = parsed.get("hand1_cmd")
        arm2_cmd = parsed["arm2_cmd"]
        gripper2_cmd = parsed.get("gripper2_cmd")
        hand2_cmd = parsed.get("hand2_cmd")

        # Execute arm1
        self._move_single_arm(
            self.arm1,
            self._gripper1,
            self.hand1,
            arm1_cmd,
            gripper1_cmd,
            hand1_cmd,
            t_cmd_target,
            binarize_gripper,
        )

        # Execute arm2
        self._move_single_arm(
            self.arm2,
            self._gripper2,
            self.hand2,
            arm2_cmd,
            gripper2_cmd,
            hand2_cmd,
            t_cmd_target,
            binarize_gripper,
        )

    def _move_single_arm(self, arm, gripper, hand, arm_cmd, gripper_cmd, hand_cmd, t_cmd_target, binarize_gripper):
        """Helper to move a single arm with gripper and hand"""
        self._dispatch_gripper(gripper, gripper_cmd, t_cmd_target, binarize=binarize_gripper)
        self._dispatch_arm(arm, arm_cmd, t_cmd_target)

        if hand is not None and hand_cmd is not None:
            hand.moveJ(hand_cmd.tolist(), t_cmd_target)

    def parse_action(self, action: np.ndarray) -> dict:
        # Parse arm command
        arm1_cmd = action[: self.arm1_dim]
        gripper1_cmd = action[self.arm1_dim]
        arm2_cmd = action[self.arm1_dim + 1 : self.arm1_dim + 1 + self.arm2_dim]
        gripper2_cmd = action[self.arm1_dim + 1 + self.arm2_dim]

        return {
            "arm1_cmd": arm1_cmd,
            "gripper1_cmd": gripper1_cmd,
            "arm2_cmd": arm2_cmd,
            "gripper2_cmd": gripper2_cmd,
            "hand1_cmd": None,
            "hand2_cmd": None,
        }

    def build_action(self, arm_cmd: np.ndarray, gripper_cmd: np.ndarray | None = None) -> np.ndarray:
        """Build bimanual action by concatenating arm and gripper commands."""

        arm1_cmd = arm_cmd[: self.arm1_dim]
        arm2_cmd = arm_cmd[self.arm1_dim : self.arm1_dim + self.arm2_dim]

        if gripper_cmd is not None:
            arm1_cmd = np.concatenate([arm1_cmd, [gripper_cmd[0]]])
            arm2_cmd = np.concatenate([arm2_cmd, [gripper_cmd[1]]])
        else:
            arm1_cmd = np.concatenate([arm1_cmd, [0.0]])
            arm2_cmd = np.concatenate([arm2_cmd, [0.0]])

        action = np.concatenate([arm1_cmd, arm2_cmd])
        return action

    def get_state(self):
        """Get state of both arms"""
        state = {"arm1": {}, "arm2": {}}

        if self.arm1:
            state["arm1"]["arm"] = self.arm1.get_state()
            if self.gripper1:
                state["arm1"]["gripper"] = self.gripper1.get_state()
            elif self._gripper1 is self.arm1:
                state["arm1"]["gripper"] = {"gripper_position": state["arm1"]["arm"].get("gripper_position", 0.0)}

            if self.hand1:
                state["arm1"]["hand"] = self.hand1.get_state()

        if self.arm2:
            state["arm2"]["arm"] = self.arm2.get_state()
            if self.gripper2:
                state["arm2"]["gripper"] = self.gripper2.get_state()
            elif self._gripper2 is self.arm2:
                state["arm2"]["gripper"] = {"gripper_position": state["arm2"]["arm"].get("gripper_position", 0.0)}

            if self.hand2:
                state["arm2"]["hand"] = self.hand2.get_state()

        return state

    def get_obs(self, cams: dict) -> BimanualObs:
        """Build bimanual observation with separate proprio for each arm"""
        robot_state = self.get_state()

        # Arm1 proprioception
        arm1_tcp_pose = robot_state["arm1"]["arm"].get("eef_pose", None)
        arm1_joint_q = robot_state["arm1"]["arm"].get("joint_q", None)
        gripper1_pos = robot_state["arm1"].get("gripper", {}).get("gripper_position", 0.0)

        if arm1_joint_q is None:
            logger.warning("Arm1 state does not contain joint_q.")

        # Arm2 proprioception
        arm2_tcp_pose = robot_state["arm2"]["arm"].get("eef_pose", None)
        arm2_joint_q = robot_state["arm2"]["arm"].get("joint_q", None)
        gripper2_pos = robot_state["arm2"].get("gripper", {}).get("gripper_position", 0.0)

        if arm2_joint_q is None:
            logger.warning("Arm2 state does not contain joint_q.")

        # Same layout as SingleArm in every action space: per arm measured joints + gripper.
        arm1_proprio = self.joint_gripper_proprio(arm1_joint_q, gripper1_pos, self.arm1_num_joints)
        if arm1_proprio is None:
            logger.warning(f"invalid arm1 proprioception detected. joint_q: {arm1_joint_q}, gripper1_pos: {gripper1_pos}")
        arm2_proprio = self.joint_gripper_proprio(arm2_joint_q, gripper2_pos, self.arm2_num_joints)
        if arm2_proprio is None:
            logger.warning(f"invalid arm2 proprioception detected. joint_q: {arm2_joint_q}, gripper2_pos: {gripper2_pos}")

        # Concatenate for full proprio
        if arm1_proprio is not None and arm2_proprio is not None:
            proprio = np.concatenate([arm1_proprio, arm2_proprio])
        else:
            proprio = None

        obs = BimanualObs(
            proprio=proprio,
            arm1_proprio_eef=arm1_tcp_pose,
            arm1_proprio_joints=arm1_joint_q,
            gripper1_position=gripper1_pos,
            hand1_pose=robot_state["arm1"].get("hand", {}).get("hand_pose"),
            hand1_joints=robot_state["arm1"].get("hand", {}).get("hand_joints"),
            arm2_proprio_eef=arm2_tcp_pose,
            arm2_proprio_joints=arm2_joint_q,
            gripper2_position=gripper2_pos,
            hand2_pose=robot_state["arm2"].get("hand", {}).get("hand_pose"),
            hand2_joints=robot_state["arm2"].get("hand", {}).get("hand_joints"),
            cameras=cams,
        )

        return obs
