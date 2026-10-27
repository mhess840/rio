"""Kassow π₀.₅ inference in joint space (moveJ). Needs a joint-trained checkpoint.

    POLICY=Pi05LerobotJointCfg uv run --group pi --no-group docs -m examples.policy_inference \\
        --policy-path ckpts/pi05_joints \\
        --instruction "pick up the blue cylinder and place it on the red plate" \\
        --policy-cfg.device cuda:1

Do not point this config at ``ckpts/pi05`` (TCP). Convert and train first:

    uv run -m examples.data.convert_to_lerobot_pi05 \\
        --input data/kassow_vive_gripper \\
        --output data/lerobot/kassow_pi05_joints \\
        --repo-id local/kassow_pi05_joints \\
        --tasks data/kassow_vive_gripper/tasks.yaml \\
        --action-space joints
"""

from dataclasses import dataclass

from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg


@dataclass
class Pi05LerobotJointCfg(Pi05LerobotCfg):
    """Same station as ``Pi05LerobotCfg``, but actions are 8D joints + gripper."""

    policy_path: str = "ckpts/pi05_joints"
    action_space: str = "joint_pos"

    def __post_init__(self):
        self.policy_cfg.action_space = "joints"
        self.policy_node_cfg.action_dim = 8
        super().__post_init__()
        self.action_space = "joint_pos"
        self.arm_cfg.robot_controller = "joint_pos"
        self.arm_cfg.stream_j_speed = 0.15
        self.policy_cfg.action_space = "joints"
        self.policy_node_cfg.action_dim = 8
