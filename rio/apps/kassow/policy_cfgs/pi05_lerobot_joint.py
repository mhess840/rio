"""Kassow π₀.₅ inference in joint space (moveJ). Needs a joint-trained checkpoint.

Convert a joint dataset of the active experiment, train it, register the model
(``uv run -m rio.apps.kassow.experiment_cli add-model``), then:

    uv run --group pi --no-group docs -m rio.apps.kassow.convert \\
        --variant joints_notrim --action-space joints --no-trim-idle --gripper-state-source commanded

    POLICY=Pi05LerobotJointCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve \\
        --policy-path experiments/<exp>/models/<joint_model> --policy-cfg.device cuda:1

A TCP checkpoint is refused. See docs/workflow/pi05_kassow_pipeline.md, section 8.
"""

from dataclasses import dataclass

from .pi05_lerobot import Pi05LerobotCfg


@dataclass
class Pi05LerobotJointCfg(Pi05LerobotCfg):
    """Same station as ``Pi05LerobotCfg``, but actions are 8D joints + gripper."""

    # Empty: the selected model of the experiment (must be a joint checkpoint).
    policy_path: str = ""
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
