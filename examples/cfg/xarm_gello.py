from dataclasses import dataclass, field

import yaml

from rio.cfg import Camera, NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class GelloCfg:
    port: str = "/dev/ttyUSB1"
    baudrate: int = 57600
    joint_ids: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    joint_offsets: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_signs: tuple = (1, 1, 1, 1, 1, 1, 1)
    gripper_config: tuple = (0, 0, 0)
    start_joints: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    freq: int = 250


@dataclass
class Xarm7GelloStation:
    arm: str = "XarmArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            robot_ip="192.168.1.205",
            robot_model="xarm7",
            joints_lowpass_alpha=0.75,
            robot_controller="joint_pos",
            freq=250,
        )
    )

    gripper: str | None = "RobotiqGripper"
    gripper_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            port="/dev/ttyUSB1",
            model="robotiq_2f140",
            calibrate=False,
            freq=200,
            filter_alpha=None,
        )
    )

    cameras: dict[str, Camera] | None = field(
        default_factory=lambda: {
            "camera_1": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="821212062747",
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            "camera_2": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="218622273888",
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            "camera_3": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="821212061298",
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
        }
    )

    teleop: str = "Gello"
    teleop_cfg: GelloCfg = field(default_factory=GelloCfg)
    teleop_cfg_yaml: str | None = None

    teleop2: str | None = None
    teleop2_cfg: GelloCfg = field(default_factory=GelloCfg)
    teleop2_cfg_yaml: str | None = None

    # Leader-follower teleoperation flags
    check_alignment: bool = True
    invert_gripper: bool = True
    use_leader_smoothing: bool = True
    startup_delay: float = 3.5

    arm_latency: float = 0.0
    gripper_latency: float = 0.01
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 15

    action_space: str = "joint_pos"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = "Place the can in the bowl."
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

    def __post_init__(self):
        def load_gello_yaml(yaml_path):
            with open(yaml_path) as f:
                teleop_cfg = yaml.safe_load(f)
            teleop_kwargs = {
                "port": teleop_cfg["agent"]["port"],
                "baudrate": teleop_cfg["agent"].get("baudrate", 57600),
                "joint_ids": tuple(teleop_cfg["agent"]["dynamixel_config"]["joint_ids"]),
                "joint_offsets": tuple(teleop_cfg["agent"]["dynamixel_config"]["joint_offsets"]),
                "joint_signs": tuple(teleop_cfg["agent"]["dynamixel_config"]["joint_signs"]),
                "gripper_config": tuple(teleop_cfg["agent"]["dynamixel_config"]["gripper_config"]),
                "start_joints": tuple(teleop_cfg["agent"]["start_joints"]),
            }
            return GelloCfg(**teleop_kwargs)

        if self.teleop_cfg_yaml:
            self.teleop_cfg = load_gello_yaml(self.teleop_cfg_yaml)

        if self.teleop2_cfg_yaml:
            self.teleop2_cfg = load_gello_yaml(self.teleop2_cfg_yaml)
