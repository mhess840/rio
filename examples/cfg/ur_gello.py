from dataclasses import dataclass, field

import yaml

from rio.cfg import Camera, NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

from .xarm_gello import GelloCfg

TASK = "stacking_plates"


@dataclass
class Ur5eGelloStation:
    arm: str = "UrArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            robot_ip="192.168.1.15",
            robot_model="ur5e",
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            joints_lowpass_alpha=0.5,
            robot_controller="joint_pos",
            freq=250,
            timeout=30.0,
        )
    )

    @dataclass
    class GripperCfg:
        addr: str = "127.0.0.1:5110"
        robot_port: str = "192.168.1.15"
        connection_type: str = "TCPIP"
        modbus_timeout: float = 10.0
        freq: int = 100
        timeout: float = 30.0

    gripper: str | None = "RobotiqGripper"
    gripper_cfg: GripperCfg = field(
        default_factory=lambda: Ur5eGelloStation.GripperCfg(
            addr="127.0.0.1:5110",
            robot_port="192.168.1.15",
            connection_type="TCPIP",
            modbus_timeout=10.0,
            timeout=30.0,
        )
    )

    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            "camera_0": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="115222071166",  # rs-enumerate-devices -S
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                timeout_ms=3000,
                timeout=30.0,
            ),
            "camera_1": Camera(
                addr="127.0.0.1:5131",
                cam_type="Zed",
                # Use get_connected_cameras() from rio_hw.cameras.zed to choose a device.
                serial="35616712",
                model="",
                freq=60,
                # ThreadClient waits this long for pub + req readiness (each phase uses the same value).
                timeout=30.0,
            ),
        }
    )

    teleop: str = "Gello"
    teleop_cfg: GelloCfg = field(default_factory=GelloCfg)
    teleop_cfg_yaml: str | None = None

    teleop_keyboard: str | None = "Keyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(addr="127.0.0.1:5570", freq=100)  # pick a free port
    )

    check_alignment: bool = True
    invert_gripper: bool = True
    use_leader_smoothing: bool = True
    startup_delay: float = 3.5

    arm_latency: float = 0.0
    gripper_latency: float = 0.01
    freq: int = 50

    mw: str = "Thread"
    mp_method: str = "spawn"
    action_space: str = "task_pos"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = "Stack the plates on top of each other."
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

    def __post_init__(self) -> None:
        def load_gello_yaml(yaml_path: str) -> GelloCfg:
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
