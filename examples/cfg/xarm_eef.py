from dataclasses import dataclass, field

from rio.cfg import Camera, NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class Xarm7EefStation:
    arm: str = "XarmArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            robot_ip="192.168.1.205",
            robot_model="xarm7",
            max_pos_speed=1.0,
            max_rot_speed=0.6,
            robot_controller="task_pos",
            freq=250,
        )
    )

    gripper: str | None = "RobotiqGripper"
    gripper_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            port="/dev/ttyUSB0",
            model="robotiq_2f140",
            calibrate=False,
            freq=200,
            filter_alpha=0.35,
        )
    )

    cameras: dict[str, Camera] = field(
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

    @dataclass
    class TeleopCfg:
        addr: str = "127.0.0.1:5000"

    teleop: str = "Spacemouse"  # Gamepad | Keyboard | Spacemouse
    teleop_cfg: TeleopCfg = field(default_factory=lambda: Xarm7EefStation.TeleopCfg())

    arm_latency: float = 0.0
    gripper_latency: float = 0.1
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    action_space: str = "task_pos"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = ""
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))
