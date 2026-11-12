from dataclasses import dataclass, field
from pathlib import Path

from rio.cfg import NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

CALIBRATION_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so100_follower"


@dataclass
class So100Station:
    arm: str = "SoArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            port="/dev/ttyACM0",
            calibration_file=str(CALIBRATION_DIR / "follower_L.json"),
            robot_controller="joint_pos",
        )
    )

    gripper: str | None = None

    teleop: str = "SoArm"
    teleop_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            port="/dev/ttyACM1",
            motors_enabled=False,
            robot_model="so100",
            robot_controller="joint_pos",
            calibration_file=str(CALIBRATION_DIR / "leader_R.json"),
        )
    )

    teleop_module: str = "robots"

    arm_latency: float = 0.01
    gripper_latency: float = 0.01
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    action_space: str = "JOINT_POS"

    instruction: str = ""
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path="data/"))
