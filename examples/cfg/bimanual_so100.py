from dataclasses import dataclass, field
from pathlib import Path

from rio.cfg import NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

CALIBRATION_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so100_follower"


@dataclass
class BimanualSo100Station:
    arm1: str = "SoArm"
    arm1_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            port="/dev/ttyACM0",
            calibration_file=str(CALIBRATION_DIR / "follower_L.json"),
            robot_controller="joint_pos",
        )
    )

    arm2: str | None = "SoArm"
    arm2_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5556",
            port="/dev/ttyACM2",
            calibration_file=str(CALIBRATION_DIR / "follower_R.json"),
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
            calibration_file=str(CALIBRATION_DIR / "leader_L.json"),
        )
    )

    teleop2: str | None = "SoArm"
    teleop2_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            port="/dev/ttyACM3",
            motors_enabled=False,
            robot_model="so100",
            robot_controller="joint_pos",
            calibration_file=str(CALIBRATION_DIR / "leader_R.json"),
        )
    )

    teleop_keyboard: str | None = "Keyboard"

    teleop_module: str = "robots"
    teleop2_module: str = "robots"
    startup_delay: float = 0.5

    arm_latency: float = 0.01
    gripper_latency: float = 0.01
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    action_space: str = "JOINT_POS"
    embodiment_type: str = "BIMANUAL"

    instruction: str = "Fold the cloth in half."
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = None
    recorder_cfg: RecorderCfg = field(
        default_factory=lambda: RecorderCfg(
            path="data/xlerobot/",
            video_codec="libx264",
            codec_options={"crf": "23", "preset": "fast"},
            verbose=True,
        )
    )
