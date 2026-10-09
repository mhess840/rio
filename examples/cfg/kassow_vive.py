from dataclasses import dataclass, field

from rio.cfg import Camera, VisualizerCfg
from rio.cfg.common import RecorderCfg
from rio.cfg.node import NodeCfg
from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

TASK = "kassow_vive_teleop"


@dataclass
class KassowViveStation:
    """Kassow KR-series end-effector teleop from a Vive tracker (no hand).

    Absolute tracker poses are clutch-retargeted onto the TCP the same way as
    `Ur5eViveInspireStation`, but the follower is a Kassow arm with local IK
    (`task_pos_ik`) and there is no Manus/Inspire path.

    Run with:

        STATION=KassowViveStation uv run -m examples.teleop_vive_hand
    """

    @dataclass
    class ArmCfg:
        addr: str = "127.0.0.1:5555"
        robot_ip: str = "192.168.1.44"
        port: int = 7582
        session_id: int = 1
        robot_controller: str = "task_pos_ik"
        max_pos_speed: float = 0.45  # m/s — keep at or above the station retargeter limit
        max_rot_speed: float = 1.0  # rad/s
        max_motor_speed: float = 1.0  # rad/s
        urdf_path: str = DEFAULT_URDF_PATH
        ee_frame: str = "end_effector"
        ik_kp: float = 12.0
        max_joint_accel: float | None = 4.0
        stream_l_mode: str = "time"
        stream_l_tt: float = 0.016
        stream_l_bt: float = 0.008
        stream_l_speed: float = 0.0
        stream_l_throttle: int = 2
        lowpass_alpha: float | None = 0.65
        log_diagnostics: bool = True
        cmd_freq: int = 50
        freq: int = 250

    arm: str = "KassowArm"
    arm_cfg: ArmCfg = field(default_factory=ArmCfg)

    gripper: str | None = None
    gripper_cfg: None = None

    cameras: dict[str, Camera] = field(default_factory=dict)

    # Leader: Vive tracker worn on the wrist / held in hand
    teleop: str = "ViveTracker"
    teleop_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5580",
            serial=None,  # first tracker SteamVR reports, or "LHR-XXXXXXXX"
            fix_base_channels=True,
            freq=250,
            timeout=60.0,
        )
    )

    # Keyboard for clutch / orientation / recorder keys
    teleop_keyboard: str | None = "Keyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(addr="127.0.0.1:5582", freq=100)
    )

    # -----------------------------------------------------------------------
    # Vive retargeting and safety (consumed by teleop_vive_hand)
    # -----------------------------------------------------------------------
    clutch_key: str = "c"
    orientation_key: str = "o"
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    orientation_enabled: bool = True
    yaw_offset: float = 0.0
    yaw_calibration_file: str | None = "vive_yaw_cal.json"
    calibrate_yaw: bool = False
    min_sweep_travel: float = 0.30
    max_sweep_skew: float = 20.0
    # Per-step limits applied by the retargeter, ahead of the arm node's own.
    max_pos_speed: float = 0.45
    max_rot_speed: float = 1.0
    # Reachable shell around the Kassow base (KR1018-scale); tighten for your cell.
    min_radius: float = 0.25
    max_radius: float = 1.40
    min_z: float = 0.05
    tracking_grace: float = 0.25
    max_lag: float = 0.25

    arm_latency: float = 0.0
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 100

    action_space: str = "task_pos"
    embodiment_type: str = "SINGLE_ARM"
    urdf_path: str = DEFAULT_URDF_PATH

    instruction: str = ""
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

    def __post_init__(self) -> None:
        self.arm_cfg.cmd_freq = self.freq
        if not self.arm_cfg.urdf_path:
            self.arm_cfg.urdf_path = self.urdf_path
        if not self.urdf_path:
            self.urdf_path = self.arm_cfg.urdf_path
        space = self.action_space.lower()
        if space == "task_pos":
            if self.arm_cfg.robot_controller not in ("task_pos", "task_pos_ik"):
                self.arm_cfg.robot_controller = "task_pos_ik"
        elif space in ("joint_pos", "joint_vel"):
            self.arm_cfg.robot_controller = space
        if self.teleop_keyboard == "Keyboard":
            import os

            from loguru import logger

            if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
                logger.warning(
                    "Wayland session detected: switching teleop_keyboard Keyboard → SshKeyboard "
                    "(pynput cannot capture keys here). Use WASD/QE in this terminal."
                )
                self.teleop_keyboard = "SshKeyboard"
