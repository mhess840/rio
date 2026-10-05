from dataclasses import dataclass, field

from rio.cfg import Camera, VisualizerCfg
from rio.cfg.common import RecorderCfg
from rio.cfg.node import NodeCfg
from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

TASK = "kassow_vive_inspire_teleop"


@dataclass
class KassowViveInspireStation:
    """Kassow arm EE teleop from a Vive tracker, with a Manus glove driving
    an Inspire hand on the wrist.

    Same clutch-retargeted Vive path as `KassowViveStation` / `Ur5eViveInspireStation`,
    plus the Manus → Inspire hand loop from `teleop_vive_hand`. Action vector is
    13-wide: 6 arm + 1 gripper slot + 6 hand DOF.

    Run with:

        STATION=KassowViveInspireStation uv run -m examples.teleop_vive_hand
    """

    @dataclass
    class ArmCfg:
        addr: str = "127.0.0.1:5555"
        robot_ip: str = "192.168.1.44"
        port: int = 7582
        session_id: int = 1
        robot_controller: str = "task_pos_ik"
        max_pos_speed: float = 0.15  # m/s — arm-node envelope behind the retargeter
        max_rot_speed: float = 0.25  # rad/s
        max_motor_speed: float = 0.4  # rad/s
        urdf_path: str = DEFAULT_URDF_PATH
        ee_frame: str = "end_effector"
        ik_kp: float = 8.0
        max_joint_accel: float | None = 2.0
        stream_l_mode: str = "time"
        stream_l_tt: float = 0.016
        stream_l_bt: float = 0.008
        stream_l_speed: float = 0.0
        stream_l_throttle: int = 2
        lowpass_alpha: float | None = 0.35
        log_diagnostics: bool = False
        cmd_freq: int = 50
        freq: int = 250

    arm: str = "KassowArm"
    arm_cfg: ArmCfg = field(default_factory=ArmCfg)

    # Inspire hand replaces a parallel-jaw gripper on the wrist.
    gripper: str | None = None
    gripper_cfg: None = None

    # -----------------------------------------------------------------------
    # Follower: Inspire RH56 dexterous hand
    # Joint order [pinky, ring, middle, index, thumb_flex, thumb_rot] in [0, 1],
    # matching what the Manus glove publishes.
    # -----------------------------------------------------------------------
    @dataclass
    class HandCfg:
        addr: str = "127.0.0.1:5556"
        port: str = "/dev/ttyUSB0"
        baudrate: int = 115200
        hand_id: int = 1
        generation: int = 3
        speed: int = 1000
        force: int = 300
        home_to_open: bool = True
        freq: int = 300
        timeout: float = 30.0

    hand: str | None = "InspireHand"
    hand_cfg: HandCfg = field(default_factory=HandCfg)

    cameras: dict[str, Camera] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Leader 1: Vive tracker → arm end-effector
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Leader 2: Manus glove → Inspire hand
    # On first run (no calibration_file) the node prompts for open/closed poses.
    # -----------------------------------------------------------------------
    teleop2: str | None = "ManusGlove"
    teleop2_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5581",
            glove_id=None,
            hand_motion="NoMotion",
            thumb_chord_weight=0.25,
            # calibration_file="manus_glove_cal.json",  # skips the prompts
            auto_calibrate=True,
            no_thumb_endpoints=False,
            curl=1.2,
            thumb_curl=1.45,
            freq=100,
            timeout=300.0,
        )
    )

    # -----------------------------------------------------------------------
    # Leader 3: keyboard clutch / orientation / recorder
    # -----------------------------------------------------------------------
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
    max_pos_speed: float = 0.15
    max_rot_speed: float = 0.5
    # Reachable shell around the Kassow base (KR1018-scale); tighten for your cell.
    min_radius: float = 0.25
    max_radius: float = 1.40
    min_z: float = 0.05
    tracking_grace: float = 0.25
    max_lag: float = 0.10

    arm_latency: float = 0.0
    gripper_latency: float = 0.0
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

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
