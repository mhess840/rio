from dataclasses import dataclass, field

from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

from rio.cfg import Camera, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class KassowStation:
    """Kassow KR-series arm teleoperated in end-effector space with a Spacemouse.

    Default EEF path (`task_pos` + `task_pos_ik`): Spacemouse Cartesian targets are
    solved with local Pinocchio IK on the KR1018 URDF, then tracked with joint
    velocities via `directJControl` (same smooth path as `teleop_joint_vel`).

    Legacy streamed `moveL` remains available with
    `--arm-cfg.robot-controller task_pos`.

    For axis→joint teleop use `--arm-cfg.robot-controller joint_vel` with
    `examples.teleop_joint_vel`.
    """

    @dataclass
    class ArmCfg:
        addr: str = "127.0.0.1:5555"
        robot_ip: str = "192.168.1.44"
        port: int = 7582
        session_id: int = 1
        robot_controller: str = "task_pos_ik"
        max_pos_speed: float = 0.15  # m/s — EEF teleop / IK envelope
        max_rot_speed: float = 0.25  # rad/s
        max_motor_speed: float = 0.4  # rad/s
        urdf_path: str = DEFAULT_URDF_PATH
        ee_frame: str = "end_effector"
        ik_kp: float = 8.0  # task-space P (1/s): lead → twist before Jacobian
        max_joint_accel: float | None = 2.0  # rad/s² — slew-limit qd
        stream_l_mode: str = "time"  # "time" (TT_TIME) | "speed" (TT_WS_TARGET_SPEED)
        stream_l_tt: float = 0.016  # 2x send period at 125 Hz (real_time_patterns.rst)
        stream_l_bt: float = 0.008  # ~50% of TT
        stream_l_speed: float = 0.0  # m/s for "speed" mode; 0 derives it
        stream_l_throttle: int = 2  # every 2nd waitSync ≈ 125 Hz
        lowpass_alpha: float | None = 0.35  # EMA on qd only (task_pos_ik)
        log_diagnostics: bool = False
        cmd_freq: int = 100  # kept in sync with the station freq below
        freq: int = 250

    arm: str = "KassowArm"
    arm_cfg: ArmCfg = field(default_factory=ArmCfg)

    @dataclass
    class GripperCfg:
        robot_port: str = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG004P3Y-if00-port0"
        connection_type: str = "SERIAL"
        baudrate: int = 115200
        member_id: int = 1
        timeout: float = 0.1
        # Let the AG95 execute one final position command instead of streaming
        # interpolated RS-485 waypoints.
        max_gripper_speed: float | None = None
        force: int = 100
        calibrate: bool = False
        startup_position: float | None = None
        feedback_freq: float = 2.0
        freq: int = 30

    gripper: str | None = "AgGripper"
    gripper_cfg: GripperCfg = field(default_factory=GripperCfg)

    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            "camera_1": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="112322070077",
                model="D400",  # D435i product line
                enable_depth=False,
                hardware_reset=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
        }
    )

    @dataclass
    class TeleopCfg:
        addr: str = "127.0.0.1:5000"
        # Spacenav device (X right, Y away, Z up) → Kassow base (Z up).
        # Cell-tuned: device Z → +X, device X → -Y, device Y → +Z.
        # Override if your cell is rotated relative to the mouse.
        tx_zup_spnav: tuple[float, ...] = (
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )

    teleop: str = "Keyboard"  # Gamepad | Keyboard | SshKeyboard | Spacemouse
    teleop_cfg: TeleopCfg = field(default_factory=TeleopCfg)

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
        # The arm node sizes its envelope and step guards against the rate targets
        # actually arrive at, so a mismatch here would silently cap the speed.
        self.arm_cfg.cmd_freq = self.freq
        if not self.arm_cfg.urdf_path:
            self.arm_cfg.urdf_path = self.urdf_path
        if not self.urdf_path:
            self.urdf_path = self.arm_cfg.urdf_path
        # Keep the KORD command path aligned with the embodiment action space.
        # task_pos defaults to local IK → joint vel; explicit `task_pos` keeps streamL.
        space = self.action_space.lower()
        if space == "task_pos":
            if self.arm_cfg.robot_controller not in ("task_pos", "task_pos_ik"):
                self.arm_cfg.robot_controller = "task_pos_ik"
        elif space in ("joint_pos", "joint_vel"):
            self.arm_cfg.robot_controller = space
        # pynput Keyboard does not receive keys under Wayland; use stdin instead.
        if self.teleop == "Keyboard":
            import os

            from loguru import logger

            if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
                logger.warning(
                    "Wayland session detected: switching teleop Keyboard → SshKeyboard "
                    "(pynput cannot capture keys here). Use WASD/QE in this terminal."
                )
                self.teleop = "SshKeyboard"
