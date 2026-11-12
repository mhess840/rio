from dataclasses import dataclass, field

from rio.cfg import NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "vive_inspire_teleop"


@dataclass
class Ur5eViveInspireStation:
    """UR5e end-effector teleop from a Vive tracker, with a Manus glove driving
    an Inspire hand mounted on the wrist.

    Run with `examples/teleop_vive_hand.py`. The action vector is 13 wide:
    6 arm (position + axis-angle) + 1 gripper slot + 6 hand DOF.
    """

    # -----------------------------------------------------------------------
    # Follower: UR5e in Cartesian servo mode, driven by moveL
    # -----------------------------------------------------------------------
    arm: str = "UrArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            robot_ip="192.168.1.15",
            robot_model="ur5e",
            robot_controller="task_pos",
            # Second speed limit behind the retargeter's own, applied by the
            # arm node's trajectory interpolator.
            max_pos_speed=0.25,
            max_rot_speed=0.3,
            freq=125,
            # Measure these for the Inspire hand mounted at the flange, otherwise
            # the tool rotates about the flange instead of the palm.
            #   tcp_offset_pose=[x, y, z, rx, ry, rz]
            tcp_offset_pose=None,
            payload_mass=None,
            timeout=30.0,
            # Scale the movement
            # pos_scale=0.5,
            # rot_scale=0.75,
        )
    )

    # The Inspire hand replaces the gripper on the wrist.
    gripper: str | None = None

    # -----------------------------------------------------------------------
    # Follower: Inspire RH56 dexterous hand
    # Joint order [pinky, ring, middle, index, thumb_flex, thumb_rot] in [0, 1],
    # which is exactly what the Manus glove publishes.
    # -----------------------------------------------------------------------
    hand: str | None = "InspireHand"
    hand_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5556",
            port="/dev/ttyUSB1",
            baudrate=115200,
            hand_id=1,
            generation=3,
            speed=1000,
            force=200,
            home_to_open=True,
            freq=300,
            timeout=30.0,
        )
    )

    # -----------------------------------------------------------------------
    # Leader 1: Vive tracker worn on the wrist, driving the arm end-effector
    # -----------------------------------------------------------------------
    teleop: str = "ViveTracker"
    teleop_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5580",
            serial=None,  # first tracker SteamVR reports, or "LHR-XXXXXXXX"
            fix_base_channels=True,
            freq=250,
            # A base station channel fix costs one Bluetooth scan at startup.
            timeout=60.0,
        )
    )

    # -----------------------------------------------------------------------
    # Leader 2: Manus glove, driving the hand
    # On first run (no calibration_file) the node prompts interactively for
    # open/closed poses before publishing.
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
    # Leader 3: keyboard, used for the arm clutch
    # -----------------------------------------------------------------------
    teleop_keyboard: str | None = "Keyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(default_factory=lambda: NodeCfg(addr="127.0.0.1:5582", freq=100))

    # -----------------------------------------------------------------------
    # Vive retargeting and safety
    # -----------------------------------------------------------------------
    clutch_key: str = "c"
    orientation_key: str = "o"
    # How far the TCP travels per unit of hand travel. Read by the teleop script,
    # not by the arm node, so these belong here rather than in arm_cfg.
    pos_scale: float = 1
    rot_scale: float = 1
    orientation_enabled: bool = True
    # SteamVR's frame is gravity-aligned, so its vertical axis already agrees with
    # the robot's and the only frame unknown is a rotation about it. The guided
    # calibration runs automatically when the file below is missing.
    yaw_offset: float = 0.0
    yaw_calibration_file: str | None = "vive_yaw_cal.json"
    calibrate_yaw: bool = False
    # Shorter sweeps let hand tremor dominate: at 20 cm, 1 cm of wobble is ~3 deg.
    min_sweep_travel: float = 0.30
    # Warn if the two sweeps are not this close to perpendicular, in degrees.
    max_sweep_skew: float = 20.0
    # Per-step limits applied by the retargeter, ahead of the arm node's own.
    max_pos_speed: float = 0.15
    max_rot_speed: float = 0.5
    # Reachable shell around the UR base, in meters.
    min_radius: float = 0.20
    max_radius: float = 0.75
    min_z: float = 0.05
    # Release the clutch after this long without a valid tracker pose.
    tracking_grace: float = 0.25
    # Release if the command runs this far ahead of the measured TCP.
    max_lag: float = 0.10

    arm_latency: float = 0.0
    gripper_latency: float = 0.0
    freq: int = 50

    mw: str = "Thread"
    mp_method: str = "spawn"
    action_space: str = "task_pos"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = ""
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))
