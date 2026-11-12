from dataclasses import dataclass, field

from rio.cfg import NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "manus_inspire_teleop_peace_sign_camera2"


@dataclass
class ManusInspireStation:
    # -----------------------------------------------------------------------
    # Follower: Inspire RH56 dexterous hand (treated as the "arm" so that
    # SingleArm.move() routes JOINT_POS commands to InspireHand.moveJ())
    # -----------------------------------------------------------------------
    arm: str = "InspireHand"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5555",
            port="/dev/ttyUSB1",
            baudrate=115200,
            hand_id=1,
            generation=3,
            speed=1000,
            force=200,
            home_to_open=True,
            freq=300,
        )
    )

    gripper: str | None = None

    # -----------------------------------------------------------------------
    # Leader: Manus haptic glove
    # Publishes calibrated finger_openness / joint_q in [0, 1]:
    #   [pinky, ring, middle, index, thumb_flex, thumb_rot]
    #
    # On first run (no calibration_file found) the node will interactively
    # prompt for open/closed poses before publishing begins.
    # Pass an existing .json file (e.g. from glove_to_inspire_calibrated_clean.py)
    # to skip the prompts.
    # -----------------------------------------------------------------------
    teleop: str = "ManusGlove"
    teleop_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5570",
            glove_id=None,  # auto-select lowest available glove ID
            hand_motion="NoMotion",
            thumb_chord_weight=0.25,
            # calibration_file="my_cal.json",  # uncomment to use an existing file
            auto_calibrate=True,  # run interactive calibration if file not found
            no_thumb_endpoints=False,  # set True for quick 2-pose calibration only
            curl=1.2,
            thumb_curl=1.45,
            freq=100,
            # Allow up to 5 minutes for the user to complete calibration poses.
            # Once a cal file exists this can be reduced (e.g. timeout=30).
            timeout=300.0,
        )
    )

    teleop_keyboard: str | None = "Keyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(default_factory=lambda: NodeCfg(addr="127.0.0.1:5571", freq=100))

    class Camera:
        def __init__(self, cam_type: str, module: str = "cameras", **kwargs):
            self.cam_type = cam_type
            self.module = module
            self.cfg = kwargs

    # cameras: dict[str, Camera] = field(
    #     default_factory=lambda: {
    #         "camera_0": ManusInspireStation.Camera(
    #             addr="127.0.0.1:5130",
    #             cam_type="Zed",
    #             # Use get_connected_cameras() from rio_hw.cameras.zed to choose a device.
    #             serial="36724908",
    #             resolution=(720, 1280),
    #             freq=60,
    #             # ThreadClient waits this long for pub + req readiness (each phase uses the same value).
    #             timeout=30.0,
    #         ),
    #     }
    # )

    # -----------------------------------------------------------------------
    # Control settings
    # -----------------------------------------------------------------------
    # Alignment check is disabled: glove openness [0,1] != hand angles [0,1]
    # in general; use the Manus SDK calibration workflow instead.
    check_alignment: bool = False
    # Smoothing is disabled: the hand firmware's own speed ramp is sufficient.
    use_leader_smoothing: bool = False
    invert_gripper: bool = False

    arm_latency: float = 0.0
    gripper_latency: float = 0.0
    freq: int = 30

    mw: str = "Thread"
    mp_method: str = "spawn"
    # JOINT_POS routes teleop joint_q directly to InspireHand.moveJ()
    action_space: str = "JOINT_POS"
    embodiment_type: str = "SINGLE_ARM"
    startup_delay: float = 1.0

    instruction: str = "Make a peace sign."
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))
