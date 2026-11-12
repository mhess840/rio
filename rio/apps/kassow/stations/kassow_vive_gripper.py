"""Kassow setup with Vive teleoperation, an AG95 gripper, and RealSense cameras."""

from dataclasses import dataclass, field

from rio.apps.kassow.hardware import GRIPPER_PORT, OVERVIEW_SERIAL, WRIST_SERIAL
from rio.cfg import Camera
from rio.cfg.common import RecorderCfg
from rio.cfg.node import NodeCfg

from ..experiments import apply_kassow_experiment
from .kassow_vive import KassowViveStation

TASK = "kassow_vive_gripper"


@dataclass
class KassowViveGripperStation(KassowViveStation):
    """Kassow arm driven by Vive, with an AG95 gripper and RealSense cameras."""

    experiment: str = ""
    overview_serial: str | None = None
    home_file: str | None = None
    pos_scale: float = 1.0
    gripper_open: float = 0.7
    gripper_close: float = 0.0

    @dataclass
    class GripperCfg:
        robot_port: str = GRIPPER_PORT
        connection_type: str = "SERIAL"
        baudrate: int = 115200
        member_id: int = 1
        timeout: float = 0.5
        ready_timeout: float = 30.0
        max_gripper_speed: float | None = None
        force: int = 50
        # One full 0xA5 homing at start. Off after a successful home.
        calibrate: bool = False
        startup_position: float | None = 0.7
        feedback_freq: float = 2.0
        freq: int = 30

    gripper: str | None = "AgGripper"
    # The Vive base station has no gripper (``None`` hides it from its CLI); this one adds the AG95.
    gripper_cfg: GripperCfg = field(default_factory=GripperCfg)  # ty: ignore[invalid-attribute-override]

    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            "camera_1": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial=WRIST_SERIAL,
                model="D400",
                enable_depth=False,
                hardware_reset=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            "camera_2": Camera(
                addr="127.0.0.1:5131",
                cam_type="Realsense",
                serial=OVERVIEW_SERIAL,
                model="D400",
                enable_depth=False,
                hardware_reset=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
        }
    )

    # Read clutch, orientation, recorder, and gripper keys from this terminal.
    teleop_keyboard: str | None = "SshKeyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(default_factory=lambda: NodeCfg(addr="127.0.0.1:5582", freq=100))
    # Spare Logitech Unifying mouse (not the HP desk mouse).
    # Left=close, right=open, back=finer scale, forward=coarser scale.
    teleop_clicker: str | None = "Clicker"
    teleop_clicker_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            addr="127.0.0.1:5583",
            freq=100,
            device_path="/dev/input/by-id/usb-Logitech_USB_Receiver-if02-event-mouse",
        )
    )

    recorder_cfg: RecorderCfg = field(
        default_factory=lambda: RecorderCfg(
            path=f"data/{TASK}/",
            codec_options={"crf": "28", "preset": "ultrafast"},
            start_recording=False,
        )
    )

    def __post_init__(self) -> None:
        apply_kassow_experiment(self)
        KassowViveStation.__post_init__(self)
