"""Kassow setup with Vive teleoperation, an AG95 gripper, and RealSense cameras."""

from dataclasses import dataclass, field

from rio.cfg import Camera
from rio.cfg.common import RecorderCfg
from rio.cfg.node import NodeCfg

from .kassow_vive import KassowViveStation

TASK = "kassow_vive_gripper"


@dataclass
class KassowViveGripperStation(KassowViveStation):
    """Kassow arm driven by Vive, with an AG95 gripper and RealSense cameras."""

    @dataclass
    class GripperCfg:
        robot_port: str = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG004P3Y-if00-port0"
        connection_type: str = "SERIAL"
        baudrate: int = 115200
        member_id: int = 1
        timeout: float = 0.1
        max_gripper_speed: float | None = None
        force: int = 50
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
    teleop_keyboard_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(addr="127.0.0.1:5582", freq=100)
    )

    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))
