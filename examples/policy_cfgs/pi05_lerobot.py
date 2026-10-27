"""Kassow + AG95 + 2× RealSense inference for a LeRobot π₀.₅ checkpoint.

Run with:

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \\
        --policy-path ckpts/pi05 \\
        --instruction "pick up the blue cylinder and place it on the red plate"

RTC off (one 50-step chunk to the end, then a pause for the next inference).
``__post_init__`` copies ``policy_node_cfg.use_rtc`` onto ``policy_cfg``:

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \\
        --policy-path ckpts/pi05 \\
        --instruction "pick up the blue cylinder and place it on the red plate" \\
        --policy-cfg.device cuda:1 \\
        --policy-node-cfg.no-use-rtc
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.cfg.kassow_vive import KassowViveStation
from examples.cfg.kassow_vive_gripper import KassowViveGripperStation
from rio.cfg.common import VisualizerCfg


@dataclass
class Pi05LerobotCfg(KassowViveGripperStation):
    """Live π₀.₅ on the Kassow station. Vive/clicker are off; action is TCP + gripper."""

    policy_path: str = "ckpts/pi05"
    instruction: str | None = "pick up the blue cylinder and place it on the red plate"

    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None
        resolutions: list[tuple[int, int]] | None = None
        proprio_dim: int = 8
        action_dim: int = 7
        chunk_size: int = 16
        use_rtc: bool = True
        freq: int = 20
        action_hz: int = 20
        # Infer when this many 20 Hz steps remain (delay ~18 plus slack).
        rtc_queue_threshold: int = 22
        # Play this many 20 Hz steps of each predicted chunk, then replan.
        # None / 0 → checkpoint ``n_action_steps`` (25 on the current ckpt).
        n_execute: int | None = None
        # After a chunk finishes, hold the last command this long before the next image.
        # 0 plans as soon as the queue is empty.
        chunk_pause_s: float = 0.0
        infer_lead_s: float = 1.5
        max_buffer_size: int = 30
        chunk_request_threshold: float = 1.0
        camera_keys: list[str] = field(default_factory=lambda: ["camera_1", "camera_2"])
        timeout: float = 180.0

        def __post_init__(self):
            if self.resolutions is None:
                self.resolutions = [(480, 640), (480, 640)]

    @dataclass
    class PolicyConfig:
        policy_path: str | None = None
        compile: bool = False
        device: str = "cuda:0"
        # Match convert_pi05: AG95 is two-state. Set False for a continuous gripper.
        binarize_gripper: bool = True
        gripper_close: float = 0.0
        gripper_open: float = 0.7
        gripper_threshold: float | None = None
        use_rtc: bool = True
        # ``tcp`` (7D xyz+rotvec) or ``joints`` (8D q + gripper). Must match the checkpoint.
        action_space: str = "tcp"

    policy: str = "Pi05"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)
    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)

    # Policy rollouts do not need the tracker or clicker.
    teleop: str | None = None
    teleop_cfg: None = None
    teleop_keyboard: str | None = None
    teleop_keyboard_cfg: None = None
    teleop_clicker: str | None = None
    teleop_clicker_cfg: None = None
    recorder: str | None = None

    arm_latency: float = 0.0
    gripper_latency: float = 0.1

    action_space: str = "task_pos"
    mw: str = "Thread"
    mp_method: str | None = "spawn"
    # 100 Hz control; ActionInterpolator x5 fills 20 Hz policy actions.
    freq: int = 100

    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    dummy_obs: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "camera_1": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_2": np.zeros((480, 640, 3), dtype=np.uint8),
            "proprio_joints": np.zeros((7,), dtype=np.float32),
            "gripper_position": 0.7,
        }
    )

    def __post_init__(self):
        KassowViveStation.__post_init__(self)
        if not self.policy_path:
            raise ValueError("policy_path is required — pass --policy-path to the LeRobot checkpoint folder")
        self.policy_cfg.policy_path = self.policy_path
        self.policy_cfg.use_rtc = bool(self.policy_node_cfg.use_rtc) and bool(self.policy_cfg.use_rtc)
        self.policy_node_cfg.use_rtc = self.policy_cfg.use_rtc
        self.policy_node_cfg.instruction = self.instruction
        ckpt_cfg = Path(self.policy_path) / "config.json"
        if ckpt_cfg.is_file():
            data = json.loads(ckpt_cfg.read_text())
            n = data.get("n_action_steps") or data.get("chunk_size")
            if n:
                self.policy_node_cfg.chunk_size = int(n)
            names = tuple(data.get("action_feature_names") or ())
            requested = str(getattr(self.policy_cfg, "action_space", "tcp") or "tcp").lower()
            if names[:6] == ("x", "y", "z", "rx", "ry", "rz") and requested == "joints":
                raise ValueError(
                    f"{self.policy_path} is a TCP checkpoint; use POLICY=Pi05LerobotCfg, not JointCfg"
                )
            if names[:7] == ("j1", "j2", "j3", "j4", "j5", "j6", "j7") and requested == "tcp":
                raise ValueError(
                    f"{self.policy_path} is a joint checkpoint; use POLICY=Pi05LerobotJointCfg"
                )
        # Inference worker poll. The control loop uses station freq (100 Hz).
        self.policy_node_cfg.freq = self.policy_node_cfg.action_hz
        self.arm_cfg.robot_controller = "task_pos_ik"
        self.policy_node_cfg.camera_keys = list(self.cameras.keys())
        self.policy_node_cfg.resolutions = [cam.cfg["resolution"] for cam in self.cameras.values()]
        if len(self.policy_node_cfg.camera_keys) != len(self.policy_node_cfg.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")
