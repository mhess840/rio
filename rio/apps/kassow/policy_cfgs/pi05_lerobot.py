"""Kassow + AG95 + 2 RealSense cameras for a LeRobot π₀.₅ checkpoint.

Run with:

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve \\
        --policy-path ckpts/pi05 \\
        --instruction "pick up the blue cylinder and place it on the red plate"

RTC on, playing 30 of the 50 predicted steps. A new chunk is requested while 18
steps remain, which covers the ~0.35 s inference. ``__post_init__`` copies
``policy_node_cfg.use_rtc`` onto ``policy_cfg``. Turn the overlap off with:

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve \\
        --policy-path ckpts/pi05 \\
        --instruction "pick up the blue cylinder and place it on the red plate" \\
        --policy-cfg.device cuda:1 \\
        --policy-node-cfg.no-use-rtc
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from rio.apps.kassow.experiments import find_model_owner, resolve_experiment, resolve_model
from rio.apps.kassow.stations import KassowViveGripperStation
from rio.cfg.common import VisualizerCfg
from rio.cfg.node import NodeCfg

POLICY_MAX_POS_SPEED = 0.30
POLICY_MAX_ROT_SPEED = 0.40
POLICY_MAX_MOTOR_SPEED = 0.40


@dataclass
class Pi05LerobotCfg(KassowViveGripperStation):
    """Live π₀.₅ on the Kassow station. Vive/clicker are off; action is TCP + gripper."""

    policy_path: str = ""
    instruction: str = ""
    allow_unlisted_instruction: bool = False

    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None
        resolutions: list[tuple[int, int]] | None = None
        proprio_dim: int = 8
        action_dim: int = 7
        chunk_size: int = 50
        use_rtc: bool = True
        freq: int = 20
        action_hz: int = 20
        # Play this many 20 Hz steps of each predicted chunk (the ckpt predicts 50).
        n_execute: int | None = 30
        # Start the next chunk while this many steps are still queued.
        # 18 steps is 0.9 s, above the ~0.35 s inference so the queue does not run dry.
        rtc_queue_threshold: int = 18
        # Guide the new chunk over this many steps. Must be <= the leftover at the
        # request (the threshold) or the prefix is padded with zeros.
        rtc_execution_horizon: int = 18
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
        gripper_state_source: str = "commanded"
        rtc_execution_horizon: int = 18
        rtc_max_guidance_weight: float = 10.0
        use_rtc: bool = True
        # ``tcp`` (7D xyz+rotvec) or ``joints`` (8D q + gripper). Must match the checkpoint.
        action_space: str = "tcp"
        resident_slot: str = ""

    policy: str = "Pi05"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)
    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)
    gripper_cfg: KassowViveGripperStation.GripperCfg = field(
        default_factory=lambda: KassowViveGripperStation.GripperCfg(startup_position=None)
    )

    # Policy rollouts do not need the tracker or clicker.
    teleop: str | None = None
    teleop_cfg: NodeCfg | None = None
    teleop_keyboard: str | None = None
    teleop_keyboard_cfg: NodeCfg | None = None
    teleop_clicker: str | None = None
    teleop_clicker_cfg: NodeCfg | None = None
    recorder: str | None = None

    arm_latency: float = 0.0
    gripper_latency: float = 0.1

    action_space: str = "task_pos"
    mw: str = "Thread"
    mp_method: str = "spawn"
    # 100 Hz control; ActionInterpolator x5 fills 20 Hz policy actions.
    freq: int = 100

    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    dummy_obs: dict[str, np.ndarray | float] = field(
        default_factory=lambda: {
            "camera_1": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_2": np.zeros((480, 640, 3), dtype=np.uint8),
            "proprio_joints": np.zeros((7,), dtype=np.float32),
            "gripper_position": 0.7,
        }
    )

    def __post_init__(self):
        KassowViveGripperStation.__post_init__(self)
        self._bind_experiment_model()
        self.policy_cfg.policy_path = self.policy_path
        self.policy_cfg.use_rtc = bool(self.policy_node_cfg.use_rtc) and bool(self.policy_cfg.use_rtc)
        self.policy_node_cfg.use_rtc = self.policy_cfg.use_rtc
        self.policy_node_cfg.instruction = self.instruction
        ckpt_cfg = Path(self.policy_path) / "config.json"
        checkpoint_steps = self.policy_node_cfg.chunk_size
        if ckpt_cfg.is_file():
            data = json.loads(ckpt_cfg.read_text())
            n = data.get("chunk_size") or data.get("n_action_steps")
            if n:
                self.policy_node_cfg.chunk_size = int(n)
            checkpoint_steps = int(data.get("n_action_steps") or self.policy_node_cfg.chunk_size)
            names = tuple(data.get("action_feature_names") or ())
            requested = str(getattr(self.policy_cfg, "action_space", "tcp") or "tcp").lower()
            if names[:6] == ("x", "y", "z", "rx", "ry", "rz") and requested == "joints":
                raise ValueError(f"{self.policy_path} is a TCP checkpoint; use POLICY=Pi05LerobotCfg, not JointCfg")
            if names[:7] == ("j1", "j2", "j3", "j4", "j5", "j6", "j7") and requested == "tcp":
                raise ValueError(f"{self.policy_path} is a joint checkpoint; use POLICY=Pi05LerobotJointCfg")
        requested = self.policy_node_cfg.n_execute
        effective = min(
            int(requested) if requested is not None and requested > 0 else checkpoint_steps,
            self.policy_node_cfg.chunk_size,
        )
        if self.policy_node_cfg.use_rtc:
            if effective < 2 or self.policy_node_cfg.rtc_queue_threshold < 1 or self.policy_node_cfg.rtc_execution_horizon < 1:
                raise ValueError("RTC needs at least two execution steps and positive queue/guidance horizons")
            self.policy_node_cfg.rtc_queue_threshold = min(self.policy_node_cfg.rtc_queue_threshold, effective - 1)
            self.policy_node_cfg.rtc_execution_horizon = min(
                self.policy_node_cfg.rtc_execution_horizon, self.policy_node_cfg.rtc_queue_threshold
            )
        self.policy_cfg.rtc_execution_horizon = self.policy_node_cfg.rtc_execution_horizon
        # Inference worker poll. The control loop uses station freq (100 Hz).
        self.policy_node_cfg.freq = self.policy_node_cfg.action_hz
        self.arm_cfg.robot_controller = "task_pos_ik"
        if self.arm_cfg.max_pos_speed == 0.45:
            self.arm_cfg.max_pos_speed = POLICY_MAX_POS_SPEED
        if self.arm_cfg.max_rot_speed == 1.0:
            self.arm_cfg.max_rot_speed = POLICY_MAX_ROT_SPEED
        if self.arm_cfg.max_motor_speed == 1.0:
            self.arm_cfg.max_motor_speed = POLICY_MAX_MOTOR_SPEED
        self.policy_node_cfg.camera_keys = list(self.cameras.keys())
        self.policy_node_cfg.resolutions = [cam.cfg["resolution"] for cam in self.cameras.values()]
        if len(self.policy_node_cfg.camera_keys) != len(self.policy_node_cfg.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")

    def start_resident(self):
        """Attach to or start this config's GPU resident; the weights stay loaded after the process exits."""
        from rio.policies.pi05_resident import start_resident

        return start_resident(self)

    def stop_resident(self) -> bool:
        """Unload this config's resident if it holds this checkpoint."""
        from rio.policies.pi05_resident import shutdown_resident

        return shutdown_resident(
            self.policy_cfg.device, policy_path=self.policy_path or None, slot=self.policy_cfg.resident_slot
        )

    def _bind_experiment_model(self) -> None:
        """Resolve the selected model and reject incompatible station or task settings."""
        if getattr(self, "unload_policy", False):
            return
        experiment = resolve_experiment(self.experiment)
        if not self.policy_path:
            name, path = resolve_model(experiment)
            self.policy_path = str(path)
            logger.info("Experiment {}: model {} ({})", experiment.name, name, path.resolve())
        owner = find_model_owner(self.policy_path)
        if owner is not None and owner[0] != experiment.name:
            raise ValueError(
                f"{self.policy_path} belongs to experiment {owner[0]!r}, not {experiment.name!r}; "
                f"pass --experiment {owner[0]} for its camera and home configuration"
            )
        instructions = list(owner[2].get("instructions") or []) if owner is not None else []
        if not self.instruction:
            if not instructions:
                raise ValueError(f"{self.policy_path} is not a registered model; pass --instruction")
            self.instruction = instructions[0]
        elif instructions and self.instruction not in instructions and not self.allow_unlisted_instruction:
            listed = "\n  ".join(repr(text) for text in instructions)
            raise ValueError(
                f"Instruction {self.instruction!r} is not a task of this model. Trained tasks:\n  {listed}\n"
                "Pass --allow-unlisted-instruction to use it anyway."
            )
