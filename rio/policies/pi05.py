# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""LeRobot π₀.₅ runtime. Hardware stays in RIO; this only loads a checkpoint."""

from pathlib import Path

import numpy as np

from rio.gripper import GripperBinarizer

try:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rtc import RTCConfig
    from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

    IMPORT_ERROR = None
except ImportError as e:
    torch = None
    PreTrainedConfig = None
    make_pre_post_processors = None
    RTCConfig = None
    _normalize_prev_actions_length = None
    IMPORT_ERROR = e

from ._policy import Policy

ACTION_DIM = 7
JOINT_ACTION_DIM = 8
STATE_JOINTS = 7
_ROTVEC_SLICE = slice(3, 6)
TCP_ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
JOINT_ACTION_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper")
GRIPPER_CLOSE = 0.0
GRIPPER_OPEN = 0.7
GRIPPER_THRESHOLD = 0.35
# Must exceed inference delay (~18 steps at 20 Hz). Keep in sync with
# examples/policy_inference.py RTC_EXECUTION_HORIZON.
RTC_EXECUTION_HORIZON = 18
RTC_MAX_GUIDANCE_WEIGHT = 10.0


def snap_gripper(
    values: np.ndarray,
    *,
    gripper_close: float = GRIPPER_CLOSE,
    gripper_open: float = GRIPPER_OPEN,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Map gripper readings to two targets. ``None`` threshold uses the midpoint."""
    return GripperBinarizer(gripper_close, gripper_open, gripper_threshold).action(values)


def _load_pi05_class():
    """Resolve the LeRobot class name across package layouts."""
    candidates = (
        "lerobot.policies.pi05.modeling_pi05",
        "lerobot.common.policies.pi05.modeling_pi05",
    )
    last_error: ImportError | None = None
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["PI05Policy", "Pi05Policy"])
        except ImportError as exc:
            last_error = exc
            continue
        for attr in ("PI05Policy", "Pi05Policy"):
            policy_cls = getattr(module, attr, None)
            if policy_cls is not None:
                return policy_cls
    raise ImportError("LeRobot π₀.₅ is not installed. Install with: pip install 'lerobot[pi]'") from last_error


def normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    """Convert HWC RGB bytes or normalized floats to contiguous float32 in [0, 1]."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or not array.size:
        raise ValueError(f"Expected a non-empty HWC RGB image, got {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array, dtype=np.float32) / 255.0
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"Expected uint8 or floating-point RGB pixels, got {array.dtype}")
    if not np.isfinite(array).all() or np.min(array) < 0.0 or np.max(array) > 1.0:
        raise ValueError("Floating-point RGB pixels must be finite and normalized to [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float32)


def format_rio_obs(
    obs: dict,
    instruction: str,
    *,
    binarize_gripper: bool = True,
    gripper_close: float = GRIPPER_CLOSE,
    gripper_open: float = GRIPPER_OPEN,
    gripper_threshold: float | None = None,
    gripper_binarizer: GripperBinarizer | None = None,
) -> dict:
    """Map a RIO observation dict to π₀.₅ numpy inputs.

    Args:
        obs: Keys ``camera_1`` / ``camera_2`` (H, W, C), uint8 or floats in [0, 1], ``proprio_joints`` (7,),
            ``gripper_position`` scalar. Optional pre-concatenated ``proprio``.
        instruction: Language task string.
        binarize_gripper: Snap analog gripper state to the two dataset targets.
        gripper_close: Closed command/state used when snapping.
        gripper_open: Open command/state used when snapping.
        gripper_threshold: Open/close cut. ``None`` uses the midpoint.

    Returns:
        Dict with ``state`` (8,), ``images`` keyed like the dataset, and ``task``.
    """
    if not instruction:
        raise ValueError("π₀.₅ needs a non-empty instruction")

    if "proprio" in obs and obs["proprio"] is not None:
        state = np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)
    else:
        joints = np.asarray(obs["proprio_joints"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(obs.get("gripper_position", 0.0), dtype=np.float32).reshape(-1)
        state = np.concatenate([joints, gripper])
    if state.shape[0] != STATE_JOINTS + 1 or not np.isfinite(state).all():
        raise ValueError(f"Expected exactly {STATE_JOINTS + 1} finite state dims, got {state.shape}")
    state = state.astype(np.float32, copy=True)
    if binarize_gripper:
        encoder = gripper_binarizer or GripperBinarizer(gripper_close, gripper_open, gripper_threshold)
        state[-1] = encoder.update(state[-1])

    images = {}
    for key in ("camera_1", "camera_2"):
        if key in obs and obs[key] is not None:
            images[key] = normalize_rgb_image(obs[key])
    if not images:
        raise ValueError("π₀.₅ observation needs camera_1 and/or camera_2 images")

    return {"state": state.astype(np.float32), "images": images, "task": instruction}


def principal_rotvec(rotvecs: np.ndarray) -> np.ndarray:
    """Map axis-angle to the principal value ``|r| ≤ π`` (same SO(3) pose).

    Training may unwrap rotvecs across ±π so the BC loss is continuous. IK
    still wants a standard rotvec; this fold is applied on the policy output.
    """
    from scipy.spatial.transform import Rotation as R

    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    if rotvecs.size == 0:
        return rotvecs.astype(np.float32)
    shape = rotvecs.shape
    out = R.from_rotvec(rotvecs.reshape(-1, 3)).as_rotvec()
    return np.asarray(out, dtype=np.float32).reshape(shape)


def _to_action_chunk(tensor, action_dim: int = ACTION_DIM) -> np.ndarray:
    """Squeeze a policy tensor to ``(T, action_dim)`` float32."""
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.array(arr[:, :action_dim], dtype=np.float32, copy=True)


def _rtc_predict_kwargs(observation: dict, device: str, execution_horizon: int = RTC_EXECUTION_HORIZON) -> dict:
    """Pad the transport prefix but guide only steps that actually exist in the old plan."""
    leftover = observation.get("rtc_leftover")
    if leftover is None or _normalize_prev_actions_length is None or torch is None:
        return {}
    leftover = np.asarray(leftover, dtype=np.float32)
    if leftover.size == 0:
        return {}
    if leftover.ndim == 1:
        leftover = leftover[None, :]
    available = int(np.asarray(observation.get("rtc_leftover_length", len(leftover))).reshape(-1)[0])
    if available < 0 or available > len(leftover):
        raise ValueError("Invalid RTC leftover length")
    if available == 0:
        return {}
    if leftover.ndim != 2 or not np.isfinite(leftover).all() or execution_horizon < 1:
        raise ValueError("RTC leftovers must be a finite action matrix with a positive horizon")
    leftover = leftover[:available]
    available = min(available, int(execution_horizon))
    delay = observation.get("rtc_inference_delay", 0)
    delay_i = int(np.asarray(delay, dtype=np.int32).reshape(-1)[0])
    leftover_t = torch.as_tensor(leftover, dtype=torch.float32, device=device)
    leftover_t = _normalize_prev_actions_length(leftover_t, target_steps=int(execution_horizon))
    return {
        "prev_chunk_left_over": leftover_t,
        "inference_delay": delay_i,
        "execution_horizon": available,
    }


def infer_action_space(policy_cfg, requested: str | None = None) -> str:
    """Return ``tcp`` or ``joints`` from checkpoint names, optionally checked against ``requested``."""
    names = tuple(getattr(policy_cfg, "action_feature_names", None) or ())
    inferred = "tcp"
    if names[:7] == JOINT_ACTION_NAMES[:7]:
        inferred = "joints"
    elif names[:6] == TCP_ACTION_NAMES[:6]:
        inferred = "tcp"
    if requested:
        want = str(requested).strip().lower()
        if want not in {"tcp", "joints"}:
            raise ValueError(f"action_space must be 'tcp' or 'joints', got {requested!r}")
        if names and inferred != want:
            raise ValueError(
                f"Checkpoint action_feature_names={list(names)} is {inferred}, "
                f"but this config requested {want}. Use Pi05LerobotCfg for TCP "
                "or Pi05LerobotJointCfg for joints."
            )
        return want
    return inferred


class Pi05(Policy):
    """Load a LeRobot π₀.₅ checkpoint and emit TCP (7) or joint (8) action chunks."""

    def __init__(
        self,
        policy_path: str | Path = "ckpts/pi05",
        compile: bool = False,
        device: str = "cuda:0",
        binarize_gripper: bool = True,
        gripper_close: float = GRIPPER_CLOSE,
        gripper_open: float = GRIPPER_OPEN,
        gripper_threshold: float | None = None,
        use_rtc: bool = True,
        rtc_execution_horizon: int = RTC_EXECUTION_HORIZON,
        rtc_max_guidance_weight: float = RTC_MAX_GUIDANCE_WEIGHT,
        action_space: str | None = None,
        **kwargs,
    ):
        if IMPORT_ERROR is not None:
            raise ImportError(
                "LeRobot π₀.₅ dependencies are missing. Install with: pip install 'lerobot[pi]'"
            ) from IMPORT_ERROR
        self.compile = compile
        self.device = device
        self.policy_path = policy_path
        self.binarize_gripper = binarize_gripper
        self.gripper_close = gripper_close
        self.gripper_open = gripper_open
        self.gripper_threshold = gripper_threshold
        self._gripper_binarizer = GripperBinarizer(gripper_close, gripper_open, gripper_threshold)
        self.use_rtc = use_rtc
        self.rtc_execution_horizon = rtc_execution_horizon
        self.rtc_max_guidance_weight = rtc_max_guidance_weight
        self.requested_action_space = action_space
        self.action_space = (action_space or "tcp").strip().lower()
        self.action_dim = JOINT_ACTION_DIM if self.action_space == "joints" else ACTION_DIM
        self.policy_cfg = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.instruction = None
        self._policy_cls = None
        self._resident = None

    def construct_policy(self, *, in_process: bool = False):
        from rio.policies.pi05_resident import resident_identity, try_connect

        if PreTrainedConfig is None or make_pre_post_processors is None or torch is None:
            raise ImportError("LeRobot and Torch are required to construct the policy") from IMPORT_ERROR
        self._policy_cls = _load_pi05_class()
        self.policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        self.action_space = infer_action_space(self.policy_cfg, self.requested_action_space)
        self.action_dim = JOINT_ACTION_DIM if self.action_space == "joints" else ACTION_DIM
        self.policy_cfg.device = self.device
        if not in_process:
            identity = resident_identity(
                self.policy_path,
                self.device,
                use_rtc=self.use_rtc,
                compile=self.compile,
                binarize_gripper=self.binarize_gripper,
                gripper_close=self.gripper_close,
                gripper_open=self.gripper_open,
                gripper_threshold=self.gripper_threshold,
                rtc_execution_horizon=self.rtc_execution_horizon,
                rtc_max_guidance_weight=self.rtc_max_guidance_weight,
                action_space=self.requested_action_space,
            )
            client = try_connect(identity)
            if client is not None:
                from loguru import logger

                logger.info("Attached to π₀.₅ resident pid {} on {}", client.pid, self.device)
                self._resident = client
                if self.instruction:
                    client.set_instruction(self.instruction)
                if client.meta.get("action_space"):
                    self.action_space = str(client.meta["action_space"])
                if client.meta.get("action_dim"):
                    self.action_dim = int(client.meta["action_dim"])
                return
        self.policy = self._policy_cls.from_pretrained(self.policy_path, config=self.policy_cfg)
        if self.use_rtc:
            if RTCConfig is None:
                raise ImportError("The installed LeRobot version has no RTC support")
            self.policy.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.rtc_execution_horizon,
                max_guidance_weight=self.rtc_max_guidance_weight,
            )
            self.policy.init_rtc_processor()
        if self.compile:
            self.policy.model.sample_actions = torch.compile(self.policy.model.sample_actions)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def set_instruction(self, instruction):
        if instruction != self.instruction:
            self.reset()
        self.instruction = instruction
        resident = getattr(self, "_resident", None)
        if resident is not None:
            resident.set_instruction(instruction)

    def reset(self):
        """Start a new rollout without carrying over gripper, processor, or action history."""
        encoder = getattr(self, "_gripper_binarizer", None)
        if encoder is not None:
            encoder.reset()
        resident = getattr(self, "_resident", None)
        if resident is not None:
            resident.reset()
            return
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def _process_observation(self, observation):
        if torch is None:
            raise ImportError("Torch is required to process policy observations") from IMPORT_ERROR
        instruction = self.instruction
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Set a non-empty task instruction before inference")
        preprocessor = self.preprocessor
        if preprocessor is None:
            raise RuntimeError("The policy preprocessor has not been constructed")
        encoder = getattr(self, "_gripper_binarizer", None)
        if encoder is None:
            encoder = GripperBinarizer(self.gripper_close, self.gripper_open, self.gripper_threshold)
            self._gripper_binarizer = encoder
        packed = format_rio_obs(
            observation,
            instruction,
            binarize_gripper=self.binarize_gripper,
            gripper_close=self.gripper_close,
            gripper_open=self.gripper_open,
            gripper_threshold=self.gripper_threshold,
            gripper_binarizer=encoder,
        )
        batch = {
            "observation.state": torch.as_tensor(packed["state"], device=self.device),
            "task": [packed["task"]],
        }
        for name, image in packed["images"].items():
            tensor = torch.as_tensor(image, device=self.device)
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            batch[f"observation.images.{name}"] = tensor
        return preprocessor(batch)

    def inference(self, observation, current_plan=None):
        resident = getattr(self, "_resident", None)
        if resident is not None:
            return resident.inference(observation)
        if self.policy is None:
            raise RuntimeError("Policy.construct_policy not called")
        rtc_kwargs = _rtc_predict_kwargs(observation, self.device, self.rtc_execution_horizon)
        batch = self._process_observation(observation)
        actions = self.policy.predict_action_chunk(batch, **rtc_kwargs)
        action_dim = int(getattr(self, "action_dim", ACTION_DIM) or ACTION_DIM)
        action_space = getattr(self, "action_space", "tcp")
        original = _to_action_chunk(actions, action_dim)
        postprocessor = self.postprocessor
        if postprocessor is None:
            raise RuntimeError("The policy postprocessor has not been constructed")
        postprocessed = postprocessor(actions)
        chunk = _to_action_chunk(postprocessed, action_dim)
        if action_space == "tcp":
            chunk[:, _ROTVEC_SLICE] = principal_rotvec(chunk[:, _ROTVEC_SLICE])
        if self.binarize_gripper:
            chunk[:, -1] = snap_gripper(
                chunk[:, -1],
                gripper_close=self.gripper_close,
                gripper_open=self.gripper_open,
                gripper_threshold=self.gripper_threshold,
            )
        else:
            chunk[:, -1] = np.clip(chunk[:, -1], 0.0, 1.0)
        return {"actions": chunk, "original_actions": original}
