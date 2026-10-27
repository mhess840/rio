# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert Kassow robodm ``.vla`` trajectories to a LeRobot v3 π₀.₅ dataset."""

from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import robodm
from loguru import logger

try:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import lerobot.datasets.video_utils as video_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _LEROBOT_IMPORT_ERROR = None
except ImportError as e:
    lerobot_dataset = None
    video_utils = None
    LeRobotDataset = None
    _LEROBOT_IMPORT_ERROR = e


def _patch_lerobot_pyav() -> None:
    """Rio pins ``av<=13``; LeRobot 0.6 reads ``Codec.canonical_name`` from newer PyAV."""
    if video_utils is None:
        return
    if getattr(video_utils.get_video_info, "_rio_av13_patch", False):
        return

    from dataclasses import asdict

    try:
        from lerobot.configs.video import DepthEncoderConfig
    except ImportError:
        DepthEncoderConfig = type(None)

    def get_video_info(video_path: Path | str, video_encoder=None, **_kwargs) -> dict:
        info: dict = {}
        with av.open(str(video_path), "r") as video_file:
            try:
                stream = video_file.streams.video[0]
            except IndexError:
                return {}
            codec = stream.codec
            info["video.height"] = stream.height
            info["video.width"] = stream.width
            info["video.codec"] = getattr(codec, "canonical_name", None) or getattr(codec, "name", "h264")
            info["video.pix_fmt"] = stream.pix_fmt
            info["video.fps"] = int(stream.base_rate) if stream.base_rate else 0
            channels_fn = getattr(video_utils, "get_pix_fmt_channels", None) or getattr(
                video_utils, "get_video_pixel_channels", lambda _fmt: 3
            )
            info["video.channels"] = channels_fn(stream.pix_fmt)
        info.update(**video_utils.get_audio_info(video_path))
        if video_encoder is not None:
            for field_name, field_value in asdict(video_encoder).items():
                if field_name == "vcodec":
                    continue
                info.setdefault(f"video.{field_name}", field_value)
        info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)
        return info

    get_video_info._rio_av13_patch = True
    video_utils.get_video_info = get_video_info
    if lerobot_dataset is not None:
        lerobot_dataset.get_video_info = get_video_info
    try:
        import lerobot.datasets.dataset_metadata as dataset_metadata

        dataset_metadata.get_video_info = get_video_info
    except ImportError:
        pass

    original_concat = video_utils.concatenate_video_files
    if getattr(original_concat, "_rio_av13_patch", False):
        return

    def concatenate_video_files(input_video_paths, output_video_path, overwrite: bool = True):
        try:
            return original_concat(input_video_paths, output_video_path, overwrite=overwrite)
        except AttributeError:
            output_video_path = Path(output_video_path)
            if output_video_path.exists() and not overwrite:
                return
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp:
                tmp.write("ffconcat version 1.0\n")
                for input_path in input_video_paths:
                    tmp.write(f"file '{Path(input_path).resolve()}'\n")
                list_path = tmp.name
            tmp_out = output_video_path.with_suffix(".concat-tmp.mp4")
            try:
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        list_path,
                        "-c",
                        "copy",
                        "-movflags",
                        "faststart",
                        str(tmp_out),
                    ],
                    check=True,
                    capture_output=True,
                )
                shutil.move(str(tmp_out), str(output_video_path))
            finally:
                Path(list_path).unlink(missing_ok=True)
                tmp_out.unlink(missing_ok=True)

    concatenate_video_files._rio_av13_patch = True
    video_utils.concatenate_video_files = concatenate_video_files
    lerobot_dataset.concatenate_video_files = concatenate_video_files


CAM1_KEY = "observation/cameras/camera_1/rgb"
CAM2_KEY = "observation/cameras/camera_2/rgb"
JOINTS_KEY = "observation/proprio_joints"
GRIPPER_KEY = "observation/gripper_position"
ACTION_KEY = "action"
INSTRUCTION_KEY = "instruction"

REQUIRED_KEYS = (CAM1_KEY, CAM2_KEY, JOINTS_KEY, GRIPPER_KEY, ACTION_KEY)

STATE_DIM = 8
ACTION_DIM = 7
JOINT_ACTION_DIM = 8
STATE_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
JOINT_ACTION_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
TCP_DATASET_DIR_NAMES = frozenset({"kassow_pi05", "kassow_pi05_alignment"})

TRAJ_NAME_RE = re.compile(r"traj_(\d+)\.vla$", re.IGNORECASE)
RANGE_RE = re.compile(r"^(\d+)(?:\s*-\s*(\d+))?$")


@dataclass
class ConvertPi05Cfg:
    """CLI/config for Kassow ``.vla`` → LeRobot v3 conversion."""

    input: Path
    output: Path | None = None
    tasks: Path | None = None
    repo_id: str = "local/kassow_pi05"
    # ``tcp``: action = xyz+rotvec+gripper (7). ``joints``: next-frame q + gripper (8).
    action_space: str = "tcp"
    src_fps: int = 100
    dst_fps: int = 20
    robot_type: str = "kassow"
    binarize_gripper: bool = True
    gripper_close: float = 0.0
    gripper_open: float = 0.7
    gripper_threshold: float | None = None
    min_frames: int = 16
    unwrap_rotvec: bool = True
    trim_idle: bool = True
    idle_xyz_m: float = 0.01
    clean: bool = False
    dry_run: bool = False
    write_tasks_template: Path | None = None
    allow_missing_tasks: bool = False
    missing_task: str = "unlabeled"
    vcodec: str = "h264"
    image_writer_threads: int = 4
    # Parallel episode encodes. Same codec settings as workers=1; only the schedule changes.
    workers: int = 1

    def __post_init__(self):
        space = str(self.action_space).strip().lower()
        if space not in {"tcp", "joints"}:
            raise ValueError(f"action_space must be 'tcp' or 'joints', got {self.action_space!r}")
        self.action_space = space
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")


def list_vla_files(path: Path) -> list[Path]:
    """Return sorted ``.vla`` files from a file or directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".vla":
            raise ValueError(f"Expected a .vla file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    files = sorted(path.glob("*.vla"))
    if not files:
        raise FileNotFoundError(f"No .vla files in {path}")
    return files


def traj_index(path: Path) -> int:
    """Parse ``traj_0012.vla`` → 12. Fallback: stem digits or -1."""
    match = TRAJ_NAME_RE.search(path.name)
    if match:
        return int(match.group(1))
    digits = re.search(r"(\d+)$", path.stem)
    if digits:
        return int(digits.group(1))
    raise ValueError(f"Cannot parse trajectory index from {path.name}")


def subsample_step(src_fps: int, dst_fps: int) -> int:
    """Integer stride from teleop Hz to dataset Hz."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be positive")
    if src_fps < dst_fps:
        raise ValueError(f"src_fps ({src_fps}) must be >= dst_fps ({dst_fps})")
    step = src_fps // dst_fps
    if src_fps % dst_fps:
        logger.warning("src_fps {} is not a multiple of dst_fps {}; using stride {}", src_fps, dst_fps, step)
    return max(step, 1)


def subsample_dict(data: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
    """Keep every ``step``-th frame, aligned across keys."""
    lengths = [len(v) for v in data.values() if hasattr(v, "__len__")]
    if not lengths:
        raise ValueError("Trajectory has no array features")
    n = min(lengths)
    if len(set(lengths)) > 1:
        logger.warning("Feature lengths differ {}; truncating to {}", lengths, n)
    idx = np.arange(0, n, step)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[0] < n:
            raise ValueError(f"{key} has {arr.shape[0]} frames, expected at least {n}")
        out[key] = arr[idx]
    return out


def unwrap_rotvec_sequence(rotvecs: np.ndarray) -> np.ndarray:
    """Pick the 2π-equivalent rotvec closest to the previous frame.

    SciPy / teleop store the principal value ``|r| ≤ π``, so a continuous
    yaw near ±π jumps by ~2π in R³. Behavior cloning treats that as a huge
    action; IK does not. The encoding stays axis-angle.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    if len(rotvecs) == 0:
        return rotvecs.astype(np.float32)
    out = np.empty_like(rotvecs)
    out[0] = rotvecs[0]
    two_pi = 2.0 * np.pi
    ks = np.arange(-3, 4, dtype=np.float64)
    for t in range(1, len(rotvecs)):
        prev = out[t - 1]
        current = rotvecs[t]
        norm = np.linalg.norm(current)
        if norm < 1e-8:
            out[t] = current
            continue
        axis = current / norm
        candidates = current[None, :] - (two_pi * ks)[:, None] * axis[None, :]
        out[t] = candidates[int(np.argmin(np.linalg.norm(candidates - prev, axis=1)))]
    return out.astype(np.float32)


def idle_start_index(xyz: np.ndarray, idle_xyz_m: float = 0.01) -> int:
    """First frame whose TCP xyz has moved ``idle_xyz_m`` from the start pose.

    Returns 0 when the episode never leaves the deadband so we do not wipe a
    static trajectory.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) == 0 or idle_xyz_m <= 0:
        return 0
    moved = np.flatnonzero(np.linalg.norm(xyz - xyz[0], axis=1) >= idle_xyz_m)
    if moved.size == 0:
        return 0
    return int(moved[0])


def slice_episode(data: dict[str, np.ndarray], start: int, end: int | None = None) -> dict[str, np.ndarray]:
    """Slice every time-major array. Length-1 extras (e.g. a scalar instruction) stay intact."""
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] <= start:
            out[key] = arr
            continue
        out[key] = arr[start:end]
    return out


def prepare_episode(data: dict[str, np.ndarray], cfg: ConvertPi05Cfg) -> tuple[dict[str, np.ndarray], dict]:
    """Subsample, unwrap TCP rotvec, drop the stationary prefix. Gripper is unchanged here."""
    step = subsample_step(cfg.src_fps, cfg.dst_fps)
    data = {key: np.asarray(value) for key, value in data.items()}
    data = subsample_dict(data, step)
    info = {
        "frames_sub": len(data[ACTION_KEY]),
        "idle_start": 0,
        "unwrap_jump_before": 0.0,
        "unwrap_jump_after": 0.0,
    }
    action = np.array(data[ACTION_KEY], dtype=np.float32, copy=True)
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"action must be (T, >=6), got {action.shape}")

    if cfg.unwrap_rotvec and cfg.action_space != "joints":
        rotvec = action[:, 3:6]
        if len(rotvec) > 1:
            info["unwrap_jump_before"] = float(np.max(np.linalg.norm(np.diff(rotvec, axis=0), axis=1)))
        unwrapped = unwrap_rotvec_sequence(rotvec)
        if len(unwrapped) > 1:
            info["unwrap_jump_after"] = float(np.max(np.linalg.norm(np.diff(unwrapped, axis=0), axis=1)))
        action[:, 3:6] = unwrapped
        data[ACTION_KEY] = action

    if cfg.trim_idle:
        xyz = np.asarray(data[ACTION_KEY], dtype=np.float32)[:, :3]
        start = idle_start_index(xyz, cfg.idle_xyz_m)
        remaining = len(xyz) - start
        if start > 0 and remaining < cfg.min_frames:
            logger.warning(
                "Idle trim would leave {} frames (min {}); keeping the prefix",
                remaining,
                cfg.min_frames,
            )
            start = 0
        if start > 0:
            data = slice_episode(data, start)
            info["idle_start"] = start

    info["frames_dst"] = len(np.asarray(data[ACTION_KEY]))
    return data, info


def joints_as_action(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build 8D joint actions: next measured ``q`` plus the commanded gripper.

    ``action[t, :7] = q[t+1]`` so the label is not the current state (identity).
    The last frame repeats ``q[-1]``. Gripper stays the teleop command at ``t``.
    """
    joints = np.asarray(data[JOINTS_KEY], dtype=np.float32).reshape(-1, 7)
    raw_action = np.asarray(data[ACTION_KEY], dtype=np.float32)
    if raw_action.ndim != 2 or raw_action.shape[0] != len(joints):
        raise ValueError(f"action {raw_action.shape} must be (T, *) aligned with joints {joints.shape}")
    grip = raw_action[:, -1]
    nxt = np.empty_like(joints)
    if len(joints) > 1:
        nxt[:-1] = joints[1:]
        nxt[-1] = joints[-1]
    else:
        nxt[0] = joints[0]
    return np.concatenate([nxt, grip.reshape(-1, 1)], axis=1).astype(np.float32)


def resolve_gripper_threshold(
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> float:
    """Return the open/close decision boundary.

    ``None`` uses the midpoint between ``gripper_close`` and ``gripper_open``.
    """
    if gripper_threshold is not None:
        return float(gripper_threshold)
    return 0.5 * (gripper_close + gripper_open)


def binarize_action_gripper(
    values: np.ndarray,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Snap commanded gripper values to the two teleop targets.

    Values at or above the threshold become ``gripper_open``. Do not low-pass
    the action — π₀.₅ would imitate the smear.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    threshold = resolve_gripper_threshold(gripper_close, gripper_open, gripper_threshold)
    return np.where(values >= threshold, gripper_open, gripper_close).astype(np.float32)


def binarize_state_gripper(
    values: np.ndarray,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
) -> np.ndarray:
    """Hysteresis-snap measured gripper position so the state does not chatter."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    threshold = resolve_gripper_threshold(gripper_close, gripper_open, gripper_threshold)
    span = gripper_open - gripper_close
    low = threshold - 0.2 * span
    high = threshold + 0.15 * span
    state = gripper_open if (values[0] if len(values) else threshold) >= threshold else gripper_close
    out = np.empty_like(values, dtype=np.float32)
    for i, value in enumerate(values):
        if value <= low:
            state = gripper_close
        elif value >= high:
            state = gripper_open
        out[i] = state
    return out


def _parse_range_key(key: str) -> tuple[int, int]:
    match = RANGE_RE.match(str(key).strip())
    if not match:
        raise ValueError(f"Invalid task-map key {key!r}; use '12' or '0-19'")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    if end < start:
        raise ValueError(f"Invalid range {key!r}")
    return start, end


def _strip_task_value(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return text.strip()


def parse_task_map(path: Path) -> dict[int, str]:
    """Load ``{index: instruction}`` from JSON or ``start-end: text`` lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Task JSON must be an object of range → instruction")
        items = payload.items()
    else:
        items = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Task line must be 'range: instruction', got {line!r}")
            key, value = stripped.split(":", 1)
            items.append((key, value))

    mapping: dict[int, str] = {}
    for key, value in items:
        task = _strip_task_value(str(value))
        if not task:
            continue
        start, end = _parse_range_key(str(key))
        for idx in range(start, end + 1):
            mapping[idx] = task
    return mapping


def write_task_template(vla_files: list[Path], path: Path) -> Path:
    """Write a fill-in YAML listing every trajectory index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [traj_index(f) for f in vla_files]
    lo, hi = min(indices), max(indices)
    lines = [
        "# Language labels for Kassow π₀.₅ conversion.",
        "# Index matches traj_XXXX.vla (traj_0000 → 0).",
        "# Fill ranges, then run the converter with --tasks <this file>.",
        "#",
        "# Example:",
        "# 0-19: pick up the blue cylinder",
        "# 20-34: put the cylinder in the bin",
        "",
        f"{lo}-{hi}:",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote task template to {} ({} trajectories, indices {}-{})", path, len(vla_files), lo, hi)
    return path


def _as_float32_vec(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    if vec.shape[0] < dim:
        raise ValueError(f"{name} has dim {vec.shape[0]}, expected {dim}")
    return vec[:dim]


def build_pi05_frame(
    data: dict[str, np.ndarray],
    frame_idx: int,
    task: str,
    *,
    binarize_gripper: bool = True,
    gripper_close: float = 0.0,
    gripper_open: float = 0.7,
    gripper_threshold: float | None = None,
    action_dim: int | None = None,
) -> dict:
    """Map one robodm frame to LeRobot π₀.₅ keys."""
    joints = _as_float32_vec(data[JOINTS_KEY][frame_idx], 7, JOINTS_KEY)
    gripper_obs = np.float32(data[GRIPPER_KEY][frame_idx]).reshape(-1)[0]
    raw_action = np.asarray(data[ACTION_KEY][frame_idx])
    dim = int(action_dim or np.asarray(data[ACTION_KEY]).shape[-1] or ACTION_DIM)
    action = _as_float32_vec(raw_action, dim, ACTION_KEY)
    if binarize_gripper:
        gripper_obs = binarize_state_gripper(
            np.asarray([gripper_obs], dtype=np.float32),
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )[0]
        action[-1] = binarize_action_gripper(
            np.asarray([action[-1]]),
            gripper_close=gripper_close,
            gripper_open=gripper_open,
            gripper_threshold=gripper_threshold,
        )[0]
    cam1 = np.asarray(data[CAM1_KEY][frame_idx])
    cam2 = np.asarray(data[CAM2_KEY][frame_idx])
    if cam1.ndim != 3 or cam1.shape[-1] != 3:
        raise ValueError(f"{CAM1_KEY} expected (H,W,3), got {cam1.shape}")
    if cam2.ndim != 3 or cam2.shape[-1] != 3:
        raise ValueError(f"{CAM2_KEY} expected (H,W,3), got {cam2.shape}")
    return {
        "observation.images.camera_1": cam1,
        "observation.images.camera_2": cam2,
        "observation.state": np.concatenate([joints, np.asarray([gripper_obs], dtype=np.float32)]),
        "action": action,
        "task": task,
    }


def _pi05_features(
    image_shape: tuple[int, int, int],
    *,
    action_dim: int = ACTION_DIM,
    action_names: list[str] | None = None,
) -> dict:
    image_spec = {
        "dtype": "video",
        "shape": image_shape,
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.camera_1": image_spec,
        "observation.images.camera_2": dict(image_spec),
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": list(action_names or ACTION_NAMES),
        },
    }


def _validate_keys(data: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Trajectory missing required keys {missing}; have {sorted(data)}")


def _resolve_task(
    idx: int,
    data: dict[str, np.ndarray],
    task_map: dict[int, str],
    *,
    allow_missing_tasks: bool,
    missing_task: str,
) -> str | None:
    if idx in task_map:
        return task_map[idx]
    if INSTRUCTION_KEY in data:
        raw = data[INSTRUCTION_KEY][0]
        text = str(raw).strip() if raw is not None else ""
        if text:
            return text
    if allow_missing_tasks:
        return missing_task
    return None


def _create_lerobot_dataset(cfg: ConvertPi05Cfg, output: Path, cam_shape: tuple[int, int, int]):
    """Create a LeRobot dataset on 0.4.x (`vcodec=`) and 0.6+ (`rgb_encoder=`)."""
    kwargs: dict = {
        "repo_id": cfg.repo_id,
        "fps": cfg.dst_fps,
        "robot_type": cfg.robot_type,
        "features": _pi05_features(
            cam_shape,
            action_dim=JOINT_ACTION_DIM if cfg.action_space == "joints" else ACTION_DIM,
            action_names=JOINT_ACTION_NAMES if cfg.action_space == "joints" else ACTION_NAMES,
        ),
        "root": str(output),
        "use_videos": True,
        "image_writer_threads": cfg.image_writer_threads,
        "image_writer_processes": 0,
    }
    params = inspect.signature(LeRobotDataset.create).parameters
    if "video_backend" in params:
        kwargs["video_backend"] = "pyav"
    if "vcodec" in params:
        kwargs["vcodec"] = cfg.vcodec
    if "rgb_encoder" in params:
        from lerobot.configs.video import RGBEncoderConfig

        kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=cfg.vcodec, video_backend="pyav")
    if "video_files_size_in_mb" in params:
        kwargs["video_files_size_in_mb"] = 1
    dataset = LeRobotDataset.create(**kwargs)
    # One MP4 per episode so we never hit PyAV concat (`add_stream_from_template`).
    if "video_files_size_in_mb" not in params and hasattr(dataset.meta, "update_chunk_settings"):
        dataset.meta.update_chunk_settings(video_files_size_in_mb=1)
    return dataset


def _convert_shard(cfg: ConvertPi05Cfg, files: list[str], shard_in: str, shard_out: str) -> str:
    """Convert one contiguous slice of ``.vla`` files into its own dataset directory."""
    src_dir = Path(shard_in)
    src_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        path = Path(src)
        link = src_dir / path.name
        if not link.exists():
            link.symlink_to(path.resolve())
    shard_cfg = replace(cfg, input=src_dir, output=Path(shard_out), clean=True, workers=1)
    convert(shard_cfg)
    return shard_out


def _list_parquet(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _shift_list_column(table, name: str, offset: float):
    """Add ``offset`` to every element of a ``list<number>`` column."""
    import pyarrow as pa
    import pyarrow.compute as pc

    column = table.column(name)
    lengths = pc.list_value_length(column).to_numpy()
    offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    values = pc.add(pc.list_flatten(column), offset).combine_chunks()
    shifted = pa.ListArray.from_arrays(pa.array(offsets), values)
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, shifted)


def _set_list_column_from_ints(table, name: str, values: np.ndarray):
    """Replace a ``list<double>`` column with one float per row."""
    import pyarrow as pa

    flat = pa.array(np.asarray(values, dtype=np.float64))
    offsets = pa.array(np.arange(len(values) + 1, dtype=np.int32))
    column = pa.ListArray.from_arrays(offsets, flat)
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, column)


def _merge_lerobot_shards(shard_dirs: list[Path], output: Path) -> dict:
    """Concatenate per-worker datasets without re-encoding video.

    Episode order follows ``shard_dirs``. Video files stay one MP4 per episode.
    Numeric stats are combined with LeRobot's ``aggregate_stats``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    info = json.loads((shard_dirs[0] / "meta" / "info.json").read_text())
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    index_shift_cols = [
        f"stats/index/{stat}" for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    ]
    episode_stat_cols = [
        f"stats/episode_index/{stat}" for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    ]

    data_parts = []
    episode_parts = []
    stats_parts = []
    episode_offset = 0
    frame_offset = 0
    for shard in shard_dirs:
        shard_info = json.loads((shard / "meta" / "info.json").read_text())
        n_episodes = int(shard_info["total_episodes"])
        n_frames = int(shard_info["total_frames"])
        if n_episodes == 0:
            continue

        data = pa.concat_tables([pq.read_table(path) for path in _list_parquet(shard / "data")])
        data = data.set_column(
            data.schema.get_field_index("episode_index"),
            "episode_index",
            pc.add(data.column("episode_index"), episode_offset),
        )
        data = data.set_column(
            data.schema.get_field_index("index"),
            "index",
            pc.add(data.column("index"), frame_offset),
        )
        data_parts.append(data)

        episodes = pq.read_table(_list_parquet(shard / "meta" / "episodes")[0])
        new_episode_index = pc.add(episodes.column("episode_index"), episode_offset)
        episodes = episodes.set_column(
            episodes.schema.get_field_index("episode_index"),
            "episode_index",
            new_episode_index,
        )
        episodes = episodes.set_column(
            episodes.schema.get_field_index("dataset_from_index"),
            "dataset_from_index",
            pc.add(episodes.column("dataset_from_index"), frame_offset),
        )
        episodes = episodes.set_column(
            episodes.schema.get_field_index("dataset_to_index"),
            "dataset_to_index",
            pc.add(episodes.column("dataset_to_index"), frame_offset),
        )
        for key in video_keys:
            column_name = f"videos/{key}/file_index"
            episodes = episodes.set_column(
                episodes.schema.get_field_index(column_name),
                column_name,
                new_episode_index,
            )
        for column_name in index_shift_cols:
            episodes = _shift_list_column(episodes, column_name, float(frame_offset))
        for column_name in episode_stat_cols:
            episodes = _set_list_column_from_ints(episodes, column_name, new_episode_index.to_numpy())
        episode_parts.append(episodes)

        for key in video_keys:
            for row in range(episodes.num_rows):
                # file_index was rewritten; the source name is the pre-merge local index.
                local_index = int(row)
                global_index = episode_offset + local_index
                src = shard / "videos" / key / "chunk-000" / f"file-{local_index:03d}.mp4"
                dst = output / "videos" / key / "chunk-000" / f"file-{global_index:03d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not src.is_file():
                    raise FileNotFoundError(src)
                os_link = dst
                shutil.copy2(src, os_link)

        loaded = load_stats(shard)
        if loaded:
            stats_parts.append(loaded)
        episode_offset += n_episodes
        frame_offset += n_frames

    if not data_parts:
        raise RuntimeError("All shards were empty")

    data_dir = output / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(data_parts), data_dir / "file-000.parquet")
    episodes_dir = output / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(episode_parts), episodes_dir / "file-000.parquet")
    shutil.copy2(shard_dirs[0] / "meta" / "tasks.parquet", output / "meta" / "tasks.parquet")

    info["total_episodes"] = episode_offset
    info["total_frames"] = frame_offset
    info["splits"] = {"train": f"0:{episode_offset}"}
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    if stats_parts:
        write_stats(aggregate_stats(stats_parts), output)
    return {"episodes": episode_offset, "frames": frame_offset}


def _convert_parallel(cfg: ConvertPi05Cfg, vla_files: list[Path]) -> dict:
    """Encode episodes on several processes, then concatenate the shards.

    Codec, fps, gripper, and idle-trim settings are unchanged. Workers only
    split the file list.
    """
    workers = min(int(cfg.workers), len(vla_files))
    chunks = [chunk.tolist() for chunk in np.array_split(np.array(vla_files, dtype=object), workers) if len(chunk)]
    logger.info("Converting {} episodes on {} workers", len(vla_files), len(chunks))
    shard_root = Path(tempfile.mkdtemp(prefix=f"{Path(cfg.output).name}.", dir=Path(cfg.output).parent))
    try:
        payloads = []
        for i, chunk in enumerate(chunks):
            payloads.append(
                (
                    replace(cfg, workers=1),
                    [str(path) for path in chunk],
                    str(shard_root / f"in_{i}"),
                    str(shard_root / f"out_{i}"),
                )
            )
        # LeRobot encodes each episode in a child process, so these workers must
        # not be daemons (a Pool worker cannot spawn its own children).
        ctx = mp.get_context("spawn")
        procs = []
        for payload in payloads:
            proc = ctx.Process(target=_convert_shard, args=payload)
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
        failed = [proc.exitcode for proc in procs if proc.exitcode]
        if failed:
            raise RuntimeError(f"Shard conversion failed, exit codes {failed}")
        shard_dirs = [Path(payload[3]) for payload in payloads]
        summary = _merge_lerobot_shards(shard_dirs, Path(cfg.output))
    finally:
        shutil.rmtree(shard_root, ignore_errors=True)
    logger.info(
        "Wrote {} episodes ({} frames) to {} with {} workers",
        summary["episodes"],
        summary["frames"],
        cfg.output,
        len(chunks),
    )
    return {
        "output": str(cfg.output),
        "kept": summary["episodes"],
        "skipped": len(vla_files) - summary["episodes"],
        "n_files": len(vla_files),
        "workers": len(chunks),
    }


def convert(cfg: ConvertPi05Cfg) -> dict:
    """Convert ``.vla`` files to a local LeRobot v3 dataset.

    Returns:
        Summary dict with episode counts. Does not upload to the Hub.
    """
    vla_files = list_vla_files(cfg.input)
    if cfg.write_tasks_template is not None:
        write_task_template(vla_files, cfg.write_tasks_template)
        return {"template": str(cfg.write_tasks_template), "n_files": len(vla_files)}

    task_map: dict[int, str] = {}
    if cfg.tasks is not None:
        task_map = parse_task_map(cfg.tasks)
        logger.info("Loaded {} task labels from {}", len(task_map), cfg.tasks)

    unlabeled = [traj_index(path) for path in vla_files if traj_index(path) not in task_map]
    if unlabeled and not cfg.allow_missing_tasks and not cfg.dry_run:
        template = Path(cfg.input) / "tasks.yaml" if Path(cfg.input).is_dir() else Path(cfg.input).with_name("tasks.yaml")
        if not template.exists():
            write_task_template(vla_files, template)
        raise ValueError(
            f"{len(unlabeled)} trajectories have no language label (e.g. {unlabeled[:5]}). "
            f"Fill {template} then re-run with --tasks {template}"
        )

    step = subsample_step(cfg.src_fps, cfg.dst_fps)
    first = robodm.Trajectory(path=str(vla_files[0]), mode="r").load()
    _validate_keys(first)
    cam_shape = tuple(int(x) for x in np.asarray(first[CAM1_KEY][0]).shape)
    n_first = len(np.asarray(first[ACTION_KEY]))
    logger.info(
        "Probe {}: {} frames @ ~{} Hz → stride {} → ~{} frames, image {}",
        vla_files[0].name,
        n_first,
        cfg.src_fps,
        step,
        (n_first + step - 1) // step,
        cam_shape,
    )
    del first

    if cfg.dry_run:
        rows = []
        for path in vla_files:
            idx = traj_index(path)
            data = robodm.Trajectory(path=str(path), mode="r").load()
            _validate_keys(data)
            n = len(np.asarray(data[ACTION_KEY]))
            prepared, info = prepare_episode(data, cfg)
            task = _resolve_task(
                idx,
                prepared,
                task_map,
                allow_missing_tasks=cfg.allow_missing_tasks,
                missing_task=cfg.missing_task,
            )
            action_g = np.asarray(prepared[ACTION_KEY], dtype=np.float32)[:, -1]
            rows.append(
                {
                    "file": path.name,
                    "index": idx,
                    "frames_src": n,
                    "frames_dst": info["frames_dst"],
                    "idle_start": info["idle_start"],
                    "unwrap_jump_before": info["unwrap_jump_before"],
                    "unwrap_jump_after": info["unwrap_jump_after"],
                    "task": task,
                    "action_gripper_unique": np.unique(np.round(action_g, 3)).tolist(),
                }
            )
            del data
        n_with_task = sum(1 for row in rows if row["task"])
        logger.info("Dry-run: {} files, {} with a task", len(rows), n_with_task)
        return {"dry_run": True, "episodes": rows}

    if cfg.output is None:
        raise ValueError("output is required unless --dry-run or --write-tasks-template")
    output = Path(cfg.output)
    if cfg.action_space == "joints" and output.name in TCP_DATASET_DIR_NAMES:
        raise ValueError(
            f"Refusing to write joint actions over the TCP dataset {output}. "
            "Use --output data/lerobot/kassow_pi05_joints --repo-id local/kassow_pi05_joints"
        )
    if output.exists():
        if not cfg.clean:
            raise FileExistsError(f"{output} already exists; pass --clean to overwrite")
        shutil.rmtree(output)

    if LeRobotDataset is None:
        raise ImportError(
            "LeRobot is required. Use Python 3.12: uv venv .venv-pi --python 3.12 && "
            "uv pip install --python .venv-pi 'lerobot[pi,training]==0.6.1'"
        ) from _LEROBOT_IMPORT_ERROR

    _patch_lerobot_pyav()
    if cfg.workers > 1:
        return _convert_parallel(cfg, vla_files)
    dataset = _create_lerobot_dataset(cfg, output, cam_shape)

    kept = 0
    skipped = 0
    for path in vla_files:
        idx = traj_index(path)
        logger.info("Episode index {} ({})", idx, path.name)
        data = robodm.Trajectory(path=str(path), mode="r").load()
        try:
            _validate_keys(data)
            data, info = prepare_episode(data, cfg)
            if cfg.action_space == "joints":
                data[ACTION_KEY] = joints_as_action(data)
                info["frames_dst"] = len(data[ACTION_KEY])
            n = info["frames_dst"]
            if info["idle_start"] > 0:
                logger.info("{}: dropped {} idle prefix frames", path.name, info["idle_start"])
            if info["unwrap_jump_before"] > np.pi:
                logger.info(
                    "{}: unwrapped rotvec max |Δr| {:.3f} → {:.3f}",
                    path.name,
                    info["unwrap_jump_before"],
                    info["unwrap_jump_after"],
                )
            if n < cfg.min_frames:
                logger.warning("Skip {}: {} frames after prepare (min {})", path.name, n, cfg.min_frames)
                skipped += 1
                continue
            task = _resolve_task(
                idx,
                data,
                task_map,
                allow_missing_tasks=cfg.allow_missing_tasks,
                missing_task=cfg.missing_task,
            )
            if not task:
                logger.warning("Skip {}: no task label", path.name)
                skipped += 1
                continue

            action_g = data[ACTION_KEY][:, -1].astype(np.float32)
            if cfg.binarize_gripper:
                snapped = binarize_action_gripper(
                    action_g,
                    cfg.gripper_close,
                    cfg.gripper_open,
                    cfg.gripper_threshold,
                )
                n_flip = int(np.sum(np.abs(action_g - snapped) > 1e-3))
                if n_flip:
                    logger.info(
                        "{}: snapped {}/{} action gripper values to {{{}, {}}} (threshold {})",
                        path.name,
                        n_flip,
                        n,
                        cfg.gripper_close,
                        cfg.gripper_open,
                        resolve_gripper_threshold(cfg.gripper_close, cfg.gripper_open, cfg.gripper_threshold),
                    )
                state_g = binarize_state_gripper(
                    data[GRIPPER_KEY],
                    cfg.gripper_close,
                    cfg.gripper_open,
                    cfg.gripper_threshold,
                )
            else:
                snapped = action_g
                state_g = np.asarray(data[GRIPPER_KEY], dtype=np.float32).reshape(-1)
            for frame_idx in range(n):
                frame = build_pi05_frame(
                    data,
                    frame_idx,
                    task,
                    binarize_gripper=False,
                )
                # Sequence-level gripper (raw or hysteresis), not per-frame restart.
                frame["observation.state"][-1] = state_g[frame_idx]
                frame["action"][-1] = snapped[frame_idx]
                dataset.add_frame(frame)
            dataset.save_episode()
            kept += 1
        finally:
            del data

    logger.info("Wrote {} episodes ({} skipped) to {}", kept, skipped, output)
    return {"output": str(output), "kept": kept, "skipped": skipped, "n_files": len(vla_files)}
