# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""LeRobot side of the conversion: version shims, dataset creation, and shard merging."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import imageio_ffmpeg

from .config import ConvertPi05Cfg
from .schema import ACTION_DIM, ACTION_NAMES, JOINT_ACTION_DIM, JOINT_ACTION_NAMES, STATE_DIM, STATE_NAMES

try:
    import lerobot.datasets.video_utils as video_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_IMPORT_ERROR = None
except ImportError as e:
    video_utils = None
    LeRobotDataset = None
    LEROBOT_IMPORT_ERROR = e

# LeRobot 0.6 modules that bind the video helpers by name at import time.
_VIDEO_HELPER_USERS = (
    "lerobot.datasets.lerobot_dataset",
    "lerobot.datasets.dataset_metadata",
    "lerobot.datasets.dataset_writer",
    "lerobot.datasets.aggregate",
)


def _rebind(name: str, original, replacement) -> None:
    """Point every loaded ``lerobot`` module that holds ``original`` as ``name`` at ``replacement``.

    Matching by identity leaves unrelated attributes (and test fakes) alone.
    """
    for module_name in _VIDEO_HELPER_USERS:
        try:
            __import__(module_name)
        except ImportError:
            continue
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("lerobot.") and module is not None and getattr(module, name, None) is original:
            setattr(module, name, replacement)


def patch_lerobot_pyav() -> None:
    """Rio pins ``av<=13``; LeRobot 0.6 reads ``Codec.canonical_name`` and uses ``add_stream_from_template`` from newer PyAV."""
    if video_utils is None:
        return
    video_module = video_utils
    if getattr(video_module.get_video_info, "_rio_av13_patch", False):
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
            channels_fn = getattr(video_module, "get_pix_fmt_channels", None) or getattr(
                video_module, "get_video_pixel_channels", lambda _fmt: 3
            )
            info["video.channels"] = channels_fn(stream.pix_fmt)
        info.update(**video_module.get_audio_info(video_path))
        if video_encoder is not None:
            for field_name, field_value in asdict(video_encoder).items():
                if field_name == "vcodec":
                    continue
                info.setdefault(f"video.{field_name}", field_value)
        info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)
        return info

    get_video_info.__dict__["_rio_av13_patch"] = True
    original_info = video_module.get_video_info
    video_module.get_video_info = get_video_info
    _rebind("get_video_info", original_info, get_video_info)

    original_concat = video_module.concatenate_video_files
    if getattr(original_concat, "_rio_av13_patch", False):
        return

    def concatenate_video_files(input_video_paths, output_video_path, overwrite: bool = True, **kwargs):
        try:
            return original_concat(input_video_paths, output_video_path, overwrite=overwrite, **kwargs)
        except AttributeError:
            output_video_path = Path(output_video_path)
            if output_video_path.exists() and not overwrite:
                return
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".rio_concat_", dir=output_video_path.parent) as work:
                lines = ["ffconcat version 1.0"]
                for input_path in input_video_paths:
                    if "\n" in str(input_path) or "\r" in str(input_path):
                        raise ValueError("Video paths must not contain line breaks") from None
                    resolved = str(Path(input_path).resolve()).replace("\\", "/")
                    if "\n" in resolved or "\r" in resolved:
                        raise ValueError("Video paths must not contain line breaks") from None
                    escaped = resolved.replace("'", "'\\''")
                    lines.append(f"file '{escaped}'")
                list_path = Path(work) / "inputs.ffconcat"
                list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                staged_output = Path(work) / "output.mp4"
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-i",
                        str(list_path),
                        "-c",
                        "copy",
                        "-movflags",
                        "faststart",
                        str(staged_output),
                    ],
                    check=True,
                    capture_output=True,
                )
                staged_output.replace(output_video_path)

    concatenate_video_files.__dict__["_rio_av13_patch"] = True
    video_module.concatenate_video_files = concatenate_video_files
    _rebind("concatenate_video_files", original_concat, concatenate_video_files)


def pi05_features(
    image_shape: tuple[int, int, int],
    *,
    action_dim: int = ACTION_DIM,
    action_names: list[str] | None = None,
) -> dict:
    """LeRobot feature spec of the two cameras, the 8D state, and the action."""
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


def create_lerobot_dataset(cfg: ConvertPi05Cfg, output: Path, cam_shape: tuple[int, int, int]):
    """Create a LeRobot dataset on 0.4.x (`vcodec=`) and 0.6+ (`rgb_encoder=`)."""
    if LeRobotDataset is None:
        raise ImportError("LeRobot dataset dependencies are unavailable") from LEROBOT_IMPORT_ERROR
    kwargs: dict = {
        "repo_id": cfg.repo_id,
        "fps": cfg.dst_fps,
        "robot_type": cfg.robot_type,
        "features": pi05_features(
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


def merge_lerobot_datasets(shard_dirs: list[Path], output: Path, repo_id: str, *, clean: bool = False) -> dict:
    """Aggregate datasets with shared task IDs and publish only a completed result."""
    output = Path(output)
    resolved_output = output.resolve()
    for source in shard_dirs:
        resolved_source = Path(source).resolve()
        if (
            resolved_source == resolved_output
            or resolved_output in resolved_source.parents
            or resolved_source in resolved_output.parents
        ):
            raise ValueError(f"{output} overlaps one of the merge sources")
    if output.is_symlink():
        raise ValueError("Refusing to replace a linked dataset")
    if output.exists() and not clean:
        raise FileExistsError(f"{output} already exists; pass --clean to replace it")
    shards = [
        Path(source)
        for source in shard_dirs
        if int(json.loads((Path(source) / "meta" / "info.json").read_text())["total_episodes"]) > 0
    ]
    if not shards:
        raise RuntimeError("All shards were empty")
    from lerobot.datasets.aggregate import aggregate_datasets

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_merge_", dir=output.parent) as work:
        staging = Path(work) / "dataset"
        aggregate_datasets(
            repo_ids=[f"{repo_id}_shard{index}" for index in range(len(shards))],
            aggr_repo_id=repo_id,
            roots=shards,
            aggr_root=staging,
            concatenate_videos=False,
        )
        info = json.loads((staging / "meta" / "info.json").read_text())
        summary = {"episodes": int(info["total_episodes"]), "frames": int(info["total_frames"])}
        backup = Path(work) / "previous"
        if output.exists():
            if not clean:
                raise FileExistsError(output)
            output.replace(backup)
        try:
            staging.replace(output)
        except OSError:
            if backup.exists() and not output.exists():
                backup.replace(output)
            raise
    return summary
