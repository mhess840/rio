# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The conversion run: labels, dry run, serial or sharded encoding, and per-segment writes."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import robodm
from loguru import logger

from . import lerobot_io
from .config import ConvertPi05Cfg
from .episode import (
    binarize_action_gripper,
    binarize_state_gripper,
    build_pi05_frame,
    episode_ranges,
    joints_as_action,
    prepare_episode,
    slice_episode,
    subsample_step,
)
from .schema import (
    ACTION_KEY,
    CAM1_KEY,
    GRIPPER_COMMANDED_KEY,
    GRIPPER_KEY,
    TCP_DATASET_DIR_NAMES,
    validate_keys,
)
from .tasks import list_vla_files, parse_task_map, resolve_task, traj_index, write_task_template


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


def _convert_parallel(cfg: ConvertPi05Cfg, vla_files: list[Path]) -> dict:
    """Encode episodes on several processes, then concatenate the shards.

    Codec, fps, gripper, and idle-trim settings are unchanged. Workers only
    split the file list.
    """
    if cfg.output is None:
        raise ValueError("output is required")
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
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
        summary = lerobot_io.merge_lerobot_datasets(shard_dirs, Path(cfg.output), cfg.repo_id, clean=cfg.clean)
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
        "skipped": len(vla_files) - summary["episodes"] if cfg.segments == "whole" else None,
        "n_files": len(vla_files),
        "workers": len(chunks),
    }


def convert(cfg: ConvertPi05Cfg) -> dict:
    """Convert ``.vla`` files to a local LeRobot v3 dataset.

    Returns:
        Summary dict with episode counts. Does not upload to the Hub.
    """
    if cfg.input is None:
        raise ValueError("input is required: a .vla file or a folder of them")
    vla_files = list_vla_files(cfg.input)
    if cfg.write_tasks_template is not None:
        write_task_template(vla_files, cfg.write_tasks_template)
        return {"template": str(cfg.write_tasks_template), "n_files": len(vla_files)}

    task_map: dict[int, str] = {}
    if cfg.tasks is not None:
        task_map = parse_task_map(cfg.tasks)
        logger.info("Loaded {} task labels from {}", len(task_map), cfg.tasks)

    unlabeled = [traj_index(path) for path in vla_files if traj_index(path) not in task_map]
    if unlabeled and not cfg.allow_missing_tasks and not cfg.dry_run and cfg.segments == "whole":
        template = Path(cfg.input) / "tasks.yaml" if Path(cfg.input).is_dir() else Path(cfg.input).with_name("tasks.yaml")
        if not template.exists():
            write_task_template(vla_files, template)
        raise ValueError(
            f"{len(unlabeled)} trajectories have no language label (e.g. {unlabeled[:5]}). "
            f"Fill {template} then re-run with --tasks {template}"
        )

    step = subsample_step(cfg.src_fps, cfg.dst_fps)
    first = robodm.Trajectory(path=str(vla_files[0]), mode="r").load()
    validate_keys(first)
    first_image = np.asarray(first[CAM1_KEY][0])
    if first_image.ndim != 3 or first_image.shape[2] != 3:
        raise ValueError("Policy camera frames must be H x W x 3 RGB images")
    cam_shape = (int(first_image.shape[0]), int(first_image.shape[1]), int(first_image.shape[2]))
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
            validate_keys(data)
            n = len(np.asarray(data[ACTION_KEY]))
            prepared, info = prepare_episode(data, cfg)
            task = resolve_task(
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
            if cfg.segments != "whole":
                rows[-1]["segments"] = episode_ranges(prepared, cfg, path.name)
            del data
        n_with_task = sum(1 for row in rows if row["task"])
        logger.info("Dry-run: {} files, {} with a task", len(rows), n_with_task)
        return {"dry_run": True, "episodes": rows}

    if cfg.output is None:
        raise ValueError("output is required unless --dry-run or --write-tasks-template")
    output = Path(cfg.output)
    source = Path(cfg.input).resolve()
    resolved_output = output.resolve()
    if (
        output.is_symlink()
        or source == resolved_output
        or resolved_output in source.parents
        or source in resolved_output.parents
    ):
        raise ValueError("Input and output must be separate, non-overlapping paths")
    if cfg.action_space == "joints" and output.name in TCP_DATASET_DIR_NAMES:
        raise ValueError(
            f"Refusing to write joint actions over the TCP dataset {output}. "
            "Use --output data/lerobot/kassow_pi05_joints --repo-id local/kassow_pi05_joints"
        )
    if output.exists():
        if not cfg.clean:
            raise FileExistsError(f"{output} already exists; pass --clean to overwrite")

    if lerobot_io.LeRobotDataset is None:
        raise ImportError(
            "LeRobot is required. Use Python 3.12: uv venv .venv-pi --python 3.12 && "
            "uv pip install --python .venv-pi 'lerobot[pi,training]==0.6.1'"
        ) from lerobot_io.LEROBOT_IMPORT_ERROR

    lerobot_io.patch_lerobot_pyav()
    if cfg.workers > 1:
        return _convert_parallel(cfg, vla_files)
    if output.exists():
        shutil.rmtree(output)
    dataset = lerobot_io.create_lerobot_dataset(cfg, output, cam_shape)

    kept = 0
    skipped = 0
    for path in vla_files:
        idx = traj_index(path)
        logger.info("Episode index {} ({})", idx, path.name)
        data = robodm.Trajectory(path=str(path), mode="r").load()
        try:
            validate_keys(data)
            data, info = prepare_episode(data, cfg)
            if info["idle_start"] > 0:
                logger.info("{}: dropped {} idle prefix frames", path.name, info["idle_start"])
            if info["unwrap_jump_before"] > np.pi:
                logger.info(
                    "{}: unwrapped rotvec max |Δr| {:.3f} → {:.3f}",
                    path.name,
                    info["unwrap_jump_before"],
                    info["unwrap_jump_after"],
                )
            ranges = episode_ranges(data, cfg, path.name)
            if not ranges:
                skipped += 1
                continue
            task = resolve_task(
                idx,
                data,
                task_map,
                allow_missing_tasks=cfg.allow_missing_tasks,
                missing_task=cfg.missing_task,
            )
            if not task:
                raise ValueError(f"{path.name} has no task label")
            for start, end in ranges:
                episode = slice_episode(data, start, end)
                if cfg.action_space == "joints":
                    episode[ACTION_KEY] = joints_as_action(episode)
                if len(episode[ACTION_KEY]) < cfg.min_frames:
                    skipped += 1
                    continue
                _save_training_episode(dataset, episode, cfg, task)
                kept += 1
        finally:
            del data

    logger.info("Wrote {} episodes ({} skipped) to {}", kept, skipped, output)
    return {"output": str(output), "kept": kept, "skipped": skipped, "n_files": len(vla_files)}


def _save_training_episode(dataset, data: dict[str, np.ndarray], cfg: ConvertPi05Cfg, task: str) -> None:
    """Write one complete segment using the shared state and action encodings."""
    action_gripper = np.asarray(data[ACTION_KEY], dtype=np.float32)[:, -1]
    state_values = (
        data.get(GRIPPER_COMMANDED_KEY, action_gripper) if cfg.gripper_state_source == "commanded" else data[GRIPPER_KEY]
    )
    if cfg.binarize_gripper:
        action_gripper = binarize_action_gripper(action_gripper, cfg.gripper_close, cfg.gripper_open, cfg.gripper_threshold)
        state_gripper = binarize_state_gripper(state_values, cfg.gripper_close, cfg.gripper_open, cfg.gripper_threshold)
    else:
        state_gripper = np.asarray(state_values, dtype=np.float32).reshape(-1)
    for frame_index in range(len(data[ACTION_KEY])):
        frame = build_pi05_frame(data, frame_index, task, binarize_gripper=False)
        frame["observation.state"][-1] = state_gripper[frame_index]
        frame["action"][-1] = action_gripper[frame_index]
        dataset.add_frame(frame)
    dataset.save_episode()
