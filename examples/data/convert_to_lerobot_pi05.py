"""Command-line entry point for the Kassow LeRobot converter.

Experiment mode reads ``experiments/<name>/raw`` and writes
``experiments/<name>/lerobot/<variant>``; ``<name>`` is the active experiment
unless ``--experiment`` is given::

    uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \\
        --variant spaces_notrim --tasks tasks_spaces.yaml \\
        --no-trim-idle --gripper-state-source commanded

``--tasks`` may name a label file inside ``raw/``; without it ``raw/tasks.yaml``
is used. The settings are recorded in ``experiment.json``. An existing variant
is never replaced unless ``--clean`` is passed, and a linked (pre-existing)
dataset is never deleted. Plain ``--input``/``--output`` work as before.

DAgger: an existing variant with the original demos plus the operator segments
of the listed rounds (``experiments/<name>/dagger/r<N>/``); the base is only read::

    uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \\
        --variant dagger_r1 --base-dataset spaces_notrim --dagger-rounds 1 \\
        --no-trim-idle --gripper-state-source commanded
"""

import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from loguru import logger

from examples.cfg.kassow_experiments import (
    active_experiment_name,
    dataset_repo_id,
    load_state,
    resolve_experiment,
    save_state,
)
from rio.data.convert_pi05 import ConvertPi05Cfg, convert, merge_lerobot_datasets

DEFAULT_REPO_ID = ConvertPi05Cfg.repo_id
# Settings recorded per variant that must agree between a base dataset and its DAgger extension.
SHARED_SETTINGS = ("action_space", "gripper_state_source", "binarize_gripper", "gripper_open", "gripper_close")


@dataclass
class ConvertExperimentCfg(ConvertPi05Cfg):
    """``ConvertPi05Cfg`` plus experiment paths."""

    # Empty: the active experiment. Only used together with --variant.
    experiment: str = ""
    # LeRobot dataset name under experiments/<name>/lerobot/, e.g. spaces_notrim.
    variant: str = ""
    # DAgger rounds to append to --base-dataset, e.g. --dagger-rounds 1 2.
    dagger_rounds: tuple[int, ...] = ()
    # Existing variant with the original demos; it is copied, never changed.
    base_dataset: str = ""


def resolve_experiment_paths(cfg: ConvertExperimentCfg) -> ConvertExperimentCfg:
    """Fill input, output, repo_id, and tasks from the experiment workspace.

    Raises:
        ValueError: If ``--experiment`` is given without ``--variant``, or the
            output is a linked dataset.
        FileNotFoundError: If the experiment has no ``raw/`` folder.
    """
    if not cfg.variant:
        if cfg.experiment:
            raise ValueError("--experiment needs --variant (the LeRobot dataset name)")
        return cfg
    experiment = resolve_experiment(cfg.experiment or active_experiment_name())
    raw = experiment.raw_dir
    input_path = Path(cfg.input) if cfg.input is not None else raw
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist; record or link-raw first")
    output = Path(cfg.output) if cfg.output is not None else experiment.lerobot_dir / cfg.variant
    if output.is_symlink():
        raise ValueError(f"{output} is a linked existing dataset; pick another --variant")
    tasks = cfg.tasks
    if tasks is not None and not Path(tasks).exists() and (raw / Path(tasks).name).is_file():
        tasks = raw / Path(tasks).name
    if tasks is None and (raw / "tasks.yaml").is_file():
        tasks = raw / "tasks.yaml"
    repo_id = cfg.repo_id
    if repo_id == DEFAULT_REPO_ID:
        repo_id = dataset_repo_id(experiment, cfg.variant)
    logger.info("Experiment {} variant {}: {} → {} (labels {})", experiment.name, cfg.variant, input_path, output, tasks)
    gripper_open = experiment.gripper_open if cfg.gripper_open == ConvertPi05Cfg.gripper_open else cfg.gripper_open
    gripper_close = experiment.gripper_close if cfg.gripper_close == ConvertPi05Cfg.gripper_close else cfg.gripper_close
    return replace(
        cfg,
        experiment=experiment.name,
        input=input_path,
        output=output,
        tasks=tasks,
        repo_id=repo_id,
        gripper_open=gripper_open,
        gripper_close=gripper_close,
    )


def dagger_segment_mode(cfg: ConvertExperimentCfg) -> str:
    """Rollouts default to the operator segments (HG-DAgger) unless ``--segments all``."""
    return "human" if cfg.segments == "whole" else cfg.segments


def check_base_settings(cfg: ConvertExperimentCfg, base_notes: dict) -> None:
    """Refuse a DAgger extension whose action/gripper encoding differs from its base.

    Raises:
        ValueError: If a setting recorded for the base differs from ``cfg``.
    """
    for key in SHARED_SETTINGS:
        if key in base_notes and base_notes[key] != getattr(cfg, key):
            raise ValueError(
                f"--base-dataset {cfg.base_dataset} was built with {key}={base_notes[key]!r}, "
                f"this conversion uses {getattr(cfg, key)!r}; pass the same setting"
            )


def convert_dagger(cfg: ConvertExperimentCfg) -> dict:
    """Base variant + DAgger rounds → ``cfg.output``. The base is only read.

    Raises:
        ValueError: Without ``--base-dataset``, or with mismatching encoding settings.
        FileNotFoundError: If the base or a round has no data.
        FileExistsError: If the output exists and ``--clean`` is not set.
    """
    experiment = resolve_experiment(cfg.experiment)
    if not cfg.base_dataset:
        raise ValueError("--dagger-rounds needs --base-dataset (the variant with the original demos)")
    base = experiment.lerobot_dir / cfg.base_dataset
    if not (base / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Base dataset {base} not found")
    check_base_settings(cfg, load_state(experiment)["datasets"].get(cfg.base_dataset, {}))
    if cfg.output is None:
        raise ValueError("output is required")
    output = Path(cfg.output)
    if output.exists() and not cfg.clean:
        raise FileExistsError(f"{output} already exists; pass --clean to overwrite")
    segments = dagger_segment_mode(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{output.name}.", dir=output.parent))
    try:
        parts = [base]
        rounds = {}
        for number in cfg.dagger_rounds:
            folder = experiment.dagger_round_dir(number)
            if not any(folder.glob("*.vla")):
                raise FileNotFoundError(f"{folder} has no saved rollouts")
            part = work / f"r{number}"
            result = convert(
                replace(
                    cfg,
                    input=folder,
                    output=part,
                    tasks=None,
                    repo_id=f"{cfg.repo_id}_r{number}",
                    clean=True,
                    segments=segments,
                )
            )
            rounds[number] = result["kept"]
            logger.info("Round {}: {} {} segment(s) from {}", number, result["kept"], segments, folder)
            parts.append(part)
        summary = merge_lerobot_datasets(parts, output, cfg.repo_id)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    logger.info("Wrote {} episodes ({} frames) to {}", summary["episodes"], summary["frames"], output)
    return {"output": str(output), "kept": summary["episodes"], "frames": summary["frames"], "dagger_episodes": rounds}


def record_variant(cfg: ConvertExperimentCfg, result: dict) -> None:
    """Note how a variant was built in the experiment's ``experiment.json``."""
    experiment = resolve_experiment(cfg.experiment)
    state = load_state(experiment)
    notes = {
        "tasks": str(cfg.tasks) if cfg.tasks is not None else None,
        "action_space": cfg.action_space,
        "trim_idle": cfg.trim_idle,
        "gripper_state_source": cfg.gripper_state_source,
        "binarize_gripper": cfg.binarize_gripper,
        "gripper_open": cfg.gripper_open,
        "gripper_close": cfg.gripper_close,
        "episodes": result.get("kept"),
        "skipped": result.get("skipped"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
    }
    if cfg.dagger_rounds:
        notes.update(
            {
                "tasks": None,
                "base_dataset": cfg.base_dataset,
                "dagger_rounds": list(cfg.dagger_rounds),
                "dagger_episodes": result.get("dagger_episodes"),
                "segments": dagger_segment_mode(cfg),
                "include_autonomous": cfg.include_autonomous,
                "pre_takeover_cut_s": cfg.pre_takeover_cut_s,
                "max_pause_s": cfg.max_pause_s if cfg.trim_operator_pauses else None,
            }
        )
    state["datasets"][cfg.variant] = notes
    state["dataset"] = state["dataset"] or cfg.variant
    save_state(experiment, state)


def main(cfg: ConvertPi05Cfg) -> dict:
    """Convert local trajectories using the shared dataset conversion implementation."""
    if not isinstance(cfg, ConvertExperimentCfg) or not (cfg.variant or cfg.experiment):
        return convert(cfg)
    cfg = resolve_experiment_paths(cfg)
    if cfg.dagger_rounds:
        if cfg.dry_run or cfg.write_tasks_template is not None:
            raise ValueError("--dagger-rounds builds a dataset; use --input <round folder> --dry-run to inspect")
        result = convert_dagger(cfg)
    else:
        result = convert(cfg)
    if not cfg.dry_run and cfg.write_tasks_template is None:
        record_variant(cfg, result)
    return result


if __name__ == "__main__":
    main(tyro.cli(ConvertExperimentCfg))
