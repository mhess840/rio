"""Kassow conversion CLI with experiment paths and HG-DAgger dataset extension."""

import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import tyro

from rio.apps.kassow.experiments import (
    active_experiment_name,
    dataset_repo_id,
    load_state,
    resolve_experiment,
    save_state,
)
from rio.data.convert_pi05 import ConvertPi05Cfg, convert, merge_lerobot_datasets

SHARED_SETTINGS = ("action_space", "gripper_state_source", "binarize_gripper", "gripper_open", "gripper_close")


@dataclass
class ConvertExperimentCfg(ConvertPi05Cfg):
    """Shared conversion settings plus experiment and DAgger workspace selection."""

    experiment: str = ""
    variant: str = ""
    dagger_rounds: tuple[int, ...] = ()
    base_dataset: str = ""


def resolve_experiment_paths(cfg: ConvertExperimentCfg) -> ConvertExperimentCfg:
    """Resolve experiment paths without allowing linked datasets to be replaced."""
    if not cfg.variant:
        if cfg.experiment:
            raise ValueError("--experiment needs --variant")
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
    return replace(
        cfg,
        experiment=experiment.name,
        input=input_path,
        output=output,
        tasks=tasks,
        repo_id=dataset_repo_id(experiment, cfg.variant) if cfg.repo_id == ConvertPi05Cfg.repo_id else cfg.repo_id,
        gripper_open=experiment.gripper_open if cfg.gripper_open == ConvertPi05Cfg.gripper_open else cfg.gripper_open,
        gripper_close=experiment.gripper_close if cfg.gripper_close == ConvertPi05Cfg.gripper_close else cfg.gripper_close,
    )


def dagger_segment_mode(cfg: ConvertExperimentCfg) -> str:
    """Default DAgger extensions to operator-controlled segments."""
    return "human" if cfg.segments == "whole" else cfg.segments


def check_base_settings(cfg: ConvertExperimentCfg, base_notes: dict) -> None:
    """Reject an extension with state or action encodings different from its base."""
    for key in SHARED_SETTINGS:
        if key in base_notes and base_notes[key] != getattr(cfg, key):
            raise ValueError(f"--base-dataset {cfg.base_dataset} uses {key}={base_notes[key]!r}; use the same setting")


def convert_dagger(cfg: ConvertExperimentCfg) -> dict:
    """Append selected operator episodes while retaining the original dataset."""
    experiment = resolve_experiment(cfg.experiment)
    if not cfg.base_dataset:
        raise ValueError("--dagger-rounds needs --base-dataset")
    base = experiment.lerobot_dir / cfg.base_dataset
    if not (base / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Base dataset {base} not found")
    check_base_settings(cfg, load_state(experiment)["datasets"].get(cfg.base_dataset, {}))
    if cfg.output is None:
        raise ValueError("output is required")
    output = Path(cfg.output)
    if output.resolve() == base.resolve() or output.resolve() in base.resolve().parents:
        raise ValueError("The DAgger output must not replace its base dataset")
    if output.exists() and not cfg.clean:
        raise FileExistsError(f"{output} already exists; pass --clean to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    rounds = {}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_dagger_", dir=output.parent) as work:
        parts = [base]
        for number in cfg.dagger_rounds:
            folder = experiment.dagger_round_dir(number)
            if not any(folder.glob("*.vla")):
                raise FileNotFoundError(f"{folder} has no saved rollouts")
            part = Path(work) / f"r{number}"
            result = convert(
                replace(
                    cfg,
                    input=folder,
                    output=part,
                    tasks=None,
                    repo_id=f"{cfg.repo_id}_r{number}",
                    clean=False,
                    segments=dagger_segment_mode(cfg),
                )
            )
            rounds[number] = result["kept"]
            parts.append(part)
        summary = merge_lerobot_datasets(parts, output, cfg.repo_id, clean=cfg.clean)
    return {"output": str(output), "kept": summary["episodes"], "frames": summary["frames"], "dagger_episodes": rounds}


def record_variant(cfg: ConvertExperimentCfg, result: dict) -> None:
    """Record the settings and provenance of a successfully built dataset."""
    experiment = resolve_experiment(cfg.experiment)
    state = load_state(experiment)
    notes = {
        **{key: getattr(cfg, key) for key in SHARED_SETTINGS},
        "tasks": str(cfg.tasks) if cfg.tasks is not None else None,
        "trim_idle": cfg.trim_idle,
        "episodes": result.get("kept"),
        "skipped": result.get("skipped"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
    }
    if cfg.dagger_rounds:
        notes.update(
            tasks=None,
            base_dataset=cfg.base_dataset,
            dagger_rounds=list(cfg.dagger_rounds),
            dagger_episodes=result.get("dagger_episodes"),
            segments=dagger_segment_mode(cfg),
            include_autonomous=cfg.include_autonomous,
            pre_takeover_cut_s=cfg.pre_takeover_cut_s,
            max_pause_s=cfg.max_pause_s if cfg.trim_operator_pauses else None,
        )
    state["datasets"][cfg.variant] = notes
    state["dataset"] = state["dataset"] or cfg.variant
    save_state(experiment, state)


def main(cfg: ConvertPi05Cfg) -> dict:
    """Convert directly or resolve a registered experiment and DAgger extension."""
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


def cli() -> None:
    """Parse ``ConvertExperimentCfg`` from the command line and convert."""
    main(tyro.cli(ConvertExperimentCfg))


if __name__ == "__main__":
    cli()
