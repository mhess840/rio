"""Switch and inspect Kassow experiments (pick, sort, organize, open).

Status of the active experiment, with ready-to-run commands::

    uv run -m examples.experiment

Switch experiment, optionally choosing the model and LeRobot dataset::

    uv run -m examples.experiment use sort
    uv run -m examples.experiment use sort --model run1 --dataset spaces_notrim

Register existing data without copying it (symlinks; nothing is moved or deleted)::

    uv run -m examples.experiment link-raw /path/to/vla_folder --experiment pick
    uv run -m examples.experiment add-dataset notrim /path/to/lerobot_dataset --experiment pick
    uv run -m examples.experiment add-model run2 /path/to/pretrained_model --dataset spaces_notrim

A frozen experiment refuses to record more ``.vla`` files::

    uv run -m examples.experiment freeze pick
"""

import argparse
import json
from pathlib import Path

from examples.cfg.kassow_experiments import (
    EXPERIMENTS,
    MAX_MODELS,
    KassowExperiment,
    active_experiment_name,
    dagger_rounds,
    dataset_tasks,
    link_entry,
    load_state,
    model_dataset_mismatch,
    resolve_experiment,
    save_state,
    set_active_experiment,
)
from rio.data.dagger import SIDECAR_SUFFIX

STATS_MATCH_TOL = 1e-4


def _experiment(name: str | None) -> KassowExperiment:
    return resolve_experiment(name or active_experiment_name())


def _describe_link(path: Path) -> str:
    if path.is_symlink():
        return f"{path} → {path.resolve()}"
    return str(path)


def _dataset_summary(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return "missing"
    info = json.loads(info_path.read_text())
    return f"{info['total_episodes']} episodes, {info['total_frames']} frames"


def dagger_round_summary(folder: Path) -> str:
    """Saved/discarded rollouts, takeovers, and operator share of one DAgger round folder."""
    saved = [json.loads(path.read_text()) for path in sorted(folder.glob(f"*{SIDECAR_SUFFIX}"))]
    discarded = len(list((folder / "discarded").glob("*.vla"))) if (folder / "discarded").is_dir() else 0
    n_vla = len(list(folder.glob("*.vla")))
    if not saved:
        return f"{n_vla} saved, {discarded} discarded"
    takeovers = sum(int(meta.get("takeovers", 0)) for meta in saved)
    autonomous = sum(1 for meta in saved if meta.get("autonomous"))
    human = sum(int(meta.get("frames", {}).get("human", 0)) for meta in saved)
    driven = human + sum(int(meta.get("frames", {}).get("policy", 0)) for meta in saved)
    share = 100.0 * human / driven if driven else 0.0
    return (
        f"{n_vla} saved ({autonomous} without takeover), {discarded} discarded; "
        f"{takeovers / len(saved):.2f} takeovers/episode, operator {share:.0f}% of frames"
    )


def _home_line(path: Path, label: str) -> str:
    if not path.is_file():
        return f"{label}: MISSING ({path.name}); capture once with --capture-home"
    home = json.loads(path.read_text())
    xyz = [round(float(v), 3) for v in home["eef_pose"][:3]]
    return f"{label}: tcp={xyz}  ({path.name})"


def _home_summary(experiment: KassowExperiment) -> str:
    if experiment.homes:
        return " | ".join(_home_line(home.path, home.pose) for home in experiment.homes)
    return _home_line(experiment.home_path, "home")


def cmd_status(args) -> None:
    """Print cameras, home, data, models, and the commands for this experiment."""
    experiment = _experiment(args.experiment)
    state = load_state(experiment)
    active = active_experiment_name()
    print(f"Experiment: {experiment.name}{'  (active)' if experiment.name == active else ''}")
    print(f"  overview camera: {experiment.overview_serial}")
    print(f"  home:            {_home_summary(experiment)}")
    start = experiment.gripper_open if experiment.gripper_start is None else experiment.gripper_start
    print(
        f"  gripper:         closed {experiment.gripper_close:g} / open {experiment.gripper_open:g}"
        f" / start {start:g}"
    )
    raw = experiment.raw_dir
    n_vla = len(list(raw.glob("traj_*.vla"))) if raw.is_dir() else 0
    frozen = "  FROZEN" if state["frozen"] else ""
    print(f"  raw .vla:        {_describe_link(raw) if raw.exists() else 'not created yet'} ({n_vla} files){frozen}")
    labels = sorted(path.name for path in raw.glob("tasks*.yaml")) if raw.is_dir() else []
    print(f"  label files:     {', '.join(labels) or 'none'}")
    rounds = dagger_rounds(experiment)
    if rounds:
        print("  DAgger rounds:")
        for number in rounds:
            print(f"     r{number}: {dagger_round_summary(experiment.dagger_round_dir(number))}")
    print("  LeRobot datasets:")
    variants = sorted(path.name for path in experiment.lerobot_dir.iterdir()) if experiment.lerobot_dir.is_dir() else []
    for variant in variants:
        mark = "*" if variant == state["dataset"] else " "
        print(f"   {mark} {variant}: {_dataset_summary(experiment.lerobot_dir / variant)}")
    if not variants:
        print("     none")
    print(f"  models (max {MAX_MODELS}):")
    names = sorted(path.name for path in experiment.models_dir.iterdir()) if experiment.models_dir.is_dir() else []
    for name in names:
        meta = state["models"].get(name, {})
        mark = "*" if name == state["model"] else " "
        print(f"   {mark} {name}: {(experiment.models_dir / name).resolve()}")
        print(f"       dataset {meta.get('dataset') or '?'}; tasks {meta.get('instructions') or '?'}")
    if not names:
        print("     none")
    print("\nCommands:")
    if experiment.homes:
        for home in experiment.homes:
            print(f"  goto {home.pose}:    uv run python -m examples.goto_kassow_q --pose {home.pose}")
            print(
                f'  record:  uv run python -m examples.kassow_vive_gripper --go-home-on-save --instruction "{home.instruction}"'
            )
    else:
        print("  teleop:  uv run python -m examples.kassow_vive_gripper --go-home-on-save")
        print("  goto:    uv run python -m examples.goto_kassow_q --pose home")
    print(
        "  convert: uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 "
        "--variant <name> --tasks tasks.yaml --no-trim-idle --gripper-state-source commanded"
    )
    print(
        "  serve:   POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference "
        "--policy-cfg.device cuda:1"
    )
    print(
        "  dagger:  POLICY=Pi05DaggerCfg uv run --group pi --no-group docs -m examples.policy_inference "
        f"--policy-cfg.device cuda:1 --dagger-round {(rounds[-1] if rounds else 1)} --instruction <task>"
    )


def cmd_list(_args) -> None:
    """One line per experiment."""
    active = active_experiment_name()
    for name, experiment in EXPERIMENTS.items():
        state = load_state(experiment)
        mark = "*" if name == active else " "
        print(f"{mark} {name}: dataset={state['dataset']} model={state['model']} overview={experiment.overview_serial}")


def cmd_use(args) -> None:
    """Activate an experiment and optionally select its dataset/model."""
    experiment = set_active_experiment(args.name)
    state = load_state(experiment)
    if args.model:
        if not (experiment.models_dir / args.model).is_dir():
            raise SystemExit(f"No model '{args.model}' in {experiment.models_dir}")
        state["model"] = args.model
    if args.dataset:
        if not (experiment.lerobot_dir / args.dataset).is_dir():
            raise SystemExit(f"No dataset '{args.dataset}' in {experiment.lerobot_dir}")
        state["dataset"] = args.dataset
    save_state(experiment, state)
    args.experiment = experiment.name
    cmd_status(args)


def cmd_link_raw(args) -> None:
    """Point ``raw/`` at an existing ``.vla`` folder."""
    experiment = _experiment(args.experiment)
    link_entry(experiment.raw_dir, Path(args.path))
    print(_describe_link(experiment.raw_dir))


def cmd_add_dataset(args) -> None:
    """Register an existing LeRobot dataset as a variant (symlink)."""
    experiment = _experiment(args.experiment)
    link_entry(experiment.lerobot_dir / args.name, Path(args.path))
    state = load_state(experiment)
    state["datasets"].setdefault(args.name, {})["source"] = str(Path(args.path).expanduser().resolve())
    if args.tasks:
        state["datasets"][args.name]["tasks"] = args.tasks
    state["dataset"] = state["dataset"] or args.name
    save_state(experiment, state)
    print(f"{args.name}: {_dataset_summary(experiment.lerobot_dir / args.name)}")


def cmd_add_model(args) -> None:
    """Register a checkpoint (symlink) with its training dataset and tasks."""
    experiment = _experiment(args.experiment)
    target = Path(args.path).expanduser().resolve()
    slot = experiment.models_dir / args.name
    # A checkpoint copied straight into models/<name>/ is registered where it is.
    in_place = slot.exists() and slot.resolve() == target
    existing = [path.name for path in experiment.models_dir.iterdir()] if experiment.models_dir.is_dir() else []
    others = [name for name in existing if not (in_place and name == args.name)]
    if len(others) >= MAX_MODELS:
        raise SystemExit(
            f"'{experiment.name}' already has {MAX_MODELS} models ({others}). "
            f"Remove a symlink in {experiment.models_dir} by hand first."
        )
    if not (target / "config.json").is_file():
        raise SystemExit(f"{target} has no config.json; point at the pretrained_model folder")
    dataset = args.dataset
    variants = sorted(path.name for path in experiment.lerobot_dir.iterdir()) if experiment.lerobot_dir.is_dir() else []
    checks = {name: model_dataset_mismatch(target, experiment.lerobot_dir / name) for name in variants}
    matches = [name for name, diff in checks.items() if diff is not None and diff < STATS_MATCH_TOL]
    if dataset is None:
        if not matches:
            raise SystemExit(f"No dataset of '{experiment.name}' matches the model stats; pass --dataset")
        dataset = matches[0]
        print(f"Training dataset by normalizer stats: {dataset} (matches: {matches})")
    elif dataset not in variants:
        raise SystemExit(f"No dataset '{dataset}' in {experiment.lerobot_dir}")
    elif dataset not in matches:
        print(f"WARNING: model stats differ from '{dataset}' (max |Δ| {checks[dataset]})")
    instructions = args.instruction or dataset_tasks(experiment.lerobot_dir / dataset)
    if not in_place:
        link_entry(slot, target)
    state = load_state(experiment)
    state["models"][args.name] = {"dataset": dataset, "instructions": instructions, "stats_matches": matches}
    state["model"] = state["model"] or args.name
    save_state(experiment, state)
    print(f"{args.name}: dataset {dataset}, tasks {instructions}")


def cmd_freeze(args, frozen: bool) -> None:
    """Allow or forbid recording into ``raw/``."""
    experiment = _experiment(args.name)
    state = load_state(experiment)
    state["frozen"] = frozen
    save_state(experiment, state)
    print(f"{experiment.name}: {'frozen' if frozen else 'recording allowed'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="examples.experiment", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="show an experiment (default: active)")
    status.add_argument("experiment", nargs="?")
    sub.add_parser("list", help="all experiments")

    use = sub.add_parser("use", help="activate an experiment")
    use.add_argument("name", choices=sorted(EXPERIMENTS))
    use.add_argument("--model")
    use.add_argument("--dataset")

    link_raw = sub.add_parser("link-raw", help="use an existing .vla folder as raw/")
    link_raw.add_argument("path")
    link_raw.add_argument("--experiment")

    add_dataset = sub.add_parser("add-dataset", help="register an existing LeRobot dataset")
    add_dataset.add_argument("name")
    add_dataset.add_argument("path")
    add_dataset.add_argument("--experiment")
    add_dataset.add_argument("--tasks", help="label file it was built from (note only)")

    add_model = sub.add_parser("add-model", help=f"register a checkpoint (max {MAX_MODELS})")
    add_model.add_argument("name")
    add_model.add_argument("path")
    add_model.add_argument("--experiment")
    add_model.add_argument("--dataset", help="training dataset variant; default: matched by stats")
    add_model.add_argument("--instruction", action="append", help="override the dataset task list")

    for name in ("freeze", "unfreeze"):
        cmd = sub.add_parser(name, help=f"{name} recording into raw/")
        cmd.add_argument("name", nargs="?")

    args = parser.parse_args(argv)
    if args.command in (None, "status"):
        args.experiment = getattr(args, "experiment", None)
        cmd_status(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "use":
        cmd_use(args)
    elif args.command == "link-raw":
        cmd_link_raw(args)
    elif args.command == "add-dataset":
        cmd_add_dataset(args)
    elif args.command == "add-model":
        cmd_add_model(args)
    else:
        cmd_freeze(args, frozen=args.command == "freeze")


if __name__ == "__main__":
    main()
