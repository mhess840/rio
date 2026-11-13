"""Named Kassow experiments and their git-ignored workspaces.

The registry below is versioned: camera serial and home pose file of each
experiment. Everything large lives in a git-ignored workspace, one folder per
experiment::

    experiments/
        active.json                 {"experiment": "sort"}
        sort/
            experiment.json         selected dataset/model, model + dataset notes
            raw/                    the .vla set and its label files (tasks*.yaml)
            dagger/r<N>/            DAgger rollouts of round N (+ discarded/)
            lerobot/<variant>/      LeRobot datasets built from raw/ (and dagger/)
            models/<name>/          up to MAX_MODELS checkpoints

``uv run -m rio.apps.kassow.experiment_cli use <name>`` switches the active experiment;
teleop, convert, serve, and goto default to it. See
docs/workflow/pi05_kassow_pipeline.md.
"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from loguru import logger

from rio.apps.kassow.hardware import JEV_SERIAL, WRIST_SERIAL
from rio.cfg import Camera
from rio.workspace import experiments_root

# Reviewed station poses shipped with the code; read-only at runtime.
PACKAGED_HOMES_DIR = Path(__file__).resolve().parent / "homes"
JEV_CAMERA_ADDR = "127.0.0.1:5132"
DEFAULT_EXPERIMENT = "sort"
MAX_MODELS = 3
_DEFAULT_RECORDER_PATH = "data/kassow_vive_gripper/"


def homes_dir() -> Path:
    """Poses captured on this station (``--capture-home``, waiting pose); they override the packaged ones."""
    return experiments_root() / "homes"


def home_file(filename: str) -> Path:
    """The captured pose of this name if one exists, else the packaged default."""
    captured = homes_dir() / filename
    return captured if captured.is_file() else PACKAGED_HOMES_DIR / filename


def writable_home(path: Path) -> Path:
    """Where a new snapshot for ``path`` goes: packaged poses are never overwritten."""
    path = Path(path)
    return homes_dir() / path.name if path.resolve().parent == PACKAGED_HOMES_DIR else path


@dataclass(frozen=True)
class InstructionHome:
    """One start pose, chosen by the task sentence or its short ``pose`` name."""

    instruction: str
    filename: str
    pose: str

    @property
    def path(self) -> Path:
        """Captured or packaged snapshot of this pose."""
        return home_file(self.filename)


@dataclass(frozen=True)
class KassowExperiment:
    """One collect/eval cell: overview RealSense, home JSON, and workspace folders.

    ``homes`` is empty for a single start pose (``home_filename``). When it is
    set, teleop and goto need ``--instruction`` or ``--pose <name>``; there is
    no silent default.
    """

    name: str
    overview_serial: str
    home_filename: str = ""
    homes: tuple[InstructionHome, ...] = ()
    # Discrete gripper targets. ``gripper_start`` is the opening at session start
    # and in the home file; None keeps the open target (the other experiments).
    gripper_close: float = 0.0
    gripper_open: float = 0.7
    gripper_start: float | None = None
    # None keeps the station force. Sort asks for a firmer jaw than that default.
    gripper_force: int | None = None

    def __post_init__(self) -> None:
        if not self.home_filename and not self.homes:
            raise ValueError(f"experiment {self.name!r} needs a home_filename or homes")
        poses = [home.pose for home in self.homes]
        texts = [home.instruction for home in self.homes]
        if len(poses) != len(set(poses)) or len(texts) != len(set(texts)):
            raise ValueError(f"experiment {self.name!r} has a duplicate home pose or instruction")
        if not self.gripper_close < self.gripper_open:
            raise ValueError(f"experiment {self.name!r} needs gripper_close < gripper_open")

    @property
    def home_path(self) -> Path:
        """Captured or packaged single home snapshot.

        Raises:
            ValueError: If this experiment has one home per instruction.
        """
        if self.homes:
            raise ValueError(f"experiment {self.name!r} has one home per instruction; pass --instruction or --pose")
        return home_file(self.home_filename)

    def select_home(self, instruction: str = "", pose: str = "") -> Path:
        """Home file for a task sentence or a short pose name.

        A single-home experiment ignores both arguments. With several homes,
        ``instruction`` and ``pose`` must name the same snapshot when both are set.

        Raises:
            ValueError: If several homes exist and the request does not match one.
        """
        if not self.homes:
            return self.home_path
        text = instruction.strip()
        key = pose.strip().lower()
        by_text = {home.instruction: home for home in self.homes}
        by_pose = {home.pose: home for home in self.homes}
        chosen = by_text.get(text) if text else None
        if text and chosen is None:
            raise ValueError(self._home_choices(f"No home for instruction {text!r}."))
        if key and key not in {"home", self.name}:
            posed = by_pose.get(key)
            if posed is None:
                raise ValueError(self._home_choices(f"No home named {key!r}."))
            if chosen is not None and posed is not chosen:
                raise ValueError(f"--pose {key} is {posed.instruction!r}, which is not {text!r}")
            chosen = posed
        if chosen is None:
            raise ValueError(self._home_choices("Pass --instruction or --pose."))
        return chosen.path

    def _home_choices(self, reason: str) -> str:
        lines = "\n  ".join(f"--pose {home.pose}    {home.instruction}" for home in self.homes)
        return f"{reason} Homes of '{self.name}':\n  {lines}"

    @property
    def root(self) -> Path:
        """Workspace folder of this experiment."""
        return experiments_root() / self.name

    @property
    def raw_dir(self) -> Path:
        """Folder the recorder writes ``traj_XXXX.vla`` into."""
        return self.root / "raw"

    @property
    def lerobot_dir(self) -> Path:
        """Parent of the LeRobot dataset variants."""
        return self.root / "lerobot"

    @property
    def models_dir(self) -> Path:
        """Parent of the registered checkpoints."""
        return self.root / "models"

    @property
    def state_path(self) -> Path:
        """Selected dataset/model and per-entry notes."""
        return self.root / "experiment.json"

    @property
    def dagger_dir(self) -> Path:
        """Parent of the DAgger rounds ``r1``, ``r2``, …"""
        return self.root / "dagger"

    def dagger_round_dir(self, number: int) -> Path:
        """Folder of DAgger round ``number``; rejected rollouts go to its ``discarded/``."""
        if int(number) < 1:
            raise ValueError(f"DAgger rounds start at 1, got {number}")
        return self.dagger_dir / f"r{int(number)}"


EXPERIMENTS = {
    "pick": KassowExperiment(
        name="pick",
        overview_serial="262522075082",
        home_filename="home_pick.json",
    ),
    "sort": KassowExperiment(
        name="sort",
        overview_serial="261222078043",
        home_filename="home_sort.json",
        # Recordings label the open jaw as 0.7. The calibrated stroke now stops near 0.3,
        # so playback opens to 0.4 and closes a bit harder than the station default.
        gripper_open=0.4,
        gripper_force=70,
    ),
    "organize": KassowExperiment(
        name="organize",
        overview_serial="261222078043",
        home_filename="home_organize.json",
        # Open jaw in these demos is 0.7, not the 0.3 of ``open``.
        gripper_open=0.7,
    ),
    # Same overview as sort. Each sentence has its own start pose.
    "open": KassowExperiment(
        name="open",
        overview_serial="261222078043",
        homes=(
            InstructionHome("put the object in space 1", "home_open_space1.json", "space1"),
            InstructionHome("put the object in space 2", "home_open_space2.json", "space2"),
            InstructionHome("put the object in space 3", "home_open_space3.json", "space3"),
        ),
        gripper_open=0.3,
        gripper_start=0.0,
    ),
}


def jev_camera(serial: str | None = None, addr: str = JEV_CAMERA_ADDR) -> Camera:
    """RealSense node spec for the Jev camera, 640x480 color at 6 fps.

    6 fps is a mode the D435i actually offers. One still photo for Jev does not
    need 30, and 30 fps of this resolution does not fit the USB 2 link that
    camera is on.

    Args:
        serial: Override ``JEV_SERIAL``.
        addr: Node address; must differ from the policy cameras (5130, 5131).
    """
    return Camera(
        addr=addr,
        cam_type="Realsense",
        serial=str(serial or JEV_SERIAL),
        model="D400",
        enable_depth=False,
        hardware_reset=False,
        freq=6,
        resolution=(480, 640),
        resolution_depth=(480, 640),
    )


def open_spaces() -> dict[int, InstructionHome]:
    """1-based spaces of ``open`` whose home file exists, e.g. ``{1: space1, 2: space2}``."""
    spaces = {}
    for home in EXPERIMENTS["open"].homes:
        number = home.pose.removeprefix("space")
        if number.isdigit() and home.path.is_file():
            spaces[int(number)] = home
    return spaces


def resolve_experiment(name: str) -> KassowExperiment:
    """Look up a named Kassow experiment.

    Args:
        name: One of ``EXPERIMENTS``.

    Returns:
        The matching preset.

    Raises:
        ValueError: If ``name`` is not a known experiment.
    """
    key = str(name).strip().lower()
    if key not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {sorted(EXPERIMENTS)}, got {name!r}")
    return EXPERIMENTS[key]


def active_experiment_name() -> str:
    """Experiment chosen with ``rio.apps.kassow.experiment_cli use``, else ``DEFAULT_EXPERIMENT``."""
    path = experiments_root() / "active.json"
    if not path.is_file():
        return DEFAULT_EXPERIMENT
    name = json.loads(path.read_text()).get("experiment") or DEFAULT_EXPERIMENT
    return resolve_experiment(name).name


def set_active_experiment(name: str) -> KassowExperiment:
    """Make ``name`` the default for teleop, convert, serve, and goto.

    Args:
        name: Experiment to activate.

    Returns:
        The activated preset.
    """
    experiment = resolve_experiment(name)
    experiment.root.mkdir(parents=True, exist_ok=True)
    _write_json(experiments_root() / "active.json", {"experiment": experiment.name})
    return experiment


def load_state(experiment: KassowExperiment) -> dict:
    """Read ``experiment.json``; missing keys are filled with empty defaults."""
    state = json.loads(experiment.state_path.read_text()) if experiment.state_path.is_file() else {}
    state.setdefault("dataset", None)
    state.setdefault("model", None)
    state.setdefault("frozen", False)
    state.setdefault("datasets", {})
    state.setdefault("models", {})
    return state


def save_state(experiment: KassowExperiment, state: dict) -> None:
    """Write ``experiment.json``."""
    _write_json(experiment.state_path, state)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def ensure_raw_dir(experiment: KassowExperiment) -> Path:
    """Return ``raw/``, creating an empty one for a new experiment.

    Raises:
        FileNotFoundError: If ``raw/`` is a symlink whose target is gone. A new
            empty folder there would split the dataset.
    """
    raw = experiment.raw_dir
    if raw.is_dir():
        return raw
    if raw.is_symlink():
        raise FileNotFoundError(f"{raw} points to {os.readlink(raw)}, which does not exist")
    raw.mkdir(parents=True)
    logger.info("Created empty {}", raw)
    return raw


def link_entry(link: Path, target: Path) -> None:
    """Create ``link`` → ``target``. Never replaces an existing entry."""
    target = Path(target).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} already exists")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def resolve_model(experiment: KassowExperiment, name: str | None = None) -> tuple[str, Path]:
    """Path of model ``name``, or of the selected model when ``name`` is ``None``.

    Raises:
        FileNotFoundError: If nothing is registered or selected.
    """
    state = load_state(experiment)
    name = name or state["model"]
    if not name:
        available = sorted(state["models"])
        raise FileNotFoundError(
            f"No model selected for '{experiment.name}' (registered: {available or 'none'}). "
            f"Run: uv run -m rio.apps.kassow.experiment_cli use {experiment.name} --model <name>"
        )
    path = experiment.models_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"Model '{name}' of '{experiment.name}' not found at {path}")
    return name, path


def resolve_dataset(experiment: KassowExperiment, name: str | None = None) -> tuple[str, Path]:
    """Path of LeRobot variant ``name``, or of the selected one when ``name`` is ``None``.

    Raises:
        FileNotFoundError: If nothing is selected or the folder is missing.
    """
    name = name or load_state(experiment)["dataset"]
    if not name:
        raise FileNotFoundError(
            f"No LeRobot dataset selected for '{experiment.name}'. "
            f"Run: uv run -m rio.apps.kassow.experiment_cli use {experiment.name} --dataset <variant>"
        )
    path = experiment.lerobot_dir / name
    if not (path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset '{name}' of '{experiment.name}' not found at {path}")
    return name, path


def dagger_rounds(experiment: KassowExperiment) -> list[int]:
    """Numbers of the existing DAgger round folders, ascending."""
    if not experiment.dagger_dir.is_dir():
        return []
    names = (path.name for path in experiment.dagger_dir.iterdir() if path.is_dir())
    return sorted(int(name[1:]) for name in names if name.startswith("r") and name[1:].isdigit())


def dataset_repo_id(experiment: KassowExperiment, variant: str) -> str:
    """``repo_id`` used for a local variant."""
    return f"local/kassow_{experiment.name}_{variant}"


def find_model_owner(policy_path: str | Path) -> tuple[str, str, dict] | None:
    """Experiment, model name, and notes of the registered model at ``policy_path``."""
    target = Path(policy_path).expanduser().resolve()
    for experiment in EXPERIMENTS.values():
        state = load_state(experiment)
        for name, meta in state["models"].items():
            path = experiment.models_dir / name
            if path.exists() and path.resolve() == target:
                return experiment.name, name, meta
    return None


def dataset_tasks(dataset_root: Path) -> list[str]:
    """Task strings of a LeRobot v3 dataset in ``task_index`` order."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(dataset_root) / "meta" / "tasks.parquet").to_pydict()
    pairs = sorted(zip(table["task_index"], table["task"], strict=True))
    return [task for _, task in pairs]


def _read_safetensors(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    """Read float tensors from a ``.safetensors`` file without the safetensors package."""
    dtypes = {"F32": np.float32, "F64": np.float64}
    with open(path, "rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
        out = {}
        for name in names:
            spec = header.get(name)
            if spec is None or spec["dtype"] not in dtypes:
                continue
            start, end = spec["data_offsets"]
            handle.seek(8 + header_size + start)
            out[name] = np.frombuffer(handle.read(end - start), dtype=dtypes[spec["dtype"]]).reshape(spec["shape"])
    return out


def model_dataset_mismatch(model_dir: Path, dataset_root: Path) -> float | None:
    """Largest difference of the model's action normalizer vs the dataset stats.

    About 1e-7 means the model was trained on exactly this dataset. ``None`` if
    either side has no action statistics.
    """
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    norm_path = Path(model_dir) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not stats_path.is_file() or not norm_path.is_file():
        return None
    keys = ("q01", "q99", "min", "max")
    tensors = _read_safetensors(norm_path, {f"action.{key}" for key in keys})
    stats = json.loads(stats_path.read_text()).get("action", {})
    diffs = [
        float(np.max(np.abs(tensors[f"action.{key}"] - np.asarray(stats[key], dtype=np.float64))))
        for key in keys
        if f"action.{key}" in tensors and key in stats
    ]
    return max(diffs) if diffs else None


def apply_kassow_experiment(station) -> KassowExperiment:
    """Bind overview serial, home file, and recorder folder for ``station.experiment``.

    An empty ``station.experiment`` selects the active experiment. Wrist
    ``camera_1`` stays ``WRIST_SERIAL``. ``overview_serial`` and ``home_file`` on
    the station, when set, win over the preset. A custom ``recorder_cfg.path``
    is left alone. ``bind_home`` False skips the home file: playback draws a
    ``.vla`` and does not drive the arm.

    Args:
        station: A ``KassowViveGripperStation`` (or subclass) instance.

    Returns:
        The resolved preset.

    Raises:
        ValueError: If recording is enabled for a frozen experiment.
    """
    experiment = resolve_experiment(getattr(station, "experiment", None) or active_experiment_name())
    station.experiment = experiment.name
    override_serial = getattr(station, "overview_serial", None)
    serial = str(override_serial).strip() if override_serial else experiment.overview_serial
    cameras = dict(getattr(station, "cameras", {}) or {})
    wrist = cameras.get("camera_1")
    overview = cameras.get("camera_2")
    if isinstance(wrist, Camera):
        cameras["camera_1"] = replace(wrist, serial=WRIST_SERIAL)
    if isinstance(overview, Camera):
        cameras["camera_2"] = replace(overview, serial=serial)
    station.cameras = cameras
    _apply_gripper_encoding(station, experiment)
    # Playback sets bind_home False. It never drives to a snapshot, so a
    # multi-home experiment must not demand --instruction or --pose.
    if getattr(station, "bind_home", True) and not getattr(station, "home_file", None):
        instruction = str(getattr(station, "instruction", "") or "")
        pose = str(getattr(station, "pose", "") or "")
        station.home_file = str(experiment.select_home(instruction, pose))
        if experiment.homes and not instruction.strip():
            key = pose.strip().lower()
            match = next(home for home in experiment.homes if home.pose == key)
            station.instruction = match.instruction
    recorder_cfg = getattr(station, "recorder_cfg", None)
    if getattr(station, "recorder", None) and recorder_cfg is not None:
        current = getattr(recorder_cfg, "path", None)
        known = {str(spec.raw_dir) for spec in EXPERIMENTS.values()}
        known.add(_DEFAULT_RECORDER_PATH)
        if not current or current in known:
            if load_state(experiment)["frozen"]:
                raise ValueError(
                    f"'{experiment.name}' is frozen; its .vla set must not grow. "
                    f"Pass --recorder None to teleop without recording, or run "
                    f"uv run -m rio.apps.kassow.experiment_cli unfreeze {experiment.name}"
                )
            recorder_cfg.path = str(ensure_raw_dir(experiment))
    logger.info(
        "Kassow experiment={}  wrist={}  overview={}  home={}  gripper={}/{} start={}  recorder={}",
        experiment.name,
        WRIST_SERIAL,
        serial,
        getattr(station, "home_file", None),
        getattr(station, "gripper_close", None),
        getattr(station, "gripper_open", None),
        getattr(station, "gripper_start", None),
        getattr(recorder_cfg, "path", None) if getattr(station, "recorder", None) else None,
    )
    return experiment


def _apply_gripper_encoding(station, experiment: KassowExperiment) -> None:
    """Copy this experiment's discrete gripper targets onto the station.

    A value that is not the station default is an explicit override and stays.
    The policy config and the gripper node's startup opening follow the same
    numbers, so teleop, conversion, and rollout discretize to one pair.
    """
    start = experiment.gripper_open if experiment.gripper_start is None else experiment.gripper_start
    if getattr(station, "gripper_open", None) == 0.7:
        station.gripper_open = float(experiment.gripper_open)
    if getattr(station, "gripper_close", None) == 0.0:
        station.gripper_close = float(experiment.gripper_close)
    station.gripper_start = float(start)
    gripper_cfg = getattr(station, "gripper_cfg", None)
    if gripper_cfg is not None and getattr(gripper_cfg, "startup_position", None) == 0.7:
        gripper_cfg.startup_position = float(start)
    if gripper_cfg is not None and experiment.gripper_force is not None and getattr(gripper_cfg, "force", None) == 50:
        gripper_cfg.force = int(experiment.gripper_force)
    policy_cfg = getattr(station, "policy_cfg", None)
    if policy_cfg is not None:
        if getattr(policy_cfg, "gripper_open", None) == 0.7:
            policy_cfg.gripper_open = float(experiment.gripper_open)
        if getattr(policy_cfg, "gripper_close", None) == 0.0:
            policy_cfg.gripper_close = float(experiment.gripper_close)
