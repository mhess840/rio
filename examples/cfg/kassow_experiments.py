"""Named Kassow setups: wrist camera stays; overview serial, home, and recorder switch together."""

from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from rio.cfg import Camera

HOMES_DIR = Path(__file__).resolve().parent / "kassow_homes"
WRIST_SERIAL = "112322070077"
_DEFAULT_RECORDER_PATH = "data/kassow_vive_gripper/"


@dataclass(frozen=True)
class KassowExperiment:
    """One collect/eval cell: overview RealSense, home JSON, and trajectory folder."""

    name: str
    overview_serial: str
    home_filename: str
    recorder_path: str

    @property
    def home_path(self) -> Path:
        """Absolute path of the named home snapshot."""
        return HOMES_DIR / self.home_filename


EXPERIMENTS = {
    "pick": KassowExperiment(
        name="pick",
        overview_serial="262522075082",
        home_filename="home_pick.json",
        recorder_path="data/kassow_vive_gripper/",
    ),
    "sort": KassowExperiment(
        name="sort",
        overview_serial="261222078043",
        home_filename="home_sort.json",
        recorder_path="data/kassow_vive_sort/",
    ),
}


def resolve_experiment(name: str) -> KassowExperiment:
    """Look up a named Kassow experiment.

    Args:
        name: ``pick`` or ``sort``.

    Returns:
        The matching preset.

    Raises:
        ValueError: If ``name`` is not a known experiment.
    """
    key = str(name).strip().lower()
    if key not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {sorted(EXPERIMENTS)}, got {name!r}")
    return EXPERIMENTS[key]


def apply_kassow_experiment(station) -> KassowExperiment:
    """Bind overview serial, home file, and recorder path for ``station.experiment``.

    Wrist ``camera_1`` stays ``WRIST_SERIAL``. ``overview_serial`` and ``home_file`` on
    the station, when set, win over the preset. A custom ``recorder_cfg.path`` that is
    not one of the preset folders is left alone.

    Args:
        station: A ``KassowViveGripperStation`` (or subclass) instance.

    Returns:
        The resolved preset.
    """
    experiment = resolve_experiment(getattr(station, "experiment", "pick"))
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
    if not getattr(station, "home_file", None):
        station.home_file = str(experiment.home_path)
    recorder_cfg = getattr(station, "recorder_cfg", None)
    if recorder_cfg is not None:
        current = getattr(recorder_cfg, "path", None)
        known = {spec.recorder_path for spec in EXPERIMENTS.values()} | {_DEFAULT_RECORDER_PATH}
        if current in known or not current:
            recorder_cfg.path = experiment.recorder_path
    logger.info(
        "Kassow experiment={}  wrist={}  overview={}  home={}  recorder={}",
        experiment.name,
        WRIST_SERIAL,
        serial,
        getattr(station, "home_file", None),
        getattr(recorder_cfg, "path", None),
    )
    return experiment
