"""π₀.₅ rollout with operator takeover (HG-DAgger) on the Kassow station.

Same model binding, instruction check, RTC, and limits as ``Pi05LerobotCfg``;
additionally the Vive tracker, terminal keys, clicker, and recorder are on::

    POLICY=Pi05DaggerCfg uv run --group pi --no-group docs -m examples.policy_inference \\
        --policy-cfg.device cuda:1 --dagger-round 1 --instruction "put the object in space 2"

Rollouts go to ``experiments/<exp>/dagger/r<N>/``; ``raw/`` is never written.
Controls are in ``examples/dagger_kassow.py``.
"""

from dataclasses import dataclass, field, fields

from examples.cfg.kassow_experiments import resolve_experiment
from examples.cfg.kassow_vive_gripper import KassowViveGripperStation
from rio.cfg.common import RecorderCfg
from rio.cfg.node import NodeCfg

from .pi05_lerobot import Pi05LerobotCfg


def _station_default(name: str):
    """Fresh default of a ``KassowViveGripperStation`` field (teleop devices, recorder)."""
    factory = {item.name: item for item in fields(KassowViveGripperStation)}[name].default_factory
    if not callable(factory):
        raise TypeError(f"KassowViveGripperStation.{name} has no default factory")
    return factory()


@dataclass
class Pi05DaggerCfg(Pi05LerobotCfg):
    """``Pi05LerobotCfg`` plus Vive takeover and recording into a DAgger round."""

    dagger_round: int = 1

    teleop: str | None = "ViveTracker"
    teleop_cfg: NodeCfg | None = field(default_factory=lambda: _station_default("teleop_cfg"))
    teleop_keyboard: str | None = "SshKeyboard"
    teleop_keyboard_cfg: NodeCfg | None = field(default_factory=lambda: _station_default("teleop_keyboard_cfg"))
    teleop_clicker: str | None = "Clicker"
    teleop_clicker_cfg: NodeCfg | None = field(default_factory=lambda: _station_default("teleop_clicker_cfg"))
    recorder_cfg: RecorderCfg = field(default_factory=lambda: _station_default("recorder_cfg"))

    def __post_init__(self):
        # The recorder stays off until the base binding is done, so the experiment
        # binding neither points it at raw/ nor applies the frozen-raw check.
        self.recorder = None
        Pi05LerobotCfg.__post_init__(self)
        if self.dagger_round < 1:
            raise ValueError(f"--dagger-round starts at 1, got {self.dagger_round}")
        if str(self.policy_cfg.action_space).lower() != "tcp":
            raise ValueError("DAgger takeover drives the TCP; use a TCP checkpoint")
        self.recorder = "Recorder"
        self.recorder_cfg.path = str(resolve_experiment(self.experiment).dagger_round_dir(self.dagger_round))
