# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator state machine with fake Jev, arm, VLA, and keys. No hardware."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples.cfg.kassow_experiments import load_state, resolve_experiment, save_state
from examples.orchestrator_kassow import (
    Args,
    CycleLog,
    GripperOpened,
    Orchestrator,
    State,
    VlaJob,
    station_cfg,
    vla_cfg,
)
from rio.jev import CrateReading, Decision
from rio.jev.client import JevError

pytestmark = pytest.mark.unit

ORGANIZE_HOME = Path("home_organize.json")
SPACES = {1: Path("home_open_space1.json"), 2: Path("home_open_space2.json"), 3: Path("home_open_space3.json")}


class ScriptedKeys:
    """Each ``pressed`` call returns the next scripted key set."""

    def __init__(self, *presses: str):
        self.presses = [set(p) for p in presses]

    def pressed(self) -> set[str]:
        if not self.presses:
            raise AssertionError("orchestrator waited for a key that was never scripted")
        return self.presses.pop(0)


def _reading(decision: Decision, space: int | None = None, reason: str = "") -> CrateReading:
    return CrateReading(
        organized_p=0.9, count=0, count_p=0.9, confidence=0.9, any_box_p=0.1, decision=decision, space=space, reason=reason
    )


def _orchestrator(tmp_path, readings, keys, vla_reasons=(), goto_ok=True):
    calls = []
    readings = list(readings)
    vla_reasons = list(vla_reasons)

    def ask():
        item = readings.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def goto(path):
        calls.append(("goto", path.name))
        return goto_ok

    def run_vla(job: VlaJob):
        calls.append(("vla", job.experiment, job.instruction))
        return vla_reasons.pop(0)

    orchestrator = Orchestrator(
        ask=ask,
        goto=goto,
        run_vla=run_vla,
        keys=keys,
        organize_home=ORGANIZE_HOME,
        spaces=SPACES,
        log=CycleLog(tmp_path / "run"),
    )
    return orchestrator, calls


def test_open_path_runs_space_n_plus_one_then_waits_for_p(tmp_path):
    orchestrator, calls = _orchestrator(
        tmp_path, [_reading("open", space=2)], ScriptedKeys("j", "p", "q"), vla_reasons=["stop_when"]
    )
    orchestrator.run()
    assert calls == [
        ("goto", "home_open_space2.json"),
        ("vla", "open", "put the object in space 2"),
        ("goto", "home_organize.json"),
    ]
    events = [json.loads(line)["event"] for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert events == ["reading", "goto", "vla", "goto"]


def test_organize_path_asks_again_after_the_rollout(tmp_path):
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize"), _reading("open", space=1)],
        ScriptedKeys("j", "p", "q"),
        vla_reasons=["timeout", "stop_when"],
    )
    orchestrator.run()
    assert calls == [
        ("goto", "home_organize.json"),
        ("vla", "organize", ""),
        ("goto", "home_open_space1.json"),
        ("vla", "open", "put the object in space 1"),
        ("goto", "home_organize.json"),
    ]


@pytest.mark.parametrize(
    ("reading", "vla_reasons", "goto_ok", "hold_contains"),
    [
        (_reading("hold", reason="count unsure"), [], True, "count unsure"),
        (JevError("HTTP 400"), [], True, "HTTP 400"),
        (_reading("open", space=1), ["timeout"], True, "without opening"),
        (_reading("open", space=1), ["aborted"], True, "aborted"),
        (_reading("open", space=1), [], False, "moveJ"),
    ],
)
def test_anything_unsure_holds_and_never_moves_on(tmp_path, reading, vla_reasons, goto_ok, hold_contains):
    orchestrator, calls = _orchestrator(tmp_path, [reading], ScriptedKeys("j", "q"), vla_reasons, goto_ok)
    orchestrator.run()
    assert hold_contains in orchestrator.hold_reason
    assert ("goto", "home_organize.json") not in calls


def test_gripper_opened_needs_a_closed_command_first():
    detect = GripperOpened(0.0, 0.3)
    assert not detect(np.array([0.0, 0.3]))
    assert not detect(np.array([0.0, 0.0]))
    assert detect(np.array([0.0, 0.3]))
    fresh = GripperOpened(0.0, 0.3)
    assert not fresh(np.array([0.3]))
    with pytest.raises(ValueError):
        GripperOpened(0.3, 0.0)


def _register(experiment_name: str, tmp_path: Path, instructions: list[str]) -> None:
    experiment = resolve_experiment(experiment_name)
    checkpoint = tmp_path / f"ckpt_{experiment_name}"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"chunk_size": 50, "n_action_steps": 50}))
    experiment.models_dir.mkdir(parents=True)
    (experiment.models_dir / "m1").symlink_to(checkpoint)
    state = load_state(experiment)
    state["model"] = "m1"
    state["models"]["m1"] = {"dataset": "d", "instructions": instructions}
    save_state(experiment, state)


def test_vla_configs_carry_slot_sentence_home_and_gripper_encoding(tmp_path):
    _register("organize", tmp_path, ["organize the inlay"])
    _register("open", tmp_path, [f"put the object in space {n}" for n in (1, 2, 3)])
    args = Args(device="cuda:1")
    organize = vla_cfg(args, VlaJob("organize", "", ORGANIZE_HOME))
    opened = vla_cfg(args, VlaJob("open", "put the object in space 2", SPACES[2]))
    assert organize.policy_cfg.resident_slot == "organize"
    assert opened.policy_cfg.resident_slot == "open"
    assert organize.instruction == "organize the inlay"
    assert opened.policy_node_cfg.instruction == "put the object in space 2"
    assert Path(opened.home_file).name == "home_open_space2.json"
    assert (organize.policy_cfg.gripper_open, opened.policy_cfg.gripper_open) == (0.7, 0.3)
    assert (
        vla_cfg(Args(single_slot=True), VlaJob("open", "put the object in space 1", SPACES[1])).policy_cfg.resident_slot == ""
    )
    station = station_cfg(args)
    assert not station.policy
    assert station.teleop_keyboard == "SshKeyboard"
    assert list(station.cameras) == ["camera_1", "camera_2"]


def test_quit_from_idle(tmp_path):
    orchestrator, calls = _orchestrator(tmp_path, [], ScriptedKeys("q"))
    orchestrator.run()
    assert orchestrator.state is State.DONE
    assert calls == []
