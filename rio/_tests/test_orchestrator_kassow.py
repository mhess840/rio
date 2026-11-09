# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator state machine with fake Jev, arm, VLA, and keys. No hardware."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples.cfg.kassow_experiments import load_state, resolve_experiment, save_state
from examples.orchestrator_board import DemoBoard, PAGE
from examples.orchestrator_kassow import (
    Args,
    CycleLog,
    GripperOpened,
    Orchestrator,
    ReturnedNearHome,
    State,
    VlaJob,
    station_cfg,
    vla_cfg,
)
from rio.arm_resume import Phase, PhaseKind
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


def _orchestrator(tmp_path, readings, keys, vla_reasons=(), goto_ok=True, goto_script=None, recover=None):
    calls = []
    readings = list(readings)
    vla_reasons = list(vla_reasons)
    script = None if goto_script is None else list(goto_script)

    def ask():
        item = readings.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def goto(path):
        calls.append(("goto", path.name))
        if script is not None:
            return script.pop(0)
        return goto_ok

    def run_vla(job: VlaJob):
        calls.append(("vla", job.experiment, job.instruction))
        return vla_reasons.pop(0)

    orchestrator = Orchestrator(
        ask=ask,
        goto=goto,
        run_vla=run_vla,
        keys=keys,
        recover=recover,
        organize_home=ORGANIZE_HOME,
        spaces=SPACES,
        log=CycleLog(tmp_path / "run"),
    )
    return orchestrator, calls


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]


def test_demo_board_shows_the_jev_reading(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path, [_reading("open", space=2, reason="2 carton(s) loaded")], ScriptedKeys("j"))
    board = DemoBoard(tmp_path / "live")
    orchestrator.board = board
    orchestrator.step()
    orchestrator.step()
    status = json.loads((tmp_path / "live" / "status.json").read_text())
    assert status["state"] == "goto"
    assert status["node"] == "space_move"
    assert status["alarm"] is False
    assert status["title"] == "Moving to the start pose"
    assert status["reading"]["decision"] == "open"
    assert "Space 2" in status["reading"]["line"]
    assert "cartons" in status["reading"]["line"]


def test_board_marks_the_remembered_step_red_during_an_alarm(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path, [], ScriptedKeys())
    orchestrator.job = VlaJob("open", "put the object in space 2", SPACES[2])
    orchestrator.after_goto = State.RUN_VLA
    orchestrator.phase_index = 1
    orchestrator.state = State.WAIT_RESET
    DemoBoard(tmp_path / "live").publish(orchestrator)
    status = json.loads((tmp_path / "live" / "status.json").read_text())
    assert status["alarm"] is True
    assert status["node"] == "open_vla"
    assert "returns to this step" in status["alarm_note"]


def test_leave_lifts_before_the_home_move(tmp_path):
    orchestrator, calls = _orchestrator(tmp_path, [], ScriptedKeys())
    seen = []
    orchestrator.lift = lambda: seen.append("lift") or True
    orchestrator.lift_before_goto = True
    orchestrator.goto_path = ORGANIZE_HOME
    orchestrator.after_goto = State.ASK
    orchestrator.state = State.GOTO
    orchestrator.step()
    assert seen == ["lift"]
    assert calls == [("goto", "home_organize.json")]
    assert orchestrator.state is State.ASK
    assert orchestrator.lift_before_goto is False


def test_open_stops_at_the_waiting_pose_before_the_space(tmp_path):
    wait = Path("home_open_wait.json")
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("open", space=2), _reading("hold", reason="placed")],
        ScriptedKeys("j", "p", "q"),
        vla_reasons=["stop_when"],
    )
    orchestrator.wait_home = wait
    orchestrator.run()
    assert calls == [
        ("goto", "home_open_wait.json"),
        ("goto", "home_open_space2.json"),
        ("vla", "open", "put the object in space 2"),
        ("goto", "home_open_wait.json"),
    ]


def test_open_path_runs_space_n_plus_one_then_waits_for_p(tmp_path):
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("open", space=2), _reading("hold", reason="placed")],
        ScriptedKeys("j", "p", "q"),
        vla_reasons=["stop_when"],
    )
    orchestrator.run()
    assert calls == [
        ("goto", "home_open_space2.json"),
        ("vla", "open", "put the object in space 2"),
        ("goto", "home_organize.json"),
    ]
    events = [json.loads(line)["event"] for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert events == ["reading", "goto", "vla", "lift", "goto", "reading", "hold"]
    assert orchestrator.count_only is True


def test_organize_path_asks_again_after_the_rollout(tmp_path):
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize"), _reading("open", space=1), _reading("hold", reason="placed")],
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


def test_goto_alarm_repeats_the_same_home_after_confirm(tmp_path):
    """The question is whether a latched moveJ is repeated without asking Jev again.

    The first moveJ reports an alarm and the arm is not asked to move until ``c``
    clears it. The second call is the same home file. The cycle counter stays
    at the reading that was already taken.
    """
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize")],
        ScriptedKeys("j", "c"),
        goto_script=["alarm", True],
        recover=lambda: True,
    )
    orchestrator.step()
    orchestrator.step()
    orchestrator.step()
    assert orchestrator.state is State.WAIT_RESET
    assert calls == [("goto", "home_organize.json")]
    assert orchestrator.log.cycle == 1
    orchestrator.step()
    orchestrator.step()
    assert calls == [("goto", "home_organize.json"), ("goto", "home_organize.json")]
    assert orchestrator.state is State.RUN_VLA
    resume = next(event for event in _events(tmp_path) if event["event"] == "resume")
    assert resume["action"] == "retry"
    assert resume["phase"] == "movej"
    assert resume["holding"] is False


def test_vla_alarm_reruns_the_same_job_from_where_it_stopped(tmp_path):
    """An empty gripper repeats the VLA job and does not go back through Jev or home."""
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize")],
        ScriptedKeys("j", "c"),
        vla_reasons=["alarm", "timeout"],
        recover=lambda: True,
    )
    for _ in range(4):
        orchestrator.step()
    assert orchestrator.state is State.WAIT_RESET
    orchestrator.step()
    orchestrator.step()
    assert calls == [
        ("goto", "home_organize.json"),
        ("vla", "organize", ""),
        ("vla", "organize", ""),
    ]
    assert orchestrator.log.cycle == 1
    resume = next(event for event in _events(tmp_path) if event["event"] == "resume")
    assert resume["action"] == "retry"
    assert resume["phase"] == "vla"
    assert resume["holding"] is False


def test_holding_continues_past_the_grasp(tmp_path):
    """A held object does not repeat the grasp. The next phase is the VLA continuation.

    The script stands in for the later pick-and-place arm: approach, grasp, then
    the VLA. The alarm is remembered on the grasp. After ``c`` the only call is
    the VLA, and the logged phase is not the grasp.
    """
    orchestrator, calls = _orchestrator(
        tmp_path,
        [],
        ScriptedKeys("c"),
        vla_reasons=["timeout"],
        recover=lambda: True,
    )
    orchestrator.job = VlaJob("organize", "", ORGANIZE_HOME)
    orchestrator.phases = (
        Phase("approach", PhaseKind.MOVE),
        Phase("grasp", PhaseKind.GRASP),
        Phase("vla", PhaseKind.MOVE),
    )
    orchestrator.phase_index = 1
    orchestrator.alarm_holding = True
    orchestrator.state = State.WAIT_RESET
    orchestrator.step()
    assert orchestrator.state is State.RUN_VLA
    assert calls == []
    resume = next(event for event in _events(tmp_path) if event["event"] == "resume")
    assert resume["action"] == "continue"
    assert resume["phase"] == "vla"
    assert resume["holding"] is True
    orchestrator.step()
    assert calls == [("vla", "organize", "")]


def test_quit_during_alarm_does_not_move(tmp_path):
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize")],
        ScriptedKeys("j", "q"),
        goto_script=["alarm"],
    )
    orchestrator.run()
    assert orchestrator.state is State.DONE
    assert calls == [("goto", "home_organize.json")]
    assert "resume" not in {event["event"] for event in _events(tmp_path)}


def test_failed_clear_stays_in_wait_reset(tmp_path):
    """``c`` with the alarm still latched must not start the remembered move."""
    orchestrator, calls = _orchestrator(
        tmp_path,
        [_reading("organize")],
        ScriptedKeys("j", "c", "q"),
        goto_script=["alarm"],
        recover=lambda: False,
    )
    orchestrator.step()
    orchestrator.step()
    orchestrator.step()
    orchestrator.step()
    assert orchestrator.state is State.WAIT_RESET
    assert calls == [("goto", "home_organize.json")]
    assert [event["event"] for event in _events(tmp_path)].count("recover_failed") == 1
    orchestrator.step()
    assert orchestrator.state is State.DONE


def test_organize_vla_stops_only_after_leaving_and_returning_home():
    home = np.array([0.0, 0.0, 0.4])
    near = ReturnedNearHome(home, radius_m=0.20)
    assert not near.consider(np.stack([home, home + np.array([0.05, 0.0, 0.0])]))
    assert not near.consider(np.stack([home, home + np.array([0.30, 0.0, 0.0])]))
    assert near.consider(np.stack([home + np.array([0.30, 0.0, 0.0]), home + np.array([0.05, 0.0, 0.0])]))
    with pytest.raises(ValueError):
        ReturnedNearHome(home, radius_m=0.0)


def test_run_diagram_names_the_reorganize_loop():
    assert ">Waiting<" in PAGE
    assert ">Ask JEV<" in PAGE
    assert ">Error<" in PAGE
    assert ">Move to reorganize<" in PAGE
    assert ">Organize<" in PAGE
    assert ">Move to space<" in PAGE
    assert ">Open space<" in PAGE
    assert ">Hold<" in PAGE
    assert ">Leave<" in PAGE
    assert "To fold" not in PAGE
    assert "Unsure" not in PAGE
    assert "move To" not in PAGE
    assert "Move To" not in PAGE


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
