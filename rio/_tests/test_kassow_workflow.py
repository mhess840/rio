# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Kassow tools for v3 workflows. No arm, no camera, no network."""

import copy
import json
from pathlib import Path

import pytest

from rio._tests.test_kassow_graph import _box_answers, _table
from rio.apps.kassow.graph.cell import cell_tools
from rio.apps.kassow.graph.run import Args, main
from rio.graph import Executor, errors, load_workflow, parse_workflow, validate_workflow

pytestmark = pytest.mark.unit

CYLINDER = Path(__file__).resolve().parents[1] / "apps/kassow/graph/workflows/cylinder_to_plate.json"


class Arm:
    """Fake skills: record calls, answer True unless told otherwise."""

    def __init__(self, **results) -> None:
        self.calls: list[str] = []
        self.results = results

    def _do(self, name: str):
        self.calls.append(name)
        return self.results.get(name, True)

    def move_to(self, pose):
        return self._do(f"move_to {Path(pose).name}")

    def lift(self, meters=0.1):
        return self._do("lift")

    def pick(self):
        return self._do("pick")

    def place(self):
        return self._do("place")

    def gripper(self, state):
        return self._do(f"gripper {state}")

    def pick_at(self, target, *, approach_m, z_offset_m):
        return self._do("pick_at")

    def place_at(self, target, *, approach_m, z_offset_m):
        return self._do("place_at")

    def move_above(self, target, *, height_m):
        return self._do("move_above")


def jev(*_):
    return _box_answers(0.5, 0.5)


def tools(arm=None, **kwargs):
    return cell_tools(arm or Arm(), jev, table=_table(), pose_exists=lambda name: True, **kwargs)


def test_shipped_cylinder_workflow_validates():
    issues = validate_workflow(load_workflow(CYLINDER), tools())
    assert errors(issues) == []


def test_cylinder_workflow_runs_on_fake_arm():
    arm = Arm()
    result = Executor(load_workflow(CYLINDER), tools(arm)).run()
    assert result.status == "success"
    assert arm.calls == ["move_to home_organize.json", "pick_at", "place_at", "move_to home_organize.json"]


def test_alarm_during_pick_goes_to_abort_and_recovers():
    arm = Arm(pick_at="KORD alarm 42")
    result = Executor(load_workflow(CYLINDER), tools(arm)).run()
    assert result.end == "abort"
    assert arm.calls[-2:] == ["gripper open", "move_to home_organize.json"]
    error = next(e for e in result.events if e["event"] == "tool_error")
    assert "KORD alarm 42" in error["error"]


def test_missed_grasp_routes_to_abort():
    result = Executor(load_workflow(CYLINDER), tools(Arm(pick_at=False))).run()
    assert result.end == "abort"


def test_cell_checks_literal_inputs():
    raw = json.loads(CYLINDER.read_text())
    raw["subgraphs"]["grasp_sg"]["nodes"]["pick"]["inputs"]["approach_m"] = 0.9
    raw["subgraphs"]["park_sg"]["nodes"]["move"]["inputs"]["pose"] = "nowhere.json"
    registry = cell_tools(Arm(), jev, table=_table(), pose_exists=lambda name: name != "nowhere.json")
    found = " | ".join(str(issue) for issue in errors(validate_workflow(parse_workflow(raw), registry)))
    assert "approach_m must be a number" in found
    assert "'nowhere.json' is not a home file" in found


OPERATOR = {
    "version": 3,
    "nodes": {
        "ask": {
            "type": "tool",
            "tool": "operator.ask",
            "inputs": {"question": "Is the crate empty?", "answers": ["empty", "full"]},
        },
        "empty": {"type": "end", "status": "success"},
        "full": {"type": "end", "status": "failure"},
    },
    "edges": [["START", "ask"]],
    "conditional_edges": {"ask": {"router_field": "exit", "mapping": {"empty": "empty", "full": "full"}}},
}


def test_operator_answer_is_the_exit():
    asked = []

    def ask_operator(node, question, answers):
        asked.append((node, question, answers))
        return "full"

    result = Executor(parse_workflow(OPERATOR), tools(ask_operator=ask_operator)).run()
    assert result.end == "full"
    assert asked == [("ask", "Is the crate empty?", ["empty", "full"])]


def test_operator_answers_are_checked():
    raw = copy.deepcopy(OPERATOR)
    raw["nodes"]["ask"]["inputs"]["answers"] = ["only"]
    found = [str(issue) for issue in errors(validate_workflow(parse_workflow(raw), tools()))]
    assert any("2 to 9 distinct" in line for line in found)


def test_jev_choice_routes_on_the_criterion():
    raw = {
        "version": 3,
        "nodes": {
            "which": {
                "type": "tool",
                "tool": "jev.ask",
                "inputs": {
                    "question": {
                        "type": "choice",
                        "instructions": "Which space of `crate` is free?",
                        "criteria": {"one": "space 1 is free", "two": "space 2 is free"},
                    }
                },
            },
            "one": {"type": "end", "status": "success"},
            "two": {"type": "end", "status": "success"},
        },
        "edges": [["START", "which"]],
        "conditional_edges": {"which": {"router_field": "exit", "mapping": {"one": "one", "two": "two"}}},
    }

    def ask(node, questions, note):
        return {"q": {"type": "choice", "choice": "two"}}

    result = Executor(parse_workflow(raw), cell_tools(Arm(), ask, pose_exists=lambda name: True)).run()
    assert result.end == "two"


def test_main_dry_runs_a_v3_file_and_logs_events(tmp_path, monkeypatch):
    monkeypatch.setattr("rio.apps.kassow.graph.run_v3.experiments_root", lambda: tmp_path)
    table = tmp_path / "table_map.json"
    _table().save(table)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"cylinder/find": _box_answers(0.4, 0.4), "plate/find": _box_answers(0.6, 0.6)}))
    assert main(Args(command="check", graph=CYLINDER, table=table)) == "ok"
    assert main(Args(command="run", graph=CYLINDER, table=table, script=script)) == "success"
    (run,) = [path for path in (tmp_path / "graph").iterdir() if path.is_dir()]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_start" and events[-1]["status"] == "success"
    assert json.loads((run / "workflow.json").read_text())["version"] == 3


def test_loop_over_items_through_subgraphs():
    raw = json.loads(CYLINDER.read_text())
    raw["conditional_edges"]["cylinder"]["mapping"]["not_found"] = "park"
    raw["conditional_edges"]["transport"]["mapping"]["placed"] = "cylinder"
    seen = []

    def ask(node, questions, note):
        seen.append(node)
        cylinders = sum(1 for n in seen if n == "cylinder/find")
        if node == "cylinder/find" and cylinders > 2:
            return _box_answers(0.4, 0.4, visible=0.0, confidence=0.1)
        return _box_answers(0.4, 0.4)

    arm = Arm()
    registry = cell_tools(arm, ask, table=_table(), pose_exists=lambda name: True)
    workflow = parse_workflow(raw)
    assert errors(validate_workflow(workflow, registry)) == []
    result = Executor(workflow, registry).run()
    assert result.end == "done"
    assert arm.calls.count("pick_at") == 2 and arm.calls.count("place_at") == 2
