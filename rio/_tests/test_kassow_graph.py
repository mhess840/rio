# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-as-Policy check and walk. No arm, no camera, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from rio.apps.kassow.graph import EXAMPLE_PATH, GraphError, check_graph, example_graph, load_graph, require_graph
from rio.apps.kassow.graph.model import Graph
from rio.apps.kassow.graph.run import Args, main
from rio.apps.kassow.graph.walk import WalkLog, walk

pytestmark = pytest.mark.unit


class FakeSkills:
    """Records skill calls. ``results`` are returned in order; the default is success."""

    def __init__(self, *results: object):
        self.results = list(results)
        self.calls: list[tuple] = []

    def _pop(self) -> object:
        return self.results.pop(0) if self.results else True

    def move_to(self, pose: Path) -> object:
        self.calls.append(("move_to", pose.name))
        return self._pop()

    def lift(self, meters: float = 0.1) -> object:
        self.calls.append(("lift", meters))
        return self._pop()

    def pick(self) -> object:
        self.calls.append(("pick",))
        return self._pop()

    def place(self) -> object:
        self.calls.append(("place",))
        return self._pop()


class ScriptedKeys:
    def __init__(self, *presses: str):
        self.presses = [set(item) for item in presses]

    def pressed(self) -> set[str]:
        return self.presses.pop(0) if self.presses else set()


def _graph(nodes: dict, start: str = "a", max_steps: int = 8, task: str = "trial") -> Graph:
    return Graph(task=task, start=start, max_steps=max_steps, nodes=nodes)


def test_the_shipped_example_checks_and_names_the_organize_home():
    graph = example_graph()
    assert check_graph(graph) == []
    assert graph.path == EXAMPLE_PATH
    assert graph.nodes["go_home"]["pose"] == "home_organize.json"


def test_a_missing_pose_and_an_unreachable_node_are_rejected():
    graph = _graph(
        {
            "go": {"kind": "move_to", "pose": "missing.json", "ok": "done", "fail": "done"},
            "done": {"kind": "stop", "reason": "end"},
            "orphan": {"kind": "stop", "reason": "unused"},
        },
        start="go",
    )
    problems = check_graph(graph, pose_exists=lambda name: False)
    assert any("missing.json" in item for item in problems)
    assert any("orphan" in item for item in problems)


def test_a_clear_view_moves_home_and_a_blocked_view_stops(tmp_path):
    graph = example_graph()
    skills = FakeSkills()
    seen = walk(
        graph,
        skills,
        lambda node, questions, note: {"view_clear": {"type": "noul", "noul": 0.91}},
        WalkLog(tmp_path / "clear"),
    )
    assert seen == "at the organize home"
    assert skills.calls == [("move_to", "home_organize.json")]
    blocked = walk(
        graph,
        FakeSkills(),
        lambda node, questions, note: {"view_clear": {"type": "noul", "noul": 0.2}},
        WalkLog(tmp_path / "blocked"),
    )
    assert blocked == "the crate is not fully visible"


def test_a_failed_move_takes_the_fail_edge_including_an_alarm(tmp_path):
    graph = example_graph()
    skills = FakeSkills("alarm")
    reason = walk(
        graph,
        skills,
        lambda node, questions, note: {"view_clear": {"type": "noul", "noul": 0.95}},
        WalkLog(tmp_path / "alarm"),
    )
    assert reason == "moveJ to the organize home failed"
    events = [json.loads(line) for line in (tmp_path / "alarm" / "events.jsonl").read_text().splitlines()]
    assert events[1]["outcome"] == "fail"
    assert events[1]["detail"] == "alarm"


def test_choice_and_score_follow_their_own_edges(tmp_path):
    graph = _graph(
        {
            "ask": {
                "kind": "jev",
                "questions": {
                    "where": {
                        "type": "choice",
                        "instructions": "Which side of `crate` holds the carton?",
                        "criteria": {"left": "the left half", "right": "the right half"},
                    },
                    "how_full": {
                        "type": "score",
                        "instructions": "How full is the crate in `crate`?",
                        "criteria": ["empty", "one carton", "several cartons"],
                    },
                },
                "route": "where",
                "when": {"left": "fullness", "right": "right_stop"},
                "else": "right_stop",
            },
            "fullness": {
                "kind": "jev",
                "questions": {
                    "how_full": {
                        "type": "score",
                        "instructions": "How full is the crate in `crate`?",
                        "criteria": ["empty", "one carton", "several cartons"],
                    }
                },
                "route": "how_full",
                "threshold": 1.5,
                "high": "full",
                "low": "empty",
            },
            "right_stop": {"kind": "stop", "reason": "right"},
            "full": {"kind": "stop", "reason": "full"},
            "empty": {"kind": "stop", "reason": "empty"},
        },
        start="ask",
    )
    require_graph(graph, pose_exists=lambda name: True)
    answers = {
        "ask": {"where": {"type": "choice", "choice": "left", "probabilities": {"left": 0.8, "right": 0.2}}},
        "fullness": {"how_full": {"type": "score", "score": 2.0, "confidence": 0.9}},
    }
    reason = walk(
        graph,
        FakeSkills(),
        lambda node, questions, note: answers[node],
        WalkLog(tmp_path / "score"),
    )
    assert reason == "full"


def test_a_loop_stops_at_max_steps_and_q_stops_before_the_next_node(tmp_path):
    graph = _graph(
        {"again": {"kind": "lift", "meters": 0.05, "ok": "again", "fail": "again"}},
        start="again",
        max_steps=3,
    )
    require_graph(graph, pose_exists=lambda name: True)
    skills = FakeSkills()
    reason = walk(graph, skills, lambda node, questions, note: {}, WalkLog(tmp_path / "loop"))
    assert reason == "stopped after 3 steps"
    assert skills.calls == [("lift", 0.05)] * 3
    quit_reason = walk(
        graph,
        FakeSkills(),
        lambda node, questions, note: {},
        WalkLog(tmp_path / "quit"),
        keys=ScriptedKeys("q"),
    )
    assert quit_reason == "quit"


def test_check_command_does_not_walk(tmp_path, monkeypatch):
    monkeypatch.setattr("rio.apps.kassow.graph.run.experiments_root", lambda: tmp_path)
    assert main(Args(command="check")) == "ok"


def test_dry_run_of_the_example_uses_the_script_and_writes_a_log(tmp_path, monkeypatch):
    monkeypatch.setattr("rio.apps.kassow.graph.run.experiments_root", lambda: tmp_path)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"look": {"view_clear": {"type": "noul", "noul": 0.88}}}))
    reason = main(Args(command="run", dry_run=True, script=script))
    assert reason == "at the organize home"
    logs = list((tmp_path / "graph").glob("*/events.jsonl"))
    assert len(logs) == 1
    assert [json.loads(line)["node"] for line in logs[0].read_text().splitlines()] == ["look", "go_home", "done"]


def test_a_sample_stores_the_tcp_and_a_jpeg(tmp_path):
    from rio_hw import time

    from rio.apps.kassow.graph.calibrate import CAMERA_KEYS, save_sample

    class Camera:
        def get_state(self):
            return {"color": np.full((480, 640, 3), 80, dtype=np.uint8), "timestamp": time.now()}

    class Env:
        cameras = {key: Camera() for key, _name in CAMERA_KEYS}

    sample = save_sample(Env(), np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]), np.zeros(7), root=tmp_path)
    assert sample.name == "001"
    assert sorted(path.name for path in sample.iterdir()) == ["jev.jpg", "overview.jpg"]
    line = json.loads((tmp_path / "poses.jsonl").read_text().splitlines()[0])
    assert line["xyz"] == [0.1, 0.2, 0.3]
    assert set(line["images"]) == {"overview", "jev"}


def test_a_graph_file_without_a_task_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"start": "a", "max_steps": 1, "nodes": {"a": {"kind": "stop"}}}))
    with pytest.raises(GraphError, match="task"):
        load_graph(path)
