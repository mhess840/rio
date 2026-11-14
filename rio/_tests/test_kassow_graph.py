# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-as-Policy check and walk. No arm, no camera, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from rio.apps.kassow.graph import EXAMPLE_PATH, GraphError, check_graph, example_graph, load_graph, require_graph
from rio.apps.kassow.graph.model import PICK_EXAMPLE_PATH, Graph
from rio.apps.kassow.graph.run import Args, main
from rio.apps.kassow.graph.table_map import TableMap, Target
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

    def pick_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> object:
        self.calls.append(("pick_at", round(target.x, 3), round(target.y, 3), approach_m, z_offset_m))
        return self._pop()

    def place_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> object:
        self.calls.append(("place_at", target.name, approach_m, z_offset_m))
        return self._pop()

    def move_above(self, target: Target, *, height_m: float) -> object:
        self.calls.append(("move_above", target.name, height_m))
        return self._pop()

    def gripper(self, state: str) -> object:
        self.calls.append(("gripper", state))
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


def test_a_sample_stores_the_tcp_first_and_the_photos_on_the_next_press(tmp_path):
    from rio_hw import time

    from rio.apps.kassow.graph.calibrate import CAMERA_KEYS, save_sample
    from rio.apps.kassow.graph.table_map import load_samples

    class Camera:
        def get_state(self):
            return {"color": np.full((480, 640, 3), 80, dtype=np.uint8), "timestamp": time.now()}

    class Env:
        cameras = {key: Camera() for key, _name in CAMERA_KEYS}

    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    sample = save_sample(Env(), pose, np.zeros(7), root=tmp_path)
    assert sample.name == "001"
    assert sorted(path.name for path in sample.iterdir()) == ["pose.json"]
    assert load_samples(tmp_path) == []
    assert save_sample(Env(), pose + 1.0, np.ones(7), root=tmp_path) == sample
    assert sorted(path.name for path in sample.iterdir()) == ["jev.jpg", "overview.jpg", "pose.json"]
    record = json.loads((sample / "pose.json").read_text())
    assert record["xyz"] == [0.1, 0.2, 0.3]
    assert [s.name for s in load_samples(tmp_path)] == ["001"]
    assert save_sample(Env(), pose, np.zeros(7), root=tmp_path).name == "002"


def test_calibration_skips_leftover_folders_without_a_pose(tmp_path):
    from rio_hw import time

    from rio.apps.kassow.graph.calibrate import CAMERA_KEYS, save_sample

    class Camera:
        def get_state(self):
            return {"color": np.full((480, 640, 3), 80, dtype=np.uint8), "timestamp": time.now()}

    class Env:
        cameras = {key: Camera() for key, _name in CAMERA_KEYS}

    # Old one-key layout: numbered folders with photos, the poses in poses.jsonl.
    for name in ("001", "002", "003"):
        (tmp_path / name).mkdir()
    (tmp_path / "poses.jsonl").write_text("{}\n")
    assert save_sample(Env(), np.zeros(6), np.zeros(7), root=tmp_path).name == "004"


def test_a_graph_file_without_a_task_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"start": "a", "max_steps": 1, "nodes": {"a": {"kind": "stop"}}}))
    with pytest.raises(GraphError, match="task"):
        load_graph(path)


# Image fractions to base x, y: about 40 cm of table, seen with a mild perspective.
TRUE_H = np.array([[0.40, 0.02, -0.70], [0.01, 0.38, 0.30], [0.05, -0.03, 1.0]])
GRASP_Z = 0.25
DOWN = [np.pi, 0.0, 0.0]


def _grid(n: int = 3) -> np.ndarray:
    values = np.linspace(0.15, 0.85, n)
    return np.array([[u, v] for u in values for v in values])


def _table(margin_m: float = 0.02):
    from rio.apps.kassow.graph.table_map import apply_homography, convex_hull
    from rio.apps.kassow.hardware import JEV_SERIAL

    xy = apply_homography(TRUE_H, np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]))
    return TableMap(
        camera="jev",
        serial=JEV_SERIAL,
        homography=TRUE_H,
        plane=np.array([0.0, 0.0, GRASP_Z]),
        rotvec=np.array(DOWN),
        hull=convex_hull(xy),
        seeds_xy=xy,
        seeds_q=np.zeros((4, 7)),
        margin_m=margin_m,
        stats={"n": 4, "loo_max_m": 0.001},
    )


def _write_samples(root: Path, uv: np.ndarray, jpeg: bytes = b"jpeg") -> dict:
    from rio.apps.kassow.graph.table_map import apply_homography

    labels = {}
    for index, ((x, y), (u, v)) in enumerate(zip(apply_homography(TRUE_H, uv), uv, strict=True), start=1):
        folder = root / f"{index:03d}"
        folder.mkdir(parents=True)
        (folder / "pose.json").write_text(json.dumps({"eef_pose": [x, y, GRASP_Z, *DOWN], "joint_q": [0.0] * 7}))
        for name in ("overview", "jev"):
            (folder / f"{name}.jpg").write_bytes(jpeg)
        labels[folder.name] = {"u": float(u), "v": float(v), "ok": True}
    return labels


def _box_answers(u: float, v: float, visible: float = 0.95, confidence: float = 0.9, half: float = 0.03) -> dict:
    return {
        "visible": {"type": "noul", "noul": visible},
        "where": {
            "type": "box",
            "box": {"left": u - half, "right": u + half, "top": v - half, "bottom": v + half},
            "confidence": confidence,
        },
    }


def test_the_homography_fit_recovers_a_known_map_and_rejects_a_line():
    from rio.apps.kassow.graph.table_map import apply_homography, fit_homography

    uv = _grid()
    assert np.allclose(fit_homography(uv, apply_homography(TRUE_H, uv)), TRUE_H, atol=1e-9)
    line = np.array([[0.1 * i, 0.1 * i] for i in range(1, 7)])
    with pytest.raises(ValueError, match="collinear"):
        fit_homography(line, line * 0.5)


def test_the_calibrated_area_is_the_hull_plus_its_margin():
    from rio.apps.kassow.graph.table_map import convex_hull, inside_hull

    hull = convex_hull(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]))
    assert len(hull) == 4
    assert inside_hull(hull, np.array([0.5, 0.5]))
    assert inside_hull(hull, np.array([1.01, 0.5]), margin=0.02)
    assert not inside_hull(hull, np.array([1.03, 0.5]), margin=0.02)


def test_fit_validates_with_leave_one_out_and_round_trips(tmp_path):
    from rio.apps.kassow.graph.table_map import fit_table_map, load_samples, map_problems

    labels = _write_samples(tmp_path, _grid())
    table = fit_table_map(load_samples(tmp_path), labels, camera="jev", serial="123")
    assert table.stats["n"] == 9
    assert table.stats["loo_max_m"] < 1e-6
    assert np.allclose(table.plane, [0.0, 0.0, GRASP_Z], atol=1e-9)
    assert map_problems(table) == []
    table.save(tmp_path / "table_map.json")
    loaded = TableMap.load(tmp_path / "table_map.json")
    assert np.allclose(loaded.to_xy(0.4, 0.6), table.to_xy(0.4, 0.6))
    assert loaded.z_at(-0.5, 0.5) == pytest.approx(GRASP_Z)

    labels["005"]["u"] += 0.1
    bad = fit_table_map(load_samples(tmp_path), labels, camera="jev", serial="123")
    assert any("leave-one-out" in problem for problem in map_problems(bad))
    with pytest.raises(ValueError, match="at least"):
        fit_table_map(load_samples(tmp_path)[:5], labels, camera="jev", serial="123")


def test_label_asks_once_per_photo_and_marks_unsure_boxes(tmp_path):
    from rio.apps.kassow.graph.table_map import label_samples
    from rio.jev import encode_jpeg

    _write_samples(tmp_path, _grid(2), jpeg=encode_jpeg(np.zeros((480, 640, 3), dtype=np.uint8)))
    replies = iter(
        [_box_answers(0.3, 0.4), _box_answers(0.5, 0.5, visible=0.2), _box_answers(0.6, 0.7), _box_answers(0.2, 0.2)]
    )
    asked: list[dict] = []

    def ask(jpeg: bytes, questions: dict) -> dict:
        asked.append(questions)
        return next(replies)

    labels = label_samples(tmp_path, ask, "the red cube")
    assert len(asked) == 4
    assert "the red cube" in asked[0]["where"]["instructions"]
    assert labels["001"]["u"] == pytest.approx(0.3) and labels["001"]["ok"]
    assert not labels["002"]["ok"]
    assert json.loads((tmp_path / "labels_jev.json").read_text()) == labels
    label_samples(tmp_path, ask, "the red cube")
    assert len(asked) == 4


@pytest.mark.parametrize(
    "box",
    [
        {"left": 0.6, "right": 0.4, "top": 0.1, "bottom": 0.2},
        {"left": 0.1, "right": 1.2, "top": 0.1, "bottom": 0.2},
        {"left": float("nan"), "right": 0.4, "top": 0.1, "bottom": 0.2},
        {"left": 0.1, "right": 0.4, "top": 0.1},
    ],
)
def test_a_malformed_box_is_rejected(box):
    from rio.apps.kassow.graph.locate import read_box

    with pytest.raises(ValueError):
        read_box({"where": {"type": "box", "box": box, "confidence": 0.9}})


def test_the_pick_example_checks_and_bad_locate_or_pick_nodes_are_rejected():
    assert check_graph(load_graph(PICK_EXAMPLE_PATH)) == []
    graph = _graph(
        {
            "find": {"kind": "locate", "object": "the `cube`", "store": "Cube", "found": "grab", "missing": "nowhere"},
            "grab": {"kind": "pick_at", "target": "ball", "z_offset_m": -0.02, "approach_m": 0.5, "ok": "end", "fail": "end"},
            "end": {"kind": "stop", "reason": "end"},
        },
        start="find",
    )
    problems = "\n".join(check_graph(graph))
    for expected in ("backticks", "store", "edge 'missing'", "target", "z_offset_m", "approach_m"):
        assert expected in problems


def test_locate_stores_the_target_and_pick_at_uses_it(tmp_path):
    from rio.apps.kassow.graph.table_map import apply_homography

    graph = load_graph(PICK_EXAMPLE_PATH)
    skills = FakeSkills()
    asked: list[dict] = []

    def ask(node: str, questions: dict, note: str) -> dict:
        asked.append(questions)
        return _box_answers(0.5, 0.5)

    assert walk(graph, skills, ask, WalkLog(tmp_path / "ok"), table=_table()) == "holding the cube"
    x, y = apply_homography(TRUE_H, [[0.5, 0.5]])[0]
    assert skills.calls == [("pick_at", round(float(x), 3), round(float(y), 3), 0.1, 0.0)]
    assert set(asked[0]) == {"visible", "where"}
    assert "the cube" in asked[0]["where"]["instructions"]


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        (_box_answers(0.5, 0.5, visible=0.2), "no cube inside the calibrated area"),
        (_box_answers(0.5, 0.5, confidence=0.3), "no cube inside the calibrated area"),
        (_box_answers(0.99, 0.99, half=0.005), "no cube inside the calibrated area"),
        ({"visible": {"type": "noul", "noul": 0.9}, "where": {"type": "noul", "noul": 0.9}}, "Jev did not answer"),
    ],
)
def test_locate_misses_an_unsure_or_far_object_and_never_picks(tmp_path, answers, reason):
    skills = FakeSkills()
    graph = load_graph(PICK_EXAMPLE_PATH)
    assert walk(graph, skills, lambda node, questions, note: answers, WalkLog(tmp_path), table=_table()) == reason
    assert skills.calls == []


def test_a_declined_pick_ends_the_walk_and_locate_needs_a_table(tmp_path):
    graph = load_graph(PICK_EXAMPLE_PATH)
    skills = FakeSkills()
    messages: list[str] = []

    def decline(message: str) -> bool:
        messages.append(message)
        return False

    ask = lambda node, questions, note: _box_answers(0.5, 0.5)
    assert walk(graph, skills, ask, WalkLog(tmp_path / "no"), table=_table(), confirm=decline) == "quit"
    assert skills.calls == []
    assert "pick_at the cube" in messages[0]
    with pytest.raises(GraphError, match="table map"):
        walk(graph, skills, ask, WalkLog(tmp_path / "none"))


def test_run_needs_the_jev_table_map_for_a_locate_graph(tmp_path, monkeypatch):
    monkeypatch.setattr("rio.apps.kassow.graph.run.experiments_root", lambda: tmp_path)
    monkeypatch.setattr("rio.apps.kassow.graph.run.table_map_path", lambda: tmp_path / "missing.json")
    with pytest.raises(GraphError, match="missing"):
        main(Args(command="check", graph=PICK_EXAMPLE_PATH))
    path = tmp_path / "table_map.json"
    _table().save(path)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"find": _box_answers(0.5, 0.5)}))
    assert main(Args(command="run", graph=PICK_EXAMPLE_PATH, table=path, script=script)) == "holding the cube"
    other = TableMap.load(path)
    TableMap(**{**other.__dict__, "serial": "000000000000"}).save(path)
    with pytest.raises(GraphError, match="not the Jev camera"):
        main(Args(command="check", graph=PICK_EXAMPLE_PATH, table=path))


PICK_PLACE_PATH = PICK_EXAMPLE_PATH.with_name("pick_place.json")


def test_every_shipped_example_checks():
    for path in sorted(EXAMPLE_PATH.parent.glob("*.json")):
        assert check_graph(load_graph(path)) == [], path.name


def test_pick_place_locates_both_then_picks_and_places(tmp_path):
    answers = {"find_cube": _box_answers(0.3, 0.4), "find_plate": _box_answers(0.7, 0.6)}
    skills = FakeSkills()
    reason = walk(
        load_graph(PICK_PLACE_PATH),
        skills,
        lambda node, questions, note: answers[node],
        WalkLog(tmp_path),
        table=_table(),
    )
    assert reason == "the cube is on the plate"
    assert [call[0] for call in skills.calls] == ["pick_at", "place_at"]
    assert skills.calls[1] == ("place_at", "the plate", 0.1, 0.02)


def test_generic_skill_and_vla_nodes_are_checked():
    nodes = {
        "hand": {"kind": "gripper", "state": "half", "ok": "end", "fail": "end"},
        "hover": {"kind": "move_above", "target": "cup", "height_m": 0.5, "ok": "end", "fail": "end"},
        "put": {"kind": "place_at", "target": "cup", "z_offset_m": 0.3, "ok": "end", "fail": "end"},
        "nope": {"kind": "vla", "experiment": "dance", "ok": "end", "fail": "end"},
        "pick_vla": {"kind": "vla", "experiment": "pick", "ok": "end", "fail": "end"},
        "open_vla": {"kind": "vla", "experiment": "open", "timeout_s": 1, "ok": "end", "fail": "end"},
        "end": {"kind": "stop", "reason": "end"},
    }
    start = {"kind": "lift", "ok": "hand", "fail": "hover"}
    chain = {"hand": "put", "put": "nope", "nope": "pick_vla", "pick_vla": "open_vla"}
    for name, nxt in chain.items():
        nodes[name] = {**nodes[name], "ok": nxt}
    problems = "\n".join(check_graph(_graph({"go": start, **nodes}, start="go"), pose_exists=lambda name: True))
    for expected in (
        "gripper state",
        "move_above target",
        "height_m",
        "place_at target",
        "'put' z_offset_m",
        "experiment must be one of",
        "needs overview camera",
        "timeout_s",
        "Pass --instruction",
    ):
        assert expected in problems, expected
    good = _graph(
        {
            "tidy": {"kind": "vla", "experiment": "organize", "timeout_s": 90, "ok": "space", "fail": "end"},
            "space": {
                "kind": "vla",
                "experiment": "open",
                "instruction": "put the object in space 2",
                "ok": "end",
                "fail": "end",
            },
            "end": {"kind": "stop", "reason": "end"},
        },
        start="tidy",
    )
    assert check_graph(good) == []


def test_a_vla_node_runs_the_injected_rollout_and_routes_on_its_result(tmp_path):
    graph = _graph(
        {
            "tidy": {"kind": "vla", "experiment": "organize", "timeout_s": 90, "ok": "done", "fail": "held"},
            "done": {"kind": "stop", "reason": "done"},
            "held": {"kind": "stop", "reason": "held"},
        },
        start="tidy",
    )
    calls: list[tuple] = []

    def run_vla(experiment: str, instruction: str, timeout_s: float) -> object:
        calls.append((experiment, instruction, timeout_s))
        return True

    assert walk(graph, FakeSkills(), lambda *a: {}, WalkLog(tmp_path / "ok"), run_vla=run_vla) == "done"
    assert calls == [("organize", "", 90.0)]
    assert walk(graph, FakeSkills(), lambda *a: {}, WalkLog(tmp_path / "f"), run_vla=lambda *a: "aborted") == "held"
    declined = walk(graph, FakeSkills(), lambda *a: {}, WalkLog(tmp_path / "q"), run_vla=run_vla, confirm=lambda m: False)
    assert declined == "quit"
    assert len(calls) == 1
    with pytest.raises(GraphError, match="VLA runner"):
        walk(graph, FakeSkills(), lambda *a: {}, WalkLog(tmp_path / "none"))


def test_the_vla_adapter_moves_home_first_and_judges_like_the_orchestrator(tmp_path, monkeypatch):
    from rio.apps.kassow.graph import vla

    reasons = iter(["timeout", "timeout", "stop_when"])
    jobs: list = []

    def orchestrator_run_vla(args, env, keys, log):
        def run(job):
            jobs.append((job.experiment, job.instruction, job.home.name, args.organize_timeout_s, args.open_timeout_s))
            return next(reasons)

        return run

    monkeypatch.setattr("rio.apps.kassow.orchestrator.actions.make_run_vla", orchestrator_run_vla)
    skills = FakeSkills(True, True, True, "alarm")
    run = vla.make_run_vla(None, None, skills, tmp_path, device="cuda:1", single_slot=False, mw="Thread")
    assert run("organize", "", 45.0) is True
    assert run("open", "put the object in space 1", 30.0) == "timeout"
    assert run("open", "put the object in space 1", 30.0) is True
    assert run("organize", "", 45.0) == "start pose not reached: alarm"
    assert [call[1] for call in skills.calls] == [
        "home_organize.json",
        "home_open_space1.json",
        "home_open_space1.json",
        "home_organize.json",
    ]
    assert jobs[0] == ("organize", "", "home_organize.json", 45.0, 45.0)
    assert len(jobs) == 3


def test_describe_covers_every_checked_kind_and_the_registry(tmp_path, monkeypatch):
    from rio.apps.kassow.graph.describe import describe
    from rio.apps.kassow.graph.model import JEV_KIND, LOCATE_KIND, SKILL_KINDS, STOP_KIND, VLA_KIND

    monkeypatch.setattr("rio.apps.kassow.graph.describe.table_map_path", lambda: tmp_path / "table_map.json")
    info = json.loads(json.dumps(describe()))
    assert set(info["nodes"]) == {*SKILL_KINDS, JEV_KIND, LOCATE_KIND, VLA_KIND, STOP_KIND}
    assert info["vla"]["pick"]["usable"] is False
    assert info["vla"]["organize"]["usable"] is True
    assert "put the object in space 2" in info["vla"]["open"]["instructions"]
    assert "home_organize.json" in info["poses"]
    assert info["table_map"]["ready"] is False
    _table().save(tmp_path / "table_map.json")
    assert describe()["table_map"]["ready"] is True


def test_ask_photo_validates_questions_and_places_a_located_object():
    from rio.apps.kassow.graph.observe import ask_photo
    from rio.apps.kassow.graph.table_map import apply_homography

    asked: list[dict] = []

    def ask(jpeg: bytes, questions: dict) -> dict:
        asked.append(questions)
        return _box_answers(0.5, 0.5)

    result = ask_photo(b"jpeg", ask, locate="the red cube", table=_table())
    x, _y = apply_homography(TRUE_H, [[0.5, 0.5]])[0]
    assert result["target"]["x"] == pytest.approx(x, abs=1e-4)
    assert result["target"]["inside_area"] is True
    assert result["target"]["grasp_z"] == pytest.approx(GRASP_Z)
    assert "target" not in ask_photo(b"jpeg", ask, locate="the red cube")
    question = {"full": {"type": "noul", "instructions": "Is the plate in `crate` full?"}}
    assert ask_photo(b"jpeg", ask, questions=question)["answers"]["visible"]["noul"] == 0.95
    for bad in ({"questions": {"x": {"type": "guess", "instructions": "?"}}}, {"locate": "the `cube`"}, {}):
        with pytest.raises(ValueError):
            ask_photo(b"jpeg", ask, **bad)
    with pytest.raises(ValueError, match="either"):
        ask_photo(b"jpeg", ask, questions=question, locate="the cube")


def test_describe_and_ask_print_json_without_a_graph(tmp_path, monkeypatch, capsys):
    from rio.jev import encode_jpeg

    monkeypatch.setattr("rio.apps.kassow.graph.run.experiments_root", lambda: tmp_path)
    assert main(Args(command="describe")) == "ok"
    assert "nodes" in json.loads(capsys.readouterr().out)

    class Reply:
        answers = _box_answers(0.5, 0.5)

    class Client:
        def __init__(self, cfg):
            pass

        def ask(self, jpeg, questions):
            return Reply()

    monkeypatch.setattr("rio.apps.kassow.graph.run.JevClient", Client)
    image = tmp_path / "jev.jpg"
    image.write_bytes(encode_jpeg(np.zeros((480, 640, 3), dtype=np.uint8)))
    table = tmp_path / "table_map.json"
    _table().save(table)
    assert main(Args(command="ask", image=image, locate="the cube", table=table)) == "ok"
    printed = json.loads(capsys.readouterr().out)
    assert printed["image"] == str(image)
    assert printed["target"]["inside_area"] is True
