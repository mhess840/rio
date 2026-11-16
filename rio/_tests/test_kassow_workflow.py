# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Kassow tools for v3 workflows. No arm, no camera, no network."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rio._tests.test_kassow_graph import _box_answers, _table
from rio.apps.kassow.graph.cell import cell_tools
from rio.apps.kassow.graph.run import Args, main
from rio.graph import Executor, errors, load_workflow, parse_workflow, validate_workflow
from rio.sam import Instance, SamReply

pytestmark = pytest.mark.unit

CYLINDER = Path(__file__).resolve().parents[1] / "apps/kassow/graph/workflows/cylinder_to_plate.json"


class Arm:
    """Fake skills: record calls, answer True unless told otherwise."""

    def __init__(self, **results) -> None:
        self.calls: list[str] = []
        self.turns: list[float] = []
        self.place_turns: list[tuple[float, bool]] = []
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

    def pick_at(self, target, *, approach_m, z_offset_m, yaw_offset_deg=0.0, outside_area=False):
        self.turns.append(yaw_offset_deg)
        return self._do("pick_at")

    def place_at(self, target, *, approach_m, z_offset_m, yaw_offset_deg=0.0, outside_area=False):
        self.place_turns.append((yaw_offset_deg, outside_area))
        return self._do("place_at")

    def move_above(self, target, *, height_m):
        return self._do("move_above")

    def move_tcp(self, xyz, rotvec):
        self.tcp = np.r_[xyz, rotvec]
        return self._do("move_tcp")


def jev(*_):
    return _box_answers(0.5, 0.5)


def sam_hit(score=0.95):
    """A SAM segmenter that finds one instance in the middle of a 640x480 photo for every prompt."""
    return lambda jpeg, prompt: SamReply([Instance(0, (300, 220, 340, 260), (640, 480), score)], "", 0.0)


def tools(arm=None, **kwargs):
    kwargs.setdefault("photo", lambda node: b"jpeg")
    kwargs.setdefault("segment", sam_hit())
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
    script.write_text(
        json.dumps(
            {
                "cylinder/find": {"blue cap": [[246, 182, 266, 202, 0.95]]},
                "plate/find": {"orange plate": [[370, 270, 400, 300, 0.98]]},
            }
        )
    )
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
    photos = []

    def photo(node):
        photos.append(node)
        return node.encode()

    def segment(jpeg, prompt):
        if jpeg == b"cylinder/find" and photos.count("cylinder/find") > 2:
            return SamReply([], "", 0.0)
        return sam_hit()(jpeg, prompt)

    arm = Arm()
    workflow = parse_workflow(raw)
    registry = tools(arm, photo=photo, segment=segment)
    assert errors(validate_workflow(workflow, registry)) == []
    result = Executor(workflow, registry).run()
    assert result.end == "done"
    assert arm.calls.count("pick_at") == 2 and arm.calls.count("place_at") == 2


def test_sam_locate_tries_prompts_in_order_and_reports_them():
    asked = []

    def segment(jpeg, prompt):
        asked.append(prompt)
        score = 0.6 if prompt == "red plate" else 0.98
        return SamReply([Instance(0, (300, 220, 340, 260), (640, 480), score)], "", 0.0)

    registry = tools(segment=segment)
    out = registry.get("sam.locate").fn(prompts=["red plate", "orange plate", "plate"])
    assert asked == ["red plate", "orange plate"]
    assert out["exit"] == "found" and out["prompt"] == "orange plate"
    assert out["tried"] == {"red plate": 0.6, "orange plate": 0.98}
    assert out["target"].name == "orange plate"
    missing = tools(segment=lambda jpeg, prompt: SamReply([], "", 0.0)).get("sam.locate").fn(prompts=["cube"])
    assert missing["exit"] == "not_found"


def test_sam_prompts_are_checked():
    check = tools().get("sam.locate").check
    assert check({"prompts": ["orange plate"]}) == []
    assert check({"prompts": []})
    assert any("one noun phrase" in p for p in check({"prompts": ["cube and plate"]}))
    assert any("one noun phrase" in p for p in check({"prompts": ["cube, plate"]}))


def test_pick_at_align_yaw_turns_the_wrist_the_short_way():
    from rio.apps.kassow.graph.table_map import Target

    arm = Arm()
    pick = tools(arm, jaw_yaw=lambda: 10.0).get("kassow.pick_at").fn
    cube = Target("cube", -0.4, 0.6, 0.037, 0.037, 0.9, yaw_deg=70.0, yaw_period_deg=90.0)
    assert pick(target=cube, align_yaw=True)["wrist_turn_deg"] == -30.0
    assert pick(target=cube)["wrist_turn_deg"] == 0.0
    round_one = Target("cap", -0.4, 0.6, 0.05, 0.05, 0.9)
    assert pick(target=round_one, align_yaw=True)["wrist_turn_deg"] == 0.0
    assert arm.turns == [-30.0, 0.0, 0.0]
    bar = Target("bar", -0.4, 0.6, 0.1, 0.03, 0.9, yaw_deg=20.0, yaw_period_deg=180.0)
    with pytest.raises(ValueError, match="closing direction"):
        pick(target=bar, align_yaw=True)
    with pytest.raises(RuntimeError, match="jaw reference"):
        tools(jaw_yaw=lambda: None).get("kassow.pick_at").fn(target=cube, align_yaw=True)


def test_sam_locate_attaches_the_depth_yaw_to_the_target():
    from rio._tests.test_kassow_depth import box_of, camera, render
    from rio.apps.kassow.graph.cell import Shot

    he = camera()
    depth = render(he, [{"x": -0.40, "y": 0.60, "h": 0.037, "l": 0.037, "w": 0.037, "yaw": 25.0}])
    box = box_of(he, depth, -0.40, 0.60, 0.03)

    def segment(jpeg, prompt):
        return SamReply([Instance(0, tuple(float(v) for v in box), (640, 480), 0.95)], "", 0.0)

    registry = tools(photo=lambda node: Shot(b"jpeg", depth), segment=segment, handeye=he)
    out = registry.get("sam.locate").fn(prompts=["red object"])
    assert out["measured"]["period_deg"] == 90.0
    yaw = out["target"].yaw_deg
    assert yaw is not None and min(abs(yaw - 25.0), 90 - abs(yaw - 25.0)) < 3.0
    assert out["target"].to_dict()["yaw_period_deg"] == 90.0


def test_sam_locate_in_another_camera_uses_its_pose():
    from rio._tests.test_kassow_depth import TABLE_Z, box_of, camera
    from rio.apps.kassow.graph.cell import TABLE_Z_M, Shot

    he = camera()  # any calibrated view; looks down from 0.86 m
    obj = np.array([-0.42, 0.62])
    box = box_of(he, np.zeros((480, 640)), obj[0], obj[1], 0.02)
    # box_of projects at TABLE_Z of the depth tests; shift the object plane to match cell.TABLE_Z_M + h/2
    height = 2 * (TABLE_Z - TABLE_Z_M)

    def segment(jpeg, prompt):
        return SamReply([Instance(0, tuple(float(v) for v in box), (640, 480), 0.95)], "", 0.0)

    registry = tools(segment=segment, views={"overview": lambda node: Shot(b"jpeg", None, he)})
    out = registry.get("sam.locate").fn(prompts=["cube"], camera="overview", height_m=height)
    assert out["camera"] == "overview"
    assert np.allclose(out["object_xyz"][:2], obj, atol=0.003)
    with pytest.raises(RuntimeError, match="not connected"):
        registry.get("sam.locate").fn(prompts=["cube"], camera="wrist")


def test_wrist_view_pose_points_the_camera_at_the_spot():
    from scipy.spatial.transform import Rotation

    from rio.apps.kassow.graph.cell import wrist_view_pose
    from rio.apps.kassow.graph.handeye import Intrinsics
    from rio.apps.kassow.graph.wristeye import WristEye, pose_matrix

    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("y", 8.0, degrees=True).as_matrix()
    mount[:3, 3] = [-0.078, 0.034, 0.028]
    wrist = WristEye("0", Intrinsics(640, 480, 606, 606, 333, 245), mount.tolist(), 3, np.eye(4).tolist())
    down = Rotation.from_euler("x", 180.0, degrees=True).as_rotvec()
    spot = np.array([-0.4, 0.6, -0.02])
    xyz, rotvec = wrist_view_pose(wrist, down, spot, 0.3)
    camera = pose_matrix(np.r_[xyz, rotvec]) @ mount
    assert camera[2, 3] == pytest.approx(spot[2] + 0.3, abs=1e-6)
    hit = wrist.pixel_to_plane(np.r_[xyz, rotvec], 333, 245, spot[2])
    assert hit is not None and np.allclose(hit[:2], spot[:2], atol=1e-6)


def test_search_walks_views_until_the_wrist_camera_sees_the_object():
    from scipy.spatial.transform import Rotation

    from rio.apps.kassow.graph.cell import TABLE_Z_M, Shot
    from rio.apps.kassow.graph.handeye import HandEye, Intrinsics
    from rio.apps.kassow.graph.wristeye import WristEye

    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("y", 8.0, degrees=True).as_matrix()
    mount[:3, 3] = [-0.078, 0.034, 0.028]
    intr = Intrinsics(640, 480, 606, 606, 320, 240)
    wrist = WristEye("0", intr, mount.tolist(), 3, np.eye(4).tolist())
    # 0.25 m from the hint toward the second ring point: out of the first view, in a later one
    obj = np.array([-0.45 + 0.25 * 0.5, 0.55 + 0.25 * np.sqrt(3) / 2, TABLE_Z_M + 0.02])
    arm = Arm()

    def photo(node):
        pose = wrist.base_from_camera(arm.tcp)
        return Shot(b"jpeg", None, HandEye("0", intr, pose.tolist(), {}, {}, {}))

    def segment(jpeg, prompt):
        pose = wrist.base_from_camera(arm.tcp)
        c = np.linalg.inv(pose) @ np.r_[obj, 1.0]
        u, v = intr.fx * c[0] / c[2] + intr.cx, intr.fy * c[1] / c[2] + intr.cy
        if c[2] <= 0 or not (20 < u < 620 and 20 < v < 460):
            return SamReply([], "", 0.0)
        return SamReply([Instance(0, (u - 15, v - 15, u + 15, v + 15), (640, 480), 0.93)], "", 0.0)

    registry = tools(arm, segment=segment, views={"wrist": photo}, wrist=wrist)
    out = registry.get("kassow.search").fn(prompts=["box"], x=-0.45, y=0.55, radius_m=0.2, max_views=7)
    assert out["exit"] in ("found", "outside_area")
    assert np.allclose(out["object_xyz"][:2], obj[:2], atol=0.002)
    assert 1 < len(out["views"]) <= 7
    assert arm.calls.count("move_tcp") == len(out["views"]) + 1  # plus the sharpening look
    missing = tools(Arm(), segment=lambda j, p: SamReply([], "", 0.0), views={"wrist": photo}, wrist=wrist)
    arm.tcp = np.r_[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    assert missing.get("kassow.search").fn(prompts=["box"], x=-0.45, y=0.55, max_views=3)["exit"] == "not_found"


def test_elongated_objects_are_grasped_and_placed_across_their_long_axis(monkeypatch):
    from dataclasses import replace

    from rio.apps.kassow.graph.table_map import Target

    monkeypatch.setattr("rio.apps.kassow.graph.table_map.load_jaw_closing", lambda path=None: 30.0)
    arm = Arm()
    registry = tools(arm, jaw_yaw=lambda: 30.0)
    bar = Target("bar", -0.4, 0.6, 0.15, 0.045, 0.9, yaw_deg=20.0, yaw_period_deg=180.0)
    out = registry.get("kassow.pick_at").fn(target=bar, align_yaw=True)
    assert out["wrist_turn_deg"] == pytest.approx(80.0)  # closing axis 30 -> 110, across the long axis 20
    slot = registry.get("table.point").fn(x=-0.37, y=0.76, name="slot")["target"]
    placed = registry.get("kassow.place_at").fn(target=slot, align_to_deg=31.0, outside_area=True)
    assert placed["wrist_turn_deg"] == pytest.approx(91.0)  # 121 across the slot axis 31; +91 is nearest the grasp's +80
    assert arm.place_turns[-1] == (pytest.approx(91.0), True)
    near_minus = tools(Arm(), jaw_yaw=lambda: 30.0)
    near_minus.get("kassow.pick_at").fn(target=replace(bar, yaw_deg=-60.0 + 180.0), align_yaw=True)  # turn -80
    again = near_minus.get("kassow.place_at").fn(target=slot, align_to_deg=31.0, outside_area=True)
    assert again["wrist_turn_deg"] == pytest.approx(-89.0)


def test_edge_yaw_of_a_dark_box_from_above():
    import cv2

    from rio._tests.test_kassow_depth import camera
    from rio.apps.kassow.graph.depth import edge_long_axis_yaw

    he = camera()
    image = np.full((480, 640, 3), 70, np.uint8)
    inv = np.linalg.inv(he.transform)
    intr = he.intrinsics

    def px(x, y, z=0.1):
        c = inv @ np.array([x, y, z, 1.0])
        return [intr.fx * c[0] / c[2] + intr.cx, intr.fy * c[1] / c[2] + intr.cy]

    yaw = np.radians(35.0)
    a, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    centre = np.array([-0.42, 0.60])
    corners = [centre + sa * 0.08 * a + sn * 0.025 * n for sa, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    poly = np.int32([px(*c) for c in corners])
    cv2.fillPoly(image, [poly], (15, 10, 35))  # dark box on the grey table, as on the cell
    x1, y1 = poly.min(axis=0) - 10
    x2, y2 = poly.max(axis=0) + 10
    got = edge_long_axis_yaw(image, (x1, y1, x2, y2), he, 0.1)
    assert got is not None and min(abs(got - 35.0), 180 - abs(got - 35.0)) < 3.0


def test_pick_at_prefers_the_equivalent_turn_near_the_planned_one(monkeypatch):
    from rio.apps.kassow.graph.table_map import Target

    monkeypatch.setattr("rio.apps.kassow.graph.table_map.load_jaw_closing", lambda path=None: 31.07)
    pick = tools(Arm(), jaw_yaw=lambda: 31.07).get("kassow.pick_at").fn
    box = Target("box", -0.25, 0.52, 0.15, 0.045, 0.9, yaw_deg=43.5, yaw_period_deg=180.0)
    assert pick(target=box, align_yaw=True)["wrist_turn_deg"] == pytest.approx(-77.6, abs=0.1)
    assert pick(target=box, align_yaw=True, prefer_turn_deg=89.0)["wrist_turn_deg"] == pytest.approx(102.4, abs=0.1)
