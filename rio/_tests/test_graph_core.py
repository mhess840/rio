# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""GaP v3 core: loading, validation and execution with fake tools. No robot."""

import copy

import pytest

from rio.graph import (
    MOTION,
    Executor,
    Param,
    Tool,
    ToolRegistry,
    WorkflowError,
    WorkflowInvalid,
    errors,
    parse_workflow,
    validate_workflow,
)

pytestmark = pytest.mark.unit


class Cell:
    """Fake tools that record their calls."""

    def __init__(self, *, visible: bool = True, held: bool = True, crash: str = "") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.visible = visible
        self.held = held
        self.crash = crash
        self.seen = 0

    def registry(self) -> ToolRegistry:
        def record(name, out):
            def fn(**kw):
                self.calls.append((name, kw))
                if self.crash == name:
                    raise RuntimeError("alarm")
                return out(**kw) if callable(out) else out

            return fn

        def locate(object: str) -> dict:
            self.seen += 1
            if not self.visible:
                return {"exit": "not_found"}
            return {"exit": "found", "target": {"name": object, "x": 0.4, "y": 0.1}}

        return ToolRegistry(
            [
                Tool(
                    "cam.locate",
                    record("cam.locate", locate),
                    "find an object",
                    inputs={"object": Param("str")},
                    outputs={"exit": "str", "target": "Target"},
                    exits=("found", "not_found"),
                ),
                Tool(
                    "arm.pick_at",
                    record("arm.pick_at", lambda target: {"exit": "held" if self.held else "missed"}),
                    "pick at a target",
                    inputs={"target": Param("Target")},
                    outputs={"exit": "str"},
                    exits=("held", "missed"),
                    tags=(MOTION,),
                ),
                Tool(
                    "arm.place_at",
                    record("arm.place_at", {"exit": "placed"}),
                    "place at a target",
                    inputs={"target": Param("Target")},
                    outputs={"exit": "str"},
                    exits=("placed",),
                    tags=(MOTION,),
                ),
                Tool("arm.open", record("arm.open", {}), "open the gripper", tags=(MOTION,)),
                Tool("arm.park", record("arm.park", {}), "park", tags=(MOTION,)),
                Tool(
                    "arm.count",
                    record("arm.count", lambda: {"exit": "more" if self.seen < 3 else "clean"}),
                    "loop helper",
                    outputs={"exit": "str"},
                    exits=("more", "clean"),
                ),
            ]
        )


def perceive(name: str, what: str) -> dict:
    return {
        "skill": "perceiving-objects",
        "outputs": {f"{name}": {"$ref": "find.target"}},
        "nodes": {
            "find": {"type": "tool", "tool": "cam.locate", "inputs": {"object": what}},
            "found": {"type": "noop"},
            "missing": {"type": "noop"},
        },
        "edges": [["START", "find"], ["found", "END"], ["missing", "END"]],
        "conditional_edges": {"find": {"router_field": "exit", "mapping": {"found": "found", "not_found": "missing"}}},
        "exit": {"router_field": None, "success_values": ["found", "missing"]},
        "on_error": "error",
    }


PICK_PLACE = {
    "version": 3,
    "meta": {"name": "cube_to_plate", "description": "Put the cube on the plate."},
    "nodes": {
        "cube": {"type": "subgraph", "ref": "cube_sg"},
        "plate": {"type": "subgraph", "ref": "plate_sg"},
        "grasp": {"type": "subgraph", "ref": "grasp_sg"},
        "put": {"type": "tool", "tool": "arm.place_at", "inputs": {"target": {"$ref": "plate.plate"}}},
        "done": {"type": "end", "status": "success"},
        "abort": {"type": "end", "status": "failure", "recovery": [{"tool": "arm.open"}, {"tool": "arm.park"}]},
    },
    "edges": [["START", "cube"]],
    "conditional_edges": {
        "cube": {"router_field": "exit", "mapping": {"found": "plate", "missing": "abort", "error": "abort"}},
        "plate": {"router_field": "exit", "mapping": {"found": "grasp", "missing": "abort", "error": "abort"}},
        "grasp": {"router_field": "exit", "mapping": {"grasped": "put", "missed": "abort", "failed": "abort"}},
        "put": {"router_field": "exit", "mapping": {"placed": "done"}},
    },
    "subgraphs": {
        "cube_sg": perceive("cube", "the red cube"),
        "plate_sg": perceive("plate", "the plate"),
        "grasp_sg": {
            "skill": "grasping-top-down",
            "inputs": {"cube": "Target"},
            "nodes": {
                "pick": {"type": "tool", "tool": "arm.pick_at", "inputs": {"target": {"$ref": "in.cube"}}},
                "grasped": {"type": "noop"},
                "missed": {"type": "noop"},
            },
            "edges": [["START", "pick"], ["grasped", "END"], ["missed", "END"]],
            "conditional_edges": {"pick": {"router_field": "exit", "mapping": {"held": "grasped", "missed": "missed"}}},
            "exit": {"router_field": None, "success_values": ["grasped", "missed"]},
            "on_error": "failed",
        },
    },
}


def workflow(**changes):
    raw = copy.deepcopy(PICK_PLACE)
    raw.update(changes)
    return parse_workflow(raw)


def test_pick_place_runs_and_passes_data_between_subgraphs():
    cell = Cell()
    result = Executor(workflow(), cell.registry()).run()
    assert result.status == "success" and result.end == "done"
    assert [name for name, _ in cell.calls] == ["cam.locate", "cam.locate", "arm.pick_at", "arm.place_at"]
    assert cell.calls[2][1]["target"]["name"] == "the red cube"
    assert cell.calls[3][1]["target"]["name"] == "the plate"
    kinds = [event["event"] for event in result.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert {"node": "grasp", "event": "route", "value": "grasped", "to": "put"}.items() <= next(
        e for e in result.events if e["event"] == "route" and e["node"] == "grasp"
    ).items()


def test_missing_object_routes_to_abort_and_recovers():
    cell = Cell(visible=False)
    result = Executor(workflow(), cell.registry()).run()
    assert result.status == "failure" and result.end == "abort"
    assert [name for name, _ in cell.calls] == ["cam.locate", "arm.open", "arm.park"]


def test_exception_inside_subgraph_becomes_on_error():
    cell = Cell(crash="arm.pick_at")
    result = Executor(workflow(), cell.registry()).run()
    assert result.end == "abort"
    end = next(e for e in result.events if e["event"] == "subgraph_end" and e["node"] == "grasp")
    assert end["exit"] == "failed" and end["error_path"] is True


def test_declined_motion_stops_without_recovery():
    cell = Cell()
    asked = []

    def confirm(path, tool, inputs):
        asked.append(path)
        return False

    result = Executor(workflow(), cell.registry(), confirm=confirm).run()
    assert result.status == "stopped"
    assert asked == ["grasp/pick"]
    assert "arm.open" not in [name for name, _ in cell.calls]


def test_stop_key_ends_before_next_node():
    cell = Cell()
    result = Executor(workflow(), cell.registry(), should_stop=lambda: len(cell.calls) >= 1).run()
    assert result.status == "stopped"
    assert [name for name, _ in cell.calls] == ["cam.locate"]


def test_conditional_back_edge_loops_until_exit():
    raw = {
        "version": 3,
        "nodes": {
            "look": {"type": "tool", "tool": "cam.locate", "inputs": {"object": "any"}},
            "check": {"type": "tool", "tool": "arm.count"},
            "done": {"type": "end", "status": "success"},
            "abort": {"type": "end", "status": "failure", "recovery": [{"tool": "arm.open"}]},
        },
        "edges": [["START", "look"]],
        "conditional_edges": {
            "look": {"router_field": "exit", "mapping": {"found": "check", "not_found": "abort"}},
            "check": {"router_field": "exit", "mapping": {"more": "look", "clean": "done"}},
        },
    }
    cell = Cell()
    result = Executor(parse_workflow(raw), cell.registry()).run()
    assert result.ok
    assert cell.seen == 3


def test_visit_cap_ends_an_endless_loop():
    raw = {
        "version": 3,
        "nodes": {
            "look": {"type": "tool", "tool": "cam.locate", "inputs": {"object": "any"}},
            "abort": {"type": "end", "status": "failure", "recovery": [{"tool": "arm.open"}]},
        },
        "edges": [["START", "look"]],
        "conditional_edges": {"look": {"router_field": "exit", "mapping": {"found": "look", "not_found": "abort"}}},
    }
    result = Executor(parse_workflow(raw), Cell().registry(), visit_cap=5).run()
    assert result.status == "limit"


def test_valid_workflow_has_no_errors():
    assert errors(validate_workflow(workflow(), Cell().registry())) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda raw: raw["conditional_edges"]["grasp"]["mapping"].pop("failed"), "exit 'failed' has no target"),
        (lambda raw: raw["conditional_edges"]["put"]["mapping"].update(blocked="abort"), "never reports"),
        (lambda raw: raw["nodes"]["put"].update(tool="arm.teleport"), "unknown tool 'arm.teleport'"),
        (lambda raw: raw["nodes"]["put"].update(inputs={}), "needs input 'target'"),
        (lambda raw: raw["nodes"]["put"]["inputs"].update(speed=2), "has no input 'speed'"),
        (lambda raw: raw["subgraphs"]["grasp_sg"]["inputs"].update(cup="Target"), "input 'cup'"),
        (lambda raw: raw["subgraphs"]["grasp_sg"]["exit"].update(success_values=["grasped"]), "missed"),
        (lambda raw: raw["edges"].append(["START", "plate"]), "parallel fan-out"),
        (lambda raw: raw["nodes"].update(lost={"type": "end", "status": "success"}), "not reachable"),
        (
            lambda raw: raw["nodes"]["put"]["inputs"].update(target={"$ref": "cube._exit"}),
            "wants Target",
        ),
    ],
)
def test_validator_reports(change, expected):
    raw = copy.deepcopy(PICK_PLACE)
    change(raw)
    found = [str(issue) for issue in errors(validate_workflow(parse_workflow(raw), Cell().registry()))]
    assert any(expected in line for line in found), found


def test_invalid_workflow_does_not_run():
    raw = copy.deepcopy(PICK_PLACE)
    raw["nodes"]["put"]["tool"] = "arm.teleport"
    cell = Cell()
    with pytest.raises(WorkflowInvalid):
        Executor(parse_workflow(raw), cell.registry()).run()
    assert cell.calls == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda raw: raw.update(version=2), "version must be 3"),
        (lambda raw: raw.update(extra=1), "unknown keys"),
        (lambda raw: raw["nodes"]["put"].update(type="script", script="x.py"), "only registered tools run"),
        (lambda raw: raw["nodes"]["put"].update(streaming=True), "streaming"),
        (lambda raw: raw["nodes"].update(START={"type": "noop"}), "reserved"),
        (lambda raw: raw["nodes"]["abort"]["recovery"].append({"service": "arm"}), "v2 recovery"),
    ],
)
def test_loader_refuses(change, expected):
    raw = copy.deepcopy(PICK_PLACE)
    change(raw)
    with pytest.raises(WorkflowError, match=expected):
        parse_workflow(raw)
