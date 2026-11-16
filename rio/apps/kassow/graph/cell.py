# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kassow cell's tools for v3 workflows (``rio.graph``).

Each tool wraps an existing skill, Jev, a VLA, or the operator. Arm skills
report ``True``, ``False`` (refused or missed) or an alarm string. A tool turns
an alarm into an exception, so a subgraph's ``on_error`` catches it, and turns
``True``/``False`` into an ``exit`` value where the planner needs to branch.

    registry = cell_tools(skills, ask, table=table, run_vla=run_vla, ask_operator=ask_operator)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rio.graph import MOTION, OPERATOR, PERCEPTION, Param, Tool, ToolRegistry, current_node

from .check import BOX_YES, NOUL_YES, _check_question, pose_is_file, resolve_pose
from .locate import BOX_KEY, VISIBLE_KEY, locate_questions, read_box, read_noul
from .model import (
    APPROACH_M,
    APPROACH_M_RANGE,
    HOVER_M,
    HOVER_M_RANGE,
    INSTRUCTION_MAX_CHARS,
    OBJECT_MAX_CHARS,
    PLACE_Z_OFFSET_M,
    PLACE_Z_OFFSET_M_RANGE,
    VLA_TIMEOUT_S,
    VLA_TIMEOUT_S_RANGE,
    Z_OFFSET_M_RANGE,
)

if TYPE_CHECKING:
    from .table_map import TableMap, Target
    from .walk import Skills

Ask = Callable[[str, dict, str], dict]
"""``ask(node_path, questions, note)`` returns Jev's ``answers`` object."""
AskOperator = Callable[[str, str, list[str]], str]
"""``ask_operator(node_path, question, answers)`` returns one of ``answers``."""
RunVla = Callable[[str, str, float], object]

TARGET = "Target"
LIFT_M = 0.1
OPERATOR_MAX_ANSWERS = 9
"""The operator answers with one key, 1 to 9."""


class SkillAlarm(RuntimeError):
    """The arm reported an alarm; a subgraph's ``on_error`` takes it."""


def cell_tools(
    skills: Skills,
    ask: Ask,
    *,
    table: TableMap | None = None,
    run_vla: RunVla | None = None,
    ask_operator: AskOperator | None = None,
    pose_exists: Callable[[str], bool] = pose_is_file,
) -> ToolRegistry:
    """Every tool of the cell, bound to live or dry-run parts.

    Args:
        skills: ``KassowSkills`` on the cell, ``DrySkills`` in a dry run.
        ask: Jev questions about a fresh Jev photo.
        table: Image-to-table map; ``jev.locate`` needs it.
        run_vla: ``run_vla(experiment, instruction, timeout_s)``; ``vla.run`` needs it.
        ask_operator: Asks the human at the cell; ``operator.ask`` needs it.
        pose_exists: Whether a home file exists (tests pass a stub).
    """

    def move_to(pose: str) -> dict:
        _must(skills.move_to(resolve_pose(pose)), f"move_to {pose}")
        return {}

    def lift(meters: float = LIFT_M) -> dict:
        _must(skills.lift(meters), "lift")
        return {}

    def gripper_open() -> dict:
        _must(skills.gripper("open"), "gripper open")
        return {}

    def gripper_close() -> dict:
        _must(skills.gripper("close"), "gripper close")
        return {}

    def gripper_grasp() -> dict:
        held = skills.pick()
        _alarm(held)
        return {"exit": "held" if held is True else "empty"}

    def pick_at(target: Target, approach_m: float = APPROACH_M, z_offset_m: float = 0.0) -> dict:
        held = skills.pick_at(target, approach_m=approach_m, z_offset_m=z_offset_m)
        _alarm(held)
        return {"exit": "held" if held is True else "missed"}

    def place_at(target: Target, approach_m: float = APPROACH_M, z_offset_m: float = PLACE_Z_OFFSET_M) -> dict:
        _must(skills.place_at(target, approach_m=approach_m, z_offset_m=z_offset_m), f"place_at {target.name}")
        return {}

    def move_above(target: Target, height_m: float = HOVER_M) -> dict:
        _must(skills.move_above(target, height_m=height_m), f"move_above {target.name}")
        return {}

    def locate(object: str, threshold: float = BOX_YES, visible_threshold: float = NOUL_YES, note: str = "") -> dict:
        if table is None:
            raise RuntimeError("jev.locate needs a table map")
        answers = ask(current_node(), locate_questions(object), note)
        visible = read_noul(answers, VISIBLE_KEY)
        box = read_box(answers, BOX_KEY)
        seen = {
            "confidence": box.confidence,
            "visible": visible,
            "box": {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom},
        }
        if visible < visible_threshold:
            return {"exit": "not_found", "detail": f"not visible ({visible:.2f})", **seen}
        if box.confidence < threshold:
            return {"exit": "not_found", "detail": f"box confidence {box.confidence:.2f}", **seen}
        target = table.target(box, object)
        if not table.contains(target.x, target.y):
            detail = f"x={target.x:.3f} y={target.y:.3f} is outside the calibrated area"
            return {"exit": "outside_area", "target": target, "detail": detail, **seen}
        return {"exit": "found", "target": target, "detail": json.dumps(target.to_dict()), **seen}

    def jev_ask(question: dict, threshold: float | None = None, note: str = "") -> dict:
        answers = ask(current_node(), {"q": question}, note)
        answer = answers.get("q") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise RuntimeError("Jev gave no answer")
        return {"exit": _jev_exit(question, answer, threshold), "answer": answer}

    def operator_ask(question: str, answers: list[str] | None = None) -> dict:
        if ask_operator is None:
            raise RuntimeError("operator.ask needs an operator (collaborative mode)")
        options = list(answers or ["yes", "no"])
        reply = ask_operator(current_node(), question, options)
        if reply not in options:
            raise RuntimeError(f"operator answer {reply!r} is not one of {options}")
        return {"exit": reply}

    def vla(experiment: str, instruction: str = "", timeout_s: float = VLA_TIMEOUT_S) -> dict:
        if run_vla is None:
            raise RuntimeError("vla.run needs a VLA runner")
        result = run_vla(experiment, instruction, timeout_s)
        return {"exit": "finished" if result is True else "stopped", "detail": "" if result is True else str(result)}

    def check_pose(inputs: dict[str, Any]) -> list[str]:
        pose = inputs.get("pose")
        if isinstance(pose, str) and not pose_exists(pose):
            return [f"pose {pose!r} is not a home file (describe lists the poses)"]
        return []

    target_param = Param(TARGET, doc="target output of jev.locate")
    approach = Param("float", False, f"metres above the target before descending, {list(APPROACH_M_RANGE)}")
    return ToolRegistry(
        [
            Tool(
                "kassow.move_to",
                move_to,
                "moveJ to a saved joint pose (park, home, a VLA start)",
                inputs={"pose": Param("str", doc="home file name from describe")},
                tags=(MOTION,),
                check=check_pose,
            ),
            Tool(
                "kassow.lift",
                lift,
                "raise the TCP straight up",
                inputs={"meters": Param("float", False, "default 0.1")},
                tags=(MOTION,),
                check=_ranges({"meters": (0.01, 0.3)}),
            ),
            Tool(
                "kassow.pick_at",
                pick_at,
                "open, move above the target, descend to the calibrated grasp height, close, lift; "
                "exit held only if the object is still in the jaws",
                inputs={
                    "target": target_param,
                    "approach_m": approach,
                    "z_offset_m": Param("float", False, f"raise the grasp, {list(Z_OFFSET_M_RANGE)}"),
                },
                outputs={"exit": "str"},
                exits=("held", "missed"),
                tags=(MOTION,),
                check=_ranges({"approach_m": APPROACH_M_RANGE, "z_offset_m": Z_OFFSET_M_RANGE}),
            ),
            Tool(
                "kassow.place_at",
                place_at,
                "move above the target, descend to grasp height plus z_offset_m, open, lift; raises when nothing is held",
                inputs={
                    "target": target_param,
                    "approach_m": approach,
                    "z_offset_m": Param("float", False, f"height of what it is set on, {list(PLACE_Z_OFFSET_M_RANGE)}"),
                },
                tags=(MOTION,),
                check=_ranges({"approach_m": APPROACH_M_RANGE, "z_offset_m": PLACE_Z_OFFSET_M_RANGE}),
            ),
            Tool(
                "kassow.move_above",
                move_above,
                "move to height_m above the calibrated grasp height at the target",
                inputs={"target": target_param, "height_m": Param("float", False, f"{list(HOVER_M_RANGE)}")},
                tags=(MOTION,),
                check=_ranges({"height_m": HOVER_M_RANGE}),
            ),
            Tool("gripper.open", gripper_open, "open the jaws", tags=(MOTION,)),
            Tool("gripper.close", gripper_close, "close the jaws without checking for an object", tags=(MOTION,)),
            Tool(
                "gripper.grasp",
                gripper_grasp,
                "close the jaws; exit held only if an object blocks them",
                outputs={"exit": "str"},
                exits=("held", "empty"),
                tags=(MOTION,),
            ),
            Tool(
                "jev.locate",
                locate,
                "find one object in a fresh Jev photo and map it onto the table",
                inputs={
                    "object": Param("str", doc=f"what to find, at most {OBJECT_MAX_CHARS} characters"),
                    "threshold": Param("float", False, f"box confidence, default {BOX_YES}"),
                    "visible_threshold": Param("float", False, f"visibility noul, default {NOUL_YES}"),
                    "note": Param("str", False, "context for Jev"),
                },
                outputs={
                    "exit": "str",
                    "target": TARGET,
                    "detail": "str",
                    "confidence": "float",
                    "visible": "float",
                    "box": "dict",
                },
                exits=("found", "not_found", "outside_area"),
                tags=(PERCEPTION,),
                check=_check_locate,
            ),
            Tool(
                "jev.ask",
                jev_ask,
                "ask Jev one typed question about a fresh Jev photo; exit yes/no (noul, box), "
                "high/low (score) or the chosen criterion (choice)",
                inputs={
                    "question": Param("dict", doc="one Jev question: type, instructions, criteria"),
                    "threshold": Param("float", False, "noul/box default 0.7/0.6; score required"),
                    "note": Param("str", False, "context for Jev"),
                },
                outputs={"exit": "str", "answer": "dict"},
                tags=(PERCEPTION,),
                check=_check_jev,
            ),
            Tool(
                "operator.ask",
                operator_ask,
                "ask the operator at the cell; the answer is the exit (collaborative mode only)",
                inputs={
                    "question": Param("str", doc="one short question about what the operator sees"),
                    "answers": Param("list", False, f"2 to {OPERATOR_MAX_ANSWERS} answers, default yes/no"),
                },
                outputs={"exit": "str"},
                tags=(OPERATOR,),
                check=_check_operator,
            ),
            Tool(
                "vla.run",
                vla,
                "moveJ to the VLA's start pose, run the learned policy until it finishes, s, x, or the timeout",
                inputs={
                    "experiment": Param("str", doc="a VLA with usable true in describe"),
                    "instruction": Param("str", False, f"one of its instructions, at most {INSTRUCTION_MAX_CHARS} characters"),
                    "timeout_s": Param("float", False, f"{list(VLA_TIMEOUT_S_RANGE)}, default {VLA_TIMEOUT_S}"),
                },
                outputs={"exit": "str", "detail": "str"},
                exits=("finished", "stopped"),
                tags=(MOTION,),
                check=_check_vla,
            ),
        ]
    )


def _alarm(outcome: object) -> None:
    if isinstance(outcome, str):
        raise SkillAlarm(outcome)


def _must(outcome: object, what: str) -> None:
    _alarm(outcome)
    if outcome is not True:
        raise RuntimeError(f"{what} failed")


def _ranges(bounds: dict[str, tuple[float, float]]) -> Callable[[dict[str, Any]], list[str]]:
    def check(inputs: dict[str, Any]) -> list[str]:
        problems = []
        for key, (low, high) in bounds.items():
            if key not in inputs:
                continue
            value = inputs[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                problems.append(f"{key} must be a number from {low} to {high}")
        return problems

    return check


def _check_locate(inputs: dict[str, Any]) -> list[str]:
    problems = _ranges({"threshold": (0.0, 1.0), "visible_threshold": (0.0, 1.0)})(inputs)
    what = inputs.get("object")
    if what is not None and (not isinstance(what, str) or not what.strip() or len(what) > OBJECT_MAX_CHARS or "`" in what):
        problems.append(f"object must be text of at most {OBJECT_MAX_CHARS} characters without backticks")
    return problems


def _check_jev(inputs: dict[str, Any]) -> list[str]:
    question = inputs.get("question")
    if question is None:
        return []
    problems = [p.replace("node 'jev.ask' ", "") for p in _check_question("jev.ask", "question", question)]
    kind = question.get("type") if isinstance(question, dict) else None
    if kind == "score" and not isinstance(inputs.get("threshold"), (int, float)):
        problems.append("a score question needs a numeric threshold")
    if kind in ("noul", "box"):
        problems += _ranges({"threshold": (0.0, 1.0)})(inputs)
    return problems


def _check_operator(inputs: dict[str, Any]) -> list[str]:
    answers = inputs.get("answers")
    if answers is None:
        return []
    ok = (
        isinstance(answers, list)
        and 2 <= len(answers) <= OPERATOR_MAX_ANSWERS
        and all(isinstance(a, str) and a for a in answers)
    )
    if not ok or len(set(answers)) != len(answers):
        return [f"answers must be 2 to {OPERATOR_MAX_ANSWERS} distinct names"]
    return []


def _check_vla(inputs: dict[str, Any]) -> list[str]:
    from rio.apps.kassow.experiments import EXPERIMENTS, resolve_experiment

    from .vla import GRAPH_STATION, vla_usable

    problems = _ranges({"timeout_s": VLA_TIMEOUT_S_RANGE})(inputs)
    experiment = inputs.get("experiment")
    if experiment is None:
        return problems
    if experiment not in EXPERIMENTS:
        return [*problems, f"experiment must be one of {sorted(EXPERIMENTS)}"]
    if not vla_usable(str(experiment)):
        serial = resolve_experiment(str(experiment)).overview_serial
        problems.append(f"vla {experiment!r} needs overview camera {serial}; the graph station runs {GRAPH_STATION!r}")
    instruction = inputs.get("instruction", "")
    if not isinstance(instruction, str) or len(instruction) > INSTRUCTION_MAX_CHARS:
        return [*problems, f"instruction must be text of at most {INSTRUCTION_MAX_CHARS} characters"]
    try:
        EXPERIMENTS[str(experiment)].select_home(instruction)
    except ValueError as error:
        problems.append(str(error))
    return problems


def _jev_exit(question: dict, answer: dict, threshold: float | None) -> str:
    kind = answer.get("type")
    if kind == "noul":
        return "yes" if float(answer["noul"]) >= (NOUL_YES if threshold is None else threshold) else "no"
    if kind == "box":
        return "yes" if float(answer.get("confidence", 0.0)) >= (BOX_YES if threshold is None else threshold) else "no"
    if kind == "score":
        if threshold is None:
            raise RuntimeError("a score question needs a threshold")
        return "high" if float(answer["score"]) >= threshold else "low"
    if kind == "choice":
        choice = str(answer.get("choice") or "")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or choice not in criteria:
            raise RuntimeError(f"Jev chose {choice!r}, not a criterion")
        return choice
    raise RuntimeError(f"Jev answer has type {kind!r}")
