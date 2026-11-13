# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Walk one graph. Skills and Jev are injected, so a test never touches the arm."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .check import BOX_YES, NOUL_YES
from .locate import BOX_KEY, VISIBLE_KEY, locate_questions, read_box, read_noul
from .model import (
    APPROACH_M,
    HOVER_M,
    LOCATE_KIND,
    PLACE_Z_OFFSET_M,
    SKILL_KINDS,
    TARGET_KINDS,
    VLA_KIND,
    VLA_TIMEOUT_S,
    Graph,
    GraphError,
)

if TYPE_CHECKING:
    from .table_map import TableMap, Target

Ask = Callable[[str, dict, str], dict]
SkillCall = Callable[[dict], object]
Confirm = Callable[[str], bool]
RunVla = Callable[[str, str, float], object]


class Skills(Protocol):
    """The arm skills a graph node may name."""

    def move_to(self, pose: Path) -> object: ...
    def lift(self, meters: float = 0.1) -> object: ...
    def pick(self) -> object: ...
    def place(self) -> object: ...
    def gripper(self, state: str) -> object: ...
    def pick_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> object: ...
    def place_at(self, target: Target, *, approach_m: float, z_offset_m: float) -> object: ...
    def move_above(self, target: Target, *, height_m: float) -> object: ...


class Keys(Protocol):
    def pressed(self) -> set[str]: ...


@dataclass
class Step:
    """One node the walker entered."""

    node: str
    kind: str
    outcome: str
    detail: str
    nxt: str | None


@dataclass
class WalkLog:
    """JSON lines beside a run. ``root`` is the folder that receives them."""

    root: Path
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, step: Step) -> None:
        self.steps.append(step)
        record = {
            "t": datetime.now().isoformat(timespec="seconds"),
            "node": step.node,
            "kind": step.kind,
            "outcome": step.outcome,
            "detail": step.detail,
            "next": step.nxt,
        }
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def walk(
    graph: Graph,
    skills: Skills,
    ask: Ask,
    log: WalkLog,
    *,
    keys: Keys | None = None,
    table: TableMap | None = None,
    confirm: Confirm | None = None,
    run_vla: RunVla | None = None,
) -> str:
    """Follow edges from ``graph.start`` until a stop node, a quit key, or the step limit.

    A skill result of True takes ``ok``. False and an alarm string take ``fail``.
    A Jev noul or box at or above its threshold takes ``yes``. A score at or above
    its threshold takes ``high``. A choice takes the matching ``when`` edge.
    A locate node takes ``found`` and stores the target when the object is visible,
    boxed confidently, and inside the calibrated area; otherwise ``missing``.
    A vla node takes ``ok`` when the rollout finished, else ``fail``.

    Args:
        graph: Already checked.
        skills: Arm skills. A dry run passes printers that return True.
        ask: ``ask(node, questions, note)`` returns the ``answers`` object.
        log: Where each step is appended.
        keys: When ``q`` is pressed before a node, the walk ends.
        table: Image-to-table map; required by locate nodes.
        confirm: Called before each move to a located target and each VLA; False ends the walk.
        run_vla: ``run_vla(experiment, instruction, timeout_s)``; required by vla nodes.

    Returns:
        The stop reason.
    """
    targets: dict[str, Target] = {}
    name = graph.start
    for _ in range(graph.max_steps):
        if keys is not None and "q" in keys.pressed():
            log.add(Step(name, "quit", "quit", "operator pressed q", None))
            return "quit"
        node = graph.node(name)
        kind = str(node.get("kind"))
        if kind == "stop":
            reason = str(node.get("reason") or "stop")
            log.add(Step(name, kind, "stop", reason, None))
            return reason
        if kind in SKILL_KINDS:
            target = targets.get(str(node.get("target"))) if kind in TARGET_KINDS else None
            if target is not None and confirm is not None and not confirm(_target_message(node, target)):
                log.add(Step(name, kind, "quit", "operator declined", None))
                return "quit"
            result = _skill(skills, node, targets)
            outcome = "ok" if result is True else "fail"
            detail = "" if result is True else str(result)
            nxt = str(node[outcome])
            log.add(Step(name, kind, outcome, detail, nxt))
            name = nxt
            continue
        if kind == VLA_KIND:
            if run_vla is None:
                raise GraphError(f"vla node {name!r} needs a VLA runner")
            experiment = str(node["experiment"])
            instruction = str(node.get("instruction") or "")
            timeout_s = float(node.get("timeout_s", VLA_TIMEOUT_S))
            message = f"vla {experiment} {instruction!r}, up to {timeout_s:.0f} s (s ends, x aborts)"
            if confirm is not None and not confirm(message):
                log.add(Step(name, kind, "quit", "operator declined", None))
                return "quit"
            result = run_vla(experiment, instruction, timeout_s)
            outcome = "ok" if result is True else "fail"
            nxt = str(node[outcome])
            log.add(Step(name, kind, outcome, "" if result is True else str(result), nxt))
            name = nxt
            continue
        if kind == LOCATE_KIND:
            if table is None:
                raise GraphError(f"locate node {name!r} needs a table map")
            try:
                answers = ask(name, locate_questions(str(node["object"])), str(node.get("note") or ""))
                outcome, found, detail = _locate(node, answers, table)
            except Exception as error:
                nxt = node.get("fail")
                detail = f"locate failed: {error}"
                if not isinstance(nxt, str):
                    log.add(Step(name, kind, "fail", detail, None))
                    return detail
                log.add(Step(name, kind, "fail", detail, nxt))
                name = nxt
                continue
            if found is not None:
                targets[str(node["store"])] = found
            nxt = str(node[outcome])
            log.add(Step(name, kind, outcome, detail, nxt))
            name = nxt
            continue
        if kind == "jev":
            try:
                answers = ask(name, node["questions"], str(node.get("note") or ""))
            except Exception as error:
                nxt = node.get("fail")
                detail = f"jev failed: {error}"
                if not isinstance(nxt, str):
                    log.add(Step(name, kind, "fail", detail, None))
                    return detail
                log.add(Step(name, kind, "fail", detail, nxt))
                name = nxt
                continue
            outcome = _jev_edge(node, answers)
            nxt = _jev_target(node, outcome)
            log.add(Step(name, kind, outcome, json.dumps(answers, default=str), nxt))
            name = nxt
            continue
        raise GraphError(f"node {name!r} has kind {kind!r}")
    log.add(Step(name, "limit", "limit", f"stopped after {graph.max_steps} steps", None))
    return f"stopped after {graph.max_steps} steps"


def _skill(skills: Skills, node: dict, targets: dict[str, Target]) -> object:
    kind = node["kind"]
    if kind == "move_to":
        from .check import resolve_pose

        return skills.move_to(resolve_pose(str(node["pose"])))
    if kind == "lift":
        meters = float(node["meters"]) if "meters" in node else 0.1
        return skills.lift(meters)
    if kind == "pick":
        return skills.pick()
    if kind == "gripper":
        return skills.gripper(str(node["state"]))
    if kind in TARGET_KINDS:
        target = targets.get(str(node["target"]))
        if target is None:
            return f"no target {node['target']!r} located yet"
        if kind == "move_above":
            return skills.move_above(target, height_m=float(node.get("height_m", HOVER_M)))
        approach_m = float(node.get("approach_m", APPROACH_M))
        if kind == "place_at":
            return skills.place_at(target, approach_m=approach_m, z_offset_m=float(node.get("z_offset_m", PLACE_Z_OFFSET_M)))
        return skills.pick_at(target, approach_m=approach_m, z_offset_m=float(node.get("z_offset_m", 0.0)))
    return skills.place()


def _target_message(node: dict, target: Target) -> str:
    return (
        f"{node['kind']} {target.name}: x={target.x:.3f} y={target.y:.3f} m, "
        f"footprint {target.width_m * 100:.1f} x {target.depth_m * 100:.1f} cm"
    )


def _locate(node: dict, answers: dict, table: TableMap) -> tuple[str, Target | None, str]:
    what = str(node["object"])
    visible = read_noul(answers, VISIBLE_KEY)
    if visible < float(node.get("visible_threshold", NOUL_YES)):
        return "missing", None, f"{what} not visible ({visible:.2f})"
    box = read_box(answers, BOX_KEY)
    if box.confidence < float(node.get("threshold", BOX_YES)):
        return "missing", None, f"box confidence {box.confidence:.2f}"
    target = table.target(box, what)
    if not table.contains(target.x, target.y):
        return "missing", None, f"{what} at x={target.x:.3f} y={target.y:.3f} is outside the calibrated area"
    return "found", target, json.dumps(target.to_dict())


def _jev_edge(node: dict, answers: dict) -> str:
    route = str(node["route"])
    if not isinstance(answers, dict) or route not in answers:
        raise GraphError(f"Jev answers have no {route!r}")
    answer = answers[route]
    if not isinstance(answer, dict):
        raise GraphError(f"Jev answer {route!r} must be an object")
    kind = answer.get("type")
    if kind == "noul":
        threshold = float(node.get("threshold", NOUL_YES))
        return "yes" if float(answer["noul"]) >= threshold else "no"
    if kind == "box":
        threshold = float(node.get("threshold", BOX_YES))
        return "yes" if float(answer.get("confidence", 0.0)) >= threshold else "no"
    if kind == "score":
        return "high" if float(answer["score"]) >= float(node["threshold"]) else "low"
    if kind == "choice":
        choice = str(answer.get("choice") or "")
        when = node.get("when")
        if isinstance(when, dict) and choice in when:
            return choice
        if "else" in node:
            return "else"
        raise GraphError(f"choice {choice!r} has no when edge")
    raise GraphError(f"Jev answer {route!r} has type {kind!r}")


def _jev_target(node: dict, outcome: str) -> str:
    if outcome in node and isinstance(node[outcome], str):
        return str(node[outcome])
    when = node.get("when")
    if isinstance(when, dict) and outcome in when:
        return str(when[outcome])
    raise GraphError(f"Jev outcome {outcome!r} has no edge")
