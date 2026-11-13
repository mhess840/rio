# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Walk one graph. Skills and Jev are injected, so a test never touches the arm."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .check import BOX_YES, NOUL_YES
from .model import SKILL_KINDS, Graph, GraphError

Ask = Callable[[str, dict, str], dict]
SkillCall = Callable[[dict], object]


class Skills(Protocol):
    """The four arm skills a graph node may name."""

    def move_to(self, pose: Path) -> object: ...
    def lift(self, meters: float = 0.1) -> object: ...
    def pick(self) -> object: ...
    def place(self) -> object: ...


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
) -> str:
    """Follow edges from ``graph.start`` until a stop node, a quit key, or the step limit.

    A skill result of True takes ``ok``. False and an alarm string take ``fail``.
    A Jev noul or box at or above its threshold takes ``yes``. A score at or above
    its threshold takes ``high``. A choice takes the matching ``when`` edge.

    Args:
        graph: Already checked.
        skills: Arm skills. A dry run passes printers that return True.
        ask: ``ask(node, questions, note)`` returns the ``answers`` object.
        log: Where each step is appended.
        keys: When ``q`` is pressed before a node, the walk ends.

    Returns:
        The stop reason.
    """
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
            result = _skill(skills, node)
            outcome = "ok" if result is True else "fail"
            detail = "" if result is True else str(result)
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


def _skill(skills: Skills, node: dict) -> object:
    kind = node["kind"]
    if kind == "move_to":
        from .check import resolve_pose

        return skills.move_to(resolve_pose(str(node["pose"])))
    if kind == "lift":
        meters = float(node["meters"]) if "meters" in node else 0.1
        return skills.lift(meters)
    if kind == "pick":
        return skills.pick()
    return skills.place()


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
