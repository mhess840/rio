# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Check a graph file before any skill runs. This stands in for the simulation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rio.apps.kassow.experiments import home_file

from .model import JEV_KIND, QUESTION_TYPES, SKILL_KINDS, STOP_KIND, Graph, GraphError

NOUL_YES = 0.7
BOX_YES = 0.6


def pose_is_file(name: str) -> bool:
    """True when ``name`` is a captured or packaged home snapshot."""
    return home_file(name).is_file()


def check_graph(graph: Graph, *, pose_exists: Callable[[str], bool] = pose_is_file) -> list[str]:
    """Every problem in the file. An empty list means the runner may start.

    Args:
        graph: A loaded graph.
        pose_exists: True when a ``move_to`` pose name resolves to a file.
    """
    problems: list[str] = []
    if graph.start not in graph.nodes:
        problems.append(f"start {graph.start!r} is not a node")
    for name, node in graph.nodes.items():
        problems.extend(_check_node(name, node, graph, pose_exists))
    if graph.start in graph.nodes:
        problems.extend(_unreachable(graph))
    return problems


def require_graph(graph: Graph, *, pose_exists: Callable[[str], bool] = pose_is_file) -> None:
    """Raise when :func:`check_graph` finds anything.

    Raises:
        GraphError: The file is not runnable. The message lists every problem.
    """
    problems = check_graph(graph, pose_exists=pose_exists)
    if problems:
        raise GraphError("\n".join(problems))


def _check_node(name: str, node: dict, graph: Graph, pose_exists: Callable[[str], bool]) -> list[str]:
    kind = node.get("kind")
    if kind in SKILL_KINDS:
        return _check_skill(name, node, graph, pose_exists)
    if kind == JEV_KIND:
        return _check_jev(name, node, graph)
    if kind == STOP_KIND:
        reason = node.get("reason", "")
        if not isinstance(reason, str):
            return [f"node {name!r} stop reason must be a string"]
        return []
    return [f"node {name!r} kind must be a skill, jev, or stop, got {kind!r}"]


def _edge(name: str, node: dict, key: str, graph: Graph, problems: list[str]) -> None:
    target = node.get(key)
    if not isinstance(target, str) or target not in graph.nodes:
        problems.append(f"node {name!r} edge {key!r} must name a node")


def _check_skill(name: str, node: dict, graph: Graph, pose_exists: Callable[[str], bool]) -> list[str]:
    problems: list[str] = []
    _edge(name, node, "ok", graph, problems)
    _edge(name, node, "fail", graph, problems)
    if node["kind"] == "move_to":
        pose = node.get("pose")
        if not isinstance(pose, str) or not pose:
            problems.append(f"node {name!r} move_to needs a pose file name")
        elif not pose_exists(pose):
            problems.append(f"node {name!r} pose {pose!r} is not a home file")
    if node["kind"] == "lift" and "meters" in node:
        meters = node["meters"]
        if isinstance(meters, bool) or not isinstance(meters, (int, float)) or not meters > 0:
            problems.append(f"node {name!r} lift meters must be a positive number")
    return problems


def _check_jev(name: str, node: dict, graph: Graph) -> list[str]:
    problems: list[str] = []
    questions = node.get("questions")
    if not isinstance(questions, dict) or not questions:
        return [f"node {name!r} jev needs a non-empty questions object"]
    for key, question in questions.items():
        problems.extend(_check_question(name, key, question))
    route = node.get("route")
    if not isinstance(route, str) or route not in questions:
        problems.append(f"node {name!r} route must name one of its questions")
        return problems
    question = questions[route]
    if not isinstance(question, dict):
        return problems
    kind = question.get("type")
    if kind in ("noul", "box"):
        _edge(name, node, "yes", graph, problems)
        _edge(name, node, "no", graph, problems)
        if "threshold" in node and not _unit(node["threshold"]):
            problems.append(f"node {name!r} threshold must be a number from 0 to 1")
    elif kind == "choice":
        when = node.get("when")
        if not isinstance(when, dict) or not when:
            problems.append(f"node {name!r} choice route needs a when map")
        else:
            criteria = question.get("criteria")
            names = set(criteria) if isinstance(criteria, dict) else set()
            for option, target in when.items():
                if option not in names:
                    problems.append(f"node {name!r} when option {option!r} is not a choice criterion")
                if not isinstance(target, str) or target not in graph.nodes:
                    problems.append(f"node {name!r} when {option!r} must name a node")
            if "else" in node:
                _edge(name, node, "else", graph, problems)
            elif isinstance(criteria, dict) and set(when) != set(criteria):
                problems.append(f"node {name!r} choice route needs an else edge for options without when")
    elif kind == "score":
        _edge(name, node, "high", graph, problems)
        _edge(name, node, "low", graph, problems)
        if "threshold" not in node or isinstance(node["threshold"], bool) or not isinstance(node["threshold"], (int, float)):
            problems.append(f"node {name!r} score route needs a numeric threshold")
    return problems


def _check_question(name: str, key: str, question: object) -> list[str]:
    if not isinstance(question, dict):
        return [f"node {name!r} question {key!r} must be an object"]
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        return [f"node {name!r} question {key!r} type must be one of {QUESTION_TYPES}"]
    instructions = question.get("instructions")
    if not isinstance(instructions, (str, dict, list)) or instructions == "" or instructions == []:
        return [f"node {name!r} question {key!r} needs instructions"]
    problems: list[str] = []
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            problems.append(f"node {name!r} question {key!r} choice needs at least two criteria")
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            problems.append(f"node {name!r} question {key!r} score needs 2 to 10 levels")
    elif kind == "noul" and "criteria" in question:
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or not {"true", "false"} <= set(criteria):
            problems.append(f"node {name!r} question {key!r} noul criteria need true and false")
    elif kind == "box" and "criteria" in question:
        problems.append(f"node {name!r} question {key!r} box takes no criteria")
    return problems


def _unit(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _unreachable(graph: Graph) -> list[str]:
    seen: set[str] = set()
    stack = [graph.start]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        node = graph.nodes[name]
        for key, value in node.items():
            if key in ("ok", "fail", "yes", "no", "high", "low", "else") and isinstance(value, str):
                stack.append(value)
            elif key == "when" and isinstance(value, dict):
                stack.extend(target for target in value.values() if isinstance(target, str))
    missing = [name for name in graph.nodes if name not in seen]
    return [f"node {name!r} is not reachable from {graph.start!r}" for name in missing]


def resolve_pose(name: str) -> Path:
    """Captured home if one exists, otherwise the packaged file of the same name."""
    path = home_file(name)
    if not path.is_file():
        raise GraphError(f"pose {name!r} is not a home file")
    return path
