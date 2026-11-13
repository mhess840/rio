# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Check a graph file before any skill runs. This stands in for the simulation."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from rio.apps.kassow.experiments import EXPERIMENTS, home_file, resolve_experiment

from .model import (
    APPROACH_M_RANGE,
    GRIPPER_STATES,
    HOVER_M_RANGE,
    INSTRUCTION_MAX_CHARS,
    JEV_KIND,
    LOCATE_KIND,
    OBJECT_MAX_CHARS,
    PLACE_Z_OFFSET_M_RANGE,
    QUESTION_TYPES,
    SKILL_KINDS,
    STOP_KIND,
    TARGET_KINDS,
    VLA_KIND,
    VLA_TIMEOUT_S_RANGE,
    Z_OFFSET_M_RANGE,
    Graph,
    GraphError,
)
from .vla import GRAPH_STATION, vla_usable

NOUL_YES = 0.7
BOX_YES = 0.6
STORE_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
EDGE_KEYS = ("ok", "fail", "yes", "no", "high", "low", "else", "found", "missing")


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


def question_problems(questions: object) -> list[str]:
    """Problems of a Jev question map asked outside a graph. Empty when it may be sent."""
    if not isinstance(questions, dict) or not questions:
        return ["questions must be a non-empty object"]
    return [problem for key, question in questions.items() for problem in _check_question("ask", str(key), question)]


def _check_node(name: str, node: dict, graph: Graph, pose_exists: Callable[[str], bool]) -> list[str]:
    kind = node.get("kind")
    if kind in SKILL_KINDS:
        return _check_skill(name, node, graph, pose_exists)
    if kind == JEV_KIND:
        return _check_jev(name, node, graph)
    if kind == LOCATE_KIND:
        return _check_locate(name, node, graph)
    if kind == VLA_KIND:
        return _check_vla(name, node, graph)
    if kind == STOP_KIND:
        reason = node.get("reason", "")
        if not isinstance(reason, str):
            return [f"node {name!r} stop reason must be a string"]
        return []
    return [f"node {name!r} kind must be a skill, jev, locate, vla, or stop, got {kind!r}"]


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
    if node["kind"] == "gripper" and node.get("state") not in GRIPPER_STATES:
        problems.append(f"node {name!r} gripper state must be one of {GRIPPER_STATES}")
    if node["kind"] in TARGET_KINDS:
        target = node.get("target")
        if not isinstance(target, str) or target not in _stored(graph):
            problems.append(f"node {name!r} {node['kind']} target must be the store of a locate node")
    if node["kind"] == "pick_at":
        _in_range(name, node, "approach_m", APPROACH_M_RANGE, problems)
        _in_range(name, node, "z_offset_m", Z_OFFSET_M_RANGE, problems)
    if node["kind"] == "place_at":
        _in_range(name, node, "approach_m", APPROACH_M_RANGE, problems)
        _in_range(name, node, "z_offset_m", PLACE_Z_OFFSET_M_RANGE, problems)
    if node["kind"] == "move_above":
        _in_range(name, node, "height_m", HOVER_M_RANGE, problems)
    return problems


def _check_vla(name: str, node: dict, graph: Graph) -> list[str]:
    problems: list[str] = []
    _edge(name, node, "ok", graph, problems)
    _edge(name, node, "fail", graph, problems)
    _in_range(name, node, "timeout_s", VLA_TIMEOUT_S_RANGE, problems)
    experiment = node.get("experiment")
    if experiment not in EXPERIMENTS:
        problems.append(f"node {name!r} vla experiment must be one of {sorted(EXPERIMENTS)}")
        return problems
    if not vla_usable(str(experiment)):
        problems.append(
            f"node {name!r} vla {experiment!r} needs overview camera {resolve_experiment(str(experiment)).overview_serial}; "
            f"the graph station runs the {GRAPH_STATION!r} cameras"
        )
    instruction = node.get("instruction", "")
    if not isinstance(instruction, str) or len(instruction) > INSTRUCTION_MAX_CHARS:
        problems.append(f"node {name!r} vla instruction must be a string of at most {INSTRUCTION_MAX_CHARS} characters")
        return problems
    try:
        EXPERIMENTS[str(experiment)].select_home(instruction)
    except ValueError as error:
        problems.append(f"node {name!r} vla: {error}")
    return problems


def _stored(graph: Graph) -> set[str]:
    return {str(node.get("store")) for node in graph.nodes.values() if node.get("kind") == LOCATE_KIND}


def _in_range(name: str, node: dict, key: str, bounds: tuple[float, float], problems: list[str]) -> None:
    if key not in node:
        return
    value = node[key]
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        problems.append(f"node {name!r} {key} must be a number from {low} to {high}")


def _check_locate(name: str, node: dict, graph: Graph) -> list[str]:
    problems: list[str] = []
    what = node.get("object")
    if not isinstance(what, str) or not what.strip() or len(what) > OBJECT_MAX_CHARS or "`" in what:
        problems.append(
            f"node {name!r} locate object must be a description of at most {OBJECT_MAX_CHARS} characters, no backticks"
        )
    store = node.get("store")
    if not isinstance(store, str) or not STORE_NAME.fullmatch(store):
        problems.append(f"node {name!r} locate store must be a lower-case name like 'cube'")
    for key in ("threshold", "visible_threshold"):
        if key in node and not _unit(node[key]):
            problems.append(f"node {name!r} {key} must be a number from 0 to 1")
    _edge(name, node, "found", graph, problems)
    _edge(name, node, "missing", graph, problems)
    if "fail" in node:
        _edge(name, node, "fail", graph, problems)
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
        if name in seen or name not in graph.nodes:
            continue
        seen.add(name)
        node = graph.nodes[name]
        for key, value in node.items():
            if key in EDGE_KEYS and isinstance(value, str):
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
