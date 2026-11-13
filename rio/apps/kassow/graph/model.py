# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""A Graph-as-Policy file: one task, named nodes, and the edges between them.

The orchestrator is a fixed graph for the crate pilot. This file is a separate
graph an agent writes for one trial. Loading it does not move the arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKILL_KINDS = ("move_to", "lift", "pick", "place", "gripper", "pick_at", "place_at", "move_above")
# Skills that act at a target stored by a locate node.
TARGET_KINDS = ("pick_at", "place_at", "move_above")
STOP_KIND = "stop"
JEV_KIND = "jev"
LOCATE_KIND = "locate"
VLA_KIND = "vla"
QUESTION_TYPES = ("noul", "choice", "score", "box")
GRIPPER_STATES = ("open", "close")
EXAMPLE_PATH = Path(__file__).resolve().parent / "example.json"
PICK_EXAMPLE_PATH = Path(__file__).resolve().parent / "pick_cube.json"

APPROACH_M = 0.10
APPROACH_M_RANGE = (0.03, 0.25)
# pick_at may only raise the calibrated grasp height, never lower it.
Z_OFFSET_M_RANGE = (0.0, 0.10)
PLACE_Z_OFFSET_M = 0.01
PLACE_Z_OFFSET_M_RANGE = (0.0, 0.15)
HOVER_M = 0.10
HOVER_M_RANGE = (0.03, 0.30)
VLA_TIMEOUT_S = 60.0
VLA_TIMEOUT_S_RANGE = (5.0, 300.0)
OBJECT_MAX_CHARS = 120
INSTRUCTION_MAX_CHARS = 200


class GraphError(ValueError):
    """The graph file is missing a field or names a node that does not exist."""


@dataclass(frozen=True)
class Graph:
    """One task graph. ``nodes`` keeps the JSON object for each id."""

    task: str
    start: str
    max_steps: int
    nodes: dict[str, dict]
    path: Path | None = None

    def node(self, name: str) -> dict:
        """The JSON object of one node.

        Raises:
            GraphError: If ``name`` is not in the file.
        """
        try:
            return self.nodes[name]
        except KeyError as error:
            raise GraphError(f"node {name!r} is not in the graph") from error

    def needs_table(self) -> bool:
        """True when a node locates an object or acts at a located one."""
        return any(node.get("kind") in (LOCATE_KIND, *TARGET_KINDS) for node in self.nodes.values())


def load_graph(path: Path) -> Graph:
    """Read one graph JSON file.

    Raises:
        GraphError: If the file is not an object or omits task, start, max_steps, or nodes.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise GraphError("a graph file must be a JSON object")
    task = raw.get("task")
    start = raw.get("start")
    max_steps = raw.get("max_steps")
    nodes = raw.get("nodes")
    if not isinstance(task, str) or not task.strip():
        raise GraphError("task must be a non-empty string")
    if not isinstance(start, str) or not start:
        raise GraphError("start must name a node")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise GraphError("max_steps must be an integer of at least 1")
    if not isinstance(nodes, dict) or not nodes:
        raise GraphError("nodes must be a non-empty object")
    for name, node in nodes.items():
        if not isinstance(node, dict):
            raise GraphError(f"node {name!r} must be an object")
    return Graph(task=task.strip(), start=start, max_steps=max_steps, nodes=nodes, path=path)


def example_graph() -> Graph:
    """The shipped sample: look at the crate, then move to the organize home."""
    return load_graph(EXAMPLE_PATH)
