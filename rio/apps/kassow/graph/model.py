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

SKILL_KINDS = ("move_to", "lift", "pick", "place")
STOP_KIND = "stop"
JEV_KIND = "jev"
QUESTION_TYPES = ("noul", "choice", "score", "box")
EXAMPLE_PATH = Path(__file__).resolve().parent / "example.json"


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
