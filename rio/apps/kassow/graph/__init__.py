# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph as Policy for one Kassow trial, beside the crate orchestrator.

An agent writes a JSON graph. ``check`` confirms every node, pose, and edge.
``run --dry-run`` walks it without moving the arm. ``run --no-dry-run`` uses the
cell. The orchestrator, its crate decisions, and the experiment registry are
not part of this package.

    uv run python -m rio.apps.kassow.graph check
    uv run python -m rio.apps.kassow.graph run --dry-run
    uv run python -m rio.apps.kassow.graph run --no-dry-run path/to/graph.json
"""

from .check import check_graph, require_graph
from .model import EXAMPLE_PATH, Graph, GraphError, example_graph, load_graph
from .walk import WalkLog, walk

__all__ = [
    "EXAMPLE_PATH",
    "Graph",
    "GraphError",
    "WalkLog",
    "check_graph",
    "example_graph",
    "load_graph",
    "require_graph",
    "walk",
]
