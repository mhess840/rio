# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph as Policy: an agent plans one task as a JSON graph, the cell walks it.

Separate from the crate orchestrator, which stays a fixed graph; its rollout code
is reused for ``vla`` nodes, not changed. Loop for any task:

1. ``describe``: node kinds, limits, poses, VLAs and their instructions, table map status.
2. ``look`` / ``ask``: photos of the Jev, overview and wrist cameras; Jev questions or ``--locate``.
3. Write the graph, ``check`` it, walk it with ``run`` (dry-run first), read ``events.jsonl``, replan.

Node kinds (``describe`` has the fields; examples: ``example.json``, ``pick_cube.json``, ``pick_place.json``):
``move_to``, ``lift``, ``pick``, ``place``, ``gripper``; ``locate`` and the target skills ``pick_at``,
``place_at``, ``move_above``; ``vla``; ``jev``; ``stop``.

    uv run python -m rio.apps.kassow.graph --command describe
    uv run python -m rio.apps.kassow.graph --command look
    uv run python -m rio.apps.kassow.graph --command ask --locate "the red cube"
    uv run python -m rio.apps.kassow.graph --command check --graph my_task.json
    uv run python -m rio.apps.kassow.graph --command run --graph my_task.json            # dry run
    uv run python -m rio.apps.kassow.graph --command run --graph my_task.json --no-dry-run
"""

from .check import check_graph, require_graph
from .model import EXAMPLE_PATH, PICK_EXAMPLE_PATH, Graph, GraphError, example_graph, load_graph
from .walk import WalkLog, walk

__all__ = [
    "EXAMPLE_PATH",
    "PICK_EXAMPLE_PATH",
    "Graph",
    "GraphError",
    "WalkLog",
    "check_graph",
    "example_graph",
    "load_graph",
    "require_graph",
    "walk",
]
