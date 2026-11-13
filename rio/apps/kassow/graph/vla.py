# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a registered VLA from a graph node through the orchestrator's own rollout code.

The orchestrator is not changed: its ``make_run_vla`` runs the rollout, with the
same stop rules (open: first gripper opening; others: plan back near home,
timeout, or ``s``), the same ``x`` abort, and the same alarm handling. A graph
node only adds the moveJ to the VLA's trained start pose before the rollout.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rio.apps.kassow.experiments import resolve_experiment
from rio.apps.terminal import term

# The graph runs on the orchestrator's station, which is set up for this experiment's cameras.
GRAPH_STATION = "open"

RunVla = Callable[[str, str, float], object]


def vla_usable(experiment: str) -> bool:
    """True when the experiment's overview camera is the one the graph station runs."""
    return resolve_experiment(experiment).overview_serial == resolve_experiment(GRAPH_STATION).overview_serial


def finished(experiment: str, reason: str) -> bool:
    """Whether a rollout ``reason`` counts as done, as the orchestrator judges it."""
    return reason == "stop_when" if experiment == GRAPH_STATION else reason in ("stop_when", "operator", "timeout")


def make_run_vla(env, keys, skills, log_root: Path, *, device: str, single_slot: bool, mw: str) -> RunVla:
    """``run_vla(experiment, instruction, timeout_s)`` for the cell. True when done, else the reason.

    Args:
        env: Started environment of the orchestrator's station.
        keys: Keyboard; ``s`` ends and ``x`` aborts the rollout.
        skills: Arm skills; ``move_to`` brings the arm to the VLA's start pose.
        log_root: Folder for the rollout diagnostics.
        device: GPU of the resident policy.
        single_slot: Keep one resident and reload on every experiment switch.
        mw: Middleware of the policy node.
    """
    from rio.apps.kassow.orchestrator.actions import Args
    from rio.apps.kassow.orchestrator.actions import make_run_vla as orchestrator_run_vla
    from rio.apps.kassow.orchestrator.machine import CycleLog, VlaJob

    log = CycleLog(log_root)

    def run(experiment: str, instruction: str, timeout_s: float) -> object:
        home = resolve_experiment(experiment).select_home(instruction)
        moved = skills.move_to(home)
        if moved is not True:
            return f"start pose not reached: {moved}"
        args = Args(device=device, single_slot=single_slot, mw=mw, open_timeout_s=timeout_s, organize_timeout_s=timeout_s)
        reason = orchestrator_run_vla(args, env, keys, log)(VlaJob(experiment, instruction, home))
        return True if finished(experiment, reason) else reason

    return run


def dry_run_vla(experiment: str, instruction: str, timeout_s: float) -> object:
    term(f"[dry run] would moveJ to the {experiment} home and run its VLA {instruction!r} for up to {timeout_s:.0f} s")
    return True
