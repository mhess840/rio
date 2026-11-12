"""Jev decides, the VLAs act: organize the crate inlay, or load the next free space.

On ``j`` the Jev camera photographs the crate and Jev answers three questions
(``rio.apps.kassow.crate``). Then:

- inlay crumpled: moveJ to the organize home, run the organize VLA until a
  predicted plan ends within 20 cm of that pose. That chunk is played out
  and no further plan is requested. When the arm has stopped, ask Jev again.
  ``s`` and the timeout still cut the rollout short.
- inlay organized, N cartons: moveJ to the home of space N+1, run the open VLA
  with "put the object in space N+1", and stop it at its first open-gripper
  command. The arm stays in the compartment, holding it open, until ``p``.
  Then it rises 10 cm and moves to the organize home, and Jev looks again.
  Every start begins with a moveJ to that same home.
- anything unsure (inlay state, count, missing space home, VLA error): hold,
  print why, and wait for ``j`` or ``q``.
- a Kassow alarm during the moveJ or the VLA: remember that phase and wait.
  Releasing the emergency stop does not move the arm. ``c`` clears the latch
  in the same session and repeats the phase from the measured pose. If the
  gripper is already holding the object, the grasp is not repeated.

Keys: ``j`` ask Jev, ``p`` proceed, ``s`` end the running VLA normally, ``x``
abort it (hold), ``c`` clear the Kassow alarm and continue, ``q`` quit.
Every cycle is logged to ``experiments/orchestrator/runs/<time>/``.
With ``--board`` a browser at http://127.0.0.1:8765 shows the state and the
latest Jev reading while the run is going.

Package layout: ``machine`` is the state machine over injected actions (no
hardware, fully unit-tested), ``actions`` builds those actions for the cell,
and ``run`` wires them together.

Decision logic with the real Jev camera, nothing moves::

    export TYPESAFE_API_KEY=...
    uv run python -m rio.apps.kassow.orchestrator --dry-run

Full run, both VLAs resident on one GPU (``--single-slot`` reloads on each switch)::

    uv run --group pi --no-group docs -m rio.apps.kassow.orchestrator --device cuda:1
"""

from .actions import KEYBOARD_ADDR, Args, NodeKeys, station_cfg, vla_cfg
from .machine import (
    JOB_PHASES,
    OPEN,
    ORGANIZE,
    CycleLog,
    GripperOpened,
    Keys,
    Orchestrator,
    ReturnedNearHome,
    State,
    VlaJob,
    wait_key,
)
from .run import cli, main

__all__ = [
    "JOB_PHASES",
    "KEYBOARD_ADDR",
    "OPEN",
    "ORGANIZE",
    "Args",
    "CycleLog",
    "GripperOpened",
    "Keys",
    "NodeKeys",
    "Orchestrator",
    "ReturnedNearHome",
    "State",
    "VlaJob",
    "cli",
    "main",
    "station_cfg",
    "vla_cfg",
    "wait_key",
]
