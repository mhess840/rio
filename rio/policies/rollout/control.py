# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Who decides when a rollout runs, pauses, and ends."""

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutControl:
    """Run one policy rollout under the caller's stop and completion conditions."""

    stop_when: Callable[[np.ndarray], bool] | None = None
    plan_done: Callable[[np.ndarray], bool] | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    reason: str = ""
    finish_plan: bool = False

    def note_plan(self, waypoints: np.ndarray) -> bool:
        """Latch completion so the remaining plan drains without replanning."""
        if not self.finish_plan and self.plan_done is not None:
            self.finish_plan = bool(self.plan_done(waypoints))
        return self.finish_plan

    def should_stop(self, last_sent: np.ndarray, has_action: bool) -> bool:
        """Check explicit stops and only evaluate commands actually sent."""
        if self.stop.is_set():
            self.reason = self.reason or "stopped"
            return True
        if has_action and self.stop_when is not None and self.stop_when(last_sent):
            self.reason = "stop_when"
            return True
        return False


def apply_serve_control(cmd: str, pause: threading.Event, quit_ev: threading.Event) -> None:
    """Map a stdin line to pause/resume/quit. Empty line toggles run vs pause."""
    cmd = cmd.strip().lower()
    if cmd in ("q", "quit"):
        quit_ev.set()
        return
    if cmd in ("p", "pause"):
        pause.set()
        return
    if pause.is_set():
        pause.clear()
    else:
        pause.set()


def _watch_stdin(pause: threading.Event, quit_ev: threading.Event) -> None:
    while not quit_ev.is_set():
        line = sys.stdin.readline()
        if line == "":
            quit_ev.set()
            return
        apply_serve_control(line, pause, quit_ev)


def start_stdin_control(pause: threading.Event, quit_ev: threading.Event) -> None:
    """Enter toggles run/pause; ``q`` quits. Daemon thread so Ctrl+C can still exit."""
    threading.Thread(target=_watch_stdin, args=(pause, quit_ev), daemon=True).start()
