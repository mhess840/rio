# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator state machine (the "task graph" of the case study). It moves nothing itself; every action is injected."""

from __future__ import annotations

import enum
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger
from rio_hw import time

from rio.apps.kassow.crate import CrateReading
from rio.apps.terminal import term
from rio.arm_resume import Phase, PhaseKind, decide_resume
from rio.jev import JevError

ORGANIZE = "organize"
OPEN = "open"


class Keys(Protocol):
    def pressed(self) -> set[str]: ...


def wait_key(keys: Keys, allowed: str, poll_s: float = 0.02) -> str:
    """Block until one of ``allowed`` is pressed and return it."""
    while True:
        hit = sorted(k for k in keys.pressed() if k in allowed)
        if hit:
            return hit[0]
        time.sleep(poll_s)


class ReturnedNearHome:
    """True when a predicted chunk ends back near the organize home.

    The rollout starts at the home pose, so a plan that never leaves the ball
    does not count. After any waypoint has been farther than ``radius_m``,
    a chunk whose last point is inside the ball is the last plan to play.
    """

    def __init__(self, home_xyz: np.ndarray, radius_m: float = 0.20):
        self.home = np.asarray(home_xyz, dtype=float).reshape(-1)[:3]
        self.radius_m = float(radius_m)
        if self.home.shape != (3,) or not np.isfinite(self.home).all():
            raise ValueError("home_xyz must be three finite coordinates")
        if not self.radius_m > 0.0:
            raise ValueError("radius_m must be positive")
        self.left = False

    def consider(self, waypoints: np.ndarray) -> bool:
        """Look at one planned TCP path. True when its end has come back."""
        points = np.asarray(waypoints, dtype=float).reshape(-1, 3)[:, :3]
        if not len(points) or not np.isfinite(points).all():
            return False
        distance = np.linalg.norm(points - self.home, axis=1)
        if np.any(distance > self.radius_m):
            self.left = True
        return self.left and float(distance[-1]) <= self.radius_m


class GripperOpened:
    """``stop_when`` for the open VLA: True at the first open command after a closed one.

    The rollout starts with the object gripped. A command that is already open
    before any closed one (a stale opening) does not count.
    """

    def __init__(self, gripper_close: float, gripper_open: float):
        if not gripper_close < gripper_open:
            raise ValueError("gripper_close must be below gripper_open")
        self.threshold = 0.5 * (gripper_close + gripper_open)
        self.armed = False

    def __call__(self, command: np.ndarray) -> bool:
        value = float(np.asarray(command).reshape(-1)[-1])
        if value < self.threshold:
            self.armed = True
            return False
        return self.armed


@dataclass(frozen=True)
class VlaJob:
    """One VLA rollout the orchestrator asks for."""

    experiment: str
    instruction: str
    home: Path


class State(enum.Enum):
    IDLE = "idle"
    ASK = "ask"
    GOTO = "goto"
    RUN_VLA = "run_vla"
    WAIT_PROCEED = "wait_proceed"
    WAIT_RESET = "wait_reset"
    HOLD = "hold"
    DONE = "done"


# The moveJ and the VLA are the two phases this cell can repeat. A later
# pick-and-place arm uses the same resume rules with grasp phases inserted.
JOB_PHASES = (
    Phase("movej", PhaseKind.MOVE),
    Phase("vla", PhaseKind.MOVE),
)


def recover_unavailable() -> bool:
    """No arm is connected, so a clear cannot succeed."""
    return False


@dataclass
class CycleLog:
    """One JSON line per event under ``runs/<time>/``, plus each Jev photo."""

    root: Path
    cycle: int = 0

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def vla_dir(self) -> Path:
        return self.root / "vla"

    def image_path(self) -> Path:
        return self.root / f"cycle_{self.cycle:03d}.jpg"

    def write(self, event: str, **payload) -> None:
        record = {"t": datetime.now().isoformat(timespec="seconds"), "cycle": self.cycle, "event": event, **payload}
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


@dataclass(frozen=True)
class Snapshot:
    """What an observer such as the demo board may read."""

    state: State
    job: VlaJob | None
    hold_reason: str
    reading: CrateReading | None
    after_goto: State
    alarm_phase: str
    photo: Path


@dataclass
class InlayTrust:
    """The next photo skips the rib check once the organize VLA has finished."""

    skip: bool = False


@dataclass
class CartonCounter:
    """Cartons a finished place leaves in the crate.

    ``expected`` stays None until the operator presses ``p`` after the open VLA.
    The count decision reads it only when ``CrateThresholds.count_prior`` is on.
    """

    expected: int | None = None


@dataclass
class Move:
    """One queued ``moveJ``. ``lift_first`` raises the TCP before it and is cleared once that has run."""

    path: Path
    lift_first: bool = False


class Orchestrator:
    """State machine over injected actions, so a dry run and the robot share one logic.

    Args:
        ask: Photograph the crate and return Jev's reading.
        goto: moveJ to a home file; False if it did not land.
        run_vla: Run one rollout; returns ``stop_when``, ``operator``, ``timeout``,
            ``aborted``, ``alarm``, ``alarm_holding``, or ``error: ...``.
        recover: Clear the controller alarm in this session. False leaves the
            cell waiting; it must not start a motion.
        keys: Key source.
        organize_home: Home of the organize VLA, also the rest pose after ``p``.
        spaces: Home per 1-based space of the open VLA.
        log: Event log.
        max_organize_attempts: Organize rollouts before a hold asks the operator. Pressing
            ``j`` in the hold grants a fresh budget; an ``open`` reading resets it too.
    """

    def __init__(
        self,
        *,
        ask: Callable[[], CrateReading],
        goto: Callable[[Path], bool | str],
        run_vla: Callable[[VlaJob], str],
        keys: Keys,
        recover: Callable[[], bool] | None = None,
        organize_home: Path,
        spaces: dict[int, Path],
        log: CycleLog,
        max_organize_attempts: int = 3,
        counter: CartonCounter | None = None,
        inlay_trust: InlayTrust | None = None,
    ):
        if max_organize_attempts < 1:
            raise ValueError("max_organize_attempts must be at least 1")
        self.ask = ask
        self.goto = goto
        self.run_vla = run_vla
        self.recover = recover if recover is not None else recover_unavailable
        self.keys = keys
        self.organize_home = organize_home
        self.wait_home: Path | None = None
        self.at_wait_pose = False
        self.spaces = spaces
        self.log = log
        self.counter = counter if counter is not None else CartonCounter()
        self.inlay_trust = inlay_trust if inlay_trust is not None else InlayTrust()
        self.max_organize_attempts = max_organize_attempts
        self.organize_attempts = 0
        self.state = State.IDLE
        self.job: VlaJob | None = None
        self.hold_reason = ""
        self.phases = JOB_PHASES
        self.phase_index = 0
        self.alarm_holding = False
        self.moves: list[Move] = []
        self.after_goto = State.RUN_VLA
        self.lift: Callable[[], bool | str] = lambda: True
        self.reading: CrateReading | None = None
        self.board = None
        self._handlers: dict[State, Callable[[], State]] = {
            State.IDLE: self._idle,
            State.ASK: self._ask,
            State.GOTO: self._goto,
            State.RUN_VLA: self._run,
            State.WAIT_PROCEED: self._wait_proceed,
            State.WAIT_RESET: self._reset,
            State.HOLD: self._held,
        }

    def _hold(self, reason: str) -> State:
        self.hold_reason = reason
        self.log.write("hold", reason=reason)
        return State.HOLD

    def _show(self) -> None:
        if self.board is not None:
            self.board.publish(self)

    def snapshot(self) -> Snapshot:
        return Snapshot(
            state=self.state,
            job=self.job,
            hold_reason=self.hold_reason,
            reading=self.reading,
            after_goto=self.after_goto,
            alarm_phase=self.phases[self.phase_index].name,
            photo=self.log.image_path(),
        )

    def step(self) -> State:
        """Run one state and return the next."""
        self._show()
        handler = self._handlers.get(self.state)
        if handler is not None:
            self.state = handler()
        self._show()
        return self.state

    def _idle(self) -> State:
        term("\nIdle. j: ask Jev | q: quit")
        return State.ASK if wait_key(self.keys, "jq") == "j" else State.DONE

    def _wait_proceed(self) -> State:
        term("\nHolding the compartment open. Place the object. p: move clear and look again | q: quit")
        if wait_key(self.keys, "pq") == "q":
            return State.DONE
        # ``p`` means the carton went into the compartment the photo had chosen.
        if self.reading is not None and self.reading.space is not None:
            self.counter.expected = self.reading.space
            self.log.write("cartons", expected=self.counter.expected)
        return self._begin_goto([self.wait_home or self.organize_home], State.ASK, lift_first=True)

    def _held(self) -> State:
        term(f"\nHOLD: {self.hold_reason}\nj: ask Jev again | q: quit")
        if wait_key(self.keys, "jq") == "j":
            # The operator has looked at the cell; organize may be retried.
            self.organize_attempts = 0
            return State.ASK
        return State.DONE

    def _ask(self) -> State:
        self.log.cycle += 1
        try:
            reading = self.ask()
            self.inlay_trust.skip = False
        except (JevError, ValueError, RuntimeError) as error:
            self.inlay_trust.skip = False
            return self._hold(f"Jev request failed: {error}")
        self.reading = reading
        term(f"Jev: {reading.summary()}")
        self.log.write("reading", reading=asdict(reading))
        self._show()
        if reading.decision == "organize":
            if self.organize_attempts >= self.max_organize_attempts:
                return self._hold(
                    f"inlay still not organized after {self.organize_attempts} organize run(s); fix it by hand, then press j"
                )
            self.organize_attempts += 1
            self.job = VlaJob(ORGANIZE, "", self.organize_home)
            return self._begin_goto([self.job.home], State.RUN_VLA)
        if reading.decision == "open" and reading.space in self.spaces:
            self.organize_attempts = 0
            self.job = VlaJob(OPEN, f"put the object in space {reading.space}", self.spaces[reading.space])
            if self.wait_home is not None and not self.at_wait_pose:
                return self._begin_goto([self.wait_home, self.job.home], State.RUN_VLA)
            return self._begin_goto([self.job.home], State.RUN_VLA)
        return self._hold(reading.reason)

    def _begin_goto(self, paths: list[Path], after: State, *, lift_first: bool = False) -> State:
        """Queue moveJs, one per path. The motion itself happens in ``GOTO``."""
        self.moves = [Move(path) for path in paths]
        self.moves[0].lift_first = lift_first
        self.after_goto = after
        return State.GOTO

    def _goto(self) -> State:
        """Run the first queued moveJ. An alarm waits for ``c`` instead of holding."""
        move = self.moves[0]
        if move.lift_first:
            lifted = self.lift()
            self.log.write("lift", ok=lifted is True)
            if lifted in ("alarm", "alarm_holding"):
                return self._alarm(0, lifted == "alarm_holding")
            move.lift_first = False
            if lifted is not True:
                return self._hold("could not lift clear of the crate")
        outcome = self.goto(move.path)
        self.log.write("goto", home=move.path.name, ok=outcome is True)
        if outcome in ("alarm", "alarm_holding"):
            return self._alarm(0, outcome == "alarm_holding")
        if not outcome:
            self.moves.clear()
            return self._hold(f"moveJ to {move.path.name} failed")
        self.moves.pop(0)
        # The flag means the arm is still at the waiting pose. A later moveJ, including the
        # compartment after the approach, leaves that pose.
        self.at_wait_pose = self.wait_home is not None and move.path == self.wait_home
        return State.GOTO if self.moves else self.after_goto

    def _alarm(self, index: int, holding: bool) -> State:
        """Remember the phase and wait. The arm does not move in this state."""
        self.at_wait_pose = False
        self.phase_index = index
        self.alarm_holding = holding
        self.log.write("alarm", phase=self.phases[index].name, holding=holding)
        return State.WAIT_RESET

    def _reset(self) -> State:
        """Clear the latch after ``c``, then repeat or continue the saved phase."""
        phase = self.phases[self.phase_index]
        term(f"\nArm alarm during {phase.name}. Release the E-stop, then c: continue | q: quit")
        if wait_key(self.keys, "cq") == "q":
            return State.DONE
        try:
            cleared = bool(self.recover())
        except Exception:
            logger.exception("Alarm clear failed")
            cleared = False
        if not cleared:
            self.log.write("recover_failed")
            return State.WAIT_RESET
        plan = decide_resume(self.phases, self.phase_index, holding=self.alarm_holding)
        try:
            phase = plan.phase(self.phases)
        except IndexError:
            return self._hold("nothing left after the alarm")
        if self.alarm_holding and phase.kind is PhaseKind.GRASP:
            return self._hold("refusing to repeat a grasp while holding")
        self.phase_index = plan.index
        self.log.write("resume", action=plan.action.value, phase=phase.name, holding=self.alarm_holding)
        if phase.name == "vla":
            return State.RUN_VLA
        if phase.name == "movej":
            return State.GOTO
        return self._hold(f"no motion for phase {phase.name}")

    def _run(self) -> State:
        job = self.job
        assert job is not None
        if job.experiment == ORGANIZE:
            term(
                "\nRunning organize VLA. A plan that ends within 20 cm of home is played out, "
                "then the arm stops. s: end | x: abort"
            )
        else:
            term(f"\nRunning {job.experiment} VLA {job.instruction!r}. s: end | x: abort")
        reason = self.run_vla(job)
        self.log.write("vla", experiment=job.experiment, instruction=job.instruction, reason=reason)
        if reason in ("alarm", "alarm_holding"):
            return self._alarm(1, reason == "alarm_holding")
        if reason == "aborted" or reason.startswith("error"):
            return self._hold(f"{job.experiment} VLA {reason}")
        if job.experiment == ORGANIZE:
            # A timeout ends the rollout like ``s``. The plan may stop 20 cm short of home,
            # so return to that same initial home before the next photo.
            if reason not in ("stop_when", "operator", "timeout"):
                return self._hold(f"organize ended without a completion condition ({reason})")
            # The VLA has straightened the rib. The next photo counts cartons and does not
            # send the arm back to organize on the organized score.
            self.inlay_trust.skip = True
            return self._begin_goto([self.organize_home], State.ASK, lift_first=True)
        if reason == "stop_when":
            return State.WAIT_PROCEED
        return self._hold(f"open VLA ended without opening the gripper ({reason})")

    def run(self) -> None:
        """Step until ``q``."""
        while self.state is not State.DONE:
            self.step()
