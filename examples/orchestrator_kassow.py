"""Jev decides, the VLAs act: organize the crate inlay, or load the next free space.

On ``j`` the Jev camera photographs the crate and Jev answers three questions
(``rio.jev.crate``). Then:

- inlay crumpled: moveJ to the organize home, run the organize VLA until ``s``
  (or the timeout), then ask Jev again.
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
latest OpenJEV reading while the run is going.

Decision logic with the real Jev camera, nothing moves::

    export TYPESAFE_API_KEY=...
    uv run python -m examples.orchestrator_kassow --dry-run

Full run, both VLAs resident on one GPU (``--single-slot`` reloads on each switch)::

    uv run --group pi --no-group docs -m examples.orchestrator_kassow --device cuda:1
"""

from __future__ import annotations

import enum
import json
import multiprocessing as mp
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from examples.cfg.kassow_experiments import (
    JEV_SERIAL,
    experiments_root,
    open_spaces,
    resolve_experiment,
    resolve_model,
)
from examples.jev_ask import fresh_frame, jev_camera_session
from examples.teleop_vive_hand import HOME_POST_MOVE_S, held_keys, make_stdout_blocking, term
from rio.arm_resume import Phase, PhaseKind, decide_resume, gripper_is_holding
from rio.cfg.node import NodeCfg
from rio.envs.factory import make_node
from rio.jev import CrateReading, CrateThresholds, JevCfg, JevClient, JevError, ask_crate, encode_jpeg

KEYBOARD_ADDR = "127.0.0.1:5582"
ORGANIZE = "organize"
OPEN = "open"


class Keys(Protocol):
    def pressed(self) -> set[str]: ...


class NodeKeys:
    """Keys newly pressed on an ``SshKeyboard`` node since the last call."""

    def __init__(self, keyboard):
        self.keyboard = keyboard
        self._previous: set[str] = set()
        self._lock = threading.Lock()

    def pressed(self) -> set[str]:
        with self._lock:
            keys = held_keys(self.keyboard.get_state())
            new = keys - self._previous
            self._previous = keys
            return new


def wait_key(keys: Keys, allowed: str, poll_s: float = 0.02) -> str:
    """Block until one of ``allowed`` is pressed and return it."""
    while True:
        hit = sorted(k for k in keys.pressed() if k in allowed)
        if hit:
            return hit[0]
        time.sleep(poll_s)


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


def _recover_unavailable() -> bool:
    """No arm is connected, so a clear cannot succeed."""
    return False


def _lift_skipped() -> bool:
    """Dry run: the arm does not rise."""
    return True


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
    ):
        self.ask = ask
        self.goto = goto
        self.run_vla = run_vla
        self.recover = recover if recover is not None else _recover_unavailable
        self.keys = keys
        self.organize_home = organize_home
        self.wait_home: Path | None = None
        self.at_wait_pose = False
        self.approach_space = False
        self.spaces = spaces
        self.log = log
        self.state = State.IDLE
        self.job: VlaJob | None = None
        self.hold_reason = ""
        self.phases = JOB_PHASES
        self.phase_index = 0
        self.alarm_holding = False
        self.goto_path: Path | None = None
        self.after_goto = State.RUN_VLA
        self.lift_before_goto = False
        self.lift: Callable[[], bool | str] = lambda: True
        # False until the first accepted inlay. Later asks only count cartons.
        self.count_only = False
        self.session: dict | None = None
        self.reading: CrateReading | None = None
        self.board = None

    def _hold(self, reason: str) -> State:
        self.hold_reason = reason
        self.log.write("hold", reason=reason)
        return State.HOLD

    def _show(self) -> None:
        if self.board is not None:
            self.board.publish(self)

    def step(self) -> State:
        """Run one state and return the next."""
        self._show()
        if self.state is State.IDLE:
            term("\nIdle. j: ask Jev | q: quit")
            self.state = State.ASK if wait_key(self.keys, "jq") == "j" else State.DONE
        elif self.state is State.ASK:
            self.state = self._ask()
        elif self.state is State.GOTO:
            self.state = self._goto()
        elif self.state is State.RUN_VLA:
            self.state = self._run()
        elif self.state is State.WAIT_PROCEED:
            term("\nHolding the compartment open. Place the object. p: move clear and look again | q: quit")
            if wait_key(self.keys, "pq") == "q":
                self.state = State.DONE
            else:
                self.goto_path = self.wait_home or self.organize_home
                self.after_goto = State.ASK
                self.lift_before_goto = True
                self.state = State.GOTO
        elif self.state is State.WAIT_RESET:
            self.state = self._reset()
        elif self.state is State.HOLD:
            term(f"\nHOLD: {self.hold_reason}\nj: ask Jev again | q: quit")
            self.state = State.ASK if wait_key(self.keys, "jq") == "j" else State.DONE
        self._show()
        return self.state

    def _ask(self) -> State:
        self.log.cycle += 1
        try:
            reading = self.ask()
        except (JevError, ValueError, RuntimeError) as error:
            return self._hold(f"Jev request failed: {error}")
        self.reading = reading
        term(f"Jev: {reading.summary()}")
        self.log.write("reading", reading=asdict(reading))
        self._show()
        if reading.decision in ("open", "organize"):
            self.count_only = True
            if self.session is not None:
                self.session["count_only"] = True
        if reading.decision == "organize":
            self.job = VlaJob(ORGANIZE, "", self.organize_home)
            return self._begin_goto(self.job.home, State.RUN_VLA)
        if reading.decision == "open" and reading.space in self.spaces:
            self.job = VlaJob(OPEN, f"put the object in space {reading.space}", self.spaces[reading.space])
            if self.wait_home is not None and not self.at_wait_pose:
                self.approach_space = True
                return self._begin_goto(self.wait_home, State.RUN_VLA)
            return self._begin_goto(self.job.home, State.RUN_VLA)
        return self._hold(reading.reason)

    def _begin_goto(self, path: Path, after: State) -> State:
        """Queue a moveJ. The motion itself happens in ``GOTO``."""
        self.goto_path = path
        self.after_goto = after
        return State.GOTO

    def _goto(self) -> State:
        """Run the queued moveJ. An alarm waits for ``c`` instead of holding."""
        path = self.goto_path
        assert path is not None
        if self.lift_before_goto:
            lifted = self.lift()
            self.log.write("lift", ok=lifted is True)
            if lifted in ("alarm", "alarm_holding"):
                return self._alarm(0, lifted == "alarm_holding")
            if lifted is not True:
                self.lift_before_goto = False
                return self._hold("could not lift clear of the crate")
            self.lift_before_goto = False
        outcome = self.goto(path)
        self.log.write("goto", home=path.name, ok=outcome is True)
        if outcome in ("alarm", "alarm_holding"):
            return self._alarm(0, outcome == "alarm_holding")
        if outcome:
            if path == self.wait_home:
                self.at_wait_pose = True
            if self.approach_space and self.job is not None:
                self.approach_space = False
                self.goto_path = self.job.home
                self.after_goto = State.RUN_VLA
                return State.GOTO
            return self.after_goto
        return self._hold(f"moveJ to {path.name} failed")

    def _alarm(self, index: int, holding: bool) -> State:
        """Remember the phase and wait. The arm does not move in this state."""
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
        term(f"\nRunning {job.experiment} VLA {job.instruction!r}. s: end | x: abort")
        reason = self.run_vla(job)
        self.log.write("vla", experiment=job.experiment, instruction=job.instruction, reason=reason)
        if reason in ("alarm", "alarm_holding"):
            return self._alarm(1, reason == "alarm_holding")
        if reason == "aborted" or reason.startswith("error"):
            return self._hold(f"{job.experiment} VLA {reason}")
        if job.experiment == ORGANIZE:
            return State.ASK
        if reason == "stop_when":
            return State.WAIT_PROCEED
        return self._hold(f"open VLA ended without opening the gripper ({reason})")

    def run(self) -> None:
        """Step until ``q``."""
        while self.state is not State.DONE:
            self.step()


@dataclass
class Args:
    """Jev-driven orchestrator for the Kassow crate."""

    jev: JevCfg = field(default_factory=JevCfg)
    thresholds: CrateThresholds = field(default_factory=CrateThresholds)
    jev_serial: str = JEV_SERIAL
    """Serial of the Jev RealSense."""
    dry_run: bool = False
    """Ask Jev for real but only print the moveJ and VLA steps. Arm and GPU stay idle."""
    device: str = "cuda:1"
    """GPU for both VLA residents."""
    single_slot: bool = False
    """Keep one resident and reload the checkpoint on every organize/open switch."""
    organize_model: str | None = None
    """Registered model of ``organize``. Unset: the selected one."""
    open_model: str | None = None
    """Registered model of ``open``. Unset: the selected one."""
    organize_timeout_s: float = 120.0
    """The organize VLA ends normally after this long, like ``s``."""
    open_timeout_s: float = 90.0
    """The open VLA is aborted (hold) if it has not opened the gripper by then."""
    home_joint_speed: float = 0.51
    """Fastest joint speed of a home moveJ, rad/s. 1.5 times the previous 0.34."""
    max_frame_age_s: float = 0.5
    runs_dir: Path | None = None
    """Log folder. Unset: experiments/orchestrator/runs."""
    board: bool = False
    """Serve the live demo page at http://127.0.0.1:8765."""
    board_port: int = 8765
    mw: str = "Thread"
    mp_method: str = "spawn"


def vla_cfg(args: Args, job: VlaJob):
    """Policy station config for one rollout: model, sentence, home, gripper encoding, slot."""
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    experiment = resolve_experiment(job.experiment)
    model = args.organize_model if job.experiment == ORGANIZE else args.open_model
    _, path = resolve_model(experiment, model)
    slot = "" if args.single_slot else job.experiment
    cfg = Pi05LerobotCfg(
        experiment=experiment.name,
        policy_path=str(path),
        instruction=job.instruction,
        policy_cfg=Pi05LerobotCfg.PolicyConfig(device=args.device, resident_slot=slot),
    )
    cfg.mw = args.mw
    return cfg


def make_ask(
    args: Args, client: JevClient, camera, spaces: set[int], log: CycleLog, session: dict
) -> Callable[[], CrateReading]:
    """``ask`` action: fresh frame, one Jev request, photo and answer in the log.

    ``session["count_only"]`` is set by the orchestrator after the inlay has been
    accepted. Later calls ask only for the carton count.
    """

    def ask() -> CrateReading:
        jpeg = encode_jpeg(fresh_frame(camera, args.max_frame_age_s))
        log.image_path().write_bytes(jpeg)
        reply, reading = ask_crate(
            client,
            jpeg,
            available_spaces=spaces,
            thresholds=args.thresholds,
            count_only=bool(session["count_only"]),
        )
        log.write("jev", model=reply.model, input_tokens=reply.input_tokens, response=reply.raw)
        if reply.image_seems_missing():
            logger.warning("input_tokens {} looks like a text-only request; was the image sent?", reply.input_tokens)
        return reading

    return ask


def dry_goto(path: Path) -> bool:
    term(f"[dry run] would moveJ to {path.name}")
    return True


def dry_run_vla(job: VlaJob) -> str:
    term(f"[dry run] would run the {job.experiment} VLA {job.instruction!r}")
    return "stop_when" if job.experiment == OPEN else "timeout"


def make_run_vla(args: Args, env, keys: Keys, log: CycleLog) -> Callable[[VlaJob], str]:
    """``run_vla`` action: resident, policy node, one ``policy_loop`` rollout with keys and timeout."""
    from examples.policy_inference import RolloutControl, policy_loop, start_resident
    from rio.arm_resume import ArmAlarm
    from rio.envs.factory import make_policy_node

    def run_vla(job: VlaJob) -> str:
        cfg = vla_cfg(args, job)
        cfg.diag_dir = str(log.vla_dir)
        if job.experiment == OPEN:
            stop_when = GripperOpened(cfg.policy_cfg.gripper_close, cfg.policy_cfg.gripper_open)
            timeout_s, timeout_reason = args.open_timeout_s, "aborted"
        else:
            stop_when = None
            timeout_s, timeout_reason = args.organize_timeout_s, "timeout"
        control = RolloutControl(stop_when=stop_when)
        done = threading.Event()

        def stop(reason: str) -> None:
            if not control.stop.is_set():
                control.reason = reason
                control.stop.set()

        def watch_keys() -> None:
            deadline = time.now() + timeout_s
            while not done.is_set():
                pressed = keys.pressed()
                if "x" in pressed:
                    stop("aborted")
                elif "s" in pressed:
                    stop("operator")
                elif time.now() > deadline:
                    stop(timeout_reason)
                time.sleep(0.02)

        try:
            start_resident(cfg)
            server, client = make_policy_node(cfg)
            watcher = threading.Thread(target=watch_keys, daemon=True)
            with ServerManager(cfg.mw, [server]), client() as policy:
                watcher.start()
                policy_loop(cfg, env, policy, control=control)
        except ArmAlarm:
            holding = gripper_is_holding(env, close=float(cfg.policy_cfg.gripper_close))
            return "alarm_holding" if holding else "alarm"
        except Exception as error:
            logger.exception("{} VLA failed", job.experiment)
            return f"error: {error}"
        finally:
            done.set()
        return control.reason or "stopped"

    return run_vla


def make_goto(args: Args, env) -> Callable[[Path], bool | str]:
    """``goto`` action: moveJ to a home file of ``organize`` or ``open``."""
    from examples.goto_kassow_q import MoveStatus, goto_home, load_pose

    def goto(path: Path) -> bool | str:
        experiment = resolve_experiment(ORGANIZE if path == resolve_experiment(ORGANIZE).home_path else OPEN)
        time.sleep(HOME_POST_MOVE_S)
        home = load_pose(path, experiment.gripper_open, experiment.name)
        status = goto_home(env, home, home_joint_speed=args.home_joint_speed)
        if status is MoveStatus.ALARM:
            holding = gripper_is_holding(env, close=float(experiment.gripper_close))
            return "alarm_holding" if holding else "alarm"
        return status is MoveStatus.LANDED

    return goto


def make_lift(args: Args, env) -> Callable[[], bool | str]:
    """Raise the TCP 10 cm so the leave moveJ clears the crate walls."""
    from examples.goto_kassow_q import MoveStatus, lift_z

    def lift() -> bool | str:
        status = lift_z(env, 0.10, home_joint_speed=args.home_joint_speed)
        if status is MoveStatus.ALARM:
            holding = gripper_is_holding(env, close=float(resolve_experiment(OPEN).gripper_close))
            return "alarm_holding" if holding else "alarm"
        return status is MoveStatus.LANDED

    return lift


def station_cfg(args: Args):
    """Arm, gripper, the two policy cameras, and the terminal keyboard; no policy node."""
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    _, path = resolve_model(resolve_experiment(OPEN), args.open_model)
    station = Pi05LerobotCfg(
        experiment=OPEN,
        policy_path=str(path),
        instruction="put the object in space 1",
        allow_unlisted_instruction=True,
    )
    station.policy = ""
    station.teleop_keyboard = "SshKeyboard"
    station.teleop_keyboard_cfg = NodeCfg(addr=KEYBOARD_ADDR, freq=100)
    station.mw = args.mw
    return station


def preload(args: Args) -> None:
    """Load both residents up front when they share the GPU in two slots."""
    from examples.policy_inference import start_resident

    if args.single_slot:
        return
    for job in (VlaJob(ORGANIZE, "", resolve_experiment(ORGANIZE).home_path), next(iter(open_jobs()), None)):
        if job is None:
            continue
        try:
            start_resident(vla_cfg(args, job))
        except (RuntimeError, TimeoutError) as error:
            raise SystemExit(
                f"Could not load the {job.experiment} VLA next to the other one on {args.device} ({error}). "
                "Free the GPU or pass --single-slot."
            ) from error


def open_jobs() -> list[VlaJob]:
    return [VlaJob(OPEN, home.instruction, home.path) for _, home in sorted(open_spaces().items())]


def make_recover(env) -> Callable[[], bool]:
    """Clear Kassow alarms without leaving the session. False means stay stopped."""

    def recover() -> bool:
        arm = env.robot.arm
        clear = getattr(arm, "recover_alarms", None)
        if not callable(clear):
            logger.error("The arm cannot clear alarms in this session")
            return False
        try:
            return bool(clear())
        except Exception:
            logger.exception("In-session alarm clear failed")
            return False

    return recover


def main(args: Args) -> None:
    make_stdout_blocking()
    client = JevClient(args.jev)
    spaces = {number: home.path for number, home in open_spaces().items()}
    organize_home = resolve_experiment(ORGANIZE).home_path
    log = CycleLog(args.runs_dir or experiments_root() / "orchestrator" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}")
    logger.info("Spaces with a home: {}. Log: {}", sorted(spaces), log.root)
    log.write("start", args=asdict(args), spaces={k: v.name for k, v in spaces.items()})

    with ExitStack() as stack:
        camera = stack.enter_context(jev_camera_session(args.jev_serial, args.mw))
        session = {"count_only": False}
        ask = make_ask(args, client, camera, set(spaces), log, session)
        if args.dry_run:
            server, keyboard_client = make_node(args.mw, "interfaces", "SshKeyboard", {"addr": KEYBOARD_ADDR, "freq": 100})
            stack.enter_context(ServerManager(args.mw, [server]))
            keys = NodeKeys(stack.enter_context(keyboard_client()))
            goto, run_vla = dry_goto, dry_run_vla
            lift = _lift_skipped
            recover = _recover_unavailable
        else:
            from rio.envs.env import make_env

            try:
                preload(args)
                station = station_cfg(args)
            except (FileNotFoundError, ValueError) as error:
                raise SystemExit(f"VLA setup: {error}") from error
            servers, clients, env = make_env(station)
            stack.enter_context(ServerManager(args.mw, list(servers.values())))
            stack.enter_context(env)
            keys = NodeKeys(stack.enter_context(clients["teleop_keyboard"]()))
            goto, run_vla = make_goto(args, env), make_run_vla(args, env, keys, log)
            lift = make_lift(args, env)
            recover = make_recover(env)
        orchestrator = Orchestrator(
            ask=ask,
            goto=goto,
            run_vla=run_vla,
            keys=keys,
            recover=recover,
            organize_home=organize_home,
            spaces=spaces,
            log=log,
        )
        orchestrator.lift = lift
        orchestrator.session = session
        if args.board:
            from examples.orchestrator_board import DemoBoard

            board = DemoBoard(log.root.parent.parent / "live", args.board_port)
            board.publish(orchestrator)
            board.serve()
            orchestrator.board = board
        from examples.goto_kassow_q import write_wait_pose

        space1 = spaces.get(1)
        if space1 is not None:
            orchestrator.wait_home = write_wait_pose(organize_home.with_name("home_open_wait.json"), space1)
        term("\nMoving to the organize home before the first ask.")
        orchestrator.state = State.GOTO
        orchestrator._show()
        if goto(organize_home) is not True:
            raise SystemExit("Could not reach the organize home before the first ask")
        orchestrator.state = State.IDLE
        orchestrator._show()
        orchestrator.run()
    log.write("quit")


if __name__ == "__main__":
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
