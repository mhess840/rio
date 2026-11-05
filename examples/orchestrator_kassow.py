"""Jev decides, the VLAs act: organize the crate inlay, or load the next free space.

On ``j`` the Jev camera photographs the crate and Jev answers three questions
(``rio.jev.crate``). Then:

- inlay crumpled: moveJ to the organize home, run the organize VLA until ``s``
  (or the timeout), then ask Jev again.
- inlay organized, N cartons: moveJ to the home of space N+1, run the open VLA
  with "put the object in space N+1", and stop it at its first open-gripper
  command. ``p`` then moves to the organize home and waits for the next ``j``.
- anything unsure (inlay state, count, missing space home, VLA error): hold,
  print why, and wait for ``j`` or ``q``.

Keys: ``j`` ask Jev, ``p`` proceed, ``s`` end the running VLA normally, ``x``
abort it (hold), ``q`` quit. Every cycle is logged to
``experiments/orchestrator/runs/<time>/``.

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
    HOLD = "hold"
    DONE = "done"


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
            ``aborted``, or ``error: ...``.
        keys: Key source.
        organize_home: Home of the organize VLA, also the rest pose after ``p``.
        spaces: Home per 1-based space of the open VLA.
        log: Event log.
    """

    def __init__(
        self,
        *,
        ask: Callable[[], CrateReading],
        goto: Callable[[Path], bool],
        run_vla: Callable[[VlaJob], str],
        keys: Keys,
        organize_home: Path,
        spaces: dict[int, Path],
        log: CycleLog,
    ):
        self.ask = ask
        self.goto = goto
        self.run_vla = run_vla
        self.keys = keys
        self.organize_home = organize_home
        self.spaces = spaces
        self.log = log
        self.state = State.IDLE
        self.job: VlaJob | None = None
        self.hold_reason = ""

    def _hold(self, reason: str) -> State:
        self.hold_reason = reason
        self.log.write("hold", reason=reason)
        return State.HOLD

    def step(self) -> State:
        """Run one state and return the next."""
        if self.state is State.IDLE:
            term("\nIdle. j: ask Jev | q: quit")
            self.state = State.ASK if wait_key(self.keys, "jq") == "j" else State.DONE
        elif self.state is State.ASK:
            self.state = self._ask()
        elif self.state is State.GOTO:
            assert self.job is not None
            ok = self.goto(self.job.home)
            self.log.write("goto", home=self.job.home.name, ok=ok)
            self.state = State.RUN_VLA if ok else self._hold(f"moveJ to {self.job.home.name} failed")
        elif self.state is State.RUN_VLA:
            self.state = self._run()
        elif self.state is State.WAIT_PROCEED:
            term("\nObject released. p: move to the organize home | q: quit")
            if wait_key(self.keys, "pq") == "q":
                self.state = State.DONE
            else:
                ok = self.goto(self.organize_home)
                self.log.write("goto", home=self.organize_home.name, ok=ok)
                self.state = State.IDLE if ok else self._hold("moveJ to the organize home failed")
        elif self.state is State.HOLD:
            term(f"\nHOLD: {self.hold_reason}\nj: ask Jev again | q: quit")
            self.state = State.ASK if wait_key(self.keys, "jq") == "j" else State.DONE
        return self.state

    def _ask(self) -> State:
        self.log.cycle += 1
        try:
            reading = self.ask()
        except (JevError, ValueError, RuntimeError) as error:
            return self._hold(f"Jev request failed: {error}")
        term(f"Jev: {reading.summary()}")
        self.log.write("reading", reading=asdict(reading))
        if reading.decision == "organize":
            self.job = VlaJob(ORGANIZE, "", self.organize_home)
            return State.GOTO
        if reading.decision == "open" and reading.space in self.spaces:
            self.job = VlaJob(OPEN, f"put the object in space {reading.space}", self.spaces[reading.space])
            return State.GOTO
        return self._hold(reading.reason)

    def _run(self) -> State:
        job = self.job
        assert job is not None
        term(f"\nRunning {job.experiment} VLA {job.instruction!r}. s: end | x: abort")
        reason = self.run_vla(job)
        self.log.write("vla", experiment=job.experiment, instruction=job.instruction, reason=reason)
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
    home_joint_speed: float = 0.15
    """Fastest joint speed of a home moveJ, rad/s."""
    max_frame_age_s: float = 0.5
    runs_dir: Path | None = None
    """Log folder. Unset: experiments/orchestrator/runs."""
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


def make_ask(args: Args, client: JevClient, camera, spaces: set[int], log: CycleLog) -> Callable[[], CrateReading]:
    """``ask`` action: fresh frame, one Jev request, photo and answer in the log."""

    def ask() -> CrateReading:
        jpeg = encode_jpeg(fresh_frame(camera, args.max_frame_age_s))
        log.image_path().write_bytes(jpeg)
        reply, reading = ask_crate(client, jpeg, available_spaces=spaces, thresholds=args.thresholds)
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
        except Exception as error:
            logger.exception("{} VLA failed", job.experiment)
            return f"error: {error}"
        finally:
            done.set()
        return control.reason or "stopped"

    return run_vla


def make_goto(args: Args, env) -> Callable[[Path], bool]:
    """``goto`` action: moveJ to a home file of ``organize`` or ``open``."""
    from examples.goto_kassow_q import goto_home, load_pose

    def goto(path: Path) -> bool:
        experiment = resolve_experiment(ORGANIZE if path == resolve_experiment(ORGANIZE).home_path else OPEN)
        time.sleep(HOME_POST_MOVE_S)
        home = load_pose(path, experiment.gripper_open, experiment.name)
        return goto_home(env, home, home_joint_speed=args.home_joint_speed)

    return goto


def station_cfg(args: Args):
    """Arm, gripper, the two policy cameras, and the terminal keyboard; no policy node."""
    from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

    _, path = resolve_model(resolve_experiment(ORGANIZE), args.organize_model)
    station = Pi05LerobotCfg(experiment=ORGANIZE, policy_path=str(path), allow_unlisted_instruction=True)
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
        ask = make_ask(args, client, camera, set(spaces), log)
        if args.dry_run:
            server, keyboard_client = make_node(args.mw, "interfaces", "SshKeyboard", {"addr": KEYBOARD_ADDR, "freq": 100})
            stack.enter_context(ServerManager(args.mw, [server]))
            keys = NodeKeys(stack.enter_context(keyboard_client()))
            goto, run_vla = dry_goto, dry_run_vla
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
        Orchestrator(ask=ask, goto=goto, run_vla=run_vla, keys=keys, organize_home=organize_home, spaces=spaces, log=log).run()
    log.write("quit")


if __name__ == "__main__":
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
