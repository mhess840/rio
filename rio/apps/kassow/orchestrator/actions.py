# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator's actions on the real cell: Jev asks, home moveJs, VLA rollouts, alarm clears."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.apps.kassow.crate import CrateReading, CrateThresholds, ask_crate
from rio.apps.kassow.experiments import JEV_SERIAL, open_spaces, resolve_experiment, resolve_model
from rio.apps.kassow.jev_ask import fresh_frame
from rio.apps.terminal import term
from rio.apps.vive.controls import held_keys
from rio.arm_resume import gripper_is_holding
from rio.cfg.node import NodeCfg
from rio.jev import JevCfg, JevClient, encode_jpeg

from .machine import OPEN, ORGANIZE, CycleLog, GripperOpened, Keys, ReturnedNearHome, VlaJob

KEYBOARD_ADDR = "127.0.0.1:5582"


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
    organize_home_radius_m: float = 0.20
    """End the organize VLA once the TCP returns inside this distance of its home."""
    open_timeout_s: float = 90.0
    """The open VLA is aborted (hold) if it has not opened the gripper by then."""
    home_joint_speed: float = 0.51
    """Fastest joint speed of a home moveJ, rad/s. 1.5 times the previous 0.34."""
    max_organize_attempts: int = 3
    """Organize rollouts in a row before a hold; ``j`` in the hold grants a fresh budget."""
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
    from rio.apps.kassow.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

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
    """``ask`` action: fresh frame, one full Jev request (view, inlay, count), photo and answer in the log."""

    def ask() -> CrateReading:
        jpeg = encode_jpeg(fresh_frame(camera, args.max_frame_age_s))
        log.image_path().write_bytes(jpeg)
        reply, reading = ask_crate(client, jpeg, available_spaces=spaces, thresholds=args.thresholds)
        log.write("jev", model=reply.model, input_tokens=reply.input_tokens, response=reply.raw)
        if reply.image_seems_missing():
            logger.warning("input_tokens {} looks like a text-only request; was the image sent?", reply.input_tokens)
        return reading

    return ask


def dry_run_vla(job: VlaJob) -> str:
    term(f"[dry run] would run the {job.experiment} VLA {job.instruction!r}")
    return "stop_when"


def make_run_vla(args: Args, env, keys: Keys, log: CycleLog) -> Callable[[VlaJob], str]:
    """``run_vla`` action: resident, policy node, one ``policy_loop`` rollout with keys and timeout."""
    from rio.arm_resume import ArmAlarm
    from rio.envs.factory import make_policy_node
    from rio.policies.rollout import RolloutControl, policy_loop

    def run_vla(job: VlaJob) -> str:
        cfg = vla_cfg(args, job)
        cfg.diag_dir = str(log.vla_dir)
        if job.experiment == OPEN:
            returned = None
            stop_when = GripperOpened(cfg.policy_cfg.gripper_close, cfg.policy_cfg.gripper_open)
            timeout_s, timeout_reason = args.open_timeout_s, "aborted"
        else:
            from rio.apps.kassow.goto import load_pose

            home = load_pose(job.home, cfg.policy_cfg.gripper_open, job.experiment)
            returned = ReturnedNearHome(home["eef_pose"][:3], args.organize_home_radius_m)
            stop_when = None
            timeout_s, timeout_reason = args.organize_timeout_s, "timeout"
        control = RolloutControl(stop_when=stop_when, plan_done=None if returned is None else returned.consider)
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
                if time.now() > deadline:
                    stop(timeout_reason)
                time.sleep(0.02)

        watcher = threading.Thread(target=watch_keys, daemon=True)
        try:
            watcher.start()
            cfg.start_resident()
            if control.stop.is_set():
                return control.reason
            server, client = make_policy_node(cfg)
            with ServerManager(cfg.mw, [server]), client() as policy:
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


def station_cfg(args: Args):
    """Arm, gripper, the two policy cameras, and the terminal keyboard; no policy node."""
    from rio.apps.kassow.policy_cfgs.pi05_lerobot import Pi05LerobotCfg

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
    if args.single_slot:
        return
    for job in (VlaJob(ORGANIZE, "", resolve_experiment(ORGANIZE).home_path), next(iter(open_jobs()), None)):
        if job is None:
            continue
        try:
            vla_cfg(args, job).start_resident()
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
