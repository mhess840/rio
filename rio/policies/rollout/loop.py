# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""``policy_loop``: run a chunked policy on the arm under stdin, DAgger, or caller control."""

import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from rio_hw import time

from rio.cleanup import cleanup_step
from rio.data.diagnostics import AsyncDiagnostics
from rio.manifest import run_manifest
from rio.schema import require_state_fields

from .actions import CommandStreamer
from .chunks import ChunkScheduler
from .control import RolloutControl, start_stdin_control
from .diagnostics import RolloutDiagnostics, dump_live_cameras
from .safety import (
    RolloutSafetyCfg,
    arm_has_stopped,
    require_arm_controller,
    require_arm_state_safe,
    stop_arm_motion,
)
from .settings import RolloutCfg, RolloutSettings

try:
    import torch
    from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig
except ImportError:
    torch = None
    ActionQueue = None
    LatencyTracker = None
    RTCConfig = None


def resolve_control_tick(t_start: float, iteration: int, dt: float, now: float) -> tuple[int, float, float]:
    """Skip missed deadlines without emitting a burst of overdue control commands."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("The control period must be finite and positive")
    cycle_end = t_start + (iteration + 1) * dt
    if cycle_end <= now:
        iteration = max(iteration, int((now - t_start) / dt) + 1)
        cycle_end = t_start + (iteration + 1) * dt
    return iteration, cycle_end, cycle_end + dt


@dataclass
class RolloutRuntime:
    """Third-party pieces of a rollout; tests pass CPU fakes instead of LeRobot and Torch."""

    action_queue: Callable[[Any], Any] | None = ActionQueue
    rtc_config: Callable[..., Any] | None = RTCConfig
    latency_tracker: Callable[[], Any] | None = LatencyTracker
    as_tensor: Callable[[Any], Any] | None = field(default_factory=lambda: None if torch is None else torch.as_tensor)
    operator_input: Callable[[threading.Event, threading.Event], None] = start_stdin_control
    dump_cameras: Callable[[dict, Path], None] = dump_live_cameras

    def require_complete(self) -> None:
        """Fail before the policy is used when LeRobot/Torch are not installed."""
        if self.action_queue is None or self.rtc_config is None or self.as_tensor is None:
            raise ImportError("LeRobot ActionQueue is required. Install with: uv sync --group pi")


class RolloutEpisode:
    """One rollout from the live pose until a pause, a stop, or an error.

    Args:
        settings: Resolved rollout settings.
        runtime: Queue and tensor factories.
        env: Started environment.
        policy: Policy client.
        writer: Background diagnostics writer.
        control: Caller-owned stop/completion, or ``None``.
        dagger: DAgger session, or ``None``.
        visualizer: Optional visualizer client.
    """

    def __init__(
        self,
        settings: RolloutSettings,
        runtime: RolloutRuntime,
        env,
        policy,
        writer: AsyncDiagnostics,
        *,
        control: RolloutControl | None = None,
        dagger=None,
        visualizer=None,
    ) -> None:
        s = settings
        self.settings = settings
        self.env = env
        self.policy = policy
        self.control = control
        self.dagger = dagger
        self.visualizer = visualizer
        self.diag = RolloutDiagnostics(
            writer,
            s.diag_root,
            {
                "started_at": time.now(),
                "policy_path": s.policy_path,
                "instruction": s.instruction,
                "control_hz": s.freq,
                "action_hz": s.action_hz,
                "n_execute": s.n_execute,
                "use_rtc": s.use_rtc,
                "rtc_queue_threshold": s.queue_threshold,
                "rtc_execution_horizon": s.execution_horizon,
                "binarize_gripper": s.binarize_gripper,
                "gripper_state_source": s.gripper_state_source,
                "action_space": "joints" if s.joint_serve else "tcp",
                "speed_limits": {"pos_m_s": s.max_pos_speed, "rot_rad_s": s.max_rot_speed, "joint_rad_s": s.max_joint_speed},
                "safety": asdict(s.safety),
                "manifest": run_manifest(),
            },
            runtime.dump_cameras,
        )
        self.chunks = ChunkScheduler(settings, runtime)
        self.stream = CommandStreamer(env, settings)
        self.t_start = time.now()
        self.iteration = 0
        env.set_start_time(self.t_start)
        env.set_instruction(s.instruction)
        self._log_plan()

    def _log_plan(self) -> None:
        s = self.settings
        logger.info(
            "Serve loop: {} Hz interpolator x{}, action {} Hz, RTC {}, {}",
            s.freq,
            s.ticks_per_action,
            s.action_hz,
            "guided" if s.use_rtc else "off",
            "joints/moveJ" if s.joint_serve else "tcp/task_pos_ik",
        )
        logger.info("Execute {} steps of each predicted chunk, then replan", s.n_execute)
        if s.use_rtc:
            logger.info(
                "RTC: request the next chunk when {} steps remain, guide the first {}", s.queue_threshold, s.execution_horizon
            )
        else:
            logger.info("RTC off: those {} steps play out before the next request", s.n_execute)
        if s.chunk_pause_s > 0:
            logger.info("Holding the last command {:.1f}s after each chunk before the next image", s.chunk_pause_s)
        if s.joint_serve:
            logger.info("Command rate limit: {:.2f} rad/s per joint ({:.1f} Hz)", s.max_joint_speed, s.freq)
        else:
            logger.info(
                "Command rate limit: {:.2f} m/s, {:.2f} rad/s ({:.1f} Hz) — first policy jump is ramped",
                s.max_pos_speed,
                s.max_rot_speed,
                s.freq,
            )
        logger.info("Diag dumps under {}", self.diag.dir)

    def run(self, pause: threading.Event, quit_ev: threading.Event) -> None:
        """Tick at the control rate until paused or quit."""
        if self.dagger is not None:
            self.dagger.begin_episode()
        while not pause.is_set() and not quit_ev.is_set():
            self.tick(pause, quit_ev)

    def tick(self, pause: threading.Event, quit_ev: threading.Event) -> None:
        """One control cycle: safety checks, operator events, planning, one command, logging."""
        s = self.settings
        self.diag.writer.check()
        if self.control is not None and self.control.stop.is_set():
            self.control.should_stop(self.stream.last_sent, has_action=False)
            quit_ev.set()
            return
        self.iteration, cycle_end, command_time = resolve_control_tick(self.t_start, self.iteration, s.dt, time.now())
        limits = s.safety if self.dagger is None else self.dagger.limits(s.safety)
        require_arm_state_safe(
            self.env.robot.arm.get_state(), self.stream.last_sent, now=time.now(), joint_serve=s.joint_serve, limits=limits
        )
        self.chunks.check_inference_timeout(time.now())
        if self.dagger is not None and not self._handle_dagger_event(self.dagger, pause, quit_ev):
            return
        if self.policy.policy_loaded:
            self._act(command_time + s.arm_latency, limits)
        if self.control is not None and self._plan_finished(self.control):
            self.control.reason = self.control.reason or "stop_when"
            quit_ev.set()
            return
        if self.control is not None and self.control.should_stop(self.stream.last_sent, self.stream.has_action):
            quit_ev.set()
            return
        time.precise_wait(cycle_end)
        self.iteration += 1

    def _handle_dagger_event(self, dagger, pause: threading.Event, quit_ev: threading.Event) -> bool:
        """Apply save/discard/quit and takeover/hand-back; False ends this tick."""
        event = dagger.poll(self.stream.last_sent)
        if event in ("save", "discard", "quit"):
            if not stop_arm_motion(self.env.robot.arm):
                raise RuntimeError("The arm could not stop before ending the DAgger episode")
            dagger.end_episode(saved=event == "save", reason="quit" if event == "quit" else None)
            (quit_ev if event == "quit" else pause).set()
            return False
        if event in ("takeover", "handback"):
            # A chunk requested before the switch was planned for the other driver; never play it.
            self.chunks.drop()
            self.stream.drop_plan()
            self.policy.reset()
        return True

    def _act(self, target_time: float, limits: RolloutSafetyCfg) -> None:
        s = self.settings
        operator_driving = self.dagger is not None and self.dagger.operator_driving
        finishing = self.control is not None and self.control.finish_plan
        if not operator_driving and not finishing and self.chunks.want_observation():
            self.chunks.request(self.env, self.policy, self.diag)
        self.chunks.poll(
            self.env,
            self.policy,
            last_command=self.stream.action,
            has_command=self.stream.has_action,
            control=self.control,
            diag=self.diag,
        )
        if operator_driving:
            measured_pose = np.asarray(self.env.robot.arm.get_state()["eef_pose"], dtype=float)
            self.stream.send_operator(self.dagger.human_command(measured_pose, s.dt), target_time, limits)
        else:
            self.stream.pull(self.chunks.queue)
            self.stream.step(target_time, limits)
        if self.dagger is not None and self.dagger.waiting_for_policy and self.stream.following:
            self.dagger.resume_policy()
        if self.stream.has_action:
            arm_now = self.env.robot.arm.get_state()
            measured = np.asarray(arm_now["joint_q" if s.joint_serve else "eef_pose"], dtype=float)
            self.diag.command(
                time.now() - self.t_start,
                self.stream.action,
                measured,
                self.chunks.queue.qsize(),
                max(self.chunks.chunk_idx - 1, 0),
                arm_now,
                s.joint_serve,
            )
        step = self.env.get_state(action=self.stream.action)
        if self.dagger is not None:
            self.dagger.record(step)
            self.dagger.status(time.now() - self.t_start)
        if self.visualizer:
            self.visualizer.log_env_state("env", step)

    def _plan_finished(self, control: RolloutControl) -> bool:
        """The caller's completion latched, the plan drained, and the arm has come to rest."""
        return bool(
            control.finish_plan
            and not self.chunks.processing
            and self.stream.idle
            and self.chunks.queue.qsize() == 0
            and arm_has_stopped(self.env.robot.arm.get_state())
        )

    def close(self, quit_ev: threading.Event) -> None:
        """Stop the arm stream and drop every plan; never hides the exception that ended the episode."""
        self.chunks.queue.clear()
        self.stream.interpolator.reset()
        if not stop_arm_motion(self.env.robot.arm):
            quit_ev.set()
            logger.error("The arm cannot acknowledge a stream stop; ending the hardware session")
        with cleanup_step("reset the policy"):
            if hasattr(self.policy, "reset"):
                self.policy.reset()
        if self.dagger is not None and self.dagger.recording:
            error = sys.exc_info()[1]
            with cleanup_step("close the DAgger episode"):
                self.dagger.end_episode(saved=False, reason=repr(error) if error is not None else "stopped")


def _wait_for_policy(env, policy, settings: RolloutSettings, control: RolloutControl | None) -> bool:
    """Block until the weights are loaded; False when the caller stopped first."""
    logger.info("Waiting for the policy to finish loading (9 GB weights onto GPU)...")
    deadline = time.now() + settings.load_timeout
    while not policy.policy_loaded:
        if control is not None and control.stop.is_set():
            control.reason = control.reason or "stopped"
            stop_arm_motion(env.robot.arm)
            return False
        if time.now() >= deadline:
            stop_arm_motion(env.robot.arm)
            raise TimeoutError("Policy loading exceeded the startup timeout")
        time.sleep(0.05)
    return True


def _drop_stale_chunks(policy) -> None:
    drained = 0
    while drained < 8:
        if not policy.get_action_chunk().get("ready"):
            break
        drained += 1
    if drained:
        logger.info("Dropped {} stale chunk(s) from before pause", drained)


def policy_loop(
    args: RolloutCfg,
    env,
    policy,
    visualizer=None,
    dagger=None,
    control: RolloutControl | None = None,
    runtime: RolloutRuntime | None = None,
):
    """Execute a policy with stdin, human takeover, or caller-owned control.

    Args:
        args: Station/policy config (see ``RolloutCfg``).
        env: Started environment.
        policy: Policy client (``PolicyInterface``).
        visualizer: Optional visualizer client.
        dagger: DAgger session for operator takeover; excludes ``control``.
        control: Caller-owned stop and completion conditions; excludes ``dagger``.
        runtime: Queue and tensor factories; defaults to LeRobot and Torch.
    """
    if control is not None and dagger is not None:
        raise ValueError("control and dagger cannot be combined")
    runtime = runtime or RolloutRuntime()
    runtime.require_complete()
    settings = RolloutSettings.from_args(args)
    require_arm_controller(env.robot.arm)
    arm_state = env.robot.arm.get_state()
    plan_done = control is not None and control.plan_done is not None
    require_state_fields(arm_state, settings.required_arm_fields(plan_done=plan_done))
    if "alarm_code" not in arm_state:
        logger.warning("The arm driver publishes no alarm_code; controller alarms cannot stop this rollout early")
    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
    if not _wait_for_policy(env, policy, settings, control):
        return
    pause = threading.Event()
    quit_ev = threading.Event()
    pause.set()
    if control is not None:
        pause.clear()
        logger.info("Policy ready. Running one caller-controlled rollout.")
    elif dagger is None:
        logger.info("Policy ready. Enter starts/resumes; p pauses; q quits.")
        runtime.operator_input(pause, quit_ev)
    else:
        logger.info("Policy ready. DAgger: n starts, c takes over, r hands back, s saves, x discards, q quits.")
    writer = AsyncDiagnostics()
    started_once = False
    try:
        while not quit_ev.is_set():
            if pause.is_set() and dagger is not None:
                if not dagger.wait_for_episode():
                    break
                pause.clear()
            elif pause.is_set():
                if started_once:
                    logger.info("Paused. Arm stopped; weights stay on GPU. Enter resumes, q quits.")
                while pause.is_set() and not quit_ev.is_set():
                    time.sleep(0.05)
                if quit_ev.is_set():
                    break
            time.sleep(0.5)
            if hasattr(policy, "reset"):
                policy.reset()
            _drop_stale_chunks(policy)
            started_once = True
            episode = RolloutEpisode(
                settings, runtime, env, policy, writer, control=control, dagger=dagger, visualizer=visualizer
            )
            try:
                episode.run(pause, quit_ev)
            except KeyboardInterrupt:
                quit_ev.set()
                logger.info("Quit. Unloading policy.")
                break
            finally:
                episode.close(quit_ev)
    finally:
        quit_ev.set()
        stop_arm_motion(env.robot.arm)
        try:
            with cleanup_step("save the recording"):
                if env.recorder:
                    env.recorder.save(wait=True)
        finally:
            with cleanup_step("flush the rollout diagnostics"):
                writer.close()
