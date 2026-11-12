# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""When to ask the policy for a new plan and how a returned chunk joins the action queue."""

import math

import numpy as np
from loguru import logger
from rio_hw import time

from rio.policies.rtc import RTC_MAX_GUIDANCE_WEIGHT

from .diagnostics import RolloutDiagnostics
from .observation import create_obs
from .safety import validate_action_chunk
from .settings import RTC_QUEUE_THRESHOLD, RolloutSettings


def slice_action_chunk(processed: np.ndarray, original: np.ndarray, n_execute: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first ``n_execute`` waypoints of a predicted chunk."""
    n = min(len(processed), len(original))
    processed = processed[:n]
    original = original[:n]
    if n_execute > 0:
        n = min(n, int(n_execute))
        processed = processed[:n]
        original = original[:n]
    return processed, original


def merge_delay_steps(
    latency_s: float,
    time_per_action: float,
    *,
    leftover_present: bool,
    consumed_steps: int | None = None,
) -> int:
    """RTC prefix skip: only drop steps the robot actually played during inference.

    LeRobot still applies ``ceil(latency/dt)`` on an empty queue. The first chunk
    then starts ~1 s into the plan (20 Hz * 1 s is about 20 steps) and the first
    waypoint is already several centimetres away.
    """
    if not leftover_present:
        return 0
    if consumed_steps is not None:
        return max(0, int(consumed_steps))
    if latency_s <= 0 or time_per_action <= 0:
        return 0
    return math.ceil(latency_s / time_per_action)


def chunk_hold_pending(now: float, hold_until: float | None) -> bool:
    """True while a between-chunk hold has not elapsed."""
    return hold_until is not None and now < hold_until


def should_send_observation(
    *,
    processing_obs: bool,
    remaining: int,
    use_rtc: bool = False,
    queue_threshold: int = RTC_QUEUE_THRESHOLD,
) -> bool:
    """Match LeRobot ``RTCInferenceEngine``: infer when ``qsize() <= queue_threshold``.

    Without RTC, wait until the queue is empty so a new absolute-TCP chunk is not
    played from index 0 over a live plan.
    """
    if processing_obs:
        return False
    if remaining <= 0:
        return True
    if not use_rtc:
        return False
    return remaining <= int(queue_threshold)


class ChunkScheduler:
    """Owns the LeRobot action queue: requests observations and merges the returned chunks.

    Args:
        settings: Resolved rollout settings.
        runtime: Queue, RTC config, latency tracker, and tensor factories.
    """

    def __init__(self, settings: RolloutSettings, runtime) -> None:
        self.settings = settings
        self.as_tensor = runtime.as_tensor
        rtc_cfg = runtime.rtc_config(
            enabled=settings.use_rtc,
            execution_horizon=settings.execution_horizon,
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
        )
        self.queue = runtime.action_queue(rtc_cfg)
        self.latency = runtime.latency_tracker() if runtime.latency_tracker is not None else None
        self.processing = False
        self.t_request: float | None = None
        self.index_before_inference = 0
        self.leftover_at_request = False
        self.hold_until: float | None = None
        self.chunk_idx = 0

    def drop(self) -> None:
        """Forget queued and in-flight plans; the caller resets the policy."""
        self.queue.clear()
        self.hold_until = None
        self.processing = False
        self.t_request = None
        self.leftover_at_request = False

    def check_inference_timeout(self, now: float) -> None:
        """Abort when the requested chunk is overdue."""
        if self.processing and self.t_request is not None and now - self.t_request > self.settings.safety.inference_timeout_s:
            raise RuntimeError("Policy inference exceeded the rollout timeout")

    def want_observation(self) -> bool:
        """Time for a new plan, honouring the optional still-hold between chunks."""
        s = self.settings
        if not should_send_observation(
            processing_obs=self.processing,
            remaining=self.queue.qsize(),
            use_rtc=s.use_rtc,
            queue_threshold=s.queue_threshold,
        ):
            return False
        if s.chunk_pause_s <= 0 or self.chunk_idx == 0:
            return True
        now = time.now()
        if self.hold_until is None:
            self.hold_until = now + s.chunk_pause_s
            logger.info("Chunk {} finished. Holding still {:.1f}s before the next plan", self.chunk_idx, s.chunk_pause_s)
        if chunk_hold_pending(now, self.hold_until):
            return False
        logger.info("Hold done after chunk {}. Taking a new image", self.chunk_idx)
        self.hold_until = None
        return True

    def request(self, env, policy, diag: RolloutDiagnostics) -> None:
        """Send a fresh observation, with the RTC leftover when a plan is still running."""
        s = self.settings
        remaining = self.queue.qsize()
        leftover = self.queue.get_left_over() if s.use_rtc else None
        self.leftover_at_request = leftover is not None and leftover.numel() > 0
        obs = create_obs(env, gripper_state_source=s.gripper_state_source, max_camera_age_s=s.safety.max_state_age_s)
        diag.cameras(obs)
        max_latency = self.latency.max() if self.latency is not None else 0.0
        infer_delay = math.ceil(max_latency / s.time_per_action) if max_latency else 0
        if leftover is not None and self.leftover_at_request:
            obs["rtc_leftover"] = leftover.detach().cpu().numpy().astype(np.float32)
            obs["rtc_inference_delay"] = np.int32(infer_delay)
        logger.warning(
            "REQUESTING OBS remaining={} leftover={} delay={}",
            remaining,
            0 if leftover is None else int(leftover.shape[0]),
            infer_delay if self.leftover_at_request else 0,
        )
        self.index_before_inference = self.queue.get_action_index()
        self.t_request = time.now()
        policy.send_observation(obs)
        self.processing = True

    def poll(self, env, policy, *, last_command: np.ndarray, has_command: bool, control, diag: RolloutDiagnostics) -> None:
        """Merge a ready chunk into the queue and log how far its first waypoint is from the arm."""
        if not self.processing:
            return
        s = self.settings
        response = policy.get_action_chunk()
        if response.get("inference_failed"):
            raise RuntimeError("Policy inference failed; see the policy worker log")
        if not response["ready"]:
            return
        processed = validate_action_chunk(response["actions"], s.action_dim)
        original = validate_action_chunk(response.get("original_actions", processed), s.action_dim)
        if original.shape != processed.shape:
            raise RuntimeError("Normalized and physical action chunk shapes do not match")
        predicted_n = min(len(processed), len(original))
        processed, original = slice_action_chunk(processed, original, s.n_execute)
        if predicted_n != len(processed):
            logger.info("Playing {}/{} predicted steps", len(processed), predicted_n)
        latency_s = 0.0 if self.t_request is None else float(time.now() - self.t_request)
        if self.latency is not None and latency_s > 0:
            self.latency.add(latency_s)
        merge_delay = merge_delay_steps(
            latency_s,
            s.time_per_action,
            leftover_present=self.leftover_at_request,
            consumed_steps=self.queue.get_action_index() - self.index_before_inference,
        )
        if s.use_rtc and not self.leftover_at_request:
            logger.info("First chunk: RTC delay skip disabled (queue was empty, latency={:.2f}s)", latency_s)
        self.queue.merge(
            self.as_tensor(original),
            self.as_tensor(processed),
            merge_delay,
            action_index_before_inference=self.index_before_inference,
        )
        self.processing = False
        self.t_request = None
        self.leftover_at_request = False
        first = self.queue.get_processed_left_over()
        if first is None or not len(first):
            return
        arm_state = env.robot.arm.get_state()
        if s.joint_serve:
            ref = np.asarray(arm_state["joint_q"], dtype=float).reshape(-1)[:7]
            vec = first[:, :7].cpu().numpy()
            unit = "rad"
        else:
            ref = np.asarray(arm_state["eef_pose"], dtype=float).reshape(-1)[:3]
            vec = first[:, :3].cpu().numpy()
            unit = "m"
        first_vec = vec[0]
        nref = min(len(first_vec), len(ref))
        raw_jump = float(np.linalg.norm(first_vec[:nref] - ref[:nref]))
        cmd_ref = last_command[:nref] if has_command else first_vec[:nref]
        cmd_jump = float(np.linalg.norm(first_vec[:nref] - cmd_ref))
        span = float(np.linalg.norm(vec[-1] - vec[0])) if len(vec) > 1 else 0.0
        step = float(np.max(np.linalg.norm(np.diff(vec, axis=0), axis=1))) if len(vec) > 1 else 0.0
        if control is not None and not s.joint_serve:
            control.note_plan(vec)
        logger.info(
            "RTC merge delay={} leftover_q={} latency={:.2f}s first {:.3f} {} vs measured, {:.3f} {} vs last cmd, "
            "span {:.3f} {}, max step {:.3f} {}",
            merge_delay,
            self.queue.qsize(),
            latency_s,
            raw_jump,
            unit,
            cmd_jump,
            unit,
            span,
            unit,
            step,
            unit,
        )
        eef = np.asarray(arm_state.get("eef_pose", ref), dtype=float)
        diag.chunk(self.chunk_idx, first.cpu().numpy(), eef, delay=merge_delay, qsize=self.queue.qsize(), latency_s=latency_s)
        self.chunk_idx += 1
