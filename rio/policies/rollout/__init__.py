# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Hardware-checked execution of chunked policies (π₀.₅, SmolVLA) with RTC and caller control.

Modules: ``settings`` (validated config), ``safety`` (fail-closed checks, speed ceilings),
``chunks`` (when to plan, how chunks merge), ``actions`` (rate-limited command stream),
``observation`` (policy input), ``diagnostics`` (per-rollout files), ``control``
(stdin/caller stops), ``loop`` (``policy_loop``). Policy-specific processes (e.g. the
π₀.₅ GPU resident) are started by the policy config, not here.
"""

from rio_hw import time

from .actions import (
    STARTUP_JOINT_EPS,
    STARTUP_JOINT_SPEED,
    STARTUP_POS_EPS,
    STARTUP_POS_SPEED,
    STARTUP_ROT_SPEED,
    CommandStreamer,
    _as_action_np,
    command_tracking_error,
    first_waypoint_needs_creep,
    limit_joint_action_step,
    limit_tcp_action_step,
    measured_joint_action,
    measured_tcp_action,
)
from .chunks import ChunkScheduler, chunk_hold_pending, merge_delay_steps, should_send_observation, slice_action_chunk
from .control import RolloutControl, apply_serve_control, start_stdin_control
from .diagnostics import (
    RolloutDiagnostics,
    append_cmd_eef,
    create_diagnostic_session,
    default_diag_dir,
    dump_live_cameras,
    write_chunk_dump,
)
from .loop import RolloutEpisode, RolloutRuntime, policy_loop, resolve_control_tick
from .observation import create_obs
from .safety import (
    MAX_POLICY_JOINT_SPEED,
    MAX_POLICY_POS_SPEED,
    MAX_POLICY_ROT_SPEED,
    RolloutSafetyCfg,
    arm_has_stopped,
    move_action_checked,
    require_arm_controller,
    require_arm_state_safe,
    require_speed_limit,
    stop_arm_motion,
    validate_action_chunk,
)
from .settings import (
    RTC_QUEUE_THRESHOLD,
    RolloutCfg,
    RolloutSettings,
    chunk_size_from_ckpt,
    is_joint_serve,
    resolve_n_execute,
)

__all__ = [
    "MAX_POLICY_JOINT_SPEED",
    "MAX_POLICY_POS_SPEED",
    "MAX_POLICY_ROT_SPEED",
    "RTC_QUEUE_THRESHOLD",
    "STARTUP_JOINT_EPS",
    "STARTUP_JOINT_SPEED",
    "STARTUP_POS_EPS",
    "STARTUP_POS_SPEED",
    "STARTUP_ROT_SPEED",
    "ChunkScheduler",
    "CommandStreamer",
    "RolloutCfg",
    "RolloutControl",
    "RolloutDiagnostics",
    "RolloutEpisode",
    "RolloutRuntime",
    "RolloutSafetyCfg",
    "RolloutSettings",
    "_as_action_np",
    "append_cmd_eef",
    "apply_serve_control",
    "arm_has_stopped",
    "chunk_hold_pending",
    "chunk_size_from_ckpt",
    "command_tracking_error",
    "create_diagnostic_session",
    "create_obs",
    "default_diag_dir",
    "dump_live_cameras",
    "first_waypoint_needs_creep",
    "is_joint_serve",
    "limit_joint_action_step",
    "limit_tcp_action_step",
    "measured_joint_action",
    "measured_tcp_action",
    "merge_delay_steps",
    "move_action_checked",
    "policy_loop",
    "require_arm_controller",
    "require_arm_state_safe",
    "require_speed_limit",
    "resolve_control_tick",
    "resolve_n_execute",
    "should_send_observation",
    "slice_action_chunk",
    "start_stdin_control",
    "stop_arm_motion",
    "time",
    "validate_action_chunk",
    "write_chunk_dump",
]
