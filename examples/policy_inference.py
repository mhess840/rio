"""Forward to the rollout runtime. Kept so existing commands keep working.

``uv run -m examples.policy_inference`` resolves ``POLICY`` the same way as before
and serves it through :mod:`rio.policies.rollout`. The Kassow entry point is
``uv run -m rio.apps.kassow.serve``.
"""

from rio.policies.rollout import (
    RTC_QUEUE_THRESHOLD,
    RolloutControl,
    RolloutSafetyCfg,
    _as_action_np,
    append_cmd_eef,
    apply_serve_control,
    chunk_hold_pending,
    chunk_size_from_ckpt,
    clamp_tcp_lead,
    command_tracking_error,
    create_obs,
    first_waypoint_needs_creep,
    is_joint_serve,
    limit_joint_action_step,
    limit_tcp_action_step,
    measured_joint_action,
    measured_tcp_action,
    merge_delay_steps,
    move_action_checked,
    policy_loop,
    require_arm_controller,
    require_arm_state_safe,
    resolve_control_tick,
    resolve_n_execute,
    should_send_observation,
    slice_action_chunk,
    stop_arm_motion,
    time,
    validate_action_chunk,
)

__all__ = [
    "RTC_QUEUE_THRESHOLD",
    "RolloutControl",
    "RolloutSafetyCfg",
    "_as_action_np",
    "append_cmd_eef",
    "apply_serve_control",
    "chunk_hold_pending",
    "chunk_size_from_ckpt",
    "clamp_tcp_lead",
    "command_tracking_error",
    "create_obs",
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
    "resolve_control_tick",
    "resolve_n_execute",
    "should_send_observation",
    "slice_action_chunk",
    "stop_arm_motion",
    "time",
    "validate_action_chunk",
]


def main() -> None:
    """Serve whichever policy ``POLICY`` names, on the station that config builds."""
    from examples import get_policy_cfg, policy_cfgs
    from rio.policies.serve import cli

    cli(get_policy_cfg(policy_cfgs))


if __name__ == "__main__":
    main()
