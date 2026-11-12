# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Play a kassow_pi05 episode through the same Kassow IK as Serve. No policy.

Same 100 Hz loop and ActionInterpolator x5 as ``examples.policy_inference``.
If the arm tracks this smoothly, IK can execute 20 Hz TCP. If it jitters on
ground-truth actions, the Kassow path is the problem, not π₀.₅.

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.replay_pi05_dataset \\
        --episode 0
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from loguru import logger
from rio_hw import time
from rio_hw.interpolators import ActionInterpolator
from rio_hw.middleware import ServerManager

from examples.diagnose_pi05_offline import _action_window, _episode_bounds, _load_dataset
from examples.policy_inference import (
    RolloutSafetyCfg,
    _as_action_np,
    command_tracking_error,
    first_waypoint_needs_creep,
    is_joint_serve,
    limit_tcp_action_step,
    measured_tcp_action,
    move_action_checked,
    require_arm_controller,
    resolve_control_tick,
    stop_arm_motion,
    validate_action_chunk,
)
from rio.envs.env import make_env
from rio.gripper import GripperEncoding


def replay_loop(args, env) -> None:
    require_arm_controller(env.robot.arm)
    if is_joint_serve(args):
        raise ValueError("This replay expects a TCP dataset and Cartesian station configuration")
    ds = _load_dataset(Path(args.dataset_root), args.repo_id)
    from_i, to_i = _episode_bounds(ds, args.episode)
    if args.start_frame is not None:
        start = int(from_i) + int(args.start_frame)
    else:
        start = int(from_i)
    if start < int(from_i) or start >= int(to_i):
        raise ValueError("start_frame must be inside the selected episode")
    n = int(to_i) - start
    if n <= 1:
        raise ValueError(f"episode {args.episode} has no actions from index {start}")
    actions = validate_action_chunk(_action_window(ds, start, n), 7)
    logger.info("Replaying episode {} from frame {} ({} steps @ 20 Hz)", args.episode, start, len(actions))

    freq = args.freq
    if freq <= 0 or freq % 20:
        raise ValueError("The replay control frequency must be a positive multiple of 20 Hz")
    dt = 1.0 / freq
    action_hz = 20.0
    ticks = max(1, round(freq / action_hz))
    binarize = bool(getattr(args.policy_cfg, "binarize_gripper", True))
    interpolator = ActionInterpolator(multiplier=ticks, rotvec=True, discrete_indices=(-1,) if binarize else ())
    gripper_encoding: GripperEncoding = {
        "gripper_close": float(getattr(args.policy_cfg, "gripper_close", 0.0)),
        "gripper_open": float(getattr(args.policy_cfg, "gripper_open", 0.7)),
        "gripper_threshold": getattr(args.policy_cfg, "gripper_threshold", None),
    }
    action_dim = 7
    safety = getattr(args, "safety_cfg", None) or RolloutSafetyCfg()

    input("Press Enter to start GT replay (E-Stop ready)...")
    previous = measured_tcp_action(env, action_dim, binarize_gripper=binarize, **gripper_encoding)
    if first_waypoint_needs_creep(command_tracking_error(previous, actions[0], joint_serve=False), joint_serve=False):
        raise RuntimeError("Align the arm with the first dataset waypoint before starting replay")
    interpolator.reset(previous)
    t_start = time.now()
    env.set_start_time(t_start)
    it = 0
    act_i = 0
    try:
        while act_i < len(actions) or not interpolator.needs_new_action():
            it, t_cycle_end, t_cmd_target = resolve_control_tick(t_start, it, dt, time.now())
            if interpolator.needs_new_action():
                interpolator.add(actions[act_i])
                act_i += 1
            interp = interpolator.get()
            if interp is not None:
                cmd = _as_action_np(interp, action_dim, binarize_gripper=binarize, **gripper_encoding)
                cmd = limit_tcp_action_step(
                    cmd,
                    previous,
                    dt=dt,
                    max_pos_speed=float(getattr(args.arm_cfg, "max_pos_speed", 0.45)),
                    max_rot_speed=float(getattr(args.arm_cfg, "max_rot_speed", 1.0)),
                )
                move_action_checked(env, cmd, t_cmd_target + args.arm_latency, joint_serve=False, limits=safety)
                previous = cmd
            time.precise_wait(t_cycle_end)
            it += 1
            if it % freq == 0:
                logger.info("replay {:.0f} s, action {}/{}", it * dt, act_i, len(actions))
    except KeyboardInterrupt:
        logger.info("Replay interrupted")
    finally:
        interpolator.reset()
        stop_arm_motion(env.robot.arm)
    logger.info("Replay done ({} interpolator ticks)", it)


def main(args):
    args.policy = None
    args.cameras = {}
    servers, _clients, env = make_env(args)
    live = [s for s in servers.values() if s is not None]
    with ServerManager(args.mw, live):
        with env:
            replay_loop(args, env)


if __name__ == "__main__":
    import examples.policy_cfgs as policy_cfgs
    from examples import get_policy_cfg

    PolicyCfg = get_policy_cfg(policy_cfgs)

    @dataclass
    class Args(PolicyCfg):
        dataset_root: str = "data/lerobot/kassow_pi05"
        repo_id: str = "local/kassow_pi05"
        episode: int = 0
        start_frame: int | None = None
        safety_cfg: RolloutSafetyCfg = field(default_factory=RolloutSafetyCfg)

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
