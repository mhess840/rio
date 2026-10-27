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
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from examples.diagnose_pi05_offline import _action_window, _episode_bounds, _load_dataset, episode_mid_index
from examples.policy_inference import _as_action_np, require_arm_controller
from rio.envs.env import make_env

try:
    import torch
    from lerobot.utils.action_interpolator import ActionInterpolator
except ImportError:
    torch = None
    ActionInterpolator = None


def replay_loop(args, env) -> None:
    require_arm_controller(env.robot.arm)
    if ActionInterpolator is None or torch is None:
        raise ImportError("LeRobot ActionInterpolator is required. Install with: uv sync --group pi")

    ds = _load_dataset(Path(args.dataset_root), args.repo_id)
    from_i, to_i = _episode_bounds(ds, args.episode)
    if args.start_frame is not None:
        start = int(from_i) + int(args.start_frame)
    else:
        horizon = min(50, int(to_i) - int(from_i))
        start = episode_mid_index(from_i, to_i, horizon)
    n = int(to_i) - start
    if n <= 1:
        raise ValueError(f"episode {args.episode} has no actions from index {start}")
    actions = _action_window(ds, start, n)
    logger.info("Replaying episode {} from frame {} ({} steps @ 20 Hz)", args.episode, start, len(actions))

    freq = args.freq
    dt = 1.0 / freq
    action_hz = 20.0
    ticks = max(1, round(freq / action_hz))
    interpolator = ActionInterpolator(multiplier=ticks)
    binarize = bool(getattr(args.policy_cfg, "binarize_gripper", True))
    action_dim = 7

    input("Press Enter to start GT replay (E-Stop ready)...")
    t_start = time.now()
    env.set_start_time(t_start)
    it = 0
    act_i = 0
    try:
        while act_i < len(actions):
            t_cycle_end = t_start + (it + 1) * dt
            t_cmd_target = t_cycle_end + dt
            if interpolator.needs_new_action():
                interpolator.add(torch.as_tensor(actions[act_i]))
                act_i += 1
            interp = interpolator.get()
            if interp is not None:
                cmd = _as_action_np(interp, action_dim, binarize_gripper=binarize)
                env.move(cmd, t_cmd_target=t_cmd_target + args.arm_latency)
            time.precise_wait(t_cycle_end)
            it += 1
            if it % freq == 0:
                logger.info("replay {:.0f} s, action {}/{}", it * dt, act_i, len(actions))
    except KeyboardInterrupt:
        logger.info("Replay interrupted")
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

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
