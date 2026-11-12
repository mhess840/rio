# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve a chunked policy on its station: build the environment, attach the policy node, run ``policy_loop``."""

import multiprocessing as mp
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import tyro
from loguru import logger
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env
from rio.policies.rollout import RolloutSafetyCfg, default_diag_dir, policy_loop

DaggerOpener = Callable[[Any, Any, dict], AbstractContextManager]


@runtime_checkable
class ResidentPolicyCfg(Protocol):
    """A policy config whose weights live in a resident process that outlives this one."""

    def start_resident(self) -> Any: ...
    def stop_resident(self) -> bool: ...


def serve(args, *, open_dagger: DaggerOpener | None = None) -> None:
    """Run the policy of ``args`` on its station until the operator quits.

    Args:
        args: Station + policy config extended by ``serve_args``.
        open_dagger: Context manager factory ``(args, env, clients)`` for operator takeover; required when
            ``args.dagger_round`` is set.
    """
    resident_cfg = args if isinstance(args, ResidentPolicyCfg) else None
    if args.unload_policy:
        if resident_cfg is None:
            raise ValueError("This policy has no resident process to unload")
        resident_cfg.stop_resident()
        return
    dagger_round = getattr(args, "dagger_round", None)
    if dagger_round is not None and open_dagger is None:
        raise ValueError("This entry point does not support DAgger; use the station application")
    if resident_cfg is not None:
        resident = resident_cfg.start_resident()
        logger.info("Policy stays loaded after this process exits (resident pid {})", resident.pid)
    servers, clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with (
            env,
            clients["policy"]() as policy,
            clients["visualizer"]() if clients["visualizer"] else nullcontext() as visualizer,
        ):
            try:
                if dagger_round is None or open_dagger is None:
                    policy_loop(args, env, policy, visualizer)
                else:
                    with open_dagger(args, env, clients) as dagger:
                        policy_loop(args, env, policy, visualizer, dagger=dagger)
            except KeyboardInterrupt:
                pass
    logger.info("Serve process exiting")


def serve_args(policy_cfg_cls: type) -> type:
    """``policy_cfg_cls`` plus the rollout options of the command line (diagnostics, unload, safety)."""
    # The station config is chosen at run time, so the base is only known dynamically.
    base: Any = policy_cfg_cls

    @dataclass
    class ServeArgs(base):
        diag_dir: str = str(default_diag_dir())
        unload_policy: bool = False
        safety_cfg: RolloutSafetyCfg = field(default_factory=RolloutSafetyCfg)

    ServeArgs.__doc__ = policy_cfg_cls.__doc__
    return ServeArgs


def cli(policy_cfg_cls: type, *, open_dagger: DaggerOpener | None = None) -> None:
    """Parse the extended ``policy_cfg_cls`` from the command line and serve it."""
    args: Any = tyro.cli(serve_args(policy_cfg_cls))
    mp.set_start_method(args.mp_method, force=True)
    serve(args, open_dagger=open_dagger)
