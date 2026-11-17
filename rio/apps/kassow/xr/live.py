# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Stream where the gripper fingertips are to the headset. Reads the arm, never moves it.

The headset view must already run (``uv run python -m rio.apps.kassow.xr --headset``).

    uv run python -m rio.apps.kassow.xr.live

Only the arm is connected; the gripper node stays off so a held object is not
released. The opening is therefore a fixed ``--opening`` (0 closed, 1 open).
This opens its own arm connection: run it while no other program drives the arm.
Ctrl-C stops.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager
from scipy.spatial.transform import Rotation

from rio.apps.kassow.goto import Args as GotoArgs
from rio.envs.env import make_env
from rio.xr.auth import read_token, token_path
from rio.xr.plan import gripper_tips
from rio.xr.publisher import XrPublisher
from rio.xr.scene import Layer, Lines, Points

LAYER_NAME = "gripper_live"
TIP = (1.0, 1.0, 1.0, 1.0)
SHAFT = (1.0, 1.0, 1.0, 0.5)
AXES = ((1.0, 0.2, 0.2, 1.0), (0.3, 1.0, 0.4, 1.0), (0.4, 0.6, 1.0, 1.0))


@dataclass
class Args(GotoArgs):
    """Show the live fingertip in the headset. Reads the arm only."""

    opening: float = 0.0
    """Assumed AG95 opening, 0 closed (tip 200 mm from the flange) to 1 open (185 mm)."""
    rate_hz: float = 15.0
    """Updates per second sent to the headset."""
    gripper: str | None = None
    gripper_cfg: None = None  # ty: ignore[invalid-attribute-override]


def gripper_layer(flange_pose6: np.ndarray, opening: float) -> Layer:
    """Fingertip cross with the flange axes, and a line from the flange to the tip."""
    pose = np.asarray(flange_pose6, dtype=float).reshape(6)
    tip = gripper_tips(pose[None], opening=opening)[0]
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    items: list[Points | Lines] = [
        Points(tip[None], TIP, 0.03),
        Lines(np.stack([pose[:3], tip]), SHAFT),
    ]
    for axis, color in enumerate(AXES):
        items.append(Lines(np.stack([tip, tip + rotation[:, axis] * 0.05]), color))
    return Layer(LAYER_NAME, "Gripper fingertip (live)", tuple(items))


def stream(args: Args, env) -> None:
    publisher = XrPublisher(token=read_token(token_path()))
    period = 1.0 / args.rate_hz
    logger.info("Streaming the fingertip at {:.0f} Hz, opening {:.2f}. Ctrl-C stops.", args.rate_hz, args.opening)
    try:
        while True:
            pose = np.asarray(env.robot.arm.get_state()["eef_pose"], dtype=float).reshape(6)
            if np.all(np.isfinite(pose)):
                publisher.publish(gripper_layer(pose, args.opening))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()


def main(args: Args) -> None:
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with env:
            stream(args, env)


def cli() -> None:
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)


if __name__ == "__main__":
    cli()
