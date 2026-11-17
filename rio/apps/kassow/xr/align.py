# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Read the arm where it stands and then walk the four marker corners.

Park the arm so the sticker faces you. Then, in a terminal you can watch:

    uv run python -m rio.apps.kassow.xr.align --here

The script reads the controller TCP once, moves the four corner targets to
that pose, and tells you when to pull the trigger. It does not move the arm.
The headset view must already be running with ``--marker``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import urllib.request

import numpy as np
from loguru import logger

from rio.xr.auth import read_token, token_path
from rio.xr.marker import CORNER_NAMES, MARKER_SIZE_M, corners_in_base
from rio.xr.publisher import DEFAULT_URL, XrPublisher

CORNER_TEXT = {
    "marker_tl": "links oben auf dem schwarzen Quadrat",
    "marker_tr": "rechts oben",
    "marker_br": "rechts unten",
    "marker_bl": "links unten",
}


def read_live_tcp() -> np.ndarray:
    """The arm's current controller TCP. Opens the arm and closes it again."""
    from rio_hw.middleware import ServerManager

    from rio.apps.kassow.goto import Args
    from rio.envs.env import make_env

    args = Args()
    mp.set_start_method(args.mp_method, force=True)
    servers, _clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with env:
            state = env.robot.arm.get_state()
    pose = np.asarray(state["eef_pose"], dtype=float).reshape(6)
    if not np.all(np.isfinite(pose)):
        raise RuntimeError(f"the arm reported a non-finite TCP: {pose.tolist()}")
    return pose


def push_corners(pose: np.ndarray) -> dict[str, np.ndarray]:
    """Tell the running headset that the marker corners are at this TCP."""
    from rio.apps.kassow.xr.layers import poses_layer

    anchors = corners_in_base(pose)
    token = read_token(token_path())
    payload = {name: point.tolist() for name, point in anchors.items()}
    request = urllib.request.Request(
        f"{DEFAULT_URL}/anchors",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2.0) as response:
        response.read()
    publisher = XrPublisher(token=token)
    try:
        publisher.publish(poses_layer(anchors))
        publisher.flush()
    finally:
        publisher.close()
    return anchors


def prompt(corners: dict[str, np.ndarray]) -> None:
    """Walk the four corners. Enter advances only after the trigger has been pulled."""
    print()
    print("Nur der rechte Controller. Der linke wird ignoriert.")
    print("Nicht den Controller an den Roboter halten. Vor dem Griff ist ein")
    print("gelber Stab, 15 cm lang. Sein Ende auf die Ecke des schwarzen Quadrats")
    print("zielen, der Griff bleibt in der Luft. Die Drehung ist egal.")
    print("Erste Ecke links oben auf dem Muster, dann im Uhrzeigersinn.")
    print("Grip nimmt den letzten Punkt zurück.")
    print(f"Kantenlänge {MARKER_SIZE_M * 1000:.0f} mm. Den Arm nicht mehr bewegen.")
    print()
    for index, name in enumerate(CORNER_NAMES, start=1):
        xyz = np.round(corners[name], 3).tolist()
        print("=" * 64)
        print(f"Ecke {index}/4   {CORNER_TEXT[name]}   {xyz}")
        print("JETZT den Trigger am rechten Controller ziehen.")
        input("Enter hier, nachdem der Trigger gezogen wurde...")
        print(f"Im Headset-Terminal muss 'saved {name}' stehen.")
        print()
    print("Vier Ecken sind durch. Ein weißes Kreuz muss auf dem Aufkleber liegen.")


def main() -> None:
    """Read the live TCP when ``--here`` is set, otherwise print how to call it."""
    if "--here" not in sys.argv:
        print("Arm an eine Stelle fahren, an der du den Aufkleber siehst, und stehen lassen.")
        print("Dann, ohne den Arm noch einmal zu bewegen:")
        print("    uv run python -m rio.apps.kassow.xr.align --here")
        return
    logger.info("Reading the arm TCP. This connects to the controller and does not move the arm.")
    pose = read_live_tcp()
    logger.info("TCP {}", np.round(pose, 3).tolist())
    corners = push_corners(pose)
    logger.info("Headset corners updated. Leave the arm where it is.")
    prompt(corners)


if __name__ == "__main__":
    main()
