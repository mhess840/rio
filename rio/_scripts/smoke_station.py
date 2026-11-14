# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Import every station package and parse the Kassow URDF. No hardware is opened.

A lockfile can resolve packages whose native libraries do not fit together
(pin 4.1 needs urdfdom 6 while the override keeps urdfdom 4). That only shows
up when the library is loaded, so this loads each one and builds the Kassow
model with Pinocchio.

    uv run --no-sync python -m rio._scripts.smoke_station
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MODULES = (
    "pinocchio",
    "sshkeyboard",
    "pyrealsense2",
    "openvr",
    "evdev",
    "serial",
    "lerobot",
    "rio_hw",
    "rio_hw.robots.kassow_kinematics",
)


def kassow_urdf() -> Path:
    """The Kassow URDF shipped with rio_hw."""
    rio_hw = importlib.import_module("rio_hw")
    paths = sorted((Path(rio_hw.__file__).parent / "assets" / "kassow").glob("*.urdf"))
    if not paths:
        raise FileNotFoundError("no Kassow URDF in rio_hw/assets/kassow")
    return paths[0]


def main() -> int:
    """Load each module and the URDF; print one line per check.

    Returns:
        Process exit code: 0 when everything loaded, 1 otherwise.
    """
    failed = 0
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:  # report every failure, not only the first
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    try:
        pinocchio = importlib.import_module("pinocchio")
        model = pinocchio.buildModelFromUrdf(str(kassow_urdf()))
        print(f"ok   Kassow URDF (nq={model.nq})")
    except Exception as error:
        failed += 1
        print(f"FAIL Kassow URDF: {type(error).__name__}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
