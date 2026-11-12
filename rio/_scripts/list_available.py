# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path


def _list(module_path: str, label: str, env_var: str | None = None) -> None:
    mod = importlib.import_module(module_path)
    names = mod.__all__
    hint = f"  (set with {env_var}=<name>)" if env_var else ""
    print(f"Available {label}{hint}:\n")
    for name in names:
        print(f"  {name}")


def list_stations() -> None:
    # Station configs live in the checkout's examples/ (not installed); append the checkout, never shadow installs.
    checkout = Path.cwd()
    if (checkout / "examples" / "cfg" / "__init__.py").is_file() and str(checkout) not in sys.path:
        sys.path.append(str(checkout))
    try:
        _list("examples.cfg", "stations", env_var="STATION")
    except ModuleNotFoundError as error:
        if error.name != "examples":
            raise
        print("Station configs live in examples/cfg of the RIO checkout; run this from the repository root.")
        _list("rio.apps.kassow.stations", "Kassow stations")


def list_robots() -> None:
    _list("rio_hw.robots", "robots")


def list_cameras() -> None:
    _list("rio_hw.cameras", "cameras")


def list_interfaces() -> None:
    _list("rio_hw.interfaces", "interfaces")
