# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Vive teleop of the Kassow station, with its start pose and KORD homing (``--go-home-on-save``)::

uv run -m rio.apps.kassow.teleop --experiment sort --go-home-on-save
"""

from typing import Any

from rio.apps.kassow.homing import KordHoming
from rio.apps.vive import teleop as vive


def teleop_vive_hand(args, env, teleop, **kwargs) -> None:
    """One Vive teleop session that loads or captures the start pose and returns to it with KORD moveJ."""
    vive.teleop_vive_hand(args, env, teleop, homing_factory=KordHoming.for_session, **kwargs)


def main(args) -> None:
    vive.main(args, KordHoming.for_session)


def cli(station_cls: type[Any] | None = None) -> None:
    """Parse a station config from the command line and teleoperate it.

    Args:
        station_cls: Station config class; defaults to ``KassowViveGripperStation``.
    """
    if station_cls is None:
        from rio.apps.kassow.stations import KassowViveGripperStation

        station_cls = KassowViveGripperStation
    vive.cli(station_cls, KordHoming.for_session)


if __name__ == "__main__":
    cli()
