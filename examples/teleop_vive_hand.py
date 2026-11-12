"""Vive teleop of the station named by ``$STATION``; implementation in :mod:`rio.apps.kassow.teleop`.

STATION=Ur5eViveInspireStation uv run -m examples.teleop_vive_hand
"""

from examples import get_station_cfg
from rio.apps.kassow.teleop import cli

if __name__ == "__main__":
    cli(get_station_cfg())
