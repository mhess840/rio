"""Run the Kassow + Vive + AG95 + RealSense setup.

Default ``--experiment sort`` uses the new overview camera and ``home_sort``.
``--experiment pick`` restores the previous overview serial and ``home_pick``.
"""

from rio.apps.kassow.stations import KassowViveGripperStation
from rio.apps.kassow.teleop import cli

if __name__ == "__main__":
    cli(KassowViveGripperStation)
