"""Run the Kassow + Vive + AG95 + RealSense setup.

Default ``--experiment sort`` uses the new overview camera and ``home_sort``.
``--experiment pick`` restores the previous overview serial and ``home_pick``.
"""

import multiprocessing as mp

import tyro

from examples.cfg import KassowViveGripperStation
from examples.teleop_vive_hand import main

if __name__ == "__main__":
    args = tyro.cli(KassowViveGripperStation)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
