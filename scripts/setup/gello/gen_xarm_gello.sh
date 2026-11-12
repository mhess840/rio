#!/bin/bash

# Generate a Gello configuration file for the xArm7 robot arm.
uv run python -m scripts.setup.gello.generate_config \
    --start-joints 0 -0.785398 0 0.785398 0 1.5708 0 \
    --joint-signs 1 1 1 1 1 1 1 \
    --output-path examples/station_cfgs/data/gello_xarm7_left.yaml