"""Command-line entry point for the Kassow LeRobot converter."""

import tyro

from rio.data.convert_pi05 import ConvertPi05Cfg, convert


def main(cfg: ConvertPi05Cfg) -> dict:
    """Convert local trajectories using the shared dataset conversion implementation."""
    return convert(cfg)


if __name__ == "__main__":
    main(tyro.cli(ConvertPi05Cfg))
