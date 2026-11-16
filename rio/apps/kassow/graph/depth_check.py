# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the Jev camera's depth against the calibration, and measure objects. The arm does not move.

Takes a few colour + depth frames from the Jev camera only, saves them as
``experiments/graph/look/<time>/jev.jpg`` and ``jev.depth.npz`` (so ``ask --image``
and ``run --image`` can reuse them), fits the table plane, and with ``--sam``
measures each prompt's best SAM instance in depth next to its image target.

    uv run python -m rio.apps.kassow.graph.depth_check --sam "orange plate|blue cap"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro

from rio.apps.kassow.experiments import experiments_root
from rio.apps.kassow.jev_ask import fresh_rgbd, jev_camera_session
from rio.jev import encode_jpeg

from .cell import _measure
from .depth import fit_table_plane
from .handeye import load_handeye
from .locate import Box
from .table_map import TableMap, table_map_path


@dataclass
class Args:
    """Depth check of the Jev camera."""

    sam: str = ""
    """SAM noun phrases separated by ``|``; each best instance is measured in depth."""
    frames: int = 5
    """Depth frames combined (median per pixel) against sensor noise."""
    mw: str = "Thread"


def main(args: Args) -> dict:
    from rio_hw import time

    folder = experiments_root() / "graph" / "look" / f"{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True)
    colors, depths = [], []
    with jev_camera_session(mw=args.mw, depth=True) as camera:
        for _ in range(max(1, args.frames)):
            color, depth = fresh_rgbd(camera, 0.5)
            if depth is None:
                raise SystemExit("the Jev camera sends no depth (USB link? try a USB 3 port)")
            colors.append(color)
            depths.append(depth)
            time.sleep(0.25)
    stack = np.stack(depths)
    stack[stack <= 0] = np.nan
    depth_m = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    jpeg = encode_jpeg(colors[-1])
    (folder / "jev.jpg").write_bytes(jpeg)
    np.savez_compressed(folder / "jev.depth.npz", depth_mm=np.round(depth_m * 1000).astype(np.uint16))
    handeye = load_handeye()
    table = TableMap.load(table_map_path())
    plane = fit_table_plane(depth_m, handeye)
    cam = handeye.stats.get("camera_xyz_m", [0.0, 0.0, 0.0])
    report: dict = {
        "folder": str(folder),
        "valid_depth": round(float((depth_m > 0).mean()), 3),
        "table_plane": {k: round(v, 5) if isinstance(v, float) else v for k, v in plane.items()},
        "table_z_below_camera_m": round(plane["a"] * cam[0] + plane["b"] * cam[1] + plane["c"], 4),
        "calibrated_object_centre_z_m": round(float(table.stats.get("object_z_m", float("nan"))), 4),
    }
    if args.sam:
        from rio.sam import SamClient

        client = SamClient()
        objects = {}
        for prompt in (p.strip() for p in args.sam.split("|") if p.strip()):
            best = client.segment(jpeg, prompt).best()
            if best is None:
                objects[prompt] = None
                continue
            target = table.target(Box(*best.fractions(), best.score or 0.0), prompt)
            objects[prompt] = {
                "score": best.score,
                "image_target": {"x": round(target.x, 4), "y": round(target.y, 4)},
                "measured": _measure(depth_m, best.box, handeye, table, target),
            }
        report["objects"] = objects
    (folder / "depth_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
