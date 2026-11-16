# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Grow the calibrated grasp area of the table map beyond its grasp samples.

The map's pixel-to-table mapping comes from the hand-eye pose and holds in the
whole image; the grasp plane and orientation are planar. What limits ``locate``
and the target skills is the area: the hull of the grasp samples, and their
joints as IK seeds. This adds points without new cube samples:

1. ``plan`` (no motion): a pixel grid over the Jev image becomes TCP targets.
   A point stays when depth shows bare table there (height relative to the old
   samples, nothing tall nearby) and the URDF IK reaches it 15 cm above and at
   grasp height, also with the wrist turned ±45° (``align_yaw``). Writes
   ``plan.json`` and ``overlay.jpg``.
2. ``survey`` (moves the arm; table cleared): visits every planned point
   ``--hover-m`` above grasp height, checks the measured TCP, keeps the joints
   as IK seeds. Every move waits for space unless ``--no-confirm``.
3. ``write``: the map's area grows to the hull of old and surveyed points; the
   old map is kept as ``table_map.<time>.json``.

    uv run python -m rio.apps.kassow.graph.extend_area                  # plan
    uv run python -m rio.apps.kassow.graph.extend_area --command survey --no-dry-run
    uv run python -m rio.apps.kassow.graph.extend_area --command write
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from rio.apps.kassow.experiments import experiments_root
from rio.apps.terminal import term

from .depth import back_project
from .handeye import HandEye, load_handeye
from .table_map import TableMap, calibration_dir, convex_hull, inside_hull, table_map_path

APPROACH_M = 0.15
TURNS_DEG = (-45.0, 0.0, 45.0)
MAX_TCP_ERROR_M = 0.005


@dataclass
class Args:
    """Grow the calibrated grasp area."""

    command: Literal["plan", "survey", "write"] = "plan"
    look: Path | None = None
    """Folder with ``jev.jpg`` and ``jev.depth.npz`` (``depth_check``). Unset: the newest such folder."""
    step_px: int = 40
    """Grid spacing in image pixels (≈ 6 cm on the table)."""
    table_tolerance_m: float = 0.008
    """Local table height may differ this much from the height at the old samples."""
    clearance_m: float = 0.05
    """Nothing taller than 15 mm above the table within this radius."""
    hover_m: float = 0.08
    """``survey``: height above the grasp height at each point."""
    dry_run: bool = True
    """``survey``: print the moves only. ``--no-dry-run`` moves the arm."""
    confirm: bool = True
    """``survey``: wait for space before each move; q stops."""
    home_joint_speed: float = 0.2
    mw: str = "Thread"


def extend_dir() -> Path:
    return calibration_dir() / "extend"


def newest(pattern: str, root: Path) -> Path | None:
    found = sorted(root.glob(pattern))
    return found[-1] if found else None


# --- plan ------------------------------------------------------------------------------


def plan(args: Args) -> Path:
    """Candidate points, their checks, ``plan.json`` and ``overlay.jpg``. Nothing moves."""
    from rio_hw.robots.kassow_kinematics import KassowKinematics

    look = args.look or _newest_look()
    image = cv2.imread(str(look / "jev.jpg"))
    depth = np.load(look / "jev.depth.npz")["depth_mm"].astype(np.float32) / 1000.0
    table = TableMap.load(table_map_path())
    handeye = load_handeye()
    kin = KassowKinematics()
    object_z = float(table.stats["object_z_m"])
    tool = table.tool_xy()
    height = _height_map(depth, handeye)
    # Compare with the table plane fitted to the depth itself: the depth is tilted (1.7 deg on 2026-11-16),
    # so one reference height would call far table "not table". The residual keeps holes, plate and crate.
    from .depth import fit_table_plane

    plane = fit_table_plane(depth, handeye)
    h0, w0 = depth.shape
    vv, uu = np.mgrid[0:h0, 0:w0]
    xyz_ok = np.isfinite(height)
    flat = np.full_like(height, np.nan)
    pts = back_project(depth, uu[xyz_ok], vv[xyz_ok], handeye)
    flat[xyz_ok] = pts[:, 2] - (plane["a"] * pts[:, 0] + plane["b"] * pts[:, 1] + plane["c"])
    height = flat
    reference = _reference_height(height, table, handeye, object_z, tool)
    radius_px = max(3, int(args.clearance_m / 0.0014))
    h, w = depth.shape
    points = []
    for v in range(args.step_px // 2, h, args.step_px):
        for u in range(args.step_px // 2, w, args.step_px):
            hit = handeye.pixel_to_plane(float(u), float(v), object_z)
            if hit is None:
                continue
            x, y = (hit[:2] - tool).tolist()
            point = {"u": u, "v": v, "x": round(x, 4), "y": round(y, 4)}
            if inside_hull(table.hull, np.array([x, y])):
                point["status"] = "calibrated"
            else:
                point |= _check(height, u, v, radius_px, reference, args.table_tolerance_m)
                if point["status"] == "ok":
                    point |= _reach(kin, table, x, y, args.hover_m)
            points.append(point)
    ok = [p for p in points if p["status"] == "ok"]
    grown = convex_hull(np.vstack([table.hull, [[p["x"], p["y"]] for p in ok]])) if ok else table.hull
    blocked = [
        p for p in points if p["status"] in ("obstacle", "not_table", "ik") and inside_hull(grown, np.array([p["x"], p["y"]]))
    ]
    out = extend_dir() / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    payload = {
        "look": str(look),
        "reference_residual_m": round(reference, 4),
        "depth_plane": plane,
        "hover_m": args.hover_m,
        "points": points,
        "new": len(ok),
        "blocked_inside_new_area": [(p["u"], p["v"], p["status"]) for p in blocked],
        "area_m2": {"old": round(_area(table.hull), 4), "new": round(_area(grown), 4)},
    }
    (out / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    _overlay(image, points, table, grown, handeye, object_z, tool, out / "overlay.jpg")
    counts = {s: sum(p["status"] == s for p in points) for s in ("calibrated", "ok", "not_table", "obstacle", "ik")}
    term(f"grid {len(points)} points: {counts}")
    term(f"area {payload['area_m2']['old']:.3f} m² → {payload['area_m2']['new']:.3f} m²")
    if blocked:
        term(f"warning: {len(blocked)} rejected points lie inside the grown area: {payload['blocked_inside_new_area']}")
    term(f"plan: {out}")
    return out


def _newest_look() -> Path:
    folders = sorted(p.parent for p in (experiments_root() / "graph" / "look").glob("*/jev.depth.npz"))
    if not folders:
        raise SystemExit("no Jev photo with depth; run graph.depth_check first (table cleared)")
    return folders[-1]


def _height_map(depth: np.ndarray, handeye: HandEye) -> np.ndarray:
    """Base-frame z of every pixel (NaN without depth)."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    us, vs = us.ravel(), vs.ravel()
    z = np.full(h * w, np.nan)
    valid = depth[vs, us] > 0
    z[valid] = back_project(depth, us[valid], vs[valid], handeye)[:, 2]
    return z.reshape(h, w)


def _project(handeye: HandEye, xyz: np.ndarray) -> tuple[int, int]:
    intr = handeye.intrinsics
    p = np.linalg.inv(handeye.transform) @ np.array([*xyz, 1.0])
    return round(intr.fx * p[0] / p[2] + intr.cx), round(intr.fy * p[1] / p[2] + intr.cy)


def _reference_height(height: np.ndarray, table: TableMap, handeye: HandEye, object_z: float, tool: np.ndarray) -> float:
    """Median depth height of the table at the old grasp samples (relative reference; depth has an offset)."""
    values = []
    for x, y in table.seeds_xy:
        u, v = _project(handeye, np.array([x + tool[0], y + tool[1], object_z]))
        patch = height[max(0, v - 6) : v + 7, max(0, u - 6) : u + 7]
        if np.isfinite(patch).any():
            values.append(float(np.nanmedian(patch)))
    if not values:
        raise SystemExit("no depth at the old samples")
    return float(np.median(values))


def _check(height: np.ndarray, u: int, v: int, radius: int, reference: float, tolerance: float) -> dict:
    patch = height[max(0, v - 6) : v + 7, max(0, u - 6) : u + 7]
    if not np.isfinite(patch).any():
        return {"status": "not_table", "why": "no depth"}
    local = float(np.nanmedian(patch))
    if abs(local - reference) > tolerance:
        return {
            "status": "not_table",
            "why": f"table {1000 * (local - reference):+.0f} mm",
            "dz_mm": round(1000 * (local - reference), 1),
        }
    around = height[max(0, v - radius) : v + radius + 1, max(0, u - radius) : u + radius + 1]
    tallest = float(np.nanpercentile(around, 99)) - reference
    if tallest > 0.015:
        return {"status": "obstacle", "why": f"{1000 * tallest:.0f} mm tall nearby"}
    return {"status": "ok", "dz_mm": round(1000 * (local - reference), 1)}


def _reach(kin, table: TableMap, x: float, y: float, hover_m: float) -> dict:
    """IK at hover, approach and grasp height, wrist turned -45/0/+45; seeds from the nearest sample."""
    from rio.apps.kassow.goto import REACH_M, REACH_ROT_RAD

    lower, upper = kin.model.lowerPositionLimit, kin.model.upperPositionLimit
    hover_q = None
    for turn in TURNS_DEG:
        tx, ty, rotvec = table.grasp_pose(x, y, turn)
        z = table.z_at(tx, ty)
        seed = table.seed_q(tx, ty)
        for dz in (hover_m, APPROACH_M, 0.0) if turn == 0.0 else (APPROACH_M, 0.0):
            goal = np.concatenate([[tx, ty, z + dz], rotvec])
            q = np.asarray(kin.ik(seed, goal), dtype=float)
            reached = np.asarray(kin.fk(q), dtype=float)
            pos = float(np.linalg.norm(reached[:3] - goal[:3]))
            delta = cast(Rotation, Rotation.from_rotvec(reached[3:]).inv() * Rotation.from_rotvec(rotvec))
            rot = float(delta.magnitude())
            if pos > REACH_M or rot > REACH_ROT_RAD or np.any(q < lower) or np.any(q > upper):
                return {"status": "ik", "why": f"turn {turn:+.0f}°, +{dz:.2f} m"}
            if turn == 0.0 and dz == hover_m:
                hover_q = q
            seed = q
    return {"hover_q": [round(float(a), 4) for a in hover_q]} if hover_q is not None else {"status": "ik"}


def _area(hull: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(hull, dtype=np.float32))) if len(hull) >= 3 else 0.0


def _overlay(image, points, table, grown, handeye, object_z, tool, path: Path) -> None:
    def pix(xy):
        return _project(handeye, np.array([xy[0] + tool[0], xy[1] + tool[1], object_z]))

    def poly(corners) -> np.ndarray:
        return np.array([pix(p) for p in corners], dtype=np.int32)

    shade = image.copy()
    cv2.fillPoly(shade, [poly(table.hull)], (60, 200, 60))
    image = cv2.addWeighted(shade, 0.25, image, 0.75, 0)
    cv2.polylines(image, [poly(grown)], True, (255, 200, 60), 2)
    cv2.polylines(image, [poly(table.hull)], True, (60, 255, 60), 2)
    colours = {"ok": (255, 200, 60), "not_table": (60, 60, 255), "obstacle": (0, 140, 255), "ik": (200, 60, 200)}
    for p in points:
        if p["status"] in colours:
            cv2.circle(image, (p["u"], p["v"]), 5 if p["status"] == "ok" else 3, colours[p["status"]], -1)
    legend = [
        ("calibrated", (60, 255, 60)),
        ("new", (255, 200, 60)),
        ("not table", (60, 60, 255)),
        ("obstacle", (0, 140, 255)),
        ("IK", (200, 60, 200)),
    ]
    for i, (text, colour) in enumerate(legend):
        cv2.putText(image, text, (8, 18 + 18 * i), 0, 0.5, colour, 2)
    cv2.imwrite(str(path), image)


# --- survey and write ------------------------------------------------------------------


def survey(args: Args) -> Path:
    """Visit the planned points above grasp height; keep measured TCP and joints."""
    folder = newest("*/plan.json", extend_dir())
    if folder is None:
        raise SystemExit("no plan; run --command plan first")
    data = json.loads(folder.read_text())  # folder is the plan file; its parent holds the run
    table = TableMap.load(table_map_path())
    todo = [p for p in data["points"] if p["status"] == "ok"]
    order = _tour(todo, start=(-0.2, 0.342))
    hover = float(data["hover_m"])
    term(f"{len(order)} points, {hover:.2f} m above grasp height, from {folder.parent.name}")
    if args.dry_run:
        for p in order:
            term(f"[dry run] would moveJ above x={p['x']:.3f} y={p['y']:.3f}")
        return folder
    from .run import Args as RunArgs
    from .run import live_cell

    results = []
    out = folder.parent / "survey.json"
    with live_cell(RunArgs(mw=args.mw, home_joint_speed=args.home_joint_speed, jev_depth=False), table, folder.parent) as cell:
        from rio.apps.kassow import goto

        env = cell.skills.env  # ty: ignore[unresolved-attribute]
        for i, p in enumerate(order, start=1):
            goal = np.array([p["x"], p["y"], table.z_at(p["x"], p["y"]) + hover])
            message = f"survey {i}/{len(order)}: moveJ above x={p['x']:.3f} y={p['y']:.3f} z={goal[2]:.3f}"
            if args.confirm and not cell.confirm(message):
                term("stopped by the operator")
                break
            if not args.confirm and "q" in cell.keys.pressed():
                term("stopped by the operator")
                break
            status = goto.move_tcp(
                env, goal, table.rotvec, home_joint_speed=args.home_joint_speed, seed_q=np.array(p["hover_q"])
            )
            state = env.robot.arm.get_state()
            measured = np.asarray(state["eef_pose"], dtype=float)[:3]
            error = float(np.linalg.norm(measured - goal))
            ok = status is goto.MoveStatus.LANDED and error <= MAX_TCP_ERROR_M
            results.append(
                {
                    **p,
                    "status": "surveyed" if ok else "failed",
                    "move": status.name,
                    "tcp_error_mm": round(error * 1000, 1),
                    "joint_q": [round(float(a), 4) for a in state["joint_q"]],
                }
            )
            term(f"  {status.name}, TCP error {error * 1000:.1f} mm")
            out.write_text(json.dumps({"plan": str(folder), "points": results}, indent=2) + "\n")
            if status is goto.MoveStatus.ALARM:
                break
        cell.skills.move_to(resolve_home())
    term(f"survey: {out}")
    return out


def resolve_home() -> Path:
    from .check import resolve_pose

    return resolve_pose("home_organize.json")


def _tour(points: list[dict], start: tuple[float, float]) -> list[dict]:
    """Nearest-neighbour order from ``start``."""
    left, order, here = list(points), [], np.array(start)
    while left:
        nearest = min(left, key=lambda p: math.hypot(p["x"] - here[0], p["y"] - here[1]))
        left.remove(nearest)
        order.append(nearest)
        here = np.array([nearest["x"], nearest["y"]])
    return order


def write(args: Args) -> Path:
    """Grow the map's area by the surveyed points; keep the old map."""
    survey_file = newest("*/survey.json", extend_dir())
    if survey_file is None:
        raise SystemExit("no survey; run --command survey --no-dry-run first")
    points = [p for p in json.loads(survey_file.read_text())["points"] if p["status"] == "surveyed"]
    if not points:
        raise SystemExit("no surveyed point")
    path = table_map_path()
    table = TableMap.load(path)
    backup = path.with_name(f"table_map.{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy(path, backup)
    xy = np.array([[p["x"], p["y"]] for p in points])
    q = np.array([p["joint_q"] for p in points])
    hull = convex_hull(np.vstack([table.hull, xy]))
    stats = {
        **table.stats,
        "extended": {
            "at": datetime.now().isoformat(timespec="seconds"),
            "points": len(points),
            "survey": str(survey_file),
            "backup": str(backup),
        },
    }
    grown = replace(
        table,
        **{
            "hull": hull,
            "seeds_xy": np.vstack([table.seeds_xy, xy]),
            "seeds_q": np.vstack([table.seeds_q, q]),
            "stats": stats,
        },
    )
    grown.save(path)
    term(f"area {_area(table.hull):.3f} m² → {_area(hull):.3f} m², {len(points)} seeds added; old map: {backup}")
    return path


def main(args: Args) -> Path:
    if args.command == "plan":
        return plan(args)
    if args.command == "survey":
        return survey(args)
    return write(args)


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
