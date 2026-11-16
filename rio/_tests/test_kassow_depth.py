# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Depth geometry on a ray-cast table scene. No camera."""

import math

import numpy as np
import pytest

from rio.apps.kassow.graph.depth import fit_table_plane, measure
from rio.apps.kassow.graph.handeye import HandEye, Intrinsics

pytestmark = pytest.mark.unit

TABLE_Z = -0.007


def camera() -> HandEye:
    """Camera 0.86 m above the table, looking straight down, slightly rolled."""
    intr = Intrinsics(640, 480, 605.7, 604.6, 320.5, 248.5)
    roll = math.radians(8.0)
    down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0.0], [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ down
    transform[:3, 3] = [-0.43, 0.54, 0.86]
    return HandEye("0", intr, transform.tolist(), {}, {}, {})


def render(handeye: HandEye, objects: list[dict]) -> np.ndarray:
    """Depth (camera z) of a flat table with boxes and cylinders on it (top faces only)."""
    intr, transform = handeye.intrinsics, handeye.transform
    vs, us = np.mgrid[0:480, 0:640]
    rays = np.stack([(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones_like(us, dtype=float)], axis=-1)
    world = rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    best = (TABLE_Z - origin[2]) / world[..., 2]
    for obj in objects:
        s = (TABLE_Z + obj["h"] - origin[2]) / world[..., 2]
        hit = origin[:2] + s[..., None] * world[..., :2] - np.array([obj["x"], obj["y"]])
        if obj.get("round"):
            inside = np.hypot(hit[..., 0], hit[..., 1]) <= obj["r"]
        else:
            c, d = math.cos(math.radians(obj["yaw"])), math.sin(math.radians(obj["yaw"]))
            along, across = hit[..., 0] * c + hit[..., 1] * d, -hit[..., 0] * d + hit[..., 1] * c
            inside = (np.abs(along) <= obj["l"] / 2) & (np.abs(across) <= obj["w"] / 2)
        best = np.where(inside, np.minimum(best, s), best)
    return best.astype(np.float32)


def box_of(handeye: HandEye, depth: np.ndarray, x: float, y: float, half: float) -> tuple[int, int, int, int]:
    """Pixel box around a base-frame square, from the rendered depth (like a SAM box)."""
    intr, transform = handeye.intrinsics, handeye.transform
    inv = np.linalg.inv(transform)
    corners = []
    for dx in (-half, half):
        for dy in (-half, half):
            p = inv @ np.array([x + dx, y + dy, TABLE_Z, 1.0])
            corners.append((intr.fx * p[0] / p[2] + intr.cx, intr.fy * p[1] / p[2] + intr.cy))
    c = np.array(corners)
    return int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 0].max()), int(c[:, 1].max())


def test_cube_centre_height_and_yaw():
    he = camera()
    depth = render(he, [{"x": -0.40, "y": 0.60, "h": 0.037, "l": 0.037, "w": 0.037, "yaw": 30.0}])
    m = measure(depth, box_of(he, depth, -0.40, 0.60, 0.03), he)
    assert m is not None
    assert m.table_z == pytest.approx(TABLE_Z, abs=0.001)
    assert m.height_m == pytest.approx(0.037, abs=0.002)
    assert (m.x, m.y) == (pytest.approx(-0.40, abs=0.002), pytest.approx(0.60, abs=0.002))
    assert m.period_deg == 90.0
    assert m.yaw_deg is not None and min(abs(m.yaw_deg - 30.0), 90 - abs(m.yaw_deg - 30.0)) < 3.0


def test_bar_yaw_has_period_180():
    he = camera()
    depth = render(he, [{"x": -0.45, "y": 0.50, "h": 0.03, "l": 0.10, "w": 0.03, "yaw": 120.0}])
    m = measure(depth, box_of(he, depth, -0.45, 0.50, 0.07), he)
    assert m is not None and m.period_deg == 180.0
    assert m.length_m == pytest.approx(0.10, abs=0.005)
    assert m.yaw_deg == pytest.approx(120.0, abs=3.0)


def test_cylinder_is_round_without_yaw():
    he = camera()
    depth = render(he, [{"x": -0.50, "y": 0.69, "h": 0.05, "r": 0.02, "round": True}])
    m = measure(depth, box_of(he, depth, -0.50, 0.69, 0.03), he)
    assert m is not None and m.yaw_deg is None
    assert m.height_m == pytest.approx(0.05, abs=0.002)


def test_flat_or_empty_box_gives_none():
    he = camera()
    depth = render(he, [])
    assert measure(depth, box_of(he, depth, -0.45, 0.55, 0.03), he) is None
    holes = np.zeros_like(depth)
    assert measure(holes, (300, 200, 340, 240), he) is None


def test_table_plane_ignores_objects():
    he = camera()
    depth = render(he, [{"x": -0.40, "y": 0.60, "h": 0.04, "l": 0.15, "w": 0.15, "yaw": 0.0}])
    plane = fit_table_plane(depth, he)
    assert plane["c"] + plane["a"] * -0.43 + plane["b"] * 0.54 == pytest.approx(TABLE_Z, abs=0.001)
    assert plane["tilt_deg"] < 0.1 and plane["rms_mm"] < 1.0


def test_grasp_pose_keeps_the_object_centre_when_the_wrist_turns():
    from scipy.spatial.transform import Rotation

    from rio._tests.test_kassow_graph import _table
    from rio.apps.kassow.graph.table_map import TableMap

    base = _table()
    tilted = Rotation.from_euler("xyz", [180.0, 15.0, 30.0], degrees=True).as_rotvec()
    table = TableMap(**{**base.__dict__, "rotvec": tilted, "stats": {**base.stats, "tool_offset_m": 0.233}})
    x, y, rotvec = table.grasp_pose(-0.40, 0.60, 0.0)
    assert (x, y) == (-0.40, 0.60) and np.allclose(rotvec, tilted)
    x, y, rotvec = table.grasp_pose(-0.40, 0.60, 30.0)
    before = np.array([-0.40, 0.60]) + table.tool_xy()
    after = np.array([x, y]) + table.tool_xy(rotvec)
    assert np.allclose(before, after, atol=1e-9)
    assert np.hypot(x + 0.40, y - 0.60) > 0.01  # the tilted tool moves the TCP
    turned = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(tilted).inv()
    assert turned.as_euler("xyz", degrees=True) == pytest.approx([0.0, 0.0, 30.0], abs=1e-6)


@pytest.mark.parametrize(
    ("yaw", "jaw", "turn"),
    [(10.0, 10.0, 0.0), (40.0, 10.0, 30.0), (70.0, 10.0, -30.0), (55.0, 10.0, -45.0), (5.0, 80.0, 15.0)],
)
def test_wrist_turn_takes_the_shortest_way(yaw, jaw, turn):
    from rio.apps.kassow.graph.table_map import wrist_turn_deg

    assert wrist_turn_deg(yaw, 90.0, jaw) == pytest.approx(turn)


def test_jaw_yaw_round_trip(tmp_path):
    from rio.apps.kassow.graph.table_map import load_jaw_yaw, save_jaw_yaw

    path = tmp_path / "jaw_yaw.json"
    assert load_jaw_yaw(path) is None
    save_jaw_yaw(97.5, "look/x", path)
    assert load_jaw_yaw(path) == pytest.approx(7.5)


def paint(he: HandEye, shapes: list[dict]) -> np.ndarray:
    """Colour image of the grey table with lit top faces (and a shaded side band) projected by ``he``."""
    import cv2

    intr, inv = he.intrinsics, np.linalg.inv(he.transform)
    image = np.full((480, 640, 3), (120, 130, 120), np.uint8)

    def pixel(p):
        c = inv @ np.array([*p, 1.0])
        return [intr.fx * c[0] / c[2] + intr.cx, intr.fy * c[1] / c[2] + intr.cy]

    for s in shapes:
        z = TABLE_Z + s["h"]
        if s.get("round"):
            ring = [(s["x"] + s["r"] * math.cos(a), s["y"] + s["r"] * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 64)]
            cv2.fillPoly(image, [np.int32([pixel(p) for p in ring])], (40, 120, 230))
            continue
        c, d, half = math.cos(math.radians(s["yaw"])), math.sin(math.radians(s["yaw"])), s["side"] / 2
        corners = [
            (s["x"] + c * a - d * b, s["y"] + d * a + c * b)
            for a, b in ((-half, -half), (half, -half), (half, half), (-half, half))
        ]
        side = [corners[0], corners[1]]
        band = [pixel((*side[0], z)), pixel((*side[1], z)), pixel((*side[1], TABLE_Z)), pixel((*side[0], TABLE_Z))]
        cv2.fillPoly(image, [np.int32(band)], (30, 60, 200))  # shaded side face: darker, more red
        cv2.fillPoly(image, [np.int32([pixel((*p, z)) for p in corners])], (90, 170, 255))  # lit top face
    return image


@pytest.mark.parametrize("yaw", [5.0, 25.0, 50.0, 80.0])
def test_top_face_yaw_of_a_cube_in_colour(yaw):
    from rio.apps.kassow.graph.depth import top_face_yaw

    he = camera()
    image = paint(he, [{"x": -0.42, "y": 0.68, "h": 0.04, "side": 0.04, "yaw": yaw}])
    box = box_of(he, np.zeros((480, 640)), -0.42, 0.68, 0.035)
    face = top_face_yaw(image, box, he, TABLE_Z + 0.04)
    assert face is not None and face["period_deg"] == 90.0
    error = abs(face["yaw_deg"] - yaw % 90.0)
    assert min(error, 90.0 - error) < 3.0


def test_top_face_of_a_disc_has_no_yaw():
    from rio.apps.kassow.graph.depth import top_face_yaw

    he = camera()
    image = paint(he, [{"x": -0.50, "y": 0.69, "h": 0.05, "r": 0.025, "round": True}])
    face = top_face_yaw(image, box_of(he, np.zeros((480, 640)), -0.50, 0.69, 0.03), he, TABLE_Z + 0.05)
    assert face is not None and face["yaw_deg"] is None


def test_extend_area_tour_and_table_check():
    from rio.apps.kassow.graph.extend_area import _check, _tour

    points = [{"x": -0.5, "y": 0.5}, {"x": -0.21, "y": 0.35}, {"x": -0.3, "y": 0.4}]
    assert [p["x"] for p in _tour(points, start=(-0.2, 0.342))] == [-0.21, -0.3, -0.5]
    flat = np.zeros((100, 100))
    assert _check(flat, 50, 50, 10, 0.0, 0.008)["status"] == "ok"
    bump = flat.copy()
    bump[45:55, 58:62] = 0.03
    assert _check(bump, 50, 50, 10, 0.0, 0.008)["status"] == "obstacle"
    low = flat - 0.02
    assert _check(low, 50, 50, 10, 0.0, 0.008)["status"] == "not_table"
