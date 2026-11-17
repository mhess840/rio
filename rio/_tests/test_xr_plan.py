# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Layer geometry, pose capture, and the VLA polyline. No headset and no server."""

import numpy as np
import pytest

from rio.xr.capture import PoseCapture
from rio.xr.geometry import layer_vertices, preview_transform, transform_points
from rio.xr.marker import corners_in_base, marker_corners
from rio.xr.plan import vla_plan_layer
from rio.xr.scene import Layer, Lines, Points

pytestmark = pytest.mark.unit


def test_a_30_mm_square_is_accepted_as_a_registration():
    from rio.xr.registration import fit_rigid

    base = marker_corners()
    moved = base + np.array([1.0, 2.0, 3.0])
    _, rms = fit_rigid(base, moved, min_spread_m=0.012)
    assert rms == pytest.approx(0.0, abs=1e-9)


def test_gripper_marker_corners_are_a_30_mm_square():
    local = marker_corners()
    assert np.linalg.norm(local[0] - local[1]) == pytest.approx(0.030)
    assert np.linalg.norm(local[1] - local[2]) == pytest.approx(0.030)
    placed = list(corners_in_base(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])).values())
    assert np.linalg.norm(placed[0] - placed[2]) == pytest.approx(0.030 * np.sqrt(2))


def test_preview_puts_robot_up_in_front_of_the_head():
    moved = transform_points(np.array([[0.0, 0.0, 1.0]]), preview_transform())
    np.testing.assert_allclose(moved[0], [0.0, 0.6, -1.2])


def test_a_point_becomes_a_cross_and_hidden_layers_drop_out():
    point = Layer("dot", "Dot", (Points(np.array([[1.0, 0.0, 0.0]]), (1.0, 0.0, 0.0, 1.0), 0.2),))
    hidden = Layer(
        "hide",
        "Hide",
        (
            Lines(
                np.zeros((2, 3)),
                (0.0, 1.0, 0.0, 1.0),
            ),
        ),
        visible=False,
    )
    lines, triangles = layer_vertices([point, hidden], np.eye(4))
    assert len(lines) == 6
    assert len(triangles) == 0
    assert lines[:, 3].max() == pytest.approx(1.0)


def test_capture_fits_after_three_spread_points_and_undo_clears_it():
    capture = PoseCapture({"a": np.zeros(3), "b": np.array([1.0, 0.0, 0.0]), "c": np.array([0.0, 1.0, 0.0])})
    capture.capture(np.array([10.0, 0.0, 0.0]))
    capture.capture(np.array([11.0, 0.0, 0.0]))
    assert capture.transform is None
    capture.capture(np.array([10.0, 1.0, 0.0]))
    assert capture.transform is not None
    assert capture.rms == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(capture.transform[:3, 3], [10.0, 0.0, 0.0])
    capture.undo()
    assert capture.transform is None
    assert capture.next_name == "c"


def test_vla_plan_skips_executed_steps_and_colours_the_gripper():
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert vla_plan_layer(actions[:1]) is None
    layer = vla_plan_layer(actions, skip=1)
    assert layer is not None and layer.name == "vla_plan"
    kinds = [type(item).__name__ for item in layer.items]
    assert kinds == ["Lines", "Lines", "Points"]
    assert layer.items[0].color[2] > layer.items[1].color[2]


def test_fingertips_sit_on_the_flange_z_axis_and_come_closer_when_open():
    from rio.xr.plan import gripper_tips

    down = np.array([0.4, 0.1, 0.5, np.pi, 0.0, 0.0])
    closed = gripper_tips(np.concatenate([down, [0.0]])[None])[0]
    opened = gripper_tips(np.concatenate([down, [1.0]])[None])[0]
    np.testing.assert_allclose(closed, [0.4, 0.1, 0.3], atol=1e-9)
    np.testing.assert_allclose(opened, [0.4, 0.1, 0.315], atol=1e-9)
    np.testing.assert_allclose(gripper_tips(down[None], opening=0.0)[0], closed)
