# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollout overlay: trail, replanning fan, gripper events, ghost arm. No headset, no robot."""

import time

import numpy as np
import pytest

from rio.xr.ghost import ghost_items, sphere, tube
from rio.xr.rollout_view import RolloutView, events_layer, fan_layer, trail_layer

pytestmark = pytest.mark.unit

DOWN = [np.pi, 0.0, 0.0]  # flange z pointing down


class _Publisher:
    def __init__(self) -> None:
        self.layers: dict[str, object] = {}

    def publish(self, layer) -> None:
        self.layers[layer.name] = layer

    def remove(self, name: str) -> None:
        self.layers[name] = None


def _wait(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_meshes_have_valid_triangles():
    vertices, triangles = tube(np.zeros(3), np.array([0.0, 0.0, 0.3]), 0.05)
    assert triangles.max() < len(vertices) and len(triangles) == 20
    np.testing.assert_allclose(np.linalg.norm(vertices[:, :2], axis=1), 0.05)
    vertices, triangles = sphere(np.ones(3), 0.1)
    assert triangles.max() < len(vertices)
    np.testing.assert_allclose(np.linalg.norm(vertices - 1.0, axis=1), 0.1)


def test_trail_fades_and_fan_and_events_split_by_kind():
    trail = trail_layer(np.linspace([0, 0, 0], [0.5, 0, 0], 40))
    assert trail is not None
    alphas = [item.color[3] for item in trail.items]
    assert alphas == sorted(alphas) and alphas[-1] > 0.9
    assert trail_layer(np.zeros((1, 3))) is None
    fan = fan_layer([np.zeros((5, 3)), np.ones((5, 3))])
    assert fan is not None and fan.items[0].color[3] < fan.items[1].color[3]
    events = events_layer([(np.zeros(3), True), (np.ones(3), False), (np.ones(3), True)])
    assert events is not None and len(events.items) == 2
    assert events_layer([]) is None


def test_the_view_records_trail_events_and_keeps_the_newest_chunk_out_of_the_fan():
    publisher = _Publisher()
    view = RolloutView(publisher, ghost=None, rate_hz=50)
    for x in np.linspace(0.3, 0.5, 30):
        view.note_step([x, 0.0, 0.4, *DOWN], 1.0)
    view.note_step([0.5, 0.0, 0.4, *DOWN], 0.0)  # close
    view.note_step([0.5, 0.0, 0.5, *DOWN], 0.0)
    view.note_step([0.5, 0.0, 0.5, *DOWN], 1.0)  # open
    chunk = np.array([[0.5, 0.0, 0.4, *DOWN, 1.0], [0.6, 0.0, 0.4, *DOWN, 0.0]])
    view.note_chunk(chunk, None, None)
    view.note_chunk(chunk + np.array([0, 0.1, 0, 0, 0, 0, 0]), None, None)
    assert _wait(lambda: {"vla_trail", "vla_events", "vla_fan"} <= set(publisher.layers))
    assert _wait(lambda: publisher.layers["vla_fan"] is not None and len(publisher.layers["vla_fan"].items) == 1)
    events = publisher.layers["vla_events"]
    assert [item.color[0] for item in events.items] == [1.0, 0.3]  # one grasp, one release
    # Fingertip, not flange: 200 mm below the downward flange when closed.
    assert events.items[0].vertices[:, 2].mean() == pytest.approx(0.2, abs=0.01)
    view.reset()
    assert _wait(lambda: publisher.layers["vla_trail"] is None and publisher.layers["vla_fan"] is None)


def test_the_ghost_arm_solves_the_kassow_flange_and_draws_it():
    pytest.importorskip("pinocchio")
    from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

    from rio.xr.ghost import GhostArm

    arm = GhostArm(DEFAULT_URDF_PATH)
    q = np.array([0.2, -0.4, 0.3, 1.2, -0.2, 0.6, 0.1])
    centres, flange = arm.joints(q)
    assert centres.shape == (8, 3)
    from scipy.spatial.transform import Rotation

    goal = np.concatenate([flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_rotvec()])
    started = time.perf_counter()
    solved = arm.solve(goal, q + 0.15)
    elapsed = time.perf_counter() - started
    assert solved is not None
    np.testing.assert_allclose(arm.joints(solved)[1][:3, 3], flange[:3, 3], atol=1e-3)
    assert elapsed < 0.05
    assert arm.solve(np.array([5.0, 0.0, 0.0, *DOWN]), q) is None  # out of reach
    tubes, joints = ghost_items(centres, flange, 0.5)
    assert tubes.color[3] < 1.0 and len(joints.vertices) > 0
