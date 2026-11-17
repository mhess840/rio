# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollout overlay: trail, replanning fan, gripper events, ghost arm. No headset, no robot."""

import time
from pathlib import Path

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

    arm = GhostArm(DEFAULT_URDF_PATH, mesh_path=Path("/nonexistent.npz"))
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


class _Presence:
    def __init__(self, worn: bool) -> None:
        self.is_worn = worn

    def worn(self) -> bool:
        return self.is_worn


def test_the_view_waits_while_nobody_wears_the_headset():
    publisher, presence = _Publisher(), _Presence(False)
    view = RolloutView(publisher, rate_hz=50.0, presence=presence)
    view.note_step(np.array([0.5, 0.0, 0.4, *DOWN]), 1.0)
    view.note_step(np.array([0.5, 0.1, 0.4, *DOWN]), 1.0)
    time.sleep(0.15)
    assert publisher.layers == {}
    presence.is_worn = True
    assert _wait(lambda: publisher.layers.get("vla_trail") is not None)


def test_smooth_joints_eases_from_start_to_goal():
    from rio.xr.move_view import smooth_joints

    q_from, q_to = np.zeros(7), np.ones(7)
    assert np.allclose(smooth_joints(q_from, q_to, 0.0), q_from)
    assert np.allclose(smooth_joints(q_from, q_to, 0.5), 0.5)
    assert np.allclose(smooth_joints(q_from, q_to, 2.0), q_to)


def test_the_move_ghost_shows_the_goal_only_when_worn_and_previews_the_move():
    pytest.importorskip("pinocchio")
    from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

    from rio.xr.ghost import GhostArm
    from rio.xr.move_view import LAYER, PREVIEW_LAYER, MoveView

    arm = GhostArm(DEFAULT_URDF_PATH, mesh_path=Path("/nonexistent.npz"))
    q_now = np.array([5.45, 0.06, 3.14, 2.5, -0.12, 0.32, 1.6])
    _, flange = arm.joints(q_now)
    eef = flange[:3, 3] - 0.003  # the controller's flange sits a few millimetres off the URDF's
    publisher, presence = _Publisher(), _Presence(False)
    view = MoveView(publisher, arm, presence)
    view.show(q_now + 0.2, q_now, eef, 0.7)
    assert publisher.layers == {}
    presence.is_worn = True
    view.show(q_now, q_now, eef, 0.7)
    tubes, _ = publisher.layers[LAYER].items
    # Drawn on the controller's frame: the ghost of the current joints sits on the measured flange.
    assert np.min(np.linalg.norm(tubes.vertices - eef, axis=1)) < 0.06
    view.clear()
    assert publisher.layers[LAYER] is None
    view.preview(q_now, q_now + 0.1, 0.05, q_now=q_now, eef_now=eef, speed=1.0)
    assert publisher.layers[PREVIEW_LAYER] is not None


def test_the_server_reports_whether_the_headset_is_worn():
    from rio.xr.hub import XrHub

    hub = XrHub()
    assert hub.headset()["worn"] is False
    hub.set_headset_state(5)
    assert hub.headset()["worn"] is True and hub.headset()["state"] == 5
    hub.set_headset_state(3)
    assert hub.headset()["worn"] is False


def test_presence_is_false_without_a_server():
    from rio.xr.presence import HeadsetPresence

    assert HeadsetPresence("http://127.0.0.1:9", timeout_s=0.1).worn() is False


def test_clustering_merges_close_vertices_and_drops_collapsed_triangles():
    from rio.xr.robodk import cluster

    soup = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.0001, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0.0002, 0, 0], [1, 0, 0]]
    )
    vertices, triangles = cluster(soup, 0.01)
    assert len(triangles) == 2  # the third triangle collapses onto an edge
    assert len(vertices) == 4 and triangles.max() == 3


def test_robodk_dh_chain_matches_the_kassow_urdf_and_the_ghost_draws_link_meshes(tmp_path):
    pytest.importorskip("pinocchio")
    from rio_hw.robots.kassow_kinematics import DEFAULT_URDF_PATH

    from rio.xr.ghost import GhostArm
    from rio.xr.robodk import link_frames, load_link_meshes

    # The RoboDK library's KR1018 rows (alpha, a [mm], theta, d [mm]).
    half = np.pi / 2
    dh = np.array([[0, 0, 0, 357.2], [-half, 62, 0, 105.2], [half, -62, 0, 522.5], [-half, 50, 0, -85.7],
                   [half, -50, 0, 477.5], [-half, 0, 0, 111.0], [half, 0, 0, 108.0]])  # fmt: skip
    frames = link_frames(DEFAULT_URDF_PATH, dh)
    assert len(frames) == 8
    with pytest.raises(ValueError, match="disagree"):
        link_frames(DEFAULT_URDF_PATH, dh * np.array([1, 1, 1, 0.9]))
    # One small triangle per link, written the way ``convert`` writes it.
    arrays = {"links": np.int64(8)}
    for i in range(8):
        arrays[f"vertices_{i}"] = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
        arrays[f"triangles_{i}"] = np.array([[0, 1, 2]], dtype=np.int32)
    path = tmp_path / "arm_mesh.npz"
    np.savez(path, **arrays)
    assert len(load_link_meshes(path)) == 8
    arm = GhostArm(DEFAULT_URDF_PATH, mesh_path=path)
    q = np.array([5.45, 0.06, 3.14, 2.5, -0.12, 0.32, 1.6])
    centres, flange = arm.joints(q)
    shapes, gripper = arm.items_at(q, q, flange[:3, 3], 0.5)
    assert shapes.lit and len(shapes.triangles) == 8
    # Each link's triangle sits at its joint origin.
    assert np.min(np.linalg.norm(shapes.vertices - centres[2], axis=1)) < 1e-6
    assert gripper.lit


def test_lit_meshes_shade_faces_by_their_tilt():
    from rio.xr.geometry import layer_vertices
    from rio.xr.scene import Layer, Mesh

    flat = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([[0, 1, 2]]), (1, 1, 1, 1), lit=True)
    wall = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1.0]]), np.array([[0, 1, 2]]), (1, 1, 1, 1), lit=True)
    _, triangles = layer_vertices([Layer("a", "a", (flat, wall))], np.eye(4))
    assert triangles[0, 3] > triangles[3, 3]  # the face looking up at the light is brighter
    assert Layer.from_json(Layer("a", "a", (flat,)).to_json()).items[0].lit


def test_the_gripper_model_closes_the_fingers_toward_the_middle(tmp_path):
    from rio.xr.ghost import GripperModel

    triangle = np.array([[0, 1, 2]], dtype=np.int32)

    def part(centre):
        return (np.asarray(centre) + np.array([[0, 0, 0], [0.001, 0, 0], [0, 0, 0.001]])).astype(np.float32)

    arrays = {"static_vertices": part([0, 0, 0.05]), "static_triangles": triangle, "stroke_m": np.array(0.095)}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        pivot, joint = np.array([sign * 0.0165, 0, 0.111]), np.array([sign * 0.0545, 0, 0.1485])
        arrays |= {
            f"{side}_tip_vertices": part(joint + np.array([sign * 0.005, 0, 0.02])),
            f"{side}_tip_triangles": triangle,
            f"{side}_turn0_vertices": part(joint),
            f"{side}_turn0_triangles": triangle,
            f"{side}_turn0_pivot": pivot,
            f"{side}_turns": np.array(1),
            f"{side}_pairs": np.array(0),
            f"{side}_pivot": pivot,
            f"{side}_joint": joint,
        }
    path = tmp_path / "gripper.npz"
    np.savez(path, **arrays)
    model = GripperModel(path)
    opened, closed = model.parts_at(1.0), model.parts_at(0.0)
    assert np.allclose(opened[0][0], closed[0][0])  # the housing stays
    left_tip_open, left_tip_closed = opened[2][0], closed[2][0]
    # Closed: the tip moved half the stroke toward the middle, and forward along z.
    assert np.allclose(left_tip_closed[:, 0] - left_tip_open[:, 0], 0.0475, atol=1e-6)
    assert np.all(left_tip_closed[:, 2] > left_tip_open[:, 2])
    # The link turned about its pivot and its end now sits where the tip group's joint went.
    assert np.allclose(closed[1][0][0], np.array([-0.0545 + 0.0475, 0, closed[1][0][0][2]]), atol=1e-6)
