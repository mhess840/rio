# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Kassow workspace layers for the headset. No arm, no headset, no pinocchio."""

import http.client
import json
import math
import socket
import threading
import time

import numpy as np
import pytest

from rio.apps.kassow.graph.table_map import TableMap
from rio.apps.kassow.xr.layers import reach_layer, saved_pose_tcps, table_layer, teleop_limits_layer
from rio.apps.kassow.xr.reach import reach_slices
from rio.xr import Layer, Lines, Mesh

pytestmark = pytest.mark.unit


class DiscKin:
    """Reaches every pose within ``radius`` of the base axis; joints are the pose itself."""

    model: object = None

    def __init__(self, radius: float = 0.5):
        self.radius = radius
        self.calls = 0

    def fk(self, q):
        return np.asarray(q, dtype=float)[:6]

    def ik(self, seed, goal):
        self.calls += 1
        if math.hypot(goal[0], goal[1]) <= self.radius:
            return np.concatenate([goal, [0.0]])
        return np.asarray(seed, dtype=float)


def test_saved_pose_tcps_include_the_packaged_homes():
    tcps = saved_pose_tcps()
    assert "home_organize" in tcps
    assert all(point.shape == (3,) for point in tcps.values())


def test_teleop_limits_lie_on_their_shells_above_the_floor():
    layer = teleop_limits_layer(0.25, 1.40, 0.05)
    assert layer.visible is False
    for item in layer.items:
        points = item.positions
        assert np.all(points[:, 2] >= 0.05 - 1e-9)
        radii = np.linalg.norm(points, axis=1)
        assert np.allclose(radii, radii[0], atol=1e-9)
        assert radii[0] == pytest.approx(1.40) or radii[0] == pytest.approx(0.25)
    assert Layer.from_json(json.loads(json.dumps(layer.to_json()))).name == "teleop_limits"


def test_the_table_layer_lies_on_the_grasp_plane():
    table = TableMap(
        camera="jev",
        serial="1",
        homography=np.eye(3),
        plane=np.array([0.01, 0.0, 0.2]),
        rotvec=np.array([np.pi, 0.0, 0.0]),
        hull=np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.3], [0.0, 0.3]]),
        seeds_xy=np.zeros((1, 2)),
        seeds_q=np.zeros((1, 7)),
    )
    mesh, edge = table_layer(table).items
    assert isinstance(mesh, Mesh) and isinstance(edge, Lines)
    assert np.allclose(mesh.vertices[:, 2], 0.2 + 0.01 * mesh.vertices[:, 0])
    assert len(edge.positions) == 8


def test_reach_fills_the_reachable_disc_and_reuses_its_cache(tmp_path):
    kin = DiscKin(0.5)
    kwargs = {"seed_xy": np.array([0.2, 0.0]), "seed_q": np.zeros(7), "step_m": 0.1, "max_radius": 1.0}
    slices = reach_slices(kin, (0.1, 0.3), np.zeros(3), cache_dir=tmp_path, model_id="disc", **kwargs)
    expected = {
        (round(0.2 + 0.1 * i, 3), round(0.1 * j, 3))
        for i in range(-8, 9)
        for j in range(-8, 9)
        if math.hypot(0.2 + 0.1 * i, 0.1 * j) <= 0.5
    }
    for z, points in slices.items():
        assert {(round(x, 3), round(y, 3)) for x, y, _ in points} == expected
        assert np.allclose(points[:, 2], z)
    calls = kin.calls
    again = reach_slices(kin, (0.1, 0.3), np.zeros(3), cache_dir=tmp_path, model_id="disc", **kwargs)
    assert kin.calls == calls
    assert all(np.allclose(again[z], slices[z]) for z in slices)
    layer = reach_layer(slices, 0.1)
    assert len(layer.items) == 2
    assert reach_slices(DiscKin(0.1), (0.1,), np.zeros(3), **kwargs)[0.1].shape == (0, 3)


def test_the_cli_serves_the_workspace_layers_until_stopped(tmp_path, monkeypatch):
    from rio.apps.kassow.xr.cli import Args, main

    monkeypatch.setenv("RIO_EXPERIMENTS_DIR", str(tmp_path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stop = threading.Event()
    runner = threading.Thread(target=main, args=(Args(port=port, reach=False), stop), daemon=True)
    runner.start()
    scene = None
    deadline = time.monotonic() + 10
    while scene is None and time.monotonic() < deadline:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/scene")
            scene = json.loads(connection.getresponse().read())
            connection.close()
        except OSError:
            time.sleep(0.1)
    stop.set()
    runner.join(5)
    assert scene is not None
    assert [layer["name"] for layer in scene["layers"]] == ["base", "poses", "teleop_limits"]
    assert "home_pick" in scene["anchors"]
    assert (tmp_path / "xr" / "publish_token").is_file()
