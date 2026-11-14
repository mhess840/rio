# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""XR scene, hub, registration, server, and publisher. Localhost only, no headset."""

import http.client
import json
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rio.xr import Layer, Lines, Mesh, Points, SceneError, XrHub, XrPublisher, XrServer, fit_rigid
from rio.xr.auth import read_token, write_token

pytestmark = pytest.mark.unit

TOKEN = "test-token-0123456789"


def _layer(name: str = "dots", count: int = 3) -> Layer:
    points = np.arange(3 * count, dtype=float).reshape(count, 3) / 10
    return Layer(name, "Dots", (Points(points, (1.0, 0.0, 0.0, 1.0), 0.02),))


def _request(server: XrServer, method: str, path: str, body: object = None, headers: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    data = None if body is None else json.dumps(body).encode()
    sent = {"Content-Type": "application/json", **(headers or {})}
    connection.request(method, path, body=data, headers=sent)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


@pytest.fixture
def server():
    hub = XrHub()
    hub.set_anchors({"home_pick": np.array([0.5, 0.1, 0.3])})
    hub.put(_layer())
    running = XrServer(hub, port=0, token=TOKEN)
    running.start()
    yield running
    running.stop()


def test_a_layer_round_trips_through_json():
    layer = Layer(
        "mix",
        "Mix",
        (
            Points(np.zeros((2, 3)), (1.0, 0.0, 0.0, 1.0), 0.02),
            Lines(np.ones((2, 3)), (0.0, 1.0, 0.0, 0.5)),
            Mesh(np.eye(3), np.array([[0, 1, 2]]), (0.0, 0.0, 1.0, 0.3)),
        ),
        visible=False,
    )
    payload = json.loads(json.dumps(layer.to_json()))
    assert Layer.from_json(payload).to_json() == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Bad", "title": "x", "items": []},
        {"name": "ok", "title": "", "items": []},
        {"name": "ok", "title": "x", "items": [{"type": "sphere"}]},
        {"name": "ok", "title": "x", "items": [{"type": "lines", "positions": [0, 0, 0], "color": [1, 1, 1, 1]}]},
        {"name": "ok", "title": "x", "items": [{"type": "points", "positions": [0, 0, float("nan")], "color": [1, 1, 1, 1]}]},
        {"name": "ok", "title": "x", "items": [{"type": "points", "positions": [0, 0, 0], "color": [2, 1, 1, 1]}]},
        {"name": "ok", "title": "x", "items": [{"type": "points", "positions": [0, 0, 0], "color": [1, 1, 1, 1], "size": 0}]},
        {
            "name": "ok",
            "title": "x",
            "items": [{"type": "mesh", "vertices": [0] * 9, "triangles": [0, 1, 3], "color": [1, 1, 1, 1]}],
        },
        {"name": "ok", "title": "x", "items": [{"type": "points", "color": [1, 1, 1, 1]}]},
    ],
)
def test_a_malformed_layer_is_refused(payload):
    with pytest.raises(SceneError):
        Layer.from_json(payload)


def test_the_hub_reports_changes_and_removals_by_version():
    hub = XrHub()
    first = hub.put(_layer("a"))
    second = hub.put(_layer("b"))
    version, layers, removed = hub.changes(first, 0.0)
    assert version == second and [layer["name"] for layer in layers] == ["b"] and removed == []
    hub.remove("a")
    version, layers, removed = hub.changes(second, 0.0)
    assert layers == [] and removed == ["a"]
    assert [layer["name"] for layer in hub.scene()["layers"]] == ["b"]
    assert hub.changes(version, 0.01) == (version, [], [])

    seen: list[int] = []
    waiter = threading.Thread(target=lambda: seen.append(hub.changes(version, 2.0)[0]))
    waiter.start()
    hub.put(_layer("c"))
    waiter.join(3.0)
    assert seen == [version + 1]


def test_registration_recovers_a_rigid_transform_and_rejects_a_line():
    rotation = Rotation.from_euler("xyz", [90, 10, -30], degrees=True)
    rng = np.random.default_rng(0)
    base = rng.uniform(-0.6, 0.6, size=(5, 3))
    xr = rotation.apply(base) + np.array([0.5, -1.2, 0.8])
    matrix, rms = fit_rigid(base, xr)
    assert np.allclose(matrix[:3, :3], rotation.as_matrix(), atol=1e-9)
    assert np.allclose(matrix[:3, 3], [0.5, -1.2, 0.8], atol=1e-9)
    assert rms < 1e-9
    _, noisy = fit_rigid(base, xr + rng.normal(0.0, 0.005, size=xr.shape))
    assert 0.001 < noisy < 0.01
    line = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="line"):
        fit_rigid(line, line)
    with pytest.raises(ValueError, match="at least"):
        fit_rigid(base[:2], xr[:2])


def test_the_token_file_round_trips(tmp_path):
    path = tmp_path / "xr" / "publish_token"
    token = write_token(path)
    assert len(token) >= 32
    assert read_token(path) == token
    assert write_token(path) != token


def test_the_server_serves_the_page_and_scene_and_refuses_foreign_hosts(server):
    response, body = _request(server, "GET", "/")
    assert response.status == 200 and b"rio XR" in body
    assert "default-src 'self'" in response.getheader("Content-Security-Policy")
    response, body = _request(server, "GET", "/xr.js")
    assert response.status == 200 and b"navigator.xr" in body
    response, body = _request(server, "GET", "/scene")
    scene = json.loads(body)
    assert [layer["name"] for layer in scene["layers"]] == ["dots"]
    assert scene["anchors"] == {"home_pick": [0.5, 0.1, 0.3]}
    for path in ("/nope", "/../server.py", "/web/xr.js"):
        assert _request(server, "GET", path)[0].status == 404
    assert _request(server, "GET", "/scene", headers={"Host": "evil.example:8770"})[0].status == 403


def test_publishing_needs_the_token_and_a_valid_layer(server):
    layer = _layer("chunk", 5).to_json()
    assert _request(server, "POST", "/layers", layer)[0].status == 401
    assert _request(server, "POST", "/layers", layer, {"Authorization": "Bearer wrong"})[0].status == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert _request(server, "POST", "/layers", layer, auth)[0].status == 200
    assert "chunk" in [item["name"] for item in server.hub.scene()["layers"]]
    assert _request(server, "POST", "/layers", {"name": "x", "title": "x", "items": "no"}, auth)[0].status == 400
    assert _request(server, "POST", "/layers/remove", {"name": "chunk"}, auth)[0].status == 200
    assert "chunk" not in [item["name"] for item in server.hub.scene()["layers"]]
    closed = XrServer(XrHub(), port=0)
    closed.start()
    try:
        assert _request(closed, "POST", "/layers", layer, auth)[0].status == 401
    finally:
        closed.stop()


def test_register_fits_the_posted_pairs(server):
    base = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    pairs = [[point, [point[0] + 1.0, point[2], -point[1]]] for point in base]
    response, body = _request(server, "POST", "/register", {"pairs": pairs})
    result = json.loads(body)
    assert response.status == 200 and result["rms_m"] < 1e-9
    assert np.allclose(result["xr_from_base"][0], [1.0, 0.0, 0.0, 1.0])
    assert _request(server, "POST", "/register", {"pairs": pairs[:2]})[0].status == 400
    assert _request(server, "POST", "/register", {"pairs": "x"})[0].status == 400


def test_the_event_stream_pushes_a_published_layer(server):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    connection.request("GET", f"/events?since={server.hub.version}")
    response = connection.getresponse()
    assert response.getheader("Content-Type") == "text/event-stream"
    server.hub.put(_layer("live", 4))
    lines = []
    while not any(line.startswith("data:") for line in lines):
        lines.append(response.fp.readline().decode().strip())
    connection.close()
    assert "event: changes" in lines
    change = json.loads(next(line for line in lines if line.startswith("data:"))[5:])
    assert [layer["name"] for layer in change["layers"]] == ["live"]


def test_the_publisher_sends_only_the_newest_layer_and_never_raises(server):
    publisher = XrPublisher(server.url, token=TOKEN)
    for count in range(1, 21):
        publisher.publish(_layer("chunk", count))
    assert publisher.flush(5.0)
    layers = {layer["name"]: layer for layer in server.hub.scene()["layers"]}
    assert len(layers["chunk"]["items"][0]["positions"]) == 20 * 3
    publisher.remove("chunk")
    assert publisher.flush(5.0)
    assert "chunk" not in {layer["name"] for layer in server.hub.scene()["layers"]}
    publisher.close()

    wrong = XrPublisher(server.url, token="wrong")
    wrong.publish(_layer("sneaky"))
    assert wrong.flush(5.0)
    wrong.close()
    assert "sneaky" not in {layer["name"] for layer in server.hub.scene()["layers"]}
    with pytest.raises(ValueError, match="this machine"):
        XrPublisher("http://10.0.0.5:8770", token=TOKEN)
