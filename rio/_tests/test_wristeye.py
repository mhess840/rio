# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Eye-in-hand fit on synthetic views of a table marker. No robot, no camera."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rio.apps.kassow.graph.handeye import Intrinsics
from rio.apps.kassow.graph.wristeye import (
    Sample,
    fit_problems,
    fit_wristeye,
    marker_points,
    pose_matrix,
)

pytestmark = pytest.mark.unit

INTR = Intrinsics(640, 480, 615.0, 615.0, 320.0, 240.0)


def views(noise_px: float = 0.0, n: int = 14):
    rng = np.random.default_rng(3)
    tcp_from_camera = np.eye(4)
    tcp_from_camera[:3, :3] = Rotation.from_euler("xyz", [-20.0, 3.0, 90.0], degrees=True).as_matrix()
    tcp_from_camera[:3, 3] = [0.06, -0.02, 0.09]
    base_from_marker = np.eye(4)
    base_from_marker[:3, :3] = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
    base_from_marker[:3, 3] = [-0.5, 0.3, 0.0]
    down = Rotation.from_euler("x", 180.0, degrees=True)
    samples = []
    while len(samples) < n:
        tilt = Rotation.from_euler("xyz", rng.uniform(-15, 15, 3), degrees=True)
        rotvec = (tilt * down).as_rotvec()
        xyz = np.array([-0.5, 0.3, 0.0]) + np.r_[rng.uniform(-0.05, 0.05, 2), 0.0]
        tcp = np.r_[xyz, rotvec]
        # place the TCP so the camera looks at the marker from about 30 cm
        base_from_camera = pose_matrix(tcp) @ tcp_from_camera
        offset = base_from_marker[:3, 3] - base_from_camera[:3, :3] @ np.array([0, 0, 0.30]) - base_from_camera[:3, 3]
        tcp[:3] = tcp[:3] + offset
        camera_from_marker = np.linalg.inv(pose_matrix(tcp) @ tcp_from_camera) @ base_from_marker
        pts, _ = cv2.projectPoints(
            marker_points(0.05),
            cv2.Rodrigues(camera_from_marker[:3, :3])[0],
            camera_from_marker[:3, 3],
            INTR.matrix,
            np.zeros(5),
        )
        pts = pts.reshape(-1, 2) + rng.normal(0, noise_px, (4, 2))
        if camera_from_marker[2, 3] > 0 and np.all((pts > 0) & (pts < [640, 480])):
            samples.append(Sample(f"{len(samples):02d}", [0.0] * 7, tcp.tolist(), pts.tolist()))
    return samples, tcp_from_camera, base_from_marker


def test_fit_recovers_the_camera_mount():
    samples, truth, marker = views(noise_px=0.2)
    fit = fit_wristeye(samples, INTR, marker=3, camera_serial="0")
    got = np.asarray(fit.tcp_from_camera)
    assert np.linalg.norm(got[:3, 3] - truth[:3, 3]) < 0.002
    angle = Rotation.from_matrix(got[:3, :3].T @ truth[:3, :3]).magnitude()
    assert np.degrees(angle) < 0.5
    assert np.linalg.norm(np.asarray(fit.base_from_marker)[:3, 3] - marker[:3, 3]) < 0.002
    assert fit_problems(fit) == []


def test_pixel_to_plane_hits_the_marker_centre():
    samples, _, marker = views()
    fit = fit_wristeye(samples, INTR, marker=3, camera_serial="0")
    s = samples[0]
    u, v = np.mean(np.asarray(s.corners), axis=0)
    hit = fit.pixel_to_plane(s.tcp, float(u), float(v), 0.0)
    assert hit is not None and np.linalg.norm(hit[:2] - marker[:2, 3]) < 0.002


def test_too_few_samples():
    samples, _, _ = views()
    with pytest.raises(ValueError, match="at least"):
        fit_wristeye(samples[:5], INTR, marker=3, camera_serial="0")
