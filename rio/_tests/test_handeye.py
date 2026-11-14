# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rio.apps.kassow.graph.handeye import (
    MARKER_SIZES_M,
    TCP,
    HandEye,
    Intrinsics,
    Sample,
    calibration_poses,
    detect_marker,
    fit_best_back_link,
    fit_handeye,
    fit_problems,
    load_handeye,
    marker_centre,
    project,
    save_handeye,
)

INTR = Intrinsics(640, 480, 605.7, 604.6, 320.6, 248.6)
# Camera 0.7 m above the start pose, looking straight down (camera z = -base z).
BASE_FROM_CAMERA = np.eye(4)
BASE_FROM_CAMERA[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
BASE_FROM_CAMERA[:3, 3] = [-0.45, 0.55, 1.15]
GRIPPER_OFFSET = np.array([0.012, -0.020, -0.045])
BACK_OFFSET = np.array([-0.030, 0.010, -0.120])
# Fingers up toward the camera, as on the station.
ANCHOR = np.array([-0.45, 0.55, 0.45, *Rotation.identity().as_rotvec()])


def _link6(pose: np.ndarray, k: int) -> np.ndarray:
    """Wrist link: the TCP without its spin about the tool axis, 10 cm behind it."""
    spin = 20.0 if k % 2 == 0 else -20.0
    r = Rotation.from_rotvec(pose[3:6]) * Rotation.from_euler("z", -spin, degrees=True)
    return np.concatenate([pose[:3] - r.apply([0.0, 0.0, 0.10]), r.as_rotvec()])


def _corners(frame_pose: np.ndarray, offset: np.ndarray, size: float, rng, noise_px: float) -> list:
    half = size / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]]) + offset
    r = Rotation.from_rotvec(frame_pose[3:6]).as_matrix()
    base = (r @ square.T).T + frame_pose[:3]
    camera_from_base = np.linalg.inv(BASE_FROM_CAMERA)
    cam = (camera_from_base[:3, :3] @ base.T).T + camera_from_base[:3, 3]
    return (project(cam, INTR) + rng.normal(0.0, noise_px, (4, 2))).tolist()


def _synthetic(noise_px: float = 0.0, seed: int = 0, back: bool = False, hide_gripper: tuple[int, ...] = ()) -> list:
    """Samples whose corners are exact projections of the markers, plus pixel noise."""
    rng = np.random.default_rng(seed)
    samples = []
    for k, pose in enumerate(calibration_poses(ANCHOR)):
        link6 = _link6(pose, k)
        # A wrong candidate: link6 turned by a pose-dependent angle.
        link5 = np.concatenate(
            [link6[:3], (Rotation.from_rotvec(link6[3:]) * Rotation.from_euler("x", 7 * k, degrees=True)).as_rotvec()]
        )
        markers = {}
        if k not in hide_gripper:
            markers["1"] = _corners(pose, GRIPPER_OFFSET, MARKER_SIZES_M[1], rng, noise_px)
        if back:
            markers["2"] = _corners(link6, BACK_OFFSET, MARKER_SIZES_M[2], rng, noise_px)
        frames = {TCP: pose.tolist(), "link6": link6.tolist(), "link5": link5.tolist()}
        samples.append(Sample(f"{k:02d}", [0.0] * 7, frames, markers))
    return samples


def test_fit_recovers_the_camera_and_the_marker_offset():
    fit = fit_handeye(_synthetic(), INTR, mounts={1: TCP}, camera_serial="x")
    assert np.allclose(fit.transform, BASE_FROM_CAMERA, atol=1e-5)
    assert np.allclose(fit.marker_offsets_m["1"], GRIPPER_OFFSET, atol=1e-5)
    assert fit.stats["loo_max_mm"] < 0.01
    assert fit_problems(fit) == []


def test_fit_with_pixel_noise_stays_within_a_millimetre():
    fit = fit_handeye(_synthetic(noise_px=0.3, seed=1), INTR, mounts={1: TCP}, camera_serial="x")
    assert np.linalg.norm(fit.transform[:3, 3] - BASE_FROM_CAMERA[:3, 3]) < 0.01
    assert np.linalg.norm(np.asarray(fit.marker_offsets_m["1"]) - GRIPPER_OFFSET) < 0.005
    assert fit.stats["loo_rms_mm"] < 1.0


def test_both_markers_fit_together_and_the_back_marker_covers_hidden_gripper_poses():
    samples = _synthetic(noise_px=0.2, seed=3, back=True, hide_gripper=(0, 3, 4, 7, 11, 15))
    fit = fit_handeye(samples, INTR, mounts={1: TCP, 2: "link6"}, camera_serial="x")
    assert fit.stats["n_samples"] == 18
    assert fit.stats["n_observations"] == {"1": 12, "2": 18}
    assert np.linalg.norm(fit.transform[:3, 3] - BASE_FROM_CAMERA[:3, 3]) < 0.01
    assert np.linalg.norm(np.asarray(fit.marker_offsets_m["2"]) - BACK_OFFSET) < 0.005
    assert fit_problems(fit) == []


def test_the_back_link_that_explains_the_pixels_is_chosen():
    samples = _synthetic(noise_px=0.2, seed=4, back=True)
    fit = fit_best_back_link(samples, INTR, gripper_marker=1, back_marker=2, camera_serial="x")
    assert fit.mounts == {"1": TCP, "2": "link6"}


def test_without_back_observations_only_the_gripper_marker_is_fitted():
    fit = fit_best_back_link(_synthetic(), INTR, gripper_marker=1, back_marker=2, camera_serial="x")
    assert fit.mounts == {"1": TCP}


def test_a_bad_sample_shows_in_the_leave_one_out_error():
    samples = _synthetic(noise_px=0.2, seed=2)
    samples[5].markers["1"] = (np.asarray(samples[5].markers["1"]) + np.array([15.0, 0.0])).tolist()
    fit = fit_handeye(samples, INTR, mounts={1: TCP}, camera_serial="x")
    worst = max(fit.stats["samples"], key=lambda name: fit.stats["samples"][name]["1"]["loo_mm"])
    assert worst == samples[5].name
    assert fit_problems(fit)


def test_too_few_samples_are_refused():
    with pytest.raises(ValueError, match="at least"):
        fit_handeye(_synthetic()[:5], INTR, mounts={1: TCP}, camera_serial="x")


def test_a_pixel_ray_meets_the_plane_where_the_point_is():
    fit = fit_handeye(_synthetic(), INTR, mounts={1: TCP}, camera_serial="x")
    point = np.array([-0.40, 0.62, 0.21])
    camera_from_base = np.linalg.inv(fit.transform)
    uv = project((camera_from_base[:3, :3] @ point + camera_from_base[:3, 3]).reshape(1, 3), INTR)[0]
    hit = fit.pixel_to_plane(float(uv[0]), float(uv[1]), z=0.21)
    assert hit is not None and np.allclose(hit, point, atol=1e-6)


def test_calibration_poses_vary_position_and_orientation():
    poses = calibration_poses(ANCHOR)
    assert len(poses) == 18
    xyz = np.array([p[:3] for p in poses])
    assert np.ptp(xyz[:, 0]) > 0.15 and np.ptp(xyz[:, 1]) > 0.15 and np.ptp(xyz[:, 2]) > 0.07
    angles = [np.degrees((Rotation.from_rotvec(ANCHOR[3:]).inv() * Rotation.from_rotvec(p[3:])).magnitude()) for p in poses]
    assert min(angles) > 15 and max(angles) < 30
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    assert np.max(steps) < 0.09  # one grid step per move, also when the height changes


def test_marker_centre_is_the_diagonal_crossing():
    corners = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    assert np.allclose(marker_centre(corners), [2.0, 1.0])
    skewed = np.array([[0.0, 0.0], [10.0, 0.0], [6.0, 4.0], [4.0, 4.0]])
    assert np.allclose(marker_centre(skewed), [5.0, 10.0 / 3.0])


def test_detect_marker_finds_the_printed_id():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[200:260, 300:360] = cv2.aruco.generateImageMarker(dictionary, 1, 60)
    rgb = np.repeat(image[..., None], 3, axis=2)
    corners = detect_marker(rgb, 1)
    assert corners is not None
    assert np.allclose(marker_centre(corners), [329.5, 229.5], atol=1.0)
    assert detect_marker(rgb, 2) is None


def test_handeye_round_trips_through_json(tmp_path):
    fit = fit_handeye(_synthetic(), INTR, mounts={1: TCP}, camera_serial="262722073415")
    path = save_handeye(fit, tmp_path / "jev.json")
    loaded = load_handeye(path)
    assert isinstance(loaded, HandEye) and isinstance(loaded.intrinsics, Intrinsics)
    assert np.allclose(loaded.transform, fit.transform)
    assert json.loads(path.read_text())["camera_serial"] == "262722073415"


def test_table_map_from_handeye_maps_the_object_pixel_to_the_grasp_tcp(tmp_path):
    from pathlib import Path

    from rio.apps.kassow.graph.table_map import Sample as Grasp
    from rio.apps.kassow.graph.table_map import TableMap, table_map_from_handeye

    handeye = HandEye("x", INTR, BASE_FROM_CAMERA.tolist(), {"1": TCP}, {"1": [0.0, 0.0, 0.0]}, {})
    handeye.stats = {"fitted": "t", "n_samples": 12, "loo_rms_mm": 0.5, "loo_max_mm": 0.7, "camera_xyz_m": [0, 0, 0]}
    tool = 0.233
    # Tool axis pointing down, tilted a little; the fingertips (object centre) 1 cm above the table.
    down = Rotation.from_euler("x", 180, degrees=True)
    grasps = []
    for k, (x, y) in enumerate([(-0.55, 0.45), (-0.35, 0.45), (-0.35, 0.65), (-0.55, 0.65), (-0.45, 0.55)]):
        rot = down * Rotation.from_euler("xy", [3 * (k % 2), -2 * (k % 3)], degrees=True)
        tcp = np.array([x, y, 0.01]) - rot.apply([0.0, 0.0, tool])
        grasps.append(Grasp(f"{k:03d}", Path(tmp_path), tcp, rot.as_rotvec(), np.zeros(7)))
    table = table_map_from_handeye(handeye, grasps, tool_offset_m=tool)
    assert abs(table.stats["object_z_m"] - 0.01) < 1e-9
    mean = Rotation.from_rotvec(table.rotvec)
    camera_from_base = np.linalg.inv(BASE_FROM_CAMERA)
    for point in ([-0.50, 0.52, 0.01], [-0.40, 0.60, 0.01]):
        cam = camera_from_base[:3, :3] @ np.asarray(point) + camera_from_base[:3, 3]
        u, v = project(cam.reshape(1, 3), INTR)[0]
        expected_tcp_xy = np.asarray(point[:2]) - mean.apply([0.0, 0.0, tool])[:2]
        assert np.allclose(table.to_xy(u / INTR.width, v / INTR.height), expected_tcp_xy, atol=1e-6)
    path = tmp_path / "table_map.json"
    table.save(path)
    assert TableMap.load(path).stats["source"] == "handeye"
