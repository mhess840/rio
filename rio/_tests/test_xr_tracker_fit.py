# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot-to-headset fit from a controller held in the gripper. No headset, no robot."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rio.xr.tracker_fit import fit_tracker, pose6_matrix

pytestmark = pytest.mark.unit

X_TRUE = pose6_matrix(np.array([0.4, -1.1, 0.7, 0.3, -1.2, 0.5]))
Y_TRUE = pose6_matrix(np.array([0.01, 0.03, 0.06, 1.0, 0.2, -0.4]))


def _poses(count: int, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for _ in range(count):
        xyz = np.array([0.5, 0.0, 0.3]) + rng.uniform(-0.15, 0.15, 3)
        tilt = Rotation.from_euler("xyz", rng.uniform(-0.4, 0.4, 3)) * Rotation.from_rotvec([np.pi, 0.0, 0.0])
        out.append(pose6_matrix(np.concatenate([xyz, tilt.as_rotvec()])))
    return out


def test_exact_poses_recover_base_and_grip():
    rng = np.random.default_rng(1)
    b = _poses(8, rng)
    a = [X_TRUE @ bi @ Y_TRUE for bi in b]
    fit = fit_tracker(b, a)
    np.testing.assert_allclose(fit.xr_from_base, X_TRUE, atol=1e-6)
    np.testing.assert_allclose(fit.tcp_from_controller, Y_TRUE, atol=1e-6)
    assert fit.max_m < 1e-6


def test_noisy_tracking_keeps_the_far_workspace_within_a_few_mm():
    rng = np.random.default_rng(2)
    b = _poses(12, rng)
    a = []
    for bi in b:
        ai = X_TRUE @ bi @ Y_TRUE
        ai[:3, 3] += rng.normal(0.0, 0.002, 3)
        ai[:3, :3] = ai[:3, :3] @ Rotation.from_rotvec(rng.normal(0.0, np.radians(0.5), 3)).as_matrix()
        a.append(ai)
    fit = fit_tracker(b, a)
    far = np.array([0.0, 0.8, 0.0, 1.0])
    error = np.linalg.norm((fit.xr_from_base @ far)[:3] - (X_TRUE @ far)[:3])
    assert error < 0.01
    assert fit.rms_m < 0.005


def test_a_single_rotation_axis_is_refused():
    b = [pose6_matrix(np.array([0.5, 0.0, 0.3 + 0.02 * i, 0.0, 0.0, 0.2 * i])) for i in range(6)]
    a = [X_TRUE @ bi @ Y_TRUE for bi in b]
    with pytest.raises(ValueError, match="one axis"):
        fit_tracker(b, a)


def test_too_few_poses_are_refused():
    with pytest.raises(ValueError, match="at least"):
        fit_tracker([np.eye(4)] * 2, [np.eye(4)] * 2)


def test_hub_keeps_device_poses_and_rejects_a_non_rigid_calibration():
    from rio.xr.hub import XrHub
    from rio.xr.scene import SceneError

    hub = XrHub()
    hub.set_tracked("right", X_TRUE, tracked=True)
    entry = hub.tracked()["devices"]["right"]
    assert entry["tracked"] and entry["age_s"] < 1.0
    np.testing.assert_allclose(entry["matrix"], X_TRUE)
    before = hub.version
    hub.set_calibration({"xr_from_base": X_TRUE.tolist(), "space": "stage"})
    assert hub.version > before and hub.calibration()["space"] == "stage"
    with pytest.raises(SceneError):
        hub.set_calibration({"xr_from_base": (2.0 * np.eye(4)).tolist()})
