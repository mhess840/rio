# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only preflight of the real Kassow station; run it before robot sessions. Nothing moves.

Starts the arm, gripper, and the two policy cameras (no teleop, recorder, or policy), then checks
the feedback the rollout and the recordings depend on::

    uv run --frozen --extra kassow --group dev pytest -m hardware rio/_tests/test_station_preflight.py -s
"""

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

ARM_FIELDS = ("timestamp", "alarm_code", "joint_q", "joint_qd", "eef_pose")
MAX_AGE_S = 0.25
STARTUP_TIMEOUT_S = 30.0


def _read_only_station():
    from rio.apps.kassow.stations import KassowViveGripperStation

    station = KassowViveGripperStation()
    for node in ("teleop", "teleop_keyboard", "teleop_clicker", "recorder", "visualizer"):
        setattr(station, node, None)
    # The AG95 otherwise drives to its startup opening when the node starts.
    station.gripper_cfg.startup_position = None
    return station


def test_kassow_station_feedback_matches_the_rollout_and_recording_contract():
    from rio_hw import time
    from rio_hw.middleware import ServerManager

    from rio.envs.env import make_env
    from rio.manifest import run_manifest
    from rio.policies.rollout import create_obs
    from rio.schema import require_state_fields

    manifest = run_manifest()
    assert manifest["kord_bridge"] is not None, "_kord_bridge is not installed for this interpreter; rebuild it"
    print(f"rio {manifest['rio']}\nrio_hw {manifest['rio_hw']}\nkord_bridge {manifest['kord_bridge']}")

    station = _read_only_station()
    servers, _clients, env = make_env(station)
    with ServerManager(station.mw, list(servers.values())), env:
        env.set_start_time(time.now())
        deadline = time.now() + STARTUP_TIMEOUT_S
        while True:
            try:
                obs = create_obs(env, max_camera_age_s=MAX_AGE_S)
                arm = env.robot.arm.get_state()
                if time.now() - float(arm.get("timestamp", 0.0)) <= MAX_AGE_S:
                    break
            except (RuntimeError, KeyError, TypeError, ValueError):
                pass
            if time.now() > deadline:
                pytest.fail(f"No fresh arm state and camera frames within {STARTUP_TIMEOUT_S:.0f} s")
            time.sleep(0.2)

        require_state_fields(arm, ARM_FIELDS)
        assert int(arm["alarm_code"]) == 0, f"Controller alarm {arm['alarm_code']} is latched; clear it at the pendant"
        for key, size in (("joint_q", 7), ("joint_qd", 7), ("eef_pose", 6)):
            values = np.asarray(arm[key], dtype=float).reshape(-1)
            assert values.shape == (size,) and np.isfinite(values).all(), f"{key} is not a finite {size}D vector"
        assert np.max(np.abs(np.asarray(arm["joint_qd"], dtype=float))) < 0.05, "The arm must be at rest"

        for key in station.cameras:
            frame = np.asarray(obs[key])
            assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8, f"{key}: {frame.shape} {frame.dtype}"
        assert obs["proprio"].shape == (8,) and np.isfinite(obs["proprio"]).all()
        assert 0.0 <= float(obs["proprio"][-1]) <= 1.0, "Gripper opening outside [0, 1]"
        # Recorded observation/proprio and the policy state must have the same layout.
        recorded = env.get_state().observation.proprio
        assert recorded is not None
        np.testing.assert_allclose(recorded, obs["proprio"], atol=1e-3)
