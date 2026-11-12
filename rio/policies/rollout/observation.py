# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Policy input built from the live environment."""

import numpy as np
from rio_hw import time


def create_obs(env, *, gripper_state_source: str = "measured", max_camera_age_s: float | None = None) -> dict:
    """Camera images plus 8D ``proprio`` (measured joints + gripper), rejecting stale frames.

    Args:
        env: Started environment.
        gripper_state_source: ``measured`` opening or the last ``commanded`` one.
        max_camera_age_s: Refuse frames older than this; ``None`` skips the check.

    Raises:
        RuntimeError: If a camera frame is stale or the commanded gripper state is unavailable.
        ValueError: If ``gripper_state_source`` is unknown.
    """
    formatted_obs = {}
    state = env.get_state()
    for key in state.observation.cameras:
        camera = state.observation.cameras[key]
        if max_camera_age_s is not None:
            timestamp = (getattr(camera, "meta", None) or {}).get("camera_receive_timestamp")
            if timestamp is None or not np.isfinite(timestamp):
                raise RuntimeError(f"Camera {key} has no valid capture/receive timestamp")
            age = time.now() - float(timestamp)
            if age < -0.01 or age > max_camera_age_s:
                raise RuntimeError(f"Camera {key} is stale (age={age:.3f}s)")
        formatted_obs[key] = camera.rgb
    joints = state.observation.proprio_joints
    gripper = state.observation.gripper_position
    if gripper_state_source == "commanded":
        gripper = getattr(state.observation, "gripper_commanded_position", None)
        if gripper is None:
            raise RuntimeError("The gripper driver does not publish commanded position; update rio-hw or select measured state")
    elif gripper_state_source != "measured":
        raise ValueError("gripper_state_source must be 'measured' or 'commanded'")
    formatted_obs["proprio_joints"] = joints
    formatted_obs["gripper_position"] = gripper
    if joints is not None:
        grip = 0.0 if gripper is None else float(gripper)
        formatted_obs["proprio"] = np.concatenate(
            [np.asarray(joints, dtype=np.float32).reshape(-1), np.asarray([grip], dtype=np.float32)]
        )
    return formatted_obs
