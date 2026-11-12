# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Operator inputs of a teleop session: held keys, live motion scale, and recorder keys."""

import numpy as np
from loguru import logger
from rio_hw.interfaces.vive_retarget import ClutchRetargeter

from rio.apps.terminal import term

# Matches rio.envs.poll Logitech presenter mapping.
_PRESENTER_CLOSE = frozenset({0x4E, 0x4F, 0xFF56, 0xFF53, 109, 106})  # PageDown, Right
_PRESENTER_OPEN = frozenset({0x4B, 0x50, 0xFF55, 0xFF51, 104, 105})  # PageUp, Left
# Clicker mouse side buttons, plus raw evdev BTN_SIDE / BTN_EXTRA / BACK / FORWARD.
_SCALE_DOWN = frozenset({0x224, 0x113, 0x116})
_SCALE_UP = frozenset({0x225, 0x114, 0x115})


def held_keys(keyboard_state) -> set[str]:
    """Characters currently held down, from the keyboard node's rollover slots."""
    keys = {chr(k) for k in keyboard_state["alphanumeric_state"] if k != 0}
    special = {int(code) for code in keyboard_state["special_state"] if code != 0}
    if 0x2C in special:
        keys.add(" ")
    if special & _PRESENTER_CLOSE:
        keys.add("pagedown")
    if special & _PRESENTER_OPEN:
        keys.add("pageup")
    if special & _SCALE_DOWN:
        keys.add("scale_down")
    if special & _SCALE_UP:
        keys.add("scale_up")
    return keys


def merge_held_keys(*keyboard_states) -> set[str]:
    """Union key sets from one or more keyboard nodes."""
    keys: set[str] = set()
    for state in keyboard_states:
        if state is not None:
            keys |= held_keys(state)
    return keys


def step_motion_scale(current: float, *, up: bool, step: float, min_scale: float, max_scale: float) -> float:
    """Multiply or divide a motion scale by ``step``, then clamp."""
    next_value = current * step if up else current / step
    return float(min(max_scale, max(min_scale, next_value)))


def sync_scaled_speeds(
    retargeter: ClutchRetargeter,
    *,
    base_max_pos_speed: float,
    base_max_rot_speed: float,
    arm=None,
    base_max_motor_speed: float | None = None,
) -> None:
    """Set speed ceilings to the unity-scale bases times the current motion scale."""
    retargeter.max_pos_speed = base_max_pos_speed * retargeter.pos_scale
    retargeter.max_rot_speed = base_max_rot_speed * retargeter.rot_scale
    if arm is None:
        return
    if hasattr(arm, "max_pos_speed"):
        arm.max_pos_speed = retargeter.max_pos_speed
    if hasattr(arm, "max_rot_speed"):
        arm.max_rot_speed = retargeter.max_rot_speed
    if base_max_motor_speed is not None and hasattr(arm, "max_motor_speed"):
        arm.max_motor_speed = base_max_motor_speed * retargeter.pos_scale


def apply_live_scale(
    retargeter: ClutchRetargeter,
    *,
    up: bool,
    step: float,
    min_scale: float,
    max_scale: float,
    tracker_pose: np.ndarray | None,
    pose_valid: bool,
    base_max_pos_speed: float | None = None,
    base_max_rot_speed: float | None = None,
    arm=None,
    base_max_motor_speed: float | None = None,
) -> tuple[float, float]:
    """Step translation and rotation scale by the same relative factor.

    Rebases the clutch origin when engaged so the TCP does not jump. Both
    scales share one step so a 20% finer translation is also 20% finer wrist
    motion. Speed ceilings, if bases are given, track the same factor.
    """
    new_pos = step_motion_scale(retargeter.pos_scale, up=up, step=step, min_scale=min_scale, max_scale=max_scale)
    new_rot = step_motion_scale(retargeter.rot_scale, up=up, step=step, min_scale=min_scale, max_scale=max_scale)
    changed = new_pos != retargeter.pos_scale or new_rot != retargeter.rot_scale
    retargeter.pos_scale = new_pos
    retargeter.rot_scale = new_rot
    if changed:
        if base_max_pos_speed is not None and base_max_rot_speed is not None:
            sync_scaled_speeds(
                retargeter,
                base_max_pos_speed=base_max_pos_speed,
                base_max_rot_speed=base_max_rot_speed,
                arm=arm,
                base_max_motor_speed=base_max_motor_speed,
            )
        if retargeter.engaged and pose_valid and tracker_pose is not None:
            retargeter.rebase(tracker_pose)
    return new_pos, new_rot


def recorder_ready_for_new(recorder) -> bool:
    """True when the previous file is closed (encode may still be running)."""
    if recorder is None:
        return False
    getter = getattr(recorder, "get_state", None)
    if getter is None:
        return True
    status = getter()
    if not isinstance(status, dict):
        return True
    return bool(status.get("is_closed", False))


def handle_recorder_start(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Start a trajectory on 'n' so this tick is the first recorded frame."""
    if not env.recorder or recording:
        return recording
    if "n" not in keys - previous:
        return recording
    if not recorder_ready_for_new(env.recorder):
        logger.warning("Still recording — press 's' before starting the next trajectory")
        return recording
    env.recorder.new_trajectory(wait=False)
    term("\n ============================================= ")
    logger.info("Started new trajectory recording")
    term("============================================= \n")
    return True


def handle_recorder_save(env, keys: set[str], previous: set[str], recording: bool) -> bool:
    """Save on 's' after this tick has been written, so the file ends with 's'."""
    if "s" not in keys - previous:
        return recording
    if not env.recorder:
        return recording
    if not recording:
        logger.warning("Not recording — press 'n' first, then 's' to save and home")
        return recording
    env.recorder.save(wait=False)
    logger.info("Saved trajectory in background (press 'n' for the next one)")
    term("============================================= \n")
    return False
