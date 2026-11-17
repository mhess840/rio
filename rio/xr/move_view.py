# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Where a discrete joint move is going, as a ghost arm in the headset.

:meth:`MoveView.show` draws the goal joints of a ``moveJ`` while the arm drives there;
:meth:`MoveView.clear` removes it once the move is over. :meth:`MoveView.preview`
plays the move with the ghost alone, so a planned sequence can be checked in the
room without moving the arm.

Everything is skipped while nobody wears the headset or no XR server runs; a
missing server is checked again at most every ``RETRY_S`` seconds.
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
from loguru import logger

from .ghost import GhostArm
from .presence import HeadsetPresence
from .publisher import DEFAULT_URL, XrPublisher
from .scene import Layer

LAYER = "move_target"
PREVIEW_LAYER = "move_preview"
TARGET = (1.0, 0.85, 0.35, 0.3)
TARGET_JOINT = (1.0, 0.9, 0.5, 0.5)
PREVIEW = (0.55, 1.0, 0.6, 0.3)
PREVIEW_JOINT = (0.7, 1.0, 0.75, 0.5)
PREVIEW_HZ = 10.0
RETRY_S = 5.0


def smooth_joints(q_from: np.ndarray, q_to: np.ndarray, fraction: float) -> np.ndarray:
    """Joints a synchronous ``moveJ`` passes at ``fraction`` of its time (cosine ease in and out).

    Args:
        q_from: Start joints.
        q_to: Goal joints.
        fraction: 0 at the start, 1 at the goal; clipped.
    """
    s = 0.5 - 0.5 * np.cos(np.pi * float(np.clip(fraction, 0.0, 1.0)))
    q_from = np.asarray(q_from, dtype=float)
    return q_from + (np.asarray(q_to, dtype=float) - q_from) * s


class MoveView:
    """Goal ghost of ``moveJ`` moves and their ghost-only previews.

    Args:
        publisher: Where the layers go.
        ghost: Arm model.
        presence: Drawing waits while nobody wears the headset; None always draws.
    """

    def __init__(self, publisher: XrPublisher, ghost: GhostArm, presence: HeadsetPresence | None = None) -> None:
        self.publisher = publisher
        self.ghost = ghost
        self.presence = presence
        self._shown = False

    def _watched(self) -> bool:
        return self.presence is None or self.presence.worn()

    def show(self, q_goal: np.ndarray, q_now: np.ndarray, eef_now: np.ndarray, opening: float | None = None) -> None:
        """Draw the arm at ``q_goal``.

        Args:
            q_goal: Goal joints of the move.
            q_now: Current joints.
            eef_now: Current measured flange pose.
            opening: Gripper opening at the goal; None draws it open.
        """
        if not self._watched():
            return
        try:
            items = self.ghost.items_at(
                q_goal, q_now, eef_now, 1.0 if opening is None else opening, color=TARGET, joint_color=TARGET_JOINT
            )
        except (ValueError, RuntimeError) as error:
            logger.warning("Move target ghost skipped: {}", error)
            return
        self.publisher.publish(Layer(LAYER, "Goal of the current moveJ", items))
        self._shown = True

    def clear(self) -> None:
        """Remove the goal ghost if one is drawn."""
        if self._shown:
            self.publisher.remove(LAYER)
            self._shown = False

    def preview(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        duration_s: float,
        *,
        q_now: np.ndarray,
        eef_now: np.ndarray,
        opening: float | None = None,
        speed: float = 1.0,
    ) -> None:
        """Play a ``moveJ`` from ``q_from`` to ``q_to`` with the ghost only; blocks for the move's time.

        The ghost stays at ``q_to`` afterwards, so the next preview starts where this one ended.

        Args:
            q_from: Start joints.
            q_to: Goal joints.
            duration_s: Time the real move would take.
            q_now: Current joints of the real arm, for the frame offset.
            eef_now: Current measured flange pose of the real arm.
            opening: Gripper opening drawn during the move; None draws it open.
            speed: Playback speed; 2 plays twice as fast.
        """
        if not self._watched():
            return
        opening = 1.0 if opening is None else opening
        play_s = max(float(duration_s) / max(speed, 1e-3), 0.0)
        frames = max(int(play_s * PREVIEW_HZ), 1)
        started = time.monotonic()
        for index in range(frames + 1):
            q = smooth_joints(q_from, q_to, index / frames)
            items = self.ghost.items_at(q, q_now, eef_now, opening, color=PREVIEW, joint_color=PREVIEW_JOINT)
            self.publisher.publish(Layer(PREVIEW_LAYER, "Dry run (ghost only)", items))
            time.sleep(max(0.0, started + (index + 1) * play_s / frames - time.monotonic()))

    def clear_preview(self) -> None:
        """Remove the dry-run ghost."""
        self.publisher.remove(PREVIEW_LAYER)


class _Shared:
    view: MoveView | None = None
    tried_at = -float("inf")
    urdf: str | None = None


def move_view(urdf_path: str | None, url: str = DEFAULT_URL) -> MoveView | None:
    """The process-wide move ghost, or None without a URDF, Pinocchio, or a running XR server.

    Args:
        urdf_path: Arm URDF.
        url: XR server.
    """
    if not urdf_path:
        return None
    if _Shared.view is not None and _Shared.urdf == str(urdf_path):
        return _Shared.view
    if time.monotonic() - _Shared.tried_at < RETRY_S:
        return None
    _Shared.tried_at = time.monotonic()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url.rstrip('/')}/calibration", timeout=0.2) as response:
            json.loads(response.read())
        publisher = XrPublisher(url)
        ghost = GhostArm(urdf_path)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        logger.debug("Move ghost off: {}", error)
        return None
    _Shared.view = MoveView(publisher, ghost, HeadsetPresence(url))
    _Shared.urdf = str(urdf_path)
    logger.info("Move ghost on: the goal of each moveJ shows in the headset")
    return _Shared.view
