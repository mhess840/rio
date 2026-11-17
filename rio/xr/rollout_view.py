# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""What a rollout did and thought, drawn into the headset next to the ``vla_plan`` line.

Layers, all in the robot base frame:

- ``vla_trail``: where the fingertips actually went, fading with age.
- ``vla_fan``: the fingertip paths of the previous chunks, fading; shows the policy replanning.
- ``vla_events``: a sphere wherever the gripper command closed (orange) or opened (green).
- ``vla_ghost``: a see-through arm at the end of the newest chunk (needs Pinocchio and a URDF).

The control loop only hands over samples (:meth:`RolloutView.note_step`,
:meth:`RolloutView.note_chunk`); a background thread builds and publishes at ``rate_hz``,
so drawing never costs the loop time. Without a running XR server nothing is sent.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque

import numpy as np
from loguru import logger

from .ghost import GhostArm, ghost_items, merge, sphere
from .plan import GRIPPER_CLOSED_BELOW, gripper_tips
from .publisher import DEFAULT_URL, XrPublisher
from .scene import Layer, Lines, Mesh

TRAIL = (0.55, 1.0, 0.85)
FAN = (0.75, 0.45, 1.0)
GRASP = (1.0, 0.55, 0.1, 0.9)
RELEASE = (0.3, 1.0, 0.45, 0.9)
EVENT_RADIUS_M = 0.018
TRAIL_BUCKETS = 8
TRAIL_SPACING_M = 0.003
TRAIL_POINTS = 1500
FAN_CHUNKS = 14
LAYER_NAMES = ("vla_trail", "vla_fan", "vla_events", "vla_ghost")


def trail_layer(points: np.ndarray) -> Layer | None:
    """A polyline that fades from transparent (oldest) to bright (newest).

    Args:
        points: ``(N, 3)`` fingertip positions, oldest first.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return None
    segments = np.stack([points[:-1], points[1:]], axis=1)
    items = []
    for index, chunk in enumerate(np.array_split(segments, min(TRAIL_BUCKETS, len(segments)))):
        if len(chunk):
            alpha = 0.12 + 0.83 * (index + 1) / TRAIL_BUCKETS
            items.append(Lines(chunk.reshape(-1, 3), (*TRAIL, alpha)))
    return Layer("vla_trail", "Fingertip trail (executed)", tuple(items))


def fan_layer(paths: list[np.ndarray]) -> Layer | None:
    """Earlier chunks' fingertip paths, oldest faintest.

    Args:
        paths: ``(N, 3)`` polylines, oldest first.
    """
    items = []
    for index, raw in enumerate(paths):
        path = np.asarray(raw, dtype=float).reshape(-1, 3)
        if len(path) < 2:
            continue
        alpha = 0.06 + 0.5 * (index + 1) / max(len(paths), 1)
        items.append(Lines(np.stack([path[:-1], path[1:]], axis=1).reshape(-1, 3), (*FAN, alpha)))
    if not items:
        return None
    return Layer("vla_fan", "Policy replanning (earlier chunks)", tuple(items))


def events_layer(events: list[tuple[np.ndarray, bool]]) -> Layer | None:
    """A sphere per gripper command change.

    Args:
        events: ``(fingertip, closed)`` pairs; ``closed`` is True for a grasp.
    """
    items = []
    for closed in (True, False):
        parts = [sphere(point, EVENT_RADIUS_M) for point, is_closed in events if is_closed == closed]
        if parts:
            vertices, triangles = merge(parts)
            items.append(Mesh(vertices, triangles, GRASP if closed else RELEASE))
    if not items:
        return None
    return Layer("vla_events", "Gripper events (orange grasp, green release)", tuple(items))


class RolloutView:
    """Collects rollout samples and publishes the overlay layers from a background thread.

    Args:
        publisher: Where the layers go.
        ghost: Arm model for ``vla_ghost``; None leaves the ghost out.
        rate_hz: Publish rate of the changing layers.
    """

    def __init__(self, publisher: XrPublisher, ghost: GhostArm | None = None, *, rate_hz: float = 10.0) -> None:
        self.publisher = publisher
        self.ghost = ghost
        self.period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._trail: deque[np.ndarray] = deque(maxlen=TRAIL_POINTS)
        self._fan: deque[np.ndarray] = deque(maxlen=FAN_CHUNKS)
        self._events: list[tuple[np.ndarray, bool]] = []
        self._closed: bool | None = None
        self._chunk: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._dirty: set[str] = set()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="xr-rollout-view", daemon=True)
        self._thread.start()

    def reset(self) -> None:
        """Forget the trail, fan, events and ghost, e.g. when a new episode starts."""
        with self._lock:
            self._trail.clear()
            self._fan.clear()
            self._events.clear()
            self._closed = None
            self._chunk = None
            self._dirty.update(LAYER_NAMES)
        self._wake.set()

    def note_step(self, eef_pose: np.ndarray, gripper: float | None) -> None:
        """One control-loop sample. Cheap: copies two small arrays under a lock.

        Args:
            eef_pose: Measured flange pose ``[x, y, z, rx, ry, rz]``.
            gripper: Commanded gripper opening (0 closed, 1 open), or None without a gripper.
        """
        pose = np.asarray(eef_pose, dtype=float).reshape(-1)[:6]
        opening = 1.0 if gripper is None else float(gripper)
        tip = gripper_tips(pose[None], opening=opening)[0]
        closed = gripper is not None and opening < GRIPPER_CLOSED_BELOW
        with self._lock:
            if not self._trail or float(np.linalg.norm(tip - self._trail[-1])) >= TRAIL_SPACING_M:
                self._trail.append(tip)
                self._dirty.add("vla_trail")
            if self._closed is not None and closed != self._closed:
                self._events.append((tip, closed))
                self._dirty.add("vla_events")
            self._closed = closed

    def note_chunk(self, actions: np.ndarray, joint_q: np.ndarray | None, eef_pose: np.ndarray | None) -> None:
        """A newly merged chunk (flange poses with gripper), with the arm state for the ghost IK seed.

        Args:
            actions: ``(N, 7)`` remaining flange poses and gripper.
            joint_q: Current joints, or None.
            eef_pose: Current measured flange pose, or None.
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] < 6 or not len(actions):
            return
        path = gripper_tips(actions)
        with self._lock:
            self._fan.append(path)
            self._dirty.add("vla_fan")
            if joint_q is not None and eef_pose is not None:
                self._chunk = (
                    actions[-1].copy(),
                    np.asarray(joint_q, dtype=float).copy(),
                    np.asarray(eef_pose, dtype=float).copy(),
                )
                self._dirty.add("vla_ghost")
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(self.period)
            self._wake.clear()
            started = time.monotonic()
            try:
                self._publish_dirty()
            except Exception as error:  # a drawing bug must never reach the rollout
                logger.warning("Rollout overlay skipped a frame: {}", error)
            time.sleep(max(0.0, self.period - (time.monotonic() - started)))

    def _publish_dirty(self) -> None:
        with self._lock:
            dirty, self._dirty = self._dirty, set()
            trail = np.array(self._trail) if "vla_trail" in dirty else None
            # The newest chunk is drawn by ``vla_plan``; the fan shows the ones before it.
            fan = list(self._fan)[:-1] if "vla_fan" in dirty else None
            events = list(self._events) if "vla_events" in dirty else None
            chunk = self._chunk if "vla_ghost" in dirty else None
        for name, layer in (
            ("vla_trail", None if trail is None else trail_layer(trail)),
            ("vla_fan", None if fan is None else fan_layer(fan)),
            ("vla_events", None if events is None else events_layer(events)),
        ):
            if name in dirty:
                self._send(name, layer)
        if "vla_ghost" in dirty:
            self._send("vla_ghost", None if chunk is None else self._ghost_layer(*chunk))

    def _send(self, name: str, layer: Layer | None) -> None:
        if layer is None:
            self.publisher.remove(name)
        else:
            self.publisher.publish(layer)

    def _ghost_layer(self, action: np.ndarray, joint_q: np.ndarray, eef_pose: np.ndarray) -> Layer | None:
        if self.ghost is None:
            return None
        # The URDF flange and the controller's eef_pose differ by a small offset; solve in URDF
        # coordinates and shift the drawing back onto the controller frame, like ``goto.move_tcp``.
        _, flange_now = self.ghost.joints(joint_q)
        offset = flange_now[:3, 3] - eef_pose[:3]
        goal = np.concatenate([action[:3] + offset, action[3:6]])
        q = self.ghost.solve(goal, joint_q)
        if q is None:
            return None
        centres, flange = self.ghost.joints(q)
        centres = centres - offset
        flange = flange.copy()
        flange[:3, 3] -= offset
        opening = float(action[6]) if len(action) > 6 else 1.0
        return Layer("vla_ghost", "Ghost arm at the end of the chunk", ghost_items(centres, flange, opening))


class _Shared:
    view: RolloutView | None = None
    tried = False


def rollout_view(urdf_path: str | None = None, url: str = DEFAULT_URL) -> RolloutView | None:
    """The process-wide overlay, created on first use; None when no XR server answers.

    Args:
        urdf_path: Arm URDF for the ghost; None or a missing Pinocchio leaves the ghost out.
        url: XR server.
    """
    if _Shared.tried:
        return _Shared.view
    _Shared.tried = True
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url.rstrip('/')}/calibration", timeout=0.5) as response:
            json.loads(response.read())
        publisher = XrPublisher(url)
    except (OSError, ValueError) as error:
        logger.info("Rollout overlay off (no XR server at {}): {}", url, error)
        return None
    ghost = None
    if urdf_path:
        try:
            ghost = GhostArm(urdf_path)
        except (ImportError, ValueError, RuntimeError) as error:
            logger.info("Ghost arm off: {}", error)
    _Shared.view = RolloutView(publisher, ghost)
    logger.info("Rollout overlay on: trail, replanning fan, gripper events{}", ", ghost arm" if ghost else "")
    return _Shared.view
