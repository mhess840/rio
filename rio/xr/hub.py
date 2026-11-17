# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The latest layers and anchors, shared by publishers and the page's event streams."""

from __future__ import annotations

import threading
import time

import numpy as np

from .scene import Layer, SceneError

ANCHOR_NAME_MAX_CHARS = 64


class XrHub:
    """Thread-safe scene store. Every change bumps ``version``; event streams wait for it.

    Anchors are named base-frame points the page offers for registration, e.g. the
    TCP of saved poses. They are set once before serving.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._version = 0
        self._layers: dict[str, tuple[int, dict]] = {}
        self._removed: dict[str, int] = {}
        self._anchors: dict[str, list[float]] = {}
        self._tracked: dict[str, tuple[float, list[list[float]], bool]] = {}
        self._buttons: dict[str, tuple[float, dict[str, bool]]] = {}
        self._calibration: dict | None = None
        self.space = ""

    def put(self, layer: Layer) -> int:
        """Add or replace ``layer``. Returns the new version."""
        payload = layer.to_json()
        with self._changed:
            self._version += 1
            self._layers.pop(layer.name, None)
            self._layers[layer.name] = (self._version, payload)
            self._removed.pop(layer.name, None)
            self._changed.notify_all()
            return self._version

    def remove(self, name: str) -> int:
        """Drop a layer if present. Returns the version after the call."""
        with self._changed:
            if self._layers.pop(name, None) is not None:
                self._version += 1
                self._removed[name] = self._version
                self._changed.notify_all()
            return self._version

    def set_anchors(self, anchors: dict[str, np.ndarray]) -> None:
        """Replace the registration anchors.

        Raises:
            SceneError: A name is empty or too long, or a point is not three finite numbers.
        """
        clean: dict[str, list[float]] = {}
        for name, point in anchors.items():
            xyz = np.asarray(point, dtype=float).reshape(-1)
            if not name or len(name) > ANCHOR_NAME_MAX_CHARS or xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise SceneError(f"anchor {name!r} must have a short name and three finite coordinates")
            clean[str(name)] = [round(float(value), 4) for value in xyz]
        with self._changed:
            self._anchors = clean

    def set_tracked(self, name: str, matrix: np.ndarray, *, tracked: bool) -> None:
        """Latest headset pose of a device, e.g. ``right`` controller, in the render space.

        Args:
            name: Device name.
            matrix: 4 x 4 ``xr_from_device``.
            tracked: The runtime reported position and orientation as tracked, not inferred.
        """
        rows = np.asarray(matrix, dtype=float).reshape(4, 4).tolist()
        with self._changed:
            self._tracked[name] = (time.monotonic(), rows, bool(tracked))

    def tracked(self) -> dict:
        """Every device pose with its age in seconds, plus the reference space name."""
        now = time.monotonic()
        with self._changed:
            devices = {
                name: {"age_s": now - stamp, "matrix": rows, "tracked": ok} for name, (stamp, rows, ok) in self._tracked.items()
            }
            return {"space": self.space, "devices": devices}

    def set_buttons(self, name: str, buttons: dict[str, bool]) -> None:
        """Latest button state of a controller, e.g. ``right``: ``{"trigger": True, "squeeze": False}``.

        Args:
            name: Device name.
            buttons: Pressed state per button name.
        """
        clean = {str(button): bool(down) for button, down in buttons.items()}
        with self._changed:
            self._buttons[name] = (time.monotonic(), clean)

    def buttons(self) -> dict:
        """Every controller's pressed buttons with the age of the sample in seconds."""
        now = time.monotonic()
        with self._changed:
            return {name: {"age_s": now - stamp, **pressed} for name, (stamp, pressed) in self._buttons.items()}

    def set_calibration(self, calibration: dict | None) -> None:
        """Replace the robot-to-headset fit; the headset draws with it from the next frame.

        Args:
            calibration: ``{"xr_from_base": 4 x 4, "space": "stage" | "local", ...}`` or None to clear.

        Raises:
            SceneError: The matrix is not a finite 4 x 4 rigid transform.
        """
        if calibration is not None:
            matrix = np.asarray(calibration.get("xr_from_base"), dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise SceneError("xr_from_base must be a finite 4 x 4 matrix")
            if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-4):
                raise SceneError("xr_from_base must be a rigid transform")
            calibration = dict(calibration)
        with self._changed:
            self._calibration = calibration
            self._version += 1
            self._changed.notify_all()

    def calibration(self) -> dict | None:
        with self._changed:
            return None if self._calibration is None else dict(self._calibration)

    @property
    def version(self) -> int:
        with self._changed:
            return self._version

    def scene(self) -> dict:
        """Everything the page needs on load."""
        with self._changed:
            return {
                "version": self._version,
                "layers": [payload for _, payload in self._layers.values()],
                "anchors": dict(self._anchors),
            }

    def changes(self, since: int, timeout_s: float) -> tuple[int, list[dict], list[str]]:
        """Wait up to ``timeout_s`` for a version above ``since``; return it, changed layers, removed names."""
        with self._changed:
            self._changed.wait_for(lambda: self._version > since, timeout=timeout_s)
            layers = [payload for version, payload in self._layers.values() if version > since]
            removed = [name for name, version in self._removed.items() if version > since]
            return self._version, layers, removed
