# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The latest layers and anchors, shared by publishers and the page's event streams."""

from __future__ import annotations

import threading

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
