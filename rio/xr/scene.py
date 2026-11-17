# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene layers for the XR view: named groups of points, line segments, and meshes.

Coordinates are metres in the robot base frame. The page maps them into the
headset with the registration. The wire format is JSON with flat coordinate
lists, so the browser can hand them to WebGL as they are. A new primitive is a
new ``type``; the page skips types it does not know.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

RGBA = tuple[float, float, float, float]
LAYER_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
MAX_ITEMS = 64
MAX_VERTICES = 200_000
TITLE_MAX_CHARS = 80
ROUND_M = 4


class SceneError(ValueError):
    """A layer is malformed or too large."""


def _vertices(values: object, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 3) if array.size % 3 == 0 else array
    if array.ndim != 2 or array.shape[1] != 3:
        raise SceneError(f"{what} must be N x 3 coordinates")
    if not np.all(np.isfinite(array)):
        raise SceneError(f"{what} must be finite")
    return array


def _color(values: object) -> RGBA:
    color = tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1))
    if len(color) != 4 or not all(0.0 <= value <= 1.0 for value in color):
        raise SceneError("color must be four values from 0 to 1")
    return color[0], color[1], color[2], color[3]


def _flat(array: np.ndarray) -> list[float]:
    return np.round(array, ROUND_M).reshape(-1).tolist()


@dataclass(frozen=True)
class Points:
    """Round dots of ``size_m`` diameter."""

    positions: np.ndarray
    color: RGBA
    size_m: float = 0.01

    def to_json(self) -> dict:
        return {"type": "points", "positions": _flat(self.positions), "color": list(self.color), "size": self.size_m}


@dataclass(frozen=True)
class Lines:
    """Independent segments: rows 0-1, 2-3, ... of ``positions``."""

    positions: np.ndarray
    color: RGBA

    def to_json(self) -> dict:
        return {"type": "lines", "positions": _flat(self.positions), "color": list(self.color)}


@dataclass(frozen=True)
class Mesh:
    """Triangles over ``vertices``; transparent when the color alpha is below 1."""

    vertices: np.ndarray
    triangles: np.ndarray
    color: RGBA

    def to_json(self) -> dict:
        return {
            "type": "mesh",
            "vertices": _flat(self.vertices),
            "triangles": np.asarray(self.triangles, dtype=int).reshape(-1).tolist(),
            "color": list(self.color),
        }


Item = Points | Lines | Mesh


@dataclass(frozen=True)
class Layer:
    """One toggle in the view. ``name`` is the key; publishing the same name replaces the layer."""

    name: str
    title: str
    items: tuple[Item, ...] = field(default_factory=tuple)
    visible: bool = True

    def __post_init__(self) -> None:
        if not LAYER_NAME.fullmatch(self.name):
            raise SceneError(f"layer name must match {LAYER_NAME.pattern}, got {self.name!r}")
        if not self.title or len(self.title) > TITLE_MAX_CHARS:
            raise SceneError(f"layer title must have 1 to {TITLE_MAX_CHARS} characters")
        if len(self.items) > MAX_ITEMS:
            raise SceneError(f"a layer holds at most {MAX_ITEMS} items")
        count = sum(len(item.vertices if isinstance(item, Mesh) else item.positions) for item in self.items)
        if count > MAX_VERTICES:
            raise SceneError(f"a layer holds at most {MAX_VERTICES} vertices, got {count}")

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "visible": self.visible,
            "items": [item.to_json() for item in self.items],
        }

    @classmethod
    def from_json(cls, payload: object) -> Layer:
        """Validated layer from its wire format.

        Raises:
            SceneError: A field is missing, of the wrong shape, out of range, or too large.
        """
        if not isinstance(payload, dict):
            raise SceneError("a layer must be a JSON object")
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or len(raw_items) > MAX_ITEMS:
            raise SceneError(f"items must be a list of at most {MAX_ITEMS}")
        items: list[Item] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise SceneError("an item must be a JSON object")
            kind = raw.get("type")
            try:
                if kind == "points":
                    size = float(raw.get("size", 0.01))
                    if not 0.0 < size <= 0.5:
                        raise SceneError("point size must be in (0, 0.5] m")
                    items.append(Points(_vertices(raw["positions"], "positions"), _color(raw["color"]), size))
                elif kind == "lines":
                    positions = _vertices(raw["positions"], "positions")
                    if len(positions) % 2:
                        raise SceneError("lines need an even number of positions")
                    items.append(Lines(positions, _color(raw["color"])))
                elif kind == "mesh":
                    vertices = _vertices(raw["vertices"], "vertices")
                    triangles = np.asarray(raw["triangles"], dtype=int).reshape(-1, 3)
                    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
                        raise SceneError("triangle indices must name vertices")
                    items.append(Mesh(vertices, triangles, _color(raw["color"])))
                else:
                    raise SceneError(f"unknown item type {kind!r}")
            except (KeyError, TypeError) as error:
                raise SceneError(f"{kind} item is incomplete: {error}") from error
        return cls(
            name=str(payload.get("name", "")),
            title=str(payload.get("title", "")),
            items=tuple(items),
            visible=bool(payload.get("visible", True)),
        )


def segments(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Segment pairs along a polyline, for :class:`Lines`."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    starts = points if closed else points[:-1]
    return np.stack([starts, ends], axis=1).reshape(-1, 3)


def circle(center: np.ndarray, radius: float, *, count: int = 64) -> np.ndarray:
    """Points on a horizontal circle; pass to :func:`segments` with ``closed=True``."""
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    center = np.asarray(center, dtype=float).reshape(3)
    return center + np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)], axis=1)


def axes(length: float = 0.2) -> list[Lines]:
    """Red x, green y, blue z from the origin."""
    origin = np.zeros(3)
    colors = ((1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (0.3, 0.5, 1.0, 1.0))
    return [Lines(np.stack([origin, length * np.eye(3)[i]]), color) for i, color in enumerate(colors)]


def fan(polygon: np.ndarray) -> np.ndarray:
    """Triangles of a convex polygon, fanned from its first corner."""
    count = len(np.asarray(polygon).reshape(-1, 3))
    return np.array([[0, i, i + 1] for i in range(1, count - 1)], dtype=int).reshape(-1, 3)
