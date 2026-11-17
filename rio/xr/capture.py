# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Collect controller touches against the saved-pose anchors and fit the headset."""

from __future__ import annotations

import numpy as np
from loguru import logger

from .registration import fit_rigid

MIN_POINTS = 3


class PoseCapture:
    """Pair touches with anchors in dict order. Three spread-out points enable the fit.

    Args:
        anchors: Name to base-frame point, in the order the operator will visit them.
    """

    def __init__(self, anchors: dict[str, np.ndarray], *, min_spread_m: float = 0.05):
        self._order: list[tuple[str, np.ndarray]] = []
        self.base: list[np.ndarray] = []
        self.xr: list[np.ndarray] = []
        self.transform: np.ndarray | None = None
        self.rms: float | None = None
        self.generation = 0
        self.min_spread_m = min_spread_m
        self.set_anchors(anchors)

    def set_anchors(self, anchors: dict[str, np.ndarray]) -> None:
        """Replace the anchor list. Same names and points keep the touches already stored."""
        order = [(str(name), np.asarray(point, dtype=float).reshape(3).copy()) for name, point in anchors.items()]
        same = len(order) == len(self._order) and all(
            left[0] == right[0] and np.allclose(left[1], right[1]) for left, right in zip(order, self._order, strict=True)
        )
        if same:
            return
        self._order = order
        self.base.clear()
        self.xr.clear()
        self.transform = None
        self.rms = None
        self.generation += 1

    @property
    def next_name(self) -> str | None:
        """Anchor the next trigger press will record, or None when every anchor has a touch."""
        if len(self.base) >= len(self._order):
            return None
        return self._order[len(self.base)][0]

    def capture(self, point: np.ndarray) -> str:
        """Record ``point`` in the headset frame against :attr:`next_name`."""
        name = self.next_name
        if name is None:
            return "every saved pose already has a point"
        _, base = self._order[len(self.base)]
        self.base.append(base.copy())
        self.xr.append(np.asarray(point, dtype=float).reshape(3).copy())
        self.generation += 1
        self._refit()
        fitted = "" if self.rms is None else f", rms {self.rms * 1000:.0f} mm"
        return f"saved {name} ({len(self.base)}/{len(self._order)}){fitted}"

    def undo(self) -> str:
        """Drop the last touch and refit whatever remains."""
        if not self.base:
            return "no point to remove"
        name = self._order[len(self.base) - 1][0]
        self.base.pop()
        self.xr.pop()
        self.generation += 1
        self._refit()
        return f"removed {name} ({len(self.base)}/{len(self._order)})"

    def _refit(self) -> None:
        self.transform = None
        self.rms = None
        if len(self.base) < MIN_POINTS:
            return
        try:
            self.transform, self.rms = fit_rigid(np.stack(self.base), np.stack(self.xr), min_spread_m=self.min_spread_m)
        except ValueError as error:
            logger.info("Registration waiting for a better set of points: {}", error)
