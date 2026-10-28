"""Shared gripper command and observation encoding for datasets and policies."""

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np


class GripperEncoding(TypedDict):
    gripper_close: float
    gripper_open: float
    gripper_threshold: float | None


@dataclass
class GripperBinarizer:
    """Keep measured-state hysteresis separate from the command decision threshold."""

    closed: float = 0.0
    opened: float = 0.7
    threshold: float | None = None
    state: float | None = field(default=None, init=False)

    def __post_init__(self):
        if self.threshold is None:
            self.threshold = 0.5 * (self.closed + self.opened)
        if not np.isfinite([self.closed, self.opened, self.threshold]).all():
            raise ValueError("Gripper encoding values must be finite")
        if not self.closed < self.threshold < self.opened:
            raise ValueError("Gripper encoding must satisfy closed < threshold < opened")

    def reset(self) -> None:
        """Start a new episode without retaining the previous grip state."""
        self.state = None

    def action(self, values) -> np.ndarray:
        """Snap commands directly at the midpoint without observation hysteresis."""
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Gripper commands must be finite")
        return np.where(values >= self.threshold, self.opened, self.closed).astype(np.float32)

    def update(self, value: float | np.floating) -> float:
        """Encode one measured opening using the same state transitions as the exporter."""
        reading = np.float32(value)
        if not np.isfinite(reading):
            raise ValueError("Gripper feedback must be finite")
        threshold = self.threshold
        if threshold is None:
            raise RuntimeError("Gripper threshold is not initialized")
        span = self.opened - self.closed
        low = threshold - 0.2 * span
        high = threshold + 0.15 * span
        if self.state is None:
            self.state = self.opened if reading >= threshold else self.closed
        if reading <= low:
            self.state = self.closed
        elif reading >= high:
            self.state = self.opened
        return float(self.state)

    def sequence(self, values) -> np.ndarray:
        """Encode one complete episode from a fresh hysteresis state."""
        self.reset()
        return np.asarray([self.update(value) for value in np.asarray(values, dtype=np.float32).reshape(-1)], dtype=np.float32)
