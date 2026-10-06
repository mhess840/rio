# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields


@dataclass
class Camera:
    """Station camera spec; driver kwargs are everything except ``cam_type`` and ``module``.

    Explicit fields (not ``**kwargs``) so tyro can build a CLI from station configs.
    """

    cam_type: str
    module: str = "cameras"
    addr: str | None = None
    serial: str | None = None
    model: str | None = None
    enable_depth: bool | None = None
    enable_color: bool | None = None
    resolution: tuple[int, int] | None = None
    resolution_depth: tuple[int, int] | None = None
    timeout_ms: int | None = None
    timeout: float | None = None
    freq: int | None = None
    bgr: bool | None = None
    hardware_reset: bool | None = None
    advanced_mode_config: str | None = None

    @property
    def cfg(self) -> dict:
        """Keyword arguments forwarded to the camera node constructor."""
        skip = {"cam_type", "module"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip and getattr(self, f.name) is not None}
