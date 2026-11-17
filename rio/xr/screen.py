# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The X11 desktop as a picture for the headset, plus where to hang it in the room.

:class:`ScreenGrabber` copies the root window with ``XGetImage`` in a background
thread (its own X connection), paints the mouse cursor from XFixes on top, and keeps
the newest frame bottom row first, as an OpenGL texture upload expects. Linux/X11 only;
Wayland sessions have no root window to read.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import POINTER, c_char_p, c_int, c_short, c_uint, c_ulong, c_ushort, c_void_p
from dataclasses import dataclass

import numpy as np
from loguru import logger

ALL_PLANES = 0xFFFFFFFF
Z_PIXMAP = 2


@dataclass
class ScreenCfg:
    """Desktop panel in the headset.

    Attributes:
        width_m: Panel width; the height follows the screen's aspect ratio.
        distance_m: How far in front of the head the panel is placed.
        drop_m: How far below eye height the panel centre sits.
        fps: Capture rate.
        downscale: Average this many screen pixels per panel pixel along each axis. 2 trades
            resolution for less shimmer when the panel covers fewer headset pixels than the screen has.
    """

    width_m: float = 1.2
    distance_m: float = 1.0
    drop_m: float = 0.1
    fps: float = 15.0
    downscale: int = 1


class _XImage(ctypes.Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("xoffset", c_int),
        ("format", c_int),
        ("data", c_void_p),
        ("byte_order", c_int),
        ("bitmap_unit", c_int),
        ("bitmap_bit_order", c_int),
        ("bitmap_pad", c_int),
        ("depth", c_int),
        ("bytes_per_line", c_int),
        ("bits_per_pixel", c_int),
        ("red_mask", c_ulong),
        ("green_mask", c_ulong),
        ("blue_mask", c_ulong),
        ("obdata", c_void_p),
        ("create_image", c_void_p),
        ("destroy_image", ctypes.CFUNCTYPE(c_int, c_void_p)),
    ]


class _CursorImage(ctypes.Structure):
    _fields_ = [
        ("x", c_short),
        ("y", c_short),
        ("width", c_ushort),
        ("height", c_ushort),
        ("xhot", c_ushort),
        ("yhot", c_ushort),
        ("cursor_serial", c_ulong),
        ("pixels", POINTER(c_ulong)),
        ("atom", c_ulong),
        ("name", c_char_p),
    ]


def _x11():
    lib = ctypes.CDLL("libX11.so.6")
    lib.XOpenDisplay.restype = c_void_p
    lib.XOpenDisplay.argtypes = [c_char_p]
    lib.XCloseDisplay.argtypes = [c_void_p]
    lib.XDefaultRootWindow.restype = c_ulong
    lib.XDefaultRootWindow.argtypes = [c_void_p]
    lib.XDefaultScreen.argtypes = [c_void_p]
    lib.XDisplayWidth.argtypes = [c_void_p, c_int]
    lib.XDisplayHeight.argtypes = [c_void_p, c_int]
    lib.XGetImage.restype = POINTER(_XImage)
    lib.XGetImage.argtypes = [c_void_p, c_ulong, c_int, c_int, c_uint, c_uint, c_ulong, c_int]
    lib.XFree.argtypes = [c_void_p]
    return lib


def _xfixes():
    try:
        lib = ctypes.CDLL("libXfixes.so.3")
    except OSError:
        return None
    lib.XFixesGetCursorImage.restype = POINTER(_CursorImage)
    lib.XFixesGetCursorImage.argtypes = [c_void_p]
    return lib


def screen_size(display: str | None = None) -> tuple[int, int]:
    """Width and height of the default X screen in pixels.

    Raises:
        RuntimeError: No X display can be opened (e.g. Wayland without XWayland, or no ``DISPLAY``).
    """
    x = _x11()
    handle = x.XOpenDisplay(display.encode() if display else None)
    if not handle:
        raise RuntimeError("cannot open the X display; the desktop panel needs an X11 session")
    try:
        number = x.XDefaultScreen(handle)
        return int(x.XDisplayWidth(handle, number)), int(x.XDisplayHeight(handle, number))
    finally:
        x.XCloseDisplay(handle)


def paint_cursor(frame: np.ndarray, argb: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a premultiplied ARGB cursor onto a top-row-first BGRA frame in place.

    Args:
        frame: ``(H, W, 4)`` uint8 BGRA, row 0 at the top.
        argb: ``(h, w)`` uint32 premultiplied ARGB, as XFixes returns it.
        x: Screen column of the cursor image's left edge.
        y: Screen row of the cursor image's top edge.
    """
    height, width = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + argb.shape[1], width), min(y + argb.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return
    patch = argb[y0 - y : y1 - y, x0 - x : x1 - x]
    bgra = patch.astype("<u4").view(np.uint8).reshape(*patch.shape, 4).astype(np.float32)
    alpha = bgra[..., 3:4] / 255.0
    region = frame[y0:y1, x0:x1, :3].astype(np.float32)
    frame[y0:y1, x0:x1, :3] = np.clip(bgra[..., :3] + region * (1.0 - alpha), 0, 255).astype(np.uint8)


def panel_pose(head_position: np.ndarray, head_rotation: np.ndarray, cfg: ScreenCfg) -> tuple[np.ndarray, np.ndarray]:
    """Upright panel ``distance_m`` ahead of the head, turned to face it.

    The reference space is y-up. The quad's visible side is its +Z, so the panel's
    +Z points back at the head along the horizontal view direction.

    Args:
        head_position: Head position in the reference space.
        head_rotation: ``3 x 3`` head orientation; the view looks along its -Z.
        cfg: Distance and drop below eye height.

    Returns:
        Panel centre and its orientation quaternion ``(x, y, z, w)``.
    """
    forward = -np.asarray(head_rotation, dtype=float)[:, 2]
    flat = np.array([forward[0], 0.0, forward[2]])
    norm = float(np.linalg.norm(flat))
    flat = flat / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    centre = np.asarray(head_position, dtype=float) + flat * cfg.distance_m
    centre[1] -= cfg.drop_m
    yaw = float(np.arctan2(-flat[0], -flat[2]))
    return centre, np.array([0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)])


def pose_matrix(centre: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """``4 x 4`` transform from a position and an ``(x, y, z, w)`` unit quaternion."""
    x, y, z, w = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = centre
    return matrix


def matrix_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and ``(x, y, z, w)`` quaternion of a rigid ``4 x 4`` transform."""
    r = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        quat = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        quat[i] = 0.25 * s
        quat[j] = (r[j, i] + r[i, j]) / s
        quat[k] = (r[k, i] + r[i, k]) / s
    quat = np.asarray(quat)
    return np.asarray(matrix, dtype=float)[:3, 3].copy(), quat / np.linalg.norm(quat)


def downscale(frame: np.ndarray, factor: int) -> np.ndarray:
    """Box-average ``factor x factor`` pixel blocks; edges that do not fill a block are dropped."""
    if factor <= 1:
        return frame
    height, width = frame.shape[0] // factor * factor, frame.shape[1] // factor * factor
    blocks = frame[:height, :width].reshape(height // factor, factor, width // factor, factor, -1)
    return (blocks.sum(axis=(1, 3), dtype=np.uint32) // (factor * factor)).astype(np.uint8)


class ScreenGrabber:
    """Newest desktop frame from a background thread; use as a context manager.

    Args:
        fps: Capture rate.
        display: X display name; None uses ``DISPLAY``.
        factor: Box-downscale factor of the published frame (see :func:`downscale`).
    """

    def __init__(self, fps: float = 15.0, display: str | None = None, factor: int = 1) -> None:
        self.period = 1.0 / fps
        self.display = display
        self.factor = max(1, int(factor))
        self.screen = screen_size(display)
        self.size = (self.screen[0] // self.factor, self.screen[1] // self.factor)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ScreenGrabber:
        self._thread = threading.Thread(target=self._run, name="xr-screen", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> tuple[int, np.ndarray | None]:
        """The capture counter and the newest ``(H, W, 4)`` BGRA frame, bottom row first."""
        with self._lock:
            return self.generation, self._frame

    def _run(self) -> None:
        x = _x11()
        fixes = _xfixes()
        handle = x.XOpenDisplay(self.display.encode() if self.display else None)
        if not handle:
            logger.warning("Desktop panel off: cannot open the X display")
            return
        root = x.XDefaultRootWindow(handle)
        width, height = self.screen
        logger.info("Desktop panel: capturing {}x{} at {:.0f} fps, panel {}x{}", width, height, 1.0 / self.period, *self.size)
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = self._grab(x, fixes, handle, root, width, height)
                if frame is not None:
                    frame = np.ascontiguousarray(downscale(frame, self.factor))
                    with self._lock:
                        self._frame = frame
                        self.generation += 1
                self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))
        finally:
            x.XCloseDisplay(handle)

    @staticmethod
    def _grab(x, fixes, handle, root: int, width: int, height: int) -> np.ndarray | None:
        image = x.XGetImage(handle, root, 0, 0, width, height, ALL_PLANES, Z_PIXMAP)
        if not image:
            return None
        try:
            info = image.contents
            if info.bits_per_pixel != 32:
                raise RuntimeError(f"desktop panel needs 32-bit pixels, the X server sends {info.bits_per_pixel}")
            raw = (ctypes.c_uint8 * (info.bytes_per_line * height)).from_address(info.data)
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, info.bytes_per_line // 4, 4)
            frame = rows[:, :width].copy()
        finally:
            image.contents.destroy_image(ctypes.cast(image, c_void_p))
        frame[..., 3] = 255
        if fixes is not None:
            cursor = fixes.XFixesGetCursorImage(handle)
            if cursor:
                try:
                    c = cursor.contents
                    count = int(c.width) * int(c.height)
                    pixels = np.ctypeslib.as_array(c.pixels, shape=(count,)).astype(np.uint32)
                    argb = pixels.reshape(int(c.height), int(c.width))
                    paint_cursor(frame, argb, int(c.x) - int(c.xhot), int(c.y) - int(c.yhot))
                finally:
                    x.XFree(cursor)
        return np.ascontiguousarray(frame[::-1])
