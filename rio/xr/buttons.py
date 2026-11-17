# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Right Touch controller buttons as a teleop key source, read from the local XR server.

The headset view publishes the right controller's trigger and grip on ``GET /buttons``.
:class:`XrButtons` polls that route and reports the same keyboard-shaped state as the
spare-mouse ``Clicker``: trigger = close (PageDown), grip = open (PageUp). A stale or
missing sample reports no key, so a headset that loses focus never holds a button down.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager, nullcontext

import numpy as np
from loguru import logger

DEFAULT_URL = "http://127.0.0.1:8770"
# HID usages the teleop key mapping reads as gripper close / open (``rio.apps.vive.controls``).
HID_CLOSE = 0x4E
HID_OPEN = 0x4B
ROLLOVER_SLOTS = 6


def buttons_to_keyboard_state(sample: dict | None, *, max_age_s: float) -> dict[str, np.ndarray]:
    """Map one ``/buttons`` controller entry to a keyboard-node state.

    Args:
        sample: The ``right`` entry of ``GET /buttons``, or None when absent.
        max_age_s: Older samples count as released.

    Returns:
        ``alphanumeric_state`` and ``special_state`` rollover arrays, zero for unused slots.
    """
    special = np.zeros(ROLLOVER_SLOTS, dtype=np.int64)
    if sample is not None and float(sample.get("age_s", np.inf)) <= max_age_s:
        codes = [code for button, code in (("trigger", HID_CLOSE), ("squeeze", HID_OPEN)) if sample.get(button)]
        special[: len(codes)] = codes
    return {"alphanumeric_state": np.zeros(ROLLOVER_SLOTS, dtype=np.int64), "special_state": special}


def open_xr_buttons(args) -> AbstractContextManager[XrButtons | None]:
    """``XrButtons`` for ``args.teleop_xr_url``, or a no-op context when unset or empty.

    Args:
        args: Station config; reads ``teleop_xr_url`` if it has one.
    """
    url = getattr(args, "teleop_xr_url", None)
    return XrButtons(url) if url else nullcontext()


class XrButtons:
    """Background poller of the XR server's controller buttons; use as a context manager.

    Args:
        url: Base URL of the XR server.
        freq: Poll rate in Hz.
        max_age_s: A sample older than this, or a failed request, reports no key.
    """

    def __init__(self, url: str = DEFAULT_URL, *, freq: float = 50.0, max_age_s: float = 0.25) -> None:
        self.url = url.rstrip("/") + "/buttons"
        self.period = 1.0 / freq
        self.max_age_s = max_age_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._sample: dict | None = None
        self._received = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reachable: bool | None = None

    def __enter__(self) -> XrButtons:
        self._thread = threading.Thread(target=self._run, name="xr-buttons", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_state(self) -> dict[str, np.ndarray]:
        """Keyboard-shaped state of the right controller; no key when the sample is stale."""
        with self._lock:
            sample, received = self._sample, self._received
        if sample is not None:
            # The server's age plus the time since this poll returned.
            sample = {**sample, "age_s": float(sample.get("age_s", np.inf)) + time.monotonic() - received}
        return buttons_to_keyboard_state(sample, max_age_s=self.max_age_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            sample = self._fetch()
            with self._lock:
                self._sample, self._received = sample, time.monotonic()
            self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))

    def _fetch(self) -> dict | None:
        try:
            with self._opener.open(self.url, timeout=0.5) as response:
                sample = json.loads(response.read()).get("right")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            self._report(False, f"XR controller buttons unavailable at {self.url}: {error}")
            return None
        self._report(True, f"XR controller buttons: right trigger closes, grip opens ({self.url})")
        return sample

    def _report(self, reachable: bool, message: str) -> None:
        if reachable != self._reachable:
            self._reachable = reachable
            (logger.info if reachable else logger.warning)(message)
