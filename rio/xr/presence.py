# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Is someone wearing the headset? Read from the XR server's ``GET /headset``.

Overlays that cost work to build (ghost arms) ask :meth:`HeadsetPresence.worn` and
skip drawing while nobody looks. The answer is cached for ``refresh_s``; a missing
server or an old server without the route counts as not worn.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from .publisher import DEFAULT_URL


class HeadsetPresence:
    """Cached ``worn`` flag of the headset view.

    Args:
        url: XR server origin.
        refresh_s: How long one answer is reused.
        timeout_s: HTTP timeout; the server is on this machine.
    """

    def __init__(self, url: str = DEFAULT_URL, *, refresh_s: float = 1.0, timeout_s: float = 0.2) -> None:
        self.url = url.rstrip("/") + "/headset"
        self.refresh_s = refresh_s
        self.timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._checked = -float("inf")
        self._worn = False

    def worn(self) -> bool:
        """True while the headset view's session is visible or focused."""
        with self._lock:
            if time.monotonic() - self._checked < self.refresh_s:
                return self._worn
            self._checked = time.monotonic()
            self._worn = self._fetch()
            return self._worn

    def _fetch(self) -> bool:
        try:
            with self._opener.open(self.url, timeout=self.timeout_s) as response:
                return bool(json.loads(response.read()).get("worn"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, AttributeError):
            return False
