# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Send layers to a running XR server from any process without blocking the caller.

A control loop calls ``publish`` and returns at once. A background thread posts
the newest layer per name; older unsent versions of the same layer are dropped,
so a slow headset never backs up the loop.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

from .auth import read_token, token_path
from .scene import Layer

DEFAULT_URL = "http://127.0.0.1:8770"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
ERROR_LOG_EVERY_S = 5.0


class XrPublisher:
    """Background sender to ``POST /layers``.

    Args:
        url: XR server origin; only plain HTTP to this machine is accepted.
        token: Bearer token. Unset: read from ``token_file``.
        token_file: Token written by the server. Unset: the workspace default.
        timeout_s: HTTP timeout per request.

    Raises:
        ValueError: ``url`` is not a loopback HTTP origin.
        FileNotFoundError: No token given and no server has written one.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        token: str | None = None,
        token_file: Path | None = None,
        timeout_s: float = 0.5,
    ):
        parts = urlsplit(url)
        if parts.scheme != "http" or parts.hostname not in LOOPBACK:
            raise ValueError(f"XR server URL must be http on this machine, got {url!r}")
        self.url = url.rstrip("/")
        self.token = token if token is not None else read_token(token_file or token_path())
        self.timeout_s = timeout_s
        # A corporate HTTP_PROXY would otherwise receive the token and the layers.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._pending: dict[str, dict | None] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._closed = threading.Event()
        self._last_error = 0.0
        self._thread = threading.Thread(target=self._run, name="xr-publisher", daemon=True)
        self._thread.start()

    def publish(self, layer: Layer) -> None:
        """Queue ``layer``; replaces any unsent layer of the same name."""
        self._queue(layer.name, layer.to_json())

    def remove(self, name: str) -> None:
        """Queue the removal of a layer."""
        self._queue(name, None)

    def flush(self, timeout_s: float = 2.0) -> bool:
        """Wait until everything queued has been sent or dropped. False on timeout."""
        return self._idle.wait(timeout_s)

    def close(self, timeout_s: float = 2.0) -> None:
        """Send what is queued, then stop the thread."""
        self._closed.set()
        self._wake.set()
        self._thread.join(timeout_s)

    def _queue(self, name: str, payload: dict | None) -> None:
        with self._lock:
            self._pending[name] = payload
            self._idle.clear()
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                batch, self._pending = self._pending, {}
            for name, payload in batch.items():
                if payload is None:
                    self._post("/layers/remove", {"name": name})
                else:
                    self._post("/layers", payload)
            with self._lock:
                if not self._pending:
                    self._idle.set()
                    if self._closed.is_set():
                        return

    def _post(self, path: str, payload: dict) -> None:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            now = time.monotonic()
            if now - self._last_error >= ERROR_LOG_EVERY_S:
                self._last_error = now
                logger.warning("XR server at {} did not take {}: {}", self.url, path, error)
