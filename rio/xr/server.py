# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only XR server: the WebXR page, the scene, and its live changes, on localhost only.

The headset reaches it over USB with ``adb reverse tcp:<port> tcp:<port>`` and opens
``http://localhost:<port>``. WebXR accepts localhost as a secure origin, so no TLS
and no open port on the network are needed. No route commands a robot.

Routes:
    GET  /, /xr.js        the page
    GET  /scene           all layers and anchors
    GET  /events          server-sent ``changes`` events after ``?since=<version>``
    POST /register        stateless: point pairs in, ``xr_from_base`` and RMS out
    POST /layers          publish one layer (bearer token)
    POST /layers/remove   drop a layer by name (bearer token)
"""

from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from loguru import logger

from .hub import XrHub
from .registration import fit_rigid
from .scene import Layer, SceneError

WEB_ROOT = Path(__file__).resolve().parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/xr.js": ("xr.js", "text/javascript; charset=utf-8"),
}
MAX_LAYER_BODY = 8 * 1024 * 1024
MAX_SMALL_BODY = 64 * 1024
HEARTBEAT_S = 15.0
CSP = "default-src 'self'; script-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'"


class XrServer:
    """Serve ``hub`` on ``127.0.0.1:port``. Port 0 picks a free one.

    Args:
        hub: Scene store shared with in-process publishers.
        port: TCP port; the headset uses the same number through ``adb reverse``.
        token: Bearer token for ``POST /layers``. Empty: publishing over HTTP is off.
        web_root: Folder with ``index.html`` and ``xr.js``.
    """

    def __init__(self, hub: XrHub, *, port: int = 8770, token: str = "", web_root: Path = WEB_ROOT):
        self.hub = hub
        self.port = port
        self.token = token
        self.web_root = web_root
        self._httpd: ThreadingHTTPServer | None = None
        self._stopping = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        """Bind and serve in a daemon thread.

        Raises:
            OSError: The port is taken.
        """
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        threading.Thread(target=self._httpd.serve_forever, name="xr-server", daemon=True).start()
        logger.info("XR server at {}", self.url)

    def stop(self) -> None:
        self._stopping.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

            def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if content_type.startswith("text/html"):
                    self.send_header("Content-Security-Policy", CSP)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload: object) -> None:
                self._send(code, json.dumps(payload).encode())

            def _host_ok(self) -> bool:
                # Rejects DNS-rebinding pages that reach localhost under a foreign host name.
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
                if host in ("localhost", "127.0.0.1", "::1"):
                    return True
                self._json(403, {"error": "host not allowed"})
                return False

            def _body(self, limit: int) -> object | None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if not 0 < length <= limit:
                    self._json(413 if length > limit else 400, {"error": f"body must be 1 to {limit} bytes"})
                    return None
                try:
                    return json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json(400, {"error": "body is not JSON"})
                    return None

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization") or ""
                given = header.removeprefix("Bearer ").strip()
                if server.token and hmac.compare_digest(given.encode(), server.token.encode()):
                    return True
                self._json(401, {"error": "publishing needs the server token"})
                return False

            def do_GET(self) -> None:
                if not self._host_ok():
                    return
                parts = urlsplit(self.path)
                if parts.path in STATIC:
                    name, content_type = STATIC[parts.path]
                    self._send(200, (server.web_root / name).read_bytes(), content_type)
                elif parts.path == "/scene":
                    self._json(200, server.hub.scene())
                elif parts.path == "/events":
                    self._events(parts.query)
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                if not self._host_ok():
                    return
                path = urlsplit(self.path).path
                if path == "/register":
                    self._register()
                elif path in ("/layers", "/layers/remove"):
                    if not self._authorized():
                        return
                    body = self._body(MAX_LAYER_BODY if path == "/layers" else MAX_SMALL_BODY)
                    if body is None:
                        return
                    try:
                        if path == "/layers":
                            version = server.hub.put(Layer.from_json(body))
                        else:
                            version = server.hub.remove(str(body.get("name", "")) if isinstance(body, dict) else "")
                    except SceneError as error:
                        self._json(400, {"error": str(error)})
                        return
                    self._json(200, {"version": version})
                else:
                    self._json(404, {"error": "not found"})

            def _register(self) -> None:
                body = self._body(MAX_SMALL_BODY)
                if body is None:
                    return
                try:
                    pairs = body.get("pairs") if isinstance(body, dict) else None
                    if not isinstance(pairs, list):
                        raise ValueError("pairs must be a list of [base_xyz, xr_xyz]")
                    base = np.asarray([pair[0] for pair in pairs], dtype=float)
                    xr = np.asarray([pair[1] for pair in pairs], dtype=float)
                    matrix, rms = fit_rigid(base, xr)
                except (ValueError, TypeError, IndexError) as error:
                    self._json(400, {"error": str(error)})
                    return
                self._json(200, {"xr_from_base": matrix.tolist(), "rms_m": rms})

            def _events(self, query: str) -> None:
                last = self.headers.get("Last-Event-ID") or (parse_qs(query).get("since") or ["0"])[0]
                try:
                    since = max(0, int(last))
                except ValueError:
                    since = 0
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not server._stopping.is_set():
                        version, layers, removed = server.hub.changes(since, HEARTBEAT_S)
                        if version == since:
                            self.wfile.write(b": ping\n\n")
                        else:
                            data = json.dumps({"version": version, "layers": layers, "removed": removed})
                            self.wfile.write(f"id: {version}\nevent: changes\ndata: {data}\n\n".encode())
                            since = version
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        return Handler
