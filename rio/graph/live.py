# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Live view of planning and runs: a local web page for demos.

The planner appends to ``<root>/plans/<task>/plan.jsonl`` and writes
``workflow.json`` there; each run writes ``<root>/<YYYYMMDD_HHMMSS>/`` with
``workflow.json``, ``events.jsonl`` and photos. This module only reads those
files and pushes every new line to the browser as Server-Sent Events, so it
never slows the arm, and a past run replays the same way.

    uv run python -m rio.graph.live --root experiments/graph
    # then open http://127.0.0.1:8766  (?plan=<task>&run=<folder> pins a plan or run)

Standard library only and no CDN: the page works offline at the cell.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PAGE_PATH = Path(__file__).resolve().parent / "live_view.html"
RUN_NAME = re.compile(r"\d{8}_\d{6}")
POLL_S = 0.25
HEARTBEAT_S = 10.0
MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".json": "application/json"}


@dataclass
class Tail:
    """New complete lines of one growing file."""

    path: Path
    offset: int = 0
    _rest: str = ""

    def read(self) -> list[dict]:
        """JSON objects appended since the last call; broken lines are skipped."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:  # rewritten
            self.offset, self._rest = 0, ""
        if size == self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        self.offset += len(chunk)
        text = self._rest + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._rest = lines.pop()
        out = []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out


@dataclass
class Feed:
    """Finds plans and runs under ``root``; pinned names win over the newest."""

    root: Path

    def runs(self) -> list[str]:
        """Run folders, oldest first."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and RUN_NAME.fullmatch(p.name))

    def plans(self) -> list[str]:
        """Task folders under ``plans/``, most recently changed first."""
        base = self.root / "plans"
        if not base.is_dir():
            return []
        found = [p for p in base.iterdir() if p.is_dir() and _stamp(p) > 0]
        return [p.name for p in sorted(found, key=_stamp, reverse=True)]

    def run(self, pinned: str | None) -> Path | None:
        names = self.runs()
        name = pinned if pinned in names else (names[-1] if names else None)
        return self.root / name if name else None

    def plan(self, pinned: str | None) -> Path | None:
        names = self.plans()
        name = pinned if pinned in names else (names[0] if names else None)
        return self.root / "plans" / name if name else None

    def file(self, rel: str) -> Path | None:
        """``root/rel`` if it is a file inside ``root``; else None (no path escapes)."""
        path = (self.root / unquote(rel)).resolve()
        root = self.root.resolve()
        if root not in path.parents or not path.is_file() or path.suffix.lower() not in MEDIA:
            return None
        return path


def _stamp(folder: Path) -> float:
    """Latest change of a task folder's plan log or workflow; 0 when it has neither."""
    stamps = [p.stat().st_mtime for p in (folder / "plan.jsonl", folder / "workflow.json") if p.is_file()]
    return max(stamps, default=0.0)


def _load(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class Stream:
    """Messages for one browser: what changed since the last poll."""

    feed: Feed
    plan_pin: str | None = None
    run_pin: str | None = None
    plan_dir: Path | None = None
    run_dir: Path | None = None
    plan_tail: Tail | None = None
    run_tail: Tail | None = None
    workflow_stamp: float = -1.0

    def poll(self) -> list[dict]:
        """Messages: ``plan_reset``, ``plan``, ``plan_workflow``, ``run_reset``, ``run``."""
        out: list[dict] = []
        plan = self.feed.plan(self.plan_pin)
        if plan != self.plan_dir:
            self.plan_dir, self.workflow_stamp = plan, -1.0
            self.plan_tail = Tail(plan / "plan.jsonl") if plan else None
            out.append({"kind": "plan_reset", "name": plan.name if plan else None, "plans": self.feed.plans()})
        if self.plan_tail is not None:
            out += [{"kind": "plan", "event": event} for event in self.plan_tail.read()]
        if self.plan_dir is not None:
            workflow = self.plan_dir / "workflow.json"
            stamp = workflow.stat().st_mtime if workflow.is_file() else 0.0
            if stamp != self.workflow_stamp:
                self.workflow_stamp = stamp
                out.append({"kind": "plan_workflow", "workflow": _load(workflow) if stamp else None, "stamp": stamp})
        run = self.feed.run(self.run_pin)
        if run != self.run_dir:
            self.run_dir = run
            self.run_tail = Tail(run / "events.jsonl") if run else None
            out.append(
                {
                    "kind": "run_reset",
                    "name": run.name if run else None,
                    "runs": self.feed.runs()[-30:],
                    "workflow": _load(run / "workflow.json") if run else None,
                }
            )
        if self.run_tail is not None and self.run_dir is not None:
            for event in self.run_tail.read():
                message = {"kind": "run", "event": event}
                if event.get("event") == "tool_end":
                    photo = self.run_dir / f"{str(event.get('node', '')).replace('/', '__')}.jpg"
                    if photo.is_file():
                        message["photo"] = f"{self.run_dir.name}/{photo.name}"
                out.append(message)
        return out


def page(messages: list[dict] | None = None) -> bytes:
    """The live page; with ``messages`` their state is embedded and shown at once.

    The embedded form is also a standalone file: opened without the server it
    shows the plan and run as they were (photos need the server).
    """
    html = PAGE_PATH.read_text()
    if messages is not None:
        data = json.dumps(messages, default=str).replace("</", "<\\/")
        html = html.replace("<script>", f"<script>window.INITIAL = {data};</script>\n<script>", 1)
    return html.encode()


def export(root: Path, out: Path, *, plan: str | None = None, run: str | None = None) -> Path:
    """Write one plan and run as a standalone HTML file (e.g. to share after a demo)."""
    out.write_bytes(page(Stream(Feed(root), plan_pin=plan, run_pin=run).poll()))
    return out


def serve(root: Path, host: str = "127.0.0.1", port: int = 8766, *, block: bool = True) -> ThreadingHTTPServer:
    """Serve the live view of ``root``.

    Args:
        root: Folder that holds the run folders and ``plans/``.
        host: Interface; ``0.0.0.0`` shows the page on another screen in the network.
        port: TCP port.
        block: Serve in this thread until Ctrl+C; else in a daemon thread.
    """
    feed = Feed(root)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                stream = Stream(feed, plan_pin=query.get("plan"), run_pin=query.get("run"))
                inline = stream.poll() if query.get("inline") else None
                self._send(page(inline), "text/html; charset=utf-8")
            elif url.path == "/events":
                self._events(Stream(feed, plan_pin=query.get("plan"), run_pin=query.get("run")))
            elif url.path.startswith("/file/"):
                path = feed.file(url.path.removeprefix("/file/"))
                if path is None:
                    self._send(b"not found", "text/plain", 404)
                else:
                    self._send(path.read_bytes(), MEDIA[path.suffix.lower()])
            else:
                self._send(b"not found", "text/plain", 404)

        def _events(self, stream: Stream) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            beat = time.monotonic()
            try:
                while True:
                    messages = stream.poll()
                    for message in messages:
                        self.wfile.write(f"data: {json.dumps(message, default=str)}\n\n".encode())
                    if not messages and time.monotonic() - beat > HEARTBEAT_S:
                        self.wfile.write(b": beat\n\n")
                        beat = time.monotonic()
                    self.wfile.flush()
                    time.sleep(POLL_S)
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"Live view of {root} at http://{host}:{port}/", flush=True)
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    """Command line: serve ``--root``, or ``--export`` one plan and run as a file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="folder with run folders and plans/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--export", type=Path, help="write a standalone HTML file instead of serving")
    parser.add_argument("--plan", help="task folder under plans/ (default: newest)")
    parser.add_argument("--run", help="run folder (default: newest)")
    args = parser.parse_args()
    if args.export is not None:
        print(export(args.root, args.export, plan=args.plan, run=args.run))
        return
    serve(args.root, args.host, args.port)


if __name__ == "__main__":
    main()
