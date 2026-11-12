"""Live page for a demo: orchestrator state and the latest Jev reading.

The orchestrator writes ``status.json`` and ``crate.jpg`` into a folder. This
module serves that folder on localhost so a browser can poll it. Nothing here
talks to the arm or to Jev.

    uv run python -m rio.apps.kassow.orchestrator --dry-run --board
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crate orchestrator</title>
<style>
  body { margin: 0; font-family: sans-serif; background: #111; color: #f4f4f4; }
  header { padding: 22px 28px 0; }
  main { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 24px; padding: 8px 28px 28px; }
  h1 { font-size: 36px; margin: 0 0 8px; }
  .detail { font-size: 18px; color: #c8c8c8; }
  .card { background: #1c1c1c; padding: 18px 20px; margin-top: 16px; }
  .label { font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; color: #9a9a9a; }
  .value { font-size: 28px; margin-top: 4px; }
  img { width: 100%; background: #000; }
  .muted { color: #9a9a9a; }
  svg { width: 100%; height: auto; }
  .edge { stroke: #6a6a6a; fill: none; }
  .node rect { fill: #1c1c1c; stroke: #6a6a6a; }
  .node text { fill: #f4f4f4; font-size: 15px; font-family: sans-serif; }
  .node.on rect { fill: #f4f4f4; stroke: #f4f4f4; }
  .node.on text { fill: #111; }
  .node.alarm rect { fill: #8f2d2d; stroke: #ff6b6b; }
  .node.alarm text { fill: #fff; }
  #alarm-note { min-height: 1.2em; color: #ff6b6b; margin: 4px 0 0; }
</style>
</head>
<body>
<header>
  <p class="label">Run</p>
  <svg viewBox="0 0 1320 360" role="img" aria-label="Crate run">
    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#6a6a6a"></path>
      </marker>
    </defs>
    <path class="edge" marker-end="url(#arrow)" d="M190,94 H250"></path>
    <path class="edge" marker-end="url(#arrow)" d="M400,94 H460"></path>
    <path class="edge" marker-end="url(#arrow)" d="M640,94 H700"></path>
    <path class="edge" marker-end="url(#arrow)" d="M860,94 H920"></path>
    <path class="edge" marker-end="url(#arrow)" d="M1050,94 H1110"></path>
    <path class="edge" marker-end="url(#arrow)" d="M1180,70 V28 H115 V70"></path>
    <path class="edge" marker-end="url(#arrow)" d="M325,118 V200"></path>
    <path class="edge" marker-end="url(#arrow)" d="M250,224 H210 V110 H250"></path>
    <path class="edge" marker-end="url(#arrow)" d="M400,118 V160 H550 V200"></path>
    <path class="edge" marker-end="url(#arrow)" d="M640,224 H700"></path>
    <path class="edge" marker-end="url(#arrow)" d="M780,248 V310 H200 V110 H250"></path>
    <g class="node" data-node="initial">
      <rect x="40" y="70" width="150" height="48"></rect>
      <text x="115" y="99" text-anchor="middle">Waiting</text>
    </g>
    <g class="node" data-node="jev">
      <rect x="250" y="70" width="150" height="48"></rect>
      <text x="325" y="99" text-anchor="middle">Ask Jev</text>
    </g>
    <g class="node" data-node="space_move">
      <rect x="460" y="70" width="180" height="48"></rect>
      <text x="550" y="99" text-anchor="middle">Move to space</text>
    </g>
    <g class="node" data-node="open_vla">
      <rect x="700" y="70" width="160" height="48"></rect>
      <text x="780" y="99" text-anchor="middle">Open space</text>
    </g>
    <g class="node" data-node="hold_open">
      <rect x="920" y="70" width="130" height="48"></rect>
      <text x="985" y="99" text-anchor="middle">Hold</text>
    </g>
    <g class="node" data-node="clear">
      <rect x="1110" y="70" width="140" height="48"></rect>
      <text x="1180" y="99" text-anchor="middle">Leave</text>
    </g>
    <g class="node" data-node="hold">
      <rect x="250" y="200" width="150" height="48"></rect>
      <text x="325" y="229" text-anchor="middle">Error</text>
    </g>
    <g class="node" data-node="organize_move">
      <rect x="460" y="200" width="180" height="48"></rect>
      <text x="550" y="229" text-anchor="middle">Move to reorganize</text>
    </g>
    <g class="node" data-node="organize_vla">
      <rect x="700" y="200" width="160" height="48"></rect>
      <text x="780" y="229" text-anchor="middle">Organize</text>
    </g>
  </svg>
  <p id="alarm-note"></p>
</header>
<main>
  <section>
    <p class="label">Now</p>
    <h1 id="title">—</h1>
    <p class="detail" id="detail"></p>
    <div class="card">
      <p class="label">Jev</p>
      <p class="value" id="decision">no reading yet</p>
      <p id="reading" class="muted"></p>
    </div>
    <div class="card"><p class="label">Current job</p><p class="value" id="job">—</p></div>
  </section>
  <section>
    <p class="label">Latest Jev camera frame</p>
    <img id="photo" alt="Crate from above">
    <p class="muted" id="updated"></p>
  </section>
</main>
<script>
async function tick() {
  const status = await fetch("/status.json?t=" + Date.now()).then(r => r.json());
  document.getElementById("title").textContent = status.title;
  document.getElementById("detail").textContent = status.detail;
  const reading = status.reading;
  document.getElementById("decision").textContent = reading ? reading.decision : "no reading yet";
  document.getElementById("reading").textContent = reading ? reading.line : "";
  document.getElementById("job").textContent = status.job || "—";
  document.getElementById("updated").textContent = status.updated;
  document.querySelectorAll("[data-node]").forEach(node => {
    const here = node.dataset.node === status.node;
    node.classList.toggle("on", here && !status.alarm);
    node.classList.toggle("alarm", here && !!status.alarm);
  });
  document.getElementById("alarm-note").textContent = status.alarm_note || "";
  if (status.photo) document.getElementById("photo").src = "/crate.jpg?t=" + Date.now();
}
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""

TITLES = {
    "idle": ("Waiting", "j asks Jev, q quits"),
    "ask": ("Ask Jev", "one photo, four questions"),
    "goto": ("Moving to the start pose", "moveJ along the recorded home"),
    "run_vla": ("Running the policy", "s ends the rollout, x aborts"),
    "wait_proceed": ("Holding the compartment open", "place the object, then p moves clear and looks again"),
    "wait_reset": ("Arm alarm", "release the e-stop, then c to continue or q to quit"),
    "hold": ("Held", "j asks Jev again, q quits"),
    "done": ("Finished", ""),
}


def _job_name(snapshot) -> str:
    job = snapshot.job
    return job.experiment if job is not None else ""


def diagram_node(snapshot) -> str:
    """Which box of the run diagram is active."""
    state = snapshot.state.value
    job = _job_name(snapshot)
    after = snapshot.after_goto.value if snapshot.after_goto is not None else ""
    if state == "idle":
        return "initial"
    if state == "ask":
        return "jev"
    if state == "hold":
        return "hold"
    if state == "wait_proceed":
        return "hold_open"
    if state == "done":
        return "initial"
    if state == "wait_reset":
        phase = snapshot.alarm_phase
        if phase == "vla":
            return "organize_vla" if job == "organize" else "open_vla"
        if job == "organize":
            return "organize_move"
        return "clear" if after == "ask" else "space_move"
    if state == "goto":
        if after == "ask":
            return "clear"
        return "organize_move" if job == "organize" else "space_move"
    if state == "run_vla":
        return "organize_vla" if job == "organize" else "open_vla"
    return "initial"


def _reading_line(reading) -> str:
    space = f", Space {reading.space}" if reading.space is not None else ""
    return (
        f"inlay q {reading.organized_q:.2f} to p {reading.organized_p:.2f}, "
        f"view {reading.view_clear_p:.2f}, "
        f"{reading.count} cartons (p {reading.count_p:.2f}), "
        f"any_box {reading.any_box_p:.2f}{space}. {reading.reason}"
    )


@dataclass
class DemoBoard:
    """Write the live snapshot and optionally serve it."""

    root: Path
    port: int = 8765

    def publish(self, orchestrator) -> None:
        """Refresh ``status.json`` and copy the latest crate photo."""
        snapshot = orchestrator.snapshot()
        self.root.mkdir(parents=True, exist_ok=True)
        photo = None
        image = snapshot.photo
        if image.is_file():
            (self.root / "crate.jpg").write_bytes(image.read_bytes())
            photo = "crate.jpg"
        state = snapshot.state.value
        node = diagram_node(snapshot)
        alarm = state == "wait_reset"
        title, detail = TITLES[state]
        if state == "hold" and snapshot.hold_reason:
            detail = snapshot.hold_reason
        job = None
        if snapshot.job is not None and state in ("goto", "run_vla", "wait_proceed"):
            job = snapshot.job.experiment
            if snapshot.job.instruction:
                job = f"{job}: {snapshot.job.instruction}"
        reading = snapshot.reading
        payload = {
            "state": state,
            "node": node,
            "alarm": alarm,
            "alarm_note": f"E-stop. c returns to this step ({node.replace('_', ' ')})." if alarm else "",
            "title": title,
            "detail": detail,
            "job": job,
            "reading": None
            if reading is None
            else {"decision": reading.decision, "line": _reading_line(reading), **asdict(reading)},
            "photo": photo,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        (self.root / "status.json").write_text(json.dumps(payload, default=str))

    def serve(self) -> None:
        """Serve the page on localhost. A taken port does not stop the arm."""
        root = self.root
        port = self.port

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

            def _send(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                    return
                if path == "/status.json":
                    file = root / "status.json"
                    body = file.read_bytes() if file.is_file() else b"{}"
                    self._send(body, "application/json")
                    return
                if path == "/crate.jpg":
                    file = root / "crate.jpg"
                    if file.is_file():
                        self._send(file.read_bytes(), "image/jpeg")
                        return
                self.send_response(404)
                self.end_headers()

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as error:
            from loguru import logger

            logger.warning("Demo page not started on port {}: {}", port, error)
            return
        threading.Thread(target=server.serve_forever, daemon=True).start()
        from loguru import logger

        logger.info("Demo page at http://127.0.0.1:{}/", port)
