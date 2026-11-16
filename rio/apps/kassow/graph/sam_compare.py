# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare SAM 3.1 with Jev on the Jev photos of past graph runs. No arm, no camera.

For every ``locate`` photo saved by a live run (``experiments/graph/<run>/<node>.jpg``)
this asks SAM for several noun phrases of the same object, maps the best box
through the table map like ``locate`` does, and writes ``report.json``, one
overlay per photo and a contact sheet into ``experiments/graph/sam_compare/<time>/``.
Each SAM request costs about $0.0025 (``docs/sam-api.md``).

    uv run python -m rio.apps.kassow.graph.sam_compare
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tyro

from rio.apps.kassow.experiments import experiments_root
from rio.sam import SamClient, SamError

from .table_map import TableMap, table_map_path

PROMPTS = {
    "cylinder": ["blue cylinder", "blue cap", "blue circle", "blue object"],
    "plate": ["red plate", "plate", "orange plate", "red object"],
    "cube": ["red cube", "cube", "red block", "red object"],
}


@dataclass
class Args:
    """Compare SAM with Jev on saved Jev photos."""

    runs: list[str] = field(default_factory=list)
    """Run folders under experiments/graph; empty: every run with a locate photo."""
    threshold: float = 0.5
    """SAM score that counts as found."""
    limit: int = 80
    """Upper bound on SAM requests (cost guard)."""


@dataclass
class Case:
    run: str
    node: str
    photo: Path
    object: str
    kind: str
    jev_outcome: str
    jev_detail: str


def cases(root: Path, runs: list[str]) -> list[Case]:
    """Locate photos of past runs with Jev's outcome for each."""
    out = []
    for folder in sorted(root.glob("2026*")):
        if runs and folder.name not in runs:
            continue
        graph_file, events_file = folder / "graph.json", folder / "events.jsonl"
        if not graph_file.is_file() or not events_file.is_file():
            continue
        nodes = json.loads(graph_file.read_text()).get("nodes", {})
        for line in events_file.read_text().splitlines():
            event = json.loads(line)
            node = nodes.get(event.get("node"), {})
            photo = folder / f"{event.get('node')}.jpg"
            if node.get("kind") != "locate" or not photo.is_file():
                continue
            what = str(node.get("object", ""))
            kind = next((k for k in PROMPTS if k in what), "")
            if kind:
                out.append(Case(folder.name, event["node"], photo, what, kind, event["outcome"], event["detail"]))
    return out


def jev_xy(case: Case) -> tuple[float, float] | None:
    if case.jev_outcome != "found":
        return None
    match = re.match(r"\{.*\}", case.jev_detail)
    if not match:
        return None
    target = json.loads(match.group(0))
    return float(target["x"]), float(target["y"])


def main(args: Args) -> Path:
    root = experiments_root() / "graph"
    out = root / "sam_compare" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    table = TableMap.load(table_map_path())
    client = SamClient()
    found = cases(root, args.runs)
    print(f"{len(found)} photos; up to {sum(len(PROMPTS[c.kind]) for c in found)} SAM requests")
    report, tiles, requests = [], [], 0
    for case in found:
        image = cv2.imread(str(case.photo))
        if image is None:
            print(f"cannot read {case.photo}")
            continue
        jpeg = case.photo.read_bytes()
        row: dict = {
            "run": case.run,
            "node": case.node,
            "object": case.object,
            "jev": {"outcome": case.jev_outcome, "detail": case.jev_detail[:120], "xy": jev_xy(case)},
            "sam": {},
        }
        for i, prompt in enumerate(PROMPTS[case.kind]):
            if requests >= args.limit:
                break
            requests += 1
            try:
                reply = client.segment(jpeg, prompt)
            except SamError as error:
                row["sam"][prompt] = {"error": str(error)[:200]}
                continue
            best = reply.best()
            entry: dict = {"n": len(reply.instances), "elapsed_s": round(reply.elapsed_s, 2)}
            if best is not None:
                x, y = table.to_xy(*best.centre)
                entry |= {
                    "score": round(best.score or 0.0, 3),
                    "box": [round(v) for v in best.box],
                    "xy": [round(float(x), 4), round(float(y), 4)],
                    "inside_area": bool(table.contains(float(x), float(y))),
                }
                jxy = row["jev"]["xy"]
                if jxy is not None:
                    entry["d_jev_mm"] = round(float(np.hypot(x - jxy[0], y - jxy[1])) * 1000, 1)
                color = [(80, 220, 80), (255, 170, 60), (60, 200, 255), (220, 100, 255)][i]
                for inst in reply.instances:
                    x1, y1, x2, y2 = (int(v) for v in inst.box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{prompt} {inst.score or 0:.2f}", (x1, max(12, y1 - 4)), 0, 0.4, color, 1)
            row["sam"][prompt] = entry
        hits = [p for p, e in row["sam"].items() if e.get("score", 0) >= args.threshold]
        row["sam_found_by"] = hits
        report.append(row)
        label = f"{case.run[-6:]} {case.node}  jev:{case.jev_outcome}  sam:{','.join(hits) or '-'}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(image, label, (6, 16), 0, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{case.run}_{case.node}.jpg"), image)
        tiles.append(cv2.resize(image, (480, 360)))
        print(f"{label}  " + "  ".join(f"{p}:{e.get('score', '-')}" for p, e in row["sam"].items()))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(str(out / "sheet.jpg"), sheet)
    print(f"{requests} SAM requests, about ${requests * 0.0025:.2f}; report: {out}")
    return out


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
