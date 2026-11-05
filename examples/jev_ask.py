"""Ask Jev about the crate on demand: is the inlay organized, how many cartons are in it.

Only the Jev RealSense runs; the arm and the VLAs are not touched. Enter asks,
``q`` quits. Each frame and answer is kept as a probe::

    export TYPESAFE_API_KEY=...
    uv run python -m examples.jev_ask

Saved photos instead of the camera. A directory is scored against hand labels
``name.json`` = ``{"organized": true, "count": 3}`` next to ``name.jpg``::

    uv run python -m examples.jev_ask --image experiments/orchestrator/probes

The local OpenJev helper instead of the hosted API (same questions)::

    uv run python -m examples.jev_ask --jev.backend local --jev.url http://127.0.0.1:3000

Each answer is written as ``name.jev-<backend>.json`` beside its JPEG, so the
two backends can be compared on the same probes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from examples.cfg.kassow_experiments import JEV_SERIAL, experiments_root, jev_camera, open_spaces
from rio.envs.factory import make_cameras
from rio.jev import CrateReading, CrateThresholds, JevCfg, JevClient, JevError, JevReply, ProbeScore, ask_crate, encode_jpeg

JEV_CAMERA = "camera_jev"
FIRST_FRAME_TIMEOUT_S = 10.0
# Auto-exposure needs about 1.5 s after start; the first frames are nearly black.
CAMERA_WARMUP_S = 2.0


def probes_root() -> Path:
    """Default folder for live probes."""
    return experiments_root() / "orchestrator" / "probes"


@contextmanager
def jev_camera_session(serial: str = JEV_SERIAL, mw: str = "Thread") -> Generator:
    """Run only the Jev RealSense node and yield its client once exposure has settled."""
    servers, clients = make_cameras(mw, {JEV_CAMERA: jev_camera(serial)})
    with ServerManager(mw, list(servers.values())), clients[JEV_CAMERA]() as camera:
        fresh_frame(camera, max_age_s=1.0)
        time.sleep(CAMERA_WARMUP_S)
        logger.info("Jev camera {} ready", serial)
        yield camera


def fresh_frame(camera, max_age_s: float, timeout_s: float = FIRST_FRAME_TIMEOUT_S) -> np.ndarray:
    """Newest RGB frame of a RealSense client, no older than ``max_age_s``.

    Raises:
        RuntimeError: If no fresh frame arrives within ``timeout_s``.
    """
    deadline = time.now() + timeout_s
    age = float("nan")
    while time.now() < deadline:
        state = camera.get_state()
        color = state.get("color")
        stamp = state.get("camera_receive_timestamp") or state.get("timestamp")
        if color is not None and stamp:
            age = time.now() - float(stamp)
            if -0.01 <= age <= max_age_s and np.asarray(color).any():
                return np.asarray(color)
        time.sleep(0.05)
    raise RuntimeError(f"No fresh Jev camera frame within {timeout_s:.0f}s (last age {age:.2f}s)")


def reply_path(image_path: Path, backend: str) -> Path:
    """Where the answer for ``image_path`` is stored."""
    return image_path.with_name(f"{image_path.stem}.jev-{backend}.json")


def save_reply(image_path: Path, backend: str, reply: JevReply, reading: CrateReading) -> Path:
    """Write the raw response and the decision beside the JPEG."""
    path = reply_path(image_path, backend)
    payload = {
        "backend": backend,
        "model": reply.model,
        "input_tokens": reply.input_tokens,
        "latency_s": round(reply.latency_s, 3),
        "response": reply.raw,
        "reading": asdict(reading),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_label(image_path: Path) -> dict | None:
    """Hand label ``name.json`` beside ``name.jpg``, if present."""
    path = image_path.with_suffix(".json")
    return json.loads(path.read_text()) if path.is_file() else None


def format_reading(name: str, reply: JevReply, reading: CrateReading) -> str:
    """Text block for one answer."""
    probs = ", ".join(f"{key}={value:.2f}" for key, value in reading.probabilities.items() if value >= 0.01)
    lines = [
        name,
        f"  {reading.summary()}",
        f"  box_count probabilities: {probs or 'all below 0.01'}",
        f"  model={reply.model} input_tokens={reply.input_tokens} latency={reply.latency_s:.2f}s",
    ]
    if reply.image_seems_missing():
        lines.append(
            f"  WARNING: input_tokens {reply.input_tokens} is about a text-only request "
            f"(~{reply.text_tokens_estimate}); the image probably did not reach the model"
        )
    return "\n".join(lines)


def iter_images(path: Path) -> list[Path]:
    """One image, or every JPEG/PNG in a directory."""
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"no JPEG or PNG files in {path}")
    return files


@dataclass
class Args:
    """Ask Jev about the crate from the Jev camera or from saved images. Nothing moves."""

    image: Path | None = None
    """File or directory of saved photos. Unset: live Jev camera."""
    serial: str = JEV_SERIAL
    """Serial of the Jev RealSense."""
    jev: JevCfg = field(default_factory=JevCfg)
    thresholds: CrateThresholds = field(default_factory=CrateThresholds)
    probes_dir: Path | None = None
    """Folder for live probes. Unset: experiments/orchestrator/probes."""
    max_frame_age_s: float = 0.5
    """Reject camera frames older than this."""
    mw: str = "Thread"
    mp_method: str = "spawn"


def run_files(args: Args, client: JevClient, spaces: set[int]) -> None:
    """Ask once per saved image and score the labelled ones."""
    assert args.image is not None
    score = ProbeScore(thresholds=args.thresholds)
    labelled = 0
    for path in iter_images(args.image):
        try:
            reply, reading = ask_crate(
                client, encode_jpeg(path.read_bytes()), available_spaces=spaces, thresholds=args.thresholds
            )
        except (JevError, ValueError) as error:
            print(f"{path.name}\n  FAILED: {error}\n")
            continue
        save_reply(path, args.jev.backend, reply, reading)
        print(format_reading(path.name, reply, reading))
        label = load_label(path)
        if label is not None:
            labelled += 1
            score.add(reading, label)
            print(f"  label: {label}")
        print()
    if labelled:
        print(f"Scored {labelled} labelled probe(s):")
        print(score.report())


def run_live(args: Args, client: JevClient, spaces: set[int]) -> None:
    """Enter asks Jev about a fresh frame; q quits."""
    folder = args.probes_dir or probes_root()
    folder.mkdir(parents=True, exist_ok=True)
    with jev_camera_session(args.serial, args.mw) as camera:
        logger.info("Backend {} at {}", args.jev.backend, args.jev.resolved_url())
        while True:
            line = input("Enter = ask Jev, q = quit: ").strip().lower()
            if line in ("q", "quit"):
                return
            jpeg = encode_jpeg(fresh_frame(camera, args.max_frame_age_s))
            path = folder / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"
            path.write_bytes(jpeg)
            try:
                reply, reading = ask_crate(client, jpeg, available_spaces=spaces, thresholds=args.thresholds)
            except (JevError, ValueError) as error:
                print(f"  FAILED: {error}")
                continue
            save_reply(path, args.jev.backend, reply, reading)
            print(format_reading(path.name, reply, reading))


def main(args: Args) -> None:
    client = JevClient(args.jev)
    spaces = set(open_spaces())
    logger.info("Spaces with a home pose: {}", sorted(spaces))
    if args.image is not None:
        run_files(args, client, spaces)
    else:
        run_live(args, client, spaces)


if __name__ == "__main__":
    args = tyro.cli(Args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
