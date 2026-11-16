# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Client for Meta's SAM 3.1 Model API: find every instance of a noun phrase in a photo.

One request sends a JPEG and a short noun phrase ("red plate") to
``POST https://api.meta.ai/v1/responses`` and reads back one line per frame in
SAM's token grammar, e.g. ``<0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=0.93|><|mask;...|>,1<|box;...|>``.
Boxes (and scores, if asked for) are parsed here; mask payloads are kept as
raw text. See ``docs/sam-api.md``.

The key comes from ``SAM_API_KEY`` or, for shells that skip ``~/.bashrc``,
from ``~/.config/rio/sam.env``. It is never logged.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

SAM_URL = "https://api.meta.ai/v1/responses"
SAM_MODEL = "sam-3.1"
KEY_ENV = "SAM_API_KEY"
KEY_FILE = Path("~/.config/rio/sam.env")
_FRAME = re.compile(r"<(\d+)f>")
_OBJECT = re.compile(r"(\d+)<\|box;([^|>]*)\|>(?:<\|mask;([^|]*)\|>)?")


class SamError(RuntimeError):
    """The request failed or the reply could not be read."""


@dataclass(frozen=True)
class Instance:
    """One object SAM found.

    Attributes:
        index: Object index within the frame.
        box: ``x1, y1, x2, y2`` in source pixels (``x2``/``y2`` inclusive).
        size: Source image ``width, height``.
        score: Confidence 0..1 when requested, else None.
        mask: Raw mask token body (``x=..;y=..;data=H,W,<enc>...``), or None.
    """

    index: int
    box: tuple[float, float, float, float]
    size: tuple[int, int]
    score: float | None = None
    mask: str | None = None

    def fractions(self) -> tuple[float, float, float, float]:
        """Box edges as fractions of the image width and height (left, top, right, bottom)."""
        w, h = self.size
        x1, y1, x2, y2 = self.box
        return x1 / w, y1 / h, (x2 + 1) / w, (y2 + 1) / h

    @property
    def centre(self) -> tuple[float, float]:
        """Box centre as image fractions ``u, v``."""
        left, top, right, bottom = self.fractions()
        return (left + right) / 2.0, (top + bottom) / 2.0


@dataclass
class SamReply:
    """Instances of frame 0, the raw text, and the round trip time."""

    instances: list[Instance]
    text: str
    elapsed_s: float
    usage: dict = field(default_factory=dict)

    def best(self) -> Instance | None:
        """The instance with the highest score (the first one when there are no scores)."""
        if not self.instances:
            return None
        return max(self.instances, key=lambda i: -1.0 if i.score is None else i.score)


def api_key(env: dict[str, str] | None = None, path: Path = KEY_FILE) -> str:
    """The SAM key from the environment, else from ``~/.config/rio/sam.env``.

    Raises:
        SamError: Neither holds a key.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    key = source.get(KEY_ENV, "").strip()
    if key:
        return key
    file = path.expanduser()
    if file.is_file():
        for raw in file.read_text().splitlines():
            name, _, value = raw.strip().removeprefix("export ").strip().partition("=")
            if name.strip() == KEY_ENV and value.strip():
                return value.strip().strip("'\"")
    raise SamError(f"no SAM key: set {KEY_ENV} or write 'export {KEY_ENV}=...' to {KEY_FILE}")


def parse_frames(text: str) -> dict[int, list[Instance]]:
    """Instances per frame index from SAM's output text."""
    frames: dict[int, list[Instance]] = {}
    parts = _FRAME.split(text)
    # split yields [before, frame, body, frame, body, ...]
    for frame, body in zip(parts[1::2], parts[2::2], strict=True):
        found = []
        for match in _OBJECT.finditer(body):
            fields = dict(item.split("=", 1) for item in match.group(2).split(";") if "=" in item)
            try:
                box = (float(fields["x1"]), float(fields["y1"]), float(fields["x2"]), float(fields["y2"]))
                size = (int(float(fields["w"])), int(float(fields["h"])))
            except (KeyError, ValueError) as error:
                raise SamError(f"unreadable box {match.group(2)!r}") from error
            score = float(fields["c"]) if "c" in fields else None
            found.append(Instance(int(match.group(1)), box, size, score, match.group(3)))
        frames.setdefault(int(frame), []).extend(found)
    return frames


def output_text(payload: dict) -> str:
    """Concatenated ``output_text`` of a non-streamed Responses API reply."""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    texts = []
    for item in payload.get("output") or []:
        for part in (item or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                texts.append(str(part.get("text") or ""))
    return "".join(texts)


@dataclass
class SamClient:
    """Segments photos by text prompt.

    Args:
        key: API key; None reads it with ``api_key()``.
        url: Responses endpoint.
        model: Model id.
        timeout_s: Per request.
        one_bit_masks: Ask for the smaller one-bit mask encoding.
    """

    key: str | None = None
    url: str = SAM_URL
    model: str = SAM_MODEL
    timeout_s: float = 60.0
    one_bit_masks: bool = True

    def request_body(self, jpeg: bytes, prompt: str) -> dict:
        """The JSON body for one photo and one noun phrase."""
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        metadata = {"include_confidence": "true"}
        if self.one_bit_masks:
            metadata["mask_encoding"] = "one_bit"
        return {
            "model": self.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": image}],
                }
            ],
            "metadata": metadata,
            "store": False,
        }

    def segment(self, jpeg: bytes, prompt: str) -> SamReply:
        """Every instance of ``prompt`` in the photo (frame 0).

        Raises:
            SamError: On an HTTP error, an unreachable endpoint, or an unreadable reply.
        """
        if not prompt.strip():
            raise SamError("empty prompt")
        key = self.key or api_key()
        request = urllib.request.Request(
            self.url,
            data=json.dumps(self.request_body(jpeg, prompt)).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise SamError(f"SAM returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SamError(f"SAM at {self.url} is unreachable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SamError(f"SAM reply is not JSON: {exc}") from exc
        text = output_text(payload)
        frames = parse_frames(text)
        return SamReply(frames.get(0, []), text, time.monotonic() - t0, dict(payload.get("usage") or {}))
