# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""One ``/v1/systemone`` call about one JPEG, against TypeSafe or a local OpenJev helper.

Both backends get the same questions and return the same ``answers`` shape. Only
the image transport differs:

- ``typesafe``: the JPEG sits once in ``assets`` under a fresh UUID, ``state``
  holds that UUID, and the request carries ``TypeSafe-Danger-Zone-Images: true``.
  Without both, the hosted model reads text only and never sees the photo.
- ``local``: the OpenJev helper (``scripts/openjev_serve.sh``) takes the image as
  a ``data:`` URL in ``state.screenshot`` and sends the rest of ``state`` as text.

Questions refer to the image as `` `crate` `` in both cases.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

TYPESAFE_URL = "https://api.typesafe.ai"
LOCAL_URL = "http://127.0.0.1:3000"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
LOCAL_TOKEN_ENV = "SHIM_TOKEN"
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
IMAGE_KEY = "crate"
# Hosted API limits on each image side, both ends included.
MIN_SIDE = 280
MAX_SIDE = 1568


class JevError(RuntimeError):
    """The Jev endpoint refused the request or answered with something unusable."""


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_url(url: str, *, with_credentials: bool = False) -> str:
    """Accept only http(s) endpoints, and credentials only over TLS or loopback.

    ``urllib`` would also open ``file://`` and ``ftp://`` URLs, so the scheme is checked first.

    Raises:
        ValueError: If the scheme is not http(s), the host is missing, or a credential
            would travel over plain HTTP to another machine.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Jev URL must be http(s) with a host, got {url!r}")
    if with_credentials and parts.scheme != "https" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(f"Refusing to send a credential over plain HTTP to {parts.hostname}; use https")
    return url


def validate_probability(value: object, name: str) -> float:
    """Require a finite numeric value in the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric probability")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


def validate_probabilities(value: object, name: str) -> dict[str, float]:
    """Validate a non-empty distribution, allowing one percent rounding error."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty probability distribution")
    probabilities = {str(key): validate_probability(probability, f"{name}.{key}") for key, probability in value.items()}
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=0.01):
        raise ValueError(f"{name} probabilities must sum to 1")
    return probabilities


@dataclass
class JevCfg:
    """Which Jev endpoint answers the questions."""

    backend: Literal["typesafe", "local"] = "typesafe"
    """Hosted TypeSafe API or the local OpenJev helper."""
    url: str = ""
    """Endpoint origin. Empty: TypeSafe, or http://127.0.0.1:3000 for local."""
    model: str = "jev-latest"
    """Model name; the local helper ignores it."""
    timeout_s: float = 60.0
    """HTTP timeout per request."""

    def resolved_url(self) -> str:
        """Endpoint origin without a trailing slash."""
        default = TYPESAFE_URL if self.backend == "typesafe" else LOCAL_URL
        return (self.url or default).rstrip("/")


@dataclass(frozen=True)
class JevReply:
    """Typed answers plus what is needed to judge whether the image arrived."""

    answers: dict
    model: str
    input_tokens: int | None
    latency_s: float
    raw: dict
    # Rough token count of the text alone (about 4 characters per token).
    text_tokens_estimate: int = 0

    def image_seems_missing(self, margin: int = 150) -> bool:
        """True when ``input_tokens`` is not clearly above a text-only request."""
        if self.input_tokens is None:
            return False
        return self.input_tokens < self.text_tokens_estimate + margin


def encode_jpeg(image: np.ndarray | bytes, *, quality: int = 90, max_side: int = MAX_SIDE) -> bytes:
    """Baseline RGB JPEG within the hosted size limits.

    Args:
        image: ``H x W x 3`` uint8 RGB frame, or the bytes of a PNG/JPEG file.
        quality: JPEG quality.
        max_side: Longest side after scaling. The default is the hosted maximum.

    Raises:
        ValueError: If a side is below ``MIN_SIDE`` pixels, or ``max_side`` is below it.
    """
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"expected an H x W x 3 RGB frame, got shape {arr.shape}")
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if max_side < MIN_SIDE:
        raise ValueError(f"max_side {max_side} is below the {MIN_SIDE} px minimum side")
    if min(img.size) < MIN_SIDE:
        raise ValueError(f"image {img.size[0]}x{img.size[1]} is below the {MIN_SIDE} px minimum side")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality, progressive=False)
    return out.getvalue()


class JevClient:
    """Send one image and a question map to the configured Jev backend."""

    def __init__(self, cfg: JevCfg | None = None):
        self.cfg = cfg or JevCfg()
        if self.cfg.backend not in ("typesafe", "local"):
            raise ValueError(f"backend must be 'typesafe' or 'local', got {self.cfg.backend!r}")

    def build_request(
        self,
        image_jpeg: bytes,
        questions: dict,
        note: str = "",
        extra_images: dict[str, bytes] | None = None,
        inlay_examples: dict[str, list[bytes]] | None = None,
    ) -> tuple[dict, dict[str, str]]:
        """Request body and headers for one call. Authorization is added by ``ask``.

        Args:
            image_jpeg: JPEG bytes, already prepared with ``encode_jpeg``.
            questions: Question map; instructions refer to the image as `` `crate` ``.
            note: Optional text context sent alongside the image.
            extra_images: More JPEGs keyed by the reference string that appears in
                ``questions``. TypeSafe lists them in ``assets``.
            inlay_examples: Local helper only. ``good`` and ``bad`` are JPEG lists
                sent as data-URLs in ``state.inlay_good`` and ``state.inlay_bad``.
        """
        if not questions:
            raise ValueError("questions must not be empty")
        encoded = base64.b64encode(image_jpeg).decode("ascii")
        headers = {"Content-Type": "application/json"}
        if self.cfg.backend == "typesafe":
            ref = f"{uuid.uuid4()}-{IMAGE_KEY}"
            state: dict = {IMAGE_KEY: ref}
            if note:
                state["note"] = note
            assets = {ref: {"type": "image", "format": "jpg", "data_base64": encoded}}
            for extra_ref, extra_jpeg in (extra_images or {}).items():
                assets[extra_ref] = {
                    "type": "image",
                    "format": "jpg",
                    "data_base64": base64.b64encode(extra_jpeg).decode("ascii"),
                }
            body = {"model": self.cfg.model, "state": state, "assets": assets, "questions": questions}
            headers[IMAGES_HEADER] = "true"
        else:
            state = {
                "screenshot": f"data:image/jpeg;base64,{encoded}",
                IMAGE_KEY: "the overview image of the crate, attached as the screenshot",
            }
            if note:
                state["note"] = note
            for key in ("good", "bad"):
                jpegs = list((inlay_examples or {}).get(key) or [])
                if jpegs:
                    urls = [f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('ascii')}" for jpeg in jpegs]
                    state[f"inlay_{key}"] = urls
            body = {"model": self.cfg.model, "state": state, "questions": questions}
        return body, headers

    def _auth_header(self) -> dict[str, str]:
        if self.cfg.backend == "typesafe":
            key = os.environ.get(TYPESAFE_KEY_ENV, "").strip()
            if not key:
                raise JevError(f"{TYPESAFE_KEY_ENV} is not set; export your TypeSafe API key first")
            return {"Authorization": f"Bearer {key}"}
        token = os.environ.get(LOCAL_TOKEN_ENV, "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def ask(
        self,
        image_jpeg: bytes,
        questions: dict,
        note: str = "",
        extra_images: dict[str, bytes] | None = None,
        inlay_examples: dict[str, list[bytes]] | None = None,
    ) -> JevReply:
        """POST one request and return the parsed answers.

        Raises:
            JevError: On an HTTP error, an unreachable endpoint, or a body without answers.
        """
        body, headers = self.build_request(
            image_jpeg, questions, note, extra_images=extra_images, inlay_examples=inlay_examples
        )
        headers.update(self._auth_header())
        validate_url(self.cfg.resolved_url(), with_credentials="Authorization" in headers)
        request = urllib.request.Request(
            f"{self.cfg.resolved_url()}/v1/systemone",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise JevError(f"Jev returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JevError(f"Jev at {self.cfg.resolved_url()} is unreachable: {exc}") from exc
        text_chars = len(json.dumps(questions)) + len(note)
        return parse_reply(payload, latency_s=time.monotonic() - t0, text_tokens_estimate=text_chars // 4)


def parse_reply(payload: dict, *, latency_s: float = 0.0, text_tokens_estimate: int = 0) -> JevReply:
    """Validate a ``/v1/systemone`` response body.

    Raises:
        JevError: If the body carries an error or no ``answers`` map.
    """
    if not isinstance(payload, dict):
        raise JevError("Jev response is not a JSON object")
    if "error" in payload:
        raise JevError(f"Jev error: {payload['error']}")
    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise JevError("Jev response has no answers")
    usage = payload.get("usage") or {}
    tokens = usage.get("input_tokens")
    return JevReply(
        answers=answers,
        model=str(payload.get("model", "")),
        input_tokens=int(tokens) if tokens is not None else None,
        latency_s=float(latency_s),
        raw=payload,
        text_tokens_estimate=int(text_tokens_estimate),
    )
