# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask Jev where one object is in the photo, and read the answer. No arm, no map.

A Jev box answer always has four edges, even when the object is absent, so a
noul about visibility is asked in the same request and decides first.
"""

from __future__ import annotations

from dataclasses import dataclass

from rio.jev.client import IMAGE_KEY, validate_probability

VISIBLE_KEY = "visible"
BOX_KEY = "where"


def locate_questions(what: str) -> dict:
    """A noul whether ``what`` is in the photo, and a box around it."""
    return {
        VISIBLE_KEY: {"type": "noul", "instructions": f"Is {what} visible in `{IMAGE_KEY}`?"},
        BOX_KEY: {"type": "box", "instructions": f"{what} in `{IMAGE_KEY}`"},
    }


@dataclass(frozen=True)
class Box:
    """Edges as fractions of the image width and height, origin at the top-left corner."""

    left: float
    top: float
    right: float
    bottom: float
    confidence: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0


def _answer(answers: dict, key: str, kind: str) -> dict:
    answer = answers.get(key) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError(f"answer {key!r} must be a {kind}")
    return answer


def read_noul(answers: dict, key: str = VISIBLE_KEY) -> float:
    """Validated noul probability.

    Raises:
        ValueError: The answer is missing, not a noul, or not a finite value in [0, 1].
    """
    return validate_probability(_answer(answers, key, "noul").get("noul"), key)


def read_box(answers: dict, key: str = BOX_KEY) -> Box:
    """Validated box answer.

    Raises:
        ValueError: The answer is missing, not a box, has an edge outside [0, 1],
            edges out of order, or no valid confidence.
    """
    answer = _answer(answers, key, "box")
    edges = answer.get("box")
    if not isinstance(edges, dict):
        raise ValueError(f"answer {key!r} has no box edges")
    left, top, right, bottom = (
        validate_probability(edges.get(side), f"{key}.{side}") for side in ("left", "top", "right", "bottom")
    )
    if not (left < right and top < bottom):
        raise ValueError(f"answer {key!r} box edges are out of order")
    confidence = validate_probability(answer.get("confidence"), f"{key}.confidence")
    return Box(left, top, right, bottom, confidence)
