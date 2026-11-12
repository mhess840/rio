# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Crate questions for Jev and the decision taken from their answers.

One request asks four things about the overview photo:

- ``view_clear`` (noul): is the crate fully visible, not covered by a person or arm?
- ``organized`` (noul): does the cardboard span both halves of the crate, with open
  compartments? TypeSafe gets ``good_1.jpg`` and the last ``bad_*.jpg`` in the noul
  criteria. The local helper gets every ``good_*.jpg`` and ``bad_*.jpg`` beside
  the live photo.
- ``box_count`` (choice ``zero`` … ``ten``): how many product cartons sit in the compartments?
- ``any_box`` (noul): is there at least one carton? A cross-check on ``box_count``.

``organized`` is mixed with a prior (default: 90 % of inlays arrive standing). A sure
crumpled noul still wins — the prior may only promote a weak yes, never hide a no.
The count is only read when the inlay is organized. A count of N means the next
carton goes into space N+1.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rio.jev.client import JevClient, JevReply, encode_jpeg, validate_probabilities, validate_probability

INLAY_DIR = Path(__file__).resolve().parent / "inlay"
INLAY_EXAMPLE_COUNT = 3
# Reference photos stay at camera size. Only a side above this is reduced.
INLAY_EXAMPLE_MAX_SIDE = 768

# Word keys: the hosted API reorders option names that are bare integers.
COUNT_KEYS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
COUNT_WORDS = {key: index for index, key in enumerate(COUNT_KEYS)}

STATE_NOTE = (
    "Top-down photo of one gray plastic crate on a perforated metal table. "
    "A tan cardboard inlay can divide the crate into compartments with upright separators. "
    "When the inlay is standing, cardboard occupies both the left and the right half of the crate. "
    "When it is collapsed, it sits in one side or corner and a large region of the crate is empty gray. "
    "Small dark retail cartons (CPU-fan boxes with a printed face) may sit in the compartments. "
    "A black foam tray to the right of the crate is outside it. "
    "A person or arm may cover part of the crate."
)

VIEW_CLEAR_INSTRUCTIONS = (
    "Is the gray crate in `crate` fully visible from above, without a person, arm, hand, or other large object covering it?"
)
VIEW_CLEAR_CRITERIA = {
    "true": "The gray crate is fully in view; no person, arm, or large object covers it.",
    "false": "A person, arm, hand, or object covers a large part of the crate.",
}
ORGANIZED_INSTRUCTIONS = (
    "Look only at the tan cardboard inlay in `crate`, ignoring product cartons and anything outside the gray crate. "
    "Standing-inlay examples and collapsed-inlay examples are attached; judge the live photo against them. "
    "Does the inlay stand so that cardboard occupies both the left and the right half of the crate, "
    "with open compartments reachable from above?"
)
ORGANIZED_CRITERIA = {
    "true": (
        "The separators stand and span both halves of the crate. A whole left or right half is not empty gray. "
        "Product cartons in the compartments do not matter."
    ),
    "false": ("The inlay is collapsed into one side or corner, so a large part of the crate is empty gray with no cardboard."),
}
COUNT_INSTRUCTIONS = (
    "How many product cartons are inside the compartments of the gray crate in `crate`? "
    "A product carton is a small dark retail box with a printed face; count it even if the printing is not readable. "
    "Do not count the tan cardboard separators, shelves, or walls. Do not count anything outside the gray crate."
)
ANY_BOX_INSTRUCTIONS = (
    "Does `crate` show at least one small dark product carton inside the gray crate, ignoring the tan cardboard separators?"
)

Decision = Literal["organize", "open", "hold"]


def _count_description(n: int) -> str:
    if n == 0:
        return "No product carton is inside the gray crate; the compartments hold only cardboard separators."
    noun = "carton" if n == 1 else "cartons"
    return f"Exactly {COUNT_KEYS[n]} dark product {noun} sit in the compartments of the gray crate."


def _example_paths(folder: Path, side: str) -> list[Path]:
    """``side_1.jpg`` … ``side_3.jpg``, plus any further numbered files that exist."""
    paths = [folder / f"{side}_{index}.jpg" for index in range(1, INLAY_EXAMPLE_COUNT + 1)]
    extra = INLAY_EXAMPLE_COUNT + 1
    while (folder / f"{side}_{extra}.jpg").is_file():
        paths.append(folder / f"{side}_{extra}.jpg")
        extra += 1
    return paths


def inlay_example_jpegs(directory: Path | None = None) -> dict[str, list[bytes]]:
    """Load the standing and collapsed inlay photos.

    ``good_1`` … ``good_3`` and ``bad_1`` … ``bad_3`` are required. Further files
    such as ``bad_4.jpg`` are included when they exist. Missing required files
    are not an error: the organized question stays text-only. A side longer than
    ``INLAY_EXAMPLE_MAX_SIDE`` is reduced.
    """
    folder = INLAY_DIR if directory is None else directory
    names = {side: _example_paths(folder, side) for side in ("good", "bad")}
    if not all(path.is_file() for path in (file for files in names.values() for file in files)):
        return {}
    return {
        side: [encode_jpeg(path.read_bytes(), max_side=INLAY_EXAMPLE_MAX_SIDE) for path in files]
        for side, files in names.items()
    }


def crate_questions(*, example_refs: dict[str, str] | None = None) -> dict:
    """The crate questions, in ``/v1/systemone`` JSON form.

    Args:
        example_refs: TypeSafe asset keys for the organized noul, with ``true``
            (standing inlay) and ``false`` (crumpled inlay). Unset: text criteria.
    """
    criteria: dict = dict(ORGANIZED_CRITERIA)
    if example_refs is not None:
        missing = {"true", "false"} - set(example_refs)
        if missing:
            raise ValueError(f"example_refs must have true and false, missing {sorted(missing)}")
        criteria = {
            "true": {"looks_like": example_refs["true"], "definition": ORGANIZED_CRITERIA["true"]},
            "false": {"looks_like": example_refs["false"], "definition": ORGANIZED_CRITERIA["false"]},
        }
    return {
        "view_clear": {
            "type": "noul",
            "instructions": VIEW_CLEAR_INSTRUCTIONS,
            "criteria": dict(VIEW_CLEAR_CRITERIA),
        },
        "organized": {"type": "noul", "instructions": ORGANIZED_INSTRUCTIONS, "criteria": criteria},
        "box_count": {
            "type": "choice",
            "instructions": COUNT_INSTRUCTIONS,
            "criteria": {key: _count_description(n) for n, key in enumerate(COUNT_KEYS)},
        },
        "any_box": {"type": "noul", "instructions": ANY_BOX_INSTRUCTIONS},
    }


@dataclass(frozen=True)
class CrateThresholds:
    """When an answer is trusted enough to drive the arm."""

    yes: float = 0.7
    """A noul at or above this reads as yes."""
    no: float = 0.3
    """A noul at or below this reads as no; between ``no`` and ``yes`` is unsure."""
    count_min_p: float = 0.7
    """Minimum probability of the chosen count before the arm may act on it."""
    organized_prior: float = 0.9
    """P(inlay is standing) before the photo. ``0.5`` leaves the noul unchanged."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.no < self.yes <= 1.0:
            raise ValueError("thresholds need 0 <= no < yes <= 1")
        if not 0.0 < self.count_min_p <= 1.0:
            raise ValueError("count_min_p must be in (0, 1]")
        if not 0.0 <= self.organized_prior <= 1.0:
            raise ValueError("organized_prior must be in [0, 1]")


@dataclass(frozen=True)
class CrateReading:
    """Jev's view of the crate and what the orchestrator should do next."""

    organized_p: float
    """Posterior P(standing | image), after mixing the noul with ``organized_prior``."""
    count: int
    count_p: float
    confidence: float | None
    """Model confidence of the count; shown only, never used to decide."""
    any_box_p: float
    probabilities: dict[str, float] = field(default_factory=dict)
    decision: Decision = "hold"
    # 1-based space the next carton goes into; set only when ``decision`` is ``open``.
    space: int | None = None
    reason: str = ""
    organized_q: float = 0.0
    """Raw organized noul, before the prior."""
    view_clear_p: float = 1.0
    """Noul that the crate is fully visible."""

    def summary(self) -> str:
        """One line for the terminal."""
        target = f" -> space {self.space}" if self.space is not None else ""
        confidence = "n/a" if self.confidence is None else f"{self.confidence:.2f}"
        return (
            f"organized q={self.organized_q:.2f}→p={self.organized_p:.2f}  view={self.view_clear_p:.2f}  "
            f"count={self.count} (p={self.count_p:.2f}, conf={confidence})  "
            f"any_box={self.any_box_p:.2f}  decision={self.decision}{target}  [{self.reason}]"
        )


def bayes_posterior(likelihood: float, prior: float) -> float:
    """Mix a model probability with a new prior, assuming the model used 0.5.

    ``prior=0.5`` returns ``likelihood`` unchanged. Values are clamped off 0/1
    so a sure noul cannot be washed out by arithmetic underflow.
    """
    prior = validate_probability(prior, "prior")
    q = min(max(validate_probability(likelihood, "likelihood"), 1e-6), 1.0 - 1e-6)
    if prior <= 0.0:
        return 0.0
    if prior >= 1.0:
        return 1.0
    return (q * prior) / (q * prior + (1.0 - q) * (1.0 - prior))


def _noul(answers: dict, key: str, default: float | None = None) -> float:
    answer = answers.get(key)
    if answer is None and default is not None:
        return default
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError(f"answer {key!r} must be a noul")
    return validate_probability(answer.get("noul"), f"answer {key!r}")


def read_crate(
    answers: dict,
    *,
    available_spaces: Collection[int],
    thresholds: CrateThresholds | None = None,
) -> CrateReading:
    """Turn the answers into a decision.

    A blocked view always holds. A sure crumpled noul organizes even when the
    prior is high. Otherwise the organized posterior (noul mixed with
    ``organized_prior``) must reach ``yes`` before the count is used.

    Args:
        answers: The ``answers`` object of the response.
        available_spaces: 1-based spaces that have a recorded home pose.
        thresholds: Trust limits; defaults to ``CrateThresholds()``.

    Raises:
        ValueError: If an answer is missing, has the wrong type, or names an unknown count.
    """
    if not isinstance(answers, dict):
        raise ValueError("Jev answers must be an object")
    limits = thresholds or CrateThresholds()
    organized_q = _noul(answers, "organized")
    organized_p = bayes_posterior(organized_q, limits.organized_prior)
    view_clear_p = _noul(answers, "view_clear")
    any_box_p = _noul(answers, "any_box")
    count_answer = answers.get("box_count")
    if not isinstance(count_answer, dict) or count_answer.get("type") != "choice":
        raise ValueError("answer 'box_count' must be a choice")
    probabilities = validate_probabilities(count_answer.get("probabilities"), "box_count")
    if not set(probabilities).issubset(COUNT_WORDS):
        raise ValueError("box_count contains an unknown count option")
    choice = str(count_answer.get("choice") or "")
    if choice not in COUNT_WORDS:
        raise ValueError(f"box_count chose {choice!r}, expected one of {COUNT_KEYS}")
    count = COUNT_WORDS[choice]
    if choice not in probabilities:
        raise ValueError("box_count has no probability for its selected count")
    count_p = probabilities[choice]
    raw_confidence = count_answer.get("confidence")
    confidence = None if raw_confidence is None else validate_probability(raw_confidence, "box_count confidence")

    def reading(decision: Decision, reason: str, space: int | None = None) -> CrateReading:
        return CrateReading(
            organized_p=organized_p,
            count=count,
            count_p=count_p,
            confidence=confidence,
            any_box_p=any_box_p,
            probabilities=probabilities,
            decision=decision,
            space=space,
            reason=reason,
            organized_q=organized_q,
            view_clear_p=view_clear_p,
        )

    if view_clear_p <= limits.no:
        return reading("hold", "view blocked")
    if view_clear_p < limits.yes:
        return reading("hold", f"view unsure ({view_clear_p:.2f})")
    # Sure crumpled: the photo wins. The prior must not hide a collapsed inlay.
    if organized_q <= limits.no:
        return reading("organize", "inlay is not organized")
    # Weak yes may be promoted by the 90 % standing prior.
    if organized_p < limits.yes:
        return reading("hold", f"inlay state unsure (q={organized_q:.2f} p={organized_p:.2f})")
    if count_p < limits.count_min_p:
        return reading("hold", f"count unsure (p={count_p:.2f} < {limits.count_min_p:.2f})")
    if count == 0 and any_box_p >= limits.yes:
        return reading("hold", "count is zero but any_box says there is a carton")
    if count > 0 and any_box_p <= limits.no:
        return reading("hold", f"count is {count} but any_box says the crate is empty")
    space = count + 1
    if space not in set(available_spaces):
        return reading("hold", f"no home pose recorded for space {space}")
    return reading("open", f"{count} carton(s) loaded", space)


def ask_crate(
    client: JevClient,
    image_jpeg: bytes,
    *,
    available_spaces: Collection[int],
    thresholds: CrateThresholds | None = None,
    inlay_dir: Path | None = None,
) -> tuple[JevReply, CrateReading]:
    """Ask the crate questions about one JPEG and decide.

    On the hosted API, the first standing and crumpled JPEGs under ``inlay_dir``
    (default: the ``inlay`` folder of this package) go into the organized noul criteria.
    The local helper receives all three of each beside the live photo.

    Args:
        client: Configured Jev client.
        image_jpeg: Overview JPEG from ``encode_jpeg``.
        available_spaces: 1-based spaces that have a recorded home pose.
        thresholds: Trust limits.
        inlay_dir: Folder with ``good_1.jpg`` … ``bad_3.jpg`` and any later ``bad_N.jpg``. Unset: the default.
    """
    extra_images: dict[str, bytes] = {}
    inlay_examples: dict[str, list[bytes]] | None = None
    questions = crate_questions()
    examples = inlay_example_jpegs(inlay_dir)
    if examples and client.cfg.backend == "typesafe":
        true_ref = f"{uuid.uuid4()}-inlay-good"
        false_ref = f"{uuid.uuid4()}-inlay-bad"
        questions = crate_questions(example_refs={"true": true_ref, "false": false_ref})
        extra_images = {true_ref: examples["good"][0], false_ref: examples["bad"][-1]}
    elif examples and client.cfg.backend == "local":
        inlay_examples = examples
    reply = client.ask(
        image_jpeg,
        questions,
        STATE_NOTE,
        extra_images=extra_images or None,
        inlay_examples=inlay_examples,
    )
    return reply, read_crate(reply.answers, available_spaces=available_spaces, thresholds=thresholds)


@dataclass
class ProbeScore:
    """Agreement of readings with hand labels ``{"organized": bool, "count": int}``.

    ``organized`` is right when a sure crumpled noul matches a false label, or
    when the posterior reaches ``yes`` on a true label. Between those it is
    unsure. ``count`` is scored only on organized probes, both on every probe
    and on the ones the gate would accept, since an accepted wrong count is the
    error that sends the arm to the wrong space.
    """

    thresholds: CrateThresholds = field(default_factory=CrateThresholds)
    organized_right: int = 0
    organized_wrong: int = 0
    organized_unsure: int = 0
    count_right: int = 0
    count_wrong: int = 0
    accepted_right: int = 0
    accepted_wrong: int = 0
    confusion: Counter = field(default_factory=Counter)

    def add(self, reading: CrateReading, label: dict) -> None:
        """Score one probe against its label."""
        organized = label.get("organized")
        if organized is not None:
            if reading.organized_q <= self.thresholds.no:
                predicted = False
            elif reading.organized_p >= self.thresholds.yes:
                predicted = True
            else:
                self.organized_unsure += 1
                predicted = None
            if predicted is not None:
                if predicted == bool(organized):
                    self.organized_right += 1
                else:
                    self.organized_wrong += 1
        if "count" not in label or organized is False:
            return
        expected = int(label["count"])
        self.confusion[(expected, reading.count)] += 1
        right = reading.count == expected
        self.count_right += right
        self.count_wrong += not right
        if reading.count_p >= self.thresholds.count_min_p:
            self.accepted_right += right
            self.accepted_wrong += not right

    def report(self) -> str:
        """Totals and the count confusion matrix as text."""
        lines = [
            f"organized: {self.organized_right} right, {self.organized_wrong} wrong, {self.organized_unsure} unsure",
            f"count (all organized probes): {self.count_right} right, {self.count_wrong} wrong",
            f"count (accepted, p >= {self.thresholds.count_min_p:.2f}): "
            f"{self.accepted_right} right, {self.accepted_wrong} wrong",
        ]
        if self.confusion:
            labels = sorted({true for true, _ in self.confusion})
            predicted = sorted({pred for _, pred in self.confusion})
            lines.append("confusion (rows = label, columns = Jev):")
            lines.append("      " + "".join(f"{p:>4}" for p in predicted))
            for true in labels:
                lines.append(f"  {true:>3} " + "".join(f"{self.confusion[(true, p)]:>4}" for p in predicted))
        return "\n".join(lines)
