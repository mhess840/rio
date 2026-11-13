# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Crate questions for Jev and the decision taken from their answers.

One request asks four things about the overview photo:

- ``view_clear`` (noul): is the crate fully visible, not covered by a person or arm?
- ``organized`` (noul): is the inlay's central rib about 90 degrees to the floor
  and in the middle of the crate? TypeSafe gets ``good_1.jpg`` and the last ``bad_*.jpg`` in the noul
  criteria. The local helper gets every ``good_*.jpg`` and ``bad_*.jpg`` beside
  the live photo.
- ``box_count`` (choice ``zero`` … ``ten``): how many product cartons sit in the compartments?
- ``any_box`` (noul): is there at least one carton? A cross-check on ``box_count``.

The organized noul is used as the model returned it. Below 0.70 the rib is not
clear enough and the arm straightens the inlay. At or above 0.70 it is standing
and the count is read. A count of N means the next carton goes into space N+1.

The count prior is off (``CrateThresholds.count_prior``). Turned on, it still
leaves a count at or above 0.70 to the photo. Below that it may pick the number
of cartons a finished place left in the crate, and only when the photo's mass
sits on two neighboring counts and that number is one of them.
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
    "A tan cardboard inlay can divide the crate into compartments. "
    "When the inlay is standing, its central rib is upright, about 90 degrees to the crate floor, "
    "and that rib is in the middle of the crate. One end may be at 90 degrees while the other is slightly off. "
    "A strong slope along the rib is not standing. Whether the side shelves are open or folded flat does not matter. "
    "When it is collapsed, the rib slopes strongly, is clearly tilted, or sits well away from the middle. "
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
    "Is the central rib standing about 90 degrees to the crate floor, and is that rib in the middle of the crate? "
    "Judge the rib, not the side shelves. "
    "A small difference between the two ends still counts as standing: "
    "one end may be at 90 degrees while the other is slightly off. "
    "It is not standing when the rib leans so much that it looks like a slanted wall along much of its length. "
    "That slope is too much even if one end is still upright and a carton would still fit. "
    "Ignore whether the side shelves are open or folded."
)
ORGANIZED_CRITERIA = {
    "true": (
        "The central rib is a thin upright edge, about 90 degrees to the floor, in the middle of the crate. "
        "One end may be upright while the other is slightly off 90 degrees. "
        "Side shelves may be open or folded flat. Product cartons do not matter."
    ),
    "false": (
        "The central rib leans like a slanted wall along much of its length. "
        "That is more than a slight difference between the two ends. "
        "Folded side shelves alone do not make it collapsed. Room for a carton does not make a sloped rib standing."
    ),
}
COUNT_INSTRUCTIONS = (
    "How many product cartons are inside the compartments of the gray crate in `crate`? "
    "A product carton is a small dark retail box with a printed face; count it even if the printing is not readable. "
    "One carton is one object, even when its printed face and its edge look like two things, "
    "and even when it sits low in a compartment. "
    "Do not count the tan cardboard separators, shelves, or walls above a carton. Do not count anything outside the gray crate."
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
    count_lead: float = 0.2
    """How far the chosen count must lead the runner-up when presence is already sure."""
    count_prior: bool = False
    """Follow the cartons a finished place left in the crate when the photo is split.

    Off until the count question is calibrated. A probability at or above
    ``count_min_p`` is never replaced by this prior.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.no < self.yes <= 1.0:
            raise ValueError("thresholds need 0 <= no < yes <= 1")
        if not 0.0 < self.count_min_p <= 1.0:
            raise ValueError("count_min_p must be in (0, 1]")
        if not 0.0 <= self.count_lead <= 1.0:
            raise ValueError("count_lead must be in [0, 1]")


@dataclass(frozen=True)
class CrateReading:
    """Jev's view of the crate and what the orchestrator should do next."""

    organized_p: float
    """Same value as ``organized_q``. Kept so saved readings and the board stay one field."""
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
    """Organized noul from the model, used as returned."""
    view_clear_p: float = 1.0
    """Noul that the crate is fully visible."""

    def summary(self) -> str:
        """One line for the terminal."""
        target = f" -> space {self.space}" if self.space is not None else ""
        confidence = "n/a" if self.confidence is None else f"{self.confidence:.2f}"
        return (
            f"organized={self.organized_q:.2f}  view={self.view_clear_p:.2f}  "
            f"count={self.count} (p={self.count_p:.2f}, conf={confidence})  "
            f"any_box={self.any_box_p:.2f}  decision={self.decision}{target}  [{self.reason}]"
        )


def _noul(answers: dict, key: str, default: float | None = None) -> float:
    answer = answers.get(key)
    if answer is None and default is not None:
        return default
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError(f"answer {key!r} must be a noul")
    return validate_probability(answer.get("noul"), f"answer {key!r}")


def count_is_accepted(
    count: int,
    count_p: float,
    any_box_p: float,
    probabilities: dict[str, float],
    choice: str,
    limits: CrateThresholds,
) -> bool:
    """True when the count may move the arm.

    A probability at or above ``count_min_p`` is enough. Below that, a count of at least one
    is still accepted when ``any_box`` is sure and the chosen word leads the runner-up by
    ``count_lead``. A sure zero is never rescued this way.
    """
    if count_p >= limits.count_min_p:
        return True
    if count <= 0 or any_box_p < limits.yes:
        return False
    others = [value for key, value in probabilities.items() if key != choice]
    lead = count_p - (max(others) if others else 0.0)
    return lead >= limits.count_lead


def count_from_process(probabilities: dict[str, float], expected: int, limits: CrateThresholds) -> int | None:
    """Process count, when the photo is split across two neighboring numbers.

    ``expected`` is how many cartons a finished place left in the crate. The
    prior stays quiet unless ``count_prior`` is on, no count reaches
    ``count_min_p``, the two strongest probabilities are neighbors and sum to
    at least ``yes``, and ``expected`` is one of those two.

    Args:
        probabilities: Count distribution from ``box_count``.
        expected: Cartons the last finished place left in the crate.
        limits: Trust limits. ``count_prior`` must be set for a count to come back.

    Returns:
        ``expected`` when the photo leaves that choice open, otherwise None.
    """
    if not limits.count_prior:
        return None
    if isinstance(expected, bool) or not isinstance(expected, int) or not 0 <= expected < len(COUNT_KEYS):
        raise ValueError(f"expected carton count must be an integer from 0 to {len(COUNT_KEYS) - 1}")
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] >= limits.count_min_p:
        return None
    if len(ranked) > 2 and ranked[2][1] == ranked[1][1]:
        return None
    first, second = COUNT_WORDS.get(ranked[0][0]), COUNT_WORDS.get(ranked[1][0])
    if first is None or second is None or abs(first - second) != 1:
        return None
    if ranked[0][1] + ranked[1][1] < limits.yes:
        return None
    if expected not in (first, second):
        return None
    return expected


def read_crate(
    answers: dict,
    *,
    available_spaces: Collection[int],
    thresholds: CrateThresholds | None = None,
    expected_count: int | None = None,
    trust_inlay: bool = False,
) -> CrateReading:
    """Turn the answers into a decision.

    A blocked view always holds. The count is read only when the organized noul
    reaches ``yes``. Anything below that is straightened, including the band
    between ``no`` and ``yes``. ``trust_inlay`` skips that rib check: the organize
    VLA has already run, so the photo is read for the carton count. With
    ``count_prior`` off, ``expected_count`` is ignored. With it on, a split below
    ``count_min_p`` may follow that count.

    Args:
        answers: The ``answers`` object of the response.
        available_spaces: 1-based spaces that have a recorded home pose.
        thresholds: Trust limits; defaults to ``CrateThresholds()``.
        expected_count: Cartons a finished place left in the crate. Unset: the photo alone.
        trust_inlay: Skip the rib check and read the count. The view check still holds.

    Raises:
        ValueError: If an answer is missing, has the wrong type, or names an unknown count.
    """
    if not isinstance(answers, dict):
        raise ValueError("Jev answers must be an object")
    limits = thresholds or CrateThresholds()
    organized_q = _noul(answers, "organized")
    organized_p = organized_q
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

    def reading(
        decision: Decision,
        reason: str,
        space: int | None = None,
        *,
        used_count: int = count,
        used_p: float = count_p,
    ) -> CrateReading:
        return CrateReading(
            organized_p=organized_p,
            count=used_count,
            count_p=used_p,
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
    if organized_q < limits.yes and not trust_inlay:
        reason = (
            "inlay is not organized" if organized_q <= limits.no else f"inlay rib is not clearly upright ({organized_q:.2f})"
        )
        return reading("organize", reason)
    chosen = count
    chosen_p = count_p
    photo_accepted = count_is_accepted(count, count_p, any_box_p, probabilities, choice, limits)
    prior_count = (
        None if photo_accepted or expected_count is None else count_from_process(probabilities, expected_count, limits)
    )
    if photo_accepted:
        reason = f"{chosen} carton(s) loaded"
    elif prior_count is not None:
        chosen = prior_count
        chosen_p = probabilities.get(COUNT_KEYS[chosen], 0.0)
        reason = f"count prior {chosen} (photo {count} at {count_p:.2f})"
    else:
        return reading("hold", f"count unsure (p={count_p:.2f} < {limits.count_min_p:.2f})")
    if chosen == 0 and any_box_p >= limits.yes:
        return reading("hold", "count is zero but any_box says there is a carton", used_count=chosen, used_p=chosen_p)
    if chosen > 0 and any_box_p <= limits.no:
        return reading("hold", f"count is {chosen} but any_box says the crate is empty", used_count=chosen, used_p=chosen_p)
    space = chosen + 1
    if space not in set(available_spaces):
        return reading("hold", f"no home pose recorded for space {space}", used_count=chosen, used_p=chosen_p)
    return reading("open", reason, space, used_count=chosen, used_p=chosen_p)


def ask_crate(
    client: JevClient,
    image_jpeg: bytes,
    *,
    available_spaces: Collection[int],
    thresholds: CrateThresholds | None = None,
    inlay_dir: Path | None = None,
    expected_count: int | None = None,
    trust_inlay: bool = False,
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
        expected_count: Cartons a finished place left in the crate. Used only when ``count_prior`` is on.
        trust_inlay: Skip the rib check. Set after the organize VLA has already straightened it.
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
    return reply, read_crate(
        reply.answers,
        available_spaces=available_spaces,
        thresholds=thresholds,
        expected_count=expected_count,
        trust_inlay=trust_inlay,
    )


@dataclass
class ProbeScore:
    """Agreement of readings with hand labels ``{"organized": bool, "count": int}``.

    ``organized`` is right when the noul is at or below ``no`` on a false label,
    or at or above ``yes`` on a true label. Between those it is unsure.
    ``count`` is scored only on organized probes, both on every probe
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
            elif reading.organized_q >= self.thresholds.yes:
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
        word = COUNT_KEYS[reading.count] if 0 <= reading.count < len(COUNT_KEYS) else ""
        if count_is_accepted(reading.count, reading.count_p, reading.any_box_p, reading.probabilities, word, self.thresholds):
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
