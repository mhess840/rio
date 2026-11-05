"""Ask OpenJev which crate slot to fill, without moving the arm.

One overview image, two questions. ``next`` picks the earliest empty slot in the
fixed order space 1, then space 2, then space 3. ``inlay`` asks whether the
cardboard insert is too shifted to fill. The winning sentence is already the
policy instruction, and the pose name is the one ``goto_kassow_q --pose`` drives.

This only prints the decision. A later caller can act on ``CrateDecision`` when
``act`` is true: home pose ``decision.pose``, instruction ``decision.instruction``.
A shifted inlay or a low choice confidence leaves ``act`` false.

    uv run python -m examples.openjev_crate photo.jpg
    uv run python -m examples.openjev_crate experiments/open/openjev_probes

A probe directory holds ``*.jpg`` files. ``name.json`` next to ``name.jpg`` may
say ``{"next": "put the object in space 1", "inlay": false}``.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tyro

from examples.cfg.kassow_experiments import EXPERIMENTS, InstructionHome

# Where each slot sits in the overview image. The fill order is space1, space2, space3.
SLOT_WHERE = {
    "space1": "the bottom slot of the right column, closest to the front lip of the crate, filled first",
    "space2": "the upper slot of the right column, behind the bottom one, filled second",
    "space3": "the left column, filled only after both slots of the right column hold an object",
}
STATE_NOTE = (
    "Overview of one crate. Cardboard dividers split it into a right column of two slots and a left column. "
    "Fill empty slots in order: space 1, then space 2, then space 3. "
    "Space 1 is the bottom slot of the right column, space 2 is the upper slot of the right column, "
    "and space 3 is the left column."
)
NEXT_INSTRUCTIONS = (
    "Choose the earliest slot in the order space 1, then space 2, then space 3 that is still empty. "
    "A slot that already holds an object is not next. "
    "If a later slot is empty but an earlier slot is also empty, choose the earlier one."
)
INLAY_INSTRUCTIONS = (
    "Is the cardboard inlay shifted so far that the slots are no longer in their usual places "
    "and it must be straightened before any slot can be filled?"
)
# Choice confidence is (max p - 1/N) / (1 - 1/N). Below this, do not drive.
CONFIDENCE_ACT_THRESHOLD = 0.5
# Calibrated yes-probability from the noul question. At or above this, do not drive.
INLAY_ACT_THRESHOLD = 0.5


@dataclass(frozen=True)
class CrateDecision:
    """What a later caller would do with one overview frame. Nothing is moved here."""

    instruction: str | None
    pose: str | None
    act: bool
    reason: str
    inlay_probability: float
    confidence: float
    probabilities: dict[str, float]


def open_homes() -> tuple[InstructionHome, ...]:
    """The three ``open`` homes, in fill order."""
    homes = EXPERIMENTS["open"].homes
    if tuple(home.pose for home in homes) != ("space1", "space2", "space3"):
        raise ValueError("open homes are no longer space1, space2, space3 in that order")
    return homes


def choice_confidence(probabilities: dict[str, float]) -> float:
    """OpenJev's choice confidence, ``(max p - 1/N) / (1 - 1/N)``."""
    if not probabilities:
        raise ValueError("choice probabilities are empty")
    top = max(probabilities.values())
    count = len(probabilities)
    if count == 1:
        return 1.0
    return (top - 1.0 / count) / (1.0 - 1.0 / count)


def request_body(image_jpeg: bytes) -> dict:
    """One ``/v1/systemone`` body: the overview JPEG plus the two questions.

    Args:
        image_jpeg: Overview camera JPEG, not a wrist image.
    """
    encoded = base64.b64encode(image_jpeg).decode("ascii")
    criteria = {home.instruction: SLOT_WHERE[home.pose] for home in open_homes()}
    return {
        "model": "openjev",
        "state": {
            "screenshot": f"data:image/jpeg;base64,{encoded}",
            "note": STATE_NOTE,
        },
        "questions": {
            "next": {"type": "choice", "instructions": NEXT_INSTRUCTIONS, "criteria": criteria},
            "inlay": {"type": "noul", "instructions": INLAY_INSTRUCTIONS},
        },
    }


def decision_from_answers(
    answers: dict,
    *,
    confidence_threshold: float = CONFIDENCE_ACT_THRESHOLD,
    inlay_threshold: float = INLAY_ACT_THRESHOLD,
) -> CrateDecision:
    """Turn the helper's ``answers`` into a pose and a sentence, or a refusal.

    Args:
        answers: The ``answers`` object from ``POST /v1/systemone``.
        confidence_threshold: Minimum choice confidence before a later caller may drive.
        inlay_threshold: Yes-probability at or above which the inlay must be straightened first.

    Raises:
        ValueError: If a required answer is missing or has the wrong type.
    """
    nxt = answers.get("next")
    inlay = answers.get("inlay")
    if not isinstance(nxt, dict) or nxt.get("type") != "choice":
        raise ValueError("answer 'next' must be a choice")
    if not isinstance(inlay, dict) or inlay.get("type") != "noul":
        raise ValueError("answer 'inlay' must be a noul")
    probabilities = {str(key): float(value) for key, value in dict(nxt.get("probabilities") or {}).items()}
    if not probabilities:
        raise ValueError("answer 'next' has no probabilities")
    confidence = choice_confidence(probabilities)
    inlay_probability = float(inlay["noul"])
    choice = str(nxt.get("choice") or "")
    pose_by_instruction = {home.instruction: home.pose for home in open_homes()}
    pose = pose_by_instruction.get(choice)
    if inlay_probability >= inlay_threshold:
        return CrateDecision(None, None, False, "inlay", inlay_probability, confidence, probabilities)
    if pose is None:
        return CrateDecision(None, None, False, "unknown_choice", inlay_probability, confidence, probabilities)
    if confidence < confidence_threshold:
        return CrateDecision(choice, pose, False, "low_confidence", inlay_probability, confidence, probabilities)
    return CrateDecision(choice, pose, True, "ok", inlay_probability, confidence, probabilities)


def ask(
    image_jpeg: bytes,
    url: str,
    *,
    timeout_s: float = 120.0,
    confidence_threshold: float = CONFIDENCE_ACT_THRESHOLD,
    inlay_threshold: float = INLAY_ACT_THRESHOLD,
) -> CrateDecision:
    """Send one overview JPEG to a local OpenJev helper.

    Args:
        image_jpeg: Overview camera JPEG.
        url: Helper origin, for example ``http://127.0.0.1:3000``.
        timeout_s: HTTP timeout. A first image read can take several seconds.
        confidence_threshold: Passed to ``decision_from_answers``.
        inlay_threshold: Passed to ``decision_from_answers``.

    Raises:
        urllib.error.URLError: If the helper is not running.
        ValueError: If the helper returns an error body or an unexpected answer.
    """
    payload = json.dumps(request_body(image_jpeg)).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenJev helper returned {exc.code}: {detail}") from exc
    if "error" in body:
        raise ValueError(f"OpenJev helper error: {body['error']}")
    return decision_from_answers(
        body["answers"],
        confidence_threshold=confidence_threshold,
        inlay_threshold=inlay_threshold,
    )


def load_label(path: Path) -> dict | None:
    """Read ``image.json`` beside an image, if the probe was named."""
    label_path = path.with_suffix(".json")
    if not label_path.is_file():
        return None
    return json.loads(label_path.read_text())


def format_decision(path: Path, decision: CrateDecision, label: dict | None) -> str:
    """One block of text for a single image."""
    probs = ", ".join(f"{key}={value:.3f}" for key, value in decision.probabilities.items())
    sentence = decision.instruction or "none"
    pose = decision.pose or "none"
    lines = [
        path.name,
        f"  next: {sentence}  pose={pose}  act={decision.act}  reason={decision.reason}",
        f"  confidence={decision.confidence:.3f}  inlay={decision.inlay_probability:.3f}",
        f"  probabilities: {probs}",
    ]
    if label is not None and "next" in label:
        expected = str(label["next"])
        slot_ok = decision.act and decision.instruction == expected
        lines.append(f"  label next={expected!r}  match_slot={slot_ok}")
    if label is not None and "inlay" in label:
        expected_inlay = bool(label["inlay"])
        inlay_ok = (decision.inlay_probability >= INLAY_ACT_THRESHOLD) == expected_inlay
        lines.append(f"  label inlay={expected_inlay}  match_inlay={inlay_ok}")
    return "\n".join(lines)


def iter_images(path: Path) -> list[Path]:
    """One JPEG, or every JPEG in a directory."""
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg"))
    if not files:
        raise FileNotFoundError(f"no JPEG files in {path}")
    return files


@dataclass
class Args:
    """Score overview JPEGs against a local OpenJev helper. The arm is not moved."""

    image: Path
    url: str = "http://127.0.0.1:3000"
    timeout_s: float = 180.0
    confidence_threshold: float = CONFIDENCE_ACT_THRESHOLD
    inlay_threshold: float = INLAY_ACT_THRESHOLD


def main(args: Args) -> None:
    """Print one decision per image."""
    for path in iter_images(args.image):
        decision = ask(
            path.read_bytes(),
            args.url,
            timeout_s=args.timeout_s,
            confidence_threshold=args.confidence_threshold,
            inlay_threshold=args.inlay_threshold,
        )
        print(format_decision(path, decision, load_label(path)))
        print()


if __name__ == "__main__":
    main(tyro.cli(Args))
