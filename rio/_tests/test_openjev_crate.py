# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""OpenJev crate decision: prompt, confidence gate, and pose mapping. No server and no robot."""

import json

import pytest

from examples.openjev_crate import (
    CONFIDENCE_ACT_THRESHOLD,
    INLAY_ACT_THRESHOLD,
    choice_confidence,
    decision_from_answers,
    format_decision,
    load_label,
    request_body,
)

pytestmark = pytest.mark.unit

SPACE1 = "put the object in space 1"
SPACE2 = "put the object in space 2"
SPACE3 = "put the object in space 3"


def _answers(choice: str, probabilities: dict[str, float], inlay: float) -> dict:
    return {
        "next": {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.0},
        "inlay": {"type": "noul", "noul": inlay},
    }


def test_request_uses_the_policy_sentences_and_both_questions():
    body = request_body(b"\xff\xd8\xff")
    assert body["state"]["screenshot"].startswith("data:image/jpeg;base64,")
    questions = body["questions"]
    assert set(questions) == {"next", "inlay"}
    assert questions["next"]["type"] == "choice"
    assert questions["inlay"]["type"] == "noul"
    assert list(questions["next"]["criteria"]) == [SPACE1, SPACE2, SPACE3]
    assert "bottom slot of the right column" in questions["next"]["criteria"][SPACE1]


def test_a_clear_empty_first_slot_may_drive():
    decision = decision_from_answers(_answers(SPACE1, {SPACE1: 0.9, SPACE2: 0.05, SPACE3: 0.05}, 0.1))
    assert decision.act is True
    assert decision.reason == "ok"
    assert decision.instruction == SPACE1
    assert decision.pose == "space1"
    assert decision.confidence == pytest.approx(choice_confidence({SPACE1: 0.9, SPACE2: 0.05, SPACE3: 0.05}))
    assert decision.confidence > CONFIDENCE_ACT_THRESHOLD


def test_a_shifted_inlay_blocks_the_slot_choice():
    decision = decision_from_answers(_answers(SPACE2, {SPACE1: 0.05, SPACE2: 0.9, SPACE3: 0.05}, 0.8))
    assert decision.act is False
    assert decision.reason == "inlay"
    assert decision.instruction is None
    assert decision.pose is None
    assert decision.inlay_probability >= INLAY_ACT_THRESHOLD


def test_a_near_tie_does_not_drive():
    decision = decision_from_answers(_answers(SPACE3, {SPACE1: 0.34, SPACE2: 0.33, SPACE3: 0.33}, 0.1))
    assert decision.act is False
    assert decision.reason == "low_confidence"
    assert decision.instruction == SPACE3
    assert decision.pose == "space3"


def test_an_unknown_sentence_does_not_drive():
    decision = decision_from_answers(_answers("open the lid", {SPACE1: 0.1, "open the lid": 0.9}, 0.0))
    assert decision.act is False
    assert decision.reason == "unknown_choice"


def test_probe_label_roundtrip(tmp_path):
    image = tmp_path / "crate.jpg"
    image.write_bytes(b"jpeg")
    (tmp_path / "crate.json").write_text(json.dumps({"next": SPACE1, "inlay": False}))
    assert load_label(image) == {"next": SPACE1, "inlay": False}
    text = format_decision(
        image,
        decision_from_answers(_answers(SPACE1, {SPACE1: 0.9, SPACE2: 0.05, SPACE3: 0.05}, 0.1)),
        {"inlay": False},
    )
    assert "match_inlay=True" in text
    assert "match_slot" not in text
