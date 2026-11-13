# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Jev request shapes, crate decisions, and probe scoring. No network and no camera."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rio.apps.kassow.crate import (
    COUNT_INSTRUCTIONS,
    COUNT_KEYS,
    ORGANIZED_CRITERIA,
    ORGANIZED_INSTRUCTIONS,
    STATE_NOTE,
    CrateThresholds,
    ProbeScore,
    ask_crate,
    crate_questions,
    inlay_example_jpegs,
    read_crate,
)
from rio.apps.kassow.jev_ask import load_label, reply_path, save_reply
from rio.jev import JevCfg, JevClient, JevError, encode_jpeg
from rio.jev.client import IMAGES_HEADER, parse_reply

pytestmark = pytest.mark.unit

JPEG = encode_jpeg(np.full((480, 640, 3), 90, dtype=np.uint8))
# Raw ``response`` objects saved by ``rio.apps.kassow.jev_ask`` (``*.jev-<backend>.json``) from the real model.
REAL_REPLIES = sorted((Path(__file__).parent / "data" / "jev_replies").glob("*.json"))


def _answers(organized: float, count: str, count_p: float, any_box: float, view_clear: float = 0.95) -> dict:
    rest = (1.0 - count_p) / (len(COUNT_KEYS) - 1)
    probs = {key: (count_p if key == count else rest) for key in COUNT_KEYS}
    return {
        "view_clear": {"type": "noul", "noul": view_clear},
        "organized": {"type": "noul", "noul": organized},
        "box_count": {"type": "choice", "choice": count, "probabilities": probs, "confidence": 0.9},
        "any_box": {"type": "noul", "noul": any_box},
    }


RAW = CrateThresholds()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_typesafe_request_sends_the_image_as_a_referenced_asset():
    body, headers = JevClient(JevCfg(backend="typesafe")).build_request(JPEG, crate_questions(), "note")
    assert headers[IMAGES_HEADER] == "true"
    (ref, asset), *others = body["assets"].items()
    assert not others
    assert asset["format"] == "jpg"
    assert base64.b64decode(asset["data_base64"]) == JPEG
    assert not asset["data_base64"].startswith("data:")
    assert body["state"]["crate"] == ref
    assert "screenshot" not in body["state"]
    assert any(ref in text for text in _strings({"state": body["state"], "questions": body["questions"]}))


def test_local_request_sends_a_screenshot_data_url_without_assets():
    body, headers = JevClient(JevCfg(backend="local")).build_request(JPEG, crate_questions(), "note")
    assert IMAGES_HEADER not in headers
    assert "assets" not in body
    assert body["state"]["screenshot"].startswith("data:image/jpeg;base64,")
    assert JevClient(JevCfg(backend="local")).cfg.resolved_url() == "http://127.0.0.1:3000"


def test_typesafe_request_attaches_extra_images_as_assets():
    extra = {"aa-inlay-good": JPEG}
    questions = crate_questions(example_refs={"true": "aa-inlay-good", "false": "bb-inlay-bad"})
    body, _ = JevClient(JevCfg(backend="typesafe")).build_request(JPEG, questions, extra_images=extra)
    assert body["assets"]["aa-inlay-good"]["data_base64"] == base64.b64encode(JPEG).decode("ascii")
    assert body["questions"]["organized"]["criteria"]["true"]["looks_like"] == "aa-inlay-good"
    local = JevClient(JevCfg(backend="local")).build_request(JPEG, crate_questions(), extra_images=extra)[0]
    assert "assets" not in local


def _write_inlay_examples(folder) -> None:
    for side in ("good", "bad"):
        for index in range(1, 4):
            (folder / f"{side}_{index}.jpg").write_bytes(JPEG)


def test_inlay_example_jpegs_need_all_six_files(tmp_path):
    assert inlay_example_jpegs(tmp_path) == {}
    _write_inlay_examples(tmp_path)
    (tmp_path / "bad_3.jpg").unlink()
    assert inlay_example_jpegs(tmp_path) == {}
    (tmp_path / "bad_3.jpg").write_bytes(JPEG)
    examples = inlay_example_jpegs(tmp_path)
    assert set(examples) == {"good", "bad"}
    assert len(examples["good"]) == len(examples["bad"]) == 3


def test_package_ships_the_inlay_reference_photos():
    examples = inlay_example_jpegs()
    assert len(examples["good"]) >= 3 and len(examples["bad"]) >= 3


@pytest.mark.parametrize("confidence", [None, "missing"])
def test_count_confidence_is_optional_because_it_never_decides(confidence):
    answers = _answers(0.9, "one", 0.95, 0.9)
    if confidence is None:
        answers["box_count"]["confidence"] = None
    else:
        del answers["box_count"]["confidence"]
    reading = read_crate(answers, available_spaces={1, 2, 3}, thresholds=RAW)
    assert reading.decision == "open" and reading.space == 2
    assert reading.confidence is None
    assert "conf=n/a" in reading.summary()


@pytest.mark.skipif(not REAL_REPLIES, reason="no saved real Jev replies in rio/_tests/data/jev_replies")
@pytest.mark.parametrize("path", REAL_REPLIES, ids=lambda path: path.name)
def test_saved_real_jev_replies_pass_validation(path):
    response = json.loads(path.read_text())["response"]
    reading = read_crate(parse_reply(response).answers, available_spaces={1, 2, 3})
    assert reading.decision in ("organize", "open", "hold")


def test_local_request_attaches_inlay_example_data_urls():
    examples = {"good": [JPEG, JPEG, JPEG], "bad": [JPEG]}
    body, _ = JevClient(JevCfg(backend="local")).build_request(JPEG, crate_questions(), inlay_examples=examples)
    assert len(body["state"]["inlay_good"]) == 3
    assert body["state"]["inlay_good"][0].startswith("data:image/jpeg;base64,")
    assert len(body["state"]["inlay_bad"]) == 1
    assert "inlay_good" not in JevClient(JevCfg(backend="typesafe")).build_request(JPEG, crate_questions())[0]["state"]


def test_ask_crate_puts_inlay_examples_on_both_backends(tmp_path):
    _write_inlay_examples(tmp_path)
    payload = {"answers": _answers(0.9, "zero", 0.95, 0.1)}

    def run(backend: str) -> dict:
        captured: dict = {}
        client = JevClient(JevCfg(backend=backend))

        def fake_ask(image_jpeg, questions, note="", extra_images=None, inlay_examples=None):
            captured["questions"] = questions
            captured["extra"] = extra_images or {}
            captured["inlay"] = inlay_examples or {}
            return parse_reply(payload)

        client.ask = fake_ask
        ask_crate(client, JPEG, available_spaces={1, 2, 3}, inlay_dir=tmp_path)
        return captured

    typesafe = run("typesafe")
    extra = typesafe["extra"]
    assert len(extra) == 2
    true_ref = typesafe["questions"]["organized"]["criteria"]["true"]["looks_like"]
    false_ref = typesafe["questions"]["organized"]["criteria"]["false"]["looks_like"]
    assert extra[true_ref] and extra[false_ref]
    assert typesafe["inlay"] == {}
    local = run("local")
    assert local["questions"]["organized"]["criteria"]["true"] == ORGANIZED_CRITERIA["true"]
    assert local["extra"] == {}
    assert len(local["inlay"]["good"]) == len(local["inlay"]["bad"]) == 3


def test_an_extra_bad_photo_is_what_the_api_compares_against(tmp_path):
    _write_inlay_examples(tmp_path)
    (tmp_path / "bad_4.jpg").write_bytes(encode_jpeg(np.full((480, 640, 3), 20, dtype=np.uint8)))
    loaded = inlay_example_jpegs(tmp_path)
    assert len(loaded["good"]) == 3
    assert len(loaded["bad"]) == 4

    captured = {}

    def fake_ask(image_jpeg, questions, note="", extra_images=None, inlay_examples=None):
        captured["extra"] = extra_images or {}
        captured["questions"] = questions
        return parse_reply({"answers": _answers(0.1, "zero", 0.9, 0.1), "model": "t"})

    client = JevClient(JevCfg(backend="typesafe"))
    client.ask = fake_ask
    ask_crate(client, JPEG, available_spaces={1, 2, 3}, inlay_dir=tmp_path)
    false_ref = captured["questions"]["organized"]["criteria"]["false"]["looks_like"]
    assert captured["extra"][false_ref] == loaded["bad"][-1]


def test_organized_question_asks_for_an_upright_rib_in_the_middle():
    """The standing check is the central rib: about 90 degrees, in the middle, without a strong slope."""
    text = " ".join((STATE_NOTE, ORGANIZED_INSTRUCTIONS, ORGANIZED_CRITERIA["true"], ORGANIZED_CRITERIA["false"]))
    assert "90 degrees" in text
    assert "middle of the crate" in text
    assert "slanted wall" in text
    assert "slightly off" in text
    assert "open or folded" in text
    assert "open compartment" not in text
    assert "reachable from above" not in text
    assert "both halves" not in text
    assert "both the left" not in text


def test_questions_avoid_bare_integer_option_names_and_name_the_image():
    questions = crate_questions()
    keys = list(questions["box_count"]["criteria"])
    assert keys == list(COUNT_KEYS)
    assert not any(key.isdigit() for key in keys)
    assert all("`crate`" in question["instructions"] for question in questions.values())


def test_typesafe_without_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient(JevCfg(backend="typesafe")).ask(JPEG, crate_questions())


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://", "example.com"])
def test_only_http_urls_are_opened(monkeypatch, url):
    monkeypatch.delenv("SHIM_TOKEN", raising=False)
    with pytest.raises(ValueError, match="http"):
        JevClient(JevCfg(backend="local", url=url)).ask(JPEG, crate_questions())


def test_a_credential_is_never_sent_over_plain_http_to_another_host(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    with pytest.raises(ValueError, match="plain HTTP"):
        JevClient(JevCfg(backend="typesafe", url="http://jev.example.com")).ask(JPEG, crate_questions())
    monkeypatch.setenv("SHIM_TOKEN", "test-token")
    with pytest.raises(ValueError, match="plain HTTP"):
        JevClient(JevCfg(backend="local", url="http://10.0.0.5:3000")).ask(JPEG, crate_questions())


def test_encode_jpeg_keeps_limits():
    big = encode_jpeg(np.zeros((2000, 3000, 3), dtype=np.uint8))
    assert max(Image.open(io.BytesIO(big)).size) == 1568
    with pytest.raises(ValueError, match="minimum"):
        encode_jpeg(np.zeros((200, 640, 3), dtype=np.uint8))


def test_reply_flags_a_text_only_token_count():
    payload = {"model": "jev-1.14.0", "answers": {"a": {"type": "noul", "noul": 0.1}}, "usage": {"input_tokens": 400}}
    assert parse_reply(payload, text_tokens_estimate=350).image_seems_missing()
    payload["usage"]["input_tokens"] = 1400
    assert not parse_reply(payload, text_tokens_estimate=350).image_seems_missing()
    with pytest.raises(JevError):
        parse_reply({"error": {"code": 400, "message": "images off"}})


@pytest.mark.parametrize(
    ("answers", "decision", "space"),
    [
        (_answers(0.1, "three", 0.9, 0.9), "organize", None),
        (_answers(0.5, "zero", 0.95, 0.1), "organize", None),
        (_answers(0.9, "zero", 0.95, 0.1), "open", 1),
        (_answers(0.9, "two", 0.9, 0.95), "open", 3),
        (_answers(0.9, "two", 0.52, 0.95), "open", 3),
        (_answers(0.9, "zero", 0.95, 0.9), "hold", None),
        (_answers(0.9, "three", 0.95, 0.95), "hold", None),
    ],
)
def test_crate_decision(answers, decision, space):
    reading = read_crate(answers, available_spaces={1, 2, 3}, thresholds=RAW)
    assert reading.decision == decision
    assert reading.space == space


def _count_answers(organized: float, probs: dict[str, float], choice: str, any_box: float) -> dict:
    return {
        "view_clear": {"type": "noul", "noul": 0.95},
        "organized": {"type": "noul", "noul": organized},
        "box_count": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.62},
        "any_box": {"type": "noul", "noul": any_box},
    }


def test_count_prior_stays_off_until_it_is_switched_on():
    """0.55 against 0.40 would follow the process count, but the switch defaults to off.

    The photo is a standing inlay and a sure carton. The process has just left one
    carton in the crate. With the prior off the hard line holds. With the prior on
    the same split opens space 2. A photo that already says zero at 0.80 still opens
    space 1, and a split whose two leaders are not neighbors still holds.
    """
    split = _count_answers(0.9, {"one": 0.55, "two": 0.40, "zero": 0.05}, "one", 0.96)
    held = read_crate(split, available_spaces={1, 2, 3}, expected_count=1)
    assert held.decision == "hold"
    assert "count unsure" in held.reason
    limits = CrateThresholds(count_prior=True)
    opened = read_crate(split, available_spaces={1, 2, 3}, thresholds=limits, expected_count=1)
    assert opened.decision == "open"
    assert opened.space == 2
    assert opened.count == 1
    assert "count prior 1" in opened.reason
    sure_zero = _count_answers(0.9, {"zero": 0.80, "one": 0.20}, "zero", 0.05)
    photo = read_crate(sure_zero, available_spaces={1, 2, 3}, thresholds=limits, expected_count=1)
    assert photo.decision == "open"
    assert photo.space == 1
    assert photo.count == 0
    far = _count_answers(0.9, {"zero": 0.50, "two": 0.40, "one": 0.10}, "zero", 0.05)
    assert read_crate(far, available_spaces={1, 2, 3}, thresholds=limits, expected_count=1).decision == "hold"
    outside = read_crate(split, available_spaces={1, 2, 3}, thresholds=limits, expected_count=0)
    assert outside.decision == "hold"


def test_count_prior_does_not_override_the_lead_rule_or_an_empty_process():
    """A lead the photo already accepts stays the photo's count. No finished place means no prior."""
    seen = _count_answers(0.83, {"zero": 0.01, "one": 0.65, "two": 0.30, "three": 0.04}, "one", 0.96)
    limits = CrateThresholds(count_prior=True)
    reading = read_crate(seen, available_spaces={1, 2, 3}, thresholds=limits, expected_count=0)
    assert reading.decision == "open"
    assert reading.space == 2
    assert "count prior" not in reading.reason
    split = _count_answers(0.9, {"one": 0.55, "two": 0.40, "zero": 0.05}, "one", 0.96)
    assert read_crate(split, available_spaces={1, 2, 3}, thresholds=limits).decision == "hold"


def test_a_sure_presence_accepts_a_count_that_leads_its_runner_up():
    """The 18:36 photo: any_box 0.96, one 0.65, two 0.30. That lead opens space 2."""
    seen = _count_answers(0.83, {"zero": 0.01, "one": 0.65, "two": 0.30, "three": 0.04}, "one", 0.96)
    reading = read_crate(seen, available_spaces={1, 2, 3})
    assert reading.decision == "open"
    assert reading.space == 2
    close = _count_answers(0.83, {"one": 0.52, "two": 0.40, "zero": 0.08}, "one", 0.96)
    assert read_crate(close, available_spaces={1, 2, 3}).decision == "hold"
    empty = _count_answers(0.82, {"zero": 1.0}, "zero", 0.01)
    empty_reading = read_crate(empty, available_spaces={1, 2, 3})
    assert empty_reading.decision == "open"
    assert empty_reading.space == 1


def test_count_question_counts_one_low_carton_and_not_the_shelf():
    assert "one object" in COUNT_INSTRUCTIONS
    assert "sits low" in COUNT_INSTRUCTIONS
    assert "shelves" in COUNT_INSTRUCTIONS


def test_trust_inlay_skips_the_rib_and_still_holds_a_blocked_view():
    """After the organize VLA the same 0.1 rib score is not straightened again.

    View and count stay a clear empty crate. Without the flag the arm organizes.
    With the flag it opens space 1. A blocked view still holds before the count.
    """
    spaces = {1, 2, 3}
    collapsed = _answers(0.1, "zero", 0.95, 0.1)
    assert read_crate(collapsed, available_spaces=spaces).decision == "organize"
    trusted = read_crate(collapsed, available_spaces=spaces, trust_inlay=True)
    assert trusted.decision == "open"
    assert trusted.space == 1
    blocked = read_crate(_answers(0.1, "zero", 0.95, 0.1, view_clear=0.1), available_spaces=spaces, trust_inlay=True)
    assert blocked.decision == "hold"
    assert "view" in blocked.reason


def test_a_middling_organized_noul_is_straightened():
    """The question is whether 0.55 still opens a space.

    View and count are held fixed at a clear empty crate. Only the organized
    noul changes. 0.55 is below the yes gate 0.70, so the arm straightens the
    inlay. 0.24 still organizes, 0.90 still opens space 1, and a blocked view
    still holds before the inlay is read.
    """
    spaces = {1, 2, 3}
    limits = CrateThresholds()
    middle = read_crate(_answers(0.55, "zero", 0.95, 0.1), available_spaces=spaces, thresholds=limits)
    assert middle.decision == "organize"
    assert middle.organized_q == pytest.approx(0.55)
    assert middle.organized_p == pytest.approx(0.55)
    assert "not clearly upright" in middle.reason
    sure_no = read_crate(_answers(0.24, "zero", 0.95, 0.1), available_spaces=spaces, thresholds=limits)
    assert sure_no.decision == "organize"
    standing = read_crate(_answers(0.90, "zero", 0.95, 0.1), available_spaces=spaces, thresholds=limits)
    assert standing.decision == "open"
    assert standing.space == 1
    blocked = read_crate(_answers(0.9, "zero", 0.95, 0.1, view_clear=0.1), available_spaces=spaces, thresholds=limits)
    assert blocked.decision == "hold"
    assert "view" in blocked.reason


def test_missing_space_home_holds_with_reason():
    reading = read_crate(_answers(0.9, "three", 0.95, 0.95), available_spaces={1, 2, 3}, thresholds=RAW)
    assert "space 4" in reading.reason


def test_unknown_count_word_is_rejected():
    answers = _answers(0.9, "zero", 0.95, 0.1)
    answers["box_count"]["choice"] = "0"
    with pytest.raises(ValueError, match="box_count"):
        read_crate(answers, available_spaces={1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, None, True, "0.9"])
@pytest.mark.parametrize("field", ["view_clear", "organized", "any_box", "count", "confidence"])
def test_invalid_model_probabilities_never_authorize_an_action(value, field):
    if field == "confidence" and value is None:
        pytest.skip("a null confidence is allowed; see test_count_confidence_is_optional_because_it_never_decides")
    answers = _answers(0.9, "zero", 0.95, 0.1)
    if field == "count":
        answers["box_count"]["probabilities"]["zero"] = value
    elif field == "confidence":
        answers["box_count"]["confidence"] = value
    else:
        answers[field]["noul"] = value
    with pytest.raises(ValueError):
        read_crate(answers, available_spaces={1, 2, 3})


@pytest.mark.parametrize("field", ["view_clear", "organized", "any_box", "box_count"])
def test_missing_visual_observations_are_rejected(field):
    answers = _answers(0.9, "zero", 0.95, 0.1)
    del answers[field]
    with pytest.raises(ValueError):
        read_crate(answers, available_spaces={1})


def test_count_distribution_must_be_normalized_and_include_the_selected_choice():
    answers = _answers(0.9, "zero", 0.95, 0.1)
    answers["box_count"]["probabilities"] = {"zero": 0.8, "one": 0.8}
    with pytest.raises(ValueError, match="sum"):
        read_crate(answers, available_spaces={1})
    answers["box_count"]["probabilities"] = {"one": 1.0}
    with pytest.raises(ValueError, match="selected"):
        read_crate(answers, available_spaces={1})


def test_probe_score_counts_accepted_wrong_counts():
    score = ProbeScore(thresholds=RAW)
    score.add(
        read_crate(_answers(0.9, "two", 0.9, 0.9), available_spaces={1, 2, 3}, thresholds=RAW),
        {"organized": True, "count": 2},
    )
    score.add(
        read_crate(_answers(0.9, "one", 0.9, 0.9), available_spaces={1, 2, 3}, thresholds=RAW),
        {"organized": True, "count": 3},
    )
    score.add(
        read_crate(_answers(0.5, "one", 0.9, 0.9), available_spaces={1, 2, 3}, thresholds=RAW),
        {"organized": False},
    )
    assert (score.organized_right, score.organized_unsure) == (2, 1)
    assert (score.accepted_right, score.accepted_wrong) == (1, 1)
    assert score.confusion[(3, 1)] == 1
    assert "confusion" in score.report()


def test_jev_ask_saves_reply_beside_the_probe(tmp_path):
    image = tmp_path / "p1.jpg"
    image.write_bytes(JPEG)
    (tmp_path / "p1.json").write_text(json.dumps({"organized": True, "count": 0}))
    payload = {"model": "m", "answers": _answers(0.9, "zero", 0.95, 0.1), "usage": {"input_tokens": 1200}}
    reply = parse_reply(payload)
    save_reply(image, "typesafe", reply, read_crate(reply.answers, available_spaces={1}, thresholds=RAW))
    saved = json.loads(reply_path(image, "typesafe").read_text())
    assert saved["reading"]["decision"] == "open"
    assert load_label(image) == {"organized": True, "count": 0}
