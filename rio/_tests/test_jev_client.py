# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Jev request shapes, crate decisions, and probe scoring. No network and no camera."""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from examples.jev_ask import load_label, reply_path, save_reply
from rio.jev import (
    CrateThresholds,
    JevCfg,
    JevClient,
    JevError,
    ProbeScore,
    ask_crate,
    bayes_posterior,
    crate_questions,
    encode_jpeg,
    read_crate,
)
from rio.jev.client import IMAGES_HEADER, parse_reply
from rio.jev.crate import COUNT_KEYS, ORGANIZED_CRITERIA, count_questions, inlay_example_jpegs

pytestmark = pytest.mark.unit

JPEG = encode_jpeg(np.full((480, 640, 3), 90, dtype=np.uint8))


def _answers(organized: float, count: str, count_p: float, any_box: float, view_clear: float = 0.95) -> dict:
    rest = (1.0 - count_p) / (len(COUNT_KEYS) - 1)
    probs = {key: (count_p if key == count else rest) for key in COUNT_KEYS}
    return {
        "view_clear": {"type": "noul", "noul": view_clear},
        "organized": {"type": "noul", "noul": organized},
        "box_count": {"type": "choice", "choice": count, "probabilities": probs, "confidence": 0.9},
        "any_box": {"type": "noul", "noul": any_box},
    }


# Isolate noul thresholds from the 90 % standing prior used in production.
RAW = CrateThresholds(organized_prior=0.5)


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


def test_count_only_skips_the_inlay_and_opens_on_the_count():
    answers = _answers(0.1, "one", 0.95, 0.9)
    del answers["organized"]
    del answers["view_clear"]
    reading = read_crate(answers, available_spaces={1, 2, 3}, count_only=True)
    assert reading.decision == "open"
    assert reading.space == 2
    assert set(count_questions()) == {"box_count", "any_box"}


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
        (_answers(0.5, "zero", 0.95, 0.1), "hold", None),
        (_answers(0.9, "zero", 0.95, 0.1), "open", 1),
        (_answers(0.9, "two", 0.9, 0.95), "open", 3),
        (_answers(0.9, "two", 0.5, 0.95), "hold", None),
        (_answers(0.9, "zero", 0.95, 0.9), "hold", None),
        (_answers(0.9, "three", 0.95, 0.95), "hold", None),
    ],
)
def test_crate_decision(answers, decision, space):
    reading = read_crate(answers, available_spaces={1, 2, 3}, thresholds=RAW)
    assert reading.decision == decision
    assert reading.space == space


def test_bayes_posterior_is_identity_at_half_prior():
    assert bayes_posterior(0.62, 0.5) == pytest.approx(0.62)
    assert bayes_posterior(0.62, 0.9) == pytest.approx(0.936, abs=0.01)
    assert bayes_posterior(0.01, 0.9) == pytest.approx(0.083, abs=0.01)


def test_standing_prior_promotes_a_weak_yes_but_not_a_sure_no():
    spaces = {1, 2, 3}
    prior = CrateThresholds(organized_prior=0.9)
    weak_yes = read_crate(_answers(0.62, "zero", 0.95, 0.1), available_spaces=spaces, thresholds=prior)
    assert weak_yes.decision == "open"
    assert weak_yes.organized_q == pytest.approx(0.62)
    assert weak_yes.organized_p > 0.9
    sure_no = read_crate(_answers(0.24, "zero", 0.95, 0.1), available_spaces=spaces, thresholds=prior)
    assert sure_no.decision == "organize"
    blocked = read_crate(_answers(0.9, "zero", 0.95, 0.1, view_clear=0.1), available_spaces=spaces, thresholds=prior)
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
