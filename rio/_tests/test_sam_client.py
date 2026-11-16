# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""SAM client: key lookup, request body, and parsing of the token grammar. No network."""

import pytest

from rio.sam import SamClient, SamError, api_key, parse_frames
from rio.sam.client import SamReply, output_text

pytestmark = pytest.mark.unit

LINE = (
    "<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561;c=0.93|><|mask;x=0;y=0;c=0.93;data=46,101,~!!F)|>,"
    "1<|box;x1=260;y1=10;x2=299;y2=59;w=537;h=561;c=0.41|><|mask;x=0;y=0;data=50,40,!abc|>"
)


def test_parse_frames_reads_boxes_scores_and_masks():
    (instances,) = parse_frames(LINE).values()
    first, second = instances
    assert first.index == 0 and first.box == (353, 382, 453, 427) and first.size == (537, 561)
    assert first.score == pytest.approx(0.93)
    assert first.mask is not None and first.mask.endswith("data=46,101,~!!F)")
    assert second.index == 1 and second.score == pytest.approx(0.41)
    left, top, right, bottom = first.fractions()
    assert (left, right) == (pytest.approx(353 / 537), pytest.approx(454 / 537))
    assert (top, bottom) == (pytest.approx(382 / 561), pytest.approx(428 / 561))
    assert SamReply(instances, LINE, 0.1).best() is first


def test_parse_frames_without_matches_is_empty():
    assert parse_frames("") == {}
    assert SamReply([], "", 0.1).best() is None


def test_unreadable_box_raises():
    with pytest.raises(SamError):
        parse_frames("<0f>0<|box;x1=1;y1=2|>")


def test_api_key_prefers_env_then_file(tmp_path):
    path = tmp_path / "sam.env"
    assert api_key({"SAM_API_KEY": " k1 "}, path) == "k1"
    path.write_text("# comment\nexport SAM_API_KEY='k2'\n")
    assert api_key({}, path) == "k2"
    path.write_text("OTHER=1\n")
    with pytest.raises(SamError, match="no SAM key"):
        api_key({}, path)


def test_request_body_puts_text_first_and_asks_for_scores():
    body = SamClient(key="k").request_body(b"jpeg", "orange plate")
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "orange plate"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert body["metadata"]["include_confidence"] == "true" and body["store"] is False


def test_output_text_reads_responses_payload():
    payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": LINE}]}]}
    assert output_text(payload) == LINE
