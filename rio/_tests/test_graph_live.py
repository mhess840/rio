# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Live view feed: tailing logs, picking plans and runs, safe file access. No browser."""

import json
import os

import pytest

from rio.graph.live import Feed, Stream, Tail, export, page

pytestmark = pytest.mark.unit


def test_tail_returns_only_complete_new_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    tail = Tail(path)
    assert tail.read() == []
    path.write_text('{"a": 1}\n{"b"')
    assert tail.read() == [{"a": 1}]
    with path.open("a") as handle:
        handle.write(': 2}\nnot json\n{"c": 3}\n')
    assert tail.read() == [{"b": 2}, {"c": 3}]
    path.write_text('{"d": 4}\n')  # rewritten shorter: start over
    assert tail.read() == [{"d": 4}]


def make_root(tmp_path):
    for name in ("20261116_100000", "20261116_110000"):
        run = tmp_path / name
        run.mkdir()
        (run / "workflow.json").write_text(json.dumps({"version": 3, "meta": {"name": name}}))
        (run / "events.jsonl").write_text(
            json.dumps({"event": "tool_end", "node": "cube/find", "tool": "jev.locate", "outputs": {"exit": "found"}}) + "\n"
        )
    (tmp_path / "20261116_110000" / "cube__find.jpg").write_bytes(b"jpg")
    (tmp_path / "look").mkdir()
    for task, stamp in (("old", 1_000), ("new", 2_000)):
        folder = tmp_path / "plans" / task
        folder.mkdir(parents=True)
        (folder / "plan.jsonl").write_text(json.dumps({"event": "task", "task": task}) + "\n")
        os.utime(folder / "plan.jsonl", (stamp, stamp))
    return tmp_path


def test_feed_picks_newest_unless_pinned(tmp_path):
    feed = Feed(make_root(tmp_path))
    assert feed.runs() == ["20261116_100000", "20261116_110000"]
    assert feed.plans() == ["new", "old"]
    assert feed.run(None).name == "20261116_110000"
    assert feed.run("20261116_100000").name == "20261116_100000"
    assert feed.run("../etc").name == "20261116_110000"
    assert feed.plan("old").name == "old"


def test_feed_serves_only_media_inside_root(tmp_path):
    feed = Feed(make_root(tmp_path))
    assert feed.file("20261116_110000/cube__find.jpg") is not None
    assert feed.file("20261116_110000/workflow.json") is not None
    assert feed.file("../secret.jpg") is None
    assert feed.file("%2e%2e/secret.jpg") is None
    assert feed.file("plans/new/plan.jsonl") is None


def test_stream_sends_resets_then_lines_with_photos(tmp_path):
    root = make_root(tmp_path)
    stream = Stream(Feed(root))
    kinds = [m["kind"] for m in stream.poll()]
    assert kinds == ["plan_reset", "plan", "plan_workflow", "run_reset", "run"]
    assert stream.poll() == []
    with (root / "plans" / "new" / "plan.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "decompose", "stages": []}) + "\n")
    (msg,) = stream.poll()
    assert msg == {"kind": "plan", "event": {"event": "decompose", "stages": []}}
    run = Stream(Feed(root)).poll()
    photo = next(m for m in run if m["kind"] == "run")["photo"]
    assert photo == "20261116_110000/cube__find.jpg"


def test_inline_page_embeds_state_safely(tmp_path):
    html = page([{"kind": "plan", "event": {"task": "</script><b>x"}}]).decode()
    assert "window.INITIAL" in html and "</script><b>" not in html
    out = export(make_root(tmp_path), tmp_path / "replay.html")
    assert "20261116_110000" in out.read_text()
