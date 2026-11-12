# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""``rio.policies.serve``: resident lifecycle and DAgger hooks without hardware."""

from contextlib import nullcontext
from dataclasses import dataclass, fields
from types import SimpleNamespace

import pytest

from rio.policies import serve
from rio.policies.rollout import RolloutSafetyCfg

pytestmark = pytest.mark.unit


class ResidentArgs:
    """Station config with a resident policy; records the lifecycle calls."""

    def __init__(self, calls: list, *, unload_policy: bool = False, dagger_round: int | None = None):
        self.calls = calls
        self.unload_policy = unload_policy
        self.dagger_round = dagger_round
        self.mw = "Thread"

    def start_resident(self):
        self.calls.append("start_resident")
        return SimpleNamespace(pid=123)

    def stop_resident(self) -> bool:
        self.calls.append("stop_resident")
        return True


@pytest.fixture
def station(monkeypatch):
    """Fake ``make_env``/``ServerManager``/``policy_loop`` that record the order of events."""
    calls: list = []

    def make_env(args):
        calls.append("make_env")
        clients = {"policy": lambda: nullcontext("policy"), "visualizer": None}
        return {}, clients, nullcontext("env")

    def policy_loop(args, env, policy, visualizer=None, dagger=None):
        calls.append(("policy_loop", dagger))

    monkeypatch.setattr(serve, "make_env", make_env)
    monkeypatch.setattr(serve, "ServerManager", lambda mw, servers: nullcontext())
    monkeypatch.setattr(serve, "policy_loop", policy_loop)
    return calls


def test_unload_stops_only_the_resident_and_never_builds_the_station(station):
    serve.serve(ResidentArgs(station, unload_policy=True))
    assert station == ["stop_resident"]


def test_unload_without_a_resident_process_is_refused(station):
    with pytest.raises(ValueError, match="no resident"):
        serve.serve(SimpleNamespace(unload_policy=True))
    assert station == []


def test_dagger_round_needs_an_entry_point_that_provides_takeover(station):
    with pytest.raises(ValueError, match="does not support DAgger"):
        serve.serve(ResidentArgs(station, dagger_round=1))
    assert station == []


def test_resident_is_loaded_before_the_station_starts(station):
    serve.serve(ResidentArgs(station))
    assert station == ["start_resident", "make_env", ("policy_loop", None)]


def test_dagger_session_wraps_the_rollout(station):
    serve.serve(ResidentArgs(station, dagger_round=2), open_dagger=lambda args, env, clients: nullcontext("dagger"))
    assert station[-1] == ("policy_loop", "dagger")


def test_serve_args_adds_the_rollout_options_and_keeps_the_station_doc():
    @dataclass
    class Station:
        """Some station."""

        mw: str = "Thread"

    extended = serve.serve_args(Station)
    assert {"diag_dir", "unload_policy", "safety_cfg"} <= {field.name for field in fields(extended)}
    assert extended.__doc__ == "Some station."
    args = extended()
    assert isinstance(args, Station) and isinstance(args.safety_cfg, RolloutSafetyCfg) and not args.unload_policy
