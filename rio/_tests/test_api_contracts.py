# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 1: upstream API surface + timing bounds
These guard against rio_hw changing method names
"""

import time as _time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from rio_hw import time as rio_time
from rio_hw.middleware import ClientFactory, ServerFactory, ServerManager
from rio_hw.middlewares import SERVERLESS_MW
from rio_hw.middlewares import __all__ as MW_ALL
from rio_hw.node import Node
from rio_hw.robots._arm import Arm
from rio_hw.robots._gripper import Gripper

from rio.envs import drivers, factory
from rio.envs.env import Env
from rio.policies._policy import Policy
from rio.policies.dummy import Dummy

pytestmark = pytest.mark.unit


def test_enabled_policy_initialization_errors_are_not_silently_disabled(monkeypatch):
    def broken(*args, **kwargs):
        raise AttributeError("broken policy implementation")

    monkeypatch.setattr(factory, "make_policy_node", broken)
    with pytest.raises(AttributeError, match="broken policy"):
        factory.instantiate_station_cfg(SimpleNamespace(policy="Broken"))


FAKE_NODE_MODULE = """
try:
    import rio_missing_driver_module
except ImportError:
    rio_missing_driver_module = None

try:
    from json import dumps as present_driver
except ImportError:
    present_driver = None


def ClickerServer(mw, **kwargs):
    raise AssertionError("the node must not start")


def ClickerClient(mw, **kwargs):
    raise AssertionError("the node must not start")
"""


def test_missing_driver_module_fails_before_any_node_starts(tmp_path, monkeypatch):
    package = tmp_path / "rio_fake_hw"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "interfaces.py").write_text(FAKE_NODE_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(drivers, "CHECKED_PACKAGES", frozenset({"rio_fake_hw"}))

    with pytest.raises(
        ImportError, match=r"Clicker needs the Python module.*rio_missing_driver_module.*--extra kassow"
    ) as error:
        factory.make_node("Thread", "interfaces", "Clicker", {}, package="rio_fake_hw")
    assert "ModuleNotFoundError" in str(error.value)
    assert "json" not in str(error.value)

    monkeypatch.setitem(drivers.OPTIONAL_DRIVER_IMPORTS, "Clicker", frozenset({"rio_missing_driver_module"}))
    server, client = factory.make_node("Thread", "interfaces", "Clicker", {}, package="rio_fake_hw")
    assert callable(server) and callable(client)


def test_driver_guards_are_read_from_the_kassow_node_and_its_kinematics():
    pytest.importorskip("rio_hw.robots.kassow_arm")

    assert ("_kord_bridge", ("_kord_bridge",)) in drivers.guarded_imports("rio_hw.robots.kassow_arm")
    assert ("pinocchio", ("pin",)) in drivers.guarded_imports("rio_hw.robots.kassow_kinematics")
    assert "rio_hw.robots.kassow_kinematics" in drivers.REQUIRED_HELPER_MODULES["KassowArm"]


def test_arm_capability_check_names_every_missing_field():
    from rio.schema import require_state_fields

    require_state_fields({"timestamp": 1.0, "joint_q": np.zeros(7)}, ("timestamp", "joint_q"))
    with pytest.raises(RuntimeError, match="does not publish eef_pose, joint_qd"):
        require_state_fields({"timestamp": 1.0, "eef_pose": None}, ("timestamp", "joint_qd", "eef_pose"))


def test_camera_initialization_errors_are_not_silently_dropped(monkeypatch):
    def broken(*args, **kwargs):
        raise AttributeError("broken camera implementation")

    monkeypatch.setattr(factory, "make_node", broken)
    camera = SimpleNamespace(module="cameras", cam_type="Broken", cfg={})
    with pytest.raises(AttributeError, match="broken camera"):
        factory.make_cameras("Thread", {"camera_1": camera})


@pytest.mark.parametrize("failure", ["embodiment", "camera"])
def test_environment_start_failure_closes_open_clients(monkeypatch, failure):
    closed = []

    @contextmanager
    def client():
        try:
            yield object()
        finally:
            closed.append(True)

    def embodiment(**kwargs):
        if failure == "embodiment":
            raise ValueError("startup failed")
        return object()

    def camera():
        raise ValueError("startup failed")

    monkeypatch.setattr("rio.envs.env.get_mbody_components", lambda **kwargs: ({"arm": client}, embodiment))
    env = Env("Thread", {}, "SINGLE_ARM", camera_clients={"broken": camera} if failure == "camera" else {})
    with pytest.raises(ValueError, match="startup failed"):
        env.start()
    assert closed == [True]
    assert env.robot is None
    assert env.cameras == {}
    assert env._exit_stack is None


def test_arm_api_unchanged():
    for m in ("moveL", "moveJ", "speedL", "speedJ"):
        assert callable(getattr(Arm, m))


def test_gripper_api_unchanged():
    assert callable(Gripper.moveG)


def test_node_api_unchanged():
    for m in ("start", "stop", "pub", "req", "pubreq"):
        assert callable(getattr(Node, m))
    for attr in ("__api__", "__pub__", "__req__"):
        assert attr in Node.__annotations__


def test_middleware_factories_present():
    assert callable(ServerManager) and callable(ServerFactory) and callable(ClientFactory)
    assert {"Shm", "Thread"}.issubset(set(SERVERLESS_MW))
    assert {"Zenoh", "Shm", "Thread"}.issubset(set(MW_ALL))


def test_policy_protocol_surface():
    for m in ("construct_policy", "set_instruction", "inference", "get_action"):
        assert callable(getattr(Policy, m))


def test_legacy_policy_cli_reexports_the_core_runtime():
    from examples import policy_inference
    from rio.policies import rollout

    assert policy_inference.policy_loop is rollout.policy_loop
    assert policy_inference.RolloutControl is rollout.RolloutControl
    assert policy_inference.append_cmd_eef is rollout.append_cmd_eef


@pytest.mark.parametrize(
    ("velocity", "stopped"),
    [(None, False), ([], False), ([float("nan")], False), ([0.2] * 7, False), ([0.0] * 7, True)],
)
def test_plan_completion_requires_measured_joint_standstill(velocity, stopped):
    from rio.policies.rollout import arm_has_stopped

    state = {} if velocity is None else {"joint_qd": velocity}
    assert arm_has_stopped(state) is stopped


def _rollout_args(**arm):
    return SimpleNamespace(
        freq=100,
        policy_node_cfg=SimpleNamespace(action_hz=20, action_dim=7, n_execute=30),
        policy_cfg=SimpleNamespace(),
        instruction="pick",
        arm_cfg=SimpleNamespace(max_pos_speed=0.3, max_rot_speed=0.4, max_motor_speed=0.4, **arm),
    )


@pytest.mark.parametrize("field", ["max_pos_speed", "max_rot_speed", "max_motor_speed"])
def test_command_line_cannot_raise_policy_speeds_above_the_reviewed_ceiling(field):
    from rio.policies.rollout import RolloutSettings

    assert RolloutSettings.from_args(_rollout_args()).max_rot_speed == pytest.approx(0.4)
    args = _rollout_args()
    setattr(args.arm_cfg, field, 5.0)
    with pytest.raises(ValueError, match=field):
        RolloutSettings.from_args(args)


def test_plan_completion_rollouts_need_measured_joint_velocities():
    from rio.policies.rollout import RolloutSettings

    settings = RolloutSettings.from_args(_rollout_args())
    assert "joint_qd" in settings.required_arm_fields(plan_done=True)
    assert "joint_qd" not in settings.required_arm_fields(plan_done=False)


def test_cleanup_failure_never_replaces_the_error_that_ended_the_rollout():
    from rio.arm_resume import ArmAlarm
    from rio.cleanup import cleanup_step

    with pytest.raises(ArmAlarm):
        try:
            raise ArmAlarm(5)
        finally:
            with cleanup_step("save the recording"):
                raise TimeoutError("recorder save timed out")
    with pytest.raises(TimeoutError), cleanup_step("save the recording"):
        raise TimeoutError("without a pending error the cleanup failure surfaces")


def test_run_manifest_identifies_the_code_that_ran():
    from rio.manifest import run_manifest

    manifest = run_manifest(policy_path="ckpts/pi05")
    assert manifest["rio"]["path"].endswith("rio")
    assert manifest["rio_hw"] is not None
    assert manifest["policy_path"] == "ckpts/pi05"
    assert "kord_bridge" in manifest
    assert manifest["rio_hw_pin"] is not None and len(manifest["rio_hw_pin"]) == 40


@pytest.mark.parametrize(
    ("installed", "dirty", "warned"),
    [("a" * 40, False, None), ("b" * 40, False, "pins aaaaaaaaaaaa"), ("a" * 40, True, "uncommitted changes")],
)
def test_rio_hw_pin_check_warns_on_another_commit_or_local_edits(monkeypatch, installed, dirty, warned):
    from rio import manifest

    monkeypatch.setattr(
        manifest.metadata, "requires", lambda name: ["numpy>=1", f"rio_hw @ git+https://example.invalid/rio-hw.git@{'A' * 40}"]
    )
    monkeypatch.setattr(manifest, "package_source", lambda *args: {"git_commit": installed, "git_dirty": dirty})
    assert manifest.pinned_commit("rio", "rio-hw") == "a" * 40
    result = manifest.check_rio_hw_pin()
    assert result is None if warned is None else warned in result


def test_rio_hw_pin_check_is_silent_when_the_commit_is_unknown(monkeypatch):
    from rio import manifest

    monkeypatch.setattr(manifest.metadata, "requires", lambda name: ["rio_hw>=0.1"])
    assert manifest.pinned_commit("rio", "rio_hw") is None
    assert manifest.check_rio_hw_pin() is None


def test_dummy_policy_output_shape():
    policy = Dummy(action_dim=7, chunk_size=8)
    actions = policy.inference({})
    assert actions.shape == (8, 7)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()


def test_rate_period():
    assert rio_time.Rate(100).period == pytest.approx(0.01)
    assert rio_time.Rate(0).period == 0


def test_now_is_monotonic():
    assert rio_time.now_ns() <= rio_time.now_ns()
    assert rio_time.now() > 0


def test_precise_wait_in_past_returns_fast():
    t0 = _time.perf_counter()
    rio_time.precise_wait(rio_time.now() - 1.0)
    assert _time.perf_counter() - t0 < 0.05


def test_precise_sleep_is_bounded():
    t0 = _time.perf_counter()
    rio_time.precise_sleep(0.02)
    elapsed = _time.perf_counter() - t0
    assert 0.015 < elapsed < 0.5  # actually sleeps, but never hangs


def test_precise_rate_skips_missed_ticks_instead_of_emitting_a_burst(monkeypatch):
    clock = [1.0]
    waits = []
    monkeypatch.setattr(rio_time, "now", lambda: clock[0])
    monkeypatch.setattr(rio_time, "precise_wait", waits.append)
    rate = rio_time.Rate(100)
    rate.start_time = 0.0
    rate.precise_sleep()
    assert rate.start_time == 1.0
    assert waits == []
    clock[0] = 1.001
    rate.precise_sleep()
    assert waits == [pytest.approx(1.01)]
