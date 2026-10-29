# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Named Kassow pick/sort experiment switch (overview camera + home)."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.cfg.kassow_experiments import (
    EXPERIMENTS,
    WRIST_SERIAL,
    apply_kassow_experiment,
    resolve_experiment,
)
from examples.cfg.kassow_vive_gripper import KassowViveGripperStation
from examples.teleop_vive_hand import home_q_path, load_home_state

pytestmark = pytest.mark.unit


def test_resolve_experiment_rejects_unknown_name():
    with pytest.raises(ValueError, match="experiment must be one of"):
        resolve_experiment("align")


def test_sort_is_the_default_and_uses_the_new_overview():
    cfg = KassowViveGripperStation()
    assert cfg.experiment == "sort"
    assert cfg.cameras["camera_1"].serial == WRIST_SERIAL
    assert cfg.cameras["camera_2"].serial == EXPERIMENTS["sort"].overview_serial
    assert cfg.home_file == str(EXPERIMENTS["sort"].home_path)
    assert cfg.recorder_cfg.path == EXPERIMENTS["sort"].recorder_path


def test_pick_restores_the_old_overview_and_home():
    cfg = KassowViveGripperStation(experiment="pick")
    assert cfg.cameras["camera_1"].serial == WRIST_SERIAL
    assert cfg.cameras["camera_2"].serial == EXPERIMENTS["pick"].overview_serial
    assert cfg.home_file == str(EXPERIMENTS["pick"].home_path)
    assert cfg.recorder_cfg.path == EXPERIMENTS["pick"].recorder_path


def test_overview_serial_and_home_file_overrides_win():
    cfg = KassowViveGripperStation(
        experiment="sort",
        overview_serial="111111111111",
        home_file="/tmp/custom_home.json",
    )
    assert cfg.cameras["camera_2"].serial == "111111111111"
    assert cfg.home_file == "/tmp/custom_home.json"


def test_custom_recorder_path_is_not_clobbered():
    cfg = KassowViveGripperStation()
    cfg.recorder_cfg.path = "data/custom_sort/"
    apply_kassow_experiment(cfg)
    assert cfg.recorder_cfg.path == "data/custom_sort/"


def test_home_q_path_prefers_home_file_then_experiment():
    explicit = SimpleNamespace(home_file="examples/cfg/kassow_homes/home_sort.json", experiment="pick")
    explicit_path = home_q_path(explicit)
    assert explicit_path is not None
    assert explicit_path.name == "home_sort.json"
    named = SimpleNamespace(home_file=None, experiment="pick", recorder_cfg=None)
    assert home_q_path(named) == EXPERIMENTS["pick"].home_path


def test_named_homes_load():
    pick = load_home_state(EXPERIMENTS["pick"].home_path, gripper_fallback=0.0)
    sort = load_home_state(EXPERIMENTS["sort"].home_path, gripper_fallback=0.0)
    np.testing.assert_allclose(pick["eef_pose"][:3], [-0.435488928244587, 0.46201706668887504, 0.3196436546871914])
    assert pick["gripper"] == pytest.approx(0.7)
    np.testing.assert_allclose(sort["eef_pose"][:3], [-0.20252609914780512, 0.35204102135298376, 0.4459602795253897])
    assert sort["joint_q"].shape == (7,)
