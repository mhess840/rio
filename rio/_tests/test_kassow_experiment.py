# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Kassow experiment switch: cameras, homes, workspace, and model binding."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.cfg.kassow_experiments import (
    EXPERIMENTS,
    JEV_CAMERA_ADDR,
    JEV_SERIAL,
    WRIST_SERIAL,
    active_experiment_name,
    apply_kassow_experiment,
    find_model_owner,
    jev_camera,
    load_state,
    model_dataset_mismatch,
    open_spaces,
    resolve_experiment,
    resolve_model,
    save_state,
    set_active_experiment,
)
from examples.cfg.kassow_vive_gripper import KassowViveGripperStation
from examples.data.convert_to_lerobot_pi05 import ConvertExperimentCfg, resolve_experiment_paths
from examples.experiment import main as experiment_cli
from examples.goto_kassow_q import goto_home, home_owner, load_pose
from examples.policy_cfgs.pi05_lerobot import Pi05LerobotCfg
from examples.teleop_vive_hand import home_q_path, load_home_state, write_home_state
from rio.data.convert_pi05 import binarize_action_gripper

pytestmark = pytest.mark.unit


def _register_model(experiment_name, tmp_path, name="run1", instructions=("put the object in space 1",)):
    experiment = EXPERIMENTS[experiment_name]
    target = tmp_path / f"ckpt_{experiment_name}_{name}"
    target.mkdir()
    (target / "config.json").write_text("{}")
    experiment.models_dir.mkdir(parents=True, exist_ok=True)
    (experiment.models_dir / name).symlink_to(target)
    state = load_state(experiment)
    state["models"][name] = {"dataset": "spaces_notrim", "instructions": list(instructions)}
    state["model"] = name
    save_state(experiment, state)
    return target


def test_resolve_experiment_rejects_unknown_name():
    with pytest.raises(ValueError, match="experiment must be one of"):
        resolve_experiment("align")


def test_sort_is_the_default_and_uses_the_new_overview():
    cfg = KassowViveGripperStation()
    assert cfg.experiment == "sort"
    assert cfg.cameras["camera_1"].serial == WRIST_SERIAL
    assert cfg.cameras["camera_2"].serial == "261222078043"
    assert cfg.home_file == str(EXPERIMENTS["sort"].home_path)
    assert cfg.recorder_cfg.path == str(EXPERIMENTS["sort"].raw_dir)


def test_pick_restores_the_old_overview_and_home():
    cfg = KassowViveGripperStation(experiment="pick")
    assert cfg.cameras["camera_1"].serial == WRIST_SERIAL
    assert cfg.cameras["camera_2"].serial == "262522075082"
    assert cfg.home_file == str(EXPERIMENTS["pick"].home_path)
    assert cfg.recorder_cfg.path == str(EXPERIMENTS["pick"].raw_dir)


def test_active_experiment_is_the_default():
    set_active_experiment("organize")
    assert active_experiment_name() == "organize"
    cfg = KassowViveGripperStation()
    assert cfg.experiment == "organize"
    assert cfg.home_file == str(EXPERIMENTS["organize"].home_path)
    assert cfg.recorder_cfg.path == str(EXPERIMENTS["organize"].raw_dir)
    assert EXPERIMENTS["organize"].raw_dir.is_dir()


def test_frozen_experiment_refuses_to_record_but_teleops_without_recorder():
    experiment = EXPERIMENTS["pick"]
    state = load_state(experiment)
    state["frozen"] = True
    save_state(experiment, state)
    with pytest.raises(ValueError, match="frozen"):
        KassowViveGripperStation(experiment="pick")
    cfg = KassowViveGripperStation(experiment="pick", recorder=None)
    assert cfg.cameras["camera_2"].serial == "262522075082"


def test_broken_raw_link_is_not_replaced_by_an_empty_folder(tmp_path):
    experiment = EXPERIMENTS["organize"]
    experiment.root.mkdir(parents=True)
    experiment.raw_dir.symlink_to(tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        KassowViveGripperStation(experiment="organize")


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


def test_named_homes_load_and_keep_their_poses():
    pick = load_home_state(EXPERIMENTS["pick"].home_path, gripper_fallback=0.0, experiment="pick")
    sort = load_home_state(EXPERIMENTS["sort"].home_path, gripper_fallback=0.0, experiment="sort")
    np.testing.assert_allclose(pick["eef_pose"][:3], [-0.435488928244587, 0.46201706668887504, 0.3196436546871914])
    assert pick["gripper"] == pytest.approx(0.7)
    np.testing.assert_allclose(sort["eef_pose"][:3], [-0.20252609914780512, 0.35204102135298376, 0.4459602795253897])
    np.testing.assert_allclose(sort["joint_q"][6], 1.6048102849336678)


def test_home_of_another_experiment_is_rejected(tmp_path):
    path = tmp_path / "home.json"
    write_home_state(path, {"joint_q": np.zeros(7), "eef_pose": np.zeros(6), "gripper": 0.7}, experiment="sort")
    assert json.loads(path.read_text())["name"] == "sort"
    with pytest.raises(ValueError, match="home of 'sort'"):
        load_home_state(path, gripper_fallback=0.0, experiment="organize")


def test_resolve_model_uses_the_selected_model(tmp_path):
    target = _register_model("sort", tmp_path)
    name, path = resolve_model(EXPERIMENTS["sort"])
    assert name == "run1"
    assert path.resolve() == target.resolve()
    owner = find_model_owner(target)
    assert owner is not None
    assert owner[:2] == ("sort", "run1")


def test_resolve_model_without_selection_explains_the_fix():
    with pytest.raises(FileNotFoundError, match=r"examples\.experiment use organize --model"):
        resolve_model(EXPERIMENTS["organize"])


def test_serve_cfg_takes_model_and_instruction_from_the_experiment(tmp_path):
    target = _register_model("sort", tmp_path)
    cfg = Pi05LerobotCfg(experiment="sort")
    assert cfg.policy_path == str(EXPERIMENTS["sort"].models_dir / "run1")
    assert cfg.policy_cfg.policy_path == cfg.policy_path
    assert cfg.instruction == "put the object in space 1"
    assert cfg.policy_node_cfg.instruction == "put the object in space 1"
    assert cfg.cameras["camera_2"].serial == "261222078043"
    explicit = Pi05LerobotCfg(experiment="sort", policy_path=str(target), instruction="put the object in space 1")
    assert explicit.instruction == "put the object in space 1"


def test_serve_cfg_rejects_untrained_instruction(tmp_path):
    _register_model("sort", tmp_path)
    with pytest.raises(ValueError, match="not a task of this model"):
        Pi05LerobotCfg(experiment="sort", instruction="pick up the blue object and place it on the red plate")
    cfg = Pi05LerobotCfg(experiment="sort", instruction="sort it", allow_unlisted_instruction=True)
    assert cfg.instruction == "sort it"


def test_serve_cfg_rejects_a_model_of_another_experiment(tmp_path):
    target = _register_model("pick", tmp_path, name="pick_cyl", instructions=("pick up the blue object",))
    with pytest.raises(ValueError, match="experiment 'pick'"):
        Pi05LerobotCfg(experiment="sort", policy_path=str(target))


def test_unregistered_checkpoint_needs_an_instruction(tmp_path):
    with pytest.raises(ValueError, match="pass --instruction"):
        Pi05LerobotCfg(experiment="sort", policy_path=str(tmp_path))


def test_convert_variant_resolves_paths_from_the_experiment():
    experiment = EXPERIMENTS["organize"]
    experiment.raw_dir.mkdir(parents=True)
    (experiment.raw_dir / "tasks_cycle.yaml").write_text("0: put it away\n")
    set_active_experiment("organize")
    cfg = resolve_experiment_paths(ConvertExperimentCfg(variant="crop", tasks=Path("tasks_cycle.yaml")))
    assert cfg.experiment == "organize"
    assert cfg.input == experiment.raw_dir
    assert cfg.output == experiment.lerobot_dir / "crop"
    assert cfg.tasks == experiment.raw_dir / "tasks_cycle.yaml"
    assert cfg.repo_id == "local/kassow_organize_crop"


def test_convert_variant_never_targets_a_linked_dataset(tmp_path):
    experiment = EXPERIMENTS["organize"]
    experiment.raw_dir.mkdir(parents=True)
    experiment.lerobot_dir.mkdir(parents=True)
    (experiment.lerobot_dir / "old").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="linked existing dataset"):
        resolve_experiment_paths(ConvertExperimentCfg(experiment="organize", variant="old"))
    with pytest.raises(ValueError, match="needs --variant"):
        resolve_experiment_paths(ConvertExperimentCfg(experiment="organize"))


def test_model_dataset_mismatch_reads_safetensors_stats(tmp_path):
    header = {
        "action.q01": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "action.q99": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
    }
    blob = json.dumps(header).encode()
    payload = np.array([0.1, 0.2, 0.9, 1.0], dtype=np.float32).tobytes()
    model = tmp_path / "model"
    model.mkdir()
    (model / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        len(blob).to_bytes(8, "little") + blob + payload
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    stats = {"action": {"q01": [0.1, 0.2], "q99": [0.9, 1.0]}}
    (dataset / "meta" / "stats.json").write_text(json.dumps(stats))
    diff = model_dataset_mismatch(model, dataset)
    assert diff is not None
    assert diff < 1e-6
    stats["action"]["q99"] = [0.9, 1.5]
    (dataset / "meta" / "stats.json").write_text(json.dumps(stats))
    assert model_dataset_mismatch(model, dataset) == pytest.approx(0.5)


def _checkpoint(folder: Path, q01: list[float]) -> Path:
    header = {"action.q01": {"dtype": "F32", "shape": [len(q01)], "data_offsets": [0, 4 * len(q01)]}}
    blob = json.dumps(header).encode()
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        len(blob).to_bytes(8, "little") + blob + np.asarray(q01, dtype=np.float32).tobytes()
    )
    return folder


def test_add_model_registers_a_copied_checkpoint_in_place_or_links_one_elsewhere(tmp_path):
    sort = EXPERIMENTS["sort"]
    dataset = sort.lerobot_dir / "dagger_r1"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "stats.json").write_text(json.dumps({"action": {"q01": [0.1, 0.2]}}))
    pq.write_table(pa.table({"task_index": [0], "task": ["put the object in space 2"]}), dataset / "meta" / "tasks.parquet")

    copied = _checkpoint(sort.models_dir / "dagger_r1", [0.1, 0.2])
    experiment_cli(["add-model", "dagger_r1", str(copied), "--experiment", "sort"])
    assert copied.is_dir() and not copied.is_symlink()
    notes = load_state(sort)["models"]["dagger_r1"]
    assert notes["dataset"] == "dagger_r1"
    assert notes["instructions"] == ["put the object in space 2"]

    elsewhere = _checkpoint(tmp_path / "outputs" / "pretrained_model", [0.1, 0.2])
    experiment_cli(["add-model", "dagger_r1_4k", str(elsewhere), "--experiment", "sort"])
    assert (sort.models_dir / "dagger_r1_4k").resolve() == elsewhere.resolve()
    with pytest.raises(FileExistsError):
        experiment_cli(["add-model", "dagger_r1", str(elsewhere), "--experiment", "sort"])


SPACE1 = "put the object in space 1"
SPACE2 = "put the object in space 2"


def test_organize_discretizes_open_at_0_7():
    cfg = KassowViveGripperStation(experiment="organize")
    assert cfg.gripper_open == pytest.approx(0.7)
    assert cfg.gripper_close == pytest.approx(0.0)
    raw = EXPERIMENTS["organize"].raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    converted = resolve_experiment_paths(
        ConvertExperimentCfg(experiment="organize", variant="gripper_probe", output=raw.parent / "_gripper_probe")
    )
    assert converted.gripper_open == pytest.approx(0.7)
    assert converted.gripper_close == pytest.approx(0.0)
    snapped = binarize_action_gripper(
        np.array([0.0, 0.3, 0.7], dtype=np.float32),
        gripper_close=converted.gripper_close,
        gripper_open=converted.gripper_open,
    )
    assert snapped.tolist() == pytest.approx([0.0, 0.0, 0.7])


def test_open_gripper_stays_on_this_experiment():
    cfg = KassowViveGripperStation(experiment="open", instruction=SPACE1)
    assert cfg.gripper_close == pytest.approx(0.0)
    assert cfg.gripper_open == pytest.approx(0.3)
    assert cfg.gripper_start == pytest.approx(0.0)
    assert cfg.gripper_cfg.startup_position == pytest.approx(0.0)
    overridden = KassowViveGripperStation(experiment="open", instruction=SPACE1, gripper_open=0.55)
    assert overridden.gripper_open == pytest.approx(0.55)
    policy = Pi05LerobotCfg(experiment="open", instruction=SPACE1, policy_path="ckpts/pi05")
    assert policy.policy_cfg.gripper_open == pytest.approx(0.3)
    assert policy.policy_cfg.gripper_close == pytest.approx(0.0)
    assert policy.gripper_cfg.startup_position is None
    sort = KassowViveGripperStation(experiment="sort")
    assert sort.gripper_open == pytest.approx(0.7)
    assert sort.gripper_cfg.startup_position == pytest.approx(0.7)
    raw = EXPERIMENTS["open"].raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    converted = resolve_experiment_paths(
        ConvertExperimentCfg(experiment="open", variant="gripper_probe", output=raw.parent / "_gripper_probe")
    )
    assert converted.gripper_open == pytest.approx(0.3)
    assert converted.gripper_close == pytest.approx(0.0)


def test_open_home_follows_the_instruction_and_uses_the_sort_camera():
    space1 = KassowViveGripperStation(experiment="open", instruction=SPACE1)
    space2 = KassowViveGripperStation(experiment="open", instruction=SPACE2)
    assert space1.cameras["camera_2"].serial == "261222078043"
    assert space1.home_file.endswith("home_open_space1.json")
    assert space2.home_file.endswith("home_open_space2.json")
    assert space1.recorder_cfg.path == str(EXPERIMENTS["open"].raw_dir)
    assert home_owner(Path(space1.home_file)) == ("open", SPACE1)
    assert home_owner(Path(space2.home_file)) == ("open", SPACE2)
    with pytest.raises(ValueError, match="Pass --instruction"):
        KassowViveGripperStation(experiment="open")


def test_playback_does_not_need_a_pose_on_open():
    @dataclass
    class Playback(KassowViveGripperStation):
        recorder: str | None = None

        def __post_init__(self) -> None:
            self.bind_home = False
            KassowViveGripperStation.__post_init__(self)

    cfg = Playback(experiment="open")
    assert not cfg.home_file
    assert cfg.cameras["camera_2"].serial == "261222078043"
    assert cfg.gripper_open == pytest.approx(0.3)


def test_open_pose_name_selects_the_sentence_and_disagreements_are_refused():
    @dataclass
    class StationWithPose(KassowViveGripperStation):
        pose: str = "home"

    cfg = StationWithPose(experiment="open", pose="space2", recorder=None)
    assert cfg.instruction == SPACE2
    assert cfg.home_file.endswith("home_open_space2.json")
    with pytest.raises(ValueError, match="space1"):
        EXPERIMENTS["open"].select_home(SPACE2, "space1")
    with pytest.raises(ValueError, match="has one home per instruction"):
        _ = EXPERIMENTS["open"].home_path


def test_home_file_keeps_the_sentence_it_was_captured_for(tmp_path):
    path = tmp_path / "home.json"
    write_home_state(
        path,
        {"joint_q": np.zeros(7), "eef_pose": np.zeros(6), "gripper": 0.7},
        experiment="open",
        instruction=SPACE1,
    )
    loaded = load_home_state(path, gripper_fallback=0.0, experiment="open", instruction=SPACE1)
    assert loaded["gripper"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="space 1"):
        load_home_state(path, gripper_fallback=0.0, experiment="open", instruction=SPACE2)


def test_jev_camera_is_not_a_policy_camera():
    cfg = KassowViveGripperStation(recorder=None)
    assert "camera_jev" not in cfg.cameras
    assert set(cfg.cameras) == {"camera_1", "camera_2"}
    camera = jev_camera()
    assert camera.serial == JEV_SERIAL
    assert camera.addr == JEV_CAMERA_ADDR
    assert camera.cam_type == "Realsense"
    assert camera.enable_depth is False
    assert camera.resolution == (480, 640)
    assert jev_camera("111111111111").serial == "111111111111"


def test_open_spaces_are_the_recorded_homes():
    spaces = open_spaces()
    assert set(spaces) == {1, 2, 3}
    assert spaces[1].pose == "space1"
    assert spaces[1].instruction == SPACE1
    assert spaces[1].path.is_file()


def test_goto_home_is_true_when_already_there_and_false_on_alarm(monkeypatch):
    monkeypatch.setattr("examples.goto_kassow_q.time.sleep", lambda _duration: None)
    home = load_pose(EXPERIMENTS["open"].homes[0].path, 0.3, "open")

    class _Arm:
        def __init__(self, q, alarm=0):
            self.q = np.asarray(q, dtype=float)
            self.alarm = alarm
            self.moves = 0

        def get_state(self):
            return {
                "joint_q": self.q,
                "joint_qd": np.zeros(7),
                "eef_pose": np.zeros(6),
                "alarm_code": self.alarm,
            }

        def moveJ_discrete(self, q, _duration):
            self.moves += 1
            self.q = np.asarray(q, dtype=float)

    at_home = SimpleNamespace(robot=SimpleNamespace(arm=_Arm(home["joint_q"])))
    assert goto_home(at_home, home, home_joint_speed=0.15) is True
    assert at_home.robot.arm.moves == 0

    alarmed = SimpleNamespace(robot=SimpleNamespace(arm=_Arm(home["joint_q"] + 0.2, alarm=1)))
    assert goto_home(alarmed, home, home_joint_speed=0.15) is False
    assert alarmed.robot.arm.moves == 1
