# Kassow π₀.₅ pipeline

Experiments, recording, labels, conversion, training, inference, the Jev crate orchestrator, and troubleshooting for the Kassow + AG95 + 2× RealSense station. This is not the xArm / OpenPI DROID path (`Pi05Cfg`, [DROID fine-tuning](pi0_droid_finetuning.md)).

Control is Cartesian (`task_pos_ik`): the policy outputs an absolute TCP pose, and the arm node turns it into joint velocities. The policy's proprioceptive input is the seven joint angles plus the gripper, not the TCP.

```text
Vive teleop ──► experiments/<exp>/raw/traj_XXXX.vla        (100 Hz, robodm)
                  + raw/tasks*.yaml                        (labels, written afterwards)
                        │  examples.data.convert_to_lerobot_pi05 --variant <v>
                        ▼
                experiments/<exp>/lerobot/<v>/             (LeRobot v3, 20 Hz)
                        │  lerobot-train (cluster)
                        ▼
                experiments/<exp>/models/<m>/              (pretrained_model)
                        │  examples.policy_inference
                        ▼
                Kassow, TCP targets at 20 Hz, control loop 100 Hz
                        │  POLICY=Pi05DaggerCfg: operator takeover (section 7)
                        ▼
                experiments/<exp>/dagger/r<N>/traj_XXXX.vla  → --base-dataset + --dagger-rounds
```

Python is 3.12. Commands that load LeRobot use `uv run --group pi --no-group docs`. LeRobot on this machine is 0.6.1.

---

## 1. Experiments

An experiment is one task setup: overview camera, home pose, one `.vla` set, any number of LeRobot datasets built from it, and up to three trained models. The current ones are `pick`, `sort`, and `organize`.

Versioned in git:

| File | Contents |
|---|---|
| `examples/cfg/kassow_experiments.py` | registry: name, overview camera serial, home file |
| `examples/cfg/kassow_homes/home_<exp>.json` | home joints + TCP, tagged with the experiment name |

Not in git (`/experiments/` is in `.gitignore`):

```text
experiments/
  active.json                  {"experiment": "sort"}
  <exp>/
    experiment.json            selected dataset/model; how each dataset was built; tasks of each model
    raw/                       traj_XXXX.vla + tasks*.yaml (label files)
    dagger/r<N>/               DAgger rollouts of round N + *.dagger.json, discarded/ (section 7)
    lerobot/<variant>/         LeRobot datasets
    models/<name>/             checkpoints (the pretrained_model folder), max 3
```

### Current inventory

| | pick | sort | organize | open |
|---|---|---|---|---|
| Overview camera (`camera_2`) | `262522075082` | `261222078043` | `261222078043` (check before recording) | `261222078043` (same cell as sort) |
| Wrist camera (`camera_1`) | `112322070077` | `112322070077` | `112322070077` | `112322070077` |
| Home TCP | `[-0.435, 0.462, 0.320]` | `[-0.203, 0.352, 0.446]` | not captured yet | one file per sentence, not captured yet |
| `.vla` | 428, **frozen** | 90 | — | — |
| Label files | `tasks.yaml` | `tasks.yaml` (one sentence), `tasks_spaces.yaml` (space 1/2/3) | — | — |
| LeRobot | `notrim` (428), `dropout` (428), `dropout_split` (856, two tasks) | `spaces_notrim` (90) | — | — |
| Models | `pick_cyl_pi05` | `run1` | — | — |
| Tasks of the model | `pick up the blue object and place it on the red plate` | `put the object in space 1` / `2` / `3` | — | same sentences, one home each |

`pick_cyl_pi05` has the same normalizer statistics as `notrim` and `dropout`, so the training set cannot be told apart by stats; it is registered as `notrim`. `run1` matches `spaces_notrim` exactly.

### Switching

```bash
uv run -m examples.experiment                         # status of the active experiment, with commands
uv run -m examples.experiment use sort                # switch
uv run -m examples.experiment use sort --model run1 --dataset spaces_notrim
```

After `use`, these need no paths:

| Step | Command | Takes from the experiment |
|---|---|---|
| Teleop | `uv run python -m examples.kassow_vive_gripper --go-home-on-save` | overview camera, home, `raw/` |
| Go home | `uv run python -m examples.goto_kassow_q` | home |
| Convert | `… convert_to_lerobot_pi05 --variant <v> …` | `raw/`, labels, output folder |
| Serve | `POLICY=Pi05LerobotCfg … examples.policy_inference --policy-cfg.device cuda:1` | camera, selected model, its tasks |
| Offline check | `… examples.diagnose_pi05_offline` | selected model and dataset |

`--experiment <name>` on any of these overrides the active one for that run.

### CLI reference (`uv run -m examples.experiment …`)

| Subcommand | Effect |
|---|---|
| *(none)* / `status [exp]` | cameras, home, `.vla` count, label files, datasets, models, ready-made commands |
| `list` | one line per experiment |
| `use <exp> [--model M] [--dataset D]` | activate, optionally select model/dataset |
| `add-model <name> <path> [--dataset D] [--experiment E]` | register a checkpoint (symlink). Without `--dataset` the training set is found by comparing normalizer stats. Stores its task strings. Refuses a 4th model. |
| `add-dataset <variant> <path> [--tasks F]` | register an existing LeRobot dataset (symlink) |
| `link-raw <path>` | use an existing `.vla` folder as `raw/` |
| `freeze [exp]` / `unfreeze [exp]` | forbid / allow recording into `raw/` |

The CLI never deletes or overwrites. To drop a model, delete its entry in `models/` by hand.

### Safety rules built in

- **Frozen `.vla` sets.** Teleop refuses to record into a frozen experiment. For driving the arm without recording, pass `--recorder None`.
- **Home poses belong to one experiment.** `--capture-home` writes the current pose to the active experiment's home file and tags it with the name. A home file of another experiment is refused at load. If an experiment has no home yet, teleop stops and asks for `--capture-home` instead of silently using the current pose. `open` has one home per sentence. `--instruction "put the object in space 1"` selects `home_open_space1.json`, and the same pattern holds for space 2 and space 3. `--pose space1`, `--pose space2`, and `--pose space3` drive to those files. A sentence with no file yet has to be captured before recording.
- **Models belong to one experiment.** Serving a model registered under `pick` while the station is set up for `sort` (camera, home) is refused.
- **Task strings must match.** Serve refuses an instruction that is not in the model's training tasks (override: `--allow-unlisted-instruction`).
- **Existing datasets are never replaced.** A convert into an existing variant needs `--clean`; a linked dataset can never be a convert target.

### Adding an experiment

Example `organize`; the registry entry already exists. For a new name, add a `KassowExperiment` to `EXPERIMENTS` in `examples/cfg/kassow_experiments.py`.

1. Check the overview camera serial in the registry (`rs-enumerate-devices -s` lists the connected ones).
2. `uv run -m examples.experiment use organize`
3. Move the arm to the start pose and capture the home once: `uv run python -m examples.kassow_vive_gripper --go-home-on-save --capture-home`. Keep that session for recording, or restart without `--capture-home`.
4. Record (section 2), write the label file (section 3), convert (section 4), train (section 5).
5. Copy the trained `pretrained_model` folder to `experiments/organize/models/<name>/`, then `uv run -m examples.experiment add-model <name> experiments/organize/models/<name>`.

Pick the home orientation with a rotation-vector norm clearly below π (shown as `rx, ry, rz` in the home file). Near π, two numerically opposite rotation vectors describe almost the same orientation, which makes episodes inconsistent.

`open` is the sort cell with a different start pose per sentence. Its gripper is closed at the start of a session and opens only to 0.3. Teleop, the LeRobot discretization, and policy rollout all use that pair; the other experiments stay at 0.0/0.7. Capture and record space 1 first (50 demos, `traj_0000`–`traj_0049`), then capture space 2 and keep recording into the same `raw/`:

```bash
uv run -m examples.experiment use open
uv run python -m examples.kassow_vive_gripper --go-home-on-save --capture-home \
  --instruction "put the object in space 1"
uv run python -m examples.goto_kassow_q --pose space1
uv run python -m examples.goto_kassow_q --pose space2
```

The capture command writes `examples/cfg/kassow_homes/home_open_space1.json` and can stay open for the 50 demos. Space 2 is the same command with `"put the object in space 2"`, which writes `home_open_space2.json`. `--pose space1` and `--pose space2` move to the saved pose without recording.

---

## 2. Recording

**Script:** `examples/teleop_vive_hand.py` via `examples/kassow_vive_gripper.py`
**Station:** `KassowViveGripperStation` (`examples/cfg/kassow_vive_gripper.py`, extends `examples/cfg/kassow_vive.py`)

```bash
uv run python -m examples.kassow_vive_gripper --go-home-on-save
# first start after the AG95 was power-cycled: add --gripper-cfg.calibrate
# slower: --max-pos-speed 0.15 --max-rot-speed 0.4 --home-joint-speed 0.08
```

| Key | Action |
|---|---|
| `n` | start recording a trajectory |
| `c` | clutch: the arm follows the tracker only while engaged |
| `o` | toggle orientation tracking |
| `s` | save the trajectory (clutch drops), then home |
| Space | toggle gripper (0.0 closed / 0.7 open) |
| Logitech mouse | left close, right open, back finer, forward coarser motion scale |

With `--go-home-on-save`, the arm drives to the home pose after Enter and after each `s`. This is one discrete KORD `moveJ` (0.15 rad/s); streaming resumes 0.5 s after it ends. Wait for `Home reached` before `c`. `n` works right after `s`, because encoding runs in the background.

What is recorded, at 100 Hz:

- The commanded TCP (`arm_target_pose`) as the action, not a second measured pose. The arm follows with `task_pos_ik` (`ik_kp=12`).
- Both RealSense images, 480×640 RGB, as the newest frames from 30 Hz ring buffers. There is no shared capture timestamp with the joints.
- The AG95: `gripper_position` is the hardware reading (2 Hz), `gripper_commanded_position` the last written target.
- The `instruction` field stays empty. Labels are added afterwards (section 3).

Keys inside a `.vla` that the converter needs:

| Key | Meaning |
|---|---|
| `observation/cameras/camera_1/rgb` | wrist RGB |
| `observation/cameras/camera_2/rgb` | overview RGB |
| `observation/proprio_joints` | 7 joint angles, rad |
| `observation/gripper_position` | AG95 opening, about 0–1 |
| `observation/gripper_commanded_position` | last AG95 target (optional) |
| `action` | commanded TCP: xyz (m) + rotvec + gripper |

---

## 3. Labels

Labels live as text next to the data, `experiments/<exp>/raw/tasks*.yaml`, and are written after recording. The index is the number in `traj_XXXX.vla`. Several files can coexist; each conversion picks one with `--tasks`.

```text
# one sentence for all (sort/tasks.yaml)
0-89: sort the packages into the next cardboard compartment

# alternating sub-tasks (sort/tasks_spaces.yaml): traj 0 → space 1, 1 → space 2, 2 → space 3, 3 → space 1, …
0: put the object in space 1
1: put the object in space 2
2: put the object in space 3
3: put the object in space 1
```

A JSON object `{"0-19": "…", "20": "…"}` is accepted too. Without a label file, the converter writes a fill-in `tasks.yaml` into `raw/` and stops.

The recorder numbers new files `max + 1`. To redo one demo, replace the file under the same name or adjust the label file; otherwise an alternating pattern shifts silently from that point on. The string a model sees in training is the string serve must send byte for byte. `add-model` stores it, so serve can check.

---

## 4. Conversion to LeRobot

**CLI:** `examples/data/convert_to_lerobot_pi05.py` · **Implementation:** `rio/data/convert_pi05.py` (`ConvertPi05Cfg`, `convert`)

```bash
uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \
  --variant spaces_notrim \
  --tasks tasks_spaces.yaml \
  --no-trim-idle \
  --gripper-state-source commanded \
  --workers 4
```

This reads `experiments/<active>/raw`, writes `experiments/<active>/lerobot/spaces_notrim` with `repo_id` `local/kassow_<exp>_spaces_notrim`, and records the flags in `experiment.json`. The first variant built becomes the selected one. `--input`/`--output`/`--repo-id` still work without an experiment.

| Flag | Default | Use |
|---|---|---|
| `--variant` | — | dataset name; required in experiment mode |
| `--tasks` | `raw/tasks.yaml` | label file, looked up in `raw/` |
| `--trim-idle` / `--no-trim-idle` | trim on | drop the start until the TCP moved 1 cm. All current models use `--no-trim-idle`. |
| `--gripper-state-source` | `measured` | `commanded` matches the serve default and all current models |
| `--workers` | 1 | parallel episodes, ~8 GB RAM each |
| `--action-space` | `tcp` | `joints` for a joint-space dataset (section 8) |
| `--clean` | off | allow replacing an existing (not linked) variant |
| `--dry-run` | off | per-episode lengths, idle start, labels, without writing |

Two defaults (`trim_idle`, `measured`) differ from what the current models use. Pass both flags explicitly.

What happens to each episode:

1. Load the `.vla` and check the keys above.
2. Subsample 100 → 20 Hz with stride 5. All features are cut on the same index.
3. Unwrap the TCP rotvec inside the episode, so a yaw near ±π does not jump by 2π. At inference it is folded back to `|r| ≤ π`.
4. Optionally trim the idle prefix.
5. Binarize the gripper. The action snaps to 0.0/0.7 at 0.35. The state uses hysteresis: it closes at ≤ ~0.21 and opens at ≥ ~0.46.
6. Attach the task string by trajectory index.
7. Write LeRobot v3: one H.264 MP4 per episode and camera, 480×640 RGB. `_patch_lerobot_pyav()` covers PyAV 13.

Parallel conversion (`--workers > 1`) converts slices in separate processes and merges them with LeRobot's `aggregate_datasets`. Each slice numbers its tasks in its own order, so the merge must remap `task_index` onto one table. The merge before 30 Oct 2026 did not, which gave wrong per-frame labels when one dataset had several tasks. A 6-episode check with 3 workers now gives identical data, statistics, byte-identical videos, and correct labels compared with `--workers 1`. All datasets listed above were built with `--workers 1`.

Output columns:

| Key | Shape | Contents |
|---|---|---|
| `observation.images.camera_1` | 480×640×3 video | wrist |
| `observation.images.camera_2` | 480×640×3 video | overview |
| `observation.state` | 8 | `j1…j7`, gripper |
| `action` | 7 | `x, y, z, rx, ry, rz`, gripper (absolute TCP) |
| `task` | string | from the label file |

---

## 5. Training (cluster)

Training is LeRobot's `lerobot-train` on the cluster, not a script in this repo. Copy the variant folder (`experiments/<exp>/lerobot/<v>`) to the cluster as is. Do not reconvert, resize, or rename cameras there.

Environment: Python 3.12, `pip install 'lerobot[pi,training]==0.6.1'`, `hf auth login` (PaliGemma is gated). LeRobot 0.4.x and Python 3.14 do not load these checkpoints.

Smoke test before a long job:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/kassow_sort_spaces_notrim", root="/abs/path/spaces_notrim", video_backend="pyav")
s = ds[0]
assert s["observation.state"].shape[-1] == 8 and s["action"].shape[-1] == 7
print(s["task"], s["observation.images.camera_1"].shape)
```

The command that produced `run1` and `pick_cyl_pi05` (from their `train_config.json`):

```bash
lerobot-train \
  --dataset.repo_id=local/kassow_sort_spaces_notrim \
  --dataset.root=/abs/path/spaces_notrim \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.compile_model=false --policy.push_to_hub=false --policy.device=cuda \
  --steps=24000 --batch_size=32 --save_freq=6000 --num_workers=4 --seed=1000 \
  --output_dir=./outputs/<run> --job_name=<run>
```

Both models used LR 2.5e-5, 1000 warmup steps, 30000 decay steps, QUANTILES normalization for state and action, and image transforms with brightness, contrast, saturation, hue, sharpness, and a RandomAffine of ±5° / 5 % shift.

Rules:

- `--policy.pretrained_path` loads weights only. `chunk_size` and `n_action_steps` fall back to 50 unless set. Do not use `--policy.path=lerobot/pi05_base`; it inherits the base camera keys.
- Keep `use_relative_actions` off (absolute TCP) and `empty_cameras=0` (two real cameras). Do not pad features; π₀.₅ pads to 32 internally.
- The RandomAffine shift moves the image by a few centimetres of workspace. That works against precise placement; consider disabling it for tasks that need accuracy.
- Deliver the whole `pretrained_model` folder: `config.json`, `model.safetensors` (~9.3 GB), and the pre/post-processor files. Then place and register it (section 1, step 5).

---

## 6. Inference

**Loop:** `examples/policy_inference.py` · **Config:** `Pi05LerobotCfg` (`examples/policy_cfgs/pi05_lerobot.py`) · **Model:** `rio/policies/pi05.py` · **GPU process:** `rio/policies/pi05_resident.py`

Drive to the home first (`uv run python -m examples.goto_kassow_q`), then:

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-cfg.device cuda:1 \
  --instruction "put the object in space 2"
```

- Without `--policy-path` the selected model of the experiment runs. Without `--instruction` it uses its first training task. An instruction the model was not trained on, or a model of another experiment, stops the run before weights load.
- `Pi05LerobotCfg` switches off Vive, clicker, and recorder and forces `task_pos_ik`. Control runs at 100 Hz, policy steps at 20 Hz; `ActionInterpolator` fills the ticks with linear position and SO(3) SLERP orientation.
- `limit_tcp_action_step` caps each 100 Hz step at `--arm-cfg.max-pos-speed` / `--arm-cfg.max-rot-speed`, so a jump in the plan becomes a ramp.
- Enter toggles pause, during which the weights stay on the GPU. `q` quits.

Defaults and knobs:

| Flag | Default | Meaning |
|---|---|---|
| `--policy-node-cfg.use-rtc` / `--no-use-rtc` | on | RTC merges the next chunk into the leftover of the current one |
| `--policy-node-cfg.n-execute` | 30 | steps played per 50-step chunk |
| `--policy-node-cfg.rtc-queue-threshold` | 18 | with RTC: request the next chunk when this many steps remain |
| `--policy-node-cfg.chunk-pause-s` | 0 | hold after a chunk before the next image |
| `--arm-cfg.max-pos-speed` / `--arm-cfg.max-rot-speed` | 0.45 m/s / 1.0 rad/s | command rate limit |

With RTC on, 30 of 50 steps, and 0.45 m/s, a late chunk that reverses toward the start pose (18 cm plan span) was executed far enough to trip the controller (see section 10). Safer first runs: `--policy-node-cfg.no-use-rtc --policy-node-cfg.n-execute 16 --arm-cfg.max-pos-speed 0.12 --arm-cfg.max-rot-speed 0.4`.

`--safety-cfg` sets fail-closed software limits. The defaults are 5 cm TCP error, 0.35 rad orientation error, 0.25 rad joint error, 0.25 s feedback age, and 2 s inference timeout. Exceeding one ends playback and stops the stream. These limits do not acknowledge controller safety and are not an emergency stop; the 5 cm limit fires only after a jump has started.

The GPU resident keeps the ~9 GB weights loaded across runs on the same device. It is reused only if checkpoint files, code versions, RTC, and gripper options match; otherwise the old one is shut down before the new one loads. To unload whatever resident runs on a device:

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference --policy-cfg.device cuda:1 --unload-policy
```

Every start or resume writes a `session_*` folder under `--diag-dir` (default `/tmp/pi05_diag`). It contains the effective configuration, camera snapshots, per-chunk `.npy` files, and `chunks.csv` (`span_m` is the plan before IK). `cmd_vs_eef.csv` holds command vs measured TCP.

The gripper is two-state (0.0/0.7). Conversion and inference share `binarize_gripper` (on). For an analog gripper later, reconvert with `--no-binarize-gripper` and serve with `--policy-cfg.no-binarize-gripper`. At policy startup the gripper gets no initial command, so a held object stays held.

---

## 7. DAgger (operator takeover)

**Loop:** `examples/policy_inference.py` with `DaggerSession` (`examples/dagger_kassow.py`) · **Config:** `Pi05DaggerCfg` (`examples/policy_cfgs/pi05_dagger.py`) · **Labels and segments:** `rio/data/dagger.py`

A model trained only on demos has seen only states the operator steered into. When it drifts, it reaches states it never saw and its errors grow. DAgger lets the policy drive, and the operator corrects it in exactly those states. The corrections are added to the demos and the model is retrained, round after round. This is the human-gated variant (HG-DAgger): the operator takes over only when the policy goes wrong.

```bash
POLICY=Pi05DaggerCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-cfg.device cuda:1 --dagger-round 1 --instruction "put the object in space 2"
```

`Pi05DaggerCfg` is `Pi05LerobotCfg` with the Vive tracker, terminal keys, clicker, and recorder switched back on. Model, instruction check, RTC, `n_execute`, and speed limits are the same as for plain serving. Rollouts go to `experiments/<exp>/dagger/r<N>/`, so `raw/` is never written and a frozen `raw/` does not block DAgger. The instruction is stored in every frame, so no label file is needed.

After the yaw calibration (if any) and Enter, the arm drives home. Every episode starts at the experiment home: after each `s`/`x` the arm drives home, and `n` checks the joints again (1° tolerance). If the arm was moved or the last move timed out, `n` drives home first and starts only once the arm is there. Keys:

| Key | Action |
|---|---|
| `n` | start an episode (at home): recording starts, the policy drives |
| `c` | **take over**: the policy stops, the Vive clutch engages at the last commanded pose (no jump). While you drive, `c` releases (arm holds) and re-engages the clutch. |
| `r` | **hand back**, clutch released only: the arm holds until the policy's first fresh chunk, then the policy drives |
| `o`, Space, clicker | orientation tracking, gripper (while you drive) |
| `s` | save the episode, then home |
| `x` | discard the episode, then home; the file moves to `discarded/`, it is never deleted |
| `q` | quit (an open episode counts as discarded) |

A safety stop or controller alarm during an episode also ends it as discarded, with the reason in its sidecar.

**What is recorded.** The whole episode, at 100 Hz and in the teleop `.vla` format, plus `meta/control_source` per frame: `0` policy, `1` operator, `2` hold (after a handback, until the fresh chunk). Next to each file, `traj_XXXX.dagger.json` holds the instruction, model, outcome, takeovers, handbacks, and frame counts per source.

**RTC across a switch.** On takeover, the action queue and RTC leftover are cleared, and a chunk still being computed is dropped when it arrives. On handback the policy starts like a fresh rollout: `policy.reset()`, empty queue, first chunk without leftover, and the startup creep if its first waypoint is more than 5 mm away. RTC then runs as configured.

**Limits while you drive.** As in teleop, the clutch releases when the command runs more than `--max-lag` (0.25 m) ahead of the arm or tracking is lost for `--tracking-grace`. The rollout position limit is raised to `max-lag + 5 cm` only while you drive; feedback age and alarm checks stay.

### When to end an episode

Measured on the 90 `sort` demos (`spaces_notrim`, 20 Hz action): every demo ends with the gripper open. From the last release to `s` took 3.6 s (median; 10–90 %: 2.5–4.4 s). In that time the TCP moved 15 cm (6–22 cm), mostly sideways; it rose at most 2.3 cm (1.1–4.3 cm), and it stood still for 0.3 s right before `s`. The model learned that ending. So:

- **After a takeover:** place, release, move about 15 cm away from the compartment as in the demos, hold still for half a second, `s`.
- **Policy succeeds alone:** wait for the release and the retreat, then `s`. If it keeps moving after the release, press `s` once it is about as far away as in the demos. Such episodes count as autonomous successes.
- **Not recoverable** (object out of reach, collision): `x`.

These numbers describe how the demos ended, not how far a retreat must be. An ending that differs from the demos teaches the model a second ending.

### Building the dataset

```bash
uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \
  --variant dagger_r1 --base-dataset spaces_notrim --dagger-rounds 1 \
  --no-trim-idle --gripper-state-source commanded
```

The base dataset (the original demos) is copied as is and never changed. Each round is converted and appended with LeRobot's `aggregate_datasets`. Each kept segment becomes its own episode. Later rounds list all of them: `--dagger-rounds 1 2`. The base's recorded `action_space`, `gripper_state_source`, and `binarize_gripper` must match, otherwise the build stops.

| Flag | Default | Effect |
|---|---|---|
| `--segments human` | yes | only the operator runs (HG-DAgger); runs shorter than `--min-segment-s` (1 s) are dropped |
| `--segments all` | | policy and operator frames, without hold frames and without the `--pre-takeover-cut-s` (1 s) before each takeover, where the policy was already going wrong |
| `--include-autonomous` | off | also episodes without any takeover |
| `--trim-operator-pauses` / `--no-…` | on | cut handover stillness from operator runs: the still start after a takeover (reaction time, hand not moving yet) and still stretches of at least `--max-pause-s` (0.3 s), e.g. clutch released to re-grip. Still stretches with a gripper switch and the still end of the episode are kept. "Still" is a commanded TCP below 1 cm/s and 0.05 rad/s. |

The 90 `sort` demos have almost no mid-task stillness: 7 still stretches of at least 0.25 s in total, none longer than 0.3 s and none at a gripper switch (the gripper was always switched while moving); the start idle was 0.6 s (median). Handover pauses would teach the model to stop in exactly the states where it needs to act.

To look at one round without writing anything: `… convert_to_lerobot_pi05 --input experiments/sort/dagger/r1 --segments human --dry-run`.

### Training and evaluation

Fine-tune from the current model on the combined dataset, with the command of section 5 and `--policy.pretrained_path=<abs>/experiments/sort/models/run1`, `--dataset.repo_id=local/kassow_sort_dagger_r1`, `--dataset.root=<abs>/experiments/sort/lerobot/dagger_r1`. Use fewer steps than the 24000 from scratch and compare the saved checkpoints. Training on the corrections alone would make the model forget the normal course of the task. Register the result with `add-model`. Its stats should match `dagger_r1`, and `run1` plus two DAgger models fill the three model slots.

`uv run -m examples.experiment` shows per round: saved and discarded rollouts, rollouts without takeover, takeovers per episode, and the operator's share of the driven frames. Both should fall from round to round, but they come from the training rollouts themselves. Whether a DAgger model is better shows only on a fixed test set: for example 15 start situations, 5 per compartment with fixed object positions, run with `run1` and the DAgger model, counting successes without takeover.

---

## 8. Joint-space variant

A TCP model outputs metres and axis-angle; its seven numbers are not joint angles and must never be sent to `moveJ`. Joint control needs a model trained on joint actions. The demos already contain the joints, so no new recording is needed.

```bash
uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \
  --variant joints_notrim --action-space joints --tasks tasks.yaml --no-trim-idle --gripper-state-source commanded
```

Here `action[t, :7] = q[t+1]` (the last frame repeats `q[-1]`), and the gripper is the command at `t`. Train as in section 5 with this dataset, then register the model and serve with `POLICY=Pi05LerobotJointCfg` (`moveJ`, `--policy-path` as needed). `Pi05LerobotJointCfg` refuses a TCP checkpoint, and `Pi05LerobotCfg` refuses a joint checkpoint. Check the joint ADE offline (section 9) before running it on the arm.

---

## 9. Offline tools

| Script | Use |
|---|---|
| `uv run --group pi --no-group docs -m examples.diagnose_pi05_offline` | Runs the selected model on frames of the selected dataset, with each frame's own label. Prints ADE and plans to `/tmp/pi05_diag/offline/`. `--model`, `--dataset`, `--sample-at start\|mid\|0,0.5`, `--num-samples`. |
| `POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.replay_pi05_dataset --episode 0` | Plays a recorded episode through the serve IK path, without a policy. If this tracks smoothly, the IK can execute 20 Hz TCP. |
| `uv run python -m examples.goto_kassow_q [--pose home\|pick\|sort\|space1\|space2\|space3\|start\|cylinder]` | One discrete `moveJ`. `home` is the active experiment's home; `space1`/`space2`/`space3` are the `open` homes; `start`/`cylinder` are the pick eval poses in `experiments/pick/raw/`. |

---

## 10. Robot connection and alarms

**Network.** The controller is `192.168.1.44`, KORD port 7582, on NIC `enp180s0f1`. That NIC ("Wired connection 2") is static `192.168.1.43/24`, with DNS off and never the default route. The office NIC is on the same /24, so a host route `192.168.1.44/32` via `enp180s0f1` is needed (`nmcli connection modify "Wired connection 2" +ipv4.routes "192.168.1.44/32"`). Check with `ip route get 192.168.1.44`. After a robot reboot, `Connection refused` for 1–2 min only means CBun is still starting.

**Bridge.** The native `_kord_bridge` must match `rio-hw`; an outdated one is rejected before connecting. To rebuild:

```bash
cd ../rio-hw
cmake -S kord_bridge -B kord_bridge/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/../rio/.venv/bin/python
cmake --build kord_bridge/build --target _kord_bridge -j2
cmake --install kord_bridge/build --component kord_bridge
```

**Alarms.** A controller alarm latches, and from then on every motion request is skipped (`skip discrete moveJ, alarm …`). The arm looks dead even though teleop runs. The alarm is often not shown on the pendant.

| Log | Meaning |
|---|---|
| `SafetyEvent/EXTERNAL_ESTOP_ACTIVATED` | the safety circuit opened. A pressed button or the controller's own reaction to a command jump (tracking/torque deviation) both cause it. In policy runs it followed large plan jumps. |
| `CBunEvent/MOVE_IN_INVALID_MOTION_STATE` | the controller is not ready (left-over halt/suspend); motion is refused |
| `JTORQUE_DEVIATION_EXCEEDED`, `JREF_*_SPAN` | the command moved faster or further than the arm could follow |

To recover: stop teleop/serve, release the physical E-stop if pressed, then `scripts/kassow_clear_alarms.sh` (no sudo; wraps `kord-clean-alarm -c 192.168.1.44 --all`, root is only needed for its unused `-r` realtime flag). Alternatively start once with `--arm-cfg.clear-alarms-on-start`. Clearing is off by default on purpose. More codes are in `rio-hw/docs/robots/kassow.md`.

---

## 11. Storage

A checkpoint is ~9.3 GB, a `.vla` set 1–2.5 GB, and a LeRobot variant 0.4–1.3 GB. `experiments/` is git-ignored, and moving folders inside it is a rename on the same disk. Check free space (`df -h /`) before recording or copying a model in. A full disk during recording corrupts the `.vla` being written.

---

## 12. Jev crate orchestrator

The third RealSense (`JEV_SERIAL` in `examples/cfg/kassow_experiments.py`) looks at the crate from above. It is **not** in `station.cameras`. The π₀.₅ observation and the `.vla` files stay two cameras. Jev reads a JPEG from that node on demand.

The same three questions go to the hosted TypeSafe API or to a local OpenJev helper (`--jev.backend typesafe|local`). Only the image transport differs. Recalibrate the noul / count thresholds on the same labelled probes after switching.

### Ask without moving the arm

```bash
export TYPESAFE_API_KEY=...
uv run python -m examples.jev_ask
```

Enter sends one request (inlay organized, carton count 0–10, any carton at all). `q` quits. Each frame is written under `experiments/orchestrator/probes/` as `timestamp.jpg` plus `timestamp.jev-typesafe.json`.

Saved photos, scored against hand labels `name.json` = `{"organized": true, "count": 3}` next to `name.jpg`:

```bash
uv run python -m examples.jev_ask --image experiments/orchestrator/probes
```

Collect about twenty probes that cover crumpled vs standing inlay, counts 0–3 (and higher if the crate can hold them), and different lighting. The script prints how often `organized` lands on the labelled side of the 0.3 / 0.7 band, a count confusion matrix, and how often an *accepted* count (p ≥ 0.8) is wrong. Those accepted errors are the ones that would send the arm to the wrong space. If counts above five stay confused, split the question into one occupancy check per compartment instead of raising the option list.

A warning that `input_tokens` looks like a text-only request means the image did not reach the model. The hosted API needs `assets` plus the header `TypeSafe-Danger-Zone-Images: true`; a `data:` URL in `state.screenshot` is only for the local helper.

### Dry run, then the arm

```bash
uv run python -m examples.orchestrator_kassow --dry-run
```

`j` asks Jev for real. moveJ and the VLAs are only printed. Keys: `j` ask, `p` proceed after an open, `s` end a VLA, `x` abort, `q` quit.

```bash
uv run --group pi --no-group docs -m examples.orchestrator_kassow --device cuda:1
```

Both the organize and the open checkpoint load as named residents on that GPU (`rio_pi05_cuda-1-organize` and `rio_pi05_cuda-1-open`). Check occupancy with `nvidia-smi` after the first load. If the second resident is refused, pass `--single-slot` so the checkpoint is swapped on each organize/open change.

What Jev's answers drive:

| Reading | Action |
|---|---|
| inlay crumpled (`organized` ≤ 0.3) | moveJ to the organize home, run the organize VLA until `s` or the timeout, then ask Jev again |
| inlay organized, N cartons, space N+1 has a home | moveJ to that space, run open with `put the object in space N+1`, stop at the first open-gripper command, wait for `p`, moveJ to the organize home |
| unsure inlay, unsure count, or no home for space N+1 | hold and print why |

Homes exist today for spaces 1–3. A count of 3 or more holds until `home_open_space4.json` (and so on) is captured. Each cycle is logged under `experiments/orchestrator/runs/<time>/`.
