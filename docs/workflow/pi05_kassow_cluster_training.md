# π₀.₅ cluster training brief — Kassow + AG95

**Audience:** coding agents that implement / launch fine-tuning on the cluster.
**Owner machine (inference only):** `jarvis`, repo `/home/hme2lo/Documents/rio`. Hardware and LeRobot 0.6.1 share the Python 3.12 `.venv`.
**Do not** collect data, command the robot, or change the converter unless this brief says so.
**Do not** use the DROID converter or the OpenPI DROID finetune path in `docs/workflow/pi0_droid_finetuning.md`. That stack is a different schema.

Goal: fine-tune `lerobot/pi05_base` on the already-converted LeRobot **v3.0** dataset with **LeRobot 0.6.1**. The folders were written by 0.4.4 and **load unchanged** on 0.6.1 — do not reconvert.

---

## 1. Success criterion

A local checkpoint directory that this exact load path accepts **with `lerobot==0.6.1`** (Python 3.12, not 3.14):

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # or Pi05Policy

cfg = PreTrainedConfig.from_pretrained(path)
policy = PI05Policy.from_pretrained(path, config=cfg)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=path
)
```

Then `policy.predict_action_chunk(preprocessor(batch))` returns a tensor that, after postprocess, can be sliced to `(chunk, 7)`.

Jarvis deploy (do **not** run this on the cluster):

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path /path/to/checkpoint \
  --instruction "pick up the blue cylinder and place it on the red plate"
```

The instruction string must match the dataset **byte-for-byte**.

---

## 2. Pin these versions

| Piece | Required | Why |
|---|---|---|
| LeRobot | **`==0.6.1`** with extras **`[pi,training]`** | Cluster constraint. `pip install lerobot` alone is not enough in 0.6. |
| Python | **3.12** (not 3.11, not 3.14) | 0.6 needs `>=3.12`. 3.14 breaks `from_pretrained` (draccus). |
| Transformers | whatever `lerobot[pi]==0.6.1` pulls (5.x) | Do not pin Transformers 4.x. |
| Base weights | **`lerobot/pi05_base`** | Not `pi05_libero_base` (extra dummy camera, different keys). |
| Dataset format | LeRobot **codebase_version `v3.0`** | Already converted. Do not reconvert for 0.6. |

Install:

```bash
pip install 'lerobot[pi,training]==0.6.1'
hf auth login   # PaliGemma is gated
```

Verified on jarvis: both `data/lerobot/kassow_pi05` and `data/lerobot/kassow_pi05_alignment` load with `LeRobotDataset(..., video_backend="pyav")` under 0.6.1. `q01`/`q99` are present.

---

## 3. Dataset (already converted — do not reconvert)

Copy this folder onto the cluster as-is:

```
data/lerobot/kassow_pi05/
```

| Field | Value |
|---|---|
| `repo_id` (inside dataset) | `local/kassow_pi05` |
| `codebase_version` | `v3.0` |
| `robot_type` | `kassow` |
| `fps` | **20** |
| episodes | **180** (`splits.train = 0:180`) |
| frames | **42104** (idle prefix trimmed) |
| tasks | **1** |
| shortest episode | **96** frames (all `>= 16`; median 195) |
| videos | H.264, **480×640×3**, RGB, `yuv420p`, one MP4 per episode (`video_files_size_in_mb=1`) |

Source teleop was 100 Hz robodm `.vla`; converter used stride `100/20 = 5`. Do not upsample. Do not change fps.

**Already applied in this folder** (do not undo, do not reconvert):

- Gripper binarized to `{0.0, 0.7}`.
- TCP rotvec **unwrapped per episode** (encoding stays axis-angle; `|r|` may exceed π).
- Stationary prefix trimmed until TCP xyz moves **≥ 1 cm**.

Pass it as a **local root**, not a Hub dataset:

```text
--dataset.repo_id=local/kassow_pi05
--dataset.root=/absolute/path/to/kassow_pi05
```

(`HF_LEROBOT_HOME` is optional; `--dataset.root` is enough.)

### 3.1 Feature contract (must match inference)

| Dataset key | Shape | Meaning |
|---|---|---|
| `observation.images.camera_1` | `(480, 640, 3)` video | RealSense D435i serial `112322070077`, **RGB** (not BGR) |
| `observation.images.camera_2` | `(480, 640, 3)` video | RealSense D435i serial `262522075082`, **RGB** |
| `observation.state` | `(8,)` float32 | 7 Kassow joints (rad) + gripper. Names: `j1…j7, gripper` |
| `action` | `(7,)` float32 | TCP pose **xyz (m) + rotvec (axis-angle, unwrapped)** + gripper. Names: `x, y, z, rx, ry, rz, gripper` |
| `task` | string | language instruction (see below) |

π₀.₅ internally pads state/action to `max_state_dim=32` / `max_action_dim=32`. **Leave features at 8 and 7.** Do not pad in the dataset.

Do **not** resize images to 224. PaliGemma does that inside the policy. Native resolution stays 480×640.

Do **not** enable relative actions (`use_relative_actions`). Actions are **absolute** TCP + gripper, executed on jarvis with `task_pos_ik` at 20 Hz.

Do **not** rename cameras to `cam_high` / `image` / `exterior_image_1_left`. Jarvis looks up `camera_1` and `camera_2`.

### 3.2 Task string

Exactly one label for all 180 episodes:

```
pick up the blue cylinder and place it on the red plate
```

Eval instruction must be identical. Do not paraphrase.

### 3.3 Gripper

Binarized in **this** dataset. Convert and jarvis inference must use the same setting.

| | This AG95 set | Later analog gripper |
|---|---|---|
| Convert | default (`binarize_gripper=True`) | `--no-binarize-gripper` |
| Inference | `Pi05LerobotCfg.policy_cfg.binarize_gripper=True` | `--policy-cfg.binarize-gripper false` |
| open / close | `0.7` / `0.0` (not 1.0) | leave raw; inference clips `[0, 1]` |

Flags live in `rio/data/convert_pi05.py` (`ConvertPi05Cfg`) and `examples/policy_cfgs/pi05_lerobot.py` (`PolicyConfig`). Teleop `gripper_open` in `examples/cfg/kassow_vive_gripper.py` is hardware only.

Unique values here: `{0.0, 0.7}`. Do not rescale open to 1.0.

### 3.4 Stats sanity (from `meta/stats.json`)

Action min → max after unwrap + idle trim:

| dim | min | max | note |
|---|---|---|---|
| x | −0.676 | −0.036 | table-scale, **x never positive** |
| y | 0.098 | 0.841 | |
| z | 0.217 | 0.486 | |
| rx | −3.181 | 3.202 | unwrapped; within-episode max ‖Δr‖ ≈ 0.07 |
| ry | −2.078 | 2.630 | |
| rz | −0.559 | 0.663 | |
| gripper | 0.0 | 0.7 | |

`q01`/`q99` exist for QUANTILES. Image stats are in `[0, 1]` (LeRobot video decode); a smaller image `count` than `total_frames` is expected.

Do **not** wrap rotvec back to ±π in the dataset. Across episodes the starting hemisphere can still sit near +π or −π; that is leftover from teleop, not a 2π jump inside a demo.

---

## 4. How jarvis will consume the checkpoint (do not break this)

File: `rio/policies/pi05.py`, config: `examples/policy_cfgs/pi05_lerobot.py`.

Live observation mapping:

```
state = concat(proprio_joints[7], gripper_position)   # 8D
images["camera_1"], images["camera_2"]                 # HWC uint8 RGB
task = instruction string
```

Images are permuted HWC → CHW before the LeRobot preprocessor.

Action used on the robot:

```
chunk = postprocessed[:, :7]          # drop 32-dim pad
chunk[:, 3:6] = principal_rotvec(...) # fold unwrap back to |r| ≤ π
chunk[:, -1] = snap {0.0, 0.7}        # if policy_cfg.binarize_gripper
# arm: task_pos_ik  (xyz + rotvec)
```

Do not analog-ize this dataset to “fix” live gripper readings.

Policy loop:

- control / inference rate **20 Hz**
- **`chunk_size = 16`** (and therefore `n_action_steps` must be 16)
- `action_dim = 7`, `proprio_dim = 8`
- `n_obs_steps = 1` (default)

Live AG95 readings are analog; `format_rio_obs` snaps them when `binarize_gripper=True`. Leave that on for this checkpoint.

---

## 5. Training command (copy, then only change paths / batch / GPU)

`--policy.pretrained_path` loads **weights only**. Feature names come from **this** dataset. Config fields such as `chunk_size` / `n_action_steps` **reset to LeRobot defaults (50)**. You must set them explicitly.

`--policy.path=lerobot/pi05_base` is **wrong**: it would inherit DROID/base camera keys.

```bash
# env: python 3.12 + pip install 'lerobot[pi,training]==0.6.1'
# hf auth login   # PaliGemma license

lerobot-train \
  --dataset.repo_id=local/kassow_pi05 \
  --dataset.root=/absolute/path/to/kassow_pi05 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --steps=5000 \
  --batch_size=8 \
  --save_freq=1000 \
  --num_workers=4 \
  --seed=1000 \
  --output_dir=./outputs/pi05_kassow \
  --job_name=pi05_kassow
```

### 5.1 Flags you must not copy from generic π₀.₅ docs

| Flag | This job |
|---|---|
| `--policy.path=…` | **Forbidden.** Use `pretrained_path`. |
| `--policy.pretrained_path=lerobot/pi05_libero_base` | **Forbidden.** |
| `--policy.empty_cameras=1` | **Forbidden.** We have two real cameras. Keep 0 / omit. |
| `--policy.n_action_steps=10` or default 50 | **Forbidden.** Must be **16**. |
| `--policy.chunk_size=50` (default) | **Forbidden.** Must be **16**. |
| `--policy.use_relative_actions=true` | **Forbidden.** Absolute TCP. |
| `--policy.normalization_mapping=MEAN_STD` | **Forbidden.** π₀.₅ default is QUANTILES for state/action. Dataset already has q01/q99. |
| `--steps=30000` | Too many for 42k frames. Use **5000–8000**. Start at **5000**. |
| `--policy.compile_model=true` | Off until one uncompiled run finishes. Then optional. |
| `--rename_map=…` | Not needed if keys stay `camera_1` / `camera_2`. |

`batch_size`: 8 is a safe start on 24 GB; raise if VRAM allows (16–32). Gradient checkpointing stays on.

`train_expert_only=true` is allowed only as a **memory fallback**, not the default. Prefer full finetune.

Scheduler: 42 104 frames / batch 16 ≈ 2632 steps per epoch. 5000 steps ≈ 1.9 epochs. 30k steps → ~11 epochs, still too many. Prefer `--batch_size=16` on an 80 GB GPU.

---

## 6. Mandatory smoke tests (before a long job)

### 6.1 Dataset actually loads on the cluster

TorchCodec often fails without matching system FFmpeg. Jarvis needed `video_backend="pyav"`.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "local/kassow_pi05",
    root="/absolute/path/to/kassow_pi05",
    video_backend="pyav",  # drop if torchcodec works
)
s = ds[0]
assert "observation.images.camera_1" in s
assert "observation.images.camera_2" in s
assert s["observation.state"].shape[-1] == 8
assert s["action"].shape[-1] == 7
print(s["task"], s["observation.images.camera_1"].shape)
```

If this raises, **do not start training**. Fix the video backend first (`pyav` and/or install FFmpeg libs for torchcodec).

Decoded images may be `float32` CHW in `[0, 1]`. That is LeRobot’s loader, not a color bug. Do not convert the dataset to BGR.

### 6.2 Config will be 16/16

After constructing the train config (or after the first checkpoint), assert:

```text
policy.chunk_size == 16
policy.n_action_steps == 16
policy.empty_cameras == 0
input features include observation.images.camera_1 and camera_2
output features include action dim 7 (padded internally to 32)
```

### 6.3 After training, before handing the folder to jarvis

From the saved checkpoint directory (usually `outputs/pi05_kassow/checkpoints/<step>/pretrained_model` — keep whatever `lerobot-train` wrote; do not delete preprocessor files):

```python
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

path = "OUTPUT_CKPT_DIR"
cfg = PreTrainedConfig.from_pretrained(path)
assert cfg.chunk_size == 16 and cfg.n_action_steps == 16
policy = PI05Policy.from_pretrained(path, config=cfg)
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path)
# dummy batch: 8D state, two 3x480x640 (or 3x224x224) images, task string
```

Deliver **the whole pretrained_model directory**, including:

- `config.json`
- weights (`model.safetensors` or equivalent)
- preprocessor / postprocessor / norm stats JSON
- any tokenizer files LeRobot wrote

Do not zip only the `.safetensors`.

---

## 7. Remaining caveats — do not silently “fix”

Idle trim and rotvec unwrap are **already in this dataset**. Do not wrap rotvec back to ±π. Do not restore idle frames.

1. **180 demos** is a usable single-task set. Prefer **5000–8000** steps, not a 30k default.
2. **Workspace is tiny.** Policy will not generalize outside the table volume in `stats.json`.
3. **Train/serve gripper.** This dataset is `{0, 0.7}`. Jarvis snaps live state/action when `binarize_gripper=True`. For a later analog gripper, disable convert **and** inference together.
4. Unwrapped rotvec is still **3D axis-angle**, not 6D/quat. Do not change the action layout.

---

## 8. Forbidden actions

- Do not use `examples/data/convert_to_lerobot_droid.py` or any DROID `LeRobotDROIDDataConfig`.
- Do not train with OpenPI `third_party/openpi` for this checkpoint (jarvis loads **LeRobot** `PI05Policy.from_pretrained`).
- Do not change camera names, action dim, state dim, fps, or image size.
- Do not add a third camera or `empty_cameras`.
- Do not set gripper open to 1.0.
- Do not push to the Hub unless asked (`push_to_hub=false`).
- Do not train with LeRobot 0.4.x. Do not use Python 3.14.
- Do not “helpfully” resize videos to 224.
- Do not command hardware.

---

## 9. What to send back

1. Checkpoint directory that passes §6.3 (path + how it was produced).
2. Exact `lerobot-train` command that ran (after any cluster-specific path edits).
3. `lerobot.__version__`, Python version, CUDA/GPU.
4. Train loss curve / final step / wall time.
5. Confirmation `config.json` has `chunk_size=16`, `n_action_steps=16`, and image keys `observation.images.camera_1` / `camera_2`.
6. If `dataset[0]` needed `video_backend="pyav"`, say so.

That is the entire job.

---

## 10. Second dataset: alignment (do not mix with §3 unless asked)

Converted separately. Same schema, same flags (`chunk_size=16`, `pretrained_path`, LeRobot **0.6.1**). **Do not** train both folders in one `repo_id` unless the human asks for a multi-task run.

| | Alignment |
|---|---|
| Folder | `data/lerobot/kassow_pi05_alignment/` |
| `repo_id` | `local/kassow_pi05_alignment` |
| episodes / frames / fps | 71 / **31936** / 20 |
| shortest episode | 128 frames |
| task string | `push the cardboard up to make space` |
| gripper | **always closed (`0.0`)** — unused on this task; `std=0` |

`--dataset.root` must point at this folder. Instruction at eval = the string above, byte-for-byte.

The motion is **only sliding the cardboard up** so the next skill has space. No object is placed in these demos. Gripper stays closed the whole time (raw `0.7` spikes were encoder glitches and did not survive 20 Hz subsample). If QUANTILES blow up on the zero-range gripper dim, that is LeRobot eps — do not inject fake gripper motion. Idle trim and rotvec unwrap are already applied (44/71 had ~2π jumps before unwrap).

---

## 11. Joint-space dataset (parallel to §3 — do not mix)

Same 180 `.vla` files, **new** LeRobot root. Action is 8D next-frame joints + gripper. Do **not** overwrite `data/lerobot/kassow_pi05`.

On jarvis:

```bash
uv run -m examples.data.convert_to_lerobot_pi05 \
  --input data/kassow_vive_gripper \
  --output data/lerobot/kassow_pi05_joints \
  --repo-id local/kassow_pi05_joints \
  --tasks data/kassow_vive_gripper/tasks.yaml \
  --action-space joints
```

| Field | Value |
|---|---|
| `repo_id` | `local/kassow_pi05_joints` |
| `observation.state` | `(8,)` `j1…j7, gripper` (same as TCP set) |
| `action` | `(8,)` `j1…j7, gripper` — `action[t,:7] = q[t+1]` |
| instruction | same byte-for-byte string as §3.2 |

`lerobot-train` is the §5 command with these substitutions:

```text
--dataset.repo_id=local/kassow_pi05_joints
--dataset.root=/absolute/path/to/kassow_pi05_joints
--output_dir=./outputs/pi05_kassow_joints
--job_name=pi05_kassow_joints
```

Keep `use_relative_actions` **false**. Copy `pretrained_model/` to jarvis as `ckpts/pi05_joints/`. Eval:

```bash
POLICY=Pi05LerobotJointCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path ckpts/pi05_joints \
  --instruction "pick up the blue cylinder and place it on the red plate" \
  --policy-cfg.device cuda:1
```

Offline A (joint ADE in rad):

```bash
uv run --group pi --no-group docs -m examples.diagnose_pi05_offline \
  --policy-path ckpts/pi05_joints --device cuda:1 \
  --dataset-root data/lerobot/kassow_pi05_joints \
  --repo-id local/kassow_pi05_joints \
  --out-dir /tmp/pi05_diag/offline_joints
```

