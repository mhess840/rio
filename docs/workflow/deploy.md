# Policy Inference

Run a fine-tuned VLA policy on the robot in closed-loop at a fixed frequency.

## 1. Install policy dependencies

Policy inference needs `openpi` and `lerobot`. They're opt-in (the install is large and conflicts with the `docs` extra).

Hardware and π₀.₅ share the Python **3.12** `.venv` (not 3.14). Pin with `.python-version`.

```bash
uv python pin 3.12
uv sync --python 3.12 --group pi --no-group docs
```

`lerobot==0.6.1` extras `[pi,training]`. After a Python minor change, rebuild `_kord_bridge` against this interpreter (`cmake -DPython_EXECUTABLE=$PWD/.venv/bin/python` in `../rio-hw/kord_bridge`).

A 0.6 checkpoint will not load on 3.11 / 0.4.4.

## 2. Pick a station and policy

| Env var | Purpose | Where defined |
|---------|---------|---------------|
| `STATION` | Robot hardware config (arm, gripper, cameras) | `examples/cfg/` |
| `POLICY`  | Policy config (checkpoint, transforms, action space) | `examples/policy_cfgs/` |

**Available policies:**

| `POLICY` | Model | Notes |
|----------|-------|-------|
| `Pi05Cfg` | [openpi](https://github.com/Physical-Intelligence/openpi) pi0.5 (DROID-style) | 3 cameras, 32-dim actions, `joint_vel` |
| `Pi05LerobotCfg` | LeRobot π₀.₅ on Kassow + AG95 | 2 cameras, 7D TCP+gripper, `task_pos_ik`, 20 Hz |
| `Pi05LerobotJointCfg` | Same station, joint-trained π₀.₅ | 8D `j1…j7`+gripper, `moveJ`. Rejects a TCP checkpoint. |
| `SmolVLACfg` | [lerobot SmolVLA](https://github.com/huggingface/lerobot) | 2 cameras, `joint_pos` |

Kassow LeRobot checkpoint: copy the cluster `pretrained_model/` folder to `ckpts/pi05/` so that `ckpts/pi05/config.json` sits at the root (not one directory deeper). Instruction must match the dataset string.

```bash
# optional: load-only, no robot
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs python -c \
  "from rio.policies.pi05 import Pi05; p=Pi05('ckpts/pi05'); p.construct_policy(); print('ok', p.policy_cfg.chunk_size)"

POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path ckpts/pi05 \
  --instruction "pick up the blue cylinder and place it on the red plate"
```

Chunk leftovers and `cmd` vs measured TCP go to `/tmp/pi05_diag/` (`chunks/chunk_XXXX.npy`, `chunks.csv`, `cmd_vs_eef.csv`). Override with `--diag-dir`.

**RTC off** (play each 50-step chunk to the end; E-Stop ready — first target can jump):

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path ckpts/pi05 \
  --instruction "pick up the blue cylinder and place it on the red plate" \
  --policy-cfg.device cuda:1 \
  --policy-node-cfg.no-use-rtc
```

Pass `--policy-node-cfg.no-use-rtc` (tyro boolean; copied onto the policy). Chunks.csv `span_m` is the model plan **before** IK.

**Joint-space serve** (needs `ckpts/pi05_joints` trained on `data/lerobot/kassow_pi05_joints`; see [joint-space concept](pi05_joint_space_concept.md)):

```bash
POLICY=Pi05LerobotJointCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path ckpts/pi05_joints \
  --instruction "pick up the blue cylinder and place it on the red plate" \
  --policy-cfg.device cuda:1
```

**Offline open-loop vs dataset GT** (no arm):

```bash
uv run --group pi --no-group docs -m examples.diagnose_pi05_offline \
  --policy-path ckpts/pi05 --device cuda:1
```

Plots and `summary.csv` land in `/tmp/pi05_diag/offline/`. xyz ADE ≲ 2 cm and pred span ≳ 5 cm means the checkpoint has a directed plan on demo frames.

**GT demo through the same IK** (no policy, same interpolator ×5):

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.replay_pi05_dataset \
  --episode 0
```


AG95 is two-state (`0.0` / `0.7`). Convert and inference share `binarize_gripper` (default on). Analog gripper later: reconvert with `--no-binarize-gripper` and run `--policy-cfg.binarize-gripper false`. See [π₀.₅ cluster brief](pi05_kassow_cluster_training.md) §3.3.

List available stations and policies:

```bash
uv run -m examples.cfg
uv run -m examples.policy_cfgs
```

## 3. Point the policy at your checkpoint

The three fields you'll typically set live at the top of each policy config:

| Field | Pi05Cfg | SmolVLACfg |
|-------|---------|------------|
| `policy_path` | Dir with `params/` + `assets/` | lerobot checkpoint dir |
| `asset_id`    | Subfolder under `<policy_path>/assets/` containing `norm_stats.json` | — |
| `instruction` | Language prompt for the task | Language prompt (optional) |

You can either edit the defaults in [`examples/policy_cfgs/pi05.py`](../../examples/policy_cfgs/pi05.py) / [`smolvla.py`](../../examples/policy_cfgs/smolvla.py), or override them on the CLI (see below).

The `action_space` in the policy config automatically sets the arm controller — no need to edit station files.

## 4. Run

```bash
# pi0.5
STATION=Xarm7EEFStation POLICY=Pi05Cfg uv run -m examples.policy_inference \
  --policy-path /data/ckpt/pi05_multi_fold \
  --asset-id robotio/multirobot_fold \
  --instruction "Fold the shirt in half."

# SmolVLA
STATION=Xarm7EEFStation POLICY=SmolVLACfg uv run -m examples.policy_inference \
  --policy-path ckpts/smolvla \
  --instruction "Place the cup on the shelf."
```

Press **Enter** when prompted to start the loop.

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | from config | Checkpoint directory |
| `--asset-id` (pi0.5) | from config | Norm-stats asset id |
| `--instruction` | from config | Language instruction |
| `--freq` | `50` | Control frequency (Hz) |
| `--visualizer` | `None` | Set to `Rerun` for live visualization |
| `--policy-node-cfg.chunk-size` | `16` | Actions per chunk |
| `--policy-node-cfg.chunk-request-threshold` | `0.1` | Fraction consumed before next chunk request |
| `--policy-node-cfg.no-use-rtc` | default on (`Pi05LerobotCfg`) | Disable RTC; play each chunk to the end |
| `--diag-dir` | `/tmp/pi05_diag` | Chunk npy + cmd-vs-eef CSV |

## Inference loop

1. Collect an observation (camera images + proprioception)
2. Send it to the policy server
3. Receive an action chunk and step through it
4. Request a new chunk before the current one is exhausted
