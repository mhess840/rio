# Policy Inference

Run a fine-tuned VLA policy on the robot in closed-loop at a fixed frequency.

## 1. Install policy dependencies

Policy inference needs `openpi` and `lerobot`. They're opt-in (the install is large and conflicts with the `docs` extra).

Hardware and π₀.₅ share the Python **3.12** `.venv` (not 3.14). Pin with `.python-version`.

```bash
uv python pin 3.12
uv sync --python 3.12 --group pi --no-group docs
```

`lerobot==0.6.1` extras `[pi,training]`. After a Python minor change, rebuild `_kord_bridge` against this interpreter (`bash scripts/setup/kord_bridge.sh`, which passes `-DPython_EXECUTABLE=$PWD/.venv/bin/python` to CMake in `../rio-hw/kord_bridge`).

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

The Kassow LeRobot configs take the model, instruction, cameras, and home from the active experiment. Experiments, safe serve flags, offline checks, and joint-space serving are in the [Kassow π₀.₅ pipeline](pi05_kassow_pipeline.md).

```bash
uv run -m examples.experiment use sort --model run1
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve \
  --policy-cfg.device cuda:1 \
  --instruction "put the object in space 2"
```

`rio.apps.kassow.serve` offers only the Kassow configs and adds DAgger takeover; `examples.policy_inference` serves
any `POLICY` from `examples/policy_cfgs/` through the same `rio.policies.serve` code.

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
STATION=Xarm7EefStation POLICY=Pi05Cfg uv run -m examples.policy_inference \
  --policy-path /data/ckpt/pi05_multi_fold \
  --asset-id robotio/multirobot_fold \
  --instruction "Fold the shirt in half."

# SmolVLA
STATION=Xarm7EefStation POLICY=SmolVLACfg uv run -m examples.policy_inference \
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
| `--policy-node-cfg.no-use-rtc` | RTC on (`Pi05LerobotCfg`) | Disable RTC; play `n_execute` steps, then request the next chunk |
| `--diag-dir` | `<experiments workspace>/pi05_diag` (`$RIO_EXPERIMENTS_DIR`, else `experiments/`) | Chunk npy + cmd-vs-eef CSV, one `session_*` folder per rollout |

## Inference loop

1. Collect an observation (camera images + proprioception)
2. Send it to the policy server
3. Receive an action chunk and step through it
4. Request a new chunk before the current one is exhausted
