# Kassow π₀.₅ pipeline — acquisition, conversion, training, inference

This is the path used on the Kassow + AG95 station. It is not the xArm / OpenPI DROID path (`Pi05Cfg`, `docs/workflow/pi0_droid_finetuning.md`).

Control of the arm is Cartesian (`task_pos_ik`): the policy outputs a TCP pose, and the arm node turns that into joint velocity. The policy’s proprioceptive input is not that TCP pose. It is seven joint angles plus the gripper.

```text
Vive teleop  →  robodm .vla (100 Hz)
                    ↓  rio/data/convert_pi05.py
              LeRobot v3 dataset (20 Hz)
                    ↓  lerobot-train  (cluster, not this machine)
              checkpoint directory
                    ↓  examples/policy_inference.py
              Kassow, Cartesian TCP at 20 Hz, control loop at 100 Hz
```

Python is 3.12. Policy commands use `uv run --group pi --no-group docs`. LeRobot on this machine is 0.6.1.

---

## 1. Data acquisition

**Script:** `examples/teleop_vive_hand.py`  
**Station:** `examples/cfg/kassow_vive_gripper.py` (`KassowViveGripperStation`, extends `examples/cfg/kassow_vive.py`)

```bash
STATION=KassowViveGripperStation uv run -m examples.teleop_vive_hand
```

What it does:

- A Vive tracker gives an absolute wrist pose. Motion is applied only while the clutch is held (`c`). The clutch snapshots the tracker and the current TCP together; later motion is the delta from that pair. There is no fixed SteamVR-to-robot calibration.
- The arm follows in Cartesian space. `KassowViveStation.ArmCfg.robot_controller` is `task_pos_ik` (`ik_kp=12`). The recorded action is the commanded TCP (`arm_target_pose`), not a second measured pose.
- The gripper is an AG95 on serial (`AgGripper`). Open is `0.7`, close is `0.0`. Finger position is read from the hardware at `feedback_freq=2` Hz (`gripper.state / 1000`).
- Two RealSense cameras, color only, 480×640. `camera_1` is serial `112322070077` (wrist). `camera_2` is serial `262522075082` (overview). Each camera thread runs at 30 Hz and writes the newest frame into a ring buffer.
- The teleop loop runs at 100 Hz. On each tick it builds the action, sends it, then calls `env.get_state(action=action)`. That call copies the latest camera frames from the ring buffers and the current joint and gripper readings. Those are not given one shared capture timestamp.
- Recording: `n` starts a trajectory, `s` saves it. Files go to `data/kassow_vive_gripper/` as robodm `traj_XXXX.vla`.

Keys inside a `.vla` that the converter requires:

| Key | Meaning |
|---|---|
| `observation/cameras/camera_1/rgb` | wrist RGB |
| `observation/cameras/camera_2/rgb` | overview RGB |
| `observation/proprio_joints` | 7 joint angles, rad |
| `observation/gripper_position` | AG95 opening, about 0–1 |
| `action` | commanded TCP: xyz (m) + rotvec + gripper |

Language labels are not taken from the `.vla`. They come from `data/kassow_vive_gripper/tasks.yaml`. The current file is one range:

```text
0-428: pick up the blue object and place it on the red plate
```

The index is the number in `traj_XXXX.vla`. The string has to match inference exactly.

---

## 2. Conversion to LeRobot

**Module:** `rio/data/convert_pi05.py`  
**Config:** `ConvertPi05Cfg`  
**Call:** `convert(cfg)`

There is no `python -m` entry point. A run looks like this:

```bash
uv run --group pi --no-group docs python -c "
from pathlib import Path
from rio.data.convert_pi05 import ConvertPi05Cfg, convert
convert(ConvertPi05Cfg(
    input=Path('data/kassow_vive_gripper'),
    output=Path('data/lerobot/kassow_pi05_pick_notrim'),
    tasks=Path('data/kassow_vive_gripper/tasks.yaml'),
    repo_id='local/kassow_pi05_pick_notrim',
    trim_idle=False,
))
"
```

`trim_idle` defaults to `True` (drop the stationary prefix until the TCP has moved 1 cm). The set used for the current runs is `data/lerobot/kassow_pi05_pick_notrim`: 428 episodes, 84407 frames, 20 Hz, idle prefix kept.

What `convert` does to each episode:

1. Load the `.vla` and check the five keys above.
2. Subsample 100 Hz → 20 Hz with stride 5 (`src_fps // dst_fps`). Features are cut on the same index.
3. Unwrap the TCP rotvec inside the episode so a yaw near ±π does not jump by 2π between frames. The encoding stays axis-angle. At inference the rotvec is folded back to `|r| ≤ π`.
4. Optionally trim the idle prefix (`idle_xyz_m=0.01`).
5. Binarize the gripper, on by default.
   - Action: snap to `0.0` or `0.7` around the midpoint `0.35`.
   - State: hysteresis. It becomes closed only at or below about `0.21`, and open again only at or above about `0.46`. The stored state is only ever `0.0` or `0.7`.
6. Attach the task string from `tasks.yaml` by trajectory index.
7. Write a LeRobot v3 dataset. One episode is one MP4 per camera (H.264, 480×640, RGB). `_patch_lerobot_pyav()` works around PyAV 13, which has no `Codec.canonical_name`.

Output columns:

| Key | Shape | Contents |
|---|---|---|
| `observation.images.camera_1` | 480×640×3 video | wrist |
| `observation.images.camera_2` | 480×640×3 video | overview |
| `observation.state` | 8 | `j1…j7`, gripper |
| `action` | 7 | `x, y, z, rx, ry, rz`, gripper |
| `task` | string | from `tasks.yaml` |

`action_space="joints"` is a second mode: the action becomes the next-frame joint vector plus gripper (8). The TCP runs do not use it. The matching inference config is `Pi05LerobotJointCfg` in `examples/policy_cfgs/pi05_lerobot_joint.py`.

---

## 3. Training

Training is not a script in this repo. It is the LeRobot CLI `lerobot-train` on the cluster, LeRobot 0.6.1, base weights `lerobot/pi05_base`.

The checked-in brief `docs/workflow/pi05_kassow_cluster_training.md` describes an older folder (`data/lerobot/kassow_pi05`, 180 episodes, task text “blue cylinder”, `chunk_size=16`). That is not the set or the checkpoint used for the current robot runs.

Current contract:

- Dataset root `data/lerobot/kassow_pi05_pick_notrim`, `repo_id` `local/kassow_pi05_pick_notrim`.
- Task string exactly `pick up the blue object and place it on the red plate`.
- Absolute TCP actions. `use_relative_actions` stays off. Normalization is QUANTILES.
- Two real cameras. `empty_cameras=0`. Do not rename them.
- The checkpoint loaded at inference is `ckpts/pi05`. Its `config.json` has `chunk_size=50` and `n_action_steps=50`. `n_obs_steps=1`: one image pair and one state vector, no history.
- Images stay 480×640 in the dataset. The policy resizes internally.

`--policy.pretrained_path` loads weights only. `chunk_size` and `n_action_steps` must be set on the `lerobot-train` command or they fall back to the LeRobot default of 50. Do not pass `--policy.path=lerobot/pi05_base`; that inherits the base camera keys. Use `--policy.pretrained_path`.

---

## 4. Inference

**Loop:** `examples/policy_inference.py`  
**Station and policy config:** `examples/policy_cfgs/pi05_lerobot.py` (`Pi05LerobotCfg`)  
**Model wrapper:** `rio/policies/pi05.py`  
**Process that keeps the weights loaded:** `rio/policies/pi05_resident.py`

```bash
POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \
  --policy-path ckpts/pi05 \
  --instruction "pick up the blue object and place it on the red plate" \
  --policy-cfg.device cuda:1 \
  --arm-cfg.max-pos-speed 0.3 \
  --arm-cfg.max-rot-speed 0.75 \
  --policy-node-cfg.n-execute 25 \
  --policy-node-cfg.no-use-rtc \
  --policy-node-cfg.chunk-pause-s 0
```

`Pi05LerobotCfg` turns Vive, clicker, and the recorder off. It forces `robot_controller=task_pos_ik`. The control loop is 100 Hz. Policy steps are 20 Hz. `ActionInterpolator` fills the four ticks between policy steps.

Each observation (`create_obs`):

- `camera_1`, `camera_2`: latest RGB from `env.get_state()`
- `proprio`: the seven joint angles concatenated with the gripper
- `task`: the `--instruction` string

`rio/policies/pi05.py` permutes images from HWC to CHW, runs the LeRobot preprocessor, calls `predict_action_chunk`, then the postprocessor. The 32-wide π₀.₅ vector is sliced to 7. The rotvec is folded back to `|r| ≤ π`. With `binarize_gripper=True` (the default, matching conversion) the gripper is snapped to `0.0` or `0.7`.

The chunk is queued and sent as absolute TCP targets. `limit_tcp_action_step` caps how far the command may move in one 100 Hz tick (`max_pos_speed`, `max_rot_speed`), so a jump in the plan becomes a short ramp instead of one huge step.

`--policy-node-cfg.no-use-rtc` is real. `Pi05LerobotCfg.__post_init__` copies that flag onto `policy_cfg.use_rtc`, and the resident is started with the same flag. With RTC off, a new observation is requested only when the action queue is empty. The new chunk is not shifted forward by the inference latency. With RTC on, the leftover tail of the previous chunk is passed in and the queues are merged.

`--policy-node-cfg.n-execute 25` plays 25 of the 50 predicted steps, then replans. `--policy-node-cfg.chunk-pause-s` holds the last command for that many seconds after a chunk ends, before the next image is taken. `0` takes the image as soon as the queue is empty.

`ensure_resident` leaves a daemon on the GPU after the process exits. A later run reuses it only if `use_rtc` and the checkpoint match. `--unload-policy` stops it.

Enter toggles pause. `q` quits.

The default `instruction` inside `Pi05LerobotCfg` is still the older “blue cylinder” sentence. The command above overrides it. The string has to match `tasks.yaml` byte for byte.

---

## 5. Scripts that are not on this path

| Script | Role |
|---|---|
| `examples/diagnose_pi05_offline.py` | Runs the checkpoint on saved LeRobot frames. No robot. |
| `examples/replay_pi05_dataset.py` | Replays a dataset trajectory. |
| `examples/goto_kassow_q.py` | One-shot `moveJ` to a saved joint pose. |
| `examples/policy_cfgs/pi05.py` (`Pi05Cfg`) | OpenPI / DROID-style stack. Different cameras and action layout. |
| `docs/workflow/pi05_kassow_cluster_training.md` | Cluster brief for the older 180-episode, chunk-16, “blue cylinder” set. |
