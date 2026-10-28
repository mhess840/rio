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
- The gripper is an AG95 on serial (`AgGripper`). Open is `0.7`, close is `0.0`. `gripper_position` is the hardware reading at `feedback_freq=2` Hz; `gripper_commanded_position` is the last successfully written target. A command no longer overwrites the measured opening.
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
| `observation/gripper_commanded_position` | optional, last AG95 target; new recordings preserve it separately |
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

The CLI forwards to the same `convert(cfg)` implementation:

```bash
uv run --group pi --no-group docs -m examples.data.convert_to_lerobot_pi05 \
      --input data/kassow_vive_gripper \
      --output data/lerobot/kassow_pi05_pick_notrim \
      --tasks data/kassow_vive_gripper/tasks.yaml \
      --repo-id local/kassow_pi05_pick_notrim \
      --no-trim-idle \
      --gripper-state-source commanded
```

Choose `--gripper-state-source measured` for genuinely measured proprioception. The default converter source remains `measured` to avoid silently changing older conversions. `commanded` reads the separate recorded command when available, otherwise the recorded action's gripper slot. Use the same source at inference. The existing Kassow policy configuration defaults to `commanded`, matching the predominantly command-like states in the older recordings. This does not turn the binary gripper command into a grasp-success sensor.

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

`Pi05LerobotCfg` turns Vive, clicker, and the recorder off. It forces `robot_controller=task_pos_ik`. The control loop is 100 Hz. Policy steps are 20 Hz. `rio_hw.interpolators.ActionInterpolator` fills the intermediate ticks with linear position interpolation and SO(3) SLERP for TCP orientation. Joint actions remain linear. Principal rotvecs are never linearly blended as ordinary Cartesian components.

Policy startup defaults to `gripper_cfg.startup_position=None`: the driver does not send an initial position target, and playback does not issue a fabricated open command while waiting for inference. This preserves an already-held object for a placement rollout. The interpolation anchor reads the actual gripper state/last command instead of assuming `0.7`. Binary gripper commands switch directly at their waypoint; only an explicitly continuous gripper is interpolated. Custom open/close/threshold values are propagated through inference, startup and playback.

Before a driver has received its first position target, the commanded-position proxy is initialized from the measured opening. Once a target has been written, this channel stays distinct from feedback. It is not a contact or grasp-success flag.

Each observation (`create_obs`):

- `camera_1`, `camera_2`: latest RGB from `env.get_state()`
- `proprio`: the seven joint angles concatenated with the gripper
- `task`: the `--instruction` string

`rio/policies/pi05.py` converts uint8 RGB to float32 `[0, 1]` exactly once, preserves already normalized float images, permutes HWC to CHW, and runs the LeRobot preprocessor. Unnormalized float images and nonfinite inputs are rejected. The output is postprocessed and sliced to seven dimensions. TCP interpolation follows SO(3) even when rotvec components cross the pi branch cut. Commands snap to `0.0` or `0.7`; observation hysteresis is shared with the converter and reset between rollouts.

The chunk is queued and sent as absolute TCP targets. `limit_tcp_action_step` caps how far the command may move in one 100 Hz tick (`max_pos_speed`, `max_rot_speed`), so a jump in the plan becomes a short ramp instead of one huge step.

`--policy-node-cfg.no-use-rtc` is real. `Pi05LerobotCfg.__post_init__` copies that flag onto `policy_cfg.use_rtc`, and the resident is started with the same flag. With RTC off, a new observation is requested only when the action queue is empty. The new chunk is not shifted forward by the inference latency. With RTC on, the leftover tail of the previous chunk is passed in and the queues are merged.

`--policy-node-cfg.n-execute 25` limits the prediction prefix to 25 steps; the default is 30. With RTC, merging replaces the queue and skips only steps actually consumed during inference, so this is not a promise to execute every one of those 25 steps. Guidance never extends into padded, nonexistent leftovers. Without RTC the prefix plays out before requesting the next image. `--policy-node-cfg.chunk-pause-s` adds an optional hold before the next request.

`ensure_resident` leaves the weights resident after the rollout. Reuse requires matching checkpoint file fingerprints, runtime code/library versions, RTC and gripper options. A mismatched authenticated resident must exit before replacement weights are loaded. Startup remains serialized until readiness. IPC keys live in an owner-only runtime directory; use the per-rollout diagnostic folders for sharing rather than copying resident credentials.

Enter toggles pause. `q` quits.

The default instruction matches the current "blue object" dataset. Pass the exact task string for any other checkpoint. Offline diagnosis uses each dataset frame's task label unless explicitly overridden and preserves normalized float pixels without a byte round trip.

### Safety and diagnostics

`--safety-cfg` exposes fail-closed software limits: 5 cm TCP position error, 0.35 rad orientation error, 0.25 rad per-joint error, 0.25 s feedback age and 2 s inference timeout. Alarms, stale feedback/camera frames, invalid chunks and exceeded limits end playback, clear buffered actions and request a stream stop. They never acknowledge controller Safety. Pause/resume also invalidates in-flight inference generations. These guards are not a certified emergency stop or collision detector.

The KORD bridge discards streams after an alarm or a command timeout. Python feedback is only republished when the controller tick advances. Missed control deadlines are skipped, not replayed in a burst, and stale streaming goals are coalesced without dropping stop or discrete requests. `clear_alarms_on_start` now defaults to `False`; resolve controller alarms explicitly before starting.

The optional native `task_pos` fallback retains KORD's XYZ-Euler TCP convention and interpolates orientation through quaternions, including a stable pitch-singularity branch. It does not interpret Euler components as RIO rotation vectors. The default `task_pos_ik` path does not use this fallback. The native changes require compilation and supervised validation on the robot host; analytic formula checks are not a substitute for that build.

Every start/resume creates a unique `session_*` directory below `--diag-dir`, containing the effective configuration, camera snapshots, chunk files and CSVs. The command log records rotation error, alarm code and state age; joint logs label radians rather than metres. Reusing the same diagnostic root cannot overwrite an earlier session.

### Deploying these changes

Update both checkouts. On the robot's Linux machine, rebuild the native bridge against its existing KORD SDK before launching hardware:

```bash
cd ../rio-hw
cmake -S kord_bridge -B kord_bridge/build -DCMAKE_BUILD_TYPE=Release
cmake --build kord_bridge/build --target _kord_bridge -j2
cmake --install kord_bridge/build --component kord_bridge
```

An older `_kord_bridge` is rejected before connecting to the controller. After updating, unload old policy residents and run offline diagnosis with corrected pixels first. A legacy resident that cannot authenticate or terminate causes startup to fail instead of spawning a second GPU process; inspect and stop the verified old process manually in that case. Real robot validation still requires operator supervision, a clear workcell and the physical emergency stop.

Existing `.vla`, LeRobot datasets and checkpoints are not rewritten by these changes. Dataset trimming, geometric augmentation and phase-dependent camera masking remain training/design choices. The restored converter entry point does not invent the unprovided two-task/masking implementation; that path still needs its own matching conversion and deployment contract.

---

## 5. Scripts that are not on this path

| Script | Role |
|---|---|
| `examples/diagnose_pi05_offline.py` | Runs the checkpoint on saved LeRobot frames. No robot. |
| `examples/replay_pi05_dataset.py` | Replays a dataset trajectory. |
| `examples/goto_kassow_q.py` | One-shot `moveJ` to a saved joint pose. |
| `examples/policy_cfgs/pi05.py` (`Pi05Cfg`) | OpenPI / DROID-style stack. Different cameras and action layout. |
| `docs/workflow/pi05_kassow_cluster_training.md` | Cluster brief for the older 180-episode, chunk-16, “blue cylinder” set. |
