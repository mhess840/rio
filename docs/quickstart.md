# Quick Start

Rio is built around one idea: **separate the hardware description from the control logic**.
A *station* declares every node in your setup; a *script* drives any station without modification.

To see all available components:

```bash
uv run rio-list-stations   
uv run rio-list-robots     
uv run rio-list-cameras     
uv run rio-list-interfaces 
```

---

## How it works

A **station** is a Python dataclass in `examples/cfg/` (Kassow stations: `rio/apps/kassow/stations/`) that declares every hardware node:

- **arm / gripper** — robot driver and gripper
- **cameras** — named dict of camera streams
- **teleop** — input device (motion controller, leader arm, or keyboard)
- **recorder** — saves demonstrations as `.vla` files

The **teleoperation scripts** (`teleop_eef`, `teleop_leader_follower`, `teleop_vive_hand`) read the station config, wire up all components, and run the control loop. The scripts are platform-agnostic — swap the station to use different hardware without touching the script.

---

## 1. Configure a station

Pick the config in `examples/cfg/` closest to your hardware and update the hardware-specific fields: robot IP or serial port, camera serial numbers, and recording path.

!!! note
    The example scripts select the station via the `STATION` environment variable (e.g. `STATION=Xarm7EefStation`). This resolves the class name from `examples/cfg/__init__.py` at startup.

See [Station Configuration](workflow/station_cfg.md) for the full field reference and how to compose your own station from scratch.

---

## 2. Pick a control mode and run

Rio has two teleoperation paradigms. Choose the one that matches your teleop device.

### EEF / Cartesian control

Use this when your input device outputs **motion deltas** (Spacemouse, Gamepad, Keyboard).
The script drives the robot in end-effector space.

```bash
STATION=Xarm7EefStation uv run -m examples.teleop_eef
```

Switch the input device without editing the config:

```bash
STATION=Xarm7EefStation uv run -m examples.teleop_eef --teleop Gamepad
STATION=Xarm7EefStation uv run -m examples.teleop_eef --teleop Keyboard
```

**Teleop modes** (switchable at runtime via a device button):

| Mode | Axes active |
|------|-------------|
| `TRANSLATION` | XYZ only (default) |
| `TRANSLATION_ROTATION` | XYZ + RPY |
| `TRANSLATION_2D` | XY only |
| `ROTATION` | RPY only |

### Leader-follower / joint mirroring

Use this when your input device is itself a **robot arm** that the follower mirrors joint-by-joint (Gello, SO100 leader).

```bash
STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower
STATION=So100Station      uv run -m examples.teleop_leader_follower
```

On startup the script checks that the leader joints are aligned with the follower — fix any misaligned joints before confirming.

### Absolute pose tracking

Use this when your input device reports an **absolute pose** rather than deltas (Vive tracker).
Motion is retargeted relative to a clutch, so nothing needs calibrating between the
tracker frame and the robot. The same loop drives a dexterous hand from a glove.

```bash
STATION=Ur5eViveInspireStation uv run -m examples.teleop_vive_hand
```

On the first run it asks you to sweep a hand along two robot axes. That solves for the yaw
between SteamVR's world frame and the robot base, which is the only frame unknown once
gravity has fixed the vertical axis. The answer is saved to `vive_yaw_cal.json` (path set by `yaw_calibration_file`, relative to the working directory; keep it out of git) and reused;
pass `--calibrate-yaw` to redo it. Calibrate standing where you will actually work, since
the correct angle depends on which way you face.

Press `c` to engage the clutch, which snapshots the tracker pose and the current TCP pose
together; wrist motion is then applied relative to that pair. Press `c` again to release and
re-center. `o` toggles wrist orientation tracking, and `n` / `s` start and save a recording.

The arm moves only while the clutch is engaged, and releases automatically if tracking is
lost or if the command runs ahead of the measured TCP. See
[Vive Tracker](https://github.com/robot-i-o/rio-hw/blob/main/docs/interfaces/vive_tracker.md)
for SteamVR setup. The loop, calibration and key handling live in `rio/apps/vive/` and work for any single-arm station;
station-specific start-pose homing is injected (Kassow: `rio/apps/kassow/homing.py`, `--go-home-on-save`).

The Kassow arm with an AG95 gripper has its own entry point and experiment handling:

```bash
uv run -m rio.apps.kassow.teleop --experiment sort --go-home-on-save
```

See the [Kassow π₀.₅ pipeline](workflow/pi05_kassow_pipeline.md).

Bring this up in stages, since it drives a real arm:

1. Tracker alone — confirm `pose_valid` stays at 1 as you move through the workspace.
2. Hand alone — `STATION=ManusInspireStation uv run -m examples.teleop_leader_follower`.
3. Arm with a low gain — add `--pos-scale 0.25 --max-pos-speed 0.05`, engage briefly, confirm
   the direction of every axis before raising the limits.
4. Full rig at the station defaults.

---

## Example stations at a glance

| Station class | Config file | Control mode | Script |
|---------------|-------------|--------------|--------|
| `Xarm7EefStation` | `examples/cfg/xarm_eef.py` | EEF (Spacemouse) | `teleop_eef` |
| `Xarm7GelloStation` | `examples/cfg/xarm_gello.py` | Leader-follower (Gello) | `teleop_leader_follower` |
| `So100Station` | `examples/cfg/so100.py` | Leader-follower (SO100) | `teleop_leader_follower` |
| `Ur5eViveInspireStation` | `examples/cfg/ur_vive_inspire.py` | Absolute pose (Vive) + hand (Manus) | `teleop_vive_hand` |
| `Ur5eGelloStation` | `examples/cfg/ur_gello.py` | Leader-follower (Gello) | `teleop_leader_follower` |
| `KassowViveInspireStation` | `rio/apps/kassow/stations/kassow_vive_inspire.py` | Absolute pose (Vive) + hand (Manus) | `teleop_vive_hand` |
| `ManusInspireStation` | `examples/cfg/manus_inspire.py` | Hand only (Manus glove) | `teleop_leader_follower` |
| `YamStation` | `examples/cfg/yam.py` | Leader-follower (YAM) | `teleop_leader_follower` |
| `BimanualYamStation` | `examples/cfg/bimanual_yam.py` | Leader-follower bimanual (YAM) | `teleop_leader_follower` |
| `BimanualSo100Station` | `examples/cfg/bimanual_so100.py` | Leader-follower bimanual | `teleop_leader_follower` |
| `G1Station` | `examples/cfg/humanoid.py` | Humanoid whole-body (XRobotoolkit) | `teleop_humanoid` |
| `KassowStation` | `rio/apps/kassow/stations/kassow.py` | EEF (Spacemouse), Kassow arm | `teleop_eef` (`teleop_joint_vel` for `joint_vel`) |
| `KassowViveStation` | `rio/apps/kassow/stations/kassow_vive.py` | Absolute pose (Vive), Kassow arm | `teleop_vive_hand` |
| `KassowViveGripperStation` | `rio/apps/kassow/stations/kassow_vive_gripper.py` | Absolute pose (Vive) + AG95 + 2 RealSense | `rio.apps.kassow.teleop` |

---

## Adding your own station

1. Copy the closest config in `examples/cfg/` and rename the class.
2. Update hardware fields (IP, ports, serials, calibration paths).
3. Register it in `examples/cfg/__init__.py` — add the import and class name to `__all__`.
4. Run with the appropriate script:

```bash
STATION=MyRobotStation uv run -m examples.teleop_eef             # EEF control
STATION=MyRobotStation uv run -m examples.teleop_leader_follower  # joint mirroring
```

---

## Common CLI overrides

Any config field can be overridden on the command line without editing the file:

```bash
STATION=Xarm7EefStation uv run -m examples.teleop_eef \
    --instruction "pick up the can" \
    --visualizer Rerun \
    --freq 30
```

Recordings are saved as `.vla` files to `recorder_cfg.path`.

Everything else a run writes (Kassow experiments, home poses, rollout diagnostics, DAgger rounds, orchestrator probes) goes to the
**experiments workspace**: `$RIO_EXPERIMENTS_DIR`, else `experiments/` in the working directory. It is git-ignored and local.
Every recording and rollout also stores a manifest with the rio and rio-hw versions and git state.

See [Collect Demonstrations](workflow/data_collection.md) for replay, conversion, and next steps.
