# Changelog

Everything that changed in this repository since work started with the assistant (2026-11-09). Newest first.
Part 1 comes from this chat. Part 2 is reconstructed from the assistant's working notes of the earlier sessions, so treat the details there as a summary.

**Verification status:** On 2026-11-12 the Linux station ran `make check` (ruff, format and `ty check` clean) and the pytest selection of `make test` and `make test-pi`: 479 passed, 17 skipped, 12 deselected. `CUDA_VISIBLE_DEVICES=0`. GPUs 1–3 stayed with the other user's training job. Not run: hardware, GPU-marked inference, RLDS.

---

## 2026-11-13

### Added

- Headset view (Meta Quest over WebXR), display only, nothing in it moves the arm:
  - `rio/xr/`, robot-independent: `Layer` of `Points`, `Lines`, `Mesh` in the base frame; `XrHub` (latest layers, versions); `XrServer` on `127.0.0.1` with the page, `GET /scene`, server-sent `changes` events, stateless `POST /register` (Kabsch fit of touched points), and token-protected `POST /layers`; `XrPublisher` for other processes, non-blocking, newest layer per name wins, never through an HTTP proxy.
  - `rio/xr/web/`: plain WebGL2 + WebXR page without third-party code. Desktop preview with orbit view; in the headset, registration by touching the TCP at three or more saved poses (trigger captures, grip undoes). Layer toggles and the registration are kept in the browser.
  - `uv run python -m rio.apps.kassow.xr`: robot base axes, saved-pose TCPs, teleop limit shells, the calibrated table area, and the area reachable with the gripper pointing down (IK flood fill per height, cached in `experiments/xr/`). The headset connects with `adb reverse tcp:8770 tcp:8770`.
  - `goto.solve_ik`: the IK acceptance rules of `move_tcp`, shared with the reach computation.
- Graph as Policy can locate and grasp an object on the table:
  - `rio/apps/kassow/graph/locate.py`: `locate_questions` (Jev noul `visible` plus box `where`), `read_box` and `read_noul` with range and order checks.
  - `rio/apps/kassow/graph/table_map.py`: `TableMap` (homography from Jev image fractions to base `x, y`, grasp-height plane, mean grasp orientation, calibrated area, IK seeds). `label` asks Jev to box the object in each sample photo; `fit` refuses the map above 20 mm leave-one-out error or 10 mm grasp-height deviation.
  - Node `locate` (`object`, `store`; edges `found`, `missing`, optional `fail`) and skill `pick_at` (`target`, `approach_m` 0.03–0.25, `z_offset_m` 0–0.10; edges `ok`, `fail`). `check` requires every `pick_at` target to be stored by a `locate` node.
  - `KassowSkills.pick_at`: open, `moveJ` above the target, check the measured TCP, descend to the calibrated height, close, lift, and report whether the object is still held. `goto.move_tcp` adds the URDF-to-controller TCP offset and refuses IK answers that miss, leave joint limits or turn a joint by more than 180°.
  - `run --no-dry-run` waits for space before each `pick_at` (`--no-confirm` turns this off). `run` needs a Jev-camera table map with the current `JEV_SERIAL` for graphs that locate.
  - `rio/apps/kassow/graph/pick_cube.json` example.
- Graph as Policy as a task-independent planning interface for an agent (the crate orchestrator is unchanged):
  - Commands `--command describe` (node kinds, limits, poses, VLAs with instructions and usability, table-map status), `look` (one photo each from the Jev, overview and wrist cameras), `ask` (Jev question map or `--locate`, with base coordinates on Jev photos). They print JSON. A failed `check` now ends with the problem list instead of a traceback.
  - Skills `gripper` (`state` open or close, no object check), `place_at` (refused while nothing is held; `z_offset_m` 0–0.15), `move_above` (`height_m` 0.03–0.30).
  - Node `vla` (`experiment`, `instruction`, `timeout_s` 5–300): moves to the experiment's start pose, then runs the orchestrator's own `make_run_vla` with its stop rules. `check` refuses VLAs whose overview camera differs from the graph station's (currently `pick`) and `open` without one of its instructions.
  - On the cell, every target move and every VLA waits for space. Each run log holds `graph.json` next to `events.jsonl`.
  - `rio/apps/kassow/graph/pick_place.json` example and `.github/agents/robot.agent.md`: a VS Code agent that runs describe → look/ask → graph → check → dry run → run (only after the user agrees) → log, and never edits code.

### Changed

- `graph/calibrate.py`: one sample takes two presses of `k`. The first stores the TCP while the object rests in the closed gripper, the second stores the photos after the arm has moved away. Samples are `NNN/pose.json` plus `overview.jpg` and `jev.jpg`; `poses.jsonl` is no longer written.

### Fixed

- `graph/check.py`: an edge to a missing node made the reachability check raise `KeyError` instead of reporting the problem.

### Open

- Not run on the cell: calibration, Jev box accuracy, `move_tcp`, `pick_at`, `place_at`, `move_above`, `look`, and `vla` nodes. The descent is a joint-space `moveJ`, not a straight line.
- Samples recorded with the old one-press `calibrate` are not read.
- MCP server around `describe`, `look`, `ask`, `check` and `run`; wrist-camera refinement of a grasp.

---

## 2026-11-12

### Renamed and moved

| Before | After |
|---|---|
| `Xarm7EEFStation` | `Xarm7EefStation` |
| `SO100Station` | `So100Station` |
| `BimanualSO100Station` | `BimanualSo100Station` |
| `Ur5eTeleopStation` (`examples/cfg/ur_gello.py`) | `Ur5eGelloStation` |
| `examples/cfg/yam_station.py` | `examples/cfg/yam.py` |
| `examples/cfg/bimanual_yam_station.py` | `examples/cfg/bimanual_yam.py` |
| `examples/cfg/kassow_vive_inspire.py` | `rio/apps/kassow/stations/kassow_vive_inspire.py` (exported from `rio.apps.kassow.stations`) |
| `rio/apps/kassow/board.py` | `rio/apps/kassow/orchestrator/board.py` |
| `scripts/exp/gen_xarm_gello.sh` | `scripts/setup/gello/gen_xarm_gello.sh` (`scripts/exp/` removed) |
| `docs/workflow/data-collection.md` | `docs/workflow/data_collection.md` |
| `docs/workflow/data-export.md` | `docs/workflow/data_export.md` |
| `Debug/` | `debug/`; `.gitignore` entry `/debug/` |

**Migration:** `STATION=<OldName>` commands must use the new class names. The station names printed by `uv run -m examples.cfg` are the new ones.

### Added

- `rio/apps/kassow/skills.py`: robot skills `move_to`, `lift`, `pick` and `place` (`KassowSkills`), plus `DrySkills` for the dry run. `pick` returns true only when an object blocks the jaws. All skills return `True`, `False`, `"alarm"` or `"alarm_holding"`.
- `rio/apps/kassow/hardware.py`: controller IP, gripper port and camera serials in one place. Environment overrides: `RIO_KASSOW_IP`, `RIO_GRIPPER_PORT`, `RIO_WRIST_SERIAL`, `RIO_OVERVIEW_SERIAL`, `RIO_JEV_SERIAL`. A command-line flag still wins.
- `Move` (queued `moveJ` with optional lift first) and `Snapshot` (read-only view for observers) in `orchestrator/machine.py`.
- `validate_url` in `rio/jev/client.py`.
- Tests: `rio/_tests/test_kassow_skills.py`; `test_failed_approach_does_not_redirect_a_later_move` in `test_orchestrator_kassow.py`; URL and credential tests in `test_jev_client.py`.
- `docs/architecture.md`: layer diagram, orchestrator state diagram, case-study-to-code table, reading order, dry-run instructions, "Same name, different job" table, security notes.
- `README.md` has a new title and intro for the Kassow pilot, with a pointer to `docs/architecture.md`. `AGENTS.md` has a project-map line. `mkdocs.yml` has an architecture entry.
- `docs/quickstart.md`, `station_cfg.md`, `data_collection.md`, `workflow/index.md`: all registered stations, the Kassow entry point, the experiments workspace, and a Kassow + Vive + AG95 tab.

### Changed

- `orchestrator/machine.py`: the flags `goto_path`, `lift_before_goto` and `approach_space` are replaced by a queue `moves`. `step()` dispatches through a handler table (`_idle`, `_wait_proceed`, `_held` plus the existing state methods). The docstring names it the "task graph" of the case study.
- `orchestrator/actions.py` and `run.py` use `KassowSkills` / `DrySkills`. Removed `make_goto`, `make_lift`, `dry_goto` and `lift_skipped`.
- `orchestrator/board.py` reads `orchestrator.snapshot()` instead of orchestrator attributes.
- Kassow stations and `experiments.py` take IP, port and serial defaults from `hardware.py`.
- Jev spelling: "Jev" for the client, API and UI texts, "OpenJev" for the local helper. Board texts changed to "Ask Jev" and "Jev"; the test asserts `">Ask Jev<"`.
- Personal absolute paths removed from defaults:
  - `ur_gello.py` and `xarm_gello.py`: `teleop_cfg_yaml` defaults to `None`.
  - `so100.py` and `bimanual_so100.py`: calibration files come from `~/.cache/huggingface/lerobot/calibration/robots/so100_follower`; the bimanual recorder path is `data/xlerobot/`.
  - `scripts/openjev_serve.sh`: `OPENJEV_ROOT` must be set.
- `docs/CONTRIBUTING.md`: `make check` and `make test` instead of `pytest -n 4`.
- `docs/workflow/pi05_kassow_pipeline.md`: intro restructured into Environment, Code layout, Experiments workspace, Error handling, Orchestrator limits, Checks and tests. Change-history wording removed. `--diag-dir` default described relative to the experiments workspace.
- `docs/workflow/deploy.md`: same `--diag-dir` wording.

### Fixed

- Orchestrator: a failed approach to the waiting pose left `approach_space` set, so a later unrelated `moveJ` could be redirected to a compartment. The queue is cleared on failure.
- `at_wait_pose` is set only while the arm is at the waiting pose. A successful `moveJ` to any other target clears it, and so does an alarm. The next open then queues the approach again.
- The organized noul is no longer mixed with a 0.9 standing prior. At or below 0.30 the inlay is reorganized, at or above 0.70 the count is used, and a value between those holds.

### Security

Checked for hard-coded secrets: none found. Credentials come from `TYPESAFE_API_KEY` and `SHIM_TOKEN`.

- `rio.jev` opened any URL scheme through `urllib` (including `file://` and `ftp://`) and could send the bearer token over plain HTTP to another host. It now accepts only `http(s)` and refuses credentials over plain HTTP unless the host is loopback. `openjev_crate.ask` checks the scheme too.
- Reviewed without change: the demo page binds to `127.0.0.1` and serves three fixed paths; the resident process uses an owner-only directory, a `0600` socket and a random key.
- Known residual risk: the resident IPC unpickles messages after authentication; the hosted Jev backend sends crate photos outside the network.

### Removed

- `Untitled` (10-byte stray file in the repository root).
- `Debug/` from the git index (`git rm --cached`); the files stay on disk and are ignored.
- Dead references to `docs/workflow/pi05_kassow_cluster_training.md`.

### Open

- No workflow step calls `pick` or `place` yet; insertion is manual (`WAIT_PROCEED`). Needs defined pick and place poses.
- Case study says "Schunk" gripper; code and docs drive an AG95 (`AgGripper`).
- The case study says "2 x Kassow"; the repository has one Kassow station.
- Machine-specific values are still defaults in code (overridable by environment variable); no local config file, because a file would override command-line flags.

---

## 2026-11-09 to 2026-11-11 (from the assistant's notes)

### Structure

- Kassow application moved to `rio/apps/kassow/` (experiments, crate, orchestrator, goto, dagger, teleop, `jev_ask`, `openjev_crate`, `experiment_cli`, `convert`, `serve`, stations, policy configs, packaged homes, inlay photos). `examples/*` became thin forwarders.
- Station-independent Vive teleop in `rio/apps/vive/`; `rio/apps/terminal.py`.
- Rollout runtime split into `rio/policies/rollout/` (settings, safety, chunks, actions, loop, diagnostics, control, observation). Generic serve CLI in `rio/policies/serve.py`.
- Resident policy process split into `rio/policies/pi05_resident/`; named slots keep `open` and `organize` separate.
- Converter split into `rio/data/convert_pi05/` (schema, config, tasks, episode, lerobot_io, pipeline).
- `rio/manifest.py` (versions and commits next to recordings and rollouts), `rio/cleanup.py` (`cleanup_step`), `rio/workspace.py` (`RIO_EXPERIMENTS_DIR` or `experiments/`), `rio/envs/drivers.py` (driver import check by AST scan).

### Behaviour

- Orchestrator always asks the full question set, never treats a timeout as success, stops after `--max-organize-attempts` organize runs (`j` in the hold grants a fresh budget), restarts from the interrupted phase after an alarm.
- Jev answers are validated as finite probabilities; view and inlay fields fail closed.
- Recorder latches write errors and publishes a save completion ID; DAgger refuses to publish a failed recording. Diagnostics are written through a bounded background queue, one `session_*` folder per start.
- π₀.₅ input: uint8 images converted to float32 once; SO(3) interpolation of TCP orientation; one shared gripper hysteresis between converter and live loop; gripper keeps a held object at startup.
- Software safety limits in the rollout (position, orientation and joint error, feedback age, inference timeout); alarms stop playback and never acknowledge controller safety. Reviewed speed ceilings are 0.5 m/s, 1.0 rad/s and 1.0 rad/s per joint in `rio/policies/rollout/safety.py`. The Kassow policy config runs at 0.30 m/s, 0.40 rad/s and 0.40 rad/s per joint; see `docs/workflow/pi05_kassow_pipeline.md`.
- KORD bridge: stream watchdog, command timeout, alarm handling; `clear_alarms_on_start` defaults to `False`; an old bridge is rejected before connecting. Native changes need a rebuild and supervised validation on the robot host.
- Recording schema version 2 (bimanual proprio is joints plus gripper).
- LeRobot formatter flushes parquet footers explicitly (found on 2026-11-11; full π₀.₅ selection then had a few failing tests that import old `examples.policy_cfgs` paths).

### Packaging and CI

- Pinned `rio-hw` commit, sibling checkout `../rio-hw` installed editable; `uv.lock` tracked; optional `kassow` extra; `make test-pi` and a CI job for it.
- `.gitattributes` for LF line endings; `.gitignore` unignores only `examples/data/convert_to_lerobot_pi05.py`.
- Added OpenJEV support (commit `8a9338e`, 2026-11-09); restructuring commit `5506406` (2026-11-12).

### Verification recorded at the time

- Windows: 424 passed, 29 skipped; Linux (WSL): 428 passed, 25 skipped; ruff, format and ty green (2026-11-10). Not verified: Linux CI run, GPU, native KORD build, hardware.
- Linux station, 2026-11-12: ruff, format and `ty check` clean. Pytest `not hardware and not gpu and not benchmark`: 479 passed, 17 skipped, 12 deselected, with the pi group installed. `CUDA_VISIBLE_DEVICES=0`; cards 1–3 were occupied by another user's job. Not run: hardware, GPU-marked inference, RLDS.
