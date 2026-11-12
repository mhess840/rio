# Changelog

Everything that changed in this repository since work started with the assistant (2026-11-09). Newest first.
Part 1 comes from this chat. Part 2 is reconstructed from the assistant's working notes of the earlier sessions, so treat the details there as a summary.

**Verification status:** none of the changes of 2026-11-12 has been run. `uv` was not available on the machine they were made on. Run `make format`, `make check`, `make test` and `make test-pi` on the Linux station before relying on them.

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

- `at_wait_pose` is set when the arm reaches the waiting pose and never cleared, so the approach is skipped on later open cycles. Decide whether that is intended.
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
- Software safety limits in the rollout (position, orientation and joint error, feedback age, inference timeout); alarms stop playback and never acknowledge controller safety. Speed ceilings 0.5 m/s, 1.0 rad/s.
- KORD bridge: stream watchdog, command timeout, alarm handling; `clear_alarms_on_start` defaults to `False`; an old bridge is rejected before connecting. Native changes need a rebuild and supervised validation on the robot host.
- Recording schema version 2 (bimanual proprio is joints plus gripper).
- LeRobot formatter flushes parquet footers explicitly (found on 2026-11-11; full π₀.₅ selection then had a few failing tests that import old `examples.policy_cfgs` paths).

### Packaging and CI

- Pinned `rio-hw` commit, sibling checkout `../rio-hw` installed editable; `uv.lock` tracked; optional `kassow` extra; `make test-pi` and a CI job for it.
- `.gitattributes` for LF line endings; `.gitignore` unignores only `examples/data/convert_to_lerobot_pi05.py`.
- Added OpenJEV support (commit `8a9338e`, 2026-11-09); restructuring commit `5506406` (2026-11-12).

### Verification recorded at the time

- Windows: 424 passed, 29 skipped; Linux (WSL): 428 passed, 25 skipped; ruff, format and ty green (2026-11-10). Not verified: Linux CI run, GPU, native KORD build, hardware.
