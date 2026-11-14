<p align="center">
  <a href="https://github.com/astral-sh/uv">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" /></a>
  <a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://github.com/astral-sh/ty">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" /></a>
</p>

# RIO: Kassow crate-loading pilot

This repository extends RIO with a Kassow + Vive + AG95 application: demonstration recording, π₀.₅ fine-tuning and serving, human-guided corrections (HG-DAgger), and a Jev-driven orchestrator with alarm-aware restart for a crate-loading pilot at the Juárez plant. Work in progress.

**New here?** Read [docs/architecture.md](docs/architecture.md) first: it maps every concept of the pilot case study to a module. The hands-on guide is [docs/workflow/pi05_kassow_pipeline.md](docs/workflow/pi05_kassow_pipeline.md); the application code is in [rio/apps/kassow/](rio/apps/kassow/).

## About RIO

RIO: flexible real-time Robot I/O for cross-embodiment robot learning.

This project provides a Python-based interface to use different robot arms (Franka, Kinova, Universal Robots, UFACTORY, SO100, ...), grippers, cameras, and teleop interfaces, with built-in support for data collection, teleoperation, and Vision-Language-Action (VLA) policy deployment.

## Setup

Tested on Ubuntu 22.04 LTS with an optional real-time kernel patch. See [the Ubuntu setup guide](docs/setup-tutorials/ubuntu.md).

```bash
git clone https://github.com/robot-i-o/rio.git
cd rio
uv sync --frozen --python 3.12 --group dev
make check
make test
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from its official distribution first.
Clone [rio-hw](https://github.com/mhess840/rio-hw) (branch `kassow-hand`) next to this repository (`../rio-hw`); uv
installs that checkout editable, so RIO always runs its current state. CI checks out the newest commit of the repo and
branch named in `pyproject.toml`. No CUDA is needed. `make test` excludes
physical-hardware, GPU tests, and optional all-backend benchmarks; `make test-pi` runs the same selection with
LeRobot/Torch installed. Symlink tests report explicit skips on Windows
without Developer Mode. `make check` checks lint, formatting, and types without modifying files; `make format`
applies formatting fixes. Type checking covers the RIO package, including the Kassow application in `rio/apps/kassow`.

Kassow/Vive/RealSense hardware and policy dependencies are opt-in. On the station use

```bash
make sync-station    # uv sync --frozen --python 3.12 --extra kassow --group pi --group dev
make smoke-station   # loads every hardware package and parses the Kassow URDF, no hardware opened
```

A plain `uv sync --group dev` removes the hardware packages again.

`uv sync` and `uv run` keep using `../rio-hw`, so the Python driver always matches the `_kord_bridge` built from the
same checkout. Recordings and rollouts store the rio/rio-hw commits and the bridge hash. A warning appears when `../rio-hw` has
uncommitted changes or is behind its upstream branch (as of the last `git fetch`). Rebuild the native Kassow bridge after rio-hw changes
(`bash scripts/setup/kord_bridge.sh`) and run the read-only station preflight
(`pytest -m hardware rio/_tests/test_station_preflight.py`) before supervised robot use. Windows supports the
hardware-free tests, but the resident daemon requires POSIX locking and the complete station targets Linux.

## Documentation

Build and browse the docs locally at http://localhost:8000:

```bash
uv sync --frozen --group docs
uv run mkdocs serve
```

## Citation

If you use RIO in your research, please cite our paper:

```bibtex
@inproceedings{ortega-kral_rio_2026,
  author    = {Pablo Ortega-Kral and Eliot Xing and Arthur Bucker and Vernon Luk and
               Junseo Kim and Owen Kwon and Angchen Xie and Nikhil Sobanbabu and
               Yifu Yuan and Megan Lee and Deepam Ameria and Bhaswanth Ayapilla and
               Jaycie Bussell and Guanya Shi and Jonathan Francis and Jean Oh},
  title     = {RIO: Flexible Real-Time Robot I/O for Cross-Embodiment Robot Learning},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026},
  eprint    = {arXiv:2605.11564},
}
```
