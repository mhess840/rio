#!/usr/bin/env bash
# Build and install the native KORD bridge (_kord_bridge) for this project's interpreter (Linux station).
#
#   bash scripts/setup/kord_bridge.sh
#
# RIO_HW_DIR (default ../rio-hw) and PYTHON (default .venv/bin/python) override the paths.
# The extension keeps an RPATH into the build folder, so do not delete kord_bridge/build afterwards.
# Rebuild after every Python minor upgrade or rio-hw update.
set -euo pipefail

RIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RIO_HW_DIR="${RIO_HW_DIR:-$RIO_DIR/../rio-hw}"
PYTHON="${PYTHON:-$RIO_DIR/.venv/bin/python}"
BUILD_DIR="$RIO_HW_DIR/kord_bridge/build"

if [[ ! -x "$PYTHON" ]]; then
    echo "No interpreter at $PYTHON; run 'uv sync --frozen --extra kassow --group pi' first." >&2
    exit 1
fi
if ! "$PYTHON" -m nanobind --cmake_dir >/dev/null 2>&1; then
    echo "nanobind is missing for $PYTHON; install it with 'uv pip install --python $PYTHON nanobind'." >&2
    exit 1
fi

# kord-api is checked out at the commit rio-hw's gitlink pins, never at a moving branch.
git -C "$RIO_HW_DIR" submodule update --init kord_bridge/kord-api

cmake -S "$RIO_HW_DIR/kord_bridge" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PYTHON"
cmake --build "$BUILD_DIR" --target _kord_bridge -j"$(nproc)"
cmake --install "$BUILD_DIR" --component kord_bridge

cd "$RIO_DIR"
"$PYTHON" - <<'EOF'
import _kord_bridge  # noqa: F401  (fails here, not on the robot, if the build is unusable)

from rio.manifest import check_rio_hw_pin, run_manifest

print("kord_bridge:", run_manifest()["kord_bridge"])
check_rio_hw_pin()
EOF
