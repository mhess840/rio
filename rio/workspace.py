# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The experiments workspace: everything runs write that is not code (recordings, poses, logs, diagnostics)."""

import os
from pathlib import Path

EXPERIMENTS_DIR_ENV = "RIO_EXPERIMENTS_DIR"


def experiments_root() -> Path:
    """``RIO_EXPERIMENTS_DIR``, else ``experiments/`` in the working directory (the checkout root)."""
    override = os.environ.get(EXPERIMENTS_DIR_ENV)
    return (Path(override).expanduser() if override else Path.cwd() / "experiments").resolve()
