# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def _isolated_kassow_experiments(tmp_path, monkeypatch):
    """Keep tests off the real ``experiments/`` workspace (active experiment, symlinks)."""
    monkeypatch.setenv("RIO_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
