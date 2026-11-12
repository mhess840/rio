# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def _isolated_kassow_experiments(tmp_path, monkeypatch):
    """Keep tests off the real ``experiments/`` workspace (active experiment, symlinks)."""
    monkeypatch.setenv("RIO_EXPERIMENTS_DIR", str(tmp_path / "experiments"))


@pytest.fixture
def symlink_capable(tmp_path):
    """Require real directory links only for tests that exercise link behavior."""
    target = tmp_path / "symlink_probe_target"
    target.mkdir()
    try:
        (tmp_path / "symlink_probe").symlink_to(target, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows directory symlinks require Developer Mode or additional privileges")
        raise
