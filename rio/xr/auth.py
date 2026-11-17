# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The publish token: only processes of this user may change what the headset shows."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from rio.workspace import experiments_root


def token_path() -> Path:
    """Default token file of the XR server in the experiments workspace."""
    return experiments_root() / "xr" / "publish_token"


def write_token(path: Path) -> str:
    """Create a fresh random token, readable only by the owner on POSIX. Returns it."""
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(token)
    if os.name == "posix":
        os.chmod(path, 0o600)
    return token


def read_token(path: Path) -> str:
    """The token written by the running server.

    Raises:
        FileNotFoundError: No XR server has written a token there.
    """
    return path.read_text().strip()
