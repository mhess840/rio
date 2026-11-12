# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Cleanup that never replaces the exception that caused it."""

import sys
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def cleanup_step(what: str) -> Generator[None]:
    """Run one cleanup action inside ``finally`` without masking the original failure.

    While another exception is propagating (for example an ``ArmAlarm`` the caller
    classifies), a failing cleanup is logged and the original exception keeps
    propagating. Without a pending exception the cleanup error is raised normally.

    Args:
        what: Short description for the log, e.g. ``"save the recording"``.
    """
    primary = sys.exc_info()[1]
    try:
        yield
    except Exception:
        if primary is None:
            raise
        logger.exception("Could not {} while handling {!r}", what, primary)
