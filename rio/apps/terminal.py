# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Terminal output that never kills a control loop."""

import os
import sys


def term(*args, **kwargs) -> None:
    """Print to the terminal; a full non-blocking stdout must not kill teleop."""
    try:
        print(*args, **kwargs)
    except BlockingIOError:
        pass


def make_stdout_blocking() -> None:
    """Undo a non-blocking stdout that some terminals leave behind (prints then raise BlockingIOError)."""
    try:
        os.set_blocking(sys.stdout.fileno(), True)
    except (AttributeError, OSError, ValueError):
        pass
