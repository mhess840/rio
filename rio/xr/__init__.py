# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Show robot data in a headset (WebXR, e.g. Meta Quest) over the real cell. Display only.

- :mod:`scene`: layers of points, segments, and meshes in the robot base frame.
- :mod:`hub`: the latest layers; :mod:`server` serves them and the page on localhost.
- :mod:`publisher`: other processes (rollout, teleop, graph) push layers without blocking.
- :mod:`registration`: base-to-headset transform from points touched with a controller.

Nothing here commands a robot. The overlay is not a safety function.
"""

from .hub import XrHub
from .publisher import XrPublisher
from .registration import fit_rigid
from .scene import Layer, Lines, Mesh, Points, SceneError
from .server import XrServer

__all__ = ["Layer", "Lines", "Mesh", "Points", "SceneError", "XrHub", "XrPublisher", "XrServer", "fit_rigid"]
