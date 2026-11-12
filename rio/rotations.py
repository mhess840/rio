# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Rotation helpers shared by datasets, policies, and the rollout."""

import numpy as np
from scipy.spatial.transform import Rotation


def principal_rotvec(rotvecs: np.ndarray) -> np.ndarray:
    """Map axis-angle to the principal value ``|r| ≤ π`` (same SO(3) pose).

    Training may unwrap rotvecs across ±π so the BC loss is continuous. IK
    still wants a standard rotvec; this fold is applied on the policy output.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    if rotvecs.size == 0:
        return rotvecs.astype(np.float32)
    shape = rotvecs.shape
    out = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_rotvec()
    return np.asarray(out, dtype=np.float32).reshape(shape)
