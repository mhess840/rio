# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Meta SAM 3.1 Model API: text-prompted instance segmentation (``docs/sam-api.md``)."""

from .client import Instance, SamClient, SamError, SamReply, api_key, parse_frames

__all__ = ["Instance", "SamClient", "SamError", "SamReply", "api_key", "parse_frames"]
