# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) defaults shared by chunked policies and the rollout."""

# Guided prefix of the next chunk; must exceed the inference delay (~18 steps at 20 Hz).
RTC_EXECUTION_HORIZON = 18
RTC_MAX_GUIDANCE_WEIGHT = 10.0
# Request the next chunk while this many steps remain.
RTC_QUEUE_THRESHOLD = 18
