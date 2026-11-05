"""Who controlled each frame of a DAgger rollout, and which frames become training episodes.

A rollout ``.vla`` stores ``meta/control_source`` per frame: the policy drove,
the operator drove (after a takeover), or the arm held still while the policy
planned its first chunk after a handback. ``dagger_segments`` turns that column
into ``[start, end)`` frame ranges, each written as its own LeRobot episode.
"""

from enum import IntEnum
from itertools import pairwise

import numpy as np

CONTROL_KEY = "meta/control_source"
SIDECAR_SUFFIX = ".dagger.json"
SEGMENT_MODES = ("whole", "human", "all")
# Commanded TCP slower than this counts as standing still.
STILL_POS_SPEED = 0.01  # m/s
STILL_ROT_SPEED = 0.05  # rad/s
# A gripper command step larger than this is a switch (0.0 ↔ 0.7).
GRIPPER_SWITCH = 0.1


class ControlSource(IntEnum):
    """Per-frame value of ``meta/control_source``."""

    POLICY = 0
    HUMAN = 1
    HOLD = 2


def control_runs(control: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a control column into ``(value, start, end)`` runs of equal value."""
    control = np.asarray(control).reshape(-1)
    if control.size == 0:
        return []
    edges = np.flatnonzero(np.diff(control)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [control.size]])
    return [(int(control[start]), int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def takeover_indices(control: np.ndarray) -> list[int]:
    """First frame of every operator run that follows policy or hold frames."""
    return [start for value, start, _ in control_runs(control) if value == ControlSource.HUMAN and start > 0]


def control_summary(control: np.ndarray, fps: float) -> dict:
    """Frame counts per source, takeovers, and handbacks of one rollout."""
    control = np.asarray(control).reshape(-1)
    runs = control_runs(control)
    human_runs = [run for run in runs if run[0] == ControlSource.HUMAN]
    handbacks = sum(1 for prev, nxt in pairwise(runs) if prev[0] == ControlSource.HUMAN and nxt[0] != ControlSource.HUMAN)
    return {
        "frames": {source.name.lower(): int(np.sum(control == source)) for source in ControlSource},
        "takeovers": len(human_runs),
        "handbacks": handbacks,
        "duration_s": round(control.size / float(fps), 3) if fps > 0 else None,
    }


def still_frames(action: np.ndarray, fps: float) -> np.ndarray:
    """Frames whose commanded TCP (xyz + unwrapped rotvec) barely moves to the next frame."""
    action = np.asarray(action, dtype=np.float64)
    if len(action) < 2:
        return np.ones(len(action), dtype=bool)
    rate = np.diff(action[:, :6], axis=0) * float(fps)
    still = (np.linalg.norm(rate[:, :3], axis=1) < STILL_POS_SPEED) & (np.linalg.norm(rate[:, 3:6], axis=1) < STILL_ROT_SPEED)
    return np.append(still, still[-1])


def operator_pauses(control: np.ndarray, action: np.ndarray, *, fps: float, max_pause_s: float) -> np.ndarray:
    """Operator frames where the arm only stood still because of the handover.

    In each operator run this marks the still start (reaction time after the
    takeover, before the hand moves) and every still stretch of at least
    ``max_pause_s`` (clutch released to re-grip, hesitation). A stretch with a
    gripper switch is kept, and so is the still end of the episode, which the
    demos have too.

    Args:
        control: ``meta/control_source`` at ``fps``.
        action: Commanded 7D TCP + gripper at ``fps`` (rotvec unwrapped).
        fps: Frame rate.
        max_pause_s: Shortest inner pause that is cut.

    Returns:
        Boolean mask, True for frames to drop.
    """
    control = np.asarray(control).reshape(-1)
    action = np.asarray(action, dtype=np.float64)
    n = int(control.size)
    still = still_frames(action[:n], fps)
    switch = np.zeros(n, dtype=bool)
    steps = np.flatnonzero(np.abs(np.diff(action[:n, -1])) > GRIPPER_SWITCH)
    switch[steps] = True
    switch[np.minimum(steps + 1, n - 1)] = True
    min_pause = max(1, round(max_pause_s * fps))
    drop = np.zeros(n, dtype=bool)
    for value, start, end in control_runs(control):
        if value != ControlSource.HUMAN:
            continue
        first = start
        while first < end and still[first] and not switch[first]:
            first += 1
        drop[start:first] = True
        for is_still, rel_start, rel_end in control_runs(still[first:end].astype(np.int8)):
            s, e = rel_start + first, rel_end + first
            if not is_still or e - s < min_pause or switch[s:e].any() or e == n:
                continue
            drop[s:e] = True
    return drop


def dagger_segments(
    control: np.ndarray,
    *,
    mode: str,
    fps: float,
    include_autonomous: bool = False,
    pre_takeover_cut_s: float = 1.0,
    min_segment_s: float = 1.0,
    action: np.ndarray | None = None,
    max_pause_s: float | None = None,
) -> list[tuple[int, int]]:
    """Frame ranges of one rollout that go into training.

    Args:
        control: ``meta/control_source`` after subsampling to ``fps``.
        mode: ``whole`` keeps everything; ``human`` keeps the operator runs only
            (HG-DAgger); ``all`` keeps policy and operator frames but drops hold
            frames and the ``pre_takeover_cut_s`` before each takeover, where the
            policy was already going wrong.
        fps: Frame rate of ``control``.
        include_autonomous: Keep a rollout without any takeover as one segment.
        pre_takeover_cut_s: Seconds dropped before each takeover in ``all`` mode.
        min_segment_s: Shorter ranges (e.g. an accidental clutch tap) are dropped.
        action: Commanded TCP + gripper at ``fps``; needed for ``max_pause_s``.
        max_pause_s: If set, cut handover stillness from operator runs
            (see ``operator_pauses``).

    Returns:
        Sorted, non-overlapping ``(start, end)`` ranges.
    """
    if mode not in SEGMENT_MODES:
        raise ValueError(f"segments must be one of {SEGMENT_MODES}, got {mode!r}")
    control = np.asarray(control).reshape(-1)
    n = int(control.size)
    min_len = max(1, round(min_segment_s * fps))
    if mode == "whole":
        return [(0, n)] if n else []
    human = control == ControlSource.HUMAN
    if not human.any():
        return [(0, n)] if include_autonomous and n >= min_len else []
    if mode == "human":
        keep = human
    else:
        keep = control != ControlSource.HOLD
        cut = max(0, round(pre_takeover_cut_s * fps))
        for start in takeover_indices(control):
            keep[max(0, start - cut) : start] = False
    if action is not None and max_pause_s is not None:
        keep &= ~operator_pauses(control, action, fps=fps, max_pause_s=max_pause_s)
    return [(start, end) for value, start, end in control_runs(keep.astype(np.int8)) if value and end - start >= min_len]
