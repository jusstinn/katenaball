"""Possession-length training weights for the team heads (S1 and S2, decision log 2026-11-10).

spellF1 counts every possession once, however long it lasts; an unweighted per-frame loss counts a 60 s
possession thirty times as much as a 2 s one. Here a training frame weighs ``1 / L``, with ``L`` the length in
seconds of the constant-label run it sits in, clipped to [``L_MIN``, ``L_MAX``]. External training labels only.
"""

from __future__ import annotations

import numpy as np

L_MIN, L_MAX = 1.0, 30.0        # seconds; fixed before the first probe run, never tuned


def run_seconds(y: np.ndarray, fps: float) -> np.ndarray:
    """For every frame of a label sequence, the length in seconds of the constant-label run it sits in."""
    y = np.asarray(y)
    if len(y) == 0:
        return np.zeros(0)
    cut = np.flatnonzero(np.diff(y) != 0) + 1
    a, b = np.concatenate(([0], cut)), np.concatenate((cut, [len(y)]))
    return np.repeat((b - a) / float(fps), b - a)


def inv_length(y: np.ndarray, fps: float) -> np.ndarray:
    """``1 / clip(run_seconds, L_MIN, L_MAX)`` for every frame (float64, not normalised)."""
    return 1.0 / np.clip(run_seconds(y, fps), L_MIN, L_MAX)


def at_rows(w_native: np.ndarray, idx: np.ndarray, step: int) -> np.ndarray:
    """Native-frame weights read at the 5 Hz rows ``idx`` (row ``i`` is native frame ``i * step``)."""
    f = np.minimum(np.asarray(idx).astype(np.int64) * int(step), len(w_native) - 1)
    return w_native[f]
