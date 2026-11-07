"""The starter baseline (`starter/baseline.py`) re-implemented on a `Match`.

1. Each frame where the ball is seen: the team of the nearest tracked slot (any slot, ghosts and
   duplicates included, exactly as the starter does). Ball not seen: nobody.
2. Centred majority vote over +/- `window` frames (50 = the published floor variant), H on ties,
   D where nobody was seen in the window. Applied over the whole timeline, like the starter.

Graded frames get H/A/D; frames outside the halves get X (ignored by the scorer either way).
"""
from __future__ import annotations

import numpy as np

from ..match import A, D, H, X, Match

NOBODY = -1


def nearest_slot_team(m: Match, chunk: int = 50_000) -> np.ndarray:
    """int8[n]: team (0/1) of the slot nearest to the ball, NOBODY where the ball or every slot is missing."""
    out = np.full(m.n, NOBODY, dtype=np.int8)
    team = np.asarray(m.team, dtype=np.int8)
    for a in range(0, m.n, chunk):
        b = m.ball[a:a + chunk]
        P = m.P[a:a + chunk]
        with np.errstate(invalid="ignore"):
            d2 = ((P - b[:, None, :]) ** 2).sum(axis=-1)
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        j = np.argmin(d2, axis=1)
        ok = np.isfinite(b).all(axis=1) & np.isfinite(d2[np.arange(len(j)), j])
        out[a:a + chunk][ok] = team[j[ok]]
    return out


def smooth_majority(codes: np.ndarray, window: int) -> np.ndarray:
    """The starter's `smooth`, returning codes 0 H / 1 A / 2 D."""
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def starter_baseline(m: Match, window: int = 50) -> np.ndarray:
    """int8[n] H/A/D on graded frames, X elsewhere."""
    lab = smooth_majority(nearest_slot_team(m), window)
    out = np.full(m.n, X, dtype=np.int8)
    g = m.graded_mask()
    out[g] = lab[g]
    return out
