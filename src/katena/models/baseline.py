"""M0: the official starter baseline algorithm on our corrupted inputs.

Nearest tracked player to the ball -> that player's team; ball missing -> D; then the starter's own
``smooth`` (majority vote, +-50 frames at 25 Hz = +-10 samples at 5 Hz), imported unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

import numpy as np

from katena.eval.protocol import REPO
from katena.types import Match

HALF_WINDOW_S = 2.0          # +-50 frames at 25 fps


@lru_cache(maxsize=1)
def _starter():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("starter_baseline", REPO / "starter" / "baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def m0_predict(m: Match) -> np.ndarray:
    """int8[n] on the work grid: 0 H, 1 A, 2 D inside the half blocks, 3 (X) outside."""
    sb = _starter()
    out = np.full(m.n, 3, dtype=np.int8)
    win = int(round(HALF_WINDOW_S * m.fps))
    for _, a, b in m.half_blocks:
        P, ball = m.P[a:b + 1], m.ball[a:b + 1]
        codes = np.full(len(ball), sb.NOBODY, dtype=np.int8)
        vis = np.isfinite(ball[:, 0])
        d = np.linalg.norm(P - ball[:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        near = np.argmin(d, axis=1)
        ok = vis & np.isfinite(d.min(1))
        codes[ok] = m.team[near[ok]]
        lab = sb.smooth(codes, win)
        out[a:b + 1] = np.select([lab == "H", lab == "A"], [0, 1], default=2)
    return out
