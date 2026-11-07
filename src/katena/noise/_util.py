"""Small vectorised helpers shared by the noise simulator, its statistics and the domain-gap check."""
from __future__ import annotations

import numpy as np


def blocks_of(m) -> list[tuple[int, int]]:
    """Half-open (first, stop) frame ranges to measure on: the graded halves, or the whole match."""
    out = [(int(f), int(l) + 1) for _, f, l in m.half_blocks if l >= f]
    return out or [(0, m.n)]


def runs_1d(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and stops (half-open) of the True runs of a 1-D boolean array."""
    pad = np.zeros(mask.shape[0] + 2, dtype=np.int8)
    pad[1:-1] = mask
    d = np.diff(pad)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def runs_2d(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(column, start, stop) of the True runs down each column of a [T, S] boolean array.

    Ordered by column, then start. Stops are exclusive.
    """
    T, S = mask.shape
    pad = np.zeros((T + 2, S), dtype=np.int8)
    pad[1:-1] = mask
    d = np.diff(pad, axis=0).T  # [S, T+1]
    col, start = np.nonzero(d == 1)
    _, stop = np.nonzero(d == -1)
    return col, start, stop


def nan_moving_mean(x: np.ndarray, half: int) -> np.ndarray:
    """Centred moving mean of a 1-D series over 2*half+1 frames, ignoring NaN."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(v)])
    k = np.concatenate([[0], np.cumsum(ok)])
    n = x.shape[0]
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def count_within(P: np.ndarray, pts: np.ndarray, r: float, chunk: int = 50_000) -> np.ndarray:
    """int16[n]: for each frame, how many rows of P[t] lie within r metres of pts[t] (NaN never counts)."""
    n = P.shape[0]
    out = np.zeros(n, dtype=np.int16)
    r2 = r * r
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = P[a:b] - pts[a:b, None, :]
        with np.errstate(invalid="ignore"):
            out[a:b] = ((d[..., 0] ** 2 + d[..., 1] ** 2) < r2).sum(1)
    return out


def pair_counts(P: np.ndarray, team: np.ndarray, idx: np.ndarray, r: float,
                chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For sampled frames `idx`: number of same-team and of cross-team slot pairs closer than r."""
    S = P.shape[1]
    iu, ju = np.triu_indices(S, 1)
    same = team[iu] == team[ju]
    out_same = np.zeros(idx.shape[0], dtype=np.int32)
    out_opp = np.zeros(idx.shape[0], dtype=np.int32)
    r2 = r * r
    for a in range(0, idx.shape[0], chunk):
        Q = P[idx[a:a + chunk]]
        d = Q[:, iu, :] - Q[:, ju, :]
        with np.errstate(invalid="ignore"):
            close = (d[..., 0] ** 2 + d[..., 1] ** 2) < r2
        out_same[a:a + chunk] = close[:, same].sum(1)
        out_opp[a:a + chunk] = close[:, ~same].sum(1)
    return out_same, out_opp


def ball_speed(ball: np.ndarray, fps: int, lag: int = 4) -> np.ndarray:
    """float32[n]: ball speed in m/s over a `lag`-frame baseline, centred, NaN where unknown."""
    n = ball.shape[0]
    v = np.full(n, np.nan, dtype=np.float32)
    if n <= lag:
        return v
    h = lag // 2
    d = ball[lag:] - ball[:-lag]
    v[h:h + n - lag] = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) * (fps / lag)
    return v


def second_diff_abs(P: np.ndarray) -> np.ndarray:
    """|x[t+1] - 2x[t] + x[t-1]| per coordinate, float32[n-2, ..., 2]; NaN where any frame is missing."""
    return np.abs(P[2:] - 2 * P[1:-1] + P[:-2])


def fnum(x) -> float:
    """Plain Python float (NaN allowed)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
