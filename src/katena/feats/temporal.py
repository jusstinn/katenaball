"""Vectorised, NaN-aware, segment-aware time-series helpers.

The feature timeline is a concatenation of segments (one per half by default). Nothing here ever
looks across a segment boundary: windows are clipped to the segment, differences never span one,
and "last seen" / "next seen" searches stop at it. Functions take `seg_lo` / `seg_hi`, the
inclusive first and last row of the segment containing each row (int64[N]).

All windows are centred and offline (non-causal). No per-frame Python loops: cumulative sums for
means, the van Herk / Gil-Werman block trick for running min / max, and accumulate-based index
fills for gaps.
"""
from __future__ import annotations

import numpy as np


def segments_from_lengths(lengths) -> tuple[np.ndarray, np.ndarray]:
    """seg_lo, seg_hi (int64[N]) for a timeline made of consecutive segments of these lengths."""
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    lo = np.repeat(starts, lengths)
    hi = np.repeat(starts + lengths - 1, lengths)
    return lo, hi


def _bshape(v: np.ndarray, ndim: int) -> np.ndarray:
    return v.reshape((-1,) + (1,) * (ndim - 1))


def last_true_index(mask: np.ndarray, seg_lo: np.ndarray) -> np.ndarray:
    """For each row t, the largest u <= t in t's segment with mask[u]; -1 if there is none.

    `mask` is bool[N] or bool[N, S]; the search runs along axis 0 independently per column.
    """
    N = mask.shape[0]
    t = _bshape(np.arange(N, dtype=np.int64), mask.ndim)
    a = np.where(mask, t, -1)
    np.maximum.accumulate(a, axis=0, out=a)
    a[a < _bshape(seg_lo, mask.ndim)] = -1
    return a


def next_true_index(mask: np.ndarray, seg_hi: np.ndarray) -> np.ndarray:
    """For each row t, the smallest u >= t in t's segment with mask[u]; -1 if there is none."""
    N = mask.shape[0]
    t = _bshape(np.arange(N, dtype=np.int64), mask.ndim)
    b = np.where(mask, t, N)
    b = np.minimum.accumulate(b[::-1], axis=0)[::-1]
    return np.where(b > _bshape(seg_hi, mask.ndim), -1, b)


def run_bounds(present: np.ndarray, seg_lo: np.ndarray, seg_hi: np.ndarray):
    """First and last row (inclusive) of the run of consecutive `present` rows containing t.

    Runs are cut at segment boundaries. Values are meaningful only where `present` is True.
    """
    rs = last_true_index(~present, seg_lo) + 1
    rs = np.maximum(rs, _bshape(seg_lo, present.ndim))
    ne = next_true_index(~present, seg_hi)
    re = np.where(ne < 0, _bshape(seg_hi, present.ndim), ne - 1)
    return rs, re


def gather_rows(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """X[idx[t]] for 1-D idx, or X[idx[t, s], s] for X of shape [N, S, ...] and idx [N, S]."""
    if idx.ndim == 1:
        return X[idx]
    return X[idx, np.arange(idx.shape[1])[None, :]]


def gap_diff(X: np.ndarray, present: np.ndarray, rs: np.ndarray, re: np.ndarray, k: int,
             fps: float) -> np.ndarray:
    """Gap-aware time derivative of X (units per second), never differencing across a gap.

    X: float[N, C] with present bool[N], or float[N, S, C] with present bool[N, S].
    At row t inside a run of present rows [rs, re], uses the widest symmetric difference
    X[t + j] - X[t - j] with j = min(t - rs, re - t, k). If j == 0 but the run has length >= 2,
    falls back to a one-sided difference. Isolated single frames and absent rows give NaN.
    """
    N = present.shape[0]
    t = _bshape(np.arange(N, dtype=np.int64), present.ndim)
    j = np.minimum(np.minimum(t - rs, re - t), k)
    j = np.where(present, j, -1)
    jj = np.maximum(j, 0)
    ip = t + jj
    im = t - jj
    fwd = (j == 0) & (re > t)
    bwd = (j == 0) & ~fwd & (rs < t)
    ip = np.where(fwd, t + 1, ip)
    im = np.where(bwd, t - 1, im)
    denom = (ip - im).astype(np.float32)
    ok = present & (denom > 0)
    ip = np.where(ok, ip, 0)
    im = np.where(ok, im, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (gather_rows(X, ip) - gather_rows(X, im)) * (np.float32(fps) / denom)[..., None]
    d[~ok] = np.nan
    return d.astype(np.float32, copy=False)


def _clip_windows(N: int, h: int, seg_lo: np.ndarray, seg_hi: np.ndarray):
    t = np.arange(N, dtype=np.int64)
    return np.maximum(t - h, seg_lo), np.minimum(t + h, seg_hi)


def roll_mean(x: np.ndarray, h: int, seg_lo: np.ndarray, seg_hi: np.ndarray) -> np.ndarray:
    """Centred NaN-aware rolling mean over [t - h, t + h] clipped to the segment.

    x: float[N] or float[N, C]. NaN where the window holds no finite value. float64 cumulative
    sums keep it accurate at 200k rows; output float32.
    """
    x = np.asarray(x)
    N = x.shape[0]
    valid = np.isfinite(x)
    z = np.zeros((1,) + x.shape[1:], dtype=np.float64)
    cs = np.concatenate([z, np.cumsum(np.where(valid, x, 0.0), axis=0, dtype=np.float64)])
    cn = np.concatenate([z, np.cumsum(valid, axis=0, dtype=np.float64)])
    lo, hi = _clip_windows(N, h, seg_lo, seg_hi)
    s = cs[hi + 1] - cs[lo]
    c = cn[hi + 1] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out.astype(np.float32)


def _sliding_max_1d(a: np.ndarray, w: int) -> np.ndarray:
    """out[i] = max(a[i:i + w]) for i in 0..len(a) - w (van Herk / Gil-Werman, O(n))."""
    n = a.shape[0]
    nb = -(-n // w)
    pad = nb * w - n
    ap = np.concatenate([a, np.full((pad,) + a.shape[1:], -np.inf, dtype=a.dtype)])
    blocks = ap.reshape((nb, w) + a.shape[1:])
    pre = np.maximum.accumulate(blocks, axis=1).reshape(ap.shape)
    suf = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape(ap.shape)
    m = n - w + 1
    return np.maximum(suf[:m], pre[w - 1:w - 1 + m])


def _segment_starts(seg_lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = seg_lo.shape[0]
    starts = np.flatnonzero(np.r_[True, seg_lo[1:] != seg_lo[:-1]])
    return starts, np.r_[starts[1:], N]


def roll_max(x: np.ndarray, h: int, seg_lo: np.ndarray, seg_hi: np.ndarray) -> np.ndarray:
    """Centred NaN-aware rolling max over [t - h, t + h] clipped to the segment (NaN if empty)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    a = np.where(np.isfinite(x), x, -np.inf)
    w = 2 * h + 1
    padv = np.full((h,) + x.shape[1:], -np.inf)
    for s, e in zip(*_segment_starts(seg_lo)):  # one iteration per segment (half), not per frame
        out[s:e] = _sliding_max_1d(np.concatenate([padv, a[s:e], padv]), w)
    out[np.isneginf(out)] = np.nan
    return out.astype(np.float32)


def roll_min(x: np.ndarray, h: int, seg_lo: np.ndarray, seg_hi: np.ndarray) -> np.ndarray:
    """Centred NaN-aware rolling min, see roll_max."""
    return -roll_max(-np.asarray(x, dtype=np.float64), h, seg_lo, seg_hi)


def shift_clip(x: np.ndarray, k: int, seg_lo: np.ndarray, seg_hi: np.ndarray) -> np.ndarray:
    """x[t + k], with the index clipped to t's segment (the edge value is repeated)."""
    N = x.shape[0]
    return x[np.clip(np.arange(N, dtype=np.int64) + k, seg_lo, seg_hi)]
