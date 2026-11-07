"""PLACEHOLDER decoder, so D4 can measure spellF1 now. J3 owns the real semi-Markov decoder.

`decode_simple(logp, t_frame, half_blocks, n_frames, cfg) -> int8[n_frames]`

1. Upsample row log-probs (one row per `t_frame`, e.g. every 5th frame) to every frame of each half:
   linear interpolation in log space between rows (held at the ends of a half), then renormalised.
2. Exact Viterbi per half over {H, A, D}: maximise the summed per-frame log-probs minus
   `switch_penalty` per label change, with every run at least `min_dwell` frames long.
3. X outside the halves.

The min-dwell Viterbi is solved as a segment DP: V[t, k] = C_k[t] + max_{s <= t - d} (W[s, k] - C_k[s]),
where C is the cumulative emission and W[s, k] = max_{k' != k} V[s, k'] - penalty. Because V on
[b, b + d) depends only on W before b, the recursion runs in blocks of d frames, vectorised over the
block and over several penalties at once (`segment_viterbi(E, penalties, d)`), which is what makes
the nested-CV parameter search cheap.
"""
from __future__ import annotations

import numpy as np

from ..match import X

DEFAULT_DECODE = {
    "switch_penalty": 8.0,   # log units per label change
    "min_dwell_s": 1.0,      # every run of H, A or D lasts at least this long
    "fps": 25,
    "upsample": "linear",    # "linear" (in log space) or "nearest"
    "round_decimals": 4,     # log-probs are rounded before decoding, for cross-machine stability
    "logp_floor": -20.0,
}


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def upsample_logp(logp: np.ndarray, t_frame: np.ndarray, half_blocks, n_frames: int,
                  mode: str = "linear") -> np.ndarray:
    """float32[n_frames, K] per-frame log-probs inside the halves, NaN outside them."""
    logp = np.asarray(logp, dtype=np.float64)
    t_frame = np.asarray(t_frame, dtype=np.int64)
    K = logp.shape[1]
    out = np.full((n_frames, K), np.nan, dtype=np.float32)
    for _, first, last in half_blocks:
        sel = (t_frame >= first) & (t_frame <= last)
        if not sel.any():
            raise ValueError(f"no prediction rows inside half block {(first, last)}")
        tr, lr = t_frame[sel], logp[sel]
        order = np.argsort(tr, kind="stable")
        tr, lr = tr[order], lr[order]
        f = np.arange(first, last + 1, dtype=np.int64)
        if mode == "linear":
            up = np.stack([np.interp(f, tr, lr[:, k]) for k in range(K)], axis=1)
        elif mode == "nearest":
            j = np.clip(np.searchsorted(tr, f), 1, len(tr) - 1) if len(tr) > 1 else np.zeros(len(f), int)
            if len(tr) > 1:
                left = j - 1
                j = np.where(np.abs(f - tr[left]) <= np.abs(tr[j] - f), left, j)
            up = lr[j]
        else:
            raise ValueError(f"unknown upsample mode {mode!r}")
        up = up - _logsumexp(up, axis=1)[:, None]
        out[first:last + 1] = up.astype(np.float32)
    return out


def segment_viterbi(E: np.ndarray, penalties, min_dwell: int) -> np.ndarray:
    """Best labelling of one half for each penalty. E: float[T, K] log-probs. Returns int8[P, T].

    Maximises sum_t E[t, y_t] - penalty * (number of label changes), subject to every run of
    constant label lasting at least `min_dwell` frames. Ties go to the lower label / earlier start.
    """
    E = np.asarray(E, dtype=np.float64)
    T, K = E.shape
    lam = np.atleast_1d(np.asarray(penalties, dtype=np.float64))
    P = lam.shape[0]
    d = max(1, int(min_dwell))
    if T == 0:
        return np.zeros((P, 0), dtype=np.int8)
    if T < 2 * d or K == 1:
        return np.full((P, T), int(np.argmax(E.sum(axis=0))), dtype=np.int8)

    C = np.zeros((T + 1, K))
    np.cumsum(E, axis=0, out=C[1:])
    NEG = -np.inf
    cand = np.full((T + 1, P, K), NEG)     # W[s] - C[s]: start a segment at s
    cand[0] = 0.0
    argK = np.zeros((T + 1, P, K), dtype=np.int8)   # previous label for a segment starting at s
    argS = np.zeros((T + 1, P, K), dtype=np.int32)  # start of the best segment ending at t
    R_prev = np.full((P, K), NEG)
    S_prev = np.zeros((P, K), dtype=np.int32)
    V_last = None
    b = d
    while b <= T:
        e = min(b + d, T + 1)
        L = e - b
        arr = np.empty((L + 1, P, K))
        arr[0] = R_prev
        arr[1:] = cand[b - d:e - d]
        cm = np.maximum.accumulate(arr, axis=0)
        pos = np.zeros((L + 1, P, K), dtype=np.int64)
        pos[1:] = np.where(arr[1:] > cm[:-1], np.arange(1, L + 1)[:, None, None], 0)
        pos = np.maximum.accumulate(pos, axis=0)[1:]
        S_blk = np.where(pos == 0, S_prev[None], (b - d - 1) + pos).astype(np.int32)
        R_blk = cm[1:]
        V = C[b:e, None, :] + R_blk                 # (L, P, K)
        argS[b:e] = S_blk
        if K == 3:
            v0, v1, v2 = V[..., 0], V[..., 1], V[..., 2]
            W = np.stack([np.maximum(v1, v2), np.maximum(v0, v2), np.maximum(v0, v1)], axis=-1)
            argK[b:e] = np.stack([np.where(v1 >= v2, 1, 2), np.where(v0 >= v2, 0, 2),
                                  np.where(v0 >= v1, 0, 1)], axis=-1).astype(np.int8)
        else:
            W = np.empty_like(V)
            for k in range(K):
                others = np.array([j for j in range(K) if j != k])
                sub = V[..., others]
                am = np.argmax(sub, axis=-1)
                W[..., k] = np.take_along_axis(sub, am[..., None], axis=-1)[..., 0]
                argK[b:e, :, k] = others[am]
        cand[b:e] = W - lam[None, :, None] - C[b:e, None, :]
        R_prev, S_prev = R_blk[-1], S_blk[-1]
        V_last = V
        b = e

    out = np.empty((P, T), dtype=np.int8)
    final = V_last[-1]                                # V[T], (P, K)
    for p in range(P):
        k = int(np.argmax(final[p]))
        t = T
        while True:
            s = int(argS[t, p, k])
            out[p, s:t] = k
            if s <= 0:
                break
            k, t = int(argK[s, p, k]), s
    return out


def _resolve(cfg: dict | None) -> dict:
    out = dict(DEFAULT_DECODE)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_DECODE)
        if unknown:
            raise KeyError(f"unknown decode cfg keys {sorted(unknown)}")
        out.update(cfg)
    return out


def prepare_emissions(logp, t_frame, half_blocks, n_frames, cfg: dict | None = None) -> np.ndarray:
    """Upsampled, rounded, floored per-frame log-probs (float64[n, K], NaN outside halves)."""
    cfg = _resolve(cfg)
    up = upsample_logp(logp, t_frame, half_blocks, n_frames, cfg["upsample"]).astype(np.float64)
    if cfg["round_decimals"] is not None:
        up = np.round(up, int(cfg["round_decimals"]))
    return np.maximum(up, float(cfg["logp_floor"]))


def decode_frames(E: np.ndarray, half_blocks, cfg: dict | None = None, penalties=None) -> np.ndarray:
    """Decode per-frame emissions (from `prepare_emissions`). Returns int8[P, n] (P = penalties)."""
    cfg = _resolve(cfg)
    pen = [cfg["switch_penalty"]] if penalties is None else list(penalties)
    d = int(round(float(cfg["min_dwell_s"]) * int(cfg["fps"])))
    n = E.shape[0]
    out = np.full((len(pen), n), X, dtype=np.int8)
    for _, first, last in half_blocks:
        out[:, first:last + 1] = segment_viterbi(E[first:last + 1], pen, d)
    return out


def decode_simple(logp, t_frame, half_blocks, n_frames: int, cfg: dict | None = None) -> np.ndarray:
    """PLACEHOLDER (J3 owns the real decoder). int8[n_frames]: 0 H, 1 A, 2 D in halves, 3 X outside."""
    E = prepare_emissions(logp, t_frame, half_blocks, n_frames, cfg)
    return decode_frames(E, half_blocks, cfg)[0]


def argmax_frames(logp, t_frame, half_blocks, n_frames: int, cfg: dict | None = None) -> np.ndarray:
    """No decoder: per-frame argmax of the upsampled log-probs (a reference point only)."""
    E = prepare_emissions(logp, t_frame, half_blocks, n_frames, cfg)
    out = np.full(n_frames, X, dtype=np.int8)
    for _, first, last in half_blocks:
        out[first:last + 1] = np.argmax(E[first:last + 1], axis=1)
    return out
