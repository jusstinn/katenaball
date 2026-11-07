"""3-state Viterbi and explicit-duration HSMM decoders (states 0 H, 1 A, 2 D).

Priors are fit on training label sequences only (``fit_priors``). Emissions are scaled likelihoods
``tau * (log p(class | x) - log prior(class))``; ``tau`` is the one tuned decoder knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import ndtr

K = 3
NEG = -1e18
KMAX = 1000                      # longest modelled segment: 200 s at 5 Hz


@dataclass
class Priors:
    logA: np.ndarray             # [3,3] per-sample transition log-probabilities (Viterbi)
    log_init: np.ndarray         # [3]
    log_class: np.ndarray        # [3] log class prior of samples
    logT: np.ndarray             # [3,3] segment-level transition log-probs, diagonal = NEG
    logdur: np.ndarray           # [3, KMAX+1] log duration pmf (samples)
    dur_params: np.ndarray       # [3,2] lognormal (mu, sigma) of segment duration in samples


def runs_of(y: np.ndarray) -> list:
    """Contiguous runs of non-X (label < 3) samples as (start, stop)."""
    ok = (y < 3).astype(np.int8)
    d = np.diff(np.concatenate(([0], ok, [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def fit_priors(ys: list, smooth: float = 1.0) -> Priors:
    """``ys``: list of int8 label arrays on the work grid (X = 3 breaks runs)."""
    A = np.full((K, K), smooth)
    seg_T = np.full((K, K), smooth)
    cls = np.full(K, smooth)
    first = np.zeros(K) + smooth
    durs = [[] for _ in range(K)]
    for y in ys:
        for a, b in runs_of(y):
            seg = y[a:b]
            cls += np.bincount(seg, minlength=K)[:K]
            first[seg[0]] += 1
            np.add.at(A, (seg[:-1], seg[1:]), 1)
            chg = np.flatnonzero(np.diff(seg) != 0) + 1
            bounds = np.concatenate(([0], chg, [len(seg)]))
            labs = seg[bounds[:-1]]
            for i in range(1, len(labs) - 1):                 # interior segments only (uncensored)
                durs[labs[i]].append(bounds[i + 1] - bounds[i])
            for i in range(len(labs) - 1):
                seg_T[labs[i], labs[i + 1]] += 1
    np.fill_diagonal(seg_T, 0.0)
    seg_T[seg_T == smooth] = smooth * 0.5
    np.fill_diagonal(seg_T, 0.0)
    logT = np.log(seg_T / seg_T.sum(1, keepdims=True) + 1e-12)
    np.fill_diagonal(logT, NEG)
    params = np.zeros((K, 2))
    logdur = np.full((K, KMAX + 1), NEG)
    k = np.arange(1, KMAX + 1)
    for s in range(K):
        x = np.log(np.maximum(np.asarray(durs[s], dtype=float), 1.0)) if durs[s] else np.array([3.0, 3.5])
        mu, sg = float(x.mean()), float(max(x.std(), 0.3))
        params[s] = (mu, sg)
        cdf = ndtr((np.log(k + 0.5) - mu) / sg) - ndtr((np.log(np.maximum(k - 0.5, 1e-6)) - mu) / sg)
        logdur[s, 1:] = np.log(np.maximum(cdf, 1e-9))
    return Priors(logA=np.log(A / A.sum(1, keepdims=True)), log_init=np.log(first / first.sum()),
                  log_class=np.log(cls / cls.sum()), logT=logT, logdur=logdur, dur_params=params)


def emissions(p_alive: np.ndarray, p_home: np.ndarray, pri: Priors, tau: float) -> np.ndarray:
    pa = np.clip(p_alive, 1e-4, 1 - 1e-4)
    ph = np.clip(p_home, 1e-4, 1 - 1e-4)
    lp = np.stack([np.log(pa * ph), np.log(pa * (1 - ph)), np.log(1 - pa)], 1)
    return tau * (lp - pri.log_class[None, :])


@njit(cache=True)
def _viterbi(le, logA, log_init):
    T = le.shape[0]
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int8)
    for s in range(K):
        delta[0, s] = log_init[s] + le[0, s]
    for t in range(1, T):
        for s in range(K):
            best, arg = -1e30, 0
            for r in range(K):
                v = delta[t - 1, r] + logA[r, s]
                if v > best:
                    best, arg = v, r
            delta[t, s] = best + le[t, s]
            psi[t, s] = arg
    out = np.empty(T, dtype=np.int8)
    s = 0
    for r in range(1, K):
        if delta[T - 1, r] > delta[T - 1, s]:
            s = r
    out[T - 1] = s
    for t in range(T - 1, 0, -1):
        s = psi[t, s]
        out[t - 1] = s
    return out


@njit(cache=True)
def _hsmm(le, logT, logdur, log_init, kmax):
    T = le.shape[0]
    cs = np.zeros((T + 1, K))
    for t in range(T):
        for s in range(K):
            cs[t + 1, s] = cs[t, s] + le[t, s]
    delta = np.full((T, K), -1e30)
    dstar = np.zeros((T, K), dtype=np.int32)
    pstar = np.zeros((T, K), dtype=np.int8)
    bp = np.full((T, K), -1e30)              # best predecessor score entering state s after sample t
    ba = np.zeros((T, K), dtype=np.int8)
    for t in range(T):
        for s in range(K):
            best, bd, bs = -1e30, 1, 0
            dmax = min(kmax, t + 1)
            for d in range(1, dmax + 1):
                start = t - d + 1
                if start == 0:
                    prev = log_init[s]
                    ps = -1
                else:
                    prev = bp[start - 1, s]
                    ps = ba[start - 1, s]
                if prev < -1e29:
                    continue
                dur = 0.0 if (start == 0 or t == T - 1) else logdur[s, d]
                v = prev + dur + cs[t + 1, s] - cs[start, s]
                if v > best:
                    best, bd, bs = v, d, ps
            delta[t, s] = best
            dstar[t, s] = bd
            pstar[t, s] = bs
        for s in range(K):                      # predecessor table for segments starting at t+1
            b2, a2 = -1e30, 0
            for r in range(K):
                if r == s:
                    continue
                v = delta[t, r] + logT[r, s]
                if v > b2:
                    b2, a2 = v, r
            bp[t, s] = b2
            ba[t, s] = a2
    out = np.empty(T, dtype=np.int8)
    s = 0
    for r in range(1, K):
        if delta[T - 1, r] > delta[T - 1, s]:
            s = r
    t = T - 1
    while t >= 0:
        d = dstar[t, s]
        out[t - d + 1:t + 1] = s
        ps = pstar[t, s]
        t -= d
        s = ps
    return out


def decode(p_alive, p_home, half, pri: Priors, tau: float, kind: str) -> np.ndarray:
    """Decode every half independently. ``kind``: 'viterbi' or 'hsmm'. Returns int8[T]."""
    le_all = emissions(p_alive, p_home, pri, tau)
    out = np.empty(len(le_all), dtype=np.int8)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        le = np.ascontiguousarray(le_all[sel])
        if kind == "viterbi":
            out[sel] = _viterbi(le, pri.logA, pri.log_init)
        elif kind == "hsmm":
            out[sel] = _hsmm(le, pri.logT, pri.logdur, pri.log_init, KMAX)
        else:
            raise ValueError(kind)
    return out
