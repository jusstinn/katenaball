"""3-state Viterbi whose switch probability varies with time (candidate B1, ``scripts/run_b.py``).

``katena.decode.hmm`` uses one transition matrix for the whole match. Here a boundary head gives, for every
sample ``t``, ``p_change[t]``: the probability that the owner changes within ``katena.models.boundary.TOL``
samples of ``t``. The odds of every off-diagonal transition into sample ``t`` are multiplied by

    (odds of p_change[t] / odds of the head's base rate) ** lam

and each row of the matrix is renormalised. ``lam = 0`` is ``hmm.decode(..., "viterbi")`` exactly. Emissions,
priors and ``tau`` are ``hmm``'s, unchanged.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from katena.decode import hmm

K = hmm.K
EPS = 1e-4                       # same clip as hmm.emissions


def bonus(p_change: np.ndarray, base: float, lam: float) -> np.ndarray:
    """Log of the factor on the switch odds at each sample: ``lam * (logit p_change - logit base)``."""
    p = np.clip(np.asarray(p_change, dtype=np.float64), EPS, 1.0 - EPS)
    b = float(np.clip(base, EPS, 1.0 - EPS))
    return lam * ((np.log(p) - np.log1p(-p)) - (np.log(b) - np.log1p(-b)))


@njit(cache=True)
def _viterbi_tv(le, logA, log_init, bon):
    T = le.shape[0]
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int8)
    for s in range(K):
        delta[0, s] = log_init[s] + le[0, s]
    A = np.empty((K, K))
    for t in range(1, T):
        b = bon[t]
        for r in range(K):
            if b == 0.0:                         # the static matrix, bit for bit
                for s in range(K):
                    A[r, s] = logA[r, s]
            else:
                tot = 0.0
                for s in range(K):
                    A[r, s] = logA[r, s] + (b if s != r else 0.0)
                    tot += np.exp(A[r, s])
                lt = np.log(tot)
                for s in range(K):
                    A[r, s] -= lt
        for s in range(K):
            best, arg = -1e30, 0
            for r in range(K):
                v = delta[t - 1, r] + A[r, s]
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


def decode(p_alive, p_home, p_change, half, pri: hmm.Priors, tau: float, lam: float, base: float) -> np.ndarray:
    """Decode every half independently, as ``hmm.decode``. Returns int8[T] (0 H, 1 A, 2 D)."""
    le_all = hmm.emissions(p_alive, p_home, pri, tau)
    bon_all = bonus(p_change, base, lam) if lam != 0.0 else np.zeros(len(le_all))
    out = np.empty(len(le_all), dtype=np.int8)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        out[sel] = _viterbi_tv(np.ascontiguousarray(le_all[sel]), pri.logA, pri.log_init,
                               np.ascontiguousarray(bon_all[sel]))
    return out
