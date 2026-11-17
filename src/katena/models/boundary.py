"""Training target of the boundary head (candidate B1, ``scripts/run_b.py``): is there a change of owner
within ``TOL`` work-grid samples of this sample?

The owner sequence is the label sequence with unknown samples (X) filled with the last known label of the same
in-play block: "possession stays with the last team that had it". So a SkillCorner run H, X, A has one change,
at the first A sample, as an IDSSE run H, A has. Dead ball is a state like the two teams. Samples before the
first known label of a block have no target.
"""

from __future__ import annotations

import numpy as np

X = 3
TOL = 2                          # samples: 0.4 s at 5 Hz


def blocks5(blocks, step: int) -> list:
    """Native in-play blocks (half, first, last), inclusive, as work-grid sample ranges [lo, hi)."""
    out = []
    for _, a, b in blocks:
        lo, hi = -(-int(a) // step), int(b) // step
        if hi >= lo:
            out.append((lo, hi + 1))
    return out


def owner(y5: np.ndarray, blocks: list) -> np.ndarray:
    """``y5`` with X filled forward inside each block; X where no label came before (and outside blocks)."""
    out = np.full(len(y5), X, dtype=np.int8)
    for lo, hi in blocks:
        seg = np.asarray(y5[lo:hi])
        last = np.where(seg != X, np.arange(len(seg)), -1)
        np.maximum.accumulate(last, out=last)
        out[lo:hi] = np.where(last >= 0, seg[np.maximum(last, 0)], X)
    return out


def change_target(y5: np.ndarray, blocks: list, tol: int = TOL) -> tuple:
    """(target uint8[n], known bool[n]). ``target[t]`` is 1 when the owner changes at some sample of
    ``[t - tol, t + tol]`` in the same block (a change at ``u`` means owner[u] differs from owner[u - 1], both
    known). ``known[t]``: the owner at ``t`` is known, so the sample has a target."""
    own = owner(y5, blocks)
    target = np.zeros(len(y5), dtype=np.uint8)
    for lo, hi in blocks:
        seg = own[lo:hi]
        chg = np.zeros(len(seg), dtype=bool)
        chg[1:] = (seg[1:] != seg[:-1]) & (seg[1:] != X) & (seg[:-1] != X)
        near = chg.copy()
        for k in range(1, tol + 1):
            near[k:] |= chg[:-k]
            near[:-k] |= chg[k:]
        target[lo:hi] = near
    return target, own != X
