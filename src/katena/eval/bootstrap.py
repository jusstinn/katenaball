"""Paired bootstrap on possessions and the pre-registered adoption rule."""

from __future__ import annotations

import numpy as np


def _f1(flags: np.ndarray, picks: np.ndarray, n_gt: int, n_pred: int) -> float:
    tp = flags[:, picks].sum(1)
    fp = np.maximum(0, n_pred - tp)
    fn = n_gt - tp
    den = 2 * tp + fp + fn
    return float(np.mean(np.where(tp > 0, 2.0 * tp / np.maximum(den, 1), 0.0)))


def paired_bootstrap(a: list, b: list, draws: int = 2000, seed: int = 0) -> dict:
    """``a``, ``b``: per-match score dicts (``score()`` output) for models A and B on the same matches.
    Resamples ground-truth possessions within each match; statistic = mean over matches of
    spellF1(A) - spellF1(B). Returns mean difference, 95 % interval, and the win rate over matches."""
    rng = np.random.RandomState(seed)
    diffs = np.empty(draws)
    for d in range(draws):
        acc = 0.0
        for ra, rb in zip(a, b):
            n = ra["n_gt"]
            picks = rng.randint(0, n, n)
            acc += _f1(ra["flags"], picks, n, ra["n_pred"]) - _f1(rb["flags"], picks, n, rb["n_pred"])
        diffs[d] = acc / len(a)
    sa = np.array([r["spell_f1"] for r in a])
    sb = np.array([r["spell_f1"] for r in b])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float((sa - sb).mean()), "lo": float(lo), "hi": float(hi),
            "win_rate": float((sa > sb).mean()), "n": len(a)}


def adopted(res: dict) -> bool:
    """Pre-registered rule: wins on >= 70 % of matches and the interval excludes 0 (positive side)."""
    return res["win_rate"] >= 0.70 and res["lo"] > 0.0


def single_ci(a: list, draws: int = 2000, seed: int = 0) -> tuple:
    """95 % interval of the mean spellF1 over matches, resampling ground-truth possessions per match."""
    rng = np.random.RandomState(seed)
    out = np.empty(draws)
    for d in range(draws):
        acc = 0.0
        for r in a:
            n = r["n_gt"]
            acc += _f1(r["flags"], rng.randint(0, n, n), n, r["n_pred"])
        out[d] = acc / len(a)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)
