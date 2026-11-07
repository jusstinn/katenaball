"""Scoring: work-grid predictions -> native frames -> official ``possession_metrics.evaluate``."""

from __future__ import annotations

import numpy as np

import possession_metrics as pm
from katena.grid import to_native
from katena.types import D, X


def to_native_codes(pred_rows: np.ndarray, idx: np.ndarray, step: int, n_native: int) -> np.ndarray:
    n5 = -(-n_native // step)
    p5 = np.full(n5, X, dtype=np.int8)
    p5[idx] = pred_rows
    return to_native(p5, step, n_native)


def score(gt: np.ndarray, pred: np.ndarray, sc_mode: bool = False) -> dict:
    """Official metrics for one match. ``sc_mode``: SkillCorner has no D label, so predicted D
    becomes X (the scorer then repeats the previous team label)."""
    if sc_mode:
        pred = np.where(pred == D, X, pred).astype(np.int8)
    res = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    flags = np.vstack([pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), t) for t in pm.SPELL_IOUS])
    return {"spell_f1": float(res["spell_f1_mean"]), "macro_f1": float(res["macro_f1"]),
            "n_gt": len(gt_sp), "n_pred": len(pr_sp), "flags": flags,
            "alive_dead_f1": float(res["alive_dead_f1"]), "team_acc": float(res["team_acc_both_alive"])}
