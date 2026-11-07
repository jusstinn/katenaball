"""Error-type census of possession spells on the native grid (what the loss is made of)."""

from __future__ import annotations

import numpy as np

import possession_metrics as pm

TYPES = ("ok", "wrong_team", "dead_miss", "split", "merged", "boundary", "mixed")


def census(gt: np.ndarray, pred: np.ndarray, iou: float = 0.5) -> dict:
    """Classify every ground-truth spell (matched at ``iou`` or by why not) and every unmatched
    predicted spell. Priority for an unmatched GT spell: majority predicted D -> dead_miss; majority
    predicted other team -> wrong_team; majority predicted right team -> split (two or more predicted
    spells carry it), merged (the covering prediction also covers a neighbouring GT spell) or boundary
    (single prediction, IoU below threshold); otherwise mixed."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gts, prs = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gts, prs), len(gts), iou)
    counts = dict.fromkeys(TYPES, 0)
    for i, (a, b, team) in enumerate(gts):
        if flags[i]:
            counts["ok"] += 1
            continue
        seg = filled[a:b]
        n = len(seg)
        if (seg == pm.D).sum() >= 0.5 * n:
            counts["dead_miss"] += 1
        elif (seg == 1 - team).sum() >= 0.5 * n:
            counts["wrong_team"] += 1
        elif (seg == team).sum() >= 0.5 * n:
            cover = [(c, d) for c, d, t in prs if t == team and min(b, d) - max(a, c) >= 0.1 * (d - c)]
            if len(cover) >= 2:
                counts["split"] += 1
            else:
                c, d = cover[0] if cover else (a, b)
                other = [1 for e, f, t in gts if t == team and (e, f) != (a, b) and min(f, d) - max(e, c) >= 0.1 * (f - e)]
                counts["merged" if other else "boundary"] += 1
        else:
            counts["mixed"] += 1
    spurious = {"in_dead": 0, "in_other": 0}
    pflags = _matched_pred(gts, prs, 0.3)
    for j, (c, d, t) in enumerate(prs):
        if pflags[j]:
            continue
        g = gt[c:d]
        g = g[g < 3]
        if len(g) and (g == pm.D).mean() >= 0.5:
            spurious["in_dead"] += 1
        else:
            spurious["in_other"] += 1
    g_all = gt[gt < 3]
    p_all = filled[gt < 3]
    dead_err = {"gt_dead_pred_alive": float(((g_all == pm.D) & (p_all != pm.D)).sum() / max((g_all == pm.D).sum(), 1)),
                "gt_alive_pred_dead": float(((g_all != pm.D) & (p_all == pm.D)).sum() / max((g_all != pm.D).sum(), 1))}
    return {"gt_spells": counts, "n_gt": len(gts), "n_pred": len(prs), "spurious_pred": spurious, "dead_frame_err": dead_err}


def _matched_pred(gts, prs, thr):
    flags = np.zeros(len(prs), bool)
    used = set()
    for iou_v, i, j in pm.spell_overlaps(gts, prs):
        if iou_v < thr:
            break
        if i in used or flags[j]:
            continue
        used.add(i)
        flags[j] = True
    return flags
