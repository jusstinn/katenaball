"""M6: post-decode spell rescoring.

For each decoded possession spell, a small LightGBM classifier decides keep / flip-team / drop-to-dead
from features of the whole spell and its neighbours. A flipped spell that sits between two spells of
the other team is, in effect, merged away. Training spells come from out-of-fold predictions only.
"""

from __future__ import annotations

import numpy as np

from katena.models import gbdt

lgb = gbdt.lgb
KEEP, FLIP, DROP = 0, 1, 2
NAMES = ["dur_s", "support_mean", "support_min3s", "support_first2s", "support_last2s", "alive_mean",
         "ctrl_aligned", "margin_aligned", "tight_own", "tight_opp", "ball_vis", "d_own", "d_opp",
         "prev_kind", "prev_dur_s", "prev_support", "next_kind", "next_dur_s", "next_support",
         "sandwich", "pos_in_half"]
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0, num_threads=8,
              verbose=-1, deterministic=True, force_row_wise=True)
N_TREES = 100
APPLY_P = 0.6


def _segments(rows: np.ndarray, half: np.ndarray):
    """[(start, stop, label)] constant-label segments of the decoded rows, per half."""
    out = []
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        lab = rows[sel]
        chg = np.flatnonzero(np.diff(lab) != 0) + 1
        b = np.concatenate(([0], chg, [len(lab)]))
        for i in range(len(b) - 1):
            out.append((int(sel[b[i]]), int(sel[b[i + 1] - 1]) + 1, int(lab[b[i]]), int(h)))
    return out


def spell_features(rows, pa, ph, X0, names, half, fps: float = 5.0):
    """Returns (segments list [(a, b, label, half)], feature matrix for H/A segments, their segment indices)."""
    col = {n: X0[:, i] for i, n in enumerate(names)}
    segs = _segments(rows, half)
    s2 = int(2 * fps)
    s3 = int(3 * fps)
    feats, which = [], []
    sup_all = lambda team: np.where(team == 0, ph, 1.0 - ph)   # noqa: E731
    for i, (a, b, lab, h) in enumerate(segs):
        if lab == 2:
            continue
        sup = sup_all(lab)[a:b]
        sm = np.convolve(sup, np.ones(min(s3, len(sup))) / min(s3, len(sup)), mode="valid")
        sign = (lambda x: x) if lab == 0 else (lambda x: 1.0 - x)    # noqa: E731
        ctrl = np.nanmean(sign(col["ctrl_c3"][a:b])) if np.isfinite(col["ctrl_c3"][a:b]).any() else np.nan
        marg = np.nanmean(col["margin_c3"][a:b] * (1 if lab == 0 else -1)) if np.isfinite(col["margin_c3"][a:b]).any() else np.nan
        own = col["tightf_c3"] if lab == 0 else col["tighto_c3"]
        opp = col["tighto_c3"] if lab == 0 else col["tightf_c3"]
        d_own = col["d_f1"] if lab == 0 else col["d_o1"]
        d_opp = col["d_o1"] if lab == 0 else col["d_f1"]

        def nb(j):
            if j < 0 or j >= len(segs) or segs[j][3] != h:
                return -1, 0.0, np.nan
            a2, b2, l2, _ = segs[j]
            kind = 3 if l2 == 2 else (1 if l2 == lab else 2)             # 1 same team, 2 other team, 3 dead
            s = np.nan if l2 == 2 else float(np.mean(sup_all(l2)[a2:b2]))
            return kind, (b2 - a2) / fps, s

        pk, pd, ps = nb(i - 1)
        nk, nd, ns = nb(i + 1)
        feats.append([
            (b - a) / fps, sup.mean(), sm.min(), sup[:s2].mean(), sup[-s2:].mean(), pa[a:b].mean(), ctrl, marg,
            np.nanmean(own[a:b]) if np.isfinite(own[a:b]).any() else np.nan,
            np.nanmean(opp[a:b]) if np.isfinite(opp[a:b]).any() else np.nan,
            np.nanmean(col["ball_vis"][a:b]), np.nanmean(d_own[a:b]) if np.isfinite(d_own[a:b]).any() else np.nan,
            np.nanmean(d_opp[a:b]) if np.isfinite(d_opp[a:b]).any() else np.nan,
            pk, pd, ps, nk, nd, ns, float(pk == 2 and nk == 2), a / max(len(rows), 1)])
        which.append(i)
    return segs, np.asarray(feats, dtype=np.float32).reshape(-1, len(NAMES)), which


def spell_targets(segs, which, gt_rows) -> np.ndarray:
    """Majority ground-truth label inside each spell: KEEP (same team), FLIP (other team), DROP (dead),
    or -1 when the ground truth is not graded there."""
    out = np.full(len(which), -1, dtype=np.int8)
    for k, i in enumerate(which):
        a, b, lab, _ = segs[i]
        g = gt_rows[a:b]
        g = g[g < 3]
        if len(g) < 3:
            continue
        cnt = np.bincount(g, minlength=3)
        top = int(np.argmax(cnt))
        out[k] = KEEP if top == lab else (DROP if top == 2 else FLIP)
    return out


def fit(F: np.ndarray, y: np.ndarray):
    ok = y >= 0
    return lgb.train(PARAMS, lgb.Dataset(F[ok], y[ok].astype(int)), N_TREES)


def apply(model, rows, segs, F, which, p_min: float = APPLY_P) -> np.ndarray:
    out = rows.copy()
    if len(F) == 0:
        return out
    p = model.predict(F)
    for k, i in enumerate(which):
        a, b, lab, _ = segs[i]
        c = int(np.argmax(p[k]))
        if c != KEEP and p[k, c] >= p_min:
            out[a:b] = (1 - lab) if c == FLIP else 2
    return out
