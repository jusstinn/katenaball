"""Training across label conventions (D4): team heads trained on one convention, on a per-row mix of
T, or on all T at once (soft labels), each scored against the ground truth of every convention.

Conventions never change D or X frames (``katena.labels.sandwich`` only swaps H and A), so the
alive head, its rows and its labels are the same at every T. Its out-of-fold probabilities are
reused from the M2 cache (``oof/full/T2``) and only the team head is refitted, with M2's rows, row
order, parameters and seed. ``S2`` is M2 itself, read from the cache.

Training label schemes (``MODELS``):

* ``S<T>``, ``S2tail``, ``H3.5``: one convention (``CONV``).
* ``MIX``: each pooled row takes the label of one T drawn uniformly from ``protocol.T_SWEEP``, once,
  with a fixed seed (the same draw in every fold).
* ``AUG``: every row carries all five conventions at once: its team target is the share of the five
  T at which the focal team has the ball, fitted as weighted 0/1 copies of the row with the same
  binary objective (the expected loss over T).

Decoding is M2's: Viterbi, priors fitted on the IDSSE and Metrica training matches' labels in the
model's own convention (pooled over T for MIX and AUG), tau chosen per fold from the other folds'
mean spellF1 against the model's own convention(s).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import possession_metrics as pm
from katena.decode import hmm
from katena.eval import corpus as C
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval.score import to_native_codes
from katena.labels import sandwich
from katena.types import D, X

CONV = {f"S{T}": dict(T=T) for T in P.T_SWEEP}
CONV.update({"S2tail": dict(T=2, tail=True), "H3.5": dict(T=3.5, tail=True, order="shortest")})
EVAL_MAIN = tuple(f"S{T}" for T in P.T_SWEEP)
EVAL_EXTRA = ("S2tail", "H3.5", "E")             # E (event-defined) exists for IDSSE folds only
SWEEP = tuple(f"S{T}" for T in P.T_SWEEP)
MODELS = {**{f"S{T}": ("single", (f"S{T}",)) for T in P.T_SWEEP},
          "MIX": ("mix", SWEEP), "AUG": ("aug", SWEEP),
          "H3.5": ("single", ("H3.5",)), "S2tail": ("single", ("S2tail",))}
CONDS = ("target", "severe")
TMIX_DIR = P.WORK / "tmix"


def conv_labels(y: np.ndarray, fps: float, name: str) -> np.ndarray:
    kw = dict(CONV[name])
    return sandwich(y, pm.valid_runs(y), fps, kw.pop("T"), **kw)


class Labels:
    """Native ground truth of every development match under every convention (from ``y_T0``)."""

    def __init__(self, refs: list, e_dir: Path | None = None):
        self.refs = refs
        self.g, self.y = {}, {}
        e_dir = e_dir or P.WORK / "labels"
        for r in refs:
            g = C.load_gt(r, 0)
            self.g[r.key] = g
            ys = {}
            for name in CONV:
                T = CONV[name]["T"]
                if set(CONV[name]) == {"T"} and T in P.T_SWEEP:
                    ys[name] = C.load_gt(r, T)["y"]         # the cached sandwich labels
                else:
                    ys[name] = conv_labels(g["y"], g["fps"], name)
            e = e_dir / f"{r.dataset}_{r.id}__E.npz"
            if e.exists():
                with np.load(e) as z:
                    ys["E"] = z["y"]
            self.y[r.key] = ys

    def gt(self, ref, name: str) -> dict | None:
        y = self.y[ref.key].get(name)
        if y is None:
            return None
        g = self.g[ref.key]
        return {"y": y, "step": g["step"], "n_native": g["n_native"], "fps": g["fps"]}

    def work(self, ref, name: str) -> np.ndarray:
        return self.y[ref.key][name][::self.g[ref.key]["step"]]

    def seqs(self, exclude: set, names: tuple) -> list:
        """Work-grid label sequences for decoder priors (IDSSE and Metrica only, as in M2)."""
        return [self.work(r, n) for r in self.refs if r.key not in exclude and not r.is_sc for n in names]


def pool_labels(pools: Lm.Pools, labels: Labels, name: str, generic: bool = False) -> np.ndarray:
    """Labels of every pooled row (``pools`` order) under convention ``name``. ``generic`` forces
    the path used for conventions the cache does not hold (a check that row order is right)."""
    if name in SWEEP and not generic:
        return pools.y[int(name[1:])]
    parts = []
    for r in pools.refs:
        with np.load(C.paths(r)["pool"]) as z:          # the row index only, not the features
            idx = z["idx"]
        parts.append(labels.work(r, name)[idx])
    return np.concatenate(parts).astype(np.int8)


def mix_draw(n: int) -> np.ndarray:
    """Which T each pooled row takes in MIX: index into T_SWEEP, fixed by seed."""
    return np.random.default_rng(P.seed_for("tmix", "mix")).integers(0, len(P.T_SWEEP), n)


def team_design(kind: str, ys: list, o: np.ndarray, draw: np.ndarray | None = None) -> tuple:
    """Team-head rows, targets and weights for one scheme on already-selected rows.

    ``ys``: label arrays (same rows) of the scheme's conventions; H/A rows are the same in all of
    them. Returns ``(rows, target, weight)``: ``rows`` index the input rows (duplicates allowed),
    ``weight`` is None for unweighted schemes."""
    base = np.flatnonzero((ys[0] == 0) | (ys[0] == 1))
    for y in ys[1:]:
        assert np.array_equal(base, np.flatnonzero((y == 0) | (y == 1))), "conventions changed D/X"
    if kind == "single":
        return base, (ys[0][base] == o[base]).astype(np.float32), None
    hits = np.stack([(y[base] == o[base]) for y in ys]).astype(np.float64)    # [nT, rows]
    if kind == "mix":
        return base, hits[draw[base], np.arange(len(base))].astype(np.float32), None
    if kind == "aug":
        w = hits.mean(0)
        split = (w > 0) & (w < 1)
        rows = np.concatenate([base, base[split]])
        target = np.concatenate([np.where(split, 1.0, w), np.zeros(int(split.sum()))]).astype(np.float32)
        weight = np.concatenate([np.where(split, w, 1.0), 1.0 - w[split]]).astype(np.float32)
        return rows, target, weight
    raise ValueError(kind)


def fit_fold(pools: Lm.Pools, labels: Labels, ref, models: list, pool_y: dict, draw: np.ndarray,
             out_dir: Path | None = None, conds=CONDS, log=print) -> None:
    """Refit the team head of every model in ``models`` without ``ref`` and store p_home per cond."""
    from katena.models import gbdt
    lgb = gbdt.lgb
    out_dir = out_dir or TMIX_DIR / "oof"
    todo = [mname for mname in models if mname != "S2" and not (out_dir / mname / f"{ref.dataset}_{ref.id}.npz").exists()]
    if not todo:
        return
    t0 = time.time()
    keep = np.flatnonzero(~np.isin(pools.keys, [ref.key]))     # M2's training rows, M2's order
    cols = pools.cols()
    every = np.array_equal(cols, np.arange(pools.X.shape[1]))

    def take(X, idx=None):                                     # same values as X[idx][:, cols]
        Z = X if idx is None else X[idx]
        return Z if every else Z[:, cols]

    o = pools.o[keep]
    feats = {c: C.load_feat(ref, c) for c in conds}
    shared = None                                    # one binned Dataset for every unweighted scheme
    for mname in todo:
        kind, names = MODELS[mname]
        ys = [pool_y[n][keep] for n in names]
        rows, target, weight = team_design(kind, ys, o, draw[keep] if kind == "mix" else None)
        if weight is None:
            if shared is None:
                shared = lgb.Dataset(take(pools.X, keep[rows]), target, free_raw_data=False)
                shared_rows = rows
            else:
                assert np.array_equal(rows, shared_rows)
                shared.set_label(target)
            ds = shared
        else:
            ds = lgb.Dataset(take(pools.X, keep[rows]), target, weight=weight, free_raw_data=True)
        team = lgb.train(gbdt.PARAMS, ds, gbdt.N_TREES)
        if ds is not shared:
            del ds
        arrays = {}
        for c, f in feats.items():
            ph = 0.5 * (team.predict(take(f["X0"])) + 1.0 - team.predict(take(f["X1"])))
            arrays[f"ph_{c}"] = ph.astype(np.float32)
        path = out_dir / mname / f"{ref.dataset}_{ref.id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        log(f"[{time.strftime('%H:%M:%S')}] {ref.key} {mname} ({time.time() - t0:.0f}s)")


def load_ph(mname: str, ref, cond: str, out_dir: Path | None = None) -> np.ndarray:
    if mname == "S2":
        return Lm.load_oof("full", P.T_MAIN, ref)[f"ph_{cond}"]
    with np.load((out_dir or TMIX_DIR / "oof") / mname / f"{ref.dataset}_{ref.id}.npz") as z:
        return z[f"ph_{cond}"]


# ---------------------------------------------------------------------------------------------
# scoring: the official spellF1 formula, without the transition metrics evaluate() also computes

def spell_overlaps(gt_spells: list, pred_spells: list) -> list:
    """Same list as ``possession_metrics.spell_overlaps`` (same IoU floats, same order), without
    the all-pairs loop: spells of one sequence are disjoint and sorted, so the predictions that
    overlap a ground-truth spell are one contiguous range."""
    if not gt_spells or not pred_spells:
        return []
    ps = np.array([s[0] for s in pred_spells])
    pe = np.array([s[1] for s in pred_spells])
    out = []
    for i, (gs, ge, gc) in enumerate(gt_spells):
        lo, hi = int(np.searchsorted(pe, gs, "right")), int(np.searchsorted(ps, ge, "left"))
        for j in range(lo, hi):
            p0, p1, pc = pred_spells[j]
            if pc != gc:
                continue
            out.append(((min(ge, p1) - max(gs, p0)) / float(max(ge, p1) - min(gs, p0)), i, j))
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def spell_score(gt: np.ndarray, pred: np.ndarray, sc_mode: bool = False) -> dict:
    """``spell_f1`` equal to ``possession_metrics.evaluate(gt, pred)['spell_f1_mean']`` plus the
    per-IoU matching flags the paired bootstrap needs (same keys as ``katena.eval.score.score``)."""
    if sc_mode:
        pred = np.where(pred == D, X, pred).astype(np.int8)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    ov = spell_overlaps(gt_sp, pr_sp)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gt_sp), t) for t in pm.SPELL_IOUS])
    f1s = []
    for row in flags:
        tp = int(row.sum())
        fp, fn = len(pr_sp) - tp, len(gt_sp) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"spell_f1": float(np.mean(f1s)), "n_gt": len(gt_sp), "n_pred": len(pr_sp), "flags": flags}


def feat_rows(ref, cond: str) -> dict:
    """``half`` and ``idx`` of a cached feature file, without loading the features."""
    with np.load(C.paths(ref)["feat"](cond)) as z:
        return {"half": z["half"], "idx": z["idx"]}


def decode_scores(pools_labels: Labels, refs: list, mname: str, conds=CONDS, taus=P.TAUS,
                  evals=EVAL_MAIN + EVAL_EXTRA, log=print) -> dict:
    """{(cond, eval convention, match key): score} for one model, tau chosen per fold from the other
    folds in ``refs`` (target cond, mean spellF1 against the model's own conventions)."""
    _, own = MODELS[mname]
    t0 = time.time()
    pa = {r.key: Lm.load_oof("full", P.T_MAIN, r) for r in refs}
    feats = {r.key: {c: feat_rows(r, c) for c in conds} for r in refs}
    pri = {r.key: hmm.fit_priors(pools_labels.seqs({r.key}, own)) for r in refs}
    ph = {r.key: {c: load_ph(mname, r, c) for c in conds} for r in refs}

    def rows_for(r, c, tau):
        f = feats[r.key][c]
        return hmm.decode(pa[r.key][f"pa_{c}"], ph[r.key][c], f["half"], pri[r.key], tau, "viterbi")

    def score_rows(r, c, rows, name):
        g = pools_labels.gt(r, name)
        if g is None:
            return None
        pred = to_native_codes(rows, feats[r.key][c]["idx"], g["step"], g["n_native"])
        return spell_score(g["y"], pred, sc_mode=r.is_sc)

    table = np.zeros((len(taus), len(refs)))
    for j, r in enumerate(refs):
        for ti, tau in enumerate(taus):
            rows = rows_for(r, "target", tau)
            table[ti, j] = np.mean([score_rows(r, "target", rows, n)["spell_f1"] for n in own])
    out = {}
    for j, r in enumerate(refs):
        tau = taus[int(np.argmax(np.delete(table, j, axis=1).mean(1)))] if len(refs) > 1 else taus[int(np.argmax(table[:, 0]))]
        out[("tau", r.key)] = tau
        for c in conds:
            rows = rows_for(r, c, tau)
            for name in evals:
                s = score_rows(r, c, rows, name)
                if s is not None:
                    out[(c, name, r.key)] = s
    log(f"[{time.strftime('%H:%M:%S')}] scored {mname} on {len(refs)} folds ({time.time() - t0:.0f}s)")
    return out


def regret(means: dict, models: list, evals: tuple) -> dict:
    """``means[(model, eval)]`` -> {model: (worst regret, eval where it happens)} with the best model
    per evaluation convention as the reference."""
    out = {}
    best = {e: max(means[(m, e)] for m in models) for e in evals}
    for m in models:
        rg = {e: best[e] - means[(m, e)] for e in evals}
        e_worst = max(rg, key=rg.get)
        out[m] = (rg[e_worst], e_worst, rg)
    return out
