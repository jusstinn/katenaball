"""D3 gate: TCN + LightGBM ensemble against M2 on the development folds.

Uses the bake-off cache (features, pools, ``oof/full`` M2 probabilities, ``results.pkl``) and writes its
own outputs next to it: ``oof/<variant>/`` probability files and ``d3_results.pkl``. Never touches the
locked benches. Scores are external-match aggregates.

Candidates (all decoded like M2: 3-state Viterbi, priors fit on the training matches' labels, tau for
match k chosen from the other matches' out-of-fold scores at target-like noise):

- ``D3E50``  logit average of the M2 heads and the TCN, TCN weight 0.5 fixed a priori. The gated one.
- ``D3T``    the deterministic TCN alone (the M5 analogue, now on every fold).
- ``D3W*``   other fixed weights, a sensitivity curve only: choosing a weight from it would be
             choosing on the gate folds.
- ``D3N``    weight and tau chosen per outer M6 group from inner out-of-fold predictions of models that
             never saw the group (strictly nested). Needs ``stage_inner``.
"""

from __future__ import annotations

import time

import numpy as np

from katena.decode import hmm
from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import stages as S
from katena.eval.bootstrap import adopted, paired_bootstrap
from katena.eval.lomo import Pools, Results, timed
from katena.models import ensemble as E

VAR_TCN = "tcn_d3"                     # LOMO TCN probabilities, deterministic configuration
N_GROUPS = 6
RES_NAME = "d3_results.pkl"            # the 28-fold gate
RES_SCREEN = "d3_screen.pkl"           # the 8-fold screen (tau chosen among those 8 folds, as for M5)
GATE_CONDS = ("target", "severe")


def d3_results(name: str = RES_NAME) -> Results:
    return Results(P.WORK / name)


def _save(out, arrays: dict) -> None:
    """Write an npz atomically, so an interrupted run never leaves a truncated cache file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)


# ---------------------------------------------------------------------------------------------
# TCN training (deterministic configuration)

def tcn_seqs(dev: list, T: int = P.T_MAIN, seed: int = 0) -> dict:
    """The M5 training sequences (same orientation draws as ``stages.stage_tcn``)."""
    return S._tcn_seqs(dev, T, np.random.default_rng(P.SEED + seed))


def fit_tcn_excluding(pools: Pools, seqs: dict, exclude: set, seed: int = 0, threads: int = E.TCN_THREADS,
                      steps: int | None = None):
    """Scaler and TCN trained on every development sequence whose match is not in ``exclude``.
    Same scaler rows as ``stages.stage_tcn`` (every 6th pooled row of the training matches)."""
    from katena.models import tcn
    rows = np.flatnonzero(~np.isin(pools.keys, list(exclude)))[::6]
    sc = tcn.Scaler(pools.X[rows])
    tr = [{**s, "X": sc(s["X"].astype(np.float32))} for s in seqs.values() if s["key"] not in exclude]
    model = E.fit_tcn(tr, seed=seed, threads=threads, steps=steps)
    del tr
    return model, sc


def predict_tcn(model, sc, ref: P.Ref, conds, threads: int = E.TCN_THREADS) -> dict:
    from katena.models import tcn
    E.set_deterministic(threads)
    out = {}
    for c in conds:
        f = C.load_feat(ref, c)
        out[f"pa_{c}"], out[f"ph_{c}"] = tcn.predict(model, sc(f["X0"]), sc(f["X1"]), f["half"])
    return out


def stage_tcn_lomo(pools: Pools, dev: list, folds: list, T: int = P.T_MAIN, seed: int = 0,
                   threads: int = E.TCN_THREADS, variant: str = VAR_TCN, steps: int | None = None) -> None:
    """LOMO TCN for every ref in ``folds`` (trained on all other development matches); cached."""
    todo = [r for r in folds if not Lm.oof_path(variant, T, r).exists()]
    if not todo:
        return
    t0 = time.time()
    seqs = tcn_seqs(dev, T, seed)
    timed(f"TCN sequences ({len(seqs)})", t0)
    for r in todo:
        t1 = time.time()
        model, sc = fit_tcn_excluding(pools, seqs, {r.key}, seed, threads, steps)
        t2 = time.time()
        _save(Lm.oof_path(variant, T, r), predict_tcn(model, sc, r, S.ALL_EVAL_CONDS, threads))
        print(f"[{time.strftime('%H:%M:%S')}] TCN fold {r.key}: train {t2 - t1:.0f}s, "
              f"predict {time.time() - t2:.0f}s", flush=True)
    timed(f"TCN LOMO {variant} ({len(todo)} folds)", t0)


# ---------------------------------------------------------------------------------------------
# blending and scoring with the M2 protocol

def wname(w: float) -> str:
    return f"D3W{int(round(100 * w)):03d}"


def write_blend(refs: list, w: float, variant: str, T: int = P.T_MAIN, gbdt: str = "full",
                tcn_variant: str = VAR_TCN) -> None:
    """Pseudo-variant OOF files holding the blended probabilities, for ``bakeoff.decode_and_score``."""
    for r in refs:
        og, ot = Lm.load_oof(gbdt, T, r), Lm.load_oof(tcn_variant, T, r)
        arrays = {}
        for c in S.ALL_EVAL_CONDS:
            if f"pa_{c}" in og and f"pa_{c}" in ot:
                arrays[f"pa_{c}"], arrays[f"ph_{c}"] = E.blend_heads((og[f"pa_{c}"], og[f"ph_{c}"]),
                                                                     (ot[f"pa_{c}"], ot[f"ph_{c}"]), w)
        _save(Lm.oof_path(variant, T, r), arrays)


def stage_score(pools: Pools, refs: list, T: int = P.T_MAIN, weights=(E.W_MAIN, 1.0, 0.0),
                res: Results | None = None) -> Results:
    """Decode and score fixed-weight blends with exactly M2's protocol. ``w = 0`` must reproduce M2."""
    res = res or d3_results()
    for w in weights:
        name = {E.W_MAIN: "D3E50", 1.0: "D3T", 0.0: "D3M2"}.get(w, wname(w))
        write_blend(refs, w, f"blend_{name}", T)
        B.decode_and_score(res, pools, refs, f"blend_{name}", T, "viterbi", name, conds=S.ALL_EVAL_CONDS)
    return res


# ---------------------------------------------------------------------------------------------
# strictly nested choice of (w, tau) on M6 groups (optional, information)

def groups(dev: list, n_groups: int = N_GROUPS) -> dict:
    return {r.key: P.group_of(r.id, n_groups) for r in dev}


def stage_inner(pools: Pools, dev: list, T: int = P.T_MAIN, seed: int = 0, threads: int = E.TCN_THREADS,
                conds=GATE_CONDS, n_groups: int = N_GROUPS, steps: int | None = None) -> None:
    """For each unordered group pair (g, h): LightGBM heads and a TCN trained without g and h, predicting
    every match of g and h. Outer group g then has inner predictions for all matches outside g."""
    grp = groups(dev, n_groups)
    cols = pools.cols()
    seqs = None
    for g, h in E.pairs(n_groups):
        members = [r for r in dev if grp[r.key] in (g, h)]
        if not members:
            continue
        ex = {r.key for r in members}
        t0 = time.time()
        vg, vt = f"inner_gbdt_{g}{h}", f"inner_tcn_{g}{h}"
        if not all(Lm.oof_path(vg, T, r).exists() for r in members):
            heads = pools.fit(ex, T, cols)
            for r in members:
                arrays = {}
                for c in conds:
                    f = C.load_feat(r, c)
                    arrays[f"pa_{c}"], arrays[f"ph_{c}"] = heads.predict(f["X0"], f["X1"])
                _save(Lm.oof_path(vg, T, r), arrays)
        if not all(Lm.oof_path(vt, T, r).exists() for r in members):
            seqs = seqs if seqs is not None else tcn_seqs(dev, T, seed)
            model, sc = fit_tcn_excluding(pools, seqs, ex, seed, threads, steps)
            for r in members:
                _save(Lm.oof_path(vt, T, r), predict_tcn(model, sc, r, conds, threads))
        timed(f"inner pair {g}{h} ({len(members)} matches held out)", t0)


def nested_grid() -> list:
    return [(w, tau) for w in E.W_GRID for tau in P.TAUS]


def stage_nested(pools: Pools, dev: list, T: int = P.T_MAIN, res: Results | None = None,
                 n_groups: int = N_GROUPS) -> dict:
    """``D3N``: (w, tau) per outer group g, chosen on inner predictions for every match outside g from
    models trained without g (and without the match's own group), decoded with priors that exclude
    both groups. Then applied to the LOMO probabilities of g's matches. Returns {group: (w, tau)}."""
    res = res or d3_results()
    grp = groups(dev, n_groups)
    grid = nested_grid()
    members = {g: {r.key for r in dev if grp[r.key] == g} for g in range(n_groups)}
    feats = {r.key: C.load_feat(r, "target") for r in dev}
    gts = {r.key: C.load_gt(r, T) for r in dev}
    choice = {}
    t0 = time.time()
    for g in range(n_groups):
        if not members[g]:
            continue
        inner = [r for r in dev if grp[r.key] != g]
        table = np.zeros((len(grid), len(inner)))
        for j, r in enumerate(inner):
            a, b = sorted((g, grp[r.key]))
            og, ot = Lm.load_oof(f"inner_gbdt_{a}{b}", T, r), Lm.load_oof(f"inner_tcn_{a}{b}", T, r)
            pri = hmm.fit_priors(pools.label_seqs(members[a] | members[b], T))
            f = feats[r.key]
            for i, (w, tau) in enumerate(grid):
                pa, ph = E.blend_heads((og["pa_target"], og["ph_target"]), (ot["pa_target"], ot["ph_target"]), w)
                rows = hmm.decode(pa, ph, f["half"], pri, tau, "viterbi")
                table[i, j] = Lm.score_rows(r, T, "target", rows, f["idx"], gts[r.key])["spell_f1"]
        choice[g] = E.choose(table, grid)
        res.put(("d3n_table", T, g, "all"), {"table": table, "grid": grid, "keys": [r.key for r in inner]})
        timed(f"nested choice group {g}: w={choice[g][0]} tau={choice[g][1]}", t0)
    for r in dev:
        w, tau = choice[grp[r.key]]
        og, ot = Lm.load_oof("full", T, r), Lm.load_oof(VAR_TCN, T, r)
        pri = B.priors_for(pools, {r.key}, T)
        for c in S.ALL_EVAL_CONDS:
            if f"pa_{c}" not in og or f"pa_{c}" not in ot:
                continue
            f = feats[r.key] if c == "target" else C.load_feat(r, c)
            pa, ph = E.blend_heads((og[f"pa_{c}"], og[f"ph_{c}"]), (ot[f"pa_{c}"], ot[f"ph_{c}"]), w)
            rows = hmm.decode(pa, ph, f["half"], pri, tau, "viterbi")
            res.put(("D3N", T, c, r.key), Lm.score_rows(r, T, c, rows, f["idx"], gts[r.key]))
    res.put(("d3n_choice", T, "all", "all"), choice)
    res.save()
    return choice


# ---------------------------------------------------------------------------------------------
# gate

def gate(base: Results, d3: Results, refs: list, cand: str, inc: str = "M2", T: int = P.T_MAIN,
         conds=GATE_CONDS) -> dict:
    """Adoption rule per condition: wins on >= 70 % of matches and the paired 95 % interval excludes 0.
    Scores are looked up in the D3 store first, then in the bake-off store (where M2 lives)."""
    def get(name, c, r):
        k = (name, T, c, r.key)
        return d3.get(k) or base.get(k)
    out = {}
    for c in conds:
        rr = [r for r in refs if get(cand, c, r) and get(inc, c, r)]
        bs = paired_bootstrap([get(cand, c, r) for r in rr], [get(inc, c, r) for r in rr])
        bs["adopted"] = adopted(bs)
        bs["cand_mean"] = float(np.mean([get(cand, c, r)["spell_f1"] for r in rr])) if rr else float("nan")
        bs["inc_mean"] = float(np.mean([get(inc, c, r)["spell_f1"] for r in rr])) if rr else float("nan")
        out[c] = bs
    return out
