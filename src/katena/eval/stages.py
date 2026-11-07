"""Heavier bake-off stages: label-semantics sweep, TCN (M5), spell rescoring (M6), oracles, error census,
locked benches. Same caching rules as ``bakeoff``."""

from __future__ import annotations

import time

import numpy as np

from katena.decode import hmm
from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import errors
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval.lomo import Pools, Results, timed
from katena.eval.score import score, to_native_codes
from katena.models import spells as SP

FOLDS8 = lambda dev: [r for r in dev if r.dataset in ("idsse", "metrica")]   # noqa: E731


# ---------------------------------------------------------------------------------------------
# label-semantics sweep

def stage_sweep(pools: Pools, res: Results, dev: list) -> None:
    t0 = time.time()
    f8 = FOLDS8(dev)
    metrica = {r.key for r in dev if r.dataset == "metrica"}
    cols = pools.cols()
    for T in P.T_SWEEP:
        # design A: all other development matches train (pre-registered)
        if T != P.T_MAIN:
            for r in f8:
                Lm.run_fold(pools, "full", T, r, cols, exclude={r.key}, conds=["target"])
        # design B: IDSSE + SkillCorner only (no Metrica rows at all)
        for r in f8:
            Lm.run_fold(pools, "xprov", T, r, cols, exclude={r.key} | metrica, conds=["target"])
    timed("sweep heads", t0)
    for T in P.T_SWEEP:
        for design, variant, excl in (("A", "full", set()), ("B", "xprov", metrica)):
            name = f"M3{design}_tr{T}"
            _decode_sweep(pools, res, f8, variant, T, name, excl)
    for T in P.T_SWEEP:                                             # no-training references per eval T
        for r in f8:
            g = C.load_gt(r, T)
            f = C.load_feat(r, "target")
            for nm, rows in (("M0", Lm.m0_rows(f)), ("M1", Lm.m1_rows(f))):
                res.put((nm, T, "target", r.key), Lm.score_rows(r, T, "target", rows, f["idx"], g))
    res.save()
    timed("sweep decode+score", t0)


def _decode_sweep(pools, res, f8, variant, trainT, name, excl):
    """Decode the stored target-cond probabilities once and score against GT of every eval T."""
    oofs = {r.key: Lm.load_oof(variant, trainT, r) for r in f8}
    feats = {r.key: C.load_feat(r, "target") for r in f8}
    pri = {r.key: hmm.fit_priors(pools.label_seqs({r.key} | excl, trainT)) for r in f8}
    table = np.zeros((len(P.TAUS), len(f8)))
    for j, r in enumerate(f8):
        for ti, tau in enumerate(P.TAUS):
            rows = Lm.decode_rows(oofs[r.key], "target", feats[r.key], pri[r.key], tau, "hsmm")
            table[ti, j] = Lm.score_rows(r, trainT, "target", rows, feats[r.key]["idx"])["spell_f1"]
    for j, r in enumerate(f8):
        tau = P.TAUS[int(np.argmax(np.delete(table, j, axis=1).mean(1)))]
        rows = Lm.decode_rows(oofs[r.key], "target", feats[r.key], pri[r.key], tau, "hsmm")
        for evalT in P.T_SWEEP:
            res.put((name, evalT, "target", r.key), Lm.score_rows(r, evalT, "target", rows, feats[r.key]["idx"]))


# ---------------------------------------------------------------------------------------------
# M5: TCN

def stage_tcn(pools: Pools, res: Results, dev: list, T: int = P.T_MAIN, seed: int = 0,
              variant: str = "M5") -> None:
    import torch  # noqa: F401  (imported lazily: only this stage needs it)

    from katena.models import tcn
    t0 = time.time()
    f8 = FOLDS8(dev)
    seqs = _tcn_seqs(dev, T, np.random.default_rng(P.SEED + seed))
    for r in f8:
        out = Lm.oof_path(variant, T, r)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        train = [s for s in seqs.values() if s["key"] != r.key]
        mask = pools.keys != r.key
        sc = tcn.Scaler(pools.X[mask][::6])
        tr = [{**s, "X": sc(s["X"].astype(np.float32))} for s in train]
        model = tcn.train(tr, seed=seed)
        arrays = {}
        for c in ALL_EVAL_CONDS:
            f = C.load_feat(r, c)
            pa, ph = tcn.predict(model, sc(f["X0"]), sc(f["X1"]), f["half"])
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = pa, ph
        np.savez(out, **arrays)
        timed(f"TCN fold {r.key}", t0)
    timed("TCN heads", t0)


ALL_EVAL_CONDS = ("native", "mild", "target", "severe")


def _tcn_seqs(dev: list, T: int, rng) -> dict:
    """Whole-match feature sequences (mild/target/severe; orientation drawn per sequence) for TCN training."""
    seqs = {}
    for r in dev:
        g = C.load_gt(r, T)
        y5 = g["y"][::g["step"]]
        for c in ("mild", "target", "severe"):
            f = C.load_feat(r, c)
            o = int(rng.integers(0, 2))
            seqs[(r.key, c)] = {"X": (f["X1"] if o else f["X0"]).astype(np.float16), "y": y5[f["idx"]],
                                "o": o, "dead_ok": r.dataset != "skillcorner", "key": r.key}
    return seqs


def stage_bench_tcn(pools: Pools, res: Results, dev: list, locked: list, T: int = P.T_MAIN) -> None:
    """M5 trained on all development matches, scored once on the locked benches."""
    from katena.models import tcn
    t0 = time.time()
    if res.get(("BENCH_M5", T, "target", locked[0].key)):
        return
    seqs = _tcn_seqs(dev, T, np.random.default_rng(P.SEED))
    sc = tcn.Scaler(pools.X[::6])
    model = tcn.train([{**s, "X": sc(s["X"].astype(np.float32))} for s in seqs.values()], seed=0)
    kind = "hsmm" if _kind_m5(res, T) == "hsmm" else "viterbi"
    tab = res.get(("tau_table", T, "M5", "all"))
    tau = P.TAUS[int(np.argmax(tab["table"].mean(1)))]
    pri = hmm.fit_priors(pools.label_seqs(set(), T))
    for r in locked:
        g = C.load_gt(r, T)
        for c in C.conds_for(r):
            f = C.load_feat(r, c)
            pa, ph = tcn.predict(model, sc(f["X0"]), sc(f["X1"]), f["half"])
            rows = hmm.decode(pa, ph, f["half"], pri, tau, kind)
            res.put(("BENCH_M5", T, c, r.key), Lm.score_rows(r, T, c, rows, f["idx"], g))
    res.save()
    timed("bench M5", t0)


def _kind_m5(res: Results, T: int) -> str:
    m2 = np.mean([v["spell_f1"] for k, v in res.d.items() if k[0] == "M2" and k[1] == T and k[2] == P.HEADLINE])
    m3 = np.mean([v["spell_f1"] for k, v in res.d.items() if k[0] == "M3" and k[1] == T and k[2] == P.HEADLINE])
    return "hsmm" if m3 >= m2 else "viterbi"


# ---------------------------------------------------------------------------------------------
# M6: nested grouped folds

def stage_m6_heads(pools: Pools, dev: list, T: int = P.T_MAIN, n_groups: int = 6) -> None:
    t0 = time.time()
    cols = pools.cols()
    grp = {r.key: P.group_of(r.id, n_groups) for r in dev}
    for g in range(n_groups):
        members = [r for r in dev if grp[r.key] == g]
        outer_ex = {r.key for r in members}
        for r in members:                                   # outer: trained without the whole group
            Lm.run_fold(pools, f"grp{g}_outer", T, r, cols, exclude=outer_ex)
        for h in range(n_groups):
            if h == g:
                continue
            inner = [r for r in dev if grp[r.key] == h]
            inner_ex = outer_ex | {r.key for r in inner}
            for r in inner:                                 # inner: without outer group g and own group h
                Lm.run_fold(pools, f"grp{g}_in{h}", T, r, cols, exclude=inner_ex, conds=list(("target", "severe")))
        timed(f"M6 group {g} heads", t0)


def stage_m6_score(pools: Pools, res: Results, dev: list, T: int = P.T_MAIN, n_groups: int = 6) -> None:
    t0 = time.time()
    grp = {r.key: P.group_of(r.id, n_groups) for r in dev}
    # outer OOF probabilities merged into one pseudo-variant so the shared decode/score code can use it
    for r in dev:
        out = Lm.oof_path("grouped", T, r)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **Lm.load_oof(f"grp{grp[r.key]}_outer", T, r))
    members = lambda r: {x.key for x in dev if grp[x.key] == grp[r.key]}   # noqa: E731
    taus = B.decode_and_score(res, pools, dev, "grouped", T, "hsmm", "M3g", prior_exclude=members)
    tau_tab = res.get(("tau_table", T, "M3g", "all"))
    tau_global = P.TAUS[int(np.argmax(tau_tab["table"].mean(1)))]
    for g in range(n_groups):
        # train the spell model on inner out-of-fold decodes of the other groups
        Fs, ys = [], []
        for h in range(n_groups):
            if h == g:
                continue
            for r in [x for x in dev if grp[x.key] == h]:
                oof = Lm.load_oof(f"grp{g}_in{h}", T, r)
                gt = C.load_gt(r, T)
                y5 = gt["y"][::gt["step"]]
                pri = hmm.fit_priors(pools.label_seqs({x.key for x in dev if grp[x.key] in (g, h)}, T))
                for c in ("target", "severe"):
                    f = C.load_feat(r, c)
                    rows = hmm.decode(oof[f"pa_{c}"], oof[f"ph_{c}"], f["half"], pri, tau_global, "hsmm")
                    segs, F, which = SP.spell_features(rows, oof[f"pa_{c}"], oof[f"ph_{c}"], f["X0"], list(f["names"]), f["half"])
                    Fs.append(F)
                    ys.append(SP.spell_targets(segs, which, y5[f["idx"]]))
        model = SP.fit(np.concatenate(Fs), np.concatenate(ys))
        for r in [x for x in dev if grp[x.key] == g]:
            oof = Lm.load_oof("grouped", T, r)
            pri_g = B.priors_for(pools, members(r), T)                     # same priors M3g used
            gt = C.load_gt(r, T)
            for c in ALL_EVAL_CONDS:
                f = C.load_feat(r, c)
                rows = hmm.decode(oof[f"pa_{c}"], oof[f"ph_{c}"], f["half"], pri_g, taus[r.key], "hsmm")
                segs, F, which = SP.spell_features(rows, oof[f"pa_{c}"], oof[f"ph_{c}"], f["X0"], list(f["names"]), f["half"])
                new = SP.apply(model, rows, segs, F, which)
                res.put(("M6", T, c, r.key), Lm.score_rows(r, T, c, new, f["idx"], gt))
        timed(f"M6 group {g} scored", t0)
    res.save()


# ---------------------------------------------------------------------------------------------
# oracles and error census (best model = M3 'full', LOMO out-of-fold probabilities)

def _gt_rows(r, T, f):
    g = C.load_gt(r, T)
    return g["y"][::g["step"]][f["idx"]], g


def stage_oracles(pools: Pools, res: Results, dev: list, T: int = P.T_MAIN, variant: str = "full") -> dict:
    t0 = time.time()
    refs = [r for r in dev if r.dataset in ("idsse", "metrica")]
    taus = res.get(("tau_table", T, "M3", "all"))["taus"]
    out = {k: [] for k in ("baseline", "clean_ball", "true_team", "true_alive", "true_both", "true_boundaries")}
    for r in refs:
        oof = Lm.load_oof(variant, T, r)
        f = C.load_feat(r, "target")
        y_rows, g = _gt_rows(r, T, f)
        pri = B.priors_for(pools, {r.key}, T)
        tau = taus[r.key]
        base = {"pa": oof["pa_target"], "ph": oof["ph_target"]}

        def run(pa, ph, feat=f):
            rows = hmm.decode(pa, ph, feat["half"], pri, tau, "hsmm")
            return Lm.score_rows(r, T, "target", rows, feat["idx"], g)

        out["baseline"].append((r.key, run(base["pa"], base["ph"])))
        if r.dataset == "idsse" and "pa_target_cb" in oof:
            fcb = C.load_feat(r, "target_cb")
            out["clean_ball"].append((r.key, run(oof["pa_target_cb"], oof["ph_target_cb"], fcb)))
        team = y_rows < 2
        ph_t = np.where(team, np.where(y_rows == 0, 0.9999, 0.0001), base["ph"])
        pa_t = np.where(y_rows < 3, np.where(y_rows == 2, 0.0001, 0.9999), base["pa"])
        out["true_team"].append((r.key, run(base["pa"], ph_t)))
        out["true_alive"].append((r.key, run(pa_t, base["ph"])))
        out["true_both"].append((r.key, run(pa_t, ph_t)))
        # perfect boundaries: GT segmentation, predicted team per spell (mean team probability)
        rows = y_rows.copy()
        rows[rows == 3] = 2
        for a, b, lab, _ in SP._segments(np.where(y_rows == 3, 2, y_rows).astype(np.int8), f["half"]):
            if lab < 2:
                rows[a:b] = 0 if base["ph"][a:b].mean() >= 0.5 else 1
        out["true_boundaries"].append((r.key, Lm.score_rows(r, T, "target", rows.astype(np.int8), f["idx"], g)))
    for k, v in out.items():
        for key, s in v:
            res.put((f"ORACLE_{k}", T, "target", key), s)
    res.save()
    timed("oracles", t0)
    return out


def stage_census(pools: Pools, res: Results, dev: list, T: int = P.T_MAIN, variant: str = "full") -> dict:
    t0 = time.time()
    taus = res.get(("tau_table", T, "M3", "all"))["taus"]
    agg = {}
    for r in dev:
        oof = Lm.load_oof(variant, T, r)
        f = C.load_feat(r, "target")
        g = C.load_gt(r, T)
        pri = B.priors_for(pools, {r.key}, T)
        rows = hmm.decode(oof["pa_target"], oof["ph_target"], f["half"], pri, taus[r.key], "hsmm")
        pred = to_native_codes(rows, f["idx"], g["step"], g["n_native"], g["blocks"])
        if r.is_sc:
            pred = np.where(pred == 2, 3, pred).astype(np.int8)
        cen = errors.census(g["y"], pred)
        d = agg.setdefault(r.dataset, {"n_gt": 0, "types": dict.fromkeys(errors.TYPES, 0), "sp_dead": 0, "sp_other": 0,
                                        "n_pred": 0, "dead_miss_frac": [], "false_dead_frac": []})
        d["n_gt"] += cen["n_gt"]
        d["n_pred"] += cen["n_pred"]
        for k, v in cen["gt_spells"].items():
            d["types"][k] += v
        d["sp_dead"] += cen["spurious_pred"]["in_dead"]
        d["sp_other"] += cen["spurious_pred"]["in_other"]
        d["dead_miss_frac"].append(cen["dead_frame_err"]["gt_dead_pred_alive"])
        d["false_dead_frac"].append(cen["dead_frame_err"]["gt_alive_pred_dead"])
    timed("census", t0)
    return agg


# ---------------------------------------------------------------------------------------------
# locked benches

def stage_bench(pools_all: Pools, res: Results, dev: list, locked: list, T: int = P.T_MAIN) -> None:
    """Train on every development match, score the two locked benches once under each severity."""
    t0 = time.time()
    if res.get(("BENCH_M3", T, "target", locked[0].key)):          # scored once; never recomputed
        return
    exclude: set = set()
    pri = hmm.fit_priors(pools_all.label_seqs(exclude, T))
    tau_tab = res.get(("tau_table", T, "M3", "all"))
    tau = P.TAUS[int(np.argmax(tau_tab["table"].mean(1)))]
    tau_v = res.get(("tau_table", T, "M2", "all"))
    tau2 = P.TAUS[int(np.argmax(tau_v["table"].mean(1)))]
    for variant, vk in B.VARIANTS.items():
        cols = pools_all.cols(**vk)
        heads = pools_all.fit(exclude, T, cols)
        for r in locked:
            g = C.load_gt(r, T)
            for c in C.conds_for(r):
                f = C.load_feat(r, c)
                pa, ph = heads.predict(f["X0"], f["X1"])
                kinds = (("M2", "viterbi", tau2), ("M3", "hsmm", tau)) if variant == "full" else ((f"M4_{variant}", "hsmm", tau),)
                for name, kind, tu in kinds:
                    rows = hmm.decode(pa, ph, f["half"], pri, tu, kind)
                    res.put(("BENCH_" + name, T, c, r.key), Lm.score_rows(r, T, c, rows, f["idx"], g))
        timed(f"bench variant {variant}", t0)
    for r in locked:
        g = C.load_gt(r, T)
        for c in C.conds_for(r):
            f = C.load_feat(r, c)
            res.put(("BENCH_M0", T, c, r.key), Lm.score_rows(r, T, c, Lm.m0_rows(f), f["idx"], g))
            res.put(("BENCH_M1", T, c, r.key), Lm.score_rows(r, T, c, Lm.m1_rows(f), f["idx"], g))
    res.save()
