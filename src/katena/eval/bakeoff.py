"""Stages of the architecture bake-off (docs/bakeoff-protocol.md). Every stage is cached on disk
and resumable; ``scripts/run_bakeoff.py`` calls them in order."""

from __future__ import annotations

import time

import numpy as np

from katena.decode import hmm
from katena.eval import corpus as C
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval.bootstrap import adopted, paired_bootstrap
from katena.eval.lomo import Pools, Results, timed

VARIANTS = {
    "full": dict(drop=()),
    "nofree": dict(drop=("free",)),
    "freeonly": dict(keep=("free", "dead_player")),
    "nodead": dict(drop=("dead_ball", "dead_player")),
    "nomiss": dict(drop=("miss",)),
    "nomissw": dict(drop=("miss",)),      # same features as nomiss; IDSSE folds with the dead-coupled worlds
}
ALL_CONDS = ("native", "mild", "target", "severe")


def dev_and_locked():
    return P.split(P.all_refs())


# ---------------------------------------------------------------------------------------------
# stage: no-training baselines

def stage_baselines(res: Results, dev: list, T: int = P.T_MAIN, conds=ALL_CONDS + tuple(C.EXTRA_CONDS)) -> None:
    t0 = time.time()
    for r in dev:
        g = C.load_gt(r, T)
        for c in C.conds_for(r):
            if c not in conds or res.get(("M0", T, c, r.key)):
                continue
            f = C.load_feat(r, c)
            res.put(("M0", T, c, r.key), Lm.score_rows(r, T, c, Lm.m0_rows(f), f["idx"], g))
            res.put(("M1", T, c, r.key), Lm.score_rows(r, T, c, Lm.m1_rows(f), f["idx"], g))
    res.save()
    timed("baselines M0/M1", t0)


# ---------------------------------------------------------------------------------------------
# stage: LOMO heads

def ablation_folds(dev: list, n_sc: int = 6) -> list:
    """Declared scope cut (Amendment A6): all IDSSE and Metrica development matches plus the ``n_sc``
    SkillCorner matches with the smallest hash."""
    import hashlib
    sc = sorted([r for r in dev if r.is_sc], key=lambda r: hashlib.sha256(("abl|" + r.id).encode()).hexdigest())
    return [r for r in dev if not r.is_sc] + sc[:n_sc]


def stage_lomo(pools: Pools, dev: list, variant: str, T: int, folds: list | None = None,
               conds: list | None = None) -> None:
    t0 = time.time()
    cols = pools.cols(**VARIANTS[variant])
    for r in (folds or dev):
        if Lm.oof_path(variant, T, r).exists():
            continue
        Lm.run_fold(pools, variant, T, r, cols, exclude={r.key}, conds=conds)
    timed(f"LOMO heads {variant} T={T} ({len(folds or dev)} folds)", t0)


# ---------------------------------------------------------------------------------------------
# decoding with tau selection from the other matches' out-of-fold scores

def priors_for(pools: Pools, exclude: set, T: int) -> hmm.Priors:
    return hmm.fit_priors(pools.label_seqs(exclude, T))


def decode_and_score(res: Results, pools: Pools, refs: list, variant: str, T: int, kind: str,
                     model_name: str, conds=ALL_CONDS, extra_conds=(), oof_variant: str | None = None,
                     prior_exclude=None) -> dict:
    """Score ``kind`` ('viterbi'/'hsmm') on the stored OOF probabilities of ``variant`` for every ref in
    ``refs`` (LOMO). tau for match k is the best mean target-cond spellF1 over the other refs. Returns
    {match_key: tau}."""
    t0 = time.time()
    ov = oof_variant or variant
    feats = {r.key: C.load_feat(r, "target") for r in refs}
    oofs = {r.key: Lm.load_oof(ov, T, r) for r in refs}
    pex = prior_exclude or (lambda r: {r.key})
    pri = {r.key: priors_for(pools, pex(r), T) for r in refs}
    gts = {r.key: C.load_gt(r, T) for r in refs}
    table = np.zeros((len(P.TAUS), len(refs)))
    for j, r in enumerate(refs):
        for ti, tau in enumerate(P.TAUS):
            rows = Lm.decode_rows(oofs[r.key], "target", feats[r.key], pri[r.key], tau, kind)
            table[ti, j] = Lm.score_rows(r, T, "target", rows, feats[r.key]["idx"], gts[r.key])["spell_f1"]
    taus = {}
    for j, r in enumerate(refs):
        others = np.delete(table, j, axis=1).mean(1)
        taus[r.key] = P.TAUS[int(np.argmax(others))]
    for j, r in enumerate(refs):
        tau = taus[r.key]
        for c in tuple(conds) + tuple(x for x in extra_conds if x in C.conds_for(r)):
            if f"pa_{c}" not in oofs[r.key]:
                continue
            f = feats[r.key] if c == "target" else C.load_feat(r, c)
            rows = Lm.decode_rows(oofs[r.key], c, f, pri[r.key], tau, kind)
            res.put((model_name, T, c, r.key), Lm.score_rows(r, T, c, rows, f["idx"], gts[r.key]))
    res.put(("tau_table", T, model_name, "all"), {"table": table, "keys": [r.key for r in refs], "taus": taus})
    res.save()
    timed(f"decode+score {model_name} T={T}", t0)
    return taus


# ---------------------------------------------------------------------------------------------
# summaries

def per_match(res: Results, model: str, T: int, cond: str, refs: list) -> list:
    return [res.get((model, T, cond, r.key)) for r in refs]


def mean_scores(res: Results, model: str, T: int, cond: str, refs: list) -> dict:
    v = [x for x in per_match(res, model, T, cond, refs) if x]
    if not v:
        return {}
    return {"spell_f1": float(np.mean([x["spell_f1"] for x in v])), "macro_f1": float(np.mean([x["macro_f1"] for x in v])),
            "n": len(v)}


def compare(res: Results, a: str, b: str, T: int, cond: str, refs: list) -> dict:
    rr = [r for r in refs if res.get((a, T, cond, r.key)) and res.get((b, T, cond, r.key))]
    out = paired_bootstrap([res.get((a, T, cond, r.key)) for r in rr], [res.get((b, T, cond, r.key)) for r in rr])
    out["adopted"] = adopted(out)
    return out
