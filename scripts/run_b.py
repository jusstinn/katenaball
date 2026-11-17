#!/usr/bin/env python3
"""B1: a boundary head that moves the decoder's switch probability in time. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final     # the cache of scripts/night_g12.sh
    uv run python scripts/run_b.py b-check                 # lam = 0 decodes exactly as the incumbent (no new score)
    uv run python scripts/run_b.py b-head b-gate           # B1 gate, 28 folds
    uv run python scripts/run_b.py b-gate --on k           # the pair: B1's decoder on K's probabilities, against K

Pre-registered in docs/decision-log.md (2026-11-11, "K1, K and B1 (pre-registration)") before any score of the
candidate. Never reads ``data/`` or ``evaluation/gt/``, never trains on or scores a locked bench. LOMO on the 27
other development matches. Every stage caches per fold under ``KATENA_WORK/oof/`` and skips what exists.

The incumbent is F3 (caches ``g1_gbdt1200`` and ``g2_tcn4800t``, w = 0.5, Viterbi at tau 2), read, not refitted.
B1 keeps F3's probabilities and changes only the decoder:

* The boundary head: one LightGBM model (F's parameters, 1,200 trees, F's 213 columns and ``d2j`` pool) trained on
  ``katena.models.boundary.change_target``: does the owner change within 0.4 s of this sample, in the T = 2
  labels with unknown samples filled with the last owner? Rows of all 27 training matches whose owner is known.
* The decoder: ``katena.decode.tv``. The odds of every switch into a sample are multiplied by the head's odds
  there over its base rate (the mean target of the fold's training rows), to the power ``LAM`` = 1.

``--tag`` keeps a smoke run's files apart; the registered size can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g12 as G  # noqa: E402  (pins the CPU environment and loads libomp before torch and LightGBM)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from katena.decode import hmm  # noqa: E402
from katena.decode import tv  # noqa: E402
from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import boundary as BD  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import gbdt  # noqa: E402

T = P.T_MAIN
N_TREES = G.N_CAND                          # 1,200, the incumbent member's size
LAM = 1.0                                   # registered
LAMS_INFO = (0.5, 2.0)                      # information only
INC_G, INC_T = "g1_gbdt1200", "g2_tcn4800t"
# --on: (LightGBM cache, TCN cache, name of the incumbent, name of the candidate)
ON = {"f3": (INC_G, INC_T, "F3", "B1"), "k1": ("k1_gbdt1200k", INC_T, "K1", "B1+K1"),
      "k": ("k1_gbdt1200k", "k2_tcn4800tk", "K", "B1+K")}
STAGES = ("b-check", "b-head", "b-gate")
log, HEAD, _row = G.log, G.HEAD, G._row


class Names:
    def __init__(self, tag: str = ""):
        self.p = f"{tag}_" if tag else ""

    def head(self, n: int) -> str:
        return f"{self.p}b1_head{n}"

    def file(self, name: str) -> Path:
        return P.WORK / f"{self.p}{name}"


def match_target(r: P.Ref) -> tuple:
    """(target, known) on the work grid of one match, from its T = 2 labels."""
    g = C.load_gt(r, T)
    return BD.change_target(g["y"][::g["step"]], BD.blocks5(g["blocks"], g["step"]))


def pool_targets(dev: list) -> tuple:
    """(target, known) for every row of ``FinalPools(dev)`` (same matches, same order)."""
    tg, kn, agree = [], [], []
    for r in dev:
        p, g = d2.load_pool(r, F.MIX), C.load_gt(r, T)
        agree.append(float((g["y"][::g["step"]][p["idx"]] == p[f"y_T{T}"]).mean()))
        t, k = match_target(r)
        tg.append(t[p["idx"]])
        kn.append(k[p["idx"]])
    log(f"pool rows against the work-grid labels: agreement {min(agree):.4f} to {max(agree):.4f} per match")
    assert min(agree) > 0.98, "pool rows do not line up with the labels"
    return np.concatenate(tg), np.concatenate(kn)


def avg_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    hit = y[order].astype(np.float64)
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float((prec * hit).sum() / max(hit.sum(), 1.0))


def b_head(nm: Names, dev: list, folds: list, n: int) -> None:
    todo = [r for r in folds if not Lm.oof_path(nm.head(n), T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev)
    tg, kn = pool_targets(dev)
    assert len(tg) == len(pools.keys)
    log(f"pool {pools.X.shape}; rows with a target {kn.mean():.3f}; positive share of those {tg[kn].mean():.4f}; "
        f"boundary heads to fit: {len(todo)}")
    for r in todo:
        t0 = time.time()
        rows = np.flatnonzero(kn & ~np.isin(pools.keys, [r.key]))
        head = lgb.train(gbdt.PARAMS, lgb.Dataset(pools.X[rows], tg[rows].astype(np.float32), free_raw_data=True), n)
        t1 = time.time()
        arrays = {"base": np.array(float(tg[rows].mean())), "trees": np.array(n)}
        for c in G.worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pc_{c}"] = (0.5 * (head.predict(X0) + head.predict(X1))).astype(np.float32)
        f = F._meta(r, "target")
        t, k = match_target(r)
        ap = avg_precision(t[f["idx"]][k[f["idx"]]], arrays["pc_target"][k[f["idx"]]])
        arrays["ap_target"] = np.array(ap)
        d3._save(Lm.oof_path(nm.head(n), T, r), arrays)
        log(f"boundary head fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s, base rate "
            f"{float(arrays['base']):.4f}, held-out average precision at target {ap:.3f}")


def _probs(r: P.Ref, c: str, on: str) -> tuple:
    vg, vt = ON[on][:2]
    g, t = Lm.load_oof(vg, T, r), Lm.load_oof(vt, T, r)
    return E.blend_heads((g[f"pa_{c}"], g[f"ph_{c}"]), (t[f"pa_{c}"], t[f"ph_{c}"]), F.W_F)


def b_check(nm: Names, dev: list, key: str) -> dict:
    """No new score. With lam = 0 ``tv.decode`` returns the incumbent's decoded rows exactly."""
    r = next(x for x in dev if x.key == key)
    pri = B.priors_for(F.PriorSource(dev), {r.key}, T)
    same = []
    for c in F.GATE_CONDS:
        pa, ph = _probs(r, c, "f3")
        f = F._meta(r, c)
        a = hmm.decode(pa, ph, f["half"], pri, F.TAU, "viterbi")
        b = tv.decode(pa, ph, np.full(len(pa), 0.5, np.float32), f["half"], pri, F.TAU, 0.0, 0.1)
        same.append(bool(np.array_equal(a, b)))
    out = {"fold": key, "lam0_equals_incumbent_decode": all(same)}
    log(f"b-check: {out}")
    return out


def b_gate(nm: Names, dev: list, folds: list, n: int, on: str) -> dict:
    vg, vt, inc, cand = ON[on]
    src = F.PriorSource(dev)
    sc, same0, aps = {}, [], []
    for r in folds:
        if not all(Lm.oof_path(v, T, r).exists() for v in (vg, vt, nm.head(n))):
            continue
        bh = Lm.load_oof(nm.head(n), T, r)
        aps.append(float(bh["ap_target"]))
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in G.worlds_here(r):
            pa, ph = _probs(r, c, on)
            f = F._meta(r, c)
            rows0 = hmm.decode(pa, ph, f["half"], pri, F.TAU, "viterbi")
            sc[(inc, c, r.key)] = Lm.score_rows(r, T, c, rows0, f["idx"], gt)
            dec = lambda lam: tv.decode(pa, ph, bh[f"pc_{c}"], f["half"], pri, F.TAU, lam, float(bh["base"]))  # noqa: E731
            sc[(cand, c, r.key)] = Lm.score_rows(r, T, c, dec(LAM), f["idx"], gt)
            if c in F.GATE_CONDS:
                same0.append(bool(np.array_equal(dec(0.0), rows0)))
                for lam in LAMS_INFO:
                    sc[(f"{cand}@{lam}", c, r.key)] = Lm.score_rows(r, T, c, dec(lam), f["idx"], gt)
    with open(nm.file(f"b_scores_{on}.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    done = [r for r in folds if get(cand, "target", r) is not None]
    sets = (("all", done), ("SkillCorner", [r for r in done if r.is_sc]), ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    registered = not nm.p and n == N_TREES and len(done) == len(dev) == 28
    out = {"candidate": cand, "incumbent": inc, "registered_run": registered, "folds": len(done), "trees": n,
           "lam": LAM, "gate": {}, "info": {}}
    print(f"\n## {cand} gate: {cand} vs {inc}, {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand, inc, c)
            if g is not None:
                out["gate"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs {inc}, {c}, {tag}", g))
    gt_, gs_, gsc = (out["gate"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["passes"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nDECISION ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): adopted at "
              f"target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on SkillCorner at target = "
              f"{gsc['worse'] if gsc else 'n/a'} -> {cand + ' PASSES' if out['passes'] else cand + ' fails'}")
    print("\n## Information: other values of lam (not candidates)\n" + HEAD)
    for lam in LAMS_INFO:
        for c in F.GATE_CONDS:
            g = F.compare(get, done, f"{cand}@{lam}", inc, c)
            if g is not None:
                out["info"][f"lam{lam}/{c}"] = g
                print(_row(f"lam = {lam} vs {inc}, {c}", g))
    print("\n## Information: every other evaluation world (coupled worlds: IDSSE + Metrica only)\n" + HEAD)
    for wld in G.W.EVAL_WORLDS:
        if wld not in F.GATE_CONDS:
            g = F.compare(get, done, cand, inc, wld)
            if g is not None:
                out["info"][f"world/{wld}"] = g
                print(_row(f"{cand} vs {inc}, {wld}", g))
    print(f"\n## Information: predicted possessions per true one\n| cond | folds | {inc} | {cand} |\n|---|---|---|---|")
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            if refs:
                v = {k: G.per_true(get, refs, k, c) for k in (inc, cand)}
                out["info"][f"per_true/{tag}/{c}"] = v
                print(f"| {c} | {tag} ({len(refs)}) | {v[inc]:.3f} | {v[cand]:.3f} |")
    out["checks"] = {"lam0_equals_incumbent_decode": (int(sum(same0)), len(same0))}
    stored = P.WORK / "s1_scores.pkl"
    if on == "f3" and stored.exists():
        with open(stored, "rb") as fh:
            old = pickle.load(fh)
        pairs = [(old.get(("LT", c, r.key)), sc[(inc, c, r.key)]) for r in done for c in F.GATE_CONDS]
        out["checks"]["F3_equals_stored_scores"] = (int(sum(F.same_score(a, b) for a, b in pairs if a is not None)),
                                                    int(sum(a is not None for a, _ in pairs)))
    if aps:
        out["info"]["head_average_precision_target"] = [min(aps), float(np.mean(aps)), max(aps)]
        print(f"\nboundary head, held-out average precision at target: {min(aps):.3f} to {max(aps):.3f}, "
              f"mean {np.mean(aps):.3f}")
    print(f"checks: {out['checks']}")
    G._dump(nm.file(f"b_gate_{on}.json"), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="boundary head size (needs --tag)")
    ap.add_argument("--on", default="f3", choices=sorted(ON), help="b-gate: whose probabilities are decoded")
    ap.add_argument("--check-fold", default=F.DET_FOLD)
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and a.trees != N_TREES:
        ap.error("the registered size can only be changed together with --tag")
    nm = Names(a.tag)
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for stage in a.stages:
        t0 = time.time()
        if stage == "b-check":
            b_check(nm, dev, a.check_fold)
        elif stage == "b-head":
            b_head(nm, dev, folds, a.trees)
        elif stage == "b-gate":
            b_gate(nm, dev, folds, a.trees, a.on)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
