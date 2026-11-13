#!/usr/bin/env python3
"""KIN: 35 ball-free timing columns in the LightGBM team head of A and of AW. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final
    uv run python scripts/run_kin.py kin-build [--workers 4]   # the columns of every world and of the pool rows
    uv run python scripts/run_kin.py kin-check                 # the code path equals the cache (no score)
    uv run python scripts/run_kin.py kin-gbdt                  # the two team heads, 28 folds (builds what is missing)
    uv run python scripts/run_kin.py kin-gate                  # the gate, where the base candidates' caches are

Pre-registered in docs/decision-log.md (2026-11-13, "KIN (pre-registration)") before any score. Never reads
``data/`` or ``evaluation/gt/``; no locked bench.

The change. ``katena.featurize.kin`` adds 35 columns that describe what the players do in the second (and the two
seconds) before a sample and after it, with no ball position. They go into the LightGBM team head and nowhere
else: the alive head, the TCN and the decoder are the base candidate's. Two team heads are fitted on every fold,
on the base candidate's rows, labels, parameters and trees, with the 213 + 35 columns:

* AK:  A's labels (the action labels on SkillCorner), unweighted, as ``run_a.py`` fits A's team head.
* AWK: the same labels with AW's possession-length weights, as ``run_aw.py`` fits AW's team head.

A file of this script holds a team head's probabilities only (``ph_<world>`` for the two gate worlds). The gate
puts them in place of the base candidate's team probabilities and leaves its alive probabilities and its TCN as
they are:

* ``_f``  blended with F3's TCN (an interim read: information only);
* no suffix: with the base candidate's 4,800-step TCN (information only);
* ``2``   with the base candidate's 9,600-step TCN: AK2 against A2 and AWK2 against AW2, the registered test.

``--tag`` names the base candidates' files (``<tag>_a_gbdt1200a`` ...) and this script's; the 9,600-step TCNs are
read under ``--tag2x`` (default ``<tag>2x``), as ``run_x2.py`` reads them.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_x2 as X2  # noqa: E402  (through run_aw, run_a and run_g12: pins the CPU environment, loads libomp first)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import kin_gate as KG  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import gbdt  # noqa: E402

RW = X2.RW
RA = RW.RA
G = RA.G
T, N_TREES, STEPS, STEPS2 = RA.T, RA.N_TREES, RA.STEPS, X2.STEPS2
STAGES = ("kin-build", "kin-check", "kin-gbdt", "kin-gate")
LEVELS = (("_f", "F3's TCN"), ("", f"the base's {STEPS}-step TCN"), ("2", f"the base's {STEPS2}-step TCN"))
log, HEAD, _row = G.log, G.HEAD, G._row


class Names(RW.Names):
    def kg(self, n: int) -> str:
        return f"{self.p}ak_gbdt{n}ak"

    def kwg(self, n: int) -> str:
        return f"{self.p}awk_gbdt{n}awk"


def fit_team(X: np.ndarray, o: np.ndarray, y: np.ndarray, n_trees: int, weight: np.ndarray | None = None):
    """``gbdt.fit_heads``'s team head alone: the same rows, target, weights, parameters and number of trees."""
    rows = np.flatnonzero((y == 0) | (y == 1))
    t = (y[rows] == o[rows]).astype(np.float32)
    wt = None if weight is None else weight[rows] / weight[rows].mean()
    return gbdt.lgb.train(gbdt.PARAMS, gbdt.lgb.Dataset(X[rows], t, weight=wt, free_raw_data=True), n_trees)


def predict_team(team, X0: np.ndarray, X1: np.ndarray, n: int) -> np.ndarray:
    return (0.5 * (team.predict(X0, num_iteration=n) + 1.0 - team.predict(X1, num_iteration=n))).astype(np.float32)


def team_hash(team) -> str:
    return hashlib.sha256(team.model_to_string().encode()).hexdigest()


def _pool(dev: list) -> tuple:
    """(pool with the kin columns, A's labels, AW's weights), rows as ``FinalPools(dev)``."""
    pools = KG.KinPools(dev)
    y = RA.act_pool_y(dev, pools)
    w = RW.pool_weights(dev, RW.native_weights(dev))
    if len(w) != len(y):
        raise RuntimeError("weights do not line up with the pool")
    return pools, y, w


def kin_build(dev: list, workers: int) -> dict:
    t0 = time.time()
    sts = KG.build_all(dev, workers)
    pools = KG.KinPools(dev)
    A = pools.X[:, pools.n_f:]
    fin = np.isfinite(A).mean(0)
    out = {"matches": len(dev), "matches_built_now": len(sts), "worlds_built_now": int(sum(s["built"] for s in sts)),
           "pool": list(pools.X.shape), "kin_columns": int(A.shape[1]), "finite_share_min": float(fin.min()),
           "finite_share_mean": float(fin.mean()), "columns_constant": int((np.nanstd(A, axis=0) == 0).sum()),
           "src": KG.SRC, "wall_s": time.time() - t0}
    log(f"kin-build: {out}")
    return out


def kin_check(nm: Names, dev: list, key: str, n: int) -> dict:
    """No score. On F3's 213 columns, ``fit_team`` gives the base candidates' cached team probabilities of one fold
    byte for byte (A's unweighted, AW's weighted), so the only difference between AK and A, and between AWK and
    AW, is the 35 columns. Also: the pool's kin rows are the per-world rows the pool was drawn from."""
    r = next(x for x in dev if x.key == key)
    pools, y, w = _pool(dev)
    m = ~np.isin(pools.keys, [key])
    Xb = pools.X[m][:, :pools.n_f]
    out = {"fold": key, "trees": n, "pool": list(pools.X.shape), "columns_before": int(pools.n_f)}
    for label, variant, weight in (("A", nm.g(n), None), ("AW", nm.wg(n), w[m])):
        if not Lm.oof_path(variant, T, r).exists():
            out[f"team_head_equals_{label}_cache"] = None
            continue
        team = fit_team(Xb, pools.o[m], y[m], n, weight)
        old = Lm.load_oof(variant, T, r)
        same = []
        for c in F.GATE_CONDS:
            _, X0, X1 = TG.features(r, c)
            same.append(predict_team(team, X0, X1, n).tobytes() == old[f"ph_{c}"].tobytes())
        out[f"team_head_equals_{label}_cache"] = bool(all(same))
    # the pool's kin rows of this match, rebuilt here from the per-world files by the pool's own layout
    with np.load(KG.pool_path(r)) as z:
        A, o = z["A"], z["o"]
    mix = d2.mix_for(r.dataset, F.MIX)
    n_rows = len(KG.load_kin(r, mix[C.POOL_CONDS[0]][0])["idx"])
    hit = off = 0
    for c, sel in d2.pool_rows(n_rows).items():
        oo, which = o[off:off + len(sel)], np.arange(len(sel)) % len(mix[c])
        for wi, wld in enumerate(mix[c]):
            k = KG.load_kin(r, wld)
            j = np.flatnonzero(which == wi)
            want = np.where(oo[j, None] == 1, k["A1"][sel[j]], k["A0"][sel[j]])
            hit += int(np.all((want == A[off + j]) | (np.isnan(want) & np.isnan(A[off + j])), axis=1).sum())
        off += len(sel)
    out["pool_rows_equal_their_world"] = [hit, int(len(A))]
    log(f"kin-check: {out}")
    return out


def kin_gbdt(nm: Names, dev: list, folds: list, n: int, workers: int) -> None:
    todo = [r for r in folds if not (Lm.oof_path(nm.kg(n), T, r).exists() and Lm.oof_path(nm.kwg(n), T, r).exists())]
    if not todo:
        return
    KG.build_all(dev, workers)                    # the pool needs every development match, whatever the folds
    pools, y, w = _pool(dev)
    log(f"pool {pools.X.shape} ({pools.n_f} + {pools.X.shape[1] - pools.n_f} columns); team heads to fit: 2 x {len(todo)}")
    for r in todo:
        m = ~np.isin(pools.keys, [r.key])
        X, o, yy = pools.X[m], pools.o[m], y[m]
        feats = {c: KG.features(r, c)[1:] for c in F.GATE_CONDS}
        for variant, weight in ((nm.kg(n), None), (nm.kwg(n), w[m])):
            out = Lm.oof_path(variant, T, r)
            if out.exists():
                continue
            t0 = time.time()
            team = fit_team(X, o, yy, n, weight)
            arrays = {f"ph_{c}": predict_team(team, X0, X1, n) for c, (X0, X1) in feats.items()}
            d3._save(out, {**arrays, "model_sha": np.array(team_hash(team)), "trees": np.array(n),
                           "columns": np.array(X.shape[1]), "src": np.array(KG.SRC)})
            log(f"team head {variant} fold {r.key}: {time.time() - t0:.0f}s")


def _with_team(base: dict, kin: dict) -> dict:
    """The base candidate's LightGBM probabilities with the team head's replaced (gate worlds only)."""
    return {**base, **{f"ph_{c}": kin[f"ph_{c}"] for c in F.GATE_CONDS}}


def score_all(nm: Names, tag2x: str, dev: list, folds: list, n: int, inc: tuple) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    ex = lambda v, r: Lm.oof_path(v, T, r).exists()                  # noqa: E731
    p2 = f"{tag2x}_" if tag2x else ""
    tcns = {"A": (nm.t(STEPS), f"{p2}a_tcn{STEPS2}ta"), "AW": (nm.wt(STEPS), f"{p2}aw_tcn{STEPS2}taw")}
    for r in folds:
        if not all(ex(v, r) for v in inc):
            continue
        g0, t0 = (Lm.load_oof(v, T, r) for v in inc)
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        gta = RA.load_act(r) if r.is_sc else None
        rows = [("F3", g0, t0, F.W_F)]
        for base, vg, vk in (("A", nm.g(n), nm.kg(n)), ("AW", nm.wg(n), nm.kwg(n))):
            if not (ex(vg, r) and ex(vk, r)):
                continue
            g = Lm.load_oof(vg, T, r)
            k = _with_team(g, Lm.load_oof(vk, T, r))
            rows += [(base + "_g", g, t0, 0.0), (base + "K_g", k, t0, 0.0), (base + "_f", g, t0, F.W_F),
                     (base + "K_f", k, t0, F.W_F)]
            for sfx, vt in zip(("", "2"), tcns[base]):
                if ex(vt, r):
                    t = Lm.load_oof(vt, T, r)
                    rows += [(base + sfx, g, t, F.W_F), (base + "K" + sfx, k, t, F.W_F)]
        for c in F.GATE_CONDS:
            for name, g, t, w in rows:
                s = G.blend_score(r, c, g, t, w, pri, gt)
                sc[(name, c, r.key)] = s
                sc[(name + "@act", c, r.key)] = RA.score_on(r, c, g, t, w, pri, gta) if r.is_sc else s
    with open(nm.file("kin_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    return sc


def gate(nm: Names, tag2x: str, dev: list, folds: list, n: int, inc: tuple) -> dict:
    sc = score_all(nm, tag2x, dev, folds, n, inc)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    out = {"trees": n, "src": KG.SRC, "cands": {}}
    views = (("all, action truth on SkillCorner", None, "@act"), ("IDSSE + Metrica, one truth", False, ""),
             ("SkillCorner, action truth", True, "@act"), ("all, standing truth", None, ""))
    for base in ("A", "AW"):
        for sfx, what in LEVELS:
            cand, inc_name = f"{base}K{sfx}", f"{base}{sfx}"
            done = [r for r in folds if get(cand, "target", r) is not None and get(inc_name, "target", r) is not None]
            if not done:
                continue
            pick = lambda sel: [r for r in done if sel is None or r.is_sc == sel]  # noqa: E731
            registered = sfx == "2" and n == N_TREES and len(done) == len(dev) == 28
            res = {"folds": len(done), "registered_run": registered, "rows": {}}
            print(f"\n## {cand} vs {inc_name}: the kin columns in the team head, {what}; {len(done)} of {len(dev)} folds\n" + HEAD)
            for label, sel, suffix in views:
                for c in F.GATE_CONDS:
                    g = F.compare(get, pick(sel), cand + suffix, inc_name + suffix, c)
                    if g is not None:
                        res["rows"][f"{label}/{c}"] = g
                        print(_row(f"{cand} vs {inc_name}, {c}, {label}", g))
            rr = res["rows"]
            res["passes"] = X2.passes(rr.get("all, action truth on SkillCorner/target"), rr.get("all, action truth on SkillCorner/severe"),
                                      rr.get("IDSSE + Metrica, one truth/target"), rr.get("SkillCorner, action truth/target"))
            f3 = {}
            for label, sel, suffix in views[1:3]:
                for c in F.GATE_CONDS:
                    g = F.compare(get, pick(sel), cand + suffix, "F3" + suffix, c)
                    if g is not None:
                        f3[f"{label}/{c}"] = res["rows"][f"vs F3/{label}/{c}"] = g
                        print(_row(f"{cand} vs F3, {c}, {label}", g))
            res["meets_rule2"] = RA.rule2(f3.get("IDSSE + Metrica, one truth/target"), f3.get("IDSSE + Metrica, one truth/severe"),
                                          f3.get("SkillCorner, action truth/target"))
            kind = "registered run" if registered else "NOT the registered run: information only"
            print(f"\nRULE ({kind}): {cand} {'passes' if res['passes'] else 'fails'} against {inc_name}; "
                  f"{'meets' if res['meets_rule2'] else 'does not meet'} rule 2 against F3")
            print("\n| folds, truth | model | under 2 s | 2-4 s | 4-8 s | 8-16 s | over 16 s | predicted per true one |\n"
                  "|---|---|---|---|---|---|---|---|")
            for label, sel, act in (("IDSSE + Metrica", False, False), ("SkillCorner, action", True, True)):
                refs = pick(sel)
                for name in (inc_name, cand):
                    s = [get(name + ("@act" if act else ""), "target", r) for r in refs]
                    if not s:
                        continue
                    v = RA.recall_by_length(sc, name, refs, "target", act)
                    per = sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)
                    res["rows"][f"recall/{label}/{name}"] = {"by_length": v, "per_true": per}
                    print(f"| {label} ({len(refs)}) | {name} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {per:.3f} |")
            out["cands"][cand] = res
        g = [get(f"{base}K_g", "target", r) for r in folds]
        if all(x is not None for x in g) and g:
            print("\n" + HEAD)
            for label, sel, suffix in views[:3]:
                refs = [r for r in folds if sel is None or r.is_sc == sel]
                c = F.compare(get, refs, f"{base}K_g" + suffix, f"{base}_g" + suffix, "target")
                if c is not None:
                    out["cands"].setdefault(f"{base}K_g", {})[label] = c
                    print(_row(f"LightGBM member alone: {base}K vs {base}, target, {label}", c))
    done = [r for r in folds if get("F3", "target", r) is not None]
    out["checks"] = {"F3_equals_stored_scores": RA.stored_check(sc, "F3", done)}
    print(f"\nchecks: {out['checks']}")
    G._dump(nm.file("kin_gate.json"), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="names the base candidates' files and this script's")
    ap.add_argument("--tag2x", default=None, help="tag of the 9,600-step TCNs (default: <tag>2x)")
    ap.add_argument("--trees", type=int, default=N_TREES, help="LightGBM size (smoke runs only; needs --tag)")
    ap.add_argument("--workers", type=int, default=2, help="processes for kin-build")
    ap.add_argument("--check-fold", default=F.DET_FOLD)
    ap.add_argument("--inc", default=f"{RA.INC_G},{RA.INC_T}", help="F3's cached members, as in run_a.py")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and a.trees != N_TREES:
        ap.error("the registered size can only be changed together with --tag")
    RA.INC_G, RA.INC_T = a.inc.split(",")
    nm = Names(a.tag)
    tag2x = a.tag2x if a.tag2x is not None else f"{a.tag}2x"
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'; kin {KG.SRC}")
    for r in dev:
        if r.is_sc:
            RA.build_act(r)
    for stage in a.stages:
        t0 = time.time()
        if stage == "kin-build":
            kin_build(dev, a.workers)
        elif stage == "kin-check":
            kin_check(nm, dev, a.check_fold, a.trees)
        elif stage == "kin-gbdt":
            kin_gbdt(nm, dev, folds, a.trees, a.workers)
        elif stage == "kin-gate":
            gate(nm, tag2x, dev, folds, a.trees, (RA.INC_G, RA.INC_T))
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
