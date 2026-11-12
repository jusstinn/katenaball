#!/usr/bin/env python3
"""AW: A (SkillCorner labels from its action events) with possession-length weights in both team heads.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final
    uv run python scripts/run_aw.py aw-gbdt               # LightGBM member, 28 folds (needs A's LightGBM cache)
    uv run python scripts/run_aw.py aw-tcn                # TCN member, 28 folds (--reverse: a second process)
    uv run python scripts/run_aw.py aw-gate               # the gate (needs A's two caches)

Pre-registered in docs/decision-log.md (2026-11-12, "AW (pre-registration)") before any score of A or of AW had
been read. External matches only; never reads ``data/`` or ``evaluation/gt/``; no locked bench.

AW is A with one more change: a training frame's weight in each member's team loss is ``lenweight.inv_length`` of
the label it trains on (1 / L, L the length in seconds of the frame's constant-label run, clipped to [1, 30],
scaled to mean 1 in the LightGBM head), exactly as S1 and S2 did on the standing labels (``scripts/run_s12.py``).
The labels are A's: the action labels on SkillCorner, the standing ones elsewhere. The alive head and the TCN's
alive loss carry no weight, so the LightGBM alive probabilities must equal A's byte for byte: checked on every
fold.

``--tag`` keeps a smoke run's files apart; the registered sizes can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_a as RA  # noqa: E402  (through run_g12: pins the CPU environment, loads libomp before torch and LightGBM)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G1  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import gbdt  # noqa: E402
from katena.models import lenweight as LW  # noqa: E402

G = RA.G
T, N_TREES, STEPS = RA.T, RA.N_TREES, RA.STEPS
STAGES = ("aw-gbdt", "aw-tcn", "aw-gate")
log, HEAD, _row = G.log, G.HEAD, G._row


class Names(RA.Names):
    def wg(self, n: int) -> str:
        return f"{self.p}aw_gbdt{n}aw"

    def wt(self, steps: int) -> str:
        return f"{self.p}aw_tcn{steps}taw"


def native_weights(dev: list) -> dict:
    """match key -> (1 / L for every native frame of the label the match trains on under A, the 5 Hz step)."""
    out = {}
    for r in dev:
        g = RA.load_act(r)
        out[r.key] = (LW.inv_length(g["y"], g["fps"]), g["step"])
    return out


def pool_weights(dev: list, wn: dict) -> np.ndarray:
    """One weight per row of ``FinalPools(dev)`` (same matches, same order)."""
    return np.concatenate([LW.at_rows(wn[r.key][0], d2.load_pool(r, F.MIX)["idx"], wn[r.key][1]) for r in dev])


def aw_gbdt(nm: Names, dev: list, folds: list, n: int) -> None:
    todo = [r for r in folds if not Lm.oof_path(nm.wg(n), T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev)
    cols, y, dead_ok = pools.cols(), RA.act_pool_y(dev, pools), np.ones(len(pools.o), bool)
    w = pool_weights(dev, native_weights(dev))
    if len(w) != len(y):
        raise RuntimeError("weights do not line up with the pool")
    log(f"pool {pools.X.shape}; weights {w.min():.4f} to {w.max():.4f}; LightGBM folds to do: {len(todo)}")
    for r in todo:
        t0 = time.time()
        m = ~np.isin(pools.keys, [r.key])
        heads = gbdt.fit_heads(pools.X[m], pools.o[m], y[m], dead_ok[m], cols, n_trees=n, team_weight=w[m])
        t1 = time.time()
        arrays = {}
        for c in G.worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = G.predict_at(heads, X0, X1, n)
        same = -1                                   # -1: no A file to compare with
        if Lm.oof_path(nm.g(n), T, r).exists():
            ref = Lm.load_oof(nm.g(n), T, r)
            same = int(all(arrays[k].tobytes() == ref[k].tobytes() for k in arrays if k.startswith("pa_")))
        d3._save(Lm.oof_path(nm.wg(n), T, r), {**arrays, "model_sha": np.array(F.gbdt_hash(heads)),
                                              "trees": np.array(n), "alive_same": np.array(same)})
        log(f"LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s, alive head as A's: {same}")


def aw_tcn(nm: Names, dev: list, folds: list, steps: int, reverse: bool = False, wait_s: float = 30.0) -> None:
    var = nm.wt(steps)
    order = list(reversed(folds)) if reverse else list(folds)
    out_of = lambda r: Lm.oof_path(var, T, r)                        # noqa: E731
    claim_of = lambda r: out_of(r).with_suffix(".claim")             # noqa: E731
    if all(out_of(r).exists() for r in order):
        return
    t0 = time.time()
    pools = F.FinalPools(dev)
    n_cols = pools.X.shape[1]
    scs = {r.key: G1.scaler(pools, {r.key}) for r in order if not out_of(r).exists()}
    del pools
    seqs, lost = RA.act_seqs(dev)
    wn, by_key = native_weights(dev), {r.key: r for r in dev}
    for (key, wld), s in seqs.items():
        with np.load(C.paths(by_key[key])["feat"](wld)) as z:
            idx = z["idx"]
        if len(idx) != len(s["y"]):
            raise RuntimeError(f"{key} {wld}: sequence rows do not line up")
        s["w"] = LW.at_rows(wn[key][0], idx, wn[key][1]).astype(np.float32)
    if {s["X"].shape[1] for s in seqs.values()} != {n_cols}:
        raise RuntimeError(f"sequence widths differ from the pool's {n_cols} columns")
    log(f"AW TCN: {len(scs)} scalers, {len(seqs)} sequences x {n_cols} columns, weighted team loss, {steps} steps; "
        f"finite values lost to float16: {lost} ({time.time() - t0:.0f}s)")
    for r in order:
        out, claim = out_of(r), claim_of(r)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                                                 # another process is training this fold
        try:
            t1 = time.time()
            model, sc = G1.fit_tcn(scs[r.key], seqs, {r.key}, G.SEED, E.TCN_THREADS, steps)
            t2 = time.time()
            arrays = G1.predict_tcn(model, sc, r, G.worlds_here(r))
            arrays["model_sha"] = np.array(E.model_hash(model))
            arrays["n_in"] = np.array(model.inp.in_channels)
            d3._save(out, arrays)
            log(f"AW TCN fold {r.key}: train {t2 - t1:.0f}s, predict {time.time() - t2:.0f}s")
        finally:
            claim.unlink(missing_ok=True)
    while any(claim_of(r).exists() for r in order):
        time.sleep(wait_s)


def score_all(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    need = (RA.INC_G, RA.INC_T, nm.g(n), nm.wg(n))
    for r in folds:
        if not all(Lm.oof_path(v, T, r).exists() for v in need):
            continue
        g0, t0, ga, gw = (Lm.load_oof(v, T, r) for v in need)
        ta = Lm.load_oof(nm.t(steps), T, r) if Lm.oof_path(nm.t(steps), T, r).exists() else None
        tw = Lm.load_oof(nm.wt(steps), T, r) if Lm.oof_path(nm.wt(steps), T, r).exists() else None
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        gta = RA.load_act(r) if r.is_sc else None
        for c in F.GATE_CONDS:
            rows = [("F3", g0, t0, F.W_F), ("AWG", gw, t0, 0.0), ("AG", ga, t0, 0.0)]
            if ta is not None:
                rows += [("A", ga, ta, F.W_F), ("AW1", gw, ta, F.W_F)]
            if tw is not None:
                rows += [("AW", gw, tw, F.W_F), ("AWT", gw, tw, 1.0)]
                if ta is not None:
                    rows += [("AW2", ga, tw, F.W_F), ("AT", ga, ta, 1.0)]
            for name, g, t, w in rows:
                s = G.blend_score(r, c, g, t, w, pri, gt)
                sc[(name, c, r.key)] = s
                sc[(name + "@act", c, r.key)] = RA.score_on(r, c, g, t, w, pri, gta) if r.is_sc else s
    with open(nm.file("aw_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    return sc


def gate(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    sc = score_all(nm, dev, folds, n, steps)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    cand = "AW" if any(k[0] == "AW" for k in sc) else "AW1"
    done = [r for r in folds if get(cand, "target", r) is not None]
    sc_refs, im_refs = [r for r in done if r.is_sc], [r for r in done if not r.is_sc]
    sets = (("all", done), ("SkillCorner", sc_refs), ("IDSSE + Metrica", im_refs))
    registered = not nm.p and cand == "AW" and n == N_TREES and steps == STEPS and len(done) == len(dev) == 28
    out = {"candidate": cand, "registered_run": registered, "folds": len(done), "trees": n, "steps": steps,
           "rule1": {}, "rule2": {}, "info": {}}
    run = "registered run" if registered else "NOT the registered run: partial, tagged or without a TCN"

    print(f"\n## Rule 1 (the standing rule): {cand} vs F3 on the standing truth, {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand, "F3", c)
            if g is not None:
                out["rule1"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs F3, {c}, {tag}", g))
    gt_, gs_, gsc = (out["rule1"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["passes_rule1"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nRULE 1 ({run}): adopted at target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on "
              f"SkillCorner at target = {gsc['worse'] if gsc else 'n/a'} -> {cand} "
              f"{'PASSES' if out['passes_rule1'] else 'fails'} rule 1")

    print(f"\n## Rule 2 (declared; counts only with David's and Justin's names): {cand} vs F3\n" + HEAD)
    for tag, refs, suffix in (("IDSSE + Metrica, one truth", im_refs, ""), ("SkillCorner, action truth", sc_refs, "@act"),
                              ("all, action truth on SkillCorner", done, "@act")):
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand + suffix, "F3" + suffix, c)
            if g is not None:
                out["rule2"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs F3, {c}, {tag}", g))
    out["passes_rule2"] = RA.rule2(out["rule2"].get("IDSSE + Metrica, one truth/target"),
                                   out["rule2"].get("IDSSE + Metrica, one truth/severe"),
                                   out["rule2"].get("SkillCorner, action truth/target"))
    print(f"\nRULE 2 ({run}): {cand} {'meets' if out['passes_rule2'] else 'does not meet'} rule 2")

    print("\n## AW against A (which of the two stands, and information)\n" + HEAD)
    pairs = [(cand, "A", f"{cand} vs A", ""), (cand, "A", f"{cand} vs A, action truth on SkillCorner", "@act"),
             ("AWG", "AG", "LightGBM member alone", ""), ("AW1", "A", "weights in the LightGBM member only, vs A", "")]
    if cand == "AW":
        pairs += [("AW2", "A", "weights in the TCN only, vs A", ""), ("AWT", "AT", "TCN member alone", "")]
    for a, b, label, suffix in pairs:
        for tag, refs in sets:
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, a + suffix, b + suffix, c)
                if g is not None:
                    out["info"][f"{label}/{tag}/{c}"] = g
                    print(_row(f"{label}, {c}, {tag}", g))

    print("\n## Information: true possessions found at IoU 0.5 by true length, target\n| folds | truth | model | under 2 s | "
          "2-4 s | 4-8 s | 8-16 s | over 16 s | predicted per true one |\n|---|---|---|---|---|---|---|---|---|")
    for tag, refs, act, truth in (("IDSSE + Metrica", im_refs, False, "one"), ("SkillCorner", sc_refs, False, "standing"),
                                  ("SkillCorner", sc_refs, True, "action")):
        for name in ("F3", "A", cand):
            s = [get(name + ("@act" if act else ""), "target", r) for r in refs]
            if not s or any(x is None for x in s):
                continue
            v = RA.recall_by_length(sc, name, refs, "target", act)
            per = sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)
            out["info"][f"recall/{tag}/{truth}/{name}"] = {"by_length": v, "per_true": per}
            print(f"| {tag} ({len(refs)}) | {truth} | {name} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {per:.3f} |")

    same = [int(Lm.load_oof(nm.wg(n), T, r)["alive_same"]) for r in done]
    out["checks"] = {"F3_equals_stored_scores": RA.stored_check(sc, "F3", done),
                     "alive_head_equals_A": (int(sum(x == 1 for x in same)), len(same))}
    print(f"\nchecks: {out['checks']}")
    G._dump(nm.file("aw_gate.json"), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every model file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="LightGBM size (needs --tag)")
    ap.add_argument("--steps", type=int, default=STEPS, help="TCN steps (needs --tag)")
    ap.add_argument("--reverse", action="store_true", help="aw-tcn: walk the folds from the other end")
    ap.add_argument("--inc", default=f"{RA.INC_G},{RA.INC_T}", help="F3's cached members, as in run_a.py")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and (a.trees != N_TREES or a.steps != STEPS):
        ap.error("the registered sizes can only be changed together with --tag")
    RA.INC_G, RA.INC_T = a.inc.split(",")
    nm = Names(a.tag)
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for r in dev:
        if r.is_sc:
            RA.build_act(r)
    for stage in a.stages:
        t0 = time.time()
        if stage == "aw-gbdt":
            aw_gbdt(nm, dev, folds, a.trees)
        elif stage == "aw-tcn":
            aw_tcn(nm, dev, folds, a.steps, a.reverse)
        elif stage == "aw-gate":
            gate(nm, dev, folds, a.trees, a.steps)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
