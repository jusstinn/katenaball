#!/usr/bin/env python3
"""S1 and S2: possession-length weights in the team heads of the two members. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final     # the cache `make model` built
    uv run python scripts/run_s12.py s1-gbdt s1-gate                  # S1 gate, 28 folds
    uv run python scripts/run_s12.py s2-train s2-report --folds 8     # S2 screen, 8 IDSSE + Metrica folds
    uv run python scripts/run_s12.py s2-train s2-report               # S2 gate, 28 folds
    scripts/night_s12.sh                                              # all of it, in that order

Pre-registered in docs/decision-log.md (2026-11-10, "S1 and S2 (pre-registration)") before any SkillCorner fold
of either candidate and before any fold of S2. Never reads ``data/`` or ``evaluation/gt/``, never trains on or
scores a locked bench. LOMO on the 27 other development matches. Every stage caches per fold under
``KATENA_WORK/oof/`` and skips what exists, so an interrupted run resumes.

The incumbent is the model that passed the night run of Nov 9 (``run_g12.py``): the 1,200-tree LightGBM member
(cache ``g1_gbdt1200``) blended at w = 0.5 with the 4,800-step TCN that reads the touch columns
(``g2_tcn4800t``). It is read from that cache, not refitted.

S1. The LightGBM member with its team head's rows weighted by ``lenweight.inv_length`` (1 / L, L the length in
seconds of the row's possession in the T = 2 training labels, clipped to [1, 30], scaled to mean 1). The alive
head is not weighted, so its probabilities must equal the cached member's byte for byte: checked on every fold.

S2. The same weights in the team loss of the 4,800-step touch TCN. Blended with the LightGBM member that stands
after S1's gate, against the unweighted TCN blended with that same member.

``--tag`` keeps a smoke run's files apart; the registered sizes can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g12 as G  # noqa: E402  (pins the CPU environment and loads libomp before torch and LightGBM)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import gbdt  # noqa: E402
from katena.models import lenweight as LW  # noqa: E402

T = P.T_MAIN
N_TREES = G.N_CAND                          # 1,200: the incumbent's and the candidate's size
STEPS = G.G2_STEPS                          # 4,800
INC_G, INC_T = "g1_gbdt1200", "g2_tcn4800t"  # the incumbent's cache (run_g12.py)
F_G, F_T = "g1_gbdt150", "g1_tcn"           # F, for information
STAGES = ("s1-gbdt", "s1-gate", "s2-train", "s2-report", "s2-pick")
log, HEAD, _row = G.log, G.HEAD, G._row


class Names:
    def __init__(self, tag: str = ""):
        self.p = f"{tag}_" if tag else ""

    def s1(self, n: int) -> str:
        return f"{self.p}s1_gbdt{n}w"

    def s2(self, steps: int) -> str:
        return f"{self.p}s2_tcn{steps}tw"

    def file(self, name: str) -> Path:
        return P.WORK / f"{self.p}{name}"


def native_weights(dev: list) -> dict:
    """match key -> (1 / L for every native frame of its T = 2 labels, the 5 Hz step)."""
    out = {}
    for r in dev:
        g = C.load_gt(r, T)
        out[r.key] = (LW.inv_length(g["y"], g["fps"]), g["step"])
    return out


def pool_weights(dev: list, wn: dict) -> np.ndarray:
    """One weight per row of ``FinalPools(dev)`` (same matches, same order)."""
    parts, agree = [], []
    for r in dev:
        p, g = d2.load_pool(r, F.MIX), C.load_gt(r, T)
        f = np.minimum(p["idx"].astype(np.int64) * g["step"], len(g["y"]) - 1)
        agree.append(float((g["y"][f] == p[f"y_T{T}"]).mean()))
        parts.append(LW.at_rows(wn[r.key][0], p["idx"], g["step"]))
    log(f"pool rows against the native labels: agreement {min(agree):.4f} to {max(agree):.4f} per match")
    assert min(agree) > 0.98, "pool rows do not line up with the native labels"
    return np.concatenate(parts)


def add_weights(dev: list, seqs: dict, wn: dict) -> None:
    """Give every TCN sequence its per-row team-loss weights (``tcn.train`` reads ``w``)."""
    by_key = {r.key: r for r in dev}
    for (key, wld), s in seqs.items():
        with np.load(C.paths(by_key[key])["feat"](wld)) as z:
            idx = z["idx"]
        w, step = wn[key]
        assert len(idx) == len(s["y"]), f"{key} {wld}: sequence rows do not line up"
        s["w"] = LW.at_rows(w, idx, step).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# S1

def s1_gbdt(nm: Names, dev: list, folds: list, n: int) -> None:
    todo = [r for r in folds if not Lm.oof_path(nm.s1(n), T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev)
    cols = pools.cols()
    w = pool_weights(dev, native_weights(dev))
    assert len(w) == len(pools.keys)
    log(f"pool {pools.X.shape}; weights {w.min():.4f} to {w.max():.4f}; LightGBM folds to do: {len(todo)}")
    for r in todo:
        t0 = time.time()
        m = ~np.isin(pools.keys, [r.key])
        heads = gbdt.fit_heads(pools.X[m], pools.o[m], pools.y[T][m], pools.dead_ok[m], cols, n_trees=n, team_weight=w[m])
        t1 = time.time()
        arrays = {}
        for c in G.worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = G.predict_at(heads, X0, X1, n)
        same = -1                                   # -1: no incumbent file to compare with
        if n == N_TREES and Lm.oof_path(INC_G, T, r).exists():
            ref = Lm.load_oof(INC_G, T, r)
            same = int(all(arrays[k].tobytes() == ref[k].tobytes() for k in arrays if k.startswith("pa_")))
        d3._save(Lm.oof_path(nm.s1(n), T, r), {**arrays, "model_sha": np.array(F.gbdt_hash(heads)),
                                              "trees": np.array(n), "alive_same": np.array(same)})
        log(f"LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s, alive head as cached: {same}")


def probe_match(sc: dict, refs: list) -> tuple:
    """(scores equal to the exploratory probe's ``inv`` at 1,200 trees, scores compared)."""
    p = P.WORK / "probe_levers.pkl"
    if not p.exists():
        return 0, 0
    with open(p, "rb") as fh:
        probe = pickle.load(fh)
    same = n = 0
    for r in refs:
        for c in F.GATE_CONDS:
            for kind, name in (("G", "S1G"), ("B", "S1")):
                old = probe.get((kind, "inv", N_TREES, c, r.key))
                if old is not None and (name, c, r.key) in sc:
                    n += 1
                    same += int(abs(old["spell_f1"] - sc[(name, c, r.key)]["spell_f1"]) < 1e-12)
    return same, n


def s1_gate(nm: Names, dev: list, folds: list, n: int) -> dict:
    src = F.PriorSource(dev)
    sc, alive = {}, []
    for r in folds:
        vs = (INC_G, nm.s1(n), INC_T, F_G, F_T)
        if not all(Lm.oof_path(v, T, r).exists() for v in vs):
            continue
        g0, g1, t, fg, ft = (Lm.load_oof(v, T, r) for v in vs)
        alive.append(int(g1["alive_same"]))
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in F.GATE_CONDS:
            for name, g, tc, w in (("LT", g0, t, F.W_F), ("S1", g1, t, F.W_F), ("LTG", g0, t, 0.0),
                                   ("S1G", g1, t, 0.0), ("F", fg, ft, F.W_F)):
                sc[(name, c, r.key)] = G.blend_score(r, c, g, tc, w, pri, gt)
    with open(nm.file("s1_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    done = [r for r in folds if get("S1", "target", r) is not None]
    sets = (("all", done), ("SkillCorner", [r for r in done if r.is_sc]), ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    registered = not nm.p and n == N_TREES and len(done) == len(dev) == 28
    out = {"registered_run": registered, "folds": len(done), "trees": n, "gate": {}, "info": {}}
    print(f"\n## S1 gate: length-weighted team head (S1) vs the unweighted {N_TREES}-tree member (LT), both blended with "
          f"the {STEPS}-step touch TCN, {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "S1", "LT", c)
            if g is not None:
                out["gate"][f"{tag}/{c}"] = g
                print(_row(f"S1 vs LT, {c}, {tag}", g))
    gt_, gs_, gsc = (out["gate"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["adopt_S1"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nDECISION ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): adopted at "
              f"target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on SkillCorner at target = "
              f"{gsc['worse'] if gsc else 'n/a'} -> {'S1 PASSES' if out['adopt_S1'] else 'the unweighted member stays'}")
    print("\n## Information: the LightGBM member alone (S1G vs LTG), and S1 against F\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            for a, b in (("S1G", "LTG"), ("S1", "F")):
                g = F.compare(get, refs, a, b, c)
                if g is not None:
                    out["info"][f"{a} vs {b}/{tag}/{c}"] = g
                    print(_row(f"{a} vs {b}, {c}, {tag}", g))
    print("\n## Information: predicted possessions per true one\n| cond | folds | LT | S1 |\n|---|---|---|---|")
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            if refs:
                v = {k: G.per_true(get, refs, k, c) for k in ("LT", "S1")}
                out["info"][f"per_true/{tag}/{c}"] = v
                print(f"| {c} | {tag} ({len(refs)}) | {v['LT']:.3f} | {v['S1']:.3f} |")
    out["checks"] = {"alive_head_equals_cache": (int(sum(a == 1 for a in alive)), len(alive)),
                     "scores_equal_probe_inv": probe_match(sc, [r for r in done if not r.is_sc])}
    print(f"\nchecks: {out['checks']}")
    G._dump(nm.file("s1_gate.json"), out)
    return out


# ---------------------------------------------------------------------------------------------
# S2

def s2_train(nm: Names, dev: list, folds: list, steps: int) -> None:
    var = nm.s2(steps)
    todo = [r for r in folds if not Lm.oof_path(var, T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev)
    t0 = time.time()
    seqs = F.tcn_seqs(dev, T, G.SEED)
    lost = G.add_touch(dev, seqs)
    add_weights(dev, seqs, native_weights(dev))
    log(f"S2: {len(seqs)} sequences x {next(iter(seqs.values()))['X'].shape[1]} columns, weighted team loss, "
        f"{steps} steps; finite values lost to float16: {lost} ({time.time() - t0:.0f}s)")
    for r in todo:
        t1 = time.time()
        model, sc = G.fit_tcn_touch(pools, seqs, r.key, steps)
        arrays = G.predict_tcn_touch(model, sc, r, G.worlds_here(r))
        arrays["model_sha"] = np.array(E.model_hash(model))
        d3._save(Lm.oof_path(var, T, r), arrays)
        log(f"S2 fold {r.key}: {time.time() - t1:.0f}s")


def s2_report(nm: Names, dev: list, folds: list, steps: int) -> dict:
    gate = json.loads(nm.file("s1_gate.json").read_text(encoding="utf-8"))
    side = {True: nm.s1(gate["trees"]), False: INC_G}
    use = bool(gate["adopt_S1"])
    need = (INC_G, nm.s1(gate["trees"]), INC_T, nm.s2(steps), F_G, F_T)
    done = [r for r in folds if all(Lm.oof_path(v, T, r).exists() for v in need)]
    src = F.PriorSource(dev)
    sc = {}
    for r in done:
        gb = {k: Lm.load_oof(v, T, r) for k, v in side.items()}
        t0, t1 = Lm.load_oof(INC_T, T, r), Lm.load_oof(nm.s2(steps), T, r)
        fg, ft = Lm.load_oof(F_G, T, r), Lm.load_oof(F_T, T, r)
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in F.GATE_CONDS:
            for k in (True, False):
                tag = "side" if k == use else "other"
                sc[(f"LT/{tag}", c, r.key)] = G.blend_score(r, c, gb[k], t0, F.W_F, pri, gt)
                sc[(f"S2/{tag}", c, r.key)] = G.blend_score(r, c, gb[k], t1, F.W_F, pri, gt)
            sc[("LT alone", c, r.key)] = G.blend_score(r, c, gb[False], t0, 1.0, pri, gt)
            sc[("S2 alone", c, r.key)] = G.blend_score(r, c, gb[False], t1, 1.0, pri, gt)
            sc[("F", c, r.key)] = G.blend_score(r, c, fg, ft, F.W_F, pri, gt)
    with open(nm.file(f"s2_scores_{len(folds)}.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    kind = "screen (8 IDSSE + Metrica folds)" if len(folds) == 8 else f"gate ({len(folds)} folds)"
    registered = (not nm.p and steps == STEPS and gate["registered_run"] and len(done) == len(folds)
                  and len(folds) in (8, 28))
    out = {"registered_run": registered, "kind": kind, "folds": len(done), "side_weighted": use, "steps": steps,
           "rule": {}, "info": {}}
    print(f"\n## S2 {kind}: length-weighted team loss in the {steps}-step touch TCN (S2) vs the unweighted one (LT), both "
          f"blended with the {'weighted' if use else 'unweighted'} LightGBM member (S1 "
          f"{'passed' if use else 'did not pass'}), {len(done)} folds scored\n" + HEAD)
    g = {c: F.compare(get, done, "S2/side", "LT/side", c) for c in F.GATE_CONDS}
    out["verdict"] = False
    if all(v is not None for v in g.values()):
        for c in F.GATE_CONDS:
            out["rule"][c] = g[c]
            print(_row(f"S2 vs LT, {c}", g[c]))
        out["verdict"] = F.decision(g["target"], g["severe"])
    print(f"\nVERDICT ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): "
          f"S2 {'passes' if out['verdict'] else 'fails'}")
    sets = (("SkillCorner", [r for r in done if r.is_sc]), ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    print("\n## Information: by dataset, with the other LightGBM member, the TCN alone, and against F\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            gg = F.compare(get, refs, "S2/side", "LT/side", c)
            if gg is not None:
                out["info"][f"side/{tag}/{c}"] = gg
                print(_row(f"S2 vs LT, {c}, {tag}", gg))
    for a, b, label in (("S2/other", "LT/other", f"with the {'unweighted' if use else 'weighted'} LightGBM member"),
                        ("S2 alone", "LT alone", "TCN alone"), ("S2/side", "F", "against F")):
        for c in F.GATE_CONDS:
            gg = F.compare(get, done, a, b, c)
            if gg is not None:
                out["info"][f"{label}/{c}"] = gg
                print(_row(f"S2 vs LT, {label}, {c}", gg))
    print("\n## Information: predicted possessions per true one, at target\n| blend | per true |\n|---|---|")
    for k in ("F", "LT/side", "S2/side"):
        if done:
            print(f"| {k} | {G.per_true(get, done, k, 'target'):.3f} |")
    G._dump(nm.file(f"s2_report_{len(folds)}.json"), out)
    return out


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="S1 size (needs --tag)")
    ap.add_argument("--steps", type=int, default=STEPS, help="S2 steps (needs --tag)")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and (a.trees != N_TREES or a.steps != STEPS):
        ap.error("the registered sizes can only be changed together with --tag")
    nm = Names(a.tag)
    if a.stages == ["s2-pick"]:                 # for night_s12.sh: prints "go" if the registered screen passed
        p = nm.file("s2_report_8.json")
        rep = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        print("go" if rep.get("verdict") and (rep.get("registered_run") or a.tag) else "")
        return 0
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for stage in a.stages:
        t0 = time.time()
        if stage == "s1-gbdt":
            s1_gbdt(nm, dev, folds, a.trees)
        elif stage == "s1-gate":
            s1_gate(nm, dev, folds, a.trees)
        elif stage == "s2-train":
            s2_train(nm, dev, folds, a.steps)
        elif stage == "s2-report":
            s2_report(nm, dev, folds, a.steps)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
