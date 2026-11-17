#!/usr/bin/env python3
"""AT and ATW: A's and AW's team labels, with the alive/dead training of F3. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<the cache of an F3 gate>
    uv run python scripts/run_at.py at-tcn                # TCN of AT, 28 folds (--reverse: a second process)
    uv run python scripts/run_at.py atw-tcn               # TCN of ATW
    uv run python scripts/run_at.py at-gate               # the gate (needs A's and AW's LightGBM caches)

Pre-registered in docs/decision-log.md (2026-11-13, "AT and ATW (pre-registration)") before any score of A, AW,
AT or ATW was known on the machine that wrote it. Never reads ``data/`` or ``evaluation/gt/``; no locked bench.

A (``scripts/run_a.py``) relabels the 20 SkillCorner matches from their action events and lets those rows train
the alive head and the TCN's alive output too. AT keeps the first half and drops the second:

* **Team heads:** A's, unchanged. The LightGBM team head *is* A's cached one. The TCN's team loss reads the action
  labels on SkillCorner rows, the standing labels elsewhere.
* **Alive/dead:** F3's, unchanged. The LightGBM alive head *is* F3's cached one (IDSSE and Metrica rows only). The
  TCN's alive loss reads IDSSE and Metrica sequences only: a SkillCorner sequence is not ``dead_ok``.

ATW is AT with AW's possession-length weights in both team heads: the LightGBM team head is AW's cached one, and
the TCN's team loss carries ``lenweight.inv_length`` of the label each frame trains on, as ``scripts/run_aw.py``.

So neither candidate fits a LightGBM model: only the two TCNs are trained here.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_aw as RW  # noqa: E402  (through run_a and run_g12: pins the CPU environment, loads libomp first)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G1  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import lenweight as LW  # noqa: E402

RA = RW.RA
G = RA.G
T, N_TREES, STEPS = RA.T, RA.N_TREES, RA.STEPS
STAGES = ("at-tcn", "atw-tcn", "at-gate")
CANDS = ("AT", "ATW")
log, HEAD, _row = G.log, G.HEAD, G._row


class Names(RW.Names):
    """``a_tag`` names another copy of A's and AW's cached members (for example one fitted on a GitHub runner and
    stored as ``gha_a_gbdt1200a``); by default they carry this run's own tag."""

    def __init__(self, tag: str = "", a_tag: str | None = None):
        super().__init__(tag)
        self.ap = self.p if a_tag is None else (f"{a_tag}_" if a_tag else "")

    def g(self, n: int) -> str:
        return f"{self.ap}a_gbdt{n}a"

    def t(self, steps: int) -> str:
        return f"{self.ap}a_tcn{steps}ta"

    def wg(self, n: int) -> str:
        return f"{self.ap}aw_gbdt{n}aw"

    def wt(self, steps: int) -> str:
        return f"{self.ap}aw_tcn{steps}taw"

    def tt(self, steps: int, weights: bool) -> str:
        tag = "atw" if weights else "at"
        return f"{self.p}{tag}_tcn{steps}t{tag}"


# ---------------------------------------------------------------------------------------------
# TCN members

def at_seqs(dev: list, weights: bool) -> tuple:
    """A's sequences (action labels on SkillCorner) with the SkillCorner ones not ``dead_ok``, so the alive loss
    reads IDSSE and Metrica only, as in F3. With ``weights``, AW's per-row weights of the team loss."""
    seqs, lost = RA.act_seqs(dev)
    by_key = {r.key: r for r in dev}
    wn = RW.native_weights(dev) if weights else None
    for (key, wld), s in seqs.items():
        r = by_key[key]
        if r.is_sc:
            s["dead_ok"] = False
        if weights:
            with np.load(C.paths(r)["feat"](wld)) as z:
                idx = z["idx"]
            if len(idx) != len(s["y"]):
                raise RuntimeError(f"{key} {wld}: sequence rows do not line up")
            s["w"] = LW.at_rows(wn[key][0], idx, wn[key][1]).astype(np.float32)
    return seqs, lost


def at_tcn(nm: Names, dev: list, folds: list, steps: int, weights: bool, reverse: bool = False,
           wait_s: float = 30.0) -> None:
    """The loop of ``run_a.a_tcn``: two processes may share the folds through claim files."""
    var = nm.tt(steps, weights)
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
    seqs, lost = at_seqs(dev, weights)
    if {s["X"].shape[1] for s in seqs.values()} != {n_cols}:
        raise RuntimeError(f"sequence widths differ from the pool's {n_cols} columns")
    log(f"{'ATW' if weights else 'AT'} TCN: {len(scs)} scalers, {len(seqs)} sequences x {n_cols} columns, {steps} steps; "
        f"sequences with a dead-ball label: {sum(bool(s['dead_ok']) for s in seqs.values())}; weighted team loss: "
        f"{weights}; finite values lost to float16: {lost} ({time.time() - t0:.0f}s)")
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
            log(f"{'ATW' if weights else 'AT'} TCN fold {r.key}: train {t2 - t1:.0f}s, predict {time.time() - t2:.0f}s")
        finally:
            claim.unlink(missing_ok=True)
    while any(claim_of(r).exists() for r in order):
        time.sleep(wait_s)


# ---------------------------------------------------------------------------------------------
# scores and gate

def _team_from(alive: dict, team: dict) -> dict:
    """A LightGBM probability file with ``alive``'s alive head and ``team``'s team head."""
    return {**alive, **{k: v for k, v in team.items() if k.startswith("ph_")}}


def score_all(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    ex = lambda v, r: Lm.oof_path(v, T, r).exists()                  # noqa: E731
    for r in folds:
        if not all(ex(v, r) for v in (RA.INC_G, RA.INC_T, nm.g(n))):
            continue
        g0, t0, ga = (Lm.load_oof(v, T, r) for v in (RA.INC_G, RA.INC_T, nm.g(n)))
        gw = Lm.load_oof(nm.wg(n), T, r) if ex(nm.wg(n), r) else None
        opt = {name: (Lm.load_oof(v, T, r) if ex(v, r) else None)
               for name, v in (("ta", nm.t(steps)), ("tw", nm.wt(steps)), ("tat", nm.tt(steps, False)),
                               ("tatw", nm.tt(steps, True)))}
        gat = _team_from(g0, ga)                                     # AT's LightGBM member
        gatw = _team_from(g0, gw) if gw is not None else None        # ATW's
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        gta = RA.load_act(r) if r.is_sc else None
        for c in F.GATE_CONDS:
            rows = [("F3", g0, t0, F.W_F), ("F3T", g0, t0, 1.0), ("AT1", gat, t0, F.W_F)]
            if opt["ta"] is not None:
                rows += [("A", ga, opt["ta"], F.W_F), ("A_T", ga, opt["ta"], 1.0)]
            if gw is not None and opt["tw"] is not None:
                rows += [("AW", gw, opt["tw"], F.W_F), ("AW_T", gw, opt["tw"], 1.0)]
            if opt["tat"] is not None:
                rows += [("AT", gat, opt["tat"], F.W_F), ("AT_T", gat, opt["tat"], 1.0), ("AT2", g0, opt["tat"], F.W_F)]
            if gatw is not None and opt["tatw"] is not None:
                rows += [("ATW", gatw, opt["tatw"], F.W_F), ("ATW_T", gatw, opt["tatw"], 1.0)]
            for name, g, t, w in rows:
                s = G.blend_score(r, c, g, t, w, pri, gt)
                sc[(name, c, r.key)] = s
                sc[(name + "@act", c, r.key)] = RA.score_on(r, c, g, t, w, pri, gta) if r.is_sc else s
    with open(nm.file("at_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    return sc


def gate(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    sc = score_all(nm, dev, folds, n, steps)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    have = lambda name: [r for r in folds if get(name, "target", r) is not None]  # noqa: E731
    out = {"trees": n, "steps": steps, "folds": {c: len(have(c)) for c in ("F3", "A", "AW") + CANDS}, "cands": {},
           "pairs": {}, "info": {}}
    print(f"\nfolds scored: {out['folds']}")

    for cand in CANDS:
        done = have(cand)
        if not done:
            continue
        sc_refs, im_refs = [r for r in done if r.is_sc], [r for r in done if not r.is_sc]
        sets = (("all", done), ("SkillCorner", sc_refs), ("IDSSE + Metrica", im_refs))
        registered = not nm.p and n == N_TREES and steps == STEPS and len(done) == len(dev) == 28
        run = "registered run" if registered else "NOT the registered run: partial or tagged"
        res = {"registered_run": registered, "folds": len(done), "rule1": {}, "rule2": {}}
        print(f"\n## Rule 1 (the standing rule): {cand} vs F3 on the standing truth, {len(done)} of {len(dev)} folds\n" + HEAD)
        for tag, refs in sets:
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, cand, "F3", c)
                if g is not None:
                    res["rule1"][f"{tag}/{c}"] = g
                    print(_row(f"{cand} vs F3, {c}, {tag}", g))
        gt_, gs_, gsc = (res["rule1"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
        res["passes_rule1"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
        print(f"\nRULE 1 ({run}): {cand} {'PASSES' if res['passes_rule1'] else 'fails'} rule 1")
        print(f"\n## Rule 2 (declared in A's entry; counts only with both names): {cand} vs F3\n" + HEAD)
        for tag, refs, suffix in (("IDSSE + Metrica, one truth", im_refs, ""),
                                  ("SkillCorner, action truth", sc_refs, "@act"),
                                  ("all, action truth on SkillCorner", done, "@act")):
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, cand + suffix, "F3" + suffix, c)
                if g is not None:
                    res["rule2"][f"{tag}/{c}"] = g
                    print(_row(f"{cand} vs F3, {c}, {tag}", g))
        res["passes_rule2"] = RA.rule2(res["rule2"].get("IDSSE + Metrica, one truth/target"),
                                       res["rule2"].get("IDSSE + Metrica, one truth/severe"),
                                       res["rule2"].get("SkillCorner, action truth/target"))
        print(f"\nRULE 2 ({run}): {cand} {'meets' if res['passes_rule2'] else 'does not meet'} rule 2")
        out["cands"][cand] = res

    print("\n## The paired tests that say which candidate stands (and information)\n" + HEAD)
    pairs = [("ATW", "AT", "weights, alive as F3: ATW vs AT"), ("AW", "A", "weights, alive as A: AW vs A"),
             ("AT", "A", "alive as F3, no weights: AT vs A"), ("ATW", "AW", "alive as F3, weights: ATW vs AW"),
             ("ATW", "A", "ATW vs A"), ("AT", "AW", "AT vs AW"),
             ("AT1", "F3", "A's LightGBM team head alone in F3 (AT's LightGBM member, F3's TCN)"),
             ("AT2", "F3", "AT's TCN alone in F3 (F3's LightGBM member)"),
             ("AT_T", "F3T", "TCN member alone: AT vs F3"), ("ATW_T", "F3T", "TCN member alone: ATW vs F3"),
             ("A_T", "F3T", "TCN member alone: A vs F3"), ("AW_T", "F3T", "TCN member alone: AW vs F3")]
    for a, b, label in pairs:
        both = [r for r in folds if get(a, "target", r) is not None and get(b, "target", r) is not None]
        if not both:
            continue
        sc_refs, im_refs = [r for r in both if r.is_sc], [r for r in both if not r.is_sc]
        for tag, refs, suffix in (("IDSSE + Metrica, one truth", im_refs, ""),
                                  ("SkillCorner, action truth", sc_refs, "@act"),
                                  ("all, action truth on SkillCorner", both, "@act"),
                                  ("all, standing truth", both, "")):
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, a + suffix, b + suffix, c)
                if g is not None:
                    out["pairs"][f"{label}/{tag}/{c}"] = g
                    print(_row(f"{label}, {c}, {tag}", g))

    print("\n## Information: frame-level parts and possessions found at IoU 0.5 by true length, target\n| folds | truth | "
          "model | alive/dead F1 | team accuracy | predicted per true one | under 2 s | 2-4 s | 4-8 s | 8-16 s | over 16 s |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("F3", "A", "AW") + CANDS:
        done = have(name)
        if not done:
            continue
        sc_refs, im_refs = [r for r in done if r.is_sc], [r for r in done if not r.is_sc]
        for tag, refs, act, truth in (("IDSSE + Metrica", im_refs, False, "one"),
                                      ("SkillCorner", sc_refs, False, "standing"),
                                      ("SkillCorner", sc_refs, True, "action")):
            s = [get(name + ("@act" if act else ""), "target", r) for r in refs]
            if not s:
                continue
            v = {"alive_dead_f1": float(np.mean([x["alive_dead_f1"] for x in s])),
                 "team_acc": float(np.mean([x["team_acc"] for x in s])),
                 "per_true": sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1),
                 "recall": RA.recall_by_length(sc, name, refs, "target", act)}
            out["info"][f"{tag}/{truth}/{name}"] = v
            print(f"| {tag} ({len(refs)}) | {truth} | {name} | {v['alive_dead_f1']:.3f} | {v['team_acc']:.3f} | "
                  f"{v['per_true']:.3f} | " + " | ".join(f"{x:.2f}" for x in v["recall"]) + " |")
    G._dump(nm.file("at_gate.json"), out)
    return out


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every model file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="LightGBM size of the cached members (needs --tag)")
    ap.add_argument("--steps", type=int, default=STEPS, help="TCN steps (needs --tag)")
    ap.add_argument("--reverse", action="store_true", help="the TCN stages: walk the folds from the other end")
    ap.add_argument("--inc", default=f"{RA.INC_G},{RA.INC_T}", help="F3's cached members, as in run_a.py")
    ap.add_argument("--a-tag", default=None,
                    help="at-gate: the tag A's and AW's cached members carry when it is not this run's (\"\" for none)")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and (a.trees != N_TREES or a.steps != STEPS):
        ap.error("the registered sizes can only be changed together with --tag")
    RA.INC_G, RA.INC_T = a.inc.split(",")
    nm = Names(a.tag, a.a_tag)
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for r in dev:
        if r.is_sc:
            RA.build_act(r)
    for stage in a.stages:
        t0 = time.time()
        if stage == "at-tcn":
            at_tcn(nm, dev, folds, a.steps, False, a.reverse)
        elif stage == "atw-tcn":
            at_tcn(nm, dev, folds, a.steps, True, a.reverse)
        elif stage == "at-gate":
            gate(nm, dev, folds, a.trees, a.steps)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
