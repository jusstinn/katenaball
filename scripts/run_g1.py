#!/usr/bin/env python
"""Gate G1 (F with its TCN member on 213 columns: M2's 160 plus D1's 53 touch columns) against F, on the 28
external development folds. Pre-registered in docs/decision-log.md before any G1 score (commit 115f7e2); the
candidate and the gate are fixed by docs/plan-final-week.md (commit ff2e0b1).

External matches only: never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench.
The F cache is read only; everything is written under KATENA_WORK. The TCN stage goes through the slot runner:

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-g1 KATENA_FINAL_CACHE=D:/katena/work-final
    PY=.venv/Scripts/python.exe; H="python D:/katena/tools/heavy.py run --name g1 --"
    $PY scripts/run_g1.py seed                 # copy F's inputs and probabilities, SHA-256 checked
    $H $PY scripts/run_g1.py tcn score gate    # G1's TCN member (28 LOMO folds), decode, the gate tables
    $PY scripts/run_g1.py cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()                             # before torch is imported anywhere

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402

T = P.T_MAIN
STAGES = ("seed", "tcn", "score", "gate", "cost")
PRE_REG = "115f7e2"
PLAN = "ff2e0b1"


def _root(a) -> Path:
    v = a.final_cache or os.environ.get("KATENA_FINAL_CACHE")
    if not v:
        sys.exit("--final-cache or KATENA_FINAL_CACHE is needed for this stage")
    return Path(v)


def _folds(dev: list, spec: str) -> list:
    if spec == "8":
        return S.FOLDS8(dev)
    if spec == "all":
        return list(dev)
    keys = set(spec.split(","))
    return [r for r in dev if r.key in keys]


# ---------------------------------------------------------------------------------------------
# report

HEAD = ("| comparison | folds | candidate | incumbent | diff | 95% paired CI | wins | adoption rule | worse |\n"
        "|---|---|---|---|---|---|---|---|---|")


def _row(label: str, g: dict) -> str:
    return (f"| {label} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {g['wins']}/{g['n']} ({100 * g['win_rate']:.0f}%) | "
            f"{'yes' if g['adopted'] else 'no'} | {'yes' if g['worse'] else 'no'} |")


def gate_report(dev: list) -> dict:
    res = G.results()
    get = F.getter(res, G.f_store())
    f8 = S.FOLDS8(dev)
    out = {"pre_registration": PRE_REG, "plan": PLAN, "folds": len(dev), "tau": F.TAU, "w": F.W_F,
           "gate": {}, "info": {}}
    print("\n## Gate: G1 vs F (F's stored scores, copied from work-final)\n" + HEAD)
    for tag, refs in (("28", dev), ("8", f8)):
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "G1", "F", c)
            out["gate" if tag == "28" else "info"][f"{tag}/{c}"] = g
            print(_row(f"G1 vs F, {c}, {tag} folds", g))
    g28t, g28s = out["gate"]["28/target"], out["gate"]["28/severe"]
    out["g1_passes"] = F.decision(g28t, g28s)
    print(f"\nDECISION (pre-registered, 28 folds): adoption rule at target = {g28t['adopted']}, worse at severe = "
          f"{g28s['worse']} -> {'G1 passes' if out['g1_passes'] else 'G1 fails'}")
    print("\n## Per dataset (G1 vs F)\n| cond | dataset | n | G1 | F | diff | wins |\n|---|---|---|---|---|---|---|")
    out["by_dataset"] = {}
    for c in F.GATE_CONDS:
        for ds in P.DATASETS:
            rr = [r for r in dev if r.dataset == ds]
            if not rr:
                continue
            a = [get("G1", c, r)["spell_f1"] for r in rr]
            b = [get("F", c, r)["spell_f1"] for r in rr]
            wins = int(sum(x > y for x, y in zip(a, b)))
            out["by_dataset"][f"{c}/{ds}"] = {"n": len(rr), "G1": float(np.mean(a)), "F": float(np.mean(b)),
                                              "wins": wins}
            print(f"| {c} | {ds} | {len(rr)} | {np.mean(a):.4f} | {np.mean(b):.4f} | {np.mean(a) - np.mean(b):+.4f} | "
                  f"{wins}/{len(rr)} |")
    print("\n## Information: the TCN member alone, G1T vs F's FT\n" + HEAD)
    for tag, refs in (("28", dev), ("8", f8)):
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "G1T", "FT", c)
            out["info"][f"G1T-FT/{tag}/{c}"] = g
            print(_row(f"G1T vs FT, {c}, {tag} folds", g))
    print("\n## Information: D2's evaluation worlds (G1 vs F, G1T vs FT, same world)\n" + HEAD)
    for wld in F.W.EVAL_WORLDS:
        for cand, inc in (("G1", "F"), ("G1T", "FT")):
            g = F.compare(get, dev, cand, inc, wld)
            if g is not None:
                out["info"][f"world/{wld}/{cand}"] = g
                print(_row(f"{cand} vs {inc}, {wld}", g))
    print("\n## Information: macroF1, IDSSE + Metrica (8 folds)\n| cond | G1 | G1T | F | FG | FT |\n"
          "|---|---|---|---|---|---|")
    out["macro8"] = {}
    for c in F.GATE_CONDS:
        v = {m: float(np.mean([get(m, c, r)["macro_f1"] for r in f8])) for m in ("G1", "G1T", "F", "FG", "FT")}
        out["macro8"][c] = v
        print(f"| {c} | {v['G1']:.4f} | {v['G1T']:.4f} | {v['F']:.4f} | {v['FG']:.4f} | {v['FT']:.4f} |")
    p = P.WORK / "g1_gate.json"
    p.write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
                 encoding="utf-8", newline="\n")
    print("wrote", p)
    return out


def cost_report() -> None:
    print("| stage | wall (min) | CPU (min) | peak memory (GB) | note |\n|---|---|---|---|---|")
    for c in F.costs():
        note = {k: v for k, v in c.items() if k not in ("stage", "wall_s", "cpu_s", "peak_gb", "utc")}
        print(f"| {c['stage']} | {c['wall_s'] / 60:.1f} | {c['cpu_s'] / 60:.1f} | {c['peak_gb']:.2f} | {note or ''} |")


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in order)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--final-cache", dest="final_cache", default="")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    dev = F.dev_refs()
    print(f"[g1] stages {a.stages}; WORK={P.WORK}; development matches: {len(dev)}", flush=True)
    folds = _folds(dev, a.folds)
    for stage in a.stages:
        cost = F.Cost(f"g1-{stage}")
        info = {}
        if stage == "seed":
            info = G.seed_cache(dev, _root(a))
        elif stage == "tcn":
            info = G.stage_tcn(dev, folds)
        elif stage == "score":
            info = G.stage_score(G.results(), dev, folds, G.f_store())
            print(f"F re-decoded against F's stored scores: {info}", flush=True)
        elif stage == "gate":
            gate_report(dev)
        elif stage == "cost":
            cost_report()
            continue
        print(f"[g1] {stage}: {info}", flush=True)
        cost.done(**info)


if __name__ == "__main__":
    main()
