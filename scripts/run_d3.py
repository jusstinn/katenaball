#!/usr/bin/env python
"""D3: TCN + LightGBM ensemble through the gate, on external development matches only.

Needs the bake-off cache in KATENA_WORK (features, pools, ``oof/full``, ``results.pkl``). Never reads
``data/`` or ``evaluation/gt/`` and never runs the locked benches. Every candidate is decoded at the fixed
tau = 2 (``katena.eval.d3.TAU``). Heavy steps go through the slot runner:

    heavy.py run --name d3 -- .venv/Scripts/python.exe scripts/run_d3.py tcn --folds 8     # screen
    .venv/Scripts/python.exe scripts/run_d3.py score --folds 8
    .venv/Scripts/python.exe scripts/run_d3.py gate --folds 8
    heavy.py run --name d3 -- .venv/Scripts/python.exe scripts/run_d3.py tcn --folds all   # gate
    .venv/Scripts/python.exe scripts/run_d3.py score --folds all
    .venv/Scripts/python.exe scripts/run_d3.py gate --folds all      # 28 folds, and the 8 IDSSE+Metrica
    .venv/Scripts/python.exe scripts/run_d3.py curve --folds all     # w = 0.25 / 0.75, information only
    heavy.py run --name d3 -- .venv/Scripts/python.exe scripts/run_d3.py inner              # optional, D3N
    .venv/Scripts/python.exe scripts/run_d3.py nested
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()                             # before torch is imported anywhere

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402
from katena.eval.lomo import Pools, Results  # noqa: E402


def refs_for(dev: list, folds: str) -> list:
    return S.FOLDS8(dev) if folds == "8" else list(dev)


def show_gate(base: Results, res: Results, refs: list, cands: list, label: str) -> None:
    T = P.T_MAIN
    print(f"\n{label}: {len(refs)} folds, fixed tau {d3.TAU:g}")
    print("| candidate | cond | n | candidate mean | M2 mean | mean diff | 95% paired CI | wins | adopted |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cand in cands:
        for c, g in d3.gate(base, res, refs, cand).items():
            if not g["n"]:
                continue
            print(f"| {cand} | {c} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
                  f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {100 * g['win_rate']:.0f}% | {'yes' if g['adopted'] else 'no'} |")
    for cand in cands:                                              # per-dataset means, target-like
        parts = []
        for ds in P.DATASETS:
            rr = [r for r in refs if r.dataset == ds and res.get((cand, T, "target", r.key))]
            if rr:
                a = np.mean([res.get((cand, T, "target", r.key))["spell_f1"] for r in rr])
                b = np.mean([base.get(("M2", T, "target", r.key))["spell_f1"] for r in rr])
                parts.append(f"{ds} {a:.4f} vs {b:.4f} (n={len(rr)})")
        nd = [r for r in refs if not r.is_sc and res.get((cand, T, "target", r.key))]
        if nd:                                                       # tie-break metric, dead-labelled data only
            parts.append("macroF1 (IDSSE+Metrica) {:.4f} vs {:.4f}".format(
                np.mean([res.get((cand, T, "target", r.key))["macro_f1"] for r in nd]),
                np.mean([base.get(("M2", T, "target", r.key))["macro_f1"] for r in nd])))
        if parts:
            print(f"{cand} vs M2 (target): " + "; ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tcn", "score", "gate", "inner", "nested", "curve", "proto-m2"])
    ap.add_argument("--folds", default="8", choices=["8", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=E.TCN_THREADS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N folds (timing)")
    ap.add_argument("--cand", nargs="*", default=["D3E50", "D3T", "D3M2"])
    ap.add_argument("--steps", type=int, default=None, help="TCN steps (smoke tests only; default tcn.STEPS)")
    ap.add_argument("--datasets", nargs="*", default=[], help="restrict the development set (prototype caches)")
    a = ap.parse_args()
    dev, locked = B.dev_and_locked()
    assert not {r.key for r in locked} & {r.key for r in dev}
    if a.datasets:
        dev = [r for r in dev if r.dataset in a.datasets]
    print("development matches:", len(dev), "| work:", P.WORK, flush=True)
    refs = refs_for(dev, a.folds)
    if a.limit:
        refs = refs[:a.limit]
    T = P.T_MAIN
    store = d3.d3_results()
    if a.stage == "gate":
        base = Results()
        show_gate(base, store, refs, a.cand, "all requested folds" if a.folds == "all" else "screen")
        if a.folds == "all":                                         # D5 audit A18: dead-labelled folds apart
            show_gate(base, store, S.FOLDS8(refs), a.cand, "IDSSE + Metrica only")
        return
    pools = Pools(dev)
    if a.stage == "proto-m2":                    # a prototype cache has no M2 yet; the real one comes from D5
        base = Results()
        assert not any(k[0] == "M2" for k in base.d), "this cache already holds M2"
        B.stage_lomo(pools, dev, "full", T)
        B.decode_and_score(base, pools, dev, "full", T, "viterbi", "M2")
    elif a.stage == "tcn":
        d3.stage_tcn_lomo(pools, dev, refs, T, seed=a.seed, threads=a.threads, steps=a.steps,
                          variant=d3.tcn_variant(a.seed))
    elif a.stage == "score":
        d3.stage_score(pools, refs, T, res=store, seed=a.seed)
    elif a.stage == "curve":                                         # sensitivity only, never a choice
        d3.stage_score(pools, refs, T, weights=(0.25, 0.75), res=store)
    elif a.stage == "inner":
        d3.stage_inner(pools, dev, T, seed=a.seed, threads=a.threads, steps=a.steps)
    elif a.stage == "nested":
        print("nested choices:", d3.stage_nested(pools, dev, T, res=store))


if __name__ == "__main__":
    main()
