#!/usr/bin/env python
"""D2: train M2 across degrader worlds and score it in every world (external development matches only).

    KATENA_EXTERNAL=... KATENA_WORK=... uv run python scripts/d2_worlds.py build [--workers 2]
    ... d2_worlds.py pools
    ... d2_worlds.py lomo  --folds 8      # 8 = IDSSE + Metrica development folds (screen), 28 = all
    ... d2_worlds.py score --folds 8      # prints the gate table and writes <KATENA_WORK>/d2/gate_<folds>.json
    ... d2_worlds.py m0                   # simulated official baseline per world vs the published scores

Never touches data/, evaluation/gt/ or the locked benches. See katena.eval.d2 for the design, and the D2
entry of docs/decision-log.md for the decision rule, which was registered before any gate result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()                      # may re-exec once (macOS without libomp)

import numpy as np  # noqa: E402

from katena.eval import d2  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.sim import worlds as W  # noqa: E402


def _fmt_gate(cmp: dict, verdict: dict) -> str:
    lines = ["| world | folds | cur | d2 | d2 - cur | 95% CI | wins | macroF1 cur / d2 | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for w, c in cmp.items():
        tag = "worse" if c["worse"] else ("better" if c["better"] else "-")
        if w not in W.EVAL_WORLDS:
            tag += " (secondary)"
        lines.append(f"| {w} | {c['n']} | {c['mean_b']:.4f} | {c['mean_a']:.4f} | {c['mean_diff']:+.4f} | "
                     f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['wins']}/{c['n']} | {c['macro_b']:.3f} / {c['macro_a']:.3f} | {tag} |")
    lines.append("")
    lines.append(f"KEEP d2 mix: {verdict['keep']}  (worse: {verdict['worse'] or 'none'}; "
                 f"better by the adoption rule: {verdict['better'] or 'none'}; missing: {verdict['missing'] or 'none'})")
    return "\n".join(lines)


def _fmt_m0(tab: dict) -> str:
    worlds = sorted({k[0] for k in tab}, key=lambda w: d2.M0_WORLDS.index(w) if w in d2.M0_WORLDS else 99)
    lines = []
    for ds in W.DEAD_LABEL_DATASETS:
        keys = sorted({k[2] for k in tab if k[2].startswith(ds + "/")})
        if not keys:
            continue
        lines += [f"\n{ds} development matches (n={len(keys)}): official baseline macroF1 / spellF1 per window, "
                  "and the realised ball-missing rates",
                  "| world | P(miss given dead) | P(miss given alive) | overall | " + " | ".join(d2.M0_WINDOWS) + " |",
                  "|---|---|---|---|" + "---|" * len(d2.M0_WINDOWS)]
        for w in worlds:
            rows = [tab[(w, name, k)] for name in d2.M0_WINDOWS for k in keys if (w, name, k) in tab]
            if not rows:
                continue
            first = [tab[(w, "smooth50", k)] for k in keys]
            cells = []
            for name in d2.M0_WINDOWS:
                v = [tab[(w, name, k)] for k in keys]
                cells.append(f"{np.mean([x['macro_f1'] for x in v]):.3f} / {np.mean([x['spell_f1'] for x in v]):.3f}")
            lines.append(f"| {w} | {np.mean([x['miss_dead'] for x in first]):.3f} | {np.mean([x['miss_alive'] for x in first]):.3f} | "
                         f"{np.mean([x['miss_all'] for x in first]):.3f} | " + " | ".join(cells) + " |")
        lines.append("| published (target, full match) | | | 0.407 | " + " | ".join(
            f"{d2.PUBLISHED_M0[n][0]:.3f} / {d2.PUBLISHED_M0[n][1]:.3f}" for n in d2.M0_WINDOWS) + " |")
    return "\n".join(lines)


STAGES = ("build", "pools", "lomo", "score", "m0", "alarm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", help="one or more of " + ", ".join(STAGES) + ", comma-separated, run in order")
    ap.add_argument("--folds", default="8", help="8 (IDSSE + Metrica), 28 (all development), or match keys")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variants", default="cur,d2")
    a = ap.parse_args()
    stages = a.stages.split(",")
    if not set(stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    for stage in stages:
        run_stage(stage, a)


def run_stage(stage: str, a) -> None:
    a.stage = stage
    t0 = time.time()
    dev = d2.dev_refs()
    print(f"[d2] stage {a.stage}; WORK={P.WORK}; development matches: {len(dev)}", flush=True)
    if a.stage == "build":
        d2.build_all(workers=a.workers)
    elif a.stage == "pools":
        for r in dev:
            d2.build_pool(r)
            print("pool", r.key, flush=True)
    elif a.stage == "lomo":
        d2.run_lomo(d2.folds(a.folds, dev), dev, variants=tuple(a.variants.split(",")))
    elif a.stage == "score":
        fr = d2.folds(a.folds, dev)
        res = d2.score_folds(fr, dev, variants=tuple(a.variants.split(",")))
        d2.save(res, f"scores_{a.folds}.pkl")
        cmp = d2.compare_worlds(res, fr)
        verdict = d2.keep_verdict(cmp)
        taus = {v: [res[("tau", v, r.key)] for r in fr] for v in a.variants.split(",")}
        print(_fmt_gate(cmp, verdict))
        print("tau per fold:", {v: dict(zip(*np.unique(t, return_counts=True))) for v, t in taus.items()})
        out = {"folds": [r.key for r in fr], "verdict": verdict, "taus": taus,
               "compare": {w: {k: v for k, v in c.items()} for w, c in cmp.items()}}
        p = d2.root() / f"gate_{a.folds}.json"
        p.write_text(json.dumps(out, indent=1, default=float))
        print("wrote", p)
    elif a.stage == "m0":
        tab = d2.m0_table(dev)
        d2.save(tab, "m0.pkl")
        print(_fmt_m0(tab))
    elif a.stage == "alarm":
        from katena.sim import alarm
        ref = d2.alarm_reference(dev)
        alarm.REF_PATH.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")
        keys = alarm.BALL_KEYS + ("objects_mean", "slot_starts_per_min")
        print("| world | " + " | ".join(keys) + " |")
        print("|---|" + "---|" * len(keys))
        for w, st in ref["worlds"].items():
            print(f"| {w} | " + " | ".join(f"{st[k][0]:.3f} ({st[k][1]:.3f})" for k in keys) + " |")
        print("wrote", alarm.REF_PATH)
    print(f"[d2] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
