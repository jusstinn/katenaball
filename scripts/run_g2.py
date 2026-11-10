#!/usr/bin/env python
"""Gate G2 (F plus set-piece tag columns in the LightGBM member) against F, on the 28 external development folds.
Pre-registered in docs/decision-log.md (2026-11-10, "Gate G2: set-piece tag features (pre-registration)") before
any G2 score was computed. ``katena.eval.g2`` has the definition.

External matches only: never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench. F's cache
is read only; everything is written under KATENA_WORK. Heavy stages go through the slot runner:

    PY=.venv/Scripts/python.exe; H="python D:/katena/tools/heavy.py run --name g2 --"
    A="--work D:/katena/work-g2 --external D:/katena/external_data --f-cache D:/katena/work-final"
    $H $PY scripts/run_g2.py seed tags check $A    # copy F's inputs, imitation tags, F's LightGBM reproduced
    $H $PY scripts/run_g2.py gbdt score gate $A    # G2's LightGBM member on 28 folds, decode, the gate tables
    $PY scripts/run_g2.py cost $A

``--work`` and ``--external`` set KATENA_WORK and KATENA_EXTERNAL (before any katena import); ``--f-cache`` or
KATENA_F_CACHE is F's cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]
for _flag, _env in (("--work", "KATENA_WORK"), ("--external", "KATENA_EXTERNAL")):   # read at import by protocol
    if _flag in sys.argv[:-1]:
        os.environ[_env] = sys.argv[sys.argv.index(_flag) + 1]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import final as F  # noqa: E402
from katena.eval import g2 as G  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402

T = P.T_MAIN
STAGES = ("seed", "tags", "check", "gbdt", "score", "gate", "cost")
CHECK_FOLD = F.DET_FOLD                     # idsse/DFL-MAT-J03WMX, F's determinism fold


def _f_root(a) -> Path:
    v = a.f_cache or os.environ.get("KATENA_F_CACHE")
    if not v:
        sys.exit("--f-cache or KATENA_F_CACHE (F's cache, read only) is needed")
    return Path(v)


def _folds(dev: list, spec: str) -> list:
    if spec == "8":
        return S.FOLDS8(dev)
    if spec == "all":
        return list(dev)
    keys = set(spec.split(","))
    return [r for r in dev if r.key in keys]


HEAD = ("| comparison | folds | candidate | incumbent | diff | 95% paired CI | wins | adoption rule | worse |\n"
        "|---|---|---|---|---|---|---|---|---|")


def _row(label: str, g: dict) -> str:
    return (f"| {label} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {g['wins']}/{g['n']} ({100 * g['win_rate']:.0f}%) | "
            f"{'yes' if g['adopted'] else 'no'} | {'yes' if g['worse'] else 'no'} |")


def gate_report(dev: list, f_root: Path, pre_reg: str) -> dict:
    res = G.results()
    base = F.load_store(f_root / F.RES_NAME)
    get = F.getter(res, base)
    f8 = S.FOLDS8(dev)
    out = {"pre_registration": pre_reg, "folds": len(dev), "tau": F.TAU, "w": F.W_F, "gate": {}, "info": {}}
    print("\n## Gate: G2 vs F (F's stored scores, work-final)\n" + HEAD)
    for tag, refs in (("28", dev), ("8", f8)):
        for c in G.CONDS:
            g = F.compare(get, refs, "G2", "F", c)
            out["gate"][f"{tag}/{c}"] = g
            print(_row(f"G2 vs F, {c}, {tag} folds", g))
    g28t, g28s = out["gate"]["28/target"], out["gate"]["28/severe"]
    out["adopt_G2"] = F.decision(g28t, g28s)
    print(f"\nDECISION (pre-registered, 28 folds): adopted at target = {g28t['adopted']}, worse at severe = "
          f"{g28s['worse']} -> {'G2 PASSES' if out['adopt_G2'] else 'G2 FAILS, keep F'}")
    print("\n## Per dataset (G2 vs F)\n| cond | dataset | n | G2 | F | diff | wins |\n|---|---|---|---|---|---|---|")
    out["by_dataset"] = {}
    for c in G.CONDS:
        for ds in P.DATASETS:
            rr = [r for r in dev if r.dataset == ds]
            a = [get("G2", c, r)["spell_f1"] for r in rr]
            b = [get("F", c, r)["spell_f1"] for r in rr]
            wins = int(sum(x > y for x, y in zip(a, b)))
            ties = int(sum(x == y for x, y in zip(a, b)))
            out["by_dataset"][f"{c}/{ds}"] = {"n": len(rr), "G2": float(np.mean(a)), "F": float(np.mean(b)),
                                              "wins": wins, "ties": ties}
            print(f"| {c} | {ds} | {len(rr)} | {np.mean(a):.4f} | {np.mean(b):.4f} | {np.mean(a) - np.mean(b):+.4f} | "
                  f"{wins}/{len(rr)} |")
    print("\n## Information: the LightGBM members alone (G2G vs FG)\n" + HEAD)
    for tag, refs in (("28", dev), ("8", f8)):
        for c in G.CONDS:
            g = F.compare(get, refs, "G2G", "FG", c)
            out["info"][f"G2G_vs_FG/{tag}/{c}"] = g
            print(_row(f"G2G vs FG, {c}, {tag} folds", g))
    print("\n## Information: macroF1, IDSSE + Metrica (8 folds)\n| cond | G2 | F | G2G | FG |\n|---|---|---|---|---|")
    out["macro8"] = {}
    for c in G.CONDS:
        v = {m: float(np.mean([get(m, c, r)["macro_f1"] for r in f8])) for m in ("G2", "F", "G2G", "FG")}
        out["macro8"][c] = v
        print(f"| {c} | {v['G2']:.4f} | {v['F']:.4f} | {v['G2G']:.4f} | {v['FG']:.4f} |")
    print("\n## Information: predicted possessions per true one, 28 folds\n| cond | G2 | F |\n|---|---|---|")
    out["spells"] = {}
    for c in G.CONDS:
        v = {m: sum(get(m, c, r)["n_pred"] for r in dev) / max(sum(get(m, c, r)["n_gt"] for r in dev), 1)
             for m in ("G2", "F")}
        out["spells"][c] = v
        print(f"| {c} | {v['G2']:.3f} | {v['F']:.3f} |")
    p = P.WORK / G.GATE_NAME
    p.write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
                 encoding="utf-8", newline="\n")
    print("wrote", p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in order)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--f-cache", dest="f_cache", default="")
    ap.add_argument("--work", default="", help="KATENA_WORK, G2's own cache")
    ap.add_argument("--external", default="", help="KATENA_EXTERNAL, the external corpus")
    ap.add_argument("--pre-registration", dest="pre_reg", default="", help="commit of the pre-registration entry")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if (a.f_cache or os.environ.get("KATENA_F_CACHE")) and _f_root(a).resolve() == P.WORK.resolve():
        sys.exit("KATENA_WORK must be G2's own cache, never F's")
    dev = F.dev_refs()
    print(f"[g2] stages {a.stages}; WORK={P.WORK}; development matches: {len(dev)}; tag code {G.SRC}", flush=True)
    folds = _folds(dev, a.folds)
    for stage in a.stages:
        cost = F.Cost(f"g2_{stage}")
        info = {}
        if stage == "seed":
            info = G.seed_cache(dev, _f_root(a))
        elif stage == "tags":
            sts = [G.build_tags(r) for r in dev]
            for st in sts:
                print(f"tags {json.dumps(st, sort_keys=True)}", flush=True)
            info = {"matches": len(sts), "with_tags": sum(s["has_tags"] for s in sts),
                    "set_pieces": sum(s["set_pieces"] for s in sts)}
        elif stage == "check":
            pools = G.G2Pools(dev)
            print(f"G2 pool: {pools.X.shape[0]} rows x {pools.X.shape[1]} columns "
                  f"({int((pools.groups == G.TAGS.GROUP).sum())} tag)", flush=True)
            ref = next(r for r in dev if r.key == CHECK_FOLD)
            info = G.check_fg(pools, ref, _f_root(a))
            del pools
        elif stage == "gbdt":
            pools = G.G2Pools(dev)
            G.stage_gbdt(pools, folds)
            del pools
            info = {"folds": len(folds)}
        elif stage == "score":
            f_root = _f_root(a)
            info = G.stage_score(G.results(), dev, folds, f_root, F.load_store(f_root / F.RES_NAME))
        elif stage == "gate":
            gate_report(dev, _f_root(a), a.pre_reg)
        elif stage == "cost":
            for c in F.costs():
                print(c)
            continue
        print(f"[g2] {stage}: {info}", flush=True)
        cost.done(**info)


if __name__ == "__main__":
    main()
