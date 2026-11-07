#!/usr/bin/env python
"""Final candidate F (D1 touch + D3 ensemble at w = 0.5 + D2 ``d2j`` mix) against M2, on the 28 external
development folds. Pre-registered in docs/decision-log.md before any score (commit 3921643).

External matches only: never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench.
The base caches are read only; everything is written under KATENA_WORK. Heavy stages go through the slot
runner:

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-final
    export KATENA_BASE_CACHE=D:/katena/work-base KATENA_D2_CACHE=D:/katena/work-d2
    PY=.venv/Scripts/python.exe; H="python D:/katena/tools/heavy.py run --name final --"
    $H $PY scripts/run_final.py seed touch pools --workers 2   # inputs, touch columns, d2j pools
    $H $PY scripts/run_final.py gbdt                           # LightGBM member, 28 LOMO folds
    $H $PY scripts/run_final.py tcn                            # TCN member, 28 LOMO folds
    $H $PY scripts/run_final.py score gate                     # F, FG, FT and M2 re-decoded; the gate tables
    $H $PY scripts/run_final.py det --out D:/katena/work-final/det/a.json   # twice, fresh processes
    $PY scripts/run_final.py det-compare D:/katena/work-final/det/a.json D:/katena/work-final/det/b.json
    $PY scripts/run_final.py cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()                             # before torch is imported anywhere

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import final as F  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402

T = P.T_MAIN
STAGES = ("seed", "touch", "pools", "gbdt", "tcn", "score", "gate", "det", "det-compare", "cost")


def _root(a, name: str, env: str) -> Path:
    v = getattr(a, name) or os.environ.get(env)
    if not v:
        sys.exit(f"--{name.replace('_', '-')} or {env} is needed for this stage")
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

def _row(label: str, g: dict) -> str:
    return (f"| {label} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {g['wins']}/{g['n']} ({100 * g['win_rate']:.0f}%) | "
            f"{'yes' if g['adopted'] else 'no'} | {'yes' if g['worse'] else 'no'} |")


HEAD = ("| comparison | folds | candidate | incumbent | diff | 95% paired CI | wins | adoption rule | worse |\n"
        "|---|---|---|---|---|---|---|---|---|")


def gate_report(dev: list, base_root: Path) -> dict:
    res = F.results()
    base = F.load_store(base_root / "results.pkl")
    get = F.getter(res, base)
    f8 = S.FOLDS8(dev)
    out = {"pre_registration": "3921643", "folds": len(dev), "tau": F.TAU, "w": F.W_F, "gate": {}, "info": {}}
    print("\n## Gate: F vs M2 (stored, work-base)\n" + HEAD)
    for tag, refs in (("28", dev), ("8", f8)):
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "F", "M2", c)
            out["gate"][f"{tag}/{c}"] = g
            print(_row(f"F vs M2, {c}, {tag} folds", g))
    g28t, g28s = out["gate"]["28/target"], out["gate"]["28/severe"]
    out["adopt_F"] = F.decision(g28t, g28s)
    print(f"\nDECISION (pre-registered, 28 folds): adopted at target = {g28t['adopted']}, worse at severe = "
          f"{g28s['worse']} -> {'ADOPT F' if out['adopt_F'] else 'keep M2'}")
    print("\n## Per dataset (F vs M2)\n| cond | dataset | n | F | M2 | diff | wins |\n|---|---|---|---|---|---|---|")
    out["by_dataset"] = {}
    for c in F.GATE_CONDS:
        for ds in P.DATASETS:
            rr = [r for r in dev if r.dataset == ds]
            a = [get("F", c, r)["spell_f1"] for r in rr]
            b = [get("M2", c, r)["spell_f1"] for r in rr]
            wins = int(sum(x > y for x, y in zip(a, b)))
            out["by_dataset"][f"{c}/{ds}"] = {"n": len(rr), "F": float(np.mean(a)), "M2": float(np.mean(b)), "wins": wins}
            print(f"| {c} | {ds} | {len(rr)} | {np.mean(a):.4f} | {np.mean(b):.4f} | {np.mean(a) - np.mean(b):+.4f} | "
                  f"{wins}/{len(rr)} |")
    print("\n## Information: the members alone vs M2\n" + HEAD)
    for cand in ("FG", "FT"):
        for tag, refs in (("28", dev), ("8", f8)):
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, cand, "M2", c)
                out["info"][f"{cand}/{tag}/{c}"] = g
                print(_row(f"{cand} vs M2, {c}, {tag} folds", g))
    print("\n## Information: D2's evaluation worlds, vs M2's heads in the same world (M2w)\n" + HEAD)
    for wld in F.W.EVAL_WORLDS:
        for cand in ("F", "FG", "FT"):
            g = F.compare(get, dev, cand, "M2w", wld)
            if g is not None:
                out["info"][f"world/{wld}/{cand}"] = g
                print(_row(f"{cand} vs M2w, {wld}", g))
    print("\n## Information: macroF1, IDSSE + Metrica (8 folds)\n| cond | F | FG | FT | M2 |\n|---|---|---|---|---|")
    out["macro8"] = {}
    for c in F.GATE_CONDS:
        v = {m: float(np.mean([get(m, c, r)["macro_f1"] for r in f8])) for m in ("F", "FG", "FT", "M2")}
        out["macro8"][c] = v
        print(f"| {c} | {v['F']:.4f} | {v['FG']:.4f} | {v['FT']:.4f} | {v['M2']:.4f} |")
    p = P.WORK / "final_gate.json"
    p.write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
                 encoding="utf-8", newline="\n")
    print("wrote", p)
    return out


# ---------------------------------------------------------------------------------------------
# determinism: one fold, both members trained afresh in this process

def det_run(a) -> None:
    import torch

    from katena.decode import hmm
    from katena.eval import bakeoff as B
    from katena.eval import corpus as C
    from katena.eval import d3
    from katena.eval import lomo as Lm
    from katena.eval.score import to_native_codes
    dev = F.dev_refs()
    ref = next(r for r in dev if r.key == a.fold)
    env = {k: os.environ.get(k) for k in sorted(E.CPU_ENV)}
    out = {"fold": ref.key, "pid": os.getpid(), "env": env, "torch": torch.__version__,
           "cpu_capability": torch.backends.cpu.get_cpu_capability(), "t": {}, "sha": {}, "lomo_equal": {}}
    conds = F.GATE_CONDS
    cost = F.Cost("det")
    t0 = time.time()
    pools = F.FinalPools(dev)
    heads = pools.fit({ref.key}, T, pools.cols())
    del pools
    pg = F.predict_gbdt(heads, ref, conds)
    out["sha"]["gbdt_model"] = F.gbdt_hash(heads)
    out["t"]["gbdt_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    pools = F.FinalPools(dev, touch=False)
    seqs = F.tcn_seqs(dev, T)
    model, sc = F.fit_tcn(pools, seqs, {ref.key})
    del seqs, pools
    pt = d3.predict_tcn(model, sc, ref, conds)
    out["sha"]["tcn_weights"] = E.model_hash(model)
    out["t"]["tcn_s"] = round(time.time() - t0, 1)
    pri = B.priors_for(F.PriorSource(dev), {ref.key}, T)
    gt = C.load_gt(ref, T)
    lg, lt = (Lm.load_oof(v, T, ref) if Lm.oof_path(v, T, ref).exists() else None for v in (F.VAR_G, F.VAR_T))
    for c in conds:
        f = F._meta(ref, c)
        pa, ph = E.blend_heads((pg[f"pa_{c}"], pg[f"ph_{c}"]), (pt[f"pa_{c}"], pt[f"ph_{c}"]), F.W_F)
        rows = hmm.decode(pa, ph, f["half"], pri, F.TAU, "viterbi")
        nat = to_native_codes(rows, f["idx"], gt["step"], gt["n_native"], gt["blocks"])
        out["sha"][f"gbdt_probs_{c}"] = E.sha256_arrays(pg[f"pa_{c}"], pg[f"ph_{c}"])
        out["sha"][f"tcn_probs_{c}"] = E.sha256_arrays(pt[f"pa_{c}"], pt[f"ph_{c}"])
        out["sha"][f"blend_probs_{c}"] = E.sha256_arrays(pa, ph)
        out["sha"][f"labels_{c}"] = E.sha256_arrays(rows)
        out["sha"][f"labels_native_{c}"] = E.sha256_arrays(nat)
        if lg is not None and lt is not None:
            out["lomo_equal"][f"gbdt_{c}"] = E.sha256_arrays(lg[f"pa_{c}"], lg[f"ph_{c}"]) == out["sha"][f"gbdt_probs_{c}"]
            out["lomo_equal"][f"tcn_{c}"] = E.sha256_arrays(lt[f"pa_{c}"], lt[f"ph_{c}"]) == out["sha"][f"tcn_probs_{c}"]
    if lg is not None and lt is not None:
        out["lomo_equal"]["gbdt_model"] = str(lg["model_sha"]) == out["sha"]["gbdt_model"]
        out["lomo_equal"]["tcn_weights"] = str(lt["model_sha"]) == out["sha"]["tcn_weights"]
    out["cost"] = cost.done(fold=ref.key)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1), flush=True)


def det_compare(files: list) -> None:
    js = [json.loads(Path(p).read_text(encoding="utf-8")) for p in files]
    for k in js[0]["sha"]:
        same = all(j["sha"].get(k) == js[0]["sha"][k] for j in js)
        print(f"{k}: {'identical' if same else 'DIFFERENT'} ({js[0]['sha'][k][:12]}...)")
    for p, j in zip(files, js):
        print(f"  {p}: pid {j['pid']}, LOMO files equal {j['lomo_equal']}")


def cost_report() -> None:
    print("| stage | wall (min) | CPU (min) | peak memory (GB) | note |\n|---|---|---|---|---|")
    for c in F.costs():
        note = {k: v for k, v in c.items() if k not in ("stage", "wall_s", "cpu_s", "peak_gb", "utc")}
        print(f"| {c['stage']} | {c['wall_s'] / 60:.1f} | {c['cpu_s'] / 60:.1f} | {c['peak_gb']:.2f} | {note or ''} |")


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in order; det-compare takes files)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-cache", dest="base_cache", default="")
    ap.add_argument("--d2-cache", dest="d2_cache", default="")
    ap.add_argument("--fold", default=F.DET_FOLD, help="det: the fold to regenerate")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.stages[0] == "det-compare":
        det_compare(a.stages[1:])
        return
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    dev = F.dev_refs()
    print(f"[final] stages {a.stages}; WORK={P.WORK}; development matches: {len(dev)}", flush=True)
    folds = _folds(dev, a.folds)
    for stage in a.stages:
        cost = F.Cost(stage)
        info = {}
        if stage == "seed":
            info = F.seed_cache(dev, _root(a, "base_cache", "KATENA_BASE_CACHE"), _root(a, "d2_cache", "KATENA_D2_CACHE"))
        elif stage == "touch":
            sts = F.build_touch_all(dev, workers=a.workers, cost=cost)
            info = {"worlds": sum(s["worlds"] for s in sts), "built": sum(s["built"] for s in sts),
                    "recovered": sum(s["recovered"] for s in sts), "rebuilt": sum(len(s["rebuilt"]) for s in sts),
                    "refeaturized": sum(s["refeaturized"] for s in sts),
                    "refeaturize_identical": sum(s["refeaturize_same"] for s in sts)}
        elif stage == "pools":
            info = F.build_pools(dev, _root(a, "d2_cache", "KATENA_D2_CACHE"))
        elif stage == "gbdt":
            pools = F.FinalPools(dev)
            print(f"joined pool: {pools.X.shape[0]} rows x {pools.X.shape[1]} columns "
                  f"({int((pools.groups == F.TF.GROUP).sum())} touch)", flush=True)
            F.stage_gbdt(pools, folds)
            del pools
            info = {"folds": len(folds)}
        elif stage == "tcn":
            pools = F.FinalPools(dev, touch=False)
            F.stage_tcn(pools, dev, folds)
            del pools
            info = {"folds": len(folds)}
        elif stage == "score":
            res = F.results()
            chk = F.stage_incumbent(res, dev, folds, _root(a, "base_cache", "KATENA_BASE_CACHE"),
                                    _root(a, "d2_cache", "KATENA_D2_CACHE"))
            print(f"incumbent checks: {chk}", flush=True)
            F.stage_score(res, dev, folds)
            info = chk
        elif stage == "gate":
            gate_report(dev, _root(a, "base_cache", "KATENA_BASE_CACHE"))
        elif stage == "det":
            det_run(a)
            continue
        elif stage == "cost":
            cost_report()
            continue
        print(f"[final] {stage}: {info}", flush=True)
        cost.done(**info)


if __name__ == "__main__":
    main()
