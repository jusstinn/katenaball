#!/usr/bin/env python
"""Locked-bench sanity read of F: F trained once on the 28 development matches, read on the two locked benches,
beside M2 retrained the way ``stages.stage_bench`` trained it. Registered in docs/decision-log.md (2026-11-08,
"Locked-bench sanity read of F", commit 4aa8bf1) before any bench score of F was computed. Report only: it
changes no decision.

External matches only: never reads ``data/`` or ``evaluation/gt/``. The source caches are read only;
everything is written under KATENA_WORK:

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-bench
    PY=.venv/Scripts/python.exe; H="python D:/katena/tools/heavy.py run --name bench --"
    $H $PY scripts/run_bench_read.py seed worlds train score \\
        --final-cache D:/katena/work-final --submit-cache D:/katena/work-submit --base-cache D:/katena/work-base

    seed    development inputs from work-final (each compared with work-submit's copy), bench inputs from
            work-submit (ground truth and base conditions compared with work-base)
    worlds  target_cpl100_ids for the benches (d2.build_ref, final.build_touch); then every bench world F reads
            is re-corrupted, re-featurized and its touch columns recomputed, all compared with the copies
    train   F's two members and M2 on the 28 development matches; bench probabilities in every condition
    score   Viterbi at tau 2, priors from the 28, official scorer; bench_results.pkl, bench_read.json, tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402

T = F.T
STAGES = ("seed", "worlds", "train", "score")
EXTRA_WORLD = "target_cpl100_ids"        # evaluation only: not in work-submit, built here for the benches
M2_ONLY = ("native",)                    # M2 is also read at native, for the Table 8 check
VAR = {"FG": "bench_final_gbdt", "FT": "bench_final_tcn", "M2": "bench_m2"}
RES_NAME = "bench_results.pkl"
OUT_JSON = "bench_read.json"
# docs/bakeoff.md Table 8, row M2 (spellF1, macroF1), as printed (3 decimals)
TABLE8_M2 = {
    ("idsse/DFL-MAT-J03WQQ", "target"): (0.567, 0.885), ("idsse/DFL-MAT-J03WQQ", "mild"): (0.613, 0.904),
    ("idsse/DFL-MAT-J03WQQ", "severe"): (0.548, 0.863), ("idsse/DFL-MAT-J03WQQ", "native"): (0.655, 0.920),
    ("metrica/Sample_Game_2", "target"): (0.631, 0.884), ("metrica/Sample_Game_2", "mild"): (0.662, 0.886),
    ("metrica/Sample_Game_2", "severe"): (0.537, 0.845), ("metrica/Sample_Game_2", "native"): (0.678, 0.894),
}


def split() -> tuple[list, list]:
    dev, locked = P.split(P.all_refs())
    assert len(dev) == 28 and sorted(r.key for r in locked) == ["idsse/" + P.LOCKED_IDSSE, "metrica/" + P.LOCKED_METRICA]
    assert not ({r.key for r in dev} & {r.key for r in locked})
    return dev, locked


def _root(a, name: str) -> Path:
    v = getattr(a, name)
    if not v:
        sys.exit(f"--{name.replace('_', '-')} is needed for this stage")
    p = Path(v)
    if p.resolve() == P.WORK.resolve():
        sys.exit(f"--{name.replace('_', '-')} must not be the bench cache")
    return p


# ---------------------------------------------------------------------------------------------
# seed: copy with a hash, compare with an independent copy

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_sha(src: Path, dst: Path) -> str:
    """Copy ``src`` to ``dst`` atomically and return the sha256 of the bytes written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 22), b""):
            h.update(chunk)
            fo.write(chunk)
    tmp.replace(dst)
    return h.hexdigest()


def _rel(ref: P.Ref, kind: str, world: str = "") -> str:
    name = f"{ref.dataset}_{ref.id}"
    return {"gt": f"gt/{name}.npz", "pool": f"pool/{name}.npz", "pool_d2j": f"pool_d2j/{name}.npz",
            "touch_pool_d2j": f"touch_pool_d2j/{name}.npz", "feat": f"feat/{name}__{world}.npz",
            "touch": f"touch/{name}__{world}.npz"}[kind]


def seed_plan(dev: list, locked: list) -> list:
    """(relative path, source tag, compared-with tag): development from work-final against work-submit,
    benches from work-submit (the base files against work-base)."""
    out = []
    for r in dev:
        for k in ("gt", "pool", "pool_d2j", "touch_pool_d2j"):
            out.append((_rel(r, k), "final", "submit"))
        for w in F.train_worlds(r.dataset):
            out.append((_rel(r, "feat", w), "final", "submit"))
    for r in locked:
        out.append((_rel(r, "gt"), "submit", "base"))
        for w in M2_ONLY + C.POOL_CONDS:
            out.append((_rel(r, "feat", w), "submit", "base"))
        for w in F.train_worlds(r.dataset):          # every training world, so build_touch builds only EXTRA_WORLD
            if w not in C.POOL_CONDS:
                out.append((_rel(r, "feat", w), "submit", None))
            out.append((_rel(r, "touch", w), "submit", None))
    return out


def stage_seed(dev: list, locked: list, roots: dict) -> dict:
    st = {"files": 0, "gb": 0.0, "compared": 0, "identical": 0, "different": [], "uncompared": 0}
    for rel, src, other in seed_plan(dev, locked):
        dst = P.WORK / rel
        sha = _copy_sha(roots[src] / rel, dst)
        st["files"] += 1
        st["gb"] += dst.stat().st_size / 2 ** 30
        if other is None:
            st["uncompared"] += 1
            continue
        st["compared"] += 1
        if _sha(roots[other] / rel) == sha:
            st["identical"] += 1
        else:
            st["different"].append(rel)
    st["gb"] = round(st["gb"], 2)
    if st["different"]:
        raise RuntimeError(f"copied inputs differ from the other cache: {st['different']}")
    return st


# ---------------------------------------------------------------------------------------------
# worlds: build the evaluation-only world, then check every bench world F reads

def verify_bench(ref: P.Ref) -> dict:
    """Re-corrupt every world F reads on ``ref``, check the recovery (rows, M0, degrader statistics), and
    compare freshly computed base features and touch columns with the cached files, byte for byte."""
    from katena.featurize import touch as TF
    from katena.grid import downsample
    from katena.types import load_match
    t0 = time.time()
    st = {"key": ref.key, "worlds": 0, "recovered": 0, "features_identical": 0, "touch_identical": 0}
    m = load_match(ref.path)
    m5 = downsample(m)
    for w in F.eval_worlds(ref):
        mc, lifted = TG.corrupted(ref, m, m5, w)
        st["worlds"] += 1
        st["recovered"] += int(F.recovery_ok(ref, w, mc))
        st["features_identical"] += int(F.features_identical(ref, w, F.base_features(mc)))
        X, names, idx = TF.touch_features(mc, TG._ball("touch", m, m5, mc, lifted))
        t = TG.load_tfeat(ref, w)
        st["touch_identical"] += int(t["A0"].tobytes() == X[0].tobytes() and t["A1"].tobytes() == X[1].tobytes()
                                     and t["idx"].tobytes() == idx.tobytes() and list(t["names"]) == list(names))
    st["wall_s"] = round(time.time() - t0, 1)
    return st


def stage_worlds(locked: list) -> dict:
    out = {"built_features": {}, "touch": [], "verify": []}
    for r in locked:
        out["built_features"][r.key] = d2.build_ref(r, [EXTRA_WORLD])
        st = F.build_touch(r)
        out["touch"].append({k: st[k] for k in ("key", "worlds", "built", "recovered", "rebuilt")})
        print(f"touch {r.key}: {st['built']} built, {st['recovered']} recovered, rebuilt {st['rebuilt'] or 'none'}",
              flush=True)
        if st["rebuilt"]:
            raise RuntimeError(f"{r.key}: copied features failed the recovery check: {st['rebuilt']}")
    for r in locked:
        v = verify_bench(r)
        out["verify"].append(v)
        print(f"verify {v}", flush=True)
        n = v["worlds"]
        if not (v["recovered"] == v["features_identical"] == v["touch_identical"] == n):
            raise RuntimeError(f"{r.key}: a bench world does not reproduce: {v}")
    return out


# ---------------------------------------------------------------------------------------------
# train and predict

def m2_conds(ref: P.Ref) -> list:
    return list(M2_ONLY) + F.eval_worlds(ref)


def stage_train(dev: list, locked: list) -> dict:
    info = {}
    t0 = time.time()
    pools = F.FinalPools(dev)
    print(f"F pool: {pools.X.shape[0]} rows x {pools.X.shape[1]} columns", flush=True)
    heads = pools.fit(set(), T, pools.cols())
    del pools
    info["gbdt_sha"] = F.gbdt_hash(heads)
    for r in locked:
        arrays = F.predict_gbdt(heads, r, F.eval_worlds(r))
        arrays["model_sha"] = np.array(info["gbdt_sha"])
        d3._save(Lm.oof_path(VAR["FG"], T, r), arrays)
    info["gbdt_s"] = round(time.time() - t0, 1)
    Lm.timed("LightGBM member", t0)

    t0 = time.time()
    pools = F.FinalPools(dev, touch=False)
    seqs = F.tcn_seqs(dev, T, 0)
    info["tcn_sequences"] = len(seqs)
    model, sc = F.fit_tcn(pools, seqs, set(), 0)
    del seqs, pools
    info["tcn_sha"] = E.model_hash(model)
    for r in locked:
        arrays = d3.predict_tcn(model, sc, r, F.eval_worlds(r))
        arrays["model_sha"] = np.array(info["tcn_sha"])
        d3._save(Lm.oof_path(VAR["FT"], T, r), arrays)
    info["tcn_s"] = round(time.time() - t0, 1)
    Lm.timed("TCN member", t0)

    t0 = time.time()
    pools = Lm.Pools(dev)                           # stages.stage_bench: the base pool, all 160 columns
    cols = pools.cols(**B.VARIANTS["full"])
    print(f"M2 pool: {pools.X.shape[0]} rows x {len(cols)} columns", flush=True)
    heads = pools.fit(set(), T, cols)
    del pools
    info["m2_sha"] = F.gbdt_hash(heads)
    for r in locked:
        arrays = {}
        for c in m2_conds(r):
            f = C.load_feat(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = heads.predict(f["X0"], f["X1"])
        arrays["model_sha"] = np.array(info["m2_sha"])
        d3._save(Lm.oof_path(VAR["M2"], T, r), arrays)
    info["m2_s"] = round(time.time() - t0, 1)
    Lm.timed("M2", t0)
    return info


# ---------------------------------------------------------------------------------------------
# score and report

def _clean(s: dict) -> dict:
    return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in s.items() if k in ("spell_f1", "macro_f1", "n_pred", "n_gt")}


def stage_score(dev: list, locked: list) -> dict:
    res = Lm.Results(P.WORK / RES_NAME)
    pri = B.priors_for(F.PriorSource(dev), set(), T)
    for r in locked:
        gt = C.load_gt(r, T)
        og, ot, om = (Lm.load_oof(VAR[k], T, r) for k in ("FG", "FT", "M2"))
        for c in F.eval_worlds(r):
            for w in (F.W_F, 0.0, 1.0):
                pa, ph = E.blend_heads((og[f"pa_{c}"], og[f"ph_{c}"]), (ot[f"pa_{c}"], ot[f"ph_{c}"]), w)
                res.put((F.NAMES[w], T, c, r.key), F.decode_score(r, c, pa, ph, pri, gt))
        for c in m2_conds(r):
            res.put(("M2", T, c, r.key), F.decode_score(r, c, om[f"pa_{c}"], om[f"ph_{c}"], pri, gt))
    res.save()

    out = {"registration": "4aa8bf1", "T": T, "tau": F.TAU, "w": F.W_F, "scores": {}, "table8_check": []}
    for k, v in res.d.items():
        out["scores"][f"{k[0]}|{k[2]}|{k[3]}"] = _clean(v)
    print("\n## Table 8 check: M2 retrained here vs bakeoff.md Table 8 (3 decimals)\n"
          "| bench | condition | spellF1 here | Table 8 | macroF1 here | Table 8 | equal at 3 decimals |\n"
          "|---|---|---|---|---|---|---|")
    for (key, c), (s8, m8) in TABLE8_M2.items():
        s = res.get(("M2", T, c, key))
        ok = round(s["spell_f1"], 3) == s8 and round(s["macro_f1"], 3) == m8
        out["table8_check"].append({"bench": key, "cond": c, "spell_f1": float(s["spell_f1"]), "table8_spell": s8,
                                    "macro_f1": float(s["macro_f1"]), "table8_macro": m8, "equal": bool(ok)})
        print(f"| {key} | {c} | {s['spell_f1']:.4f} | {s8:.3f} | {s['macro_f1']:.4f} | {m8:.3f} | {'yes' if ok else 'NO'} |")
    print("\n## F vs M2 on the locked benches (spellF1 / macroF1)\n"
          "| bench | condition | F spellF1 | M2 spellF1 | diff | F macroF1 | M2 macroF1 | diff | FG spellF1 | FT spellF1 |"
          "\n|---|---|---|---|---|---|---|---|---|---|")
    out["rows"] = []
    for r in locked:
        for c in F.eval_worlds(r):
            f, m, fg, ft = (res.get((n, T, c, r.key)) for n in ("F", "M2", "FG", "FT"))
            row = {"bench": r.key, "cond": c, "F_spell": float(f["spell_f1"]), "M2_spell": float(m["spell_f1"]),
                   "F_macro": float(f["macro_f1"]), "M2_macro": float(m["macro_f1"]),
                   "FG_spell": float(fg["spell_f1"]), "FT_spell": float(ft["spell_f1"]),
                   "F_pred_per_true": f["n_pred"] / f["n_gt"], "M2_pred_per_true": m["n_pred"] / m["n_gt"]}
            out["rows"].append(row)
            print(f"| {r.key} | {c} | {row['F_spell']:.4f} | {row['M2_spell']:.4f} | {row['F_spell'] - row['M2_spell']:+.4f} | "
                  f"{row['F_macro']:.4f} | {row['M2_macro']:.4f} | {row['F_macro'] - row['M2_macro']:+.4f} | "
                  f"{row['FG_spell']:.4f} | {row['FT_spell']:.4f} |")
    gate = [x for x in out["rows"] if x["cond"] in F.GATE_CONDS]
    out["gross_failure"] = any(x["F_spell"] - x["M2_spell"] < -0.05 for x in gate)
    print(f"\ngross failure (F below M2 by more than 0.05 at target or severe on either bench): {out['gross_failure']}")
    return out


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in order)")
    ap.add_argument("--final-cache", dest="final_cache", default="")
    ap.add_argument("--submit-cache", dest="submit_cache", default="")
    ap.add_argument("--base-cache", dest="base_cache", default="")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    dev, locked = split()
    print(f"[bench] stages {a.stages}; WORK={P.WORK}; development {len(dev)}, benches {[r.key for r in locked]}",
          flush=True)
    summary_path = P.WORK / OUT_JSON
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for stage in a.stages:
        cost = F.Cost(f"bench_{stage}")
        if stage == "seed":
            roots = {"final": _root(a, "final_cache"), "submit": _root(a, "submit_cache"), "base": _root(a, "base_cache")}
            info = stage_seed(dev, locked, roots)
        elif stage == "worlds":
            info = stage_worlds(locked)
        elif stage == "train":
            info = stage_train(dev, locked)
        else:
            info = stage_score(dev, locked)
        rec = cost.done()
        summary[stage] = {"info": info, "cost": rec}
        print(f"[bench] {stage}: {json.dumps(info, default=str)[:2000]}", flush=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8", newline="\n")
    print("wrote", summary_path, flush=True)


if __name__ == "__main__":
    main()
