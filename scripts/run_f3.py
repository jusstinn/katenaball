#!/usr/bin/env python
"""F3 (F with the 1,200-tree LightGBM member and the 4,800-step touch TCN) against F on the 28 external development
folds, on the Windows PC: the second-machine gate of the pair that passed on the Mac (``scripts/run_g12.py``).
Pre-registered in docs/decision-log.md (2026-11-11, commit 005cd6c) before any Windows score of either member.

External matches only: never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench.
The F cache is read only; everything is written under KATENA_WORK. Fits go through the slot runner:

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-f3 KATENA_FINAL_CACHE=D:/katena/work-final
    PY=.venv/Scripts/python.exe; H="python D:/katena/tools/heavy.py run --name f3 --"
    $PY scripts/run_f3.py seed                  # copy F's inputs, probabilities and scores, SHA-256 checked
    $PY scripts/run_f3.py equiv                 # the TCN code here equals run_g12.py's LT code (20 steps, no score)
    $H $PY scripts/run_f3.py gbdt tcn score gate
    $PY scripts/run_f3.py cost
"""

from __future__ import annotations

import argparse
import importlib.util
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

from katena.eval import f3 as F3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G1  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402

T = P.T_MAIN
STAGES = ("seed", "equiv", "gbdt", "tcn", "score", "gate", "cost")
PRE_REG = "005cd6c"
# The Mac's means on the same folds (decision log, 2026-11-10, "Justin's branch: G1 and G2: result").
MAC = {"target": {"F": 0.6019, "F3": 0.6180}, "severe": {"F": 0.5696, "F3": 0.5840}}
HEAD = ("| comparison | folds | candidate | incumbent | diff | 95% paired CI | wins | adoption rule | worse |\n"
        "|---|---|---|---|---|---|---|---|---|")


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


def _row(label: str, g: dict | None) -> str:
    if g is None:
        return f"| {label} | 0 | | | | | | | |"
    return (f"| {label} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {g['wins']}/{g['n']} ({100 * g['win_rate']:.0f}%) | "
            f"{'yes' if g['adopted'] else 'no'} | {'yes' if g['worse'] else 'no'} |")


# ---------------------------------------------------------------------------------------------
# equivalence of the two implementations of LT (no score)

def equiv(dev: list, key: str, steps: int) -> dict:
    """``katena.eval.g1``'s TCN code (used here and in the pipeline) against ``scripts/run_g12.py``'s ``LT`` code,
    which the Mac's gate ran: the same sequences, scaler, trained weights and probabilities. A handful of matches and
    ``steps`` training steps; nothing is scored."""
    spec = importlib.util.spec_from_file_location("run_g12", REPO / "scripts" / "run_g12.py")
    G12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G12)
    held = next(r for r in dev if r.key == key)
    others = [r for r in dev if r.key != key]
    sub = [held] + [r for r in others if not r.is_sc][:2] + [r for r in others if r.is_sc][:2]
    pools = F.FinalPools(sub)
    a_seqs = F.tcn_seqs(sub, T, 0)
    lost = G12.add_touch(sub, a_seqs)
    b_seqs = G1.tcn_seqs(sub, T, 0)
    seq_same = (list(a_seqs) == list(b_seqs) and all(
        a_seqs[k]["X"].tobytes() == b_seqs[k]["X"].tobytes() and np.array_equal(a_seqs[k]["y"], b_seqs[k]["y"])
        and a_seqs[k]["o"] == b_seqs[k]["o"] for k in a_seqs))
    ma, sa = G12.fit_tcn_touch(pools, a_seqs, key, steps)
    mb, sb = G1.fit_tcn(G1.scaler(pools, {key}), b_seqs, {key}, 0, E.TCN_THREADS, steps)
    pa = G12.predict_tcn_touch(ma, sa, held, ["target"])
    pb = G1.predict_tcn(mb, sb, held, ["target"])
    out = {"matches": len(sub), "sequences": len(a_seqs), "columns": int(pools.X.shape[1]), "steps": steps,
           "float16_lost": int(lost), "sequences_identical": bool(seq_same),
           "scaler_identical": bool(sa.mu.tobytes() == sb.mu.tobytes() and sa.sd.tobytes() == sb.sd.tobytes()),
           "weights_identical": E.model_hash(ma) == E.model_hash(mb),
           "probabilities_identical": all(pa[k].tobytes() == pb[k].tobytes() for k in pa)}
    out["equivalent"] = all(out[k] for k in ("sequences_identical", "scaler_identical", "weights_identical",
                                             "probabilities_identical"))
    return out


# ---------------------------------------------------------------------------------------------
# report

def gate_report(dev: list) -> dict:
    res = F3.results()
    get = F.getter(res, F3.f_store())
    done = [r for r in dev if get("F3", "target", r) is not None]
    sets = (("all", done), ("SkillCorner", [r for r in done if r.is_sc]),
            ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    registered = len(done) == len(dev) == 28
    out = {"pre_registration": PRE_REG, "registered_run": registered, "folds": len(done), "tau": F.TAU, "w": F.W_F,
           "trees": F3.N_TREES, "steps": F3.STEPS, "gate": {}, "info": {}}
    print(f"\n## Gate: F3 vs F (F's stored Windows scores), {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "F3", "F", c)
            if g is not None:
                out["gate"][f"{tag}/{c}"] = g
                print(_row(f"F3 vs F, {c}, {tag}", g))
    gt_, gs_, gsc = (out["gate"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["f3_passes"] = bool(gt_ and gs_ and F3.passes(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nDECISION ({'registered run' if registered else 'NOT the registered run: partial'}): adoption rule at "
              f"target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on SkillCorner at target = "
              f"{gsc['worse'] if gsc else 'n/a'} -> {'F3 PASSES' if out['f3_passes'] else 'F3 fails'}")
    print("\n## Information: one change at a time and the members alone\n" + HEAD)
    for cand, inc in (("T1200", "F"), ("LT", "F"), ("F3", "T1200"), ("F3", "LT"), ("F3G", "FG"), ("F3T", "FT")):
        for tag, refs in sets:
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, cand, inc, c)
                if g is not None:
                    out["info"][f"{cand}-{inc}/{tag}/{c}"] = g
                    print(_row(f"{cand} vs {inc}, {c}, {tag}", g))
    print("\n## Information: every evaluation world (coupled worlds: IDSSE + Metrica only)\n" + HEAD)
    for wld in F.W.EVAL_WORLDS:
        g = F.compare(get, done, "F3", "F", wld)
        if g is not None:
            out["info"][f"world/{wld}"] = g
            print(_row(f"F3 vs F, {wld}", g))
    print("\n## Information: means, macroF1, predicted possessions per true one, and the Mac's means\n"
          "| cond | folds | F | F3 | Mac F | Mac F3 | macroF1 F | macroF1 F3 | per true F | per true F3 |\n"
          "|---|---|---|---|---|---|---|---|---|---|")
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            if not refs:
                continue
            mean = lambda name, key="spell_f1": float(np.mean([get(name, c, r)[key] for r in refs]))  # noqa: E731
            per = lambda name: (sum(get(name, c, r)["n_pred"] for r in refs)                           # noqa: E731
                                / max(sum(get(name, c, r)["n_gt"] for r in refs), 1))
            v = {"F": mean("F"), "F3": mean("F3"), "macro_F": mean("F", "macro_f1"), "macro_F3": mean("F3", "macro_f1"),
                 "per_true_F": per("F"), "per_true_F3": per("F3")}
            out["info"][f"means/{tag}/{c}"] = v
            mac = MAC[c] if tag == "all" else {"F": float("nan"), "F3": float("nan")}
            print(f"| {c} | {tag} ({len(refs)}) | {v['F']:.4f} | {v['F3']:.4f} | {mac['F']:.4f} | {mac['F3']:.4f} | "
                  f"{v['macro_F']:.4f} | {v['macro_F3']:.4f} | {v['per_true_F']:.3f} | {v['per_true_F3']:.3f} |")
    p = P.WORK / "f3_gate.json"
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
    ap.add_argument("--equiv-fold", default=F.DET_FOLD)
    ap.add_argument("--equiv-steps", type=int, default=20)
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    dev = F.dev_refs()
    print(f"[f3] stages {a.stages}; WORK={P.WORK}; development matches: {len(dev)}", flush=True)
    folds = _folds(dev, a.folds)
    for stage in a.stages:
        cost = F.Cost(f"f3-{stage}")
        info = {}
        if stage == "seed":
            info = G1.seed_cache(dev, _root(a))
        elif stage == "equiv":
            info = equiv(dev, a.equiv_fold, a.equiv_steps)
        elif stage == "gbdt":
            info = F3.stage_gbdt(F.FinalPools(dev), folds)
        elif stage == "tcn":
            info = F3.stage_tcn(dev, folds)
        elif stage == "score":
            info = F3.stage_score(F3.results(), dev, folds, F3.f_store())
            print(f"F re-decoded against F's stored scores: {info}", flush=True)
        elif stage == "gate":
            gate_report(dev)
        elif stage == "cost":
            cost_report()
            continue
        print(f"[f3] {stage}: {info}", flush=True)
        cost.done(**info)


if __name__ == "__main__":
    main()
