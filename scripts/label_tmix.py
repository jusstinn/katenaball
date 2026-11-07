#!/usr/bin/env python
"""D4: train across label conventions and score against the ground truth of every convention.

    KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-d4 \
        uv run python scripts/label_tmix.py --folds 8 [--check] [--models MIX AUG ...]

Needs the bake-off cache in $KATENA_WORK (gt/, pool/, feat/, oof/full/T2/, results.pkl) and, for
the event-defined convention E, ``scripts/label_evidence.py`` first. Team heads are cached under
$KATENA_WORK/tmix/oof/<model>/ (resumable); scores go to $KATENA_WORK/tmix/scores_<n>.pkl. Prints
aggregate tables only. Development matches only: the locked benches are never loaded.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

import numpy as np  # noqa: E402

from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402
from katena.eval import tmix as TM  # noqa: E402
from katena.eval.bootstrap import adopted, paired_bootstrap  # noqa: E402

DEFAULT_MODELS = ["S0", "S1", "S2", "S3", "S5", "MIX", "AUG", "H3.5", "S2tail"]
CANDIDATES = ["MIX", "AUG", "H3.5", "S2tail"]


def check(pools, labels, folds, pool_y, draw) -> None:
    """Pipeline check: the refitted T = 2 team head must equal M2's cached one, and the S2 scores
    must equal results.pkl's M2 scores (28 folds only: tau depends on the fold set)."""
    r = folds[0]
    TM.MODELS["S2r"] = ("single", ("S2",))
    TM.fit_fold(pools, labels, r, ["S0", "S2r"], pool_y, draw)      # S2r reuses S0's binned Dataset
    for c in TM.CONDS:
        a, b = TM.load_ph("S2r", r, c), TM.load_ph("S2", r, c)
        print(f"check {r.key} {c}: max |p_home refit - cached| = {float(np.abs(a - b).max()):.3g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, choices=(8, 28), default=8)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--stage", choices=("fit", "score", "all"), default="all")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    dev, locked = P.split(P.all_refs())
    lk = {r.key for r in locked}
    folds = S.FOLDS8(dev) if a.folds == 8 else dev
    assert not lk & {r.key for r in dev + folds}
    print(f"locked benches excluded: {sorted(lk)} | folds: {len(folds)} | training pool: {len(dev)} dev matches",
          flush=True)
    labels = TM.Labels(dev)
    if a.stage in ("fit", "all") or a.check:
        pools = Lm.Pools(dev)
        need = sorted({n for m in a.models + ["S0", "S2"] for n in TM.MODELS[m][1]})
        pool_y = {n: TM.pool_labels(pools, labels, n) for n in need}
        draw = TM.mix_draw(len(pools.o))
        team = pool_y["S2"] < 2
        share = np.mean([pool_y[n][team] == pools.o[team] for n in TM.SWEEP if n in pool_y], axis=0)
        print(f"pool rows {len(pools.o)}; team rows {int(team.sum())}; rows whose team differs across "
              f"T: {float(((share > 0) & (share < 1)).mean()):.1%}", flush=True)
        if a.check:
            check(pools, labels, folds, pool_y, draw)
        if a.stage in ("fit", "all"):
            for r in folds:
                TM.fit_fold(pools, labels, r, a.models, pool_y, draw,
                            log=lambda s: print(s, flush=True))
        del pools
    if a.stage == "fit":
        return 0

    scores = {}
    for mname in a.models:
        for k, v in TM.decode_scores(labels, folds, mname, log=lambda s: print(s, flush=True)).items():
            scores[(mname,) + k] = v
    out = TM.TMIX_DIR / f"scores_{len(folds)}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(scores, open(out, "wb"))
    if a.check and len(folds) == 28 and Lm.RES_PATH.exists():
        res = Lm.Results()
        d = [abs(scores[("S2", "target", "S2", r.key)]["spell_f1"] - res.get(("M2", 2, "target", r.key))["spell_f1"])
             for r in folds]
        print(f"check: S2 re-score vs results.pkl M2, max |diff| over {len(d)} folds = {max(d):.3g}")
    report(scores, folds, [m for m in a.models if m in TM.MODELS])
    print(f"done in {time.time() - t0:.0f}s")
    return 0


def report(scores: dict, folds: list, models: list) -> None:
    evals_all = TM.EVAL_MAIN + TM.EVAL_EXTRA
    for cond in TM.CONDS:
        print(f"\n## {cond}: mean spellF1 over folds (rows = training scheme, columns = evaluation convention)\n")
        print("| train \\ eval | " + " | ".join(evals_all) + " |")
        print("|---|" + "---|" * len(evals_all))
        means = {}
        for m in models:
            cells = []
            for e in evals_all:
                v = [scores[(m, cond, e, r.key)]["spell_f1"] for r in folds if (m, cond, e, r.key) in scores]
                means[(m, e)] = float(np.mean(v)) if v else float("nan")
                cells.append(f"{means[(m, e)]:.3f}" + ("" if len(v) == len(folds) else f" (n={len(v)})"))
            print(f"| {m} | " + " | ".join(cells) + " |")
        for label, evals in (("S0-S5", TM.EVAL_MAIN), ("S0-S5 + S2tail + H3.5", TM.EVAL_MAIN + ("S2tail", "H3.5"))):
            rg = TM.regret(means, models, evals)
            print(f"\nworst-case regret over {label} ({cond}): " +
                  ", ".join(f"{m} {rg[m][0]:.3f} (at {rg[m][1]})" for m in sorted(models, key=lambda x: rg[x][0])))
        if "S2" not in models:
            continue
        print(f"\n### {cond}: candidates vs M2 (= S2), paired bootstrap over possessions, 2000 draws\n")
        print("| candidate | eval | n | mean diff | 95% CI | wins | adopted |")
        print("|---|---|---|---|---|---|---|")
        for m in [x for x in models if x != "S2"]:
            for e in evals_all:
                ks = [r.key for r in folds if (m, cond, e, r.key) in scores and ("S2", cond, e, r.key) in scores]
                if not ks:
                    continue
                res = paired_bootstrap([scores[(m, cond, e, k)] for k in ks], [scores[("S2", cond, e, k)] for k in ks])
                print(f"| {m} | {e} | {res['n']} | {res['mean_diff']:+.4f} | [{res['lo']:+.4f}, {res['hi']:+.4f}] | "
                      f"{res['win_rate']:.0%} | {'yes' if adopted(res) else 'no'} |")
        by_ds = {}
        for r in folds:
            by_ds.setdefault(r.dataset, []).append(r.key)
        print(f"\n### {cond}: mean spellF1 by dataset at eval S2 and S5\n")
        print("| train | " + " | ".join(f"{ds} S2 | {ds} S5" for ds in by_ds) + " |")
        print("|---|" + "---|---|" * len(by_ds))
        for m in models:
            cells = []
            for ds, ks in by_ds.items():
                for e in ("S2", "S5"):
                    cells.append(f"{np.mean([scores[(m, cond, e, k)]['spell_f1'] for k in ks]):.3f}")
            print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.exit(main())
