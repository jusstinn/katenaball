#!/usr/bin/env python
"""D4: train across label conventions and score against the ground truth of every convention.

    KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-d4 \
        uv run python scripts/label_tmix.py --folds 8 [--check] [--models MIX AUG ...]

Needs the bake-off cache in $KATENA_WORK (gt/, pool/, feat/, oof/full/T2/, results.pkl) and
``scripts/label_evidence.py --metrica`` first (event-defined labels E, Metrica restart labels R).
Team heads are cached under $KATENA_WORK/tmix/oof/<model>/ (resumable); scores go to
$KATENA_WORK/tmix/scores_<n>.pkl. tau is fixed at 2 unless --tau-select. Prints aggregate tables
only, for all folds run and for the 8 IDSSE + Metrica folds separately. Development matches only:
the locked benches are never loaded.
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

DEFAULT_MODELS = ["S0", "S1", "S2", "S3", "S5", "MIX", "AUG", "H3.5", "S2tail", "MR"]
T_SCHEMES = ["S0", "S1", "S2", "S3", "S5", "MIX", "AUG", "H3.5", "S2tail"]     # the regret set
SUBSETS = {"all folds": lambda r: True, "8 IDSSE + Metrica": lambda r: r.dataset in ("idsse", "metrica"),
           "IDSSE": lambda r: r.dataset == "idsse", "Metrica": lambda r: r.dataset == "metrica",
           "SkillCorner": lambda r: r.dataset == "skillcorner"}


def check(pools, labels, folds, pool_y, draw) -> None:
    """Pipeline checks: pooled labels rebuilt from the native ground truth equal the cache, and the
    refitted T = 2 heads (team head alone, and both heads) equal M2's cached ones."""
    r = folds[0]
    for name in ("S0", "S2", "S5"):
        same = np.array_equal(TM.pool_labels(pools, labels, name, generic=True), pools.y[int(name[1:])])
        print(f"check: pooled {name} labels rebuilt from the native ground truth equal the cache: {same}", flush=True)
        assert same
    TM.MODELS["S2r"] = ("single", ("S2",))
    TM.MODELS["S2b"] = ("both", ("S2",))
    TM.fit_fold(pools, labels, r, ["S0", "S2r", "S2b"], pool_y, draw)    # S2r reuses S0's binned Dataset
    for c in TM.CONDS:
        cached = Lm.load_oof("full", P.T_MAIN, r)
        d_team = float(np.abs(TM.load_ph("S2r", r, c) - cached[f"ph_{c}"]).max())
        d_both = max(float(np.abs(TM.load_ph("S2b", r, c) - cached[f"ph_{c}"]).max()),
                     float(np.abs(TM.load_pa("S2b", r, c) - cached[f"pa_{c}"]).max()))
        print(f"check {r.key} {c}: max |refit - cached| team head {d_team:.3g}, both heads {d_both:.3g}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, choices=(8, 28), default=8)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--stage", choices=("fit", "score", "all"), default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tau-select", action="store_true", help="choose tau per fold as the bake-off did")
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
        print(f"pool rows {len(pools.o)}; team rows {int(team.sum())}; team rows whose label differs across T: "
              f"{float(((share > 0) & (share < 1)).mean()):.1%}", flush=True)
        if "S2R" in pool_y:
            mt = np.isin(pools.keys, [r.key for r in dev if r.dataset == "metrica"])
            print(f"Metrica rows {int(mt.sum())}: restart-rule labels differ on {float((pool_y['S2R'][mt] != pool_y['S2'][mt]).mean()):.1%}; "
                  f"share of the alive head's rows {float(mt[pools.dead_ok & (pool_y['S2'] < 3)].mean()):.1%}", flush=True)
        if a.check:
            check(pools, labels, folds, pool_y, draw)
        if a.stage in ("fit", "all"):
            for r in folds:
                TM.fit_fold(pools, labels, r, a.models, pool_y, draw, log=lambda s: print(s, flush=True))
        del pools
    if a.stage == "fit":
        return 0

    tau = None if a.tau_select else TM.TAU
    scores = {}
    for mname in a.models:
        for k, v in TM.decode_scores(labels, folds, mname, tau=tau, log=lambda s: print(s, flush=True)).items():
            scores[(mname,) + k] = v
    out = TM.TMIX_DIR / f"scores_{len(folds)}{'_tausel' if a.tau_select else ''}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(scores, open(out, "wb"))
    if a.check and len(folds) == 28 and Lm.RES_PATH.exists() and "S2" in a.models:
        res = Lm.Results()
        d = [abs(scores[("S2", "target", "S2", r.key)]["spell_f1"] - res.get(("M2", 2, "target", r.key))["spell_f1"])
             for r in folds]
        print(f"check: S2 re-score vs results.pkl M2, max |diff| over {len(d)} folds = {max(d):.3g}")
    report(scores, folds, [m for m in a.models if m in TM.MODELS])
    print(f"done in {time.time() - t0:.0f}s")
    return 0


def _mean(scores, m, cond, e, keys):
    v = [scores[(m, cond, e, k)]["spell_f1"] for k in keys if (m, cond, e, k) in scores]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def report(scores: dict, folds: list, models: list) -> None:
    evals_all = TM.EVAL_MAIN + TM.EVAL_EXTRA
    schemes = [m for m in models if m in T_SCHEMES]
    subsets = {name: [r.key for r in folds if f(r)] for name, f in SUBSETS.items()}
    subsets = {k: v for k, v in subsets.items() if v and not (k != "all folds" and len(v) == len(folds))}
    for cond in TM.CONDS:
        for sname, keys in subsets.items():
            print(f"\n## {cond}, {sname} ({len(keys)}): mean spellF1 (rows = training, columns = evaluation)\n")
            print("| train \\ eval | " + " | ".join(evals_all) + " |")
            print("|---|" + "---|" * len(evals_all))
            means = {}
            for m in models:
                cells = []
                for e in evals_all:
                    means[(m, e)], n = _mean(scores, m, cond, e, keys)
                    cells.append("-" if not n else f"{means[(m, e)]:.3f}" + ("" if n == len(keys) else f" (n={n})"))
                print(f"| {m} | " + " | ".join(cells) + " |")
            for label, evals in (("S0-S5", TM.EVAL_MAIN), ("S0-S5, S2tail, H3.5", TM.EVAL_MAIN + ("S2tail", "H3.5"))):
                if len(schemes) > 1:
                    rg = TM.regret(means, schemes, evals)
                    print(f"\nworst-case regret over {label}: " +
                          ", ".join(f"{m} {rg[m][0]:.3f} ({rg[m][1]})" for m in sorted(schemes, key=lambda x: rg[x][0])))
            if "S2" not in models or cond != "target":
                continue
            print(f"\n### {cond}, {sname}: against M2 (= S2), paired bootstrap over possessions, 2000 draws\n")
            print("| candidate | eval | n | mean diff | 95% CI | wins | passes the rule |")
            print("|---|---|---|---|---|---|---|")
            for m in [x for x in models if x != "S2"]:
                for e in (("S2R", "S2") if m == "MR" else ("S2", "S0", "S5", "H3.5")):
                    ks = [k for k in keys if (m, cond, e, k) in scores and ("S2", cond, e, k) in scores]
                    if not ks:
                        continue
                    res = paired_bootstrap([scores[(m, cond, e, k)] for k in ks], [scores[("S2", cond, e, k)] for k in ks])
                    print(f"| {m} | {e} | {res['n']} | {res['mean_diff']:+.4f} | [{res['lo']:+.4f}, {res['hi']:+.4f}] | "
                          f"{res['win_rate']:.0%} | {'yes' if adopted(res) else 'no'} |")


if __name__ == "__main__":
    sys.exit(main())
