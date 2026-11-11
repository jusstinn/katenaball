#!/usr/bin/env python3
"""Two exploratory probes of the LightGBM member on the 8 IDSSE + Metrica development folds.

    uv run python scripts/probe_levers.py curve      # would more training matches help, and of which kind?
    uv run python scripts/probe_levers.py weights    # team head trained with possession-length weights
    uv run python scripts/probe_levers.py curve weights

External data only, LOMO, Viterbi at tau 2, priors and everything else as F. Exploratory, as
``screen_gbdt_trees.py`` was: nothing here is registered, nothing is adopted from it, and the 20 SkillCorner
folds are not scored, so they stay unseen for a registered gate. The unchanged model is not refitted: it is read
from the cache ``scripts/run_g12.py`` wrote (``g1_gbdt150``, ``g1_gbdt1200``, and ``g2_tcn4800t`` for the blend).

``curve`` refits the member with fewer training matches. For a held-out IDSSE or Metrica match the other 27 are
7 of its own kind (human or event labels, dead ball labelled) and 20 SkillCorner (labels from broadcast
tracking, no dead ball):

* ``7 + 10``  every second SkillCorner match dropped
* ``7 + 0``   no SkillCorner match
* ``4 + 20``  every second match of its own kind dropped

``weights`` changes one thing: each training row of the team head is weighted by the length L (seconds) of the
possession it sits in, in the T = 2 training labels, clipped to [1, 30] and scaled to mean 1. Both forms were
fixed before the first run:

* ``inv``   1 / L        every possession weighs about the same, as in spellF1
* ``sqrt``  1 / sqrt(L)  halfway
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("KATENA_WORK", str(REPO / "work" / "final"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()
from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.eval.bootstrap import paired_bootstrap  # noqa: E402
from katena.sim import worlds as W  # noqa: E402

T = P.T_MAIN
SIZES = (150, 1200)
CONDS = ("target", "severe")
VG, VT = {150: "g1_gbdt150", 1200: "g1_gbdt1200"}, "g2_tcn4800t"
CURVE = ("7 + 10", "7 + 0", "4 + 20")
WEIGHTS = ("inv", "sqrt")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit(pools, m: np.ndarray, w_team: np.ndarray | None = None) -> tuple:
    """The two heads on rows ``m``, every column, 1,200 rounds; ``w_team`` weights the team head's rows."""
    X, o, y, dead_ok = pools.X[m], pools.o[m], pools.y[T][m], pools.dead_ok[m]
    tr, ar = np.flatnonzero((y == 0) | (y == 1)), np.flatnonzero(dead_ok & (y < 3))
    wt = None if w_team is None else w_team[m][tr] / w_team[m][tr].mean()
    team = lgb.train(gbdt.PARAMS, lgb.Dataset(X[tr], (y[tr] == o[tr]).astype(np.float32), weight=wt), max(SIZES))
    alive = lgb.train(gbdt.PARAMS, lgb.Dataset(X[ar], (y[ar] != 2).astype(np.float32)), max(SIZES))
    return team, alive


def heads_at(team, alive, X0, X1, n: int) -> tuple:
    pa = (0.5 * (alive.predict(X0, num_iteration=n) + alive.predict(X1, num_iteration=n))).astype(np.float32)
    ph = (0.5 * (team.predict(X0, num_iteration=n) + 1.0 - team.predict(X1, num_iteration=n))).astype(np.float32)
    return pa, ph


def row_spell_seconds(dev: list) -> np.ndarray:
    """For every pool row, the length in seconds of the constant-label run it sits in (T = 2 labels)."""
    out, agree = [], []
    for r in dev:
        p, g = d2.load_pool(r, F.MIX), C.load_gt(r, T)
        y = g["y"]
        cut = np.flatnonzero(np.diff(y) != 0) + 1
        a, b = np.concatenate(([0], cut)), np.concatenate((cut, [len(y)]))
        seg = np.repeat((b - a) / g["fps"], b - a)
        f = np.minimum(p["idx"].astype(np.int64) * g["step"], len(y) - 1)
        agree.append(float((y[f] == p[f"y_T{T}"]).mean()))
        out.append(seg[f])
    log(f"pool rows against the native labels: agreement {min(agree):.4f} to {max(agree):.4f} per match")
    assert min(agree) > 0.98, "pool rows do not line up with the native labels"
    return np.concatenate(out)


def run(probes: list, out: Path) -> dict:
    dev = F.dev_refs()
    folds = [r for r in dev if r.dataset in W.DEAD_LABEL_DATASETS]
    assert len(folds) == 8
    src, pools = F.PriorSource(dev), F.FinalPools(dev)
    res = pickle.load(open(out, "rb")) if out.exists() else {}
    wrow = {}
    if "weights" in probes:
        L = np.clip(row_spell_seconds(dev), 1.0, 30.0)
        wrow = {"inv": 1.0 / L, "sqrt": 1.0 / np.sqrt(L)}
    own = sorted(r.key for r in folds)
    sc = sorted(r.key for r in dev if r.is_sc)
    for r in folds:
        gt, pri = C.load_gt(r, T), B.priors_for(src, {r.key}, T)
        tcn = Lm.load_oof(VT, T, r)
        feats = {c: TG.features(r, c) for c in CONDS}

        def put(name: str, n: int, c: str, pa, ph) -> None:
            res[("G", name, n, c, r.key)] = F.decode_score(r, c, pa, ph, pri, gt)
            bp = E.blend_heads((pa, ph), (tcn[f"pa_{c}"], tcn[f"ph_{c}"]), F.W_F)
            res[("B", name, n, c, r.key)] = F.decode_score(r, c, *bp, pri, gt)

        for n in SIZES:                                             # the unchanged model, from the gate's cache
            g = Lm.load_oof(VG[n], T, r)
            for c in CONDS:
                put("all", n, c, g[f"pa_{c}"], g[f"ph_{c}"])
        others = [k for k in own if k != r.key]
        jobs = []
        if "curve" in probes:
            jobs += [("7 + 10", others + sc[::2], None), ("7 + 0", others, None), ("4 + 20", others[::2] + sc, None)]
        if "weights" in probes:
            jobs += [(k, others + sc, wrow[k]) for k in WEIGHTS]
        for name, keys, w in jobs:
            if ("G", name, max(SIZES), CONDS[-1], r.key) in res:
                continue
            t0 = time.time()
            team, alive = fit(pools, np.isin(pools.keys, keys), w)
            for c, (_, X0, X1) in feats.items():
                for n in SIZES:
                    put(name, n, c, *heads_at(team, alive, X0, X1, n))
            log(f"{r.key} {name}: {time.time() - t0:.0f}s")
            with open(out, "wb") as fh:
                pickle.dump(res, fh)
    return res


def report(res: dict, probes: list) -> None:
    keys = sorted({k[4] for k in res})
    names = (list(CURVE) if "curve" in probes else []) + (list(WEIGHTS) if "weights" in probes else [])
    names = [x for x in names if all(("G", x, max(SIZES), CONDS[-1], k) in res for k in keys)]
    for kind, title in (("G", "LightGBM member alone"), ("B", "blended with the 4,800-step touch TCN (w = 0.5)")):
        print(f"\n## {title}: variant minus the unchanged model, {len(keys)} folds\n")
        print("| variant | trees | cond | variant | unchanged | diff | 95% paired CI | folds won | predicted per true |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name in names:
            for n in SIZES:
                for c in CONDS:
                    x = [res[(kind, name, n, c, k)] for k in keys]
                    y = [res[(kind, "all", n, c, k)] for k in keys]
                    bs = paired_bootstrap(x, y)
                    wins = int(sum(p["spell_f1"] > q["spell_f1"] for p, q in zip(x, y)))
                    per = sum(p["n_pred"] for p in x) / sum(p["n_gt"] for p in x)
                    per0 = sum(q["n_pred"] for q in y) / sum(q["n_gt"] for q in y)
                    print(f"| {name} | {n} | {c} | {np.mean([p['spell_f1'] for p in x]):.4f} | {np.mean([q['spell_f1'] for q in y]):.4f} | "
                          f"{bs['mean_diff']:+.4f} | [{bs['lo']:+.4f}, {bs['hi']:+.4f}] | {wins}/{len(keys)} | {per:.3f} (from {per0:.3f}) |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probes", nargs="+", choices=("curve", "weights"))
    ap.add_argument("--out", type=Path, default=P.WORK / "probe_levers.pkl")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    res = pickle.load(open(a.out, "rb")) if a.report_only else run(a.probes, a.out)
    report(res, a.probes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
