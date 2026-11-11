#!/usr/bin/env python3
"""Screen the LightGBM member's size and training mix on the 8 IDSSE + Metrica development folds.

    uv run python scripts/screen_gbdt_trees.py                       # KATENA_WORK=work/final (needs `make model`)
    uv run python scripts/screen_gbdt_trees.py --trees 1200,2000,3000 --mixes d2j --worlds target,severe

External data only. Exploratory: it informs what goes to a pre-registered 28-fold gate and decides nothing
(docs/improvement-plan.md, docs/decision-log.md 2026-11-09). LOMO on the 27 other development matches; the
benches are never trained on or scored. One model of max(--trees) trees is fitted per fold and mix and read
at each smaller size (`num_iteration`), which equals fitting that size: LightGBM's rounds are sequential.

Mixes: ``d2j`` is F's pool. ``smooth`` uses the same rows, orientations and labels with every severity's
rows taken from its ``_smooth`` world alone. Where the cached TCN probabilities of ``screen_tcn_seeds.py``
exist (``target``, ``severe``), the F-style blend (w = 0.5, Viterbi at tau 2) is reported as well.
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
VT = "screen_tcn_s0"                                    # cached by scripts/screen_tcn_seeds.py
BLEND_CONDS = ("target", "severe")


class MixPools(F.FinalPools):
    """F's pool with every severity's rows taken from one world of its own (``{severity}{suffix}``)."""

    def __init__(self, like: F.FinalPools, suffix: str):
        mix = {c: [c + suffix] for c in C.POOL_CONDS}
        parts = []
        for r in like.refs:
            base = C.load_pool(r)
            parts.append(np.concatenate([d2.assemble_pool(r, mix, base)["X"],
                                         d2.assemble_pool(r, mix, base, load=F._touch_as_feat)["X"]], 1))
        self.X = np.concatenate(parts)
        for k in ("refs", "n_base", "keys", "o", "y", "dead_ok", "names", "groups"):
            setattr(self, k, getattr(like, k))
        assert self.X.shape == like.X.shape


def fit(pool, exclude: set, n_trees: int) -> tuple:
    """``gbdt.fit_heads`` with every column and ``n_trees`` rounds; returns the two boosters."""
    m = ~np.isin(pool.keys, list(exclude))
    X, o, y, dead_ok = pool.X[m], pool.o[m], pool.y[T][m], pool.dead_ok[m]
    tr, ar = np.flatnonzero((y == 0) | (y == 1)), np.flatnonzero(dead_ok & (y < 3))
    team = lgb.train(gbdt.PARAMS, lgb.Dataset(X[tr], (y[tr] == o[tr]).astype(np.float32)), n_trees)
    alive = lgb.train(gbdt.PARAMS, lgb.Dataset(X[ar], (y[ar] != 2).astype(np.float32)), n_trees)
    return team, alive


def run(args) -> dict:
    dev = F.dev_refs()
    folds = [r for r in dev if r.dataset in W.DEAD_LABEL_DATASETS]
    assert len(folds) == 8
    src = F.PriorSource(dev)
    pools = {"d2j": F.FinalPools(dev)}
    if "smooth" in args.mixes:
        pools["smooth"] = MixPools(pools["d2j"], "_smooth")
    res = {}
    for r in folds:
        gt, pri = C.load_gt(r, T), B.priors_for(src, {r.key}, T)
        feats = {w: TG.features(r, w) for w in args.worlds}
        tcn = Lm.load_oof(VT, T, r) if Lm.oof_path(VT, T, r).exists() else None
        for mix in args.mixes:
            t0 = time.time()
            team, alive = fit(pools[mix], {r.key}, max(args.trees))
            for w, (f, X0, X1) in feats.items():
                for n in args.trees:
                    pa = (0.5 * (alive.predict(X0, num_iteration=n) + alive.predict(X1, num_iteration=n))).astype(np.float32)
                    ph = (0.5 * (team.predict(X0, num_iteration=n) + 1.0 - team.predict(X1, num_iteration=n))).astype(np.float32)
                    res[("G", mix, n, w, r.key)] = F.decode_score(r, w, pa, ph, pri, gt)
                    if tcn is not None and w in BLEND_CONDS:
                        bp = E.blend_heads((pa, ph), (tcn[f"pa_{w}"], tcn[f"ph_{w}"]), F.W_F)
                        res[("F", mix, n, w, r.key)] = F.decode_score(r, w, *bp, pri, gt)
            print(f"[{time.strftime('%H:%M:%S')}] {r.key} {mix}: {time.time() - t0:.0f}s", flush=True)
        with open(args.out, "wb") as fh:
            pickle.dump(res, fh)
    return res


def report(res: dict, args) -> None:
    keys = sorted({k[4] for k in res})
    mean = lambda kind, mix, n, w: float(np.mean([res[(kind, mix, n, w, k)]["spell_f1"] for k in keys]))  # noqa: E731
    for kind, worlds in (("G", args.worlds), ("F", [w for w in args.worlds if ("F", args.mixes[0], args.trees[0], w, keys[0]) in res])):
        if not worlds:
            continue
        print(f"\n{'LightGBM member alone' if kind == 'G' else 'F-style blend with the cached TCN (seed 0)'}: mean spellF1, {len(keys)} folds")
        print(f"{'mix':7s} {'trees':>5s}  " + "  ".join(f"{w:>14s}" for w in worlds))
        for mix in args.mixes:
            for n in args.trees:
                print(f"{mix:7s} {n:5d}  " + "  ".join(f"{mean(kind, mix, n, w):14.4f}" for w in worlds))
        print("paired difference [95% CI over possessions], folds won:")

        def cmp(a, b, label):
            for w in worlds:
                x, y = [res[(kind, *a, w, k)] for k in keys], [res[(kind, *b, w, k)] for k in keys]
                bs = paired_bootstrap(x, y)
                wins = int(sum(p["spell_f1"] > q["spell_f1"] for p, q in zip(x, y)))
                print(f"  {label:34s} {w:14s} {bs['mean_diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}]  {wins}/{len(keys)}")
        for mix in args.mixes:
            for n in args.trees[1:]:
                cmp((mix, n), (mix, args.trees[0]), f"{mix}: {n} trees - {args.trees[0]} trees")
        if kind == "G" and "smooth" in args.mixes and "d2j" in args.mixes:
            for n in args.trees:
                cmp(("smooth", n), ("d2j", n), f"smooth - d2j, {n} trees")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trees", default="150,300,600,1200")
    ap.add_argument("--mixes", default="d2j,smooth")
    ap.add_argument("--worlds", default="target,severe,mild,target_cpl100,target_smooth,severe_smooth,mild_smooth")
    ap.add_argument("--out", default=None, help="results pickle (default: $KATENA_WORK/screen_gbdt_trees.pkl)")
    ap.add_argument("--report", action="store_true", help="print the tables from an existing results pickle")
    args = ap.parse_args()
    args.trees = [int(x) for x in args.trees.split(",")]
    args.mixes, args.worlds = args.mixes.split(","), args.worlds.split(",")
    args.out = Path(args.out) if args.out else P.WORK / "screen_gbdt_trees.pkl"
    if args.report:
        with open(args.out, "rb") as fh:
            res = pickle.load(fh)
    else:
        res = run(args)
    report(res, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
