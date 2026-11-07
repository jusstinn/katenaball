"""Smoke test: do the D3 features carry possession signal? (Synthetic, so not evidence.)

    python scripts/feature_smoke.py [--matches 5] [--half-minutes 8]

Builds a few synthetic matches, degrades them like ATD (46 anonymous slots with dropouts,
duplicates and ghosts; ~41% bursty ball loss concentrated in crowded moments and passes), then runs
leave-one-match-out LightGBM on per-frame features with team-swap augmentation. Reports frame
accuracy and macro-F1 over H/A/D, overall and on ball-missing frames, for all features and for the
ball-free subset. Deterministic (fixed seeds, deterministic=True, fixed threads).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from katena.feats import ball_free, featurize, swap_teams  # noqa: E402
from katena.feats.stress import drop_ball_bursts, fragment_slots  # noqa: E402
from katena.match import A, H  # noqa: E402
from katena.synth import synthetic_match  # noqa: E402

PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 50, "feature_fraction": 1.0, "bagging_fraction": 1.0,
          "lambda_l2": 1.0, "deterministic": True, "force_col_wise": True, "num_threads": 4,
          "seed": 0, "verbose": -1}
ROUNDS = 300


def swap_labels(y):
    out = y.copy()
    out[y == H], out[y == A] = A, H
    return out


def scores(y, p, sub=None):
    if sub is not None:
        y, p = y[sub], p[sub]
    return float((y == p).mean()), float(f1_score(y, p, labels=[0, 1, 2], average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--half-minutes", type=float, default=8.0)
    ap.add_argument("--train-stride", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    data = []
    for s in range(args.matches):
        m = synthetic_match(seed=100 + s, half_minutes=args.half_minutes)
        m = drop_ball_bursts(fragment_slots(m, seed=s), seed=s)
        full = featurize(m)
        data.append((full, m.y[full.t_frame].astype(np.int64)))
    names = data[0][0].names
    sets = {"all": names, "ball-free": ball_free(names)}
    miss = np.mean([np.mean(f.col("ball_present") == 0) for f, _ in data])
    print(f"{args.matches} synthetic matches x {2 * args.half_minutes:g} min, "
          f"{len(names)} features ({len(sets['ball-free'])} ball-free), ball missing on "
          f"{miss:.1%} of graded frames; built in {time.time() - t0:.0f} s")

    res = {k: [] for k in sets}
    gains = {}
    for i in range(len(data)):
        train = [d for j, d in enumerate(data) if j != i]
        Xs, ys = [], []
        for f, y in train:
            sub = np.arange(0, f.T, args.train_stride)
            fs = f.take(sub)
            Xs += [fs.X, swap_teams(fs).X]
            ys += [y[sub], swap_labels(y[sub])]
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        fte, yte = data[i]
        missing = fte.col("ball_present") == 0
        for k, cols in sets.items():
            idx = [names.index(c) for c in cols]
            ds = lgb.Dataset(Xtr[:, idx], ytr, feature_name=cols, free_raw_data=True)
            booster = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
            pred = booster.predict(fte.X[:, idx]).argmax(axis=1)
            res[k].append(scores(yte, pred) + scores(yte, pred, missing))
            if i == 0:
                gains[k] = sorted(zip(booster.feature_importance("gain"), cols), reverse=True)

    print(f"\nleave-one-match-out over {len(data)} matches (mean +- sd), frame-level argmax:")
    print(f"{'features':<10} {'acc':>13} {'macroF1':>13} {'acc|noball':>13} {'mF1|noball':>13}")
    for k, r in res.items():
        r = np.array(r)
        cells = [f"{r[:, c].mean():.3f}+-{r[:, c].std():.3f}" for c in range(4)]
        print(f"{k:<10} " + " ".join(f"{c:>13}" for c in cells))
    for k, g in gains.items():
        tot = sum(v for v, _ in g)
        top = ", ".join(f"{n} {v / tot:.0%}" for v, n in g[:6])
        print(f"top gain ({k}, fold 0): {top}")
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
