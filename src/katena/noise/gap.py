"""Domain-gap check: can a classifier tell real windows from simulated ones?

Each match is cut into fixed windows inside its graded blocks. Each window gets label-free summary
features (ball presence and gaps, present-object counts, fragment churn, jumps, jitter, speed,
duplicates, distances to the ball, team shape). A classifier is cross-validated on real-vs-sim with
folds grouped by time block. AUC near 0.5 means the simulator is hard to tell apart from the real
input; the top features say what gives it away. Only aggregates are returned.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..match import Match
from ._util import ball_speed, blocks_of, pair_counts

FEATURE_NAMES = [
    "ball_present", "ball_gaps", "ball_longest_gap_frac", "ball_speed_med", "ball_speed_q90",
    "ball_d2", "ball_near", "ball_nearest", "n_present", "n_present_std", "n_present_min",
    "n_present_max", "team_count_diff", "slot_onsets", "slot_offsets", "slot_jumps", "player_d2",
    "speed_mean", "speed_nan_frac", "speed_fast_frac", "dup_pairs", "opp_close_pairs", "x_spread",
    "y_spread", "team_centroid_dist", "oob",
]


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)


def _frame_series(m: Match, stride: int, ball_radius_m: float, dup_dist_m: float, jump_m: float):
    fps = int(m.fps)
    P = np.asarray(m.P, dtype=np.float32)
    ball = np.asarray(m.ball, dtype=np.float32)
    speed = np.asarray(m.speed, dtype=np.float32)
    team = np.asarray(m.team)
    n, S = P.shape[:2]
    L, W = m.pitch
    pres = np.isfinite(P[..., 0])
    bp = np.isfinite(ball[:, 0])
    out = {}
    out["ball_present"] = bp.astype(np.float64)
    last_seen = np.maximum.accumulate(np.where(bp, np.arange(n), -1))
    out["_miss_run"] = (np.arange(n) - last_seen).astype(np.float64)  # 0 where present
    out["ball_speed"] = ball_speed(ball, fps).astype(np.float64)
    d2b = np.full(n, np.nan)
    if n > 2:
        d2b[1:-1] = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=-1)
    out["ball_d2"] = d2b
    near = np.full(n, np.nan)
    nearest = np.full(n, np.nan)
    for a in range(0, n, 50_000):
        b = min(n, a + 50_000)
        d = P[a:b] - ball[a:b, None, :]
        dist = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
        with np.errstate(invalid="ignore"):
            near[a:b] = np.where(bp[a:b], (dist < ball_radius_m).sum(1), np.nan)
        ok = np.isfinite(dist).any(1)
        tmp = np.full(b - a, np.nan)
        tmp[ok] = np.nanmin(dist[ok], axis=1)
        nearest[a:b] = tmp
    out["ball_near"], out["ball_nearest"] = near, nearest
    cnt = pres.sum(1).astype(np.float64)
    out["n_present"] = cnt
    nh = pres[:, team == 0].sum(1).astype(np.float64)
    out["team_count_diff"] = np.abs(nh - (cnt - nh))
    prev = np.vstack([np.zeros((1, S), bool), pres[:-1]])
    nxt = np.vstack([pres[1:], np.zeros((1, S), bool)])
    out["slot_onsets"] = (pres & ~prev).sum(1).astype(np.float64)
    out["slot_offsets"] = (pres & ~nxt).sum(1).astype(np.float64)
    step = np.full((n, S), np.nan, dtype=np.float32)
    step[1:] = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    with np.errstate(invalid="ignore"):
        out["slot_jumps"] = (step > jump_m).sum(1).astype(np.float64)
    d2 = np.full((n, S), np.nan, dtype=np.float32)
    if n > 2:
        d2[1:-1] = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1)
    out["player_d2"] = _nanmean_rows(np.minimum(d2, 2.0))
    sp = np.where(pres, speed, np.nan)
    out["speed_mean"] = _nanmean_rows(np.minimum(sp, 12.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["speed_nan_frac"] = np.where(cnt > 0, (pres & ~np.isfinite(speed)).sum(1) / cnt, np.nan)
        out["speed_fast_frac"] = np.where(cnt > 0, (sp > 7.0).sum(1) / cnt, np.nan)
    idx = np.arange(0, n, stride)
    same, opp = pair_counts(P, team, idx, dup_dist_m)
    dp = np.full(n, np.nan)
    op = np.full(n, np.nan)
    dp[idx], op[idx] = same, opp
    out["dup_pairs"], out["opp_close_pairs"] = dp, op
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["x_spread"] = np.nanstd(P[..., 0], axis=1).astype(np.float64)
        out["y_spread"] = np.nanstd(P[..., 1], axis=1).astype(np.float64)
        ch = np.nanmean(P[:, team == 0], axis=1)
        ca = np.nanmean(P[:, team == 1], axis=1)
    out["team_centroid_dist"] = np.linalg.norm(ch - ca, axis=-1).astype(np.float64)
    with np.errstate(invalid="ignore"):
        oob = (np.abs(P[..., 0]) > L / 2 + 3) | (np.abs(P[..., 1]) > W / 2 + 3)
    out["oob"] = oob.sum(1).astype(np.float64)
    return out


def window_features(m: Match, window_s: float = 10.0, *, block_s: float = 60.0, stride: int = 5,
                    ball_radius_m: float = 3.0, dup_dist_m: float = 0.5, jump_m: float = 1.5):
    """Label-free features per window: (X float64[nw, F], names, groups int64[nw]).

    Windows tile each graded block; `groups` is the time-block id (block_s long) used for CV.
    """
    fps = int(m.fps)
    W = max(2, int(round(window_s * fps)))
    starts, groups = [], []
    for bi, (f, e) in enumerate(blocks_of(m)):
        k = (e - f) // W
        ws = f + W * np.arange(k)
        starts.append(ws)
        groups.append(bi * 100_000 + ((ws - f) // max(1, int(block_s * fps))))
    ws = np.concatenate(starts) if starts else np.zeros(0, int)
    groups = np.concatenate(groups) if groups else np.zeros(0, int)
    if ws.size == 0:
        return np.zeros((0, len(FEATURE_NAMES))), list(FEATURE_NAMES), groups
    s = _frame_series(m, stride, ball_radius_m, dup_dist_m, jump_m)
    fr = ws[:, None] + np.arange(W)[None, :]
    g = {k: v[fr] for k, v in s.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bpw = g["ball_present"]
        miss = np.minimum(g["_miss_run"], np.arange(W)[None, :] + 1)
        cols = {
            "ball_present": bpw.mean(1),
            "ball_gaps": ((bpw[:, :-1] == 1) & (bpw[:, 1:] == 0)).sum(1).astype(float),
            "ball_longest_gap_frac": miss.max(1) / W,
            "ball_speed_med": np.nanmedian(g["ball_speed"], axis=1),
            "ball_speed_q90": np.nanquantile(g["ball_speed"], 0.9, axis=1),
            "ball_d2": np.nanmedian(g["ball_d2"], axis=1),
            "ball_near": np.nanmean(g["ball_near"], axis=1),
            "ball_nearest": np.nanmedian(g["ball_nearest"], axis=1),
            "n_present": g["n_present"].mean(1),
            "n_present_std": g["n_present"].std(1),
            "n_present_min": g["n_present"].min(1),
            "n_present_max": g["n_present"].max(1),
        }
        for k in ("team_count_diff", "slot_onsets", "slot_offsets", "slot_jumps", "player_d2",
                  "speed_mean", "speed_nan_frac", "speed_fast_frac", "dup_pairs", "opp_close_pairs",
                  "x_spread", "y_spread", "team_centroid_dist", "oob"):
            cols[k] = np.nanmean(g[k], axis=1)
    X = np.column_stack([cols[k] for k in FEATURE_NAMES]).astype(np.float64)
    return X, list(FEATURE_NAMES), groups


def _make_model(model: str, seed: int):
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                                  deterministic=True, force_col_wise=True, n_jobs=1, verbose=-1)
    if model == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError("model must be 'lgbm' or 'logreg'")


def _importance(clf, model: str, F: int) -> np.ndarray:
    if model == "lgbm":
        imp = clf.booster_.feature_importance(importance_type="gain").astype(np.float64)
    else:
        imp = np.abs(clf[-1].coef_[0]).astype(np.float64)
    if imp.shape[0] != F:
        return np.zeros(F)
    s = imp.sum()
    return imp / s if s > 0 else imp


def domain_gap(real: Match, sim: Match, *, window_s: float = 10.0, n_folds: int = 5, seed: int = 0,
               model: str = "lgbm", block_s: float = 60.0, top_k: int = 10) -> dict:
    """Cross-validated AUC of real-vs-sim window classification, plus the features that drive it.

    Returns aggregates only: {"auc", "auc_folds", "auc_std", "n_real_windows", "n_sim_windows",
    "n_features", "model", "top_features": [[name, importance], ...]}.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    Xr, names, gr = window_features(real, window_s, block_s=block_s)
    Xs, _, gs = window_features(sim, window_s, block_s=block_s)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(Xr), int), np.zeros(len(Xs), int)])
    groups = np.concatenate([gr, gs])
    n_groups = np.unique(groups).size
    k = min(n_folds, n_groups)
    if k < 2 or len(Xr) < 4 or len(Xs) < 4:
        raise ValueError("not enough windows for a cross-validated domain-gap check")
    oof = np.full(len(y), np.nan)
    aucs, imps = [], []
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in cv.split(X, y, groups):
            if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
                continue
            clf = _make_model(model, seed)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            oof[te] = p
            aucs.append(float(roc_auc_score(y[te], p)))
            imps.append(_importance(clf, model, X.shape[1]))
    ok = np.isfinite(oof)
    if not aucs or np.unique(y[ok]).size < 2:
        raise ValueError("cross-validation produced no usable folds")
    auc = float(roc_auc_score(y[ok], oof[ok]))
    imp = np.mean(imps, axis=0)
    order = np.argsort(-imp, kind="stable")[:top_k]
    return {
        "auc": auc,
        "auc_folds": aucs,
        "auc_std": float(np.std(aucs)),
        "n_real_windows": int(len(Xr)),
        "n_sim_windows": int(len(Xs)),
        "n_features": int(X.shape[1]),
        "model": model,
        "top_features": [[names[i], float(imp[i])] for i in order],
    }
