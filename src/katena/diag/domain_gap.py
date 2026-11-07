"""Domain-gap alarm: can a classifier tell real tracking windows from the degrader's output?

Ported from origin/david/foundation (``katena.noise.gap``) and adapted to this repository: corner-
origin metres (``katena.types.Match``), and the 5 Hz work grid on which ``katena.sim.degrade``
corrupts and ``katena.featurize`` reads matches. Speed-channel statistics are left out, because the
model never reads ``Match.speed`` (features use positions only) and the target's provider speed
would differ from derived speed for reasons that do not matter here.

Each match is cut into fixed windows inside its half blocks. Every window gets label-free summary
statistics (ball presence and gaps, present objects, slot churn, jumps, jitter, duplicates, distances
to the ball, team shape). A classifier is cross-validated on real versus simulated windows, with folds
grouped by time block. An AUC near 0.5 means the degrader is hard to tell from the real input; the top
features say what gives it away.

Rules (docs/PLAN.md, "Validating blind", step 5): this is an alarm. It returns aggregates only (AUC,
window counts, feature names and importances), it is never used to select between versions, and it
is never used to tune the degrader: fitting the degrader to the target is out until the organisers
confirm. ``domain_gap`` returns no configuration, model or per-window value, so there is nothing to
feed back. The labels of either side are never read. It does fit classifiers on both sides' window
statistics, so on the target ``scripts/check_domain_gap.py`` prints only one alarm bit.
"""

from __future__ import annotations

import warnings

import numpy as np

from katena.types import Match

FEATURE_NAMES = [
    "ball_present", "ball_gaps", "ball_longest_gap_frac", "ball_speed_med", "ball_speed_q90",
    "ball_d2", "ball_near", "ball_nearest", "n_present", "n_present_std", "n_present_min",
    "n_present_max", "team_count_diff", "slot_onsets", "slot_offsets", "slot_jumps", "player_d2",
    "dup_pairs", "opp_close_pairs", "x_spread", "y_spread", "team_centroid_dist", "oob",
]
RESULT_KEYS = ("auc", "auc_folds", "auc_std", "n_real_windows", "n_sim_windows", "n_features", "model",
               "top_features")


def _blocks(m: Match) -> list[tuple[int, int]]:
    """Half-open (first, stop) ranges of the half blocks, or the whole match if it has none."""
    out = [(int(a), int(b) + 1) for _, a, b in m.half_blocks if b >= a]
    return out or [(0, m.n)]


def _ball_speed(ball: np.ndarray, fps: float, lag: int) -> np.ndarray:
    n, h = ball.shape[0], lag // 2
    v = np.full(n, np.nan, dtype=np.float32)
    if n > lag:
        d = ball[lag:] - ball[:-lag]
        v[h:h + n - lag] = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) * (fps / lag)
    return v


def _pair_counts(P: np.ndarray, team: np.ndarray, r: float, chunk: int = 2048):
    """Per frame: same-team and cross-team slot pairs closer than r metres."""
    S = P.shape[1]
    iu, ju = np.triu_indices(S, 1)
    same = team[iu] == team[ju]
    out_s = np.zeros(P.shape[0], dtype=np.float64)
    out_o = np.zeros(P.shape[0], dtype=np.float64)
    for a in range(0, P.shape[0], chunk):
        Q = P[a:a + chunk]
        d = Q[:, iu, :] - Q[:, ju, :]
        with np.errstate(invalid="ignore"):
            close = (d[..., 0] ** 2 + d[..., 1] ** 2) < r * r
        out_s[a:a + chunk] = close[:, same].sum(1)
        out_o[a:a + chunk] = close[:, ~same].sum(1)
    return out_s, out_o


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)


def _frame_series(m: Match, ball_radius_m: float, dup_dist_m: float, jump_m: float) -> dict:
    fps = float(m.fps)
    P = np.asarray(m.P, dtype=np.float32)
    ball = np.asarray(m.ball, dtype=np.float32)
    team = np.asarray(m.team)
    n, S = P.shape[:2]
    L, W = m.pitch
    pres = np.isfinite(P[..., 0])
    bp = np.isfinite(ball[:, 0])
    out = {"ball_present": bp.astype(np.float64)}
    last_seen = np.maximum.accumulate(np.where(bp, np.arange(n), -1))
    out["_miss_run"] = (np.arange(n) - last_seen).astype(np.float64)
    out["ball_speed"] = _ball_speed(ball, fps, lag=max(2, int(round(0.4 * fps)))).astype(np.float64)
    d2b = np.full(n, np.nan)
    if n > 2:
        d2b[1:-1] = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=-1)
    out["ball_d2"] = d2b
    near, nearest = np.full(n, np.nan), np.full(n, np.nan)
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
        out["slot_jumps"] = (step > jump_m * 25.0 / fps).sum(1).astype(np.float64)
    d2 = np.full((n, S), np.nan, dtype=np.float32)
    if n > 2:
        d2[1:-1] = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1)
    out["player_d2"] = _nanmean_rows(np.minimum(d2, 2.0))
    out["dup_pairs"], out["opp_close_pairs"] = _pair_counts(P, team, dup_dist_m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["x_spread"] = np.nanstd(P[..., 0], axis=1).astype(np.float64)
        out["y_spread"] = np.nanstd(P[..., 1], axis=1).astype(np.float64)
        ch = np.nanmean(P[:, team == 0], axis=1)
        ca = np.nanmean(P[:, team == 1], axis=1)
    out["team_centroid_dist"] = np.linalg.norm(ch - ca, axis=-1).astype(np.float64)
    with np.errstate(invalid="ignore"):
        oob = (P[..., 0] < -3) | (P[..., 0] > L + 3) | (P[..., 1] < -3) | (P[..., 1] > W + 3)
    out["oob"] = oob.sum(1).astype(np.float64)
    return out


def window_features(m: Match, window_s: float = 10.0, *, block_s: float = 60.0, ball_radius_m: float = 3.0,
                    dup_dist_m: float = 0.5, jump_m: float = 1.5):
    """Label-free statistics per window: (X float64[n_windows, F], names, groups int64[n_windows]).

    Windows tile each half block; ``groups`` is a time-block id (``block_s`` long) used for CV.
    ``jump_m`` is per 25 Hz frame and scaled to the match's frame rate."""
    w = max(2, int(round(window_s * m.fps)))
    starts, groups = [], []
    for bi, (f, e) in enumerate(_blocks(m)):
        ws = f + w * np.arange((e - f) // w)
        starts.append(ws)
        groups.append(bi * 100_000 + (ws - f) // max(1, int(round(block_s * m.fps))))
    ws = np.concatenate(starts) if starts else np.zeros(0, int)
    groups = np.concatenate(groups) if groups else np.zeros(0, int)
    if ws.size == 0:
        return np.zeros((0, len(FEATURE_NAMES))), list(FEATURE_NAMES), groups
    s = _frame_series(m, ball_radius_m, dup_dist_m, jump_m)
    fr = ws[:, None] + np.arange(w)[None, :]
    g = {k: v[fr] for k, v in s.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bpw = g["ball_present"]
        miss = np.minimum(g["_miss_run"], np.arange(w)[None, :] + 1)
        cols = {
            "ball_present": bpw.mean(1),
            "ball_gaps": ((bpw[:, :-1] == 1) & (bpw[:, 1:] == 0)).sum(1).astype(float),
            "ball_longest_gap_frac": miss.max(1) / w,
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
        for k in ("team_count_diff", "slot_onsets", "slot_offsets", "slot_jumps", "player_d2", "dup_pairs",
                  "opp_close_pairs", "x_spread", "y_spread", "team_centroid_dist", "oob"):
            cols[k] = np.nanmean(g[k], axis=1)
    X = np.column_stack([cols[k] for k in FEATURE_NAMES]).astype(np.float64)
    return X, list(FEATURE_NAMES), groups


def _model(kind: str, seed: int):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, deterministic=True, force_col_wise=True, n_jobs=1,
                                  verbose=-1)
    if kind == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError("model must be 'lgbm' or 'logreg'")


def _importance(clf, kind: str, n_features: int) -> np.ndarray:
    imp = (clf.booster_.feature_importance(importance_type="gain") if kind == "lgbm"
           else np.abs(clf[-1].coef_[0])).astype(np.float64)
    if imp.shape[0] != n_features:
        return np.zeros(n_features)
    return imp / imp.sum() if imp.sum() > 0 else imp


def domain_gap(real: Match, sims, *, window_s: float = 10.0, n_folds: int = 5, seed: int = 0,
               model: str = "lgbm", block_s: float = 60.0, top_k: int = 10) -> dict:
    """Cross-validated AUC of real-versus-simulated window classification and what drives it.

    ``sims``: one Match or a list (several degraded matches pooled as the simulated side). Both
    sides must be on the same frame grid. Returns aggregates only (``RESULT_KEYS``)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    sims = [sims] if isinstance(sims, Match) else list(sims)
    if any(abs(s.fps - real.fps) > 1e-9 for s in sims):
        raise ValueError("real and simulated matches must be on the same frame grid")
    Xr, names, gr = window_features(real, window_s, block_s=block_s)
    parts = [window_features(s, window_s, block_s=block_s) for s in sims]
    Xs = np.vstack([p[0] for p in parts])
    gs = np.concatenate([(i + 1) * 10_000_000 + p[2] for i, p in enumerate(parts)])
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(Xr), int), np.zeros(len(Xs), int)])
    groups = np.concatenate([gr, gs])
    k = min(n_folds, np.unique(groups).size)
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
            clf = _model(model, seed)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
            aucs.append(float(roc_auc_score(y[te], oof[te])))
            imps.append(_importance(clf, model, X.shape[1]))
    ok = np.isfinite(oof)
    if not aucs or np.unique(y[ok]).size < 2:
        raise ValueError("cross-validation produced no usable folds")
    imp = np.mean(imps, axis=0)
    order = np.argsort(-imp, kind="stable")[:top_k]
    return {"auc": float(roc_auc_score(y[ok], oof[ok])), "auc_folds": aucs, "auc_std": float(np.std(aucs)),
            "n_real_windows": int(len(Xr)), "n_sim_windows": int(len(Xs)), "n_features": int(X.shape[1]),
            "model": model, "top_features": [[names[i], float(imp[i])] for i in order]}
