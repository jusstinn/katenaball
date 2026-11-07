#!/usr/bin/env python
"""Estimate degrader constants from real automatic tracking (SkillCorner + Metrica).

    uv run python scripts/calibrate_degrader.py

Writes src/katena/sim/calibration.json. Uses only external matches. Gap durations are measured on
the 5 Hz grid inside contiguous stretches where tracking exists (>= 10 players), so camera
cut-aways are not counted as tracker ball loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from katena.grid import block_mask, downsample  # noqa: E402
from katena.types import D, load_match  # noqa: E402

ROOT = Path(__import__("os").environ.get("KATENA_EXTERNAL") or REPO / "external_data") / "canonical"
OUT = REPO / "src" / "katena" / "sim" / "calibration.json"


def runs(mask: np.ndarray):
    """[(start, stop)] of True runs and whether each touches an array end (censored)."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(s.tolist(), e.tolist()))


def lognormal_fit(secs: np.ndarray) -> dict:
    x = np.log(np.maximum(secs, 0.2))
    return {"mu": float(x.mean()), "sigma": float(x.std()), "n": int(len(x)),
            "mean_s": float(secs.mean()), "median_s": float(np.median(secs)),
            "p90_s": float(np.percentile(secs, 90))}


metrica_alive_gaps: list = []


def ball_stats(m5, missing: np.ndarray, valid: np.ndarray, dt: float):
    """Gap durations (s), onset-hazard dataset rows (congestion, speed, dead), missing fraction."""
    gaps, rows = [], []
    ball = m5.ball.copy()
    seen = ~missing
    last = np.full(2, np.nan)
    last_known = np.full((m5.n, 2), np.nan, dtype=np.float32)
    for i in range(m5.n):
        if seen[i] and np.isfinite(ball[i, 0]):
            last = ball[i]
        last_known[i] = last
    d = np.linalg.norm(m5.P - last_known[:, None, :], axis=-1)
    cong = np.nansum(d < 4.0, axis=1)
    sp = np.full(m5.n, np.nan)
    sp[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=-1) / dt
    dead = (m5.y == D) if m5.y is not None else np.zeros(m5.n, bool)
    # stretches of valid tracking
    for a, b in runs(valid):
        mm = missing[a:b]
        for s, e in runs(mm):
            if s > 0 and e < b - a:                       # uncensored
                gaps.append((e - s) * dt)
                if m5.source == 'metrica' and m5.y is not None and m5.y[a + s] < 2:
                    metrica_alive_gaps.append((e - s) * dt)
        for i in range(a + 1, b):
            if seen[i - 1] and np.isfinite(sp[i - 1]):
                rows.append((cong[i - 1], min(sp[i - 1], 30.0), float(dead[i - 1]), float(missing[i])))
    return np.array(gaps), np.array(rows), float(missing[valid].mean())


def main():
    result = {"sources": {}}
    all_ball_gaps, all_player_gaps, hazard_rows = [], [], []
    for source in ("skillcorner", "metrica"):
        bg, pg, rows, fr = [], [], [], []
        pm_miss_d = [0, 0, 0, 0]
        for p in sorted((ROOT / source).glob("*.npz")):
            m5 = downsample(load_match(p))
            dt = 1.0 / m5.fps
            blk = block_mask(m5)
            npl = np.isfinite(m5.P[..., 0]).sum(1)
            valid = blk & (npl >= 10)
            ball_nan = ~np.isfinite(m5.ball[:, 0])
            if source == "skillcorner":
                det = m5.extras["ball_detected"] == 1
                missing = ball_nan | ~det
            else:
                missing = ball_nan
            g, r, f = ball_stats(m5, missing, valid, dt)
            bg.append(g); rows.append(r); fr.append(f)
            if m5.y is not None and source == "metrica":
                dead = m5.y == D
                alive = (m5.y < 2)
                pm_miss_d[0] += int((missing & dead & valid).sum()); pm_miss_d[1] += int((dead & valid).sum())
                pm_miss_d[2] += int((missing & alive & valid).sum()); pm_miss_d[3] += int((alive & valid).sum())
            if source == "skillcorner":
                pd_ = m5.extras["player_detected"][:, :] if m5.extras["player_detected"].ndim == 2 else None
                if pd_ is not None:
                    und = (pd_ != 1)
                    for s in range(und.shape[1]):
                        for a, b in runs(und[:, s] & valid):
                            if a > 0 and b < m5.n:
                                pg.append((b - a) * dt)
        bgc, rowsc = np.concatenate(bg), np.concatenate(rows)
        entry = {"ball_missing_fraction_mean": float(np.mean(fr)),
                 "ball_missing_fraction_range": [float(min(fr)), float(max(fr))],
                 "ball_gap": lognormal_fit(bgc), "n_hazard_rows": int(len(rowsc))}
        if source == "metrica":
            entry["p_missing_given_dead_label"] = pm_miss_d[0] / max(pm_miss_d[1], 1)
            entry["p_missing_given_alive_label"] = pm_miss_d[2] / max(pm_miss_d[3], 1)
        if pg:
            entry["player_gap"] = lognormal_fit(np.array(pg))
            all_player_gaps.append(np.array(pg))
        result["sources"][source] = entry
        all_ball_gaps.append(bgc); hazard_rows.append(rowsc)

    # onset hazard models. Metrica rows are split by label: the "alive" fit is the one the degrader
    # uses (pooling alive and dead mixes in the dead-ball ball loss, which has the opposite sign).
    def fit(rows_, with_dead=False):
        cols = [np.minimum(rows_[:, 0], 8.0) / 4.0, rows_[:, 1] / 10.0]
        if with_dead:
            cols.append(rows_[:, 2])
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(np.c_[tuple(cols)], rows_[:, 3])
        return lr
    sc_rows, me_rows = hazard_rows
    lr_sc = fit(sc_rows)
    me_alive = me_rows[me_rows[:, 2] == 0]
    lr_me = fit(me_alive)
    lr_dead = fit(me_rows, with_dead=True)
    result["onset_logit_metrica_alive"] = {
        "intercept": float(lr_me.intercept_[0]), "per_4_congested_players": float(lr_me.coef_[0, 0]),
        "per_10mps_ball_speed": float(lr_me.coef_[0, 1]), "onset_rate": float(me_alive[:, 3].mean())}
    result["onset_logit_skillcorner"] = {
        "intercept": float(lr_sc.intercept_[0]), "per_4_congested_players": float(lr_sc.coef_[0, 0]),
        "per_10mps_ball_speed": float(lr_sc.coef_[0, 1]), "onset_rate": float(sc_rows[:, 3].mean())}
    result["metrica_dead_onset_odds_ratio"] = float(np.exp(lr_dead.coef_[0, 2]))
    # gap durations that start while the ball is alive (Metrica) are free of the long dead-ball losses
    result["ball_gap_metrica_alive_onset"] = lognormal_fit(np.array(metrica_alive_gaps))
    bg = np.concatenate(all_ball_gaps)
    result["ball_gap_pooled"] = lognormal_fit(bg)
    if all_player_gaps:
        result["player_gap_pooled"] = lognormal_fit(np.concatenate(all_player_gaps))
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
