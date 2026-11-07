"""Label-free aggregate statistics that characterise tracking noise.

`target_stats(m)` never looks at `m.y` and returns only plain floats (rates, counts, quantiles), so it
is safe to run on the competition match and print. Everything is measured on graded frames (the
half blocks), or on the whole match if it has none.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..match import Match
from ._util import (ball_speed, blocks_of, fnum, nan_moving_mean, pair_counts, runs_1d, runs_2d,
                    second_diff_abs)

NAN = float("nan")


def _q(x: np.ndarray, qs) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [NAN] * len(qs)
    return [float(v) for v in np.quantile(x, qs)]


def _rate(events: np.ndarray, at: np.ndarray, min_n: int = 30) -> tuple[float, float]:
    """(rate, log of add-half smoothed rate); NaN if fewer than min_n trials."""
    k, n = int(np.sum(events & at)), int(np.sum(at))
    if n < min_n:
        return NAN, NAN
    return k / n, float(np.log((k + 0.5) / (n + 1.0)))


def _ball_geometry(P: np.ndarray, ball: np.ndarray, r: float, chunk: int = 50_000):
    """Per frame: slots within r of the ball, and the nearest slot's distance (NaN if none)."""
    n = P.shape[0]
    cnt = np.zeros(n, dtype=np.int16)
    near = np.full(n, np.nan, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = P[a:b] - ball[a:b, None, :]
        dist = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
        with np.errstate(invalid="ignore"):
            cnt[a:b] = (dist < r).sum(1)
        ok = np.isfinite(dist).any(1)
        near[a:b][ok] = np.nanmin(dist[ok], axis=1)
    return cnt, near


def target_stats(m: Match, *, stride: int = 5, ball_radius_m: float = 3.0, dup_dist_m: float = 0.5,
                 jump_m: float = 1.5, fast_mps: float = 12.0) -> dict:
    """Aggregate, label-free noise statistics of a match. All values are plain floats."""
    fps = int(m.fps)
    P = np.asarray(m.P, dtype=np.float32)
    ball = np.asarray(m.ball, dtype=np.float32)
    n, S = P.shape[:2]
    L_pitch, W_pitch = m.pitch
    blocks = blocks_of(m)
    g = np.zeros(n, dtype=bool)
    trans = np.zeros(n, dtype=bool)  # t and t+1 both inside one block
    for f, e in blocks:
        g[f:e] = True
        trans[f:max(f, e - 1)] = True
    ng = int(g.sum())
    minutes = max(ng / fps / 60.0, 1e-9)
    out: dict = {
        "n_frames": float(n), "n_graded": float(ng), "n_slots": float(S),
        "home_slot_frac": fnum(np.mean(np.asarray(m.team) == 0)) if S else NAN,
    }

    # ---- ball presence and gaps ------------------------------------------------------------------
    bp = np.isfinite(ball[:, 0])
    out["ball_present_rate"] = fnum(bp[g].mean()) if ng else NAN
    out["ball_present_rate_all"] = fnum(bp.mean()) if n else NAN
    gaps, vis, gap_starts = [], [], []
    for f, e in blocks:
        s, t = runs_1d(~bp[f:e])
        gaps.append(t - s)
        gap_starts.append(np.where(s > 0, s + f, -1))  # -1: gap touches the block start
        s, t = runs_1d(bp[f:e])
        vis.append(t - s)
    gl = np.concatenate(gaps) if gaps else np.zeros(0, int)
    gs = np.concatenate(gap_starts) if gap_starts else np.zeros(0, int)
    vl = np.concatenate(vis) if vis else np.zeros(0, int)
    q = _q(gl / fps, [0.25, 0.5, 0.75, 0.9, 0.99])
    for name, v in zip(("q25", "q50", "q75", "q90", "q99"), q):
        out[f"ball_gap_{name}_s"] = v
    out["ball_gap_mean_s"] = fnum(gl.mean() / fps) if gl.size else NAN
    out["ball_gaps_per_min"] = gl.size / minutes
    out["ball_iso_share"] = fnum(gl[gl <= 3].sum() / gl.sum()) if gl.sum() else NAN
    burst = gl[gl > 3]  # gaps longer than 3 frames: the bursty part, without isolated drops
    out["ball_bursts_per_min"] = burst.size / minutes
    out["ball_burst_q50_s"], out["ball_burst_q90_s"] = _q(burst / fps, [0.5, 0.9])
    q = _q(vl / fps, [0.5, 0.9])
    out["ball_vis_q50_s"], out["ball_vis_q90_s"] = q

    # ---- ball loss hazard vs congestion, speed and activity --------------------------------------
    c, nearest = _ball_geometry(P, ball, ball_radius_m)
    at = trans & bp
    lost = np.zeros(n, dtype=bool)
    lost[:-1] = ~bp[1:]
    h0, l0 = _rate(lost, at & (c == 0))
    h1, _ = _rate(lost, at & (c == 1))
    h2, l2 = _rate(lost, at & (c >= 2))
    _, l01 = _rate(lost, at & (c <= 1))
    _, l1p = _rate(lost, at & (c >= 1))
    out["ball_loss_hazard_c0"], out["ball_loss_hazard_c1"], out["ball_loss_hazard_c2p"] = h0, h1, h2
    out["ball_loss_logratio_contest"] = l2 - l01
    out["ball_loss_logratio_near"] = l1p - l0
    v = ball_speed(ball, fps)
    fin = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        hf, lf = _rate(lost, at & fin & (v > fast_mps))
        hs, ls = _rate(lost, at & fin & (v <= fast_mps))
    out["ball_loss_hazard_fast"], out["ball_loss_hazard_slow"] = hf, hs
    out["ball_loss_logratio_fast"] = lf - ls
    ok = gs > 0
    if ok.any():
        c_on = c[gs[ok] - 1]
        lg = np.log(gl[ok].astype(np.float64))
        a, b = lg[c_on >= 1], lg[c_on == 0]
        out["ball_gap_logratio_near"] = fnum(a.mean() - b.mean()) if a.size >= 20 and b.size >= 20 else NAN
    else:
        out["ball_gap_logratio_near"] = NAN
    out["ball_speed_q50"], out["ball_speed_q90"] = _q(v[g], [0.5, 0.9])
    out["ball_nearest_q50_m"], out["ball_nearest_q90_m"] = _q(nearest[g & bp], [0.5, 0.9])
    out["ball_near_mean"] = fnum(c[g & bp].mean()) if (g & bp).any() else NAN
    db = np.linalg.norm(np.diff(ball, axis=0), axis=-1)
    both = trans[:-1] & bp[:-1] & bp[1:]
    out["ball_jump_rate"] = fnum((db[both] > 3.0).mean()) if both.any() else NAN
    d2b = second_diff_abs(ball)
    d2b = d2b[np.isfinite(d2b).all(1) & g[1:-1]]
    out["ball_d2_mad"] = fnum(np.median(d2b)) if d2b.size else NAN

    speed = np.asarray(m.speed, dtype=np.float32)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        act = np.nanmean(np.minimum(speed, 12.0), axis=1)
    act = nan_moving_mean(act, fps)
    ag = act[g]
    if np.isfinite(ag).sum() > 100:
        thr = np.nanquantile(ag, 0.2)
        lo, hi = g & (act <= thr), g & (act > thr)
        p_lo = fnum(bp[lo].mean()) if lo.any() else NAN
        p_hi = fnum(bp[hi].mean()) if hi.any() else NAN
    else:
        p_lo = p_hi = NAN
    out["ball_present_rate_lowact"], out["ball_present_rate_highact"] = p_lo, p_hi
    out["ball_present_lowact_ratio"] = p_lo / p_hi if p_hi and np.isfinite(p_hi) else NAN

    # ---- present objects ---------------------------------------------------------------------------
    pres = np.isfinite(P[..., 0])
    cnt = pres.sum(1)
    cg = cnt[g]
    out["slots_present_mean"] = fnum(cg.mean()) if ng else NAN
    out["slots_present_q05"], out["slots_present_q50"], out["slots_present_q95"] = _q(cg, [0.05, 0.5, 0.95])
    out["slots_present_frac_gt22"] = fnum((cg > 22).mean()) if ng else NAN
    out["slots_present_frac_lt18"] = fnum((cg < 18).mean()) if ng else NAN
    out["slots_present_std"] = fnum(cg.std()) if ng else NAN
    win, wstd = 5 * fps, []
    for f, e in blocks:  # short-time fluctuation of the count: mean std inside 5 s windows
        k = (e - f) // win
        if k:
            wstd.append(cnt[f:f + k * win].reshape(k, win).std(1))
    out["slots_present_std_5s"] = fnum(np.concatenate(wstd).mean()) if wstd else NAN
    team = np.asarray(m.team)
    nh = pres[g][:, team == 0].sum(1)
    na = pres[g][:, team == 1].sum(1)
    out["home_present_mean"] = fnum(nh.mean()) if ng else NAN
    out["away_present_mean"] = fnum(na.mean()) if ng else NAN
    out["team_count_absdiff_mean"] = fnum(np.abs(nh - na).mean()) if ng else NAN

    # ---- fragments (contiguous runs per slot) ----------------------------------------------------
    run_len, slot_gap = [], []
    for f, e in blocks:
        col, s, t = runs_2d(pres[f:e])
        run_len.append(t - s)
        same = col[1:] == col[:-1]
        slot_gap.append((s[1:] - t[:-1])[same])
    rl = np.concatenate(run_len) if run_len else np.zeros(0, int)
    sg = np.concatenate(slot_gap) if slot_gap else np.zeros(0, int)
    q = _q(rl / fps, [0.1, 0.5, 0.9])
    out["slot_run_q10_s"], out["slot_run_q50_s"], out["slot_run_q90_s"] = q
    out["slot_run_mean_s"] = fnum(rl.mean() / fps) if rl.size else NAN
    out["slot_onsets_per_min"] = rl.size / minutes
    out["slot_gap_q50_s"], out["slot_gap_q90_s"] = _q(sg / fps, [0.5, 0.9])
    occ = pres[g].mean(0) if ng else np.zeros(S)
    out["slots_used"] = float((occ > 0).sum())
    out["slot_occ_q10"], out["slot_occ_q50"], out["slot_occ_q90"] = _q(occ, [0.1, 0.5, 0.9])

    # ---- jumps, jitter, duplicates -----------------------------------------------------------------
    dP = np.diff(P, axis=0)
    with np.errstate(invalid="ignore"):
        step = np.sqrt(dP[..., 0] ** 2 + dP[..., 1] ** 2)
    del dP
    both = pres[:-1] & pres[1:] & trans[:-1, None]
    nb = int(both.sum())
    with np.errstate(invalid="ignore"):
        out["slot_jump_rate"] = fnum((step[both] > jump_m).sum() / nb) if nb else NAN
    idx = np.flatnonzero(g[1:-1])[::stride] + 1
    idx = idx[g[idx - 1] & g[idx + 1]]
    if idx.size:
        d2 = np.abs(P[idx + 1] - 2 * P[idx] + P[idx - 1]).reshape(-1)
        d2 = d2[np.isfinite(d2)]
        out["player_d2_mad"], out["player_d2_q90"] = (_q(d2, [0.5, 0.9]) if d2.size else [NAN, NAN])
        cv = np.sqrt(((P[idx + 1] - P[idx - 1]) ** 2).sum(-1)) * fps / 2.0
        sp = speed[idx]
        dev = np.abs(sp - cv)
        dev = dev[np.isfinite(dev)]
        out["speed_pos_consistency"] = fnum(np.median(dev)) if dev.size else NAN
        same, opp = pair_counts(P, team, idx, dup_dist_m)
        out["dup_pairs_per_frame"] = fnum(same.mean())
        out["dup_frame_frac"] = fnum((same > 0).mean())
        out["opp_close_pairs_per_frame"] = fnum(opp.mean())
    else:
        for k in ("player_d2_mad", "player_d2_q90", "speed_pos_consistency", "dup_pairs_per_frame",
                  "dup_frame_frac", "opp_close_pairs_per_frame"):
            out[k] = NAN

    # ---- speed and bounds --------------------------------------------------------------------------
    pg = pres & g[:, None]
    sfin = np.isfinite(speed)
    out["speed_avail"] = fnum(sfin[pg].mean()) if pg.any() else NAN
    out["speed_q50"], out["speed_q95"], out["speed_q99"] = _q(speed[pg & sfin], [0.5, 0.95, 0.99])
    with np.errstate(invalid="ignore"):
        oob = (np.abs(P[..., 0]) > L_pitch / 2 + 3) | (np.abs(P[..., 1]) > W_pitch / 2 + 3)
    out["oob_frac"] = fnum(oob[pg].mean()) if pg.any() else NAN
    return {k: fnum(v) for k, v in out.items()}
