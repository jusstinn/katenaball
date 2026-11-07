"""Native-frame-rate touch and turnover features (feature group ``touch``) on the 5 Hz work grid.

The base groups (``build.py``) see the match after it has been downsampled to 5 Hz. A touch (a kick, a
reception, a deflection) changes the ball's velocity within a frame or two, and at 5 Hz most of that is
gone. This module works on the ball at the source frame rate (25 Hz for IDSSE, Metrica and the target,
10 Hz for SkillCorner):

1. **Velocity breaks.** At every native frame, straight lines are fitted by least squares to the visible
   ball positions in the window just before and just after it (``K_EVENT_S`` each). A touch is a local
   maximum of the velocity change between the two fits that is large in m/s and large against the noise
   (a z-score from the local jitter estimate). Fits use visible frames only, so nothing differences across
   a ball gap.
2. **Attribution.** Each break is given to the nearest player of either team at that instant when he is
   within ``R_TOUCH`` and clearly nearer than the other team's nearest player.
3. **Features** for the focal team, aggregated onto each work-grid sample: time since and until each
   team's last and next touch, who touched last and before that, turnovers between touches, touch counts
   and balance in short windows (0.5-2 s, shorter than the base group's), the strongest break near the
   sample and who was near it, each team's closest approach to the ball between work-grid samples and the
   control margin (opponent's minus focal's nearest distance) in 0.2-1 s windows, ball speed and flight,
   and the local jitter estimate. Distances use the ball smoothed by a short centred line fit (``K_SMOOTH_S``
   each side), which averages frame-level jitter and leaves a smooth track as it is.

Inputs. The ball comes from ``NativeBall``: for training, the native ball corrupted consistently with the
5 Hz corruption (``lift_ball``); at inference, the real native ball. Players come from the (corrupted)
5 Hz match, interpolated linearly to native frames: players move slowly, so the interpolation error
(centimetres) is far below the tracking jitter, and both feature groups see one player path.

The features do not depend on attack direction (distances, speeds, counts, times), so no rotation is
needed. Team-signed values are +1 for the focal team and -1 for the opponent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from katena.grid import step_for
from katena.types import Match

GROUP = "touch"

K_EVENT_S = 0.40        # one-sided fit window before / after a candidate touch
K_SMOOTH_S = 0.12       # half-window of the centred fit used for speed and position
NMS_S = 0.40            # breaks closer than this are one touch
Z_EVENT = 3.5           # a break must exceed this many noise standard deviations ...
DV_MIN = 2.0            # ... and this many m/s
R_TOUCH = 3.0           # m: the nearest player this close to the break point touched the ball ...
MARGIN_TOUCH = 0.3      # ... if he is nearer than the other team's nearest player by this much
FLIGHT_SPEED = 8.0      # m/s
FLIGHT_CLEAR = 2.0      # m: nobody this close -> the ball is in flight
SIGMA_FLOOR = 0.02      # m: lower bound of the jitter estimate (clean optical tracking)
SIGMA_WIN_S = 10.0      # rolling window of the jitter estimate
MAX_FILL_S = 0.10       # ball gaps up to this long are interpolated before fitting
CAP_S = 30.0            # time-since / time-until cap
MIN_FIT_FRAC = 0.5      # a one-sided fit needs at least this share of its window visible
BIG = 1e6


@dataclass(frozen=True)
class NativeBall:
    """The ball at the source frame rate. Native frame ``i * step`` is work-grid sample ``i``."""

    ball: np.ndarray        # float32[n_native, 2] metres, NaN = missing
    fps: float
    step: int

    @property
    def n(self) -> int:
        return int(len(self.ball))


def native_ball(m: Match, hz: float = 5.0) -> NativeBall:
    """The ball of a native-grid match, as the input of ``touch_features``."""
    return NativeBall(np.asarray(m.ball, dtype=np.float32), float(m.fps), step_for(m.fps, hz))


# ----------------------------------------------------------------------------------------------
# noise consistency: a native-rate ball that agrees with the 5 Hz corruption

def lift_ball(ball_native: np.ndarray, ball5_clean: np.ndarray, ball5_corrupt: np.ndarray, step: int,
              sigma: float, seed: int) -> np.ndarray:
    """Native-rate corrupted ball that downsamples exactly to ``ball5_corrupt``.

    ``katena.sim.degrade.corrupt`` works on the 5 Hz match. Its ball corruption is lifted to the native
    frames around each sample (the frames whose nearest sample it is):

    * a sample the degrader dropped (finite in ``ball5_clean``, NaN in ``ball5_corrupt``) drops all its
      native frames, so bursts keep their durations; so does a sample already missing at the source
      whose nearest source-visible samples on both sides were dropped (it sits inside a burst, and its
      native frames must not show the ball in the middle of it);
    * a false detection (moved further than any jitter could, ``> max(2, 4 sqrt(2) sigma)`` m) is held
      at its position on its native frames, as the degrader holds it over its samples;
    * every other visible native frame gets the clean native position plus i.i.d. N(0, sigma^2) jitter
      per axis, the degrader's jitter model applied at the native rate;
    * native frames that are NaN at the source stay NaN;
    * the frame of each sample (``i * step``) is set to ``ball5_corrupt[i]`` exactly.

    So ``lifted[::step] == ball5_corrupt`` and the base features need no change. With no corruption
    (``ball5_corrupt == ball5_clean``, ``sigma == 0``) the lift returns the native ball.
    """
    ball_native = np.asarray(ball_native, dtype=np.float64)
    n = len(ball_native)
    n5 = len(ball5_clean)
    if n5 != -(-n // step) or len(ball5_corrupt) != n5:
        raise ValueError("5 Hz arrays do not match the native length")
    c5 = np.asarray(ball5_clean, dtype=np.float64)
    k5 = np.asarray(ball5_corrupt, dtype=np.float64)
    j = np.arange(n)
    anc = np.minimum((j + step // 2) // step, n5 - 1)
    clean_ok = np.isfinite(c5[:, 0])
    vis = np.isfinite(k5[:, 0])
    dropped = clean_ok & ~vis
    i5 = np.arange(n5)
    left = np.maximum.accumulate(np.where(clean_ok, i5, -1))                # last source-visible <= i
    right = np.minimum.accumulate(np.where(clean_ok, i5, n5)[::-1])[::-1]   # next source-visible >= i
    inside = ~clean_ok & (left >= 0) & (right < n5)
    inside[inside] = dropped[left[inside]] & dropped[right[inside]]
    dropped = dropped | inside
    delta = np.where((clean_ok & vis)[:, None], k5 - c5, 0.0)
    tol = max(2.0, 4.0 * np.sqrt(2.0) * float(sigma))
    false = clean_ok & vis & (np.hypot(delta[:, 0], delta[:, 1]) > tol)
    out = ball_native.copy()
    if sigma > 0:
        out += np.random.default_rng(seed).normal(0.0, float(sigma), (n, 2))
    fa = false[anc]
    out[fa] = k5[anc[fa]]
    out[dropped[anc]] = np.nan
    at = j % step == 0
    out[at] = k5[j[at] // step]
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------
# helpers

def players_at(P5: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Slot positions at fractional work-grid times ``tq`` (sample units). Linear between the two
    neighbouring samples when both are finite, else the nearer one, else NaN. float32[len(tq), S, 2]."""
    n5 = len(P5)
    i0 = np.clip(np.floor(tq).astype(np.int64), 0, n5 - 1)
    i1 = np.minimum(i0 + 1, n5 - 1)
    w = np.clip(tq - i0, 0.0, 1.0).astype(np.float32)[:, None, None]
    p0, p1 = P5[i0], P5[i1]
    both = np.isfinite(p0[..., :1]) & np.isfinite(p1[..., :1])
    near = np.where(w < 0.5, p0, p1)
    return np.where(both, p0 + (p1 - p0) * w, near)


def _fill_short(b: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs of at most ``max_gap`` frames (interior runs only)."""
    out = b.copy()
    if max_gap <= 0:
        return out
    vis = np.isfinite(b[:, 0])
    idx = np.flatnonzero(vis)
    if len(idx) < 2:
        return out
    gaps = np.flatnonzero((np.diff(idx) > 1) & (np.diff(idx) - 1 <= max_gap))
    for k in gaps:
        a, c = idx[k], idx[k + 1]
        w = (np.arange(a + 1, c) - a) / float(c - a)
        out[a + 1:c] = b[a] * (1 - w)[:, None] + b[c] * w[:, None]
    return out


def _shift(x: np.ndarray, k: int, fill) -> np.ndarray:
    """y[t] = x[t + k] (zero/NaN padded)."""
    if k == 0:
        return x
    out = np.full_like(x, fill)
    if k > 0:
        out[:-k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def _line_fit(x: np.ndarray, w: np.ndarray, lo: int, hi: int):
    """Least-squares line through the visible frames at offsets lo..hi around every frame t.
    Returns slope (per frame, [n, 2]), value at t ([n, 2]), sum of squared offset deviations ([n]) and
    the number of visible frames ([n])."""
    n = len(x)
    wf = w.astype(np.float64)
    xz = np.where(w[:, None], x, 0.0)
    S0, S1, S2 = np.zeros(n), np.zeros(n), np.zeros(n)
    Sx, Stx = np.zeros((n, 2)), np.zeros((n, 2))
    for k in range(lo, hi + 1):
        ww, xx = _shift(wf, k, 0.0), _shift(xz, k, 0.0)
        S0 += ww
        S1 += ww * k
        S2 += ww * (k * k)
        Sx += xx
        Stx += xx * k
    den = S0 * S2 - S1 * S1
    need = max(3, int(np.ceil(MIN_FIT_FRAC * (hi - lo + 1))))
    ok = (S0 >= need) & (den > 1e-9)
    slope = np.full((n, 2), np.nan)
    val = np.full((n, 2), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope[ok] = (S0[ok, None] * Stx[ok] - S1[ok, None] * Sx[ok]) / den[ok, None]
        val[ok] = (Sx[ok] - slope[ok] * S1[ok, None]) / S0[ok, None]
        sxx = np.where(ok, den / np.maximum(S0, 1.0), np.nan)
    return slope, val, sxx, S0


def _jitter(b: np.ndarray, fps: float) -> np.ndarray:
    """Local jitter (m, per axis) from the rolling median absolute second difference of visible
    triples, per axis then averaged: i.i.d. jitter sigma gives a second difference with sd
    sigma * sqrt(6), whose absolute value has median 0.6745 sigma sqrt(6)."""
    vis = np.isfinite(b[:, 0])
    ok = vis & _shift(vis, 1, False) & _shift(vis, -1, False)
    d2 = np.abs(_shift(b, 1, np.nan) - 2 * b + _shift(b, -1, np.nan))
    d2[~ok] = np.nan
    win = max(3, int(round(SIGMA_WIN_S * fps)))
    est = []
    for ax in (0, 1):
        s = pd.Series(d2[:, ax]).rolling(win, center=True, min_periods=max(3, win // 10)).median().to_numpy()
        glob = np.nanmedian(d2[:, ax]) if np.isfinite(d2[:, ax]).any() else np.nan
        s = np.where(np.isfinite(s), s, glob)
        est.append(np.where(np.isfinite(s), s, 0.0))
    s = 0.5 * (est[0] + est[1]) / (0.6745 * np.sqrt(6.0))
    return np.maximum(s, SIGMA_FLOOR)


def _roll(x: np.ndarray, back: int, fwd: int, how: str) -> np.ndarray:
    """Rolling max / min / sum / mean over [t - back, t + fwd], truncated at the ends (NaN ignored;
    NaN where the window holds no finite value)."""
    s = pd.Series(x)
    if fwd:
        s = s.shift(-fwd)           # the NaN padding truncates the last windows
    return getattr(s.rolling(back + fwd + 1, min_periods=1), how)().to_numpy()


def _since(t_ev: np.ndarray, n: int) -> np.ndarray:
    """For every frame, frames since the last event at or before it (BIG if none)."""
    last = np.full(n, -BIG)
    if len(t_ev):
        last[t_ev] = t_ev
        last = np.maximum.accumulate(last)
    return np.arange(n) - last


def _until(t_ev: np.ndarray, n: int) -> np.ndarray:
    nxt = np.full(n, BIG)
    if len(t_ev):
        nxt[t_ev] = t_ev
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return nxt - np.arange(n)


def _last_value(t_ev: np.ndarray, val: np.ndarray, n: int, default=0.0) -> np.ndarray:
    """Value of the last event at or before each frame."""
    out = np.full(n, default, dtype=np.float64)
    if len(t_ev):
        pos = np.searchsorted(t_ev, np.arange(n), side="right") - 1
        ok = pos >= 0
        out[ok] = val[pos[ok]]
    return out


def _next_value(t_ev: np.ndarray, val: np.ndarray, n: int, default=0.0) -> np.ndarray:
    out = np.full(n, default, dtype=np.float64)
    if len(t_ev):
        pos = np.searchsorted(t_ev, np.arange(n), side="left")
        ok = pos < len(t_ev)
        out[ok] = val[pos[ok]]
    return out


# ----------------------------------------------------------------------------------------------
# one half

def detect_touches(b: np.ndarray, fps: float, P_nat_fn, team: np.ndarray, k_s: float = K_EVENT_S,
                   z_min: float = Z_EVENT, dv_min: float = DV_MIN) -> dict:
    """Velocity breaks of one half's native ball ``b`` ([n, 2], NaN = missing) and their attribution.

    ``P_nat_fn(frames)`` returns the slot positions at those native frames ([len, S, 2]); ``team`` is
    the slot team (0 home, 1 away). Returns per-frame arrays (``dv``, ``z``, ``speed``, ``sigma``,
    ``vis``) and the events (``t``, ``pos``, ``d_home``, ``d_away``, ``who`` in {0, 1, -1})."""
    bf = _fill_short(b.astype(np.float64), int(round(MAX_FILL_S * fps)))
    vis = np.isfinite(bf[:, 0])
    K = max(2, int(round(k_s * fps)))
    Ks = max(1, int(round(K_SMOOTH_S * fps)))
    sb, vb, xb, _ = _line_fit(bf, vis, -K, 0)
    sa, va, xa, _ = _line_fit(bf, vis, 0, K)
    sc, vc, _, _ = _line_fit(bf, vis, -Ks, Ks)
    sigma = _jitter(bf, fps)
    with np.errstate(invalid="ignore"):
        dv = np.hypot(*(sa - sb).T) * fps
        sd = sigma * fps * np.sqrt(1.0 / xb + 1.0 / xa)
        z = dv / sd
        speed = np.hypot(*sc.T) * fps
    m = max(1, int(round(NMS_S * fps / 2)))
    zz = np.where(np.isfinite(z), z, -1.0)
    peak = zz >= _roll(zz, m, m, "max")
    cand = np.flatnonzero(peak & (zz >= z_min) & (np.nan_to_num(dv) >= dv_min))
    if len(cand) > 1:                                   # plateaus: keep the first frame of a run
        cand = cand[np.concatenate(([True], np.diff(cand) > m))]
    pos = 0.5 * (vb[cand] + va[cand])
    d_home = np.full(len(cand), np.nan)
    d_away = np.full(len(cand), np.nan)
    if len(cand):
        Pe = P_nat_fn(cand).astype(np.float64)               # [E, S, 2]
        d = np.linalg.norm(Pe - pos[:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, BIG)
        dh = d[:, team == 0].min(1) if (team == 0).any() else np.full(len(cand), BIG)
        da = d[:, team == 1].min(1) if (team == 1).any() else np.full(len(cand), BIG)
        d_home, d_away = np.where(dh >= BIG, np.nan, dh), np.where(da >= BIG, np.nan, da)
    dh0, da0 = np.nan_to_num(d_home, nan=BIG), np.nan_to_num(d_away, nan=BIG)
    who = np.where((dh0 < R_TOUCH) & (dh0 + MARGIN_TOUCH < da0), 0,
                   np.where((da0 < R_TOUCH) & (da0 + MARGIN_TOUCH < dh0), 1, -1)).astype(np.int8)
    # smoothed ball for proximity: the centred fit where it exists, else the (short-gap filled) ball
    smooth = np.where(np.isfinite(vc[:, :1]), vc, bf)
    return {"dv": dv, "z": z, "speed": speed, "sigma": sigma, "vis": vis, "ball": bf, "ball_s": smooth,
            "t": cand.astype(np.int64), "pos": pos, "d_home": d_home, "d_away": d_away, "who": who,
            "v_before": sb[cand] * fps, "v_after": sa[cand] * fps}


def _nearest_per_frame(bf: np.ndarray, vis: np.ndarray, P_nat_fn, team: np.ndarray, chunk: int = 8192):
    """Nearest home and away distance to the ball at every native frame (NaN where no ball)."""
    n = len(bf)
    dh, da = np.full(n, np.nan), np.full(n, np.nan)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fr = np.arange(s, e)
        ok = vis[s:e]
        if not ok.any():
            continue
        P = P_nat_fn(fr[ok]).astype(np.float64)
        d = np.linalg.norm(P - bf[fr[ok]][:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, BIG)
        for tm, out in ((0, dh), (1, da)):
            sel = team == tm
            v = d[:, sel].min(1) if sel.any() else np.full(len(d), BIG)
            out[fr[ok]] = np.where(v >= BIG, np.nan, v)
    return dh, da


def _half_features(det: dict, dh_nat: np.ndarray, da_nat: np.ndarray, focal: int, fps: float,
                   anchors: np.ndarray, step: int) -> tuple[list, list]:
    """Per-sample focal features of one half. ``anchors``: native frame of each work-grid row."""
    n = len(det["dv"])
    names, cols = [], []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    s1 = fps
    cap = CAP_S * fps
    t_ev, who = det["t"], det["who"]
    sign = np.where(who < 0, 0, np.where(who == focal, 1, -1)).astype(np.float64)   # +1 focal, -1 opp
    d_f_ev = det["d_home"] if focal == 0 else det["d_away"]
    d_o_ev = det["d_away"] if focal == 0 else det["d_home"]
    d_f = dh_nat if focal == 0 else da_nat
    d_o = da_nat if focal == 0 else dh_nat
    tf, to = t_ev[sign > 0], t_ev[sign < 0]
    ident = sign != 0
    t_id, s_id = t_ev[ident], sign[ident]

    a = anchors
    half_lo = step // 2
    half_hi = (step - 1) // 2
    cell = lambda x, how: _roll(x, half_lo, half_hi, how)[a]           # noqa: E731
    win = lambda x, w, how: _roll(x, int(round(w * s1)), int(round(w * s1)), how)[a]   # noqa: E731

    # who touched, and when
    for nm, arr in (("t_since_f", _since(tf, n)), ("t_since_o", _since(to, n)),
                    ("t_until_f", _until(tf, n)), ("t_until_o", _until(to, n))):
        add(nm, np.minimum(arr, cap)[a] / s1)
    last_team = _last_value(t_id, s_id, n)
    next_team = _next_value(t_id, s_id, n)
    prev_team = _last_value(t_id[1:], s_id[:-1], n) if len(t_id) > 1 else np.zeros(n)
    add("t_last_team", last_team[a])
    add("t_next_team", next_team[a])
    add("t_prev_team", prev_team[a])
    add("t_last_age", np.minimum(_since(t_id, n), cap)[a] / s1)
    add("t_next_wait", np.minimum(_until(t_id, n), cap)[a] / s1)
    margin_ev = np.clip(np.nan_to_num(d_o_ev, nan=8.0) - np.nan_to_num(d_f_ev, nan=8.0), -8, 8)
    add("t_last_margin", _last_value(t_ev, margin_ev, n, np.nan)[a])
    add("t_next_margin", _next_value(t_ev, margin_ev, n, np.nan)[a])

    # turnovers: identified touches whose team differs from the previous identified touch
    if len(t_id) > 1:
        chg = np.flatnonzero(s_id[1:] != s_id[:-1]) + 1
        t_turn, s_turn = t_id[chg], s_id[chg]
    else:
        t_turn, s_turn = np.zeros(0, np.int64), np.zeros(0)
    add("t_since_turn", np.minimum(_since(t_turn, n), cap)[a] / s1)
    add("t_until_turn", np.minimum(_until(t_turn, n), cap)[a] / s1)
    add("t_turn_sign", _last_value(t_turn, s_turn, n)[a])
    add("t_next_turn_sign", _next_value(t_turn, s_turn, n)[a])

    # touch counts and balance in short windows
    ind_f, ind_o = np.zeros(n), np.zeros(n)
    ind_f[tf] = 1.0
    ind_o[to] = 1.0
    for w in (0.5, 1.0, 2.0):
        nf, no = win(ind_f, w, "sum"), win(ind_o, w, "sum")
        add(f"t_nf_{w:g}s", nf)
        add(f"t_no_{w:g}s", no)
        if w >= 1.0:
            add(f"t_bal_{w:g}s", (nf - no) / (nf + no + 1.0))
    for nm, back, fwd in (("past2", 2.0, 0.0), ("fut2", 0.0, 2.0)):
        bk, fw = int(round(back * s1)), int(round(fwd * s1))
        add(f"t_nf_{nm}", _roll(ind_f, bk, fw, "sum")[a])
        add(f"t_no_{nm}", _roll(ind_o, bk, fw, "sum")[a])

    # break strength near the sample, and who was near the strongest break
    z = np.nan_to_num(det["z"], nan=0.0)
    dvv = np.nan_to_num(det["dv"], nan=0.0)
    add("t_zmax_cell", cell(z, "max"))
    add("t_dvmax_cell", cell(dvv, "max"))
    add("t_zmax_0.5s", win(z, 0.5, "max"))
    ev_z = np.zeros(n)
    if len(t_ev):
        ev_z[t_ev] = z[t_ev]
    w05 = int(round(0.5 * s1))
    # strongest event inside +-0.5 s: pick by rolling max of the event z, then look it up
    zmax_ev = _roll(np.where(ev_z > 0, ev_z, np.nan), w05, w05, "max")[a]
    sel_sign, sel_df, sel_do = np.zeros(len(a)), np.full(len(a), np.nan), np.full(len(a), np.nan)
    if len(t_ev):
        lo = np.searchsorted(t_ev, a - w05, side="left")
        hi = np.searchsorted(t_ev, a + w05, side="right")
        for r in np.flatnonzero(hi > lo):
            k = lo[r] + int(np.argmax(z[t_ev[lo[r]:hi[r]]]))
            sel_sign[r], sel_df[r], sel_do[r] = sign[k], d_f_ev[k], d_o_ev[k]
    add("t_evz_0.5s", np.nan_to_num(zmax_ev, nan=0.0))
    add("t_ev_team_0.5s", sel_sign)
    add("t_ev_df_0.5s", sel_df)
    add("t_ev_do_0.5s", sel_do)

    # closest approach of each team to the (smoothed) ball between work-grid samples
    add("t_dminf_cell", cell(d_f, "min"))
    add("t_dmino_cell", cell(d_o, "min"))
    with np.errstate(invalid="ignore"):
        closer = np.where(np.isfinite(d_f) & np.isfinite(d_o), (d_f < d_o).astype(float), np.nan)
    add("t_closer_cell", cell(closer, "mean"))
    add("t_dminf_0.5s", win(d_f, 0.5, "min"))
    add("t_dmino_0.5s", win(d_o, 0.5, "min"))
    add("t_closer_1s", win(closer, 1.0, "mean"))

    # control margin at the native rate: who is nearer the smoothed ball, and how clearly
    with np.errstate(invalid="ignore"):
        margin = np.where(np.isfinite(d_f) & np.isfinite(d_o), np.clip(d_o - d_f, -8.0, 8.0), np.nan)
        tight_f = np.where(np.isfinite(d_f), (d_f < 2.0).astype(float), np.nan)
        tight_o = np.where(np.isfinite(d_o), (d_o < 2.0).astype(float), np.nan)
    add("t_margin_cell", cell(margin, "mean"))
    add("t_margin_0.5s", win(margin, 0.5, "mean"))
    add("t_margin_1s", win(margin, 1.0, "mean"))
    w1 = int(round(1.0 * s1))
    add("t_margin_past1", _roll(margin, w1, 0, "mean")[a])
    add("t_margin_fut1", _roll(margin, 0, w1, "mean")[a])
    add("t_tightf_0.5s", win(tight_f, 0.5, "mean"))
    add("t_tighto_0.5s", win(tight_o, 0.5, "mean"))

    # ball speed, flight, visibility, jitter
    sp = det["speed"]
    add("t_speed", sp[a])
    add("t_speed_max_cell", cell(sp, "max"))
    with np.errstate(invalid="ignore"):
        dmin = np.fmin(d_f, d_o)
        flight = np.where(np.isfinite(sp) & np.isfinite(dmin),
                          ((sp > FLIGHT_SPEED) & (dmin > FLIGHT_CLEAR)).astype(float), np.nan)
    add("t_flight_cell", cell(flight, "mean"))
    add("t_flight_1s", win(flight, 1.0, "mean"))
    add("t_vis_cell", cell(det["vis"].astype(float), "mean"))
    add("t_sigma", det["sigma"][a])
    return names, cols


def count_touches(m: Match, nb: NativeBall) -> dict:
    """Aggregate event counts over the halves of ``m`` (for logging on the real match: counts only)."""
    out = {"events": 0, "identified": 0, "minutes": 0.0}
    step = nb.step
    for h, a5, b5 in m.half_blocks:
        j_lo = max(0, a5 * step - step // 2)
        j_hi = min(nb.n - 1, b5 * step + (step - 1) // 2)
        fn = lambda fr, _lo=j_lo: players_at(m.P, (fr + _lo) / float(step))   # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            det = detect_touches(nb.ball[j_lo:j_hi + 1], nb.fps, fn, m.team)
        out["events"] += int(len(det["t"]))
        out["identified"] += int((det["who"] >= 0).sum())
        out["minutes"] += (j_hi - j_lo + 1) / nb.fps / 60.0
    return out


def touch_features(m: Match, nb: NativeBall, focals=(0, 1)) -> tuple[dict, list, np.ndarray]:
    """Touch features of the work-grid match ``m`` (players; usually corrupted) and its native ball.

    Returns ({focal: float32[T, F]}, names, idx) with rows on the same in-play samples, in the same
    order, as ``katena.featurize.build.featurize(m, ...)``."""
    step = nb.step
    if abs(nb.fps / step - m.fps) > 1e-6:
        raise ValueError(f"native fps {nb.fps} / step {step} != work-grid fps {m.fps}")
    if -(-nb.n // step) != m.n:
        raise ValueError("native ball length does not match the work-grid match")
    out = {f: [] for f in focals}
    names = None
    P5 = m.P
    for h, a5, b5 in m.half_blocks:
        j_lo = max(0, a5 * step - step // 2)
        j_hi = min(nb.n - 1, b5 * step + (step - 1) // 2)
        b = nb.ball[j_lo:j_hi + 1]
        fn = lambda fr, _lo=j_lo: players_at(P5, (fr + _lo) / float(step))   # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            det = detect_touches(b, nb.fps, fn, m.team)
            dh, da = _nearest_per_frame(det["ball_s"], det["vis"], fn, m.team)
            anchors = np.arange(a5, b5 + 1) * step - j_lo
            for f in focals:
                nm, cols = _half_features(det, dh, da, f, nb.fps, anchors, step)
                out[f].append(np.stack(cols, 1))
                names = nm
    idx = np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])
    X = {f: np.concatenate(v, 0).astype(np.float32) for f, v in out.items()}
    return X, names, idx
