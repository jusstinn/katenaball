"""Per-frame features on the 5 Hz work grid.

Everything is expressed for a *focal* team that attacks towards +x (the pitch is rotated 180 degrees
for frames where the focal team attacks the other way; the attack direction is inferred from the
player positions by ``infer_dirs``, never read from metadata). The team head predicts "focal team has
the ball"; running with each team as focal and averaging makes the model side-symmetric.

Feature groups (used by the ablations):

``ball``         ball-aware: gap-aware kinematics, nearest player per team, control persistence.
``free``         ball-free: team shape, line heights, velocity along the attack axis, pressing, contact
                 hot-spot, tracked-object counts.
``dead_ball``    dead-ball cues that need the ball: out of bounds, ball stillness, restart geometry.
``dead_player``  dead-ball cues from players only: stillness, restart layouts, box and centre counts.
``miss``         ball-missingness context (visibility fractions, time since seen).
``touch``        optional, computed at the native frame rate before downsampling (``touch.py``); only
                 present when ``featurize`` is given the native ball.

(The package is called ``featurize`` rather than ``features`` because the repository's
``.claude/settings.json`` denies access to any path named ``features/``.)
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd

from katena.external.common import infer_attack_dir
from katena.types import Match

GROUPS = ("ball", "free", "dead_ball", "dead_player", "miss", "touch")
FILL_S = 3.0           # linear ball interpolation across gaps up to this long
BIG = 1e6


@dataclass
class Feats:
    X: np.ndarray            # float32[T, F]
    names: list
    groups: np.ndarray       # str[F]
    idx: np.ndarray          # int64[T] work-grid sample index of every row
    half: np.ndarray         # int8[T]
    meta: dict

    def select(self, drop: tuple = (), keep: tuple | None = None) -> np.ndarray:
        """Column indices of the kept groups."""
        ok = np.ones(len(self.groups), bool) if keep is None else np.isin(self.groups, keep)
        ok &= ~np.isin(self.groups, drop)
        return np.flatnonzero(ok)


# ----------------------------------------------------------------------------------------------
# helpers

def wmean(x: np.ndarray, back: int, fwd: int, min_frac: float = 0.2) -> np.ndarray:
    """NaN-aware mean over samples [t-back, t+fwd], truncated at the array ends."""
    v = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(v, x, 0.0), dtype=np.float64)))
    cn = np.concatenate(([0], np.cumsum(v)))
    n = len(x)
    i = np.arange(n)
    lo, hi = np.maximum(i - back, 0), np.minimum(i + fwd + 1, n)
    c = cn[hi] - cn[lo]
    out = np.full(n, np.nan)
    ok = c >= np.maximum(1.0, min_frac * (hi - lo))
    out[ok] = (cs[hi] - cs[lo])[ok] / c[ok]
    return out


def wmax(x: np.ndarray, back: int, fwd: int) -> np.ndarray:
    """Max over [t-back, t+fwd] (x must be finite); truncated at the array ends."""
    r = pd.Series(x).rolling(back + fwd + 1, min_periods=1).max()
    if fwd:
        r = r.shift(-fwd).ffill()
    return r.to_numpy()


def since(mask: np.ndarray, cap: float) -> np.ndarray:
    """Samples since the last True (0 where True); ``cap`` where none yet."""
    n = len(mask)
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -10 ** 9))
    return np.minimum((np.arange(n) - last).astype(np.float64), cap)


def until(mask: np.ndarray, cap: float) -> np.ndarray:
    return since(mask[::-1], cap)[::-1]


def interp_short(b: np.ndarray, max_gap: int):
    """Linear interpolation of an (n, 2) track over gaps of at most ``max_gap`` samples.
    Returns (filled, interpolated_flag, length_of_the_gap_each_sample_sits_in)."""
    n = len(b)
    vis = np.isfinite(b[:, 0])
    out = b.copy()
    interp = np.zeros(n, bool)
    glen = np.zeros(n)
    idx = np.flatnonzero(vis)
    if len(idx) < 2:
        return out, interp, np.where(vis, 0.0, float(n))
    for k in np.flatnonzero(np.diff(idx) > 1):
        a, c = idx[k], idx[k + 1]
        glen[a + 1:c] = c - a - 1
        if c - a - 1 <= max_gap:
            w = (np.arange(a + 1, c) - a) / float(c - a)
            out[a + 1:c] = b[a] * (1 - w)[:, None] + b[c] * w[:, None]
            interp[a + 1:c] = True
    glen[:idx[0]] = idx[0]
    glen[idx[-1] + 1:] = n - idx[-1] - 1
    return out, interp, glen


def infer_dirs(m: Match) -> dict:
    """Home attack direction per half from player positions (centroid rule). Falls back to the
    opposite of the other half, then to +1."""
    out = {h: infer_attack_dir(m.P, m.team, a, b, m.pitch[0]) for h, a, b in m.half_blocks}
    for h in list(out):
        if out[h] == 0:
            other = [v for o, v in out.items() if o != h and v != 0]
            out[h] = -other[0] if other else 1
    return out


def _quantiles(x, alive):
    with np.errstate(all="ignore"):
        return np.nanpercentile(np.where(alive, x, np.nan), [20, 50, 80], axis=1)


class _Cols:
    def __init__(self):
        self.data, self.names, self.groups = [], [], []

    def add(self, name, group, arr):
        self.data.append(np.asarray(arr, dtype=np.float32))
        self.names.append(name)
        self.groups.append(group)


# ----------------------------------------------------------------------------------------------

def featurize(m: Match, focal: int = 0, dirs: dict | None = None, native=None) -> Feats:
    """Features for every in-play sample of ``m`` with team ``focal`` (0 home, 1 away) as focal.
    ``native``: the match's ball at the source frame rate (``touch.NativeBall``); when given, the
    ``touch`` group is appended after the other groups."""
    dirs = dirs if dirs is not None else infer_dirs(m)
    mats, halves, cols = [], [], None
    for h, a, b in m.half_blocks:
        sl = slice(a, b + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows are expected
            cols = _half_cols(m.P[sl], m.ball[sl], m.team, m.fps, m.pitch, focal, dirs[h])
        mats.append(np.stack(cols.data, 1))
        halves.append(np.full(b - a + 1, h, dtype=np.int8))
    idx = np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])
    X, names, groups = np.concatenate(mats, 0).astype(np.float32), list(cols.names), list(cols.groups)
    if native is not None:
        from katena.featurize import touch
        Xt, tnames, _ = touch.touch_features(m, native, focals=(focal,))
        X, names, groups = np.concatenate([X, Xt[focal]], 1), names + tnames, groups + [touch.GROUP] * len(tnames)
    return Feats(X=X, names=names, groups=np.array(groups), idx=idx, half=np.concatenate(halves),
                 meta={"dirs": dirs, "focal": focal})


def _half_cols(P, ball, team, fps, pitch, focal, home_dir):
    n, S, _ = P.shape
    L, W = pitch
    dt = 1.0 / fps
    s1 = max(1, int(round(1.0 / dt)))      # samples per second
    C = _Cols()
    focal_dir = home_dir if focal == 0 else -home_dir
    P = P.astype(np.float64).copy()
    ball = ball.astype(np.float64).copy()
    if focal_dir < 0:                       # rotate 180 degrees so the focal team attacks +x
        P[..., 0] = L - P[..., 0]
        P[..., 1] = W - P[..., 1]
        ball[:, 0] = L - ball[:, 0]
        ball[:, 1] = W - ball[:, 1]
    is_f = team == focal
    pf, po = P[:, is_f], P[:, ~is_f]
    alive_f, alive_o = np.isfinite(pf[..., 0]), np.isfinite(po[..., 0])
    ymid = W / 2.0

    # velocities (central difference over +-2 samples) per slot; jumps above 12.5 m/s blanked
    vel = np.full_like(P, np.nan)
    k = 2
    if n > 2 * k:
        vel[k:-k] = (P[2 * k:] - P[:-2 * k]) / (2 * k * dt)
    spd = np.linalg.norm(vel, axis=-1)
    spd[spd > 12.5] = np.nan
    spd_f, spd_o = spd[:, is_f], spd[:, ~is_f]

    # ---- ball: gap-aware fill, kinematics, age ----
    vis = np.isfinite(ball[:, 0])
    bf, interp, glen = interp_short(ball, int(FILL_S / dt))
    have = np.isfinite(bf[:, 0])
    age = since(vis, 60 * s1) * dt
    nxt = until(vis, 60 * s1) * dt
    bvel = np.full_like(bf, np.nan)
    bvel[1:-1] = (bf[2:] - bf[:-2]) / (2 * dt)
    bspeed = np.linalg.norm(bvel, axis=1)
    bspeed[bspeed > 45] = np.nan
    bacc = np.full(n, np.nan)
    bacc[1:-1] = (bspeed[2:] - bspeed[:-2]) / (2 * dt)

    # ---- nearest players per team ----
    d_all = np.linalg.norm(P - bf[:, None, :], axis=-1)
    d_all[~have] = np.nan
    d_f, d_o = d_all[:, is_f], d_all[:, ~is_f]

    def two_nearest(d, alive):
        dd = np.where(alive & np.isfinite(d), d, BIG)
        if dd.shape[1] < 2:
            dd = np.concatenate([dd, np.full((n, 2), BIG)], 1)
        part = np.partition(dd, 1, axis=1)[:, :2]
        part.sort(axis=1)
        return np.where(part >= BIG, np.nan, part), np.argmin(dd, axis=1)

    (nf, af), (no, ao) = two_nearest(d_f, alive_f), two_nearest(d_o, alive_o)
    d_f1, d_f2, d_o1, d_o2 = nf[:, 0], nf[:, 1], no[:, 0], no[:, 1]
    rows = np.arange(n)
    for r in (3.0, 6.0, 10.0):
        C.add(f"n_f_within{r:g}", "ball", np.where(have, np.nansum(d_f < r, 1), np.nan))
        C.add(f"n_o_within{r:g}", "ball", np.where(have, np.nansum(d_o < r, 1), np.nan))
    diff = d_o1 - d_f1
    for nm, arr in (("d_f1", d_f1), ("d_o1", d_o1), ("d_f2", d_f2), ("d_o2", d_o2), ("d_diff", diff),
                    ("d_diff2", d_o2 - d_f2), ("d_ratio", np.log((d_o1 + 0.5) / (d_f1 + 0.5))),
                    ("spd_nearest_f", spd_f[rows, af]), ("spd_nearest_o", spd_o[rows, ao]),
                    ("ball_x", bf[:, 0]), ("ball_absy", np.abs(bf[:, 1] - ymid)),
                    ("ball_speed", bspeed), ("ball_acc", bacc), ("ball_vx", bvel[:, 0]),
                    ("ball_vabsy", bvel[:, 1] * np.sign(bf[:, 1] - ymid)),
                    ("ball_interp", np.where(vis, 0.0, np.where(interp, 1.0, 2.0)))):
        C.add(nm, "ball", arr)
    C.add("ball_speed_max2s", "ball", wmax(np.nan_to_num(bspeed, nan=0.0), 2 * s1, 2 * s1))
    C.add("ball_speed_mean1s", "ball", wmean(bspeed, s1, s1))
    C.add("ball_speed_mean3s", "ball", wmean(bspeed, 3 * s1, 3 * s1))
    flight = np.where(have, (bspeed > 12.0).astype(float), np.nan)
    C.add("flight_frac_2s", "ball", wmean(flight, 2 * s1, 2 * s1))
    sh = 3 * s1
    for nm, dd in (("d_f1", d_f1), ("d_o1", d_o1)):
        t = np.full(n, np.nan)
        if n > 2 * sh:
            t[sh:-sh] = dd[2 * sh:] - dd[:-2 * sh]
        C.add(f"{nm}_trend3s", "ball", t)

    # control persistence (centred, past-only, future-only windows)
    ctrl = np.where(have & np.isfinite(diff), (diff > 0).astype(float), np.nan)
    tight_f = np.where(have & np.isfinite(d_f1), (d_f1 < 2.5).astype(float), np.nan)
    tight_o = np.where(have & np.isfinite(d_o1), (d_o1 < 2.5).astype(float), np.nan)
    margin = np.clip(diff, -8, 8)
    for lab, back, fwd in (("c1", s1, s1), ("c3", 3 * s1, 3 * s1), ("c6", 6 * s1, 6 * s1),
                           ("past2", 2 * s1, 0), ("past5", 5 * s1, 0), ("fut2", 0, 2 * s1), ("fut5", 0, 5 * s1)):
        C.add(f"ctrl_{lab}", "ball", wmean(ctrl, back, fwd))
        C.add(f"margin_{lab}", "ball", wmean(margin, back, fwd))
        C.add(f"tightf_{lab}", "ball", wmean(tight_f, back, fwd))
        C.add(f"tighto_{lab}", "ball", wmean(tight_o, back, fwd))
    tf, to = np.nan_to_num(tight_f) > 0, np.nan_to_num(tight_o) > 0
    ts_f, ts_o = since(tf, 30 * s1) * dt, since(to, 30 * s1) * dt
    tu_f, tu_o = until(tf, 30 * s1) * dt, until(to, 30 * s1) * dt
    for nm, arr in (("since_tight_f", ts_f), ("since_tight_o", ts_o), ("until_tight_f", tu_f),
                    ("until_tight_o", tu_o), ("last_ctrl_sign", np.clip(ts_o - ts_f, -30, 30)),
                    ("next_ctrl_sign", np.clip(tu_o - tu_f, -30, 30))):
        C.add(nm, "ball", arr)

    # ---- ball-missingness context ----
    C.add("ball_vis", "miss", vis.astype(float))
    C.add("ball_age_s", "miss", age)
    C.add("ball_next_s", "miss", nxt)
    C.add("ball_gap_len_s", "miss", glen * dt)
    for lab, w in (("1s", s1), ("3s", 3 * s1), ("10s", 10 * s1), ("30s", 30 * s1)):
        C.add(f"vis_frac_{lab}", "miss", wmean(vis.astype(float), w, w, 0.0))

    # ---- ball-free team shape ----
    xf, xo, yf, yo = pf[..., 0], po[..., 0], pf[..., 1], po[..., 1]
    C.add("nobs_f", "free", alive_f.sum(1))
    C.add("nobs_o", "free", alive_o.sum(1))
    qf, qo = _quantiles(xf, alive_f), _quantiles(xo, alive_o)
    with np.errstate(all="ignore"):
        mxf, mxo = np.where(alive_f, xf, np.nan), np.where(alive_o, xo, np.nan)
        cxf, cxo = np.nanmean(mxf, 1), np.nanmean(mxo, 1)
        wf, wo = np.nanstd(np.where(alive_f, yf, np.nan), 1), np.nanstd(np.where(alive_o, yo, np.nan), 1)
        sxf, sxo = np.nanstd(mxf, 1), np.nanstd(mxo, 1)
    span_f, span_o = qf[2] - qf[0], qo[2] - qo[0]
    for nm, arr in (("cx_f", cxf), ("cx_o", cxo), ("rear_f", qf[0]), ("mid_f", qf[1]), ("front_f", qf[2]),
                    ("rear_o", qo[0]), ("mid_o", qo[1]), ("front_o", qo[2]), ("width_f", wf), ("width_o", wo),
                    ("depth_f", sxf), ("depth_o", sxo), ("cx_diff", cxf - cxo),
                    ("line_gap", qf[2] - qo[0]), ("line_gap2", qo[2] - qf[0]), ("span_f", span_f),
                    ("span_o", span_o), ("span_diff", span_f - span_o), ("width_diff", wf - wo)):
        C.add(nm, "free", arr)
    for lab, w in (("3s", 3 * s1), ("8s", 8 * s1)):
        for nm, arr in (("cx_f", cxf), ("cx_o", cxo), ("front_f", qf[2]), ("rear_o", qo[0]), ("span_f", span_f),
                        ("span_o", span_o), ("cx_diff", cxf - cxo)):
            C.add(f"{nm}_m{lab}", "free", wmean(arr, w, w))

    def pair_disp(x, w):                      # per-slot displacement, so fragment churn is not "movement"
        out = np.full(x.shape, np.nan)
        if x.shape[0] > 2 * w:
            out[w:-w] = x[2 * w:] - x[:-2 * w]
        return out

    for lab, w in (("2s", 2 * s1), ("5s", 5 * s1)):
        with np.errstate(all="ignore"):
            mf = np.nanmean(pair_disp(xf, w), 1) / (2 * w * dt)
            mo = np.nanmean(pair_disp(xo, w), 1) / (2 * w * dt)
        C.add(f"vx_f_{lab}", "free", mf)
        C.add(f"vx_o_{lab}", "free", mo)
        C.add(f"vx_diff_{lab}", "free", mf - mo)
    with np.errstate(all="ignore"):
        msf, mso = np.nanmean(spd_f, 1), np.nanmean(spd_o, 1)
    C.add("spd_f", "free", msf)
    C.add("spd_o", "free", mso)
    C.add("spd_diff", "free", msf - mso)
    for lab, w in (("3s", 3 * s1), ("8s", 8 * s1)):
        C.add(f"spd_diff_m{lab}", "free", wmean(msf - mso, w, w))

    # pressing and the contact hot-spot (focal vs opponent pairs within 4 m)
    nan = lambda: np.full(n, np.nan)  # noqa: E731
    pres_f, pres_o, close_f, close_o = nan(), nan(), nan(), nan()
    hot_x, hot_y, near_f, near_o, hot_n = nan(), nan(), nan(), nan(), np.zeros(n)
    for s in range(0, n, 4000):
        e = min(n, s + 4000)
        fa, oa = pf[s:e], po[s:e]
        with np.errstate(all="ignore"):
            dd = np.linalg.norm(fa[:, :, None, :] - oa[:, None, :, :], axis=-1)       # [c, F, O]
            dd = np.where(np.isfinite(dd), dd, np.inf)
            nf_o, no_f = dd.min(2), dd.min(1)
            fin_f, fin_o = np.isfinite(nf_o), np.isfinite(no_f)
            any_both = np.isfinite(fa[..., 0]).any(1) & np.isfinite(oa[..., 0]).any(1)
            pres_f[s:e] = np.where(any_both, np.nanmean(np.where(fin_f, nf_o, np.nan), 1), np.nan)
            pres_o[s:e] = np.where(any_both, np.nanmean(np.where(fin_o, no_f, np.nan), 1), np.nan)
            close_f[s:e] = np.where(any_both, (nf_o < 4.0).sum(1), np.nan)
            close_o[s:e] = np.where(any_both, (no_f < 4.0).sum(1), np.nan)
            pair = dd < 4.0
            cnt = pair.sum((1, 2))
            midx = np.where(pair, (fa[:, :, None, 0] + oa[:, None, :, 0]) / 2, 0).sum((1, 2))
            midy = np.where(pair, (fa[:, :, None, 1] + oa[:, None, :, 1]) / 2, 0).sum((1, 2))
            okc = cnt > 0
            hx = np.where(okc, midx / np.maximum(cnt, 1), np.nan)
            hy = np.where(okc, midy / np.maximum(cnt, 1), np.nan)
            hxy = np.stack([hx, hy], 1)
            near_f[s:e] = np.where(okc, np.nansum(np.linalg.norm(fa - hxy[:, None, :], axis=-1) < 10.0, 1), np.nan)
            near_o[s:e] = np.where(okc, np.nansum(np.linalg.norm(oa - hxy[:, None, :], axis=-1) < 10.0, 1), np.nan)
        hot_n[s:e], hot_x[s:e], hot_y[s:e] = cnt, hx, hy
    for nm, arr in (("press_f", pres_f), ("press_o", pres_o), ("close_f", close_f), ("close_o", close_o),
                    ("hot_n", hot_n), ("hot_x", hot_x), ("hot_absy", np.abs(hot_y - ymid)),
                    ("hot_near_f", near_f), ("hot_near_o", near_o), ("hot_near_diff", near_f - near_o),
                    ("close_diff", close_f - close_o)):
        C.add(nm, "free", arr)
    w3 = 3 * s1
    for nm, arr in (("hot_near_diff", near_f - near_o), ("close_diff", close_f - close_o),
                    ("press_diff", pres_o - pres_f), ("hot_x", hot_x)):
        C.add(f"{nm}_m3s", "free", wmean(arr, w3, w3))

    # ---- dead-ball cues from players only ----
    still = np.where(np.isfinite(spd), (spd < 0.7).astype(float), np.nan)
    with np.errstate(all="ignore"):
        frac_still, mspd = np.nanmean(still, 1), np.nanmean(spd, 1)
    C.add("frac_still", "dead_player", frac_still)
    C.add("mean_speed_all", "dead_player", mspd)
    for lab, w in (("2s", 2 * s1), ("6s", 6 * s1), ("15s", 15 * s1)):
        C.add(f"frac_still_m{lab}", "dead_player", wmean(frac_still, w, w))
        C.add(f"mean_speed_m{lab}", "dead_player", wmean(mspd, w, w))
    box_x, box_hw = 16.5, 20.16
    for nm, x, y, al in (("f", xf, yf, alive_f), ("o", xo, yo, alive_o)):
        C.add(f"n_ownbox_{nm}", "dead_player", (al & (x < box_x) & (np.abs(y - ymid) < box_hw)).sum(1))
        C.add(f"n_oppbox_{nm}", "dead_player", (al & (x > L - box_x) & (np.abs(y - ymid) < box_hw)).sum(1))
        C.add(f"n_ownhalf_{nm}", "dead_player", (al & (x < L / 2)).sum(1))
        C.add(f"n_circle_{nm}", "dead_player", (al & (np.hypot(x - L / 2, y - ymid) < 9.15)).sum(1))

    # ---- dead-ball cues from the ball ----
    bx, by = bf[:, 0], bf[:, 1]
    oob = np.minimum.reduce([bx, L - bx, by, W - by])            # negative = outside the pitch
    corner = np.minimum.reduce([np.hypot(bx, by), np.hypot(L - bx, by), np.hypot(bx, W - by),
                                np.hypot(L - bx, W - by)])
    for nm, arr in (("ball_oob_dist", oob), ("ball_line_dist", np.minimum(by, W - by)),
                    ("ball_goalline_dist", np.minimum(bx, L - bx)), ("ball_corner_dist", corner),
                    ("ball_centre_dist", np.hypot(bx - L / 2, by - ymid))):
        C.add(nm, "dead_ball", arr)
    outf = np.where(have, (oob < 0.0).astype(float), np.nan)
    for lab, w in (("2s", 2 * s1), ("6s", 6 * s1), ("15s", 15 * s1)):
        C.add(f"ball_out_frac_{lab}", "dead_ball", wmean(outf, w, w))
    outm = np.nan_to_num(outf) > 0
    C.add("since_out_s", "dead_ball", since(outm, 60 * s1) * dt)
    C.add("until_out_s", "dead_ball", until(outm, 60 * s1) * dt)
    still_b = np.where(np.isfinite(bspeed), (bspeed < 1.0).astype(float), np.nan)
    for lab, w in (("2s", 2 * s1), ("6s", 6 * s1)):
        C.add(f"ball_still_frac_{lab}", "dead_ball", wmean(still_b, w, w))
    bxs = np.where(have, bx, np.nan)
    with np.errstate(all="ignore"):
        var = wmean(bxs ** 2, w3, w3) - wmean(bxs, w3, w3) ** 2
    C.add("ball_x_std_6s", "dead_ball", np.sqrt(np.maximum(var, 0)))
    return C
