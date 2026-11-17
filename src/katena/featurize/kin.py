"""Ball-free timing columns: what the players do in the second before and the second after a sample.

Candidate KIN (decision log, 2026-11-13). The base groups describe team movement with windows centred on the
sample (``vx_f_2s`` is the displacement from 2 s before to 2 s after), with no acceleration and no rate of the
lines or of the spread. A tree model reads one row at a time, so it cannot difference neighbouring rows, and a
centred window says the same thing at a change of owner as half a second either side of it. These columns are
one-sided: each quantity is measured over the window that ends at the sample ("b", back) and over the window that
starts at it ("a", ahead), and their difference is the reversal at the sample.

No ball position is read. Everything is in the focal team's frame (it attacks +x), as in ``build.py``, so the
columns for focal team 1 are those of focal team 0 with the teams swapped.

* ``kin_vx_{f,o}_{b,a}{1s,2s}``: mean over a team's slots of the slot's own displacement along x, in m/s. A slot
  counts only where it is seen at both ends and did not switch player in between (``_jumps``), so a fragment
  that starts, ends or is taken over by another player is not movement.
* ``kin_vx_{f,o,all}_rev{1s,2s}``: ahead minus back; ``all`` is the mean of the two teams (both teams move towards
  the same goal when the ball changes owner).
* ``kin_fwd_{f,o}_{b,a}1s``: share of the team's slots moving towards +x faster than ``V_MOVE`` minus the share
  moving towards -x faster than it.
* ``kin_rear_f_*``, ``kin_front_o_*`` (1 s): rate of each team's defensive line (the 20th percentile of the focal
  team's x, the 80th of the opponent's).
* ``kin_span_{f,o}_*``, ``kin_width_{f,o}_*`` (2 s): rate of the team's length (80th minus 20th percentile of x)
  and of its width (standard deviation of y).
* ``kin_pair_min``, ``kin_pair_min_{b,a}1s``: distance of the closest pair of opposing players, and its mean over
  the second before and the second after.
* ``kin_spd_{f,o}_acc1s``: the team's mean speed over the second after minus the second before.

Player positions carry jitter (0.3 m a sample in the ``target`` world), so slot positions and the team series are
averaged over ``SMOOTH`` samples either side before they are differenced, and nothing here is per player.
"""

from __future__ import annotations

import warnings

import numpy as np

from katena.featurize.build import infer_dirs, wmean
from katena.types import Match

GROUP = "kin"
SMOOTH = 1            # samples either side averaged before differencing (0.6 s at 5 Hz)
V_MOVE = 1.0          # m/s: a slot faster than this along x counts as moving forward or backward
V_MAX = 12.5          # m/s: faster slot movement is a fragment switching player, as in build.py
JUMP_M = 5.0          # m: a slot that moves further than this between two sightings, faster than V_MAX, switched player
MIN_SLOTS = 3         # a team quantity needs this many slots


def _smooth_slots(x: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware mean of every slot over samples [t-k, t+k]; needs two finite samples (one when ``k`` is 0)."""
    if k == 0:
        return x.copy()
    v = np.isfinite(x)
    n = len(x)
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(np.where(v, x, 0.0), axis=0)])
    cn = np.concatenate([np.zeros((1, x.shape[1]), dtype=np.int64), np.cumsum(v, axis=0)])
    i = np.arange(n)
    lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, n)
    c = cn[hi] - cn[lo]
    out = np.full(x.shape, np.nan)
    ok = c >= 2
    out[ok] = (cs[hi] - cs[lo])[ok] / c[ok]
    return out


def _sides(x: np.ndarray, w: int, dt: float) -> tuple:
    """(back, ahead) rates of ``x`` along axis 0 over ``w`` samples: back[t] = (x[t] - x[t-w]) / (w dt),
    ahead[t] = (x[t+w] - x[t]) / (w dt). NaN where the window leaves the array."""
    back, ahead = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
    if len(x) > w:
        d = (x[w:] - x[:-w]) / (w * dt)
        back[w:], ahead[:-w] = d, d
    return back, ahead


def _jumps(p: np.ndarray, dt: float) -> np.ndarray:
    """int64[n + 1, S]: running count, per slot, of sightings that lie more than ``JUMP_M`` from the slot's
    previous sighting at an implied speed above ``V_MAX`` (the slot now follows another player). The count of
    jumps at samples (a, b] is ``out[b + 1] - out[a + 1]``."""
    n, S = p.shape[:2]
    fin = np.isfinite(p).all(-1)
    last = np.maximum.accumulate(np.where(fin, np.arange(n)[:, None], -1), axis=0)
    prev = np.concatenate([np.full((1, S), -1, dtype=last.dtype), last[:-1]])
    ok = fin & (prev >= 0)
    cols = np.broadcast_to(np.arange(S), (n, S))
    was = p[np.maximum(prev, 0), cols]
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(p - was, axis=-1)
        el = (np.arange(n)[:, None] - prev) * dt
        jump = ok & (dist > JUMP_M) & (dist > V_MAX * el)
    return np.concatenate([np.zeros((1, S), dtype=np.int64), np.cumsum(jump, axis=0)])


def _no_jump(cj: np.ndarray, w: int, k: int) -> tuple:
    """(back, ahead) masks [n, S]: no jump inside the samples a one-sided rate over ``w`` samples reads
    (its two ends, each smoothed over ``k`` samples either side)."""
    n = len(cj) - 1
    i = np.arange(n)
    lo_b, hi_b = np.clip(i - w - k, 0, n - 1), np.clip(i + k, 0, n - 1)
    lo_a, hi_a = np.clip(i - k, 0, n - 1), np.clip(i + w + k, 0, n - 1)
    return cj[hi_b + 1] - cj[lo_b + 1] == 0, cj[hi_a + 1] - cj[lo_a + 1] == 0


def _team_mean(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    c = ok.sum(1)
    return np.where(c >= MIN_SLOTS, np.where(ok, v, 0.0).sum(1) / np.maximum(c, 1), np.nan)


def _fwd_share(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    c = ok.sum(1)
    with np.errstate(invalid="ignore"):
        net = (v > V_MOVE).sum(1) - (v < -V_MOVE).sum(1)
    return np.where(c >= MIN_SLOTS, net / np.maximum(c, 1), np.nan)


def _nanq(x: np.ndarray, q: float) -> np.ndarray:
    """``np.nanpercentile(x, q, axis=1)`` (linear interpolation) without its per-row Python call; NaN for a row
    with no finite value."""
    xs = np.sort(x, axis=1)                                    # NaN sorts last
    c = np.isfinite(xs).sum(1)
    pos = np.maximum(c - 1, 0) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(c - 1, 0))
    rows = np.arange(len(xs))
    if xs.shape[1] == 0:
        return np.full(len(xs), np.nan)
    a, b = xs[rows, lo], xs[rows, hi]
    return np.where(c > 0, a + (b - a) * (pos - lo), np.nan)


def _half_cols(P: np.ndarray, team: np.ndarray, fps: float, pitch: tuple, focal: int, home_dir: int) -> tuple:
    """(columns float32[n, F], names) of one half in the focal team's frame."""
    n = P.shape[0]
    L, W = pitch
    dt = 1.0 / fps
    s1 = max(1, int(round(fps)))
    P = P.astype(np.float64).copy()
    if (home_dir if focal == 0 else -home_dir) < 0:
        P[..., 0] = L - P[..., 0]
        P[..., 1] = W - P[..., 1]
    is_f = team == focal
    pf, po = P[:, is_f], P[:, ~is_f]
    data, names = [], []

    def add(name, arr):
        data.append(np.asarray(arr, dtype=np.float32))
        names.append(f"kin_{name}")

    # per-slot displacement along x, one-sided
    xs_f, xs_o = _smooth_slots(pf[..., 0], SMOOTH), _smooth_slots(po[..., 0], SMOOTH)
    cj_f, cj_o = _jumps(pf, dt), _jumps(po, dt)
    for lab, w in (("1s", s1), ("2s", 2 * s1)):
        side = {}
        for tm, xs, cj in (("f", xs_f, cj_f), ("o", xs_o, cj_o)):
            vb, va = _sides(xs, w, dt)
            for v, clean in zip((vb, va), _no_jump(cj, w, SMOOTH)):
                with np.errstate(invalid="ignore"):
                    v[(np.abs(v) > V_MAX) | ~clean] = np.nan
            side[tm] = (vb, va)
        mean = {tm: (_team_mean(vb), _team_mean(va)) for tm, (vb, va) in side.items()}
        for tm in ("f", "o"):
            add(f"vx_{tm}_b{lab}", mean[tm][0])
            add(f"vx_{tm}_a{lab}", mean[tm][1])
        rev = {tm: mean[tm][1] - mean[tm][0] for tm in ("f", "o")}
        add(f"vx_f_rev{lab}", rev["f"])
        add(f"vx_o_rev{lab}", rev["o"])
        add(f"vx_all_rev{lab}", 0.5 * (rev["f"] + rev["o"]))
        if lab == "1s":
            for tm in ("f", "o"):
                add(f"fwd_{tm}_b1s", _fwd_share(side[tm][0]))
                add(f"fwd_{tm}_a1s", _fwd_share(side[tm][1]))

    # lines and spread: team series, smoothed, then one-sided rates
    def series(p):
        x, y = p[..., 0], p[..., 1]
        enough = np.isfinite(x).sum(1) >= MIN_SLOTS
        xe = np.where(enough[:, None], x, np.nan)
        q20, q80 = _nanq(xe, 20.0), _nanq(xe, 80.0)
        width = np.where(enough, np.nanstd(np.where(enough[:, None], y, np.nan), axis=1), np.nan)
        return q20, q80, width

    rear_f, front_f, width_f = series(pf)
    rear_o, front_o, width_o = series(po)
    sm = lambda a: wmean(a, SMOOTH, SMOOTH)  # noqa: E731

    def rates(name, arr, lab, w):
        b, a = _sides(sm(arr), w, dt)
        add(f"{name}_b{lab}", b)
        add(f"{name}_a{lab}", a)

    rates("rear_f", rear_f, "1s", s1)
    rates("front_o", front_o, "1s", s1)
    for name, arr in (("span_f", front_f - rear_f), ("span_o", front_o - rear_o), ("width_f", width_f),
                      ("width_o", width_o)):
        rates(name, arr, "2s", 2 * s1)

    # closest opposing pair
    pair = np.full(n, np.nan)
    for s in range(0, n, 4000):
        e = min(n, s + 4000)
        dd = np.linalg.norm(pf[s:e, :, None, :] - po[s:e, None, :, :], axis=-1)
        dd = np.where(np.isfinite(dd), dd, np.inf).reshape(e - s, -1)
        mn = dd.min(1) if dd.shape[1] else np.full(e - s, np.inf)
        pair[s:e] = np.where(np.isfinite(mn), mn, np.nan)
    add("pair_min", sm(pair))
    add("pair_min_b1s", wmean(pair, s1, 0))
    add("pair_min_a1s", wmean(pair, 0, s1))

    # mean speed of each team (central difference over +-2 samples, as build.py), after minus before
    vel = np.full_like(P, np.nan)
    k = 2
    if n > 2 * k:
        vel[k:-k] = (P[2 * k:] - P[:-2 * k]) / (2 * k * dt)
    spd = np.linalg.norm(vel, axis=-1)
    with np.errstate(invalid="ignore"):
        spd[spd > V_MAX] = np.nan
    for tm, sel in (("f", is_f), ("o", ~is_f)):
        ms = _team_mean(spd[:, sel])
        add(f"spd_{tm}_acc1s", wmean(ms, 0, s1) - wmean(ms, s1, 0))
    return np.stack(data, 1), names


def kin_features(m: Match, focals=(0, 1), dirs: dict | None = None) -> tuple[dict, list, np.ndarray]:
    """({focal: float32[T, F]}, column names, work-grid sample index of every row) for the in-play samples of
    ``m``, in the row order of ``build.featurize``. Each half is computed by itself."""
    dirs = dirs if dirs is not None else infer_dirs(m)
    out, names = {}, []
    for focal in focals:
        mats = []
        for h, a, b in m.half_blocks:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows are expected
                X, names = _half_cols(m.P[a:b + 1], m.team, m.fps, m.pitch, focal, dirs[h])
            mats.append(X)
        out[focal] = np.concatenate(mats, 0).astype(np.float32)
    idx = np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])
    return out, names, idx
