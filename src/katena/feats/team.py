"""Team-level, set-based features over whatever slots are present on each frame.

Slots are never treated as people. Per-slot quantities are used only over a few frames (gap-aware
velocity with a jump guard, so an identity swap reads as NaN rather than a sprint); everything
else is a per-frame statistic of the set of present slots of each team, after compaction (present
slots first) and de-duplication (a slot within `dedup_m` of a lower-index teammate is dropped).
"""
from __future__ import annotations

import numpy as np

from .temporal import gap_diff, run_bounds

BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16


class TeamSet:
    """Compacted present players of one team on the attack-normalised timeline.

    P: float32[N, K, 2] (NaN where absent), V: float32[N, K, 2] velocity, spd: float32[N, K],
    pres: bool[N, K]. `sign` is +1 for home and -1 for away: u = sign * x is the team's own
    advance coordinate (own goal at u = -L/2).
    """

    def __init__(self, P, V, spd, pres, sign):
        self.P, self.V, self.spd, self.pres, self.sign = P, V, spd, pres, sign

    @property
    def K(self) -> int:
        return self.P.shape[1]


def build_team(P: np.ndarray, speed: np.ndarray, seg_lo, seg_hi, fps: float, cfg: dict,
               sign: int) -> TeamSet:
    """P: float32[N, S_t, 2] attack-normalised slots of one team; speed: float32[N, S_t] or None."""
    N, S = P.shape[0], P.shape[1]
    if S == 0:
        z = np.full((N, 1, 2), np.nan, dtype=np.float32)
        return TeamSet(z, z.copy(), np.full((N, 1), np.nan, np.float32),
                       np.zeros((N, 1), dtype=bool), sign)
    pres = np.isfinite(P).all(axis=-1)
    rs, re = run_bounds(pres, seg_lo, seg_hi)
    V = gap_diff(P, pres, rs, re, int(cfg["vel_half_frames"]), fps)
    del rs, re
    vn = np.hypot(V[..., 0], V[..., 1])
    V[vn > cfg["player_vmax"]] = np.nan  # identity swaps and re-acquisitions, not running
    vn = np.hypot(V[..., 0], V[..., 1])
    spd = vn if speed is None else np.where(np.isfinite(speed), speed, vn).astype(np.float32)
    spd = np.where(pres, spd, np.nan).astype(np.float32)

    # compact: present slots first (stable, so the order among present slots is preserved)
    K = max(1, int(pres.sum(axis=1).max()))
    order = np.argsort(~pres, axis=1, kind="stable")[:, :K]
    P = np.take_along_axis(P, order[..., None], axis=1)
    V = np.take_along_axis(V, order[..., None], axis=1)
    spd = np.take_along_axis(spd, order, axis=1)
    pres = np.take_along_axis(pres, order, axis=1)

    if cfg["dedup_m"] > 0 and K > 1:
        dup = np.zeros_like(pres)
        upper = np.triu(np.ones((K, K), dtype=bool), k=1)
        c = int(cfg["chunk"])
        for a in range(0, N, c):  # chunked over frames, vectorised inside
            p = P[a:a + c]
            d = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
            dup[a:a + c] = ((d < cfg["dedup_m"]) & upper).any(axis=1)
        pres = pres & ~dup
        P[~pres] = np.nan
        V[~pres] = np.nan
        spd[~pres] = np.nan
    return TeamSet(P, V, spd, pres, sign)


def _masked_mean(v: np.ndarray, m: np.ndarray, min_count: int = 1) -> np.ndarray:
    m = m & np.isfinite(v)
    c = m.sum(axis=1)
    s = np.where(m, v, 0).sum(axis=1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < min_count] = np.nan
    return out.astype(np.float32)


def shape_features(ts: TeamSet, cfg: dict, pitch) -> tuple[dict, dict]:
    """Ball-free per-team shape columns (unprefixed core names) and aux (gk index, outfield mask)."""
    L = float(pitch[0])
    f32 = np.float32
    P, pres, sign = ts.P, ts.pres, ts.sign
    x, y = P[..., 0], P[..., 1]
    cnt = pres.sum(axis=1)
    out: dict[str, np.ndarray] = {"n": cnt.astype(f32)}

    cx = _masked_mean(x, pres)
    cy = _masked_mean(y, pres)
    out["cx"], out["cy"] = cx, cy
    with np.errstate(invalid="ignore"):
        out["sx"] = np.sqrt(_masked_mean((x - cx[:, None]) ** 2, pres, 2))
        out["sy"] = np.sqrt(_masked_mean((y - cy[:, None]) ** 2, pres, 2))
    out["dcent"] = _masked_mean(np.hypot(x - cx[:, None], y - cy[:, None]), pres, 2)

    # own advance coordinate: u = -L/2 at the own goal line
    u = np.where(pres, sign * x, np.nan)
    us = np.sort(u, axis=1)  # NaN sorts last
    gk_u = us[:, 0]
    line_u = us[:, 1] if us.shape[1] > 1 else np.full(len(us), np.nan, f32)
    top_u = np.take_along_axis(us, np.maximum(cnt - 1, 0)[:, None], axis=1)[:, 0]
    top_u = np.where(cnt > 0, top_u, np.nan)
    out["gk_x"] = (sign * gk_u).astype(f32)
    out["line_x"] = (sign * line_u).astype(f32)
    out["top_x"] = (sign * top_u).astype(f32)
    out["length"] = (top_u - line_u).astype(f32)

    gk_idx = np.argmin(np.where(pres, u, np.inf), axis=1)
    outfield = pres & (np.arange(ts.K)[None, :] != gk_idx[:, None])
    n_of = outfield.sum(axis=1)
    ymax = np.where(outfield, y, -np.inf).max(axis=1)
    ymin = np.where(outfield, y, np.inf).min(axis=1)
    width = np.where(n_of >= 2, ymax - ymin, np.nan).astype(f32)
    out["width"] = width
    out["area"] = (out["length"] * width).astype(f32)

    out["cvx"] = _masked_mean(ts.V[..., 0], pres)
    out["mspd"] = _masked_mean(ts.spd, pres)
    with np.errstate(invalid="ignore"):
        out["nsprint"] = (pres & (ts.spd > cfg["sprint_speed"])).sum(axis=1).astype(f32)
        out["nopp_half"] = (pres & (u > 0)).sum(axis=1).astype(f32)
        inbox_y = np.abs(y) < BOX_HALF_WIDTH
        c = f32(L / 2 - BOX_DEPTH)
        out["box_att"] = (pres & (u > c) & inbox_y).sum(axis=1).astype(f32)
        box_def = (pres & (u < -c) & inbox_y).sum(axis=1).astype(f32)
    aux = {"outfield": outfield, "box_def": box_def}
    return out, aux


def mark_distances(h: TeamSet, a: TeamSet, of_h: np.ndarray, of_a: np.ndarray, chunk: int):
    """Mean over each team's outfield players of the distance to the nearest opponent."""
    N = h.P.shape[0]
    mh = np.full(N, np.nan, np.float32)
    ma = np.full(N, np.nan, np.float32)
    for s in range(0, N, chunk):
        ph, pa = h.P[s:s + chunk], a.P[s:s + chunk]
        d = np.hypot(ph[:, :, None, 0] - pa[:, None, :, 0], ph[:, :, None, 1] - pa[:, None, :, 1])
        d = np.where(np.isnan(d), np.inf, d)
        dh = d.min(axis=2)  # home player -> nearest away
        da = d.min(axis=1)  # away player -> nearest home
        dh[~np.isfinite(dh)] = np.nan
        da[~np.isfinite(da)] = np.nan
        mh[s:s + chunk] = _masked_mean(dh, of_h[s:s + chunk])
        ma[s:s + chunk] = _masked_mean(da, of_a[s:s + chunk])
    return mh, ma


def ball_relations(ts: TeamSet, b: np.ndarray, vb: np.ndarray, cfg: dict) -> dict:
    """Players of one team against a ball position b (float32[N, 2], NaN = unknown).

    Returns nearest / second-nearest distance, counts within the radii, the nearest player's
    index and position, and its speed relative to the ball (vb, NaN where unknown).
    """
    N, K = ts.P.shape[0], ts.K
    known = np.isfinite(b).all(axis=1)
    d = np.hypot(ts.P[..., 0] - b[:, None, 0], ts.P[..., 1] - b[:, None, 1])
    d = np.where(ts.pres & known[:, None], d, np.inf)
    idx = np.argmin(d, axis=1)
    d1 = np.take_along_axis(d, idx[:, None], axis=1)[:, 0]
    if K >= 2:
        d2 = np.partition(d, 1, axis=1)[:, 1]
    else:
        d2 = np.full(N, np.inf, d.dtype)
    has = np.isfinite(d1)
    r_small, r_big = cfg["near_r"]
    n_small = np.where(known, (d < r_small).sum(axis=1), np.nan).astype(np.float32)
    n_big = np.where(known, (d < r_big).sum(axis=1), np.nan).astype(np.float32)
    rows = np.arange(N)
    pos = np.where(has[:, None], ts.P[rows, idx], np.nan).astype(np.float32)
    vp = ts.V[rows, idx]
    rel = np.hypot(vp[:, 0] - vb[:, 0], vp[:, 1] - vb[:, 1]).astype(np.float32)
    rel[~has] = np.nan
    return {
        "d1": np.where(has, d1, np.nan).astype(np.float32),
        "d2": np.where(np.isfinite(d2), d2, np.nan).astype(np.float32),
        "n2": n_small, "n5": n_big, "pos": pos, "has": has, "rel": rel,
    }


def pressure_on(carrier_pos: np.ndarray, opp: TeamSet, radius: float):
    """(# opponents within radius of the carrier, distance from the carrier to the nearest one)."""
    known = np.isfinite(carrier_pos).all(axis=1)
    d = np.hypot(opp.P[..., 0] - carrier_pos[:, None, 0], opp.P[..., 1] - carrier_pos[:, None, 1])
    d = np.where(opp.pres & known[:, None], d, np.inf)
    cnt = np.where(known, (d < radius).sum(axis=1), np.nan).astype(np.float32)
    dmin = d.min(axis=1)
    dmin = np.where(np.isfinite(dmin), dmin, np.nan).astype(np.float32)
    return cnt, dmin
