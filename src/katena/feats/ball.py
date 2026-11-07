"""Ball state: presence, gap timers, short-gap interpolation, hold positions, gap-aware kinematics.

Everything works on the attack-normalised timeline (home attacks +x) built by `core.featurize`,
one segment per half, and never crosses a segment boundary.
"""
from __future__ import annotations

import numpy as np

from .temporal import gap_diff, last_true_index, next_true_index, run_bounds


def ball_state(B: np.ndarray, seg_lo: np.ndarray, seg_hi: np.ndarray, fps: float, cfg: dict,
               pitch) -> tuple[dict, dict]:
    """Per-row ball columns, plus the positions other modules need.

    B: float32[N, 2] attack-normalised ball, NaN = unknown.
    Returns (cols, aux): `cols` are output columns keyed by name; `aux` holds `present`,
    `known_i`, `b_ref` (interpolated, else last seen, else next seen), `ls` / `ns` (index of last
    / next observed row in the segment, -1 if none) and `v` (observed ball velocity).
    """
    N = B.shape[0]
    L, W = float(pitch[0]), float(pitch[1])
    t = np.arange(N, dtype=np.int64)
    cap = int(round(cfg["gap_cap_s"] * fps))
    max_gap = int(round(cfg["max_interp_s"] * fps))

    present = np.isfinite(B).all(axis=1)
    B = np.where(present[:, None], B, np.nan).astype(np.float32)
    ls = last_true_index(present, seg_lo)
    ns = next_true_index(present, seg_hi)
    has_l, has_n = ls >= 0, ns >= 0

    since = np.where(present, 0, np.where(has_l, t - ls, cap))
    until = np.where(present, 0, np.where(has_n, ns - t, cap))
    gap_len = np.where(present, 0, np.where(has_l & has_n, ns - ls - 1, cap))

    # short-gap linear interpolation between the last and next observed positions
    b_last = np.where(has_l[:, None], B[np.maximum(ls, 0)], np.nan).astype(np.float32)
    b_next = np.where(has_n[:, None], B[np.maximum(ns, 0)], np.nan).astype(np.float32)
    interp = ~present & has_l & has_n & (ns - ls - 1 <= max_gap)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = ((t - ls) / np.maximum(ns - ls, 1)).astype(np.float32)[:, None]
    b_i = np.where(interp[:, None], (1 - w) * b_last + w * b_next, B).astype(np.float32)
    known_i = present | interp
    b_ref = np.where(known_i[:, None], b_i, np.where(has_l[:, None], b_last, b_next))

    cols: dict[str, np.ndarray] = {}
    f32 = np.float32
    cols["ball_present"] = present.astype(f32)
    cols["ball_since"] = np.minimum(since, cap).astype(f32)
    cols["ball_until"] = np.minimum(until, cap).astype(f32)
    cols["ball_gap_len"] = np.minimum(gap_len, cap).astype(f32)
    cols["ball_interp"] = interp.astype(f32)
    cols["ball_known_i"] = known_i.astype(f32)

    bx, by = B[:, 0], B[:, 1]
    cols["ball_x"] = bx
    cols["ball_absy"] = np.abs(by)
    cols["home_ballgoal_dist"] = np.hypot(bx - f32(L / 2), by)
    cols["away_ballgoal_dist"] = np.hypot(bx + f32(L / 2), by)
    cols["ball_out_x"] = np.abs(bx) - f32(L / 2)
    cols["ball_out_y"] = np.abs(by) - f32(W / 2)
    cols["ball_edge_dist"] = np.minimum(f32(L / 2) - np.abs(bx), f32(W / 2) - np.abs(by))
    cols["ball_x_i"] = b_i[:, 0]
    cols["ball_absy_i"] = np.abs(b_i[:, 1])

    # kinematics on observed frames only, never differencing across a gap
    rs, re = run_bounds(present, seg_lo, seg_hi)
    v = gap_diff(B, present, rs, re, int(cfg["vel_half_frames"]), fps)
    spd = np.hypot(v[:, 0], v[:, 1])
    v[spd > cfg["ball_vmax"]] = np.nan  # detector jumps, not ball motion
    spd = np.hypot(v[:, 0], v[:, 1])
    cols["ball_vx"] = v[:, 0]
    cols["ball_speed"] = spd

    pv = np.isfinite(v).all(axis=1)
    rsv, rev = run_bounds(pv, seg_lo, seg_hi)
    ka = int(cfg["acc_half_frames"])
    acc = gap_diff(v, pv, rsv, rev, ka, fps)
    cols["ball_accel"] = np.hypot(acc[:, 0], acc[:, 1])

    ja = np.where(pv, np.minimum(np.minimum(t - rsv, rev - t), ka), 0)
    v1 = v[np.maximum(t - ja, 0)]
    v2 = v[np.minimum(t + ja, N - 1)]
    n1, n2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (v1 * v2).sum(axis=1) / (n1 * n2)
    turn = np.arccos(np.clip(cosang, -1.0, 1.0)).astype(f32)
    ok = pv & (ja >= 1) & (n1 >= cfg["turn_min_speed"]) & (n2 >= cfg["turn_min_speed"])
    turn[~ok] = np.nan
    cols["ball_turn"] = turn

    # straightness of the observed path over [t - hs, t + hs]: displacement / path length
    hs = max(1, int(round(cfg["straight_s"] * fps / 2)))
    step = np.zeros(N, dtype=np.float64)
    same = present[:-1] & present[1:] & (seg_lo[:-1] == seg_lo[1:])
    d = np.hypot(B[1:, 0] - B[:-1, 0], B[1:, 1] - B[:-1, 1])
    step[:-1] = np.where(same, d, 0.0)
    cpath = np.concatenate([[0.0], np.cumsum(step)])
    cover = present & (t - rs >= hs) & (re - t >= hs)
    lo, hi = np.clip(t - hs, 0, N - 1), np.clip(t + hs, 0, N - 1)
    path = cpath[hi] - cpath[lo]
    disp = np.hypot(B[hi, 0] - B[lo, 0], B[hi, 1] - B[lo, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        straight = (disp / path).astype(f32)
    straight[~(cover & (path > cfg["straight_min_path"]))] = np.nan
    cols["ball_straight"] = np.minimum(straight, f32(1.0))

    fl = (spd > cfg["flight_speed"]) | ((cols["ball_straight"] > cfg["straight_thr"])
                                        & (spd > cfg["straight_min_speed"]))
    cols["ball_inflight"] = np.where(np.isfinite(spd), fl, np.nan).astype(f32)

    aux = {"present": present, "known_i": known_i, "b_ref": b_ref.astype(f32), "ls": ls,
           "ns": ns, "v": v}
    return cols, aux
