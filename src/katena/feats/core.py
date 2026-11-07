"""`featurize(m, cfg, extras) -> Feats` and `swap_teams(feats) -> Feats`."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..match import Match
from . import spec
from .ball import ball_state
from .team import build_team, mark_distances, ball_relations, pressure_on, shape_features
from .temporal import roll_max, roll_mean, roll_min, segments_from_lengths, shift_clip

FEATURE_VERSION = "v1"

DEFAULT_CFG: dict = {
    # rows
    "rows": "halves",          # "halves": one row per graded frame; "all": every frame, mask = graded
    "stride": 1,               # keep every stride-th frame of each segment (training subsample)
    "stride_offset": 0,
    # ball
    "max_interp_s": 1.0,       # linear interpolation over ball gaps up to this long
    "gap_cap_s": 10.0,         # cap for ball_since / ball_until / ball_gap_len (reported in frames)
    "vel_half_frames": 2,      # gap-aware central difference half-width for velocities
    "acc_half_frames": 3,      # half-width for acceleration and direction change
    "ball_vmax": 40.0,         # m/s; faster implied ball motion is a detector jump -> NaN
    "turn_min_speed": 1.0,     # m/s; direction change undefined below this
    "straight_s": 0.4,         # window for path straightness
    "straight_min_path": 1.0,  # m; straightness undefined over shorter paths
    "straight_thr": 0.95,
    "straight_min_speed": 4.0,
    "flight_speed": 8.0,       # m/s; in-flight proxy threshold
    # players
    "player_vmax": 12.0,       # m/s; faster implied slot motion is an identity swap -> NaN
    "dedup_m": 0.5,            # same-team slots closer than this are one person
    "near_r": (2.0, 5.0),
    "press_r": 5.0,
    "sprint_speed": 5.5,
    "static_speed": 1.0,
    # temporal context
    "context": True,
    "windows_s": (0.5, 2.0, 6.0),
    "step_s": (1.0, 4.0),
    # engineering
    "chunk": 16384,
}


@dataclass
class Feats:
    X: np.ndarray            # float32[T, F]
    t_frame: np.ndarray      # int64[T], frame index of each row
    mask: np.ndarray         # bool[T], row lies in a graded half
    names: list              # F column names
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.X.shape[0])

    @property
    def F(self) -> int:
        return int(self.X.shape[1])

    def col(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]

    def select(self, names) -> "Feats":
        """Same rows, only these columns, in this order."""
        names = list(names)
        idx = [self.names.index(n) for n in names]
        return replace(self, X=np.ascontiguousarray(self.X[:, idx]), names=names,
                       meta=dict(self.meta))

    def take(self, rows) -> "Feats":
        """Subset of rows (index array or bool mask)."""
        return replace(self, X=self.X[rows], t_frame=self.t_frame[rows], mask=self.mask[rows],
                       meta=dict(self.meta))


def _resolve_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_CFG)
        if unknown:
            raise ValueError(f"unknown featurize cfg keys: {sorted(unknown)}")
        out.update(cfg)
    if out["rows"] not in ("halves", "all"):
        raise ValueError("cfg['rows'] must be 'halves' or 'all'")
    if int(out["stride"]) < 1 or not 0 <= int(out["stride_offset"]) < int(out["stride"]):
        raise ValueError("need stride >= 1 and 0 <= stride_offset < stride")
    return out


def _timeline(m: Match, rows: str):
    """Frame index, attack direction and graded flag per timeline row, plus segment lengths."""
    blocks = sorted(m.half_blocks, key=lambda b: b[1])
    if not blocks:
        raise ValueError("match has no half_blocks")
    if rows == "halves":
        idx = np.concatenate([np.arange(f, l + 1, dtype=np.int64) for _, f, l in blocks])
        d = np.concatenate([np.full(l - f + 1, m.attack_dir[h], np.int8) for h, f, l in blocks])
        lengths = [l - f + 1 for _, f, l in blocks]
        graded = np.ones(len(idx), dtype=bool)
        return idx, d, graded, lengths
    # every frame; ungraded stretches take the attack direction of the nearest half
    n = m.n
    t = np.arange(n, dtype=np.int64)
    dist = np.stack([np.maximum(np.maximum(f - t, t - l), 0) for _, f, l in blocks])
    near = np.argmin(dist, axis=0)
    graded = m.graded_mask()
    d = np.array([m.attack_dir[h] for h, _, _ in blocks], dtype=np.int8)[near]
    key = near * 2 + graded
    cuts = np.flatnonzero(np.diff(key)) + 1
    lengths = np.diff(np.r_[0, cuts, n]).tolist()
    return t, d, graded, lengths


def featurize(m: Match, cfg: dict | None = None, extras: dict | None = None) -> Feats:
    """Per-frame feature matrix for one match. See `katena.feats` for the column list and policy."""
    cfg = _resolve_cfg(cfg)
    fps = float(m.fps)
    idx, d, graded, lengths = _timeline(m, cfg["rows"])
    seg_lo, seg_hi = segments_from_lengths(lengths)
    N = len(idx)
    f32 = np.float32

    # attack-normalised frame: rotate by 180 degrees in halves where home attacks -x
    df = d.astype(f32)
    B = m.ball[idx].astype(f32) * df[:, None]
    cols, baux = ball_state(B, seg_lo, seg_hi, fps, cfg, m.pitch)
    present, known_i, b_ref = baux["present"], baux["known_i"], baux["b_ref"]

    team = np.asarray(m.team)
    teams = {}
    for side, code, sign in (("home", 0, 1), ("away", 1, -1)):
        slots = np.flatnonzero(team == code)
        P = m.P[np.ix_(idx, slots)].astype(f32) * df[:, None, None]
        spd = None if m.speed is None else m.speed[np.ix_(idx, slots)].astype(f32)
        teams[side] = build_team(P, spd, seg_lo, seg_hi, fps, cfg, sign)
        del P, spd
    H, A = teams["home"], teams["away"]

    # ball-free team shape
    shp, saux = {}, {}
    for side, ts in teams.items():
        shp[side], saux[side] = shape_features(ts, cfg, m.pitch)
    shp["home"]["box_n"] = shp["home"]["box_att"] + saux["away"]["box_def"]
    shp["away"]["box_n"] = shp["away"]["box_att"] + saux["home"]["box_def"]
    shp["home"]["mark_dist"], shp["away"]["mark_dist"] = mark_distances(
        H, A, saux["home"]["outfield"], saux["away"]["outfield"], int(cfg["chunk"]))

    # players vs ball: one pass against b_ref (interpolated, else last / next seen), then masked
    v = baux["v"]
    rel = {s: ball_relations(ts, b_ref, v, cfg) for s, ts in teams.items()}
    nan = np.float32(np.nan)
    for s in ("home", "away"):
        r = rel[s]
        cols[f"{s}_nearest_dist"] = np.where(present, r["d1"], nan)
        cols[f"{s}_second_dist"] = np.where(present, r["d2"], nan)
        cols[f"{s}_n2"] = np.where(present, r["n2"], nan)
        cols[f"{s}_n5"] = np.where(present, r["n5"], nan)
        cols[f"{s}_relspeed"] = np.where(present, r["rel"], nan)
    margin = rel["away"]["d1"] - rel["home"]["d1"]
    cols["nearest_margin"] = np.where(present, margin, nan)
    for s in ("home", "away"):
        cols[f"{s}_nearest_dist_i"] = np.where(known_i, rel[s]["d1"], nan)
        cols[f"{s}_n5_i"] = np.where(known_i, rel[s]["n5"], nan)
    cols["nearest_margin_i"] = np.where(known_i, margin, nan)
    for s in ("home", "away"):
        cols[f"{s}_ball_dx_i"] = cols["ball_x_i"] - shp[s]["cx"]
    mo = cols["nearest_margin"]
    ls, ns = baux["ls"], baux["ns"]
    cols["nearest_margin_ffill"] = np.where(ls >= 0, mo[np.maximum(ls, 0)], nan)
    cols["nearest_margin_bfill"] = np.where(ns >= 0, mo[np.maximum(ns, 0)], nan)

    # pressing around each team's ball-nearest player
    hp, hs = pressure_on(rel["home"]["pos"], A, cfg["press_r"])  # away players on home's carrier
    ap, as_ = pressure_on(rel["away"]["pos"], H, cfg["press_r"])
    cols["away_pressers"], cols["home_carrier_space"] = hp, hs
    cols["home_pressers"], cols["away_carrier_space"] = ap, as_

    for s in ("home", "away"):
        for k, val in shp[s].items():
            cols[f"{s}_{k}"] = val
    sh, sa = shp["home"], shp["away"]
    cols["adv_diff"] = sh["cx"] + sa["cx"]
    cols["cx_gap"] = sa["cx"] - sh["cx"]
    cols["line_diff"] = sh["line_x"] + sa["line_x"]
    cols["width_diff"] = sh["width"] - sa["width"]
    cols["length_diff"] = sh["length"] - sa["length"]
    cols["dcent_diff"] = sh["dcent"] - sa["dcent"]

    # stillness and layout over both teams
    spd_all = np.concatenate([H.spd, A.spd], axis=1)
    sp_ok = np.isfinite(spd_all)
    n_sp = sp_ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["all_mspd"] = (np.where(sp_ok, spd_all, 0).sum(axis=1, dtype=np.float64)
                            / n_sp).astype(f32)
        cols["frac_static"] = ((sp_ok & (spd_all < cfg["static_speed"])).sum(axis=1)
                               / n_sp).astype(f32)
    cols["all_mspd"][n_sp == 0] = np.nan
    cols["frac_static"][n_sp == 0] = np.nan
    del spd_all, sp_ok
    P_all = np.concatenate([H.P, A.P], axis=1)
    pr_all = np.concatenate([H.pres, A.pres], axis=1)
    n_all = pr_all.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_all = np.where(pr_all[..., None], P_all, 0).sum(axis=1, dtype=np.float64) / n_all[:, None]
        dc = np.hypot(P_all[..., 0] - c_all[:, None, 0], P_all[..., 1] - c_all[:, None, 1])
        cols["all_dcent"] = (np.where(pr_all, dc, 0).sum(axis=1) / n_all).astype(f32)
    cols["all_dcent"][n_all < 2] = np.nan
    cols["n_present"] = n_all.astype(f32)
    del P_all, pr_all, dc

    names = spec.base_names()
    missing = [n for n in names if n not in cols]
    if missing:
        raise RuntimeError(f"featurize did not compute {missing}")

    # rows to emit
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    st, off = int(cfg["stride"]), int(cfg["stride_offset"])
    rows = np.concatenate([np.arange(s + off, s + ln, st) for s, ln in zip(starts, lengths)])
    rows = rows.astype(np.int64)

    out_cols = [cols[n][rows] for n in names]
    if cfg["context"]:
        ctx_names = spec.context_names(cfg["windows_s"], cfg["step_s"])
        out_cols += _context(cols, cfg, fps, seg_lo, seg_hi, rows)
        names = names + ctx_names

    t_frame = idx[rows]
    if extras:
        for k in sorted(extras):
            v_ = np.asarray(extras[k])
            if v_.shape != (m.n,):
                raise ValueError(f"extra {k!r} must have shape ({m.n},), got {v_.shape}")
            names.append(f"x_{k}")
            out_cols.append(v_[t_frame].astype(f32))
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")

    X = np.empty((len(rows), len(names)), dtype=f32)
    for j, c in enumerate(out_cols):
        X[:, j] = c
    np.nan_to_num(X, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    meta = {"version": FEATURE_VERSION, "cfg": cfg, "match": m.name, "fps": m.fps,
            "swapped": False}
    return Feats(X=X, t_frame=t_frame.astype(np.int64), mask=graded[rows].astype(bool),
                 names=names, meta=meta)


def _context(cols: dict, cfg: dict, fps: float, seg_lo, seg_hi, rows) -> list:
    out = []
    for w in cfg["windows_s"]:
        h = max(1, int(round(w * fps / 2)))
        out += [roll_mean(cols[b], h, seg_lo, seg_hi)[rows] for b in spec.CONTEXT_MEAN]
        out += [roll_min(cols[b], h, seg_lo, seg_hi)[rows] for b in spec.CONTEXT_MIN]
        out += [roll_max(cols[b], h, seg_lo, seg_hi)[rows] for b in spec.CONTEXT_MAX]
    for w in cfg["step_s"]:
        # mean over the next w seconds minus mean over the previous w seconds
        h = max(1, int(round(w * fps / 2)))
        for b in spec.CONTEXT_STEP:
            r = roll_mean(cols[b], h, seg_lo, seg_hi)
            out.append((shift_clip(r, h, seg_lo, seg_hi) - shift_clip(r, -h, seg_lo, seg_hi))[rows])
    return out


def swap_teams(feats: Feats) -> Feats:
    """Relabel home <-> away: rotate the pitch, swap paired columns, negate odd ones.

    An involution, and equal (to float rounding) to featurizing the team-swapped match. Labels are
    not part of Feats; swap them alongside (H <-> A, D unchanged).
    """
    perm, sign = spec.swap_spec(feats.names)
    X = feats.X[:, perm] * sign[None, :]
    meta = dict(feats.meta)
    meta["swapped"] = not meta.get("swapped", False)
    return replace(feats, X=X.astype(np.float32, copy=False), meta=meta)
