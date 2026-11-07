"""Noise simulator: make clean tracking with identities look like Metrica's automatic tracking (ATD).

What the ATD looks like (public facts, docs/upstream/DATA.md): 46 track slots for 22 players, slots
appear, vanish and swap identity, team membership per slot is known but identities are not, about
20.5 objects present per graded frame and occasionally more than 22 (duplicates), the ball present
on 59.3% of graded frames and missing disproportionately in crowds, duels and aerial play, speed per
slot in m/s, NaN = missing.

`corrupt(m_clean, cfg, seed)` reproduces those properties from a clean match:

(a) identity stripping and fragmentation. Each clean player's on-pitch time is cut into fragments
    (lognormal durations in a congestion-accelerated clock). After a break the player is either
    re-acquired at once in a new slot or lost for a gap. Fragments are packed into a pool of
    `n_slots` team-owned slots, so slots appear, vanish and change occupant. Close same-team
    encounters can swap two players' slots mid-run. A small share of fragments land in a slot of the
    wrong team (team classification errors).
(b) occlusion dropouts: the gaps above. Breaks are more likely when other players are close, and the
    gap scale is solved so the mean number of present objects hits `players_present_mean`.
(c) bursty ball loss: a two-state Gilbert-Elliott process. Onsets of a "bad" state come at a rate
    that rises with players near the ball, with ball speed (flight proxy), with ball height when the
    clean match carries `meta["ball_z"]`, and in dead-ball time; bad
    bursts last a lognormal time that also grows with congestion; the "good" state drops isolated
    frames. The onset rate is solved so the graded ball-present rate hits `ball_present_rate`.
(d) duplicates (a second track shadowing a player) and ghosts (tracks that are nobody).
(e) positional error: white jitter, a slow AR(1) wobble per slot, and a constant bias per fragment.
(f) speed recomputed per slot from the noisy positions, NaN where a slot is empty.

Labels `y` are copied unchanged, and per-slot identity entries of `meta` (such as `slot_ids`) are
dropped. Deterministic for a given (clean match, cfg, seed).
"""
from __future__ import annotations

import bisect
import math
import warnings

import numpy as np
from scipy.signal import lfilter

from ..match import D, X, Match
from ._util import ball_speed, blocks_of, count_within, nan_moving_mean, runs_2d

# Values marked [public] are anchored to docs/upstream/DATA.md. Everything else is a guess, meant to
# be replaced by `fit_noise_cfg` on the target match.
DEFAULT_CFG: dict = {
    "version": 1,
    # -- slots --------------------------------------------------------------------------------
    "n_slots": 46,                 # [public] 46 track slots
    "home_slot_frac": 0.5,         # share of slots owned by the home team (fit reads it off target)
    "slot_policy": "lowest",       # which free slot a new fragment takes: lowest | lru | random
    # -- ball ---------------------------------------------------------------------------------
    "ball_present_rate": 0.593,    # [public] ball present on 59.3% of graded frames
    "ball_iso_drop_share": 0.10,   # share of missing ball frames that are isolated good-state drops
    "ball_gap_median_s": 0.6,      # median bad-burst length for an uncontested ball
    "ball_gap_sigma": 1.2,         # lognormal sigma of bad-burst lengths
    "ball_gap_max_s": 30.0,        # cap on one bad burst
    "ball_cong_radius_m": 3.0,     # "near the ball"
    "ball_beta_cong": 0.4,         # log onset-rate per player near the ball (capped at 4)
    "ball_gap_cong_gamma": 0.15,   # log burst-length per player near the ball at onset
    "ball_fast_mps": 15.0,         # flight proxy threshold on clean ball speed
    "ball_beta_speed": 0.7,        # log onset-rate when the ball is fast
    "ball_air_z_m": 1.8,           # ball above this height counts as aerial (needs meta["ball_z"])
    "ball_beta_air": 0.8,          # log onset-rate when the ball is aerial
    "ball_beta_dead": 0.5,         # log onset-rate in dead-ball / ungraded time
    "ball_jitter_sd_m": 0.3,
    "ball_outlier_rate": 0.002,    # per observed frame: start of a short false-position burst
    "ball_outlier_len_s": 0.12,
    "ball_outlier_sd_m": 3.0,
    # -- players ------------------------------------------------------------------------------
    "players_present_mean": 20.5,  # [public] mean tracked objects per graded frame (per 22 on pitch)
    "frag_median_s": 10.0,         # median fragment length in congestion-adjusted time
    "frag_sigma": 1.2,
    "occl_radius_m": 1.5,          # "close" for occlusion
    "frag_beta_cong": 0.3,         # log break-rate per close player (capped at 4)
    "p_gap": 0.6,                  # P(dropout gap after a break); else instant re-acquisition
    "gap_sigma": 1.0,              # lognormal sigma of gap lengths; the median is solved
    "gap_max_s": 60.0,             # cap on one dropout gap
    "swap_dist_m": 1.0,            # same-team encounter distance that can swap identities
    "swap_prob": 0.1,              # P(swap) per encounter
    "team_flip_rate": 0.01,        # P(fragment lands in a slot of the wrong team)
    "dup_rate_per_min": 3.0,       # duplicate-track onsets per minute
    "dup_median_s": 1.5,
    "dup_offset_sd_m": 0.3,
    "ghost_rate_per_min": 1.0,     # ghost-track onsets per minute
    "ghost_median_s": 2.0,
    "ghost_speed_mps": 1.5,
    # -- positions and speed ------------------------------------------------------------------
    "pos_jitter_sd_m": 0.05,       # white, per frame
    "pos_wobble_sd_m": 0.2,        # AR(1), per slot
    "pos_wobble_rho": 0.98,        # per frame (time constant about 2 s at 25 fps)
    "frag_bias_sd_m": 0.15,        # constant per fragment
    "speed_win_s": 0.2,            # half-window of the position smoother before differencing
}

ANCHORED = {
    "n_slots": "DATA.md: 46 track slots",
    "ball_present_rate": "DATA.md: ball present on 59.3% of graded frames (94,214 / 158,857)",
    "players_present_mean": "DATA.md: mean 20.5 tracked objects over graded frames",
}

SLOT_POLICIES = ("lowest", "lru", "random")
_CAP = 4  # congestion counts are capped here before entering a log-rate


def resolve_cfg(noise_cfg: dict | None) -> dict:
    """DEFAULT_CFG overlaid with `noise_cfg`. Unknown keys raise, except informational `_...` keys."""
    cfg = dict(DEFAULT_CFG)
    if noise_cfg:
        unknown = sorted(k for k in noise_cfg if k not in DEFAULT_CFG and not str(k).startswith("_"))
        if unknown:
            raise KeyError(f"unknown noise_cfg keys: {unknown}")
        cfg.update(noise_cfg)
    if cfg["slot_policy"] not in SLOT_POLICIES:
        raise ValueError(f"slot_policy must be one of {SLOT_POLICIES}")
    return cfg


class _Stream:
    """Index-addressable random draws: value k is the same however many times a loop is re-run."""
    __slots__ = ("rng", "kind", "buf")

    def __init__(self, rng: np.random.Generator, kind: str):
        self.rng, self.kind, self.buf = rng, kind, []

    def grow(self, k: int) -> list:
        while len(self.buf) <= k:
            m = max(1024, len(self.buf))
            if self.kind == "exp":
                new = self.rng.standard_exponential(m)
            elif self.kind == "norm":
                new = self.rng.standard_normal(m)
            else:
                new = self.rng.random(m)
            self.buf.extend(new.tolist())
        return self.buf


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


# ------------------------------------------------------------------------------------------------
# congestion and encounters on the clean match
# ------------------------------------------------------------------------------------------------

def _player_congestion(P0, team0, r_occ, swap_dist, stride, chunk=2048):
    """Per player-frame count of other players within r_occ (held between sampled frames), plus
    onsets of same-team encounters closer than swap_dist as arrays (frame, i, j)."""
    n, S0 = P0.shape[:2]
    idx = np.arange(0, n, stride)
    cong_s = np.zeros((idx.shape[0], S0), dtype=np.int8)
    iu, ju = np.triu_indices(S0, 1)
    same = team0[iu] == team0[ju]
    pi, pj = iu[same], ju[same]
    prev = np.zeros(pi.shape[0], dtype=bool)
    enc_t, enc_i, enc_j = [], [], []
    r2, s2 = r_occ * r_occ, swap_dist * swap_dist
    diag = np.arange(S0)
    for a in range(0, idx.shape[0], chunk):
        Q = P0[idx[a:a + chunk]]
        d = Q[:, :, None, :] - Q[:, None, :, :]
        with np.errstate(invalid="ignore"):
            d2 = d[..., 0] ** 2 + d[..., 1] ** 2
        d2[:, diag, diag] = np.inf
        with np.errstate(invalid="ignore"):
            cong_s[a:a + chunk] = np.minimum((d2 < r2).sum(2), 127)
            close = d2[:, pi, pj] < s2
        if pi.shape[0]:
            onset = close & ~np.vstack([prev[None], close[:-1]])
            k, p = np.nonzero(onset)
            enc_t.append(idx[a + k]); enc_i.append(pi[p]); enc_j.append(pj[p])
            prev = close[-1]
    cong = np.repeat(cong_s, stride, axis=0)[:n]
    if enc_t:
        enc = (np.concatenate(enc_t), np.concatenate(enc_i), np.concatenate(enc_j))
    else:
        enc = (np.zeros(0, int), np.zeros(0, int), np.zeros(0, int))
    return cong, enc


def _dead_mask(m: Match) -> np.ndarray:
    """Dead-ball or ungraded time: from labels when the clean match has them, else low activity."""
    if m.y is not None:
        return np.isin(m.y, (D, X))
    if m.n < 2:
        return np.zeros(m.n, dtype=bool)
    v = np.linalg.norm(np.diff(m.P, axis=0), axis=-1) * m.fps
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        act = np.nanmean(np.minimum(v, 12.0), axis=1)
    act = nan_moving_mean(np.concatenate([act[:1], act]), m.fps)
    if not np.isfinite(act).any():
        return np.zeros(m.n, dtype=bool)
    return act <= np.nanquantile(act, 0.2)


# ------------------------------------------------------------------------------------------------
# ball
# ------------------------------------------------------------------------------------------------

def _bursts(H, lam, len_mult, sigma, cap, E: _Stream, Z: _Stream):
    """Bad-state onsets and lengths of a semi-Markov process in the clock H (cumulative rate)."""
    n = H.shape[0]
    ons, lens = [], []
    t, k = 0, 0
    e, z = E.buf, Z.buf
    while t < n:
        if k >= len(e):
            e, z = E.grow(k), Z.grow(k)
        base = H[t - 1] if t > 0 else 0.0
        t_on = int(np.searchsorted(H, base + e[k] / lam))
        if t_on >= n:
            break
        L = int(len_mult[t_on] * math.exp(sigma * z[k]) + 0.5)
        L = 1 if L < 1 else (cap if L > cap else L)
        ons.append(t_on)
        lens.append(L)
        t = t_on + L
        k += 1
    return np.asarray(ons, dtype=np.int64), np.asarray(lens, dtype=np.int64)


def _interval_mask(n, starts, lens):
    inc = np.zeros(n + 1, dtype=np.int32)
    np.add.at(inc, starts, 1)
    np.add.at(inc, np.minimum(starts + lens, n), -1)
    return np.cumsum(inc[:-1]) > 0


def _ball_height(m: Match) -> np.ndarray | None:
    z = (m.meta or {}).get("ball_z")
    if isinstance(z, np.ndarray) and z.shape == (m.n,):
        return z.astype(np.float32)
    return None


# meta entries that carry per-slot identities of the clean match; they must not survive corruption
_IDENTITY_META = ("slot_ids", "player_ids", "slot_names", "shirt_numbers", "jersey_numbers")


def _strip_identity_meta(meta: dict, n: int, S0: int) -> tuple[dict, list]:
    keep, dropped = {}, []
    for k, v in (meta or {}).items():
        slotted = False
        if k in _IDENTITY_META:
            slotted = True
        elif isinstance(v, np.ndarray):
            slotted = not (v.ndim == 1 and v.shape[0] == n) and S0 in v.shape
        elif isinstance(v, (list, tuple)):
            slotted = len(v) == S0 and S0 != n
        if slotted:
            dropped.append(k)
        else:
            keep[k] = v
    return keep, dropped


def _ball_observed(ball0, c_ball, fast, air, dead, graded, cfg, fps, rngs):
    n = ball0.shape[0]
    valid = np.isfinite(ball0[:, 0])
    vg = valid & graded
    info = {"ball_onset_rate_per_s": 0.0, "ball_bad_frac": 0.0}
    present_clean = float(vg.sum()) / max(1, int(graded.sum()))
    if present_clean <= 0:
        return valid, info
    M = float(np.clip(1.0 - cfg["ball_present_rate"] / present_clean, 0.0, 0.98))
    share = float(np.clip(cfg["ball_iso_drop_share"], 0.0, 1.0))
    fB_t = (1.0 - share) * M
    cb = np.minimum(c_ball, _CAP).astype(np.float64)
    bad = np.zeros(n, dtype=bool)
    fB = 0.0
    if fB_t > 0:
        logm = (cfg["ball_beta_cong"] * cb + cfg["ball_beta_speed"] * fast + cfg["ball_beta_air"] * air
                + cfg["ball_beta_dead"] * dead)
        mult = np.exp(logm)
        mult[~valid] = 0.0
        mult /= mult[vg].mean() if vg.any() else 1.0
        H = np.cumsum(mult)
        gap_med = cfg["ball_gap_median_s"] * fps
        len_mult = gap_med * np.exp(cfg["ball_gap_cong_gamma"] * cb)
        cap = max(1, int(cfg["ball_gap_max_s"] * fps))
        sig = cfg["ball_gap_sigma"]
        mean_len = float(np.mean(len_mult[vg])) * math.exp(sig * sig / 2) if vg.any() else gap_med
        lam = max(fB_t / ((1 - fB_t) * max(mean_len, 1.0)), 1e-9)
        E, Z = _Stream(rngs[0], "exp"), _Stream(rngs[2], "norm")
        # Solve the onset rate. The bad fraction is a noisy function of the rate (one long burst
        # moves it), so aim to land just above the target and then drop whole bursts to fit.
        above = closest = None
        for _ in range(14):
            ons, lens = _bursts(H, lam, len_mult, sig, cap, E, Z)
            bad = _interval_mask(n, ons, lens)
            fB = float(bad[vg].mean()) if vg.any() else 0.0
            rec = (fB, lam, ons, lens)
            if fB >= fB_t and (above is None or fB < above[0]):
                above = rec
            if closest is None or abs(fB - fB_t) < abs(closest[0] - fB_t):
                closest = rec
            if above is not None and above[0] - fB_t < 0.01:
                break
            step = _odds(fB_t) / _odds(fB)
            lam *= max(step, 1.25) if fB < fB_t else step
        fB, lam, ons, lens = above if above is not None else closest
        if fB > fB_t and ons.size:
            # drop whole bursts in random order, keeping the bad fraction closest to the target
            cvg = np.concatenate([[0], np.cumsum(vg)])
            w = cvg[np.minimum(ons + lens, n)] - cvg[ons]
            perm = rngs[1].permutation(ons.size)
            removed = np.concatenate([[0], np.cumsum(w[perm])])
            k = int(np.argmin(np.abs(removed - (fB - fB_t) * vg.sum())))
            keep = np.ones(ons.size, dtype=bool)
            keep[perm[:k]] = False
            ons, lens = ons[keep], lens[keep]
        bad = _interval_mask(n, ons, lens)
        fB = float(bad[vg].mean()) if vg.any() else 0.0
        info = {"ball_onset_rate_per_s": float(lam * fps), "ball_bad_frac": fB}
    # isolated drops make up whatever the bursts left, so the graded present rate lands on target
    q = 1.0 - (1.0 - M) / max(1e-6, 1.0 - fB)
    iso = rngs[1].random(n) < float(np.clip(q, 0.0, 0.95))
    return valid & ~bad & ~iso, info


def _noisy_ball(ball0, observed, cfg, fps, rng):
    n = ball0.shape[0]
    out = ball0 + rng.normal(0.0, cfg["ball_jitter_sd_m"], size=(n, 2)).astype(np.float32)
    rate = cfg["ball_outlier_rate"]
    if rate > 0:
        ons = np.flatnonzero(observed & (rng.random(n) < rate))
        mean_len = max(1.0, cfg["ball_outlier_len_s"] * fps)
        lens = rng.geometric(1.0 / mean_len, size=ons.shape[0])
        offs = rng.normal(0.0, cfg["ball_outlier_sd_m"], size=(ons.shape[0], 2)).astype(np.float32)
        for t, L, o in zip(ons, lens, offs):
            out[t:t + L] += o
    out[~observed] = np.nan
    return out


# ------------------------------------------------------------------------------------------------
# players
# ------------------------------------------------------------------------------------------------

def _player_frags(Hp, segs, frag_med, frag_sig, p_gap, gap_med, gap_sig, gap_cap, Z1, Z2, U):
    """Fragments [start, stop) of one player over its on-pitch segments."""
    starts, stops = [], []
    k = 0
    z1, z2, u = Z1.buf, Z2.buf, U.buf
    for s0, e0 in segs:
        t = s0
        while t < e0:
            if k >= len(z1):
                z1, z2, u = Z1.grow(k), Z2.grow(k), U.grow(k)
            dur = frag_med * math.exp(frag_sig * z1[k])
            base = Hp[t - 1] if t > 0 else 0.0
            te = int(np.searchsorted(Hp, base + dur)) + 1
            if te <= t:
                te = t + 1
            if te > e0:
                te = e0
            starts.append(t)
            stops.append(te)
            if te >= e0:
                k += 1
                break
            if u[k] < p_gap:
                G = int(gap_med * math.exp(gap_sig * z2[k]) + 0.5)
                t = te + (1 if G < 1 else (gap_cap if G > gap_cap else G))
            else:
                t = te
            k += 1
    return starts, stops


def _seg_end(seg_s, seg_e, t):
    i = bisect.bisect_right(seg_s, t) - 1
    return seg_e[i] if i >= 0 and t < seg_e[i] else t


def _slot_speed(P, fps, half, chunk=8):
    """Speed per slot from positions smoothed within each contiguous run; NaN where empty."""
    n, S = P.shape[:2]
    out = np.full((n, S), np.nan, dtype=np.float32)
    if n < 2:
        return out
    t = np.arange(n)[:, None]
    for a in range(0, S, chunk):
        Q = P[:, a:a + chunk].astype(np.float64)
        pres = np.isfinite(Q[..., 0])
        s = pres.shape[1]
        z = np.zeros((1, s), dtype=bool)
        first = pres & ~np.vstack([z, pres[:-1]])
        last = pres & ~np.vstack([pres[1:], z])
        rs = np.maximum.accumulate(np.where(first, t, 0), axis=0)
        re = np.minimum.accumulate(np.where(last, t, n - 1)[::-1], axis=0)[::-1]
        C = np.zeros((n + 1, s, 2))
        np.cumsum(np.where(pres[..., None], Q, 0.0), axis=0, out=C[1:])
        cols = np.arange(s)[None, :]
        lo = np.maximum(t - half, rs)
        hi = np.maximum(np.minimum(t + half, re), lo)
        Sm = (C[hi + 1, cols] - C[lo, cols]) / (hi - lo + 1)[..., None]
        tp = np.clip(np.minimum(t + 1, re), 0, n - 1)
        tm = np.clip(np.maximum(t - 1, rs), 0, n - 1)
        dt = tp - tm
        dv = Sm[tp, cols] - Sm[tm, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.sqrt((dv ** 2).sum(-1)) * fps / np.maximum(dt, 1)
        v[(dt <= 0) | ~pres] = np.nan
        out[:, a:a + chunk] = v
    return out


def _ar1_noise(rng, shape, sd, rho):
    """Stationary AR(1) noise along axis 0, float32."""
    n = shape[0]
    flat = int(np.prod(shape[1:]))
    e = rng.standard_normal((n, flat), dtype=np.float32)
    x0 = rng.standard_normal((1, flat)).astype(np.float32) * sd
    b = np.array([math.sqrt(max(0.0, 1 - rho * rho)) * sd], dtype=np.float32)
    a = np.array([1.0, -rho], dtype=np.float32)
    y, _ = lfilter(b, a, e, axis=0, zi=(rho * x0).astype(np.float32))
    return y.astype(np.float32).reshape(shape)


# ------------------------------------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------------------------------------

def corrupt(m_clean: Match, noise_cfg: dict | None = None, seed: int = 0, *,
            return_truth: bool = False):
    """New Match with ATD-like tracking noise; labels `y` unchanged.

    With `return_truth=True` returns (match, truth) where truth holds `slot_player` int16[n, S]
    (clean player index, -1 empty, -2 ghost), `slot_kind` int8[n, S] (0 player, 1 duplicate,
    2 ghost, -1 empty) and `ball_observed` bool[n]. Never feed truth to a model.
    """
    cfg = resolve_cfg(noise_cfg)
    m = m_clean
    n, S0, fps = m.n, m.n_slots, int(m.fps)
    L_pitch, W_pitch = m.pitch
    root = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x4B4432])
    ss = root.spawn(10)
    rg = [np.random.default_rng(s) for s in ss]
    r_ball, r_iso, r_swap, r_dup, r_ghost, r_slot, r_pos, r_bnoise, r_blen = rg[:9]
    frag_ss = ss[9].spawn(S0)

    P0 = np.asarray(m.P, dtype=np.float32)
    ball0 = np.asarray(m.ball, dtype=np.float32)
    team0 = np.asarray(m.team, dtype=np.int8)
    on = np.isfinite(P0[..., 0])
    graded = np.zeros(n, dtype=bool)
    for f, e in blocks_of(m):
        graded[f:e] = True
    dead = _dead_mask(m)
    stride = max(1, int(round(0.2 * fps)))

    # ---- (c) ball -------------------------------------------------------------------------------
    c_ball = count_within(P0, ball0, cfg["ball_cong_radius_m"])
    with np.errstate(invalid="ignore"):
        fast = ball_speed(ball0, fps) > cfg["ball_fast_mps"]
        z = _ball_height(m)
        air = np.zeros(n, dtype=bool) if z is None else z > cfg["ball_air_z_m"]
    observed, ball_info = _ball_observed(ball0, c_ball, fast, air, dead, graded, cfg, fps,
                                         (r_ball, r_iso, r_blen))
    ball_out = _noisy_ball(ball0, observed, cfg, fps, r_bnoise)

    # ---- congestion and encounters --------------------------------------------------------------
    cong, (enc_t, enc_i, enc_j) = _player_congestion(P0, team0, cfg["occl_radius_m"],
                                                     cfg["swap_dist_m"], stride)
    col, seg_start, seg_stop = runs_2d(on)
    segs = [[] for _ in range(S0)]
    for p, s, e in zip(col.tolist(), seg_start.tolist(), seg_stop.tolist()):
        segs[p].append((s, e))
    seg_s = [[s for s, _ in sg] for sg in segs]
    seg_e = [[e for _, e in sg] for sg in segs]

    on_g = on & graded[:, None]
    n_graded = max(1, int(graded.sum()))
    S_on = float(on_g.sum()) / n_graded  # mean clean players on pitch per graded frame
    Cg = np.concatenate([[0], np.cumsum(graded)])  # graded frames before t

    # ---- (d) duplicates and ghosts --------------------------------------------------------------
    sig_short = 0.8
    dups = []  # (player, start, stop, offset)
    if cfg["dup_rate_per_min"] > 0 and S_on > 0:
        p_on = cfg["dup_rate_per_min"] / (60.0 * fps) / max(S_on, 1.0)
        tt, pp = np.nonzero(on & (r_dup.random((n, S0), dtype=np.float32) < p_on))
        lens = np.maximum(1, np.round(cfg["dup_median_s"] * fps *
                                      np.exp(sig_short * r_dup.standard_normal(tt.shape[0])))).astype(int)
        offs = r_dup.normal(0.0, cfg["dup_offset_sd_m"], size=(tt.shape[0], 2)).astype(np.float32)
        for t, p, L, o in zip(tt.tolist(), pp.tolist(), lens.tolist(), offs):
            e = min(t + L, _seg_end(seg_s[p], seg_e[p], t))
            if e > t:
                dups.append((p, t, e, o))
    ghosts = []  # (start, stop, team)
    if cfg["ghost_rate_per_min"] > 0:
        tt = np.flatnonzero(r_ghost.random(n) < cfg["ghost_rate_per_min"] / (60.0 * fps))
        lens = np.maximum(1, np.round(cfg["ghost_median_s"] * fps *
                                      np.exp(sig_short * r_ghost.standard_normal(tt.shape[0])))).astype(int)
        tms = r_ghost.integers(0, 2, size=tt.shape[0])
        for t, L, tm in zip(tt.tolist(), lens.tolist(), tms.tolist()):
            ghosts.append((t, min(n, t + L), tm))
    extra_cnt = (sum(Cg[e] - Cg[s] for _, s, e, _ in dups) +
                 sum(Cg[e] - Cg[s] for s, e, _ in ghosts)) / n_graded

    # ---- (a, b) fragmentation and dropouts ------------------------------------------------------
    cnt_target = cfg["players_present_mean"] * S_on / 22.0
    cov_t = float(np.clip((cnt_target - extra_cnt) / max(S_on, 1e-9), 0.02, 1.0))
    pm = np.exp(cfg["frag_beta_cong"] * np.minimum(cong, _CAP).astype(np.float64))
    norm = pm[on_g].mean() if on_g.any() else 1.0
    Hps = [np.cumsum(pm[:, p] / norm) for p in range(S0)]
    del pm
    streams = []
    for p in range(S0):
        rp = np.random.default_rng(frag_ss[p])
        streams.append((_Stream(rp, "norm"), _Stream(rp, "norm"), _Stream(rp, "unif")))
    frag_med = cfg["frag_median_s"] * fps
    fsig, gsig, p_gap = cfg["frag_sigma"], cfg["gap_sigma"], float(cfg["p_gap"])
    gap_cap = max(1, int(cfg["gap_max_s"] * fps))
    F_mean = frag_med * math.exp(fsig * fsig / 2)
    if p_gap > 0:
        gap_med = F_mean * ((1 - cov_t) / cov_t) / p_gap / math.exp(gsig * gsig / 2)
    else:
        gap_med = 1.0
    gap_med = float(np.clip(gap_med, 0.5, gap_cap))
    on_graded_total = max(1, int(on_g.sum()))
    best = None
    for _ in range(10):
        frags = [_player_frags(Hps[p], segs[p], frag_med, fsig, p_gap, gap_med, gsig, gap_cap,
                               *streams[p]) for p in range(S0)]
        covered = sum(int(np.sum(Cg[np.asarray(e)] - Cg[np.asarray(s)])) for s, e in frags if s)
        cov = covered / on_graded_total
        if best is None or abs(cov - cov_t) < abs(best[0] - cov_t):
            best = (cov, gap_med, frags)
        if abs(cov - cov_t) < 0.001 or p_gap <= 0:
            break
        new = float(np.clip(gap_med * _odds(1 - cov_t) / _odds(1 - cov), 0.5, gap_cap))
        if new == gap_med:
            break
        gap_med = new
    cov, gap_med, frags = best
    del Hps

    # swaps: split both players' fragments at the encounter and exchange slots afterwards
    starts = [list(s) for s, _ in frags]
    stops = [list(e) for _, e in frags]
    take = [[None] * len(s) for s in starts]
    n_swaps = 0
    if enc_t.shape[0] and cfg["swap_prob"] > 0:
        pick = r_swap.random(enc_t.shape[0]) < cfg["swap_prob"]
        order = np.argsort(enc_t[pick], kind="stable")
        for t, i, j in zip(enc_t[pick][order].tolist(), enc_i[pick][order].tolist(),
                           enc_j[pick][order].tolist()):
            ki = bisect.bisect_right(starts[i], t) - 1
            kj = bisect.bisect_right(starts[j], t) - 1
            if ki < 0 or kj < 0:
                continue
            if not (starts[i][ki] < t < stops[i][ki] and starts[j][kj] < t < stops[j][kj]):
                continue
            for p, k, q in ((i, ki, j), (j, kj, i)):
                starts[p].insert(k + 1, t)
                stops[p].insert(k + 1, stops[p][k])
                stops[p][k] = t
                take[p].insert(k + 1, (q, t))
            n_swaps += 1

    # ---- assemble every fragment -----------------------------------------------------------------
    f_src, f_kind, f_s, f_e, f_team, f_take = [], [], [], [], [], []
    key = {}
    for p in range(S0):
        for s, e, tk in zip(starts[p], stops[p], take[p]):
            key[(p, e)] = len(f_s)
            f_src.append(p); f_kind.append(0); f_s.append(s); f_e.append(e)
            f_team.append(int(team0[p])); f_take.append(tk)
    dup_off = {}
    for p, s, e, o in dups:
        dup_off[len(f_s)] = o
        f_src.append(p); f_kind.append(1); f_s.append(s); f_e.append(e)
        f_team.append(int(team0[p])); f_take.append(None)
    for s, e, tm in ghosts:
        f_src.append(-1); f_kind.append(2); f_s.append(s); f_e.append(e)
        f_team.append(int(tm)); f_take.append(None)
    nf = len(f_s)
    f_s_a = np.asarray(f_s, dtype=np.int64)
    f_e_a = np.asarray(f_e, dtype=np.int64)
    f_kind_a = np.asarray(f_kind, dtype=np.int8)
    flip = (r_slot.random(nf) < cfg["team_flip_rate"]) & (f_kind_a != 2)
    f_team_a = np.asarray(f_team, dtype=np.int8) ^ flip.astype(np.int8)
    take_idx = [key.get(tk, -1) if tk is not None else -1 for tk in f_take]
    u_slot = r_slot.random(nf).tolist()

    # ---- slot assignment -------------------------------------------------------------------------
    NS = int(cfg["n_slots"])
    n_home = int(np.clip(round(NS * cfg["home_slot_frac"]), 1, NS - 1))
    team_out = np.array([0] * n_home + [1] * (NS - n_home), dtype=np.int8)
    pools = [list(range(n_home)), list(range(n_home, NS))]
    busy = [-(10 ** 12)] * NS  # exclusive stop of the slot's last occupant
    slot_of = [-1] * nf
    policy = cfg["slot_policy"]
    n_dropped = 0
    order = np.lexsort((f_kind_a, f_s_a)).tolist()
    ts, te = f_s_a.tolist(), f_e_a.tolist()
    tteam = f_team_a.tolist()
    for f in order:
        s, tm = ts[f], tteam[f]
        sl = -1
        tf = take_idx[f]
        if tf >= 0:
            cand = slot_of[tf]
            if cand >= 0 and team_out[cand] == tm and busy[cand] == s:
                sl = cand
        if sl < 0:
            free = [k for k in pools[tm] if busy[k] < s]  # at least one empty frame in between
            if not free:
                n_dropped += 1
                continue
            if policy == "lowest":
                sl = free[0]
            elif policy == "lru":
                sl = min(free, key=busy.__getitem__)
            else:
                sl = free[int(u_slot[f] * len(free))]
        slot_of[f] = sl
        busy[sl] = te[f]

    # ---- (e) positions ---------------------------------------------------------------------------
    P_out = np.full((n, NS, 2), np.nan, dtype=np.float32)
    bias = r_pos.normal(0.0, cfg["frag_bias_sd_m"], size=(nf, 2)).astype(np.float32)
    if return_truth:
        slot_player = np.full((n, NS), -1, dtype=np.int16)
        slot_kind = np.full((n, NS), -1, dtype=np.int8)
    gs = cfg["ghost_speed_mps"]
    for f in range(nf):
        sl = slot_of[f]
        if sl < 0:
            continue
        s, e, k = ts[f], te[f], f_kind[f]
        if k == 2:
            L = e - s
            v = _ar1_noise(r_ghost, (L, 2), gs, 0.97)
            start = r_ghost.uniform([-L_pitch / 2 - 3, -W_pitch / 2 - 3], [L_pitch / 2 + 3, W_pitch / 2 + 3])
            P_out[s:e, sl] = (start + np.cumsum(v, axis=0) / fps).astype(np.float32) + bias[f]
        else:
            src = P0[s:e, f_src[f]]
            off = dup_off.get(f)
            P_out[s:e, sl] = src + (bias[f] if off is None else bias[f] + off)
        if return_truth:
            slot_player[s:e, sl] = f_src[f] if k != 2 else -2
            slot_kind[s:e, sl] = k
    if cfg["pos_jitter_sd_m"] > 0:
        P_out += r_pos.standard_normal((n, NS, 2), dtype=np.float32) * np.float32(cfg["pos_jitter_sd_m"])
    if cfg["pos_wobble_sd_m"] > 0:
        P_out += _ar1_noise(r_pos, (n, NS, 2), cfg["pos_wobble_sd_m"], cfg["pos_wobble_rho"])

    # ---- (f) speed -------------------------------------------------------------------------------
    speed_out = _slot_speed(P_out, fps, max(0, int(round(cfg["speed_win_s"] * fps))))

    pres_out = np.isfinite(P_out[..., 0])
    summary = {
        "n_fragments": int(nf - n_dropped),
        "n_dropped": int(n_dropped),
        "n_swaps": int(n_swaps),
        "n_duplicates": len(dups),
        "n_ghosts": len(ghosts),
        "coverage": float(cov),
        "gap_median_s": float(gap_med / fps),
        "ball_present_rate": float(observed[graded].mean()) if graded.any() else float("nan"),
        "players_present_mean": float(pres_out[graded].sum(1).mean()) if graded.any() else float("nan"),
        **ball_info,
    }
    meta, dropped = _strip_identity_meta(m.meta, n, S0)
    meta["noise"] = {"cfg": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
                     "seed": int(seed), "clean": m.name, "summary": summary,
                     "dropped_meta": dropped}
    out = Match(P=P_out, team=team_out, speed=speed_out, ball=ball_out.astype(np.float32),
                half_blocks=list(m.half_blocks), attack_dir=dict(m.attack_dir),
                y=None if m.y is None else m.y.copy(), name=f"{m.name}+noise{int(seed)}",
                fps=m.fps, pitch=tuple(m.pitch), meta=meta)
    out.validate()
    if return_truth:
        return out, {"slot_player": slot_player, "slot_kind": slot_kind, "ball_observed": observed}
    return out
