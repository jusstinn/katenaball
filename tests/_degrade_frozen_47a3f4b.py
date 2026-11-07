"""Make clean tracking look like automatic ATD (anonymous, fragmented, gappy, noisy).

``corrupt(m, cfg, seed)`` is a pure function of its inputs and the seed. It takes a Match on the
working grid (5 Hz) and returns a new Match with:

* identity stripped: every person is cut into anonymous track fragments (identity switches plus
  tracker loss gaps) that are re-assigned to 23 anonymous slots per team (46 slots, as in the target);
* player gaps with lognormal durations (SkillCorner calibration, ``calibration.json``);
* ghost objects and duplicates, tuned so the mean tracked-object count matches the config;
* bursty ball dropout whose onset hazard rises with congestion around the ball and with ball speed
  (flight), with lognormal burst durations; the global hazard level is solved so the realised ball
  missing fraction of in-play frames hits the config target (matches that are already missing the
  ball, e.g. Metrica and SkillCorner, are topped up, never reduced);
* positional jitter, occasional false-ball detections, and a small rate of wrong-team slots.

Labels, half blocks and attack directions are carried over untouched.

The coupling between ball loss and dead ball is a separate knob (``dead_odds``). The corpus supports
both a strong coupling (Metrica sample data: ball NaN on about 80 % of dead-label frames versus 2 % of
alive ones) and none (SkillCorner has no dead label); the default is no coupling.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from katena.external.common import derive_speed
from katena.grid import block_mask
from katena.types import D, Match

SLOTS_PER_TEAM = 23
CAL_PATH = Path(__file__).with_name("calibration.json")


@dataclass(frozen=True)
class DegradeCfg:
    name: str
    ball_missing: float           # target missing fraction of in-play frames
    player_avail: float           # observed share of person-time before ghosts
    ghost_mean: float             # mean concurrent ghost/duplicate objects
    frag_life_s: float            # mean time between identity switches
    sigma_player: float           # m
    sigma_ball: float             # m
    false_ball_rate: float        # per visible sample, starts a 1-3 sample false detection
    team_flip: float              # per fragment, wrong-team slot
    cong_coef: float = 1.0        # onset log-hazard per 4 players within 4 m of the ball
    speed_coef: float = 0.6       # onset log-hazard per 10 m/s of ball speed
    dead_odds: float = 1.0        # onset hazard multiplier while the ball is dead (needs labels)
    # burst mixture: share drawn from the long (SkillCorner) lognormal vs the short (Metrica-alive)
    short_share: float = 0.2


MILD = DegradeCfg("mild", 0.20, 0.96, 0.3, 120.0, 0.15, 0.30, 0.002, 0.0)
TARGET = DegradeCfg("target", 0.41, 0.90, 0.5, 60.0, 0.30, 0.60, 0.005, 0.01)
SEVERE = DegradeCfg("severe", 0.55, 0.78, 0.9, 30.0, 0.50, 1.00, 0.010, 0.03)
CONFIGS = {c.name: c for c in (MILD, TARGET, SEVERE)}


def with_dead_odds(cfg: DegradeCfg, odds: float) -> DegradeCfg:
    return dataclasses.replace(cfg, name=f"{cfg.name}_dead{odds:g}", dead_odds=odds)


def _cal() -> dict:
    return json.loads(CAL_PATH.read_text())


def _lognormal_samples(rng, mu, sigma, size):
    return np.exp(rng.normal(mu, sigma, size))


# ----------------------------------------------------------------------------------------------
# players

def _person_runs(rng, pres: np.ndarray, dt: float, cfg: DegradeCfg, cal: dict, blk: np.ndarray):
    """Observed fragments for one person: list of (start, stop) half-open sample ranges."""
    n = len(pres)
    gap = cal["player_gap_pooled"]
    probe = np.exp(np.random.default_rng(0).normal(gap["mu"], gap["sigma"], 20000))
    mean_gap = float(np.minimum(probe, 60.0).mean())            # E[min(gap, 60 s)], as drawn below
    a = cfg.player_avail
    mean_obs = mean_gap * a / (1.0 - a)
    obs = np.zeros(n, dtype=bool)
    t = 0
    observed = bool(rng.random() < a)                            # stationary start state
    while t < n:
        if observed:
            length = max(1, int(rng.exponential(mean_obs) / dt))
            obs[t:t + length] = True
        else:
            length = max(1, int(np.minimum(_lognormal_samples(rng, gap["mu"], gap["sigma"], 1)[0], 60.0) / dt))
        t += length
        observed = not observed
    obs &= pres
    # identity switches cut observed stretches into fragments
    runs = []
    edges = np.diff(np.concatenate(([0], obs.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        cur = s
        while cur < e:
            length = max(2, int(rng.exponential(cfg.frag_life_s) / dt))
            runs.append((int(cur), int(min(e, cur + length))))
            cur += length
    return runs


def _ghost_runs(rng, n, dt, cfg, blk, P_true, team):
    """Ghost and duplicate objects as (start, stop, team, xy[stop-start, 2])."""
    out = []
    if cfg.ghost_mean <= 0:
        return out
    mean_len_s = 5.0
    rate = cfg.ghost_mean / (mean_len_s / dt)                    # starts per sample
    starts = np.flatnonzero((rng.random(n) < rate) & blk)
    L, W = 105.0, 68.0
    for s in starts:
        length = int(np.clip(rng.lognormal(np.log(mean_len_s), 0.6) / dt, 2, 200 / dt))
        e = min(n, s + length)
        k = e - s
        if rng.random() < 0.5:                                   # duplicate of a real player
            cand = np.flatnonzero(np.isfinite(P_true[s, :, 0]))
            if len(cand):
                p = int(rng.choice(cand))
                xy = P_true[s:e, p].copy() + rng.normal(0, 1.0, 2).astype(np.float32)
                xy = np.where(np.isfinite(xy), xy, np.nan)
                out.append((int(s), int(e), int(team[p]), xy))
                continue
        start_xy = np.array([rng.uniform(0, L), rng.uniform(0, W)])
        if rng.random() < 0.5:
            xy = np.tile(start_xy, (k, 1))
        else:
            xy = start_xy + np.cumsum(rng.normal(0, 0.4, (k, 2)), axis=0)
        out.append((int(s), int(e), int(rng.integers(0, 2)), xy.astype(np.float32)))
    return out


def _assign_slots(rng, runs, n):
    """runs: list of (start, stop, team, xy). Returns P_out[n, 46, 2] and slot team."""
    P = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    team = np.repeat(np.array([0, 1], dtype=np.int8), SLOTS_PER_TEAM)
    free_at = np.zeros(2 * SLOTS_PER_TEAM, dtype=np.int64)
    for s, e, tm, xy in sorted(runs, key=lambda r: (r[0], r[1])):
        base = tm * SLOTS_PER_TEAM
        ok = np.flatnonzero(free_at[base:base + SLOTS_PER_TEAM] <= s)
        if len(ok) == 0:
            continue
        slot = base + int(rng.choice(ok))
        P[s:e, slot] = xy
        free_at[slot] = e + 1
    return P, team


# ----------------------------------------------------------------------------------------------
# ball

def _ffill(x: np.ndarray, limit: int) -> np.ndarray:
    idx = np.where(np.isfinite(x[:, 0]), np.arange(len(x)), -1)
    idx = np.maximum.accumulate(idx)
    ok = (idx >= 0) & (np.arange(len(x)) - idx <= limit)
    out = np.full_like(x, np.nan)
    out[ok] = x[idx[ok]]
    return out


def ball_dropout(m: Match, cfg: DegradeCfg, rng: np.random.Generator, cal: dict) -> np.ndarray:
    """bool[n]: True where the ball is dropped (in addition to what is already NaN)."""
    n, dt = m.n, 1.0 / m.fps
    blk = block_mask(m)
    native_nan = ~np.isfinite(m.ball[:, 0])
    ball = _ffill(m.ball, int(10 / dt))
    d = np.linalg.norm(m.P - ball[:, None, :], axis=-1)
    cong = np.minimum(np.nansum(d < 4.0, axis=1), 8) / 4.0
    sp = np.zeros(n)
    sp[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=-1) / dt
    sp = np.nan_to_num(np.minimum(sp, 30.0)) / 10.0
    logit = cfg.cong_coef * (cong - cong[blk].mean()) + cfg.speed_coef * (sp - sp[blk].mean())
    if cfg.dead_odds != 1.0 and m.y is not None:
        logit = logit + np.log(cfg.dead_odds) * (m.y == D)
    # burst durations (samples), common random numbers across bisection steps
    g_long, g_short = cal["sc_burst"], cal["short_burst"]
    use_short = rng.random(n) < cfg.short_share
    secs = np.where(use_short, _lognormal_samples(rng, g_short["mu"], g_short["sigma"], n),
                    _lognormal_samples(rng, g_long["mu"], g_long["sigma"], n))
    lens = np.clip(np.ceil(secs / dt), 1, int(120 / dt)).astype(np.int64)
    u = rng.random(n)
    target = cfg.ball_missing

    def run(base: float) -> np.ndarray:
        h = np.minimum(base * np.exp(logit), 0.9)
        drop = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if blk[i] and u[i] < h[i]:
                drop[i:i + lens[i]] = True
                i += lens[i]
            else:
                i += 1
        return drop

    def frac(base: float) -> float:
        return float((run(base) | native_nan)[blk].mean())

    if native_nan[blk].mean() >= target:
        return np.zeros(n, dtype=bool)
    lo, hi = 1e-5, 0.5
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return run(np.sqrt(lo * hi))


# ----------------------------------------------------------------------------------------------

def corrupt(m: Match, cfg: DegradeCfg, seed: int) -> Match:
    """Return a degraded copy of ``m`` (work-grid Match). Deterministic given ``seed``."""
    cal = _cal_cached()
    rng = np.random.default_rng(seed)
    n, S, _ = m.P.shape
    dt = 1.0 / m.fps
    blk = block_mask(m)
    pres = np.isfinite(m.P[..., 0])

    runs = []
    n_frag = 0
    for s in range(S):
        if not pres[:, s].any():
            continue
        for a, b in _person_runs(rng, pres[:, s], dt, cfg, cal, blk):
            if b - a < 2:
                continue
            tm = int(m.team[s])
            if rng.random() < cfg.team_flip:
                tm = 1 - tm
            runs.append((a, b, tm, m.P[a:b, s].copy()))
            n_frag += 1
    runs += _ghost_runs(rng, n, dt, cfg, blk, m.P, m.team)

    # jitter, then slot assignment
    jittered = []
    for a, b, tm, xy in runs:
        jittered.append((a, b, tm, xy + rng.normal(0, cfg.sigma_player, xy.shape).astype(np.float32)))
    P_out, team = _assign_slots(rng, jittered, n)
    P_out[~np.isfinite(P_out[..., 0])] = np.nan

    # ball
    drop = ball_dropout(m, cfg, rng, cal)
    ball = m.ball.copy()
    ball[drop] = np.nan
    vis = np.isfinite(ball[:, 0])
    ball[vis] += rng.normal(0, cfg.sigma_ball, (int(vis.sum()), 2)).astype(np.float32)
    starts = np.flatnonzero(vis & (rng.random(n) < cfg.false_ball_rate))
    for s in starts:
        k = int(rng.integers(1, 4))
        cand = np.flatnonzero(np.isfinite(P_out[s, :, 0]))
        if len(cand):
            p = int(rng.choice(cand))
            ball[s:s + k] = np.where(vis[s:s + k, None], P_out[s, p] + rng.normal(0, 1.0, 2), ball[s:s + k])

    speed = derive_speed(P_out, m.fps)
    meta = dict(m.meta)
    n_obj = np.isfinite(P_out[..., 0]).sum(1)
    meta["degrade"] = {
        "cfg": cfg.name, "seed": int(seed), "ball_missing": float((~np.isfinite(ball[:, 0]))[blk].mean()),
        "mean_objects": float(n_obj[blk].mean()), "fragments": int(n_frag),
        "slots_used": int(np.isfinite(P_out[..., 0]).any(0).sum()),
    }
    return dataclasses.replace(m, P=P_out, team=team, speed=speed, ball=ball.astype(np.float32),
                               meta=meta, extras={})


_CAL_CACHE: dict = {}


def _cal_cached() -> dict:
    if not _CAL_CACHE:
        raw = _cal()
        _CAL_CACHE.update(
            player_gap_pooled=raw["player_gap_pooled"],
            sc_burst=raw["sources"]["skillcorner"]["ball_gap"],
            short_burst=raw["ball_gap_metrica_alive_onset"])
    return _CAL_CACHE


def native_ball_missing(m: Match) -> float:
    return float((~np.isfinite(m.ball[:, 0]))[block_mask(m)].mean())
