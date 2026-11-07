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

Two optional mechanisms (D2, 2026-11-07) are off by default, so every existing config gives
byte-identical output for the same seed (``tests/test_d2_degrader.py`` checks this against a frozen
copy of the previous module):

* **Dead-span ball loss** (``dead_miss >= 0``, needs dead labels). The ball disappears an exponential
  lag (``dead_lag_s``) after a stoppage and stays lost until the restart, on a share of the dead
  spans solved so that P(ball missing | dead) hits ``dead_miss``. Alive-time loss keeps the bursty
  hazard above, with onsets on alive frames only, solved so the overall missing fraction still hits
  ``ball_missing``. If the two cannot both hold, alive frames keep at least ``alive_floor`` and the
  dead share gives way. ``dead_odds`` is ignored on this path.
* **Encounter identity switches** (``frag_cong_coef``, ``swap_prob`` or ``xswap_prob`` > 0), ported
  from the D2 noise simulator on ``david/foundation`` (``katena.noise.sim``). Fragment cuts run on a
  clock that speeds up per nearby player (``occl_radius_m``), so slots change hands in crowds. Two
  players of the same team who come within ``swap_dist_m`` swap slots with ``swap_prob``. Two
  opponents swap with ``xswap_prob``: each slot then follows a player of the other team until the
  tracker notices, after an exponential ``xswap_hold_s``, and the player is re-acquired in a slot of
  his own team. Presence is unchanged, so the tracked-object count stays where ``player_avail`` puts it.
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
    # D2 options (module docstring). The defaults switch them off and keep the output unchanged.
    dead_miss: float = -1.0       # target P(ball missing | dead) on in-play frames; < 0 = off
    dead_lag_s: float = 1.4       # mean delay from a stoppage to the ball loss (exponential)
    alive_floor: float = 0.015    # least P(ball missing | alive) when dead_miss must give way
    frag_cong_coef: float = 0.0   # fragment-cut log-rate per other player within occl_radius_m (cap 4)
    occl_radius_m: float = 1.5
    swap_prob: float = 0.0        # P(slot swap) when two team-mates come within swap_dist_m
    xswap_prob: float = 0.0       # P(slot swap) when two opponents come within swap_dist_m
    swap_dist_m: float = 1.0
    xswap_hold_s: float = 3.0     # mean time a swapped player stays in the other team's slot


MILD = DegradeCfg("mild", 0.20, 0.96, 0.3, 120.0, 0.15, 0.30, 0.002, 0.0)
TARGET = DegradeCfg("target", 0.41, 0.90, 0.5, 60.0, 0.30, 0.60, 0.005, 0.01)
SEVERE = DegradeCfg("severe", 0.55, 0.78, 0.9, 30.0, 0.50, 1.00, 0.010, 0.03)
CONFIGS = {c.name: c for c in (MILD, TARGET, SEVERE)}


def with_dead_odds(cfg: DegradeCfg, odds: float) -> DegradeCfg:
    return dataclasses.replace(cfg, name=f"{cfg.name}_dead{odds:g}", dead_odds=odds)


def dead_coupling_on(cfg: DegradeCfg) -> bool:
    """True when ``cfg`` uses the dead-span ball loss instead of the legacy dropout."""
    return cfg.dead_miss >= 0.0


def identity_switches_on(cfg: DegradeCfg) -> bool:
    """True when ``cfg`` uses the encounter identity model instead of the legacy fragments."""
    return cfg.frag_cong_coef != 0.0 or cfg.swap_prob > 0.0 or cfg.xswap_prob > 0.0


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
# players: encounter identity switches (D2; only when identity_switches_on(cfg))

_CONG_CAP = 4                        # crowding counts are capped here before entering a log-rate


def _close_pairs(P: np.ndarray, r_occ: float, r_swap: float, chunk: int = 2048):
    """For clean positions ``P[n, S, 2]``: how many other people are within ``r_occ`` of each person
    at each sample (int16[n, S]), and the onsets (t, i, j), i < j, ordered by t, of pairs that come
    within ``r_swap`` (closer now, not closer at the previous sample)."""
    n, S = P.shape[:2]
    cong = np.zeros((n, S), dtype=np.int16)
    iu, ju = np.triu_indices(S, 1)
    if len(iu) == 0:
        z = np.zeros(0, dtype=np.int64)
        return cong, (z, z, z)
    inc = np.zeros((len(iu), S), dtype=np.float32)          # pair -> its two people
    inc[np.arange(len(iu)), iu] = 1.0
    inc[np.arange(len(iu)), ju] = 1.0
    prev = np.zeros(len(iu), dtype=bool)
    ts, ps = [], []
    for a in range(0, n, chunk):
        Q = P[a:a + chunk].astype(np.float64)
        d = Q[:, iu] - Q[:, ju]
        with np.errstate(invalid="ignore"):
            d2 = d[..., 0] ** 2 + d[..., 1] ** 2
            occ = d2 < r_occ * r_occ
            close = d2 < r_swap * r_swap
        cong[a:a + chunk] = np.rint(occ.astype(np.float32) @ inc).astype(np.int16)
        onset = close & ~np.vstack([prev[None], close[:-1]])
        t, p = np.nonzero(onset)
        ts.append(t + a)
        ps.append(p)
        prev = close[-1]
    t, p = np.concatenate(ts), np.concatenate(ps)
    order = np.argsort(t, kind="stable")
    return cong, (t[order].astype(np.int64), iu[p[order]].astype(np.int64), ju[p[order]].astype(np.int64))


def _person_runs_clock(rng, pres: np.ndarray, dt: float, cfg: DegradeCfg, cal: dict, cong_p: np.ndarray):
    """``_person_runs`` with the fragment cuts on a crowding clock: it runs exp(frag_cong_coef * people
    within occl_radius_m) times faster, normalised to mean 1 over the person's samples, so the cut rate
    per match stays 1 / frag_life_s while the cuts move towards contested moments. Returns a list of
    [start, stop, take] with take None (filled in by the swaps)."""
    n = len(pres)
    # observed / lost stretches: the same process as _person_runs
    gap = cal["player_gap_pooled"]
    probe = np.exp(np.random.default_rng(0).normal(gap["mu"], gap["sigma"], 20000))
    mean_gap = float(np.minimum(probe, 60.0).mean())
    a = cfg.player_avail
    mean_obs = mean_gap * a / (1.0 - a)
    obs = np.zeros(n, dtype=bool)
    t = 0
    observed = bool(rng.random() < a)
    while t < n:
        if observed:
            length = max(1, int(rng.exponential(mean_obs) / dt))
            obs[t:t + length] = True
        else:
            length = max(1, int(np.minimum(_lognormal_samples(rng, gap["mu"], gap["sigma"], 1)[0], 60.0) / dt))
        t += length
        observed = not observed
    obs &= pres
    rate = np.exp(cfg.frag_cong_coef * np.minimum(cong_p, _CONG_CAP).astype(np.float64))
    norm = float(rate[pres].mean()) if pres.any() else 1.0
    clock = np.cumsum(rate / norm)
    runs = []
    edges = np.diff(np.concatenate(([0], obs.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        cur = int(s)
        while cur < e:
            base = clock[cur - 1] if cur > 0 else 0.0
            stop = int(np.searchsorted(clock, base + rng.exponential(cfg.frag_life_s) / dt)) + 1
            stop = max(stop, cur + 2)
            runs.append([cur, int(min(e, stop)), None])
            cur = stop
    return runs


def _frag_at(frags: list, t: int) -> int:
    """Index of the fragment with start < t < stop, or -1."""
    for k, (s, e, _) in enumerate(frags):
        if s < t < e:
            return k
        if s >= t:
            break
    return -1


def _apply_swaps(frags: dict, team: np.ndarray, enc, cfg: DegradeCfg, rng, dt: float) -> tuple[int, int]:
    """Swap slots at encounter onsets, in time order. ``frags``: {person: start-sorted [start, stop,
    take]}. A swap at t splits both people's fragments at t, and each continuation takes the slot of
    the other's fragment that ends at t (take = (other person, t)). After a swap between opponents
    each continuation is cut again after the hold, and the rest is a fresh fragment of the player's
    own team. Returns (team-mate swaps, opponent swaps)."""
    enc_t, enc_i, enc_j = enc
    u = rng.random(len(enc_t))
    hold = rng.exponential(cfg.xswap_hold_s / dt, len(enc_t))
    n_same = n_opp = 0
    for k in range(len(enc_t)):
        t, i, j = int(enc_t[k]), int(enc_i[k]), int(enc_j[k])
        same = bool(team[i] == team[j])
        if u[k] >= (cfg.swap_prob if same else cfg.xswap_prob) or i not in frags or j not in frags:
            continue
        fi, fj = _frag_at(frags[i], t), _frag_at(frags[j], t)
        if fi < 0 or fj < 0:
            continue
        si, ei, tki = frags[i][fi]
        sj, ej, tkj = frags[j][fj]
        frags[i][fi] = [si, t, tki]
        frags[j][fj] = [sj, t, tkj]
        cont_i, cont_j = [[t, ei, (j, t)]], [[t, ej, (i, t)]]
        if same:
            n_same += 1
        else:
            n_opp += 1
            h = max(1, int(round(hold[k])))
            if t + h < ei:
                cont_i = [[t, t + h, (j, t)], [t + h, ei, None]]
            if t + h < ej:
                cont_j = [[t, t + h, (i, t)], [t + h, ej, None]]
        frags[i][fi + 1:fi + 1] = cont_i
        frags[j][fj + 1:fj + 1] = cont_j
    return n_same, n_opp


def _identity_runs(rng, m: Match, cfg: DegradeCfg, cal: dict):
    """Fragments of every person under the encounter identity model.

    Returns (runs, takes, n_frag, info): runs as in ``corrupt`` (start, stop, team, xy), ``takes[k]``
    the index of the run whose slot run k continues in (-1 for a fresh slot), and swap counters."""
    n, S, _ = m.P.shape
    dt = 1.0 / m.fps
    pres = np.isfinite(m.P[..., 0])
    cong, enc = _close_pairs(m.P, cfg.occl_radius_m, cfg.swap_dist_m)
    frags = {}
    for s in range(S):
        if pres[:, s].any():
            frags[s] = _person_runs_clock(rng, pres[:, s], dt, cfg, cal, cong[:, s])
    n_same, n_opp = _apply_swaps(frags, m.team, enc, cfg, rng, dt)
    runs, refs, index = [], [], {}
    for p in sorted(frags):
        for a, b, take in frags[p]:
            if b - a < 2:
                continue
            index[(p, b)] = len(runs)
            runs.append((a, b, int(m.team[p]), m.P[a:b, p].copy()))
            refs.append(take)
    flips = rng.random(len(runs)) < cfg.team_flip
    takes = [index.get(r, -1) if r is not None else -1 for r in refs]
    runs = [(a, b, (1 - tm) if (flips[k] and takes[k] < 0) else tm, xy)
            for k, (a, b, tm, xy) in enumerate(runs)]
    info = {"swaps_same": int(n_same), "swaps_opp": int(n_opp), "encounters": int(len(enc[0]))}
    return runs, takes, len(runs), info


def _assign_slots_take(rng, runs, takes, n):
    """``_assign_slots`` for the identity model. A run whose ``takes`` entry names an earlier run
    continues in that run's slot when the slot was released exactly at its start (a seamless
    hand-over: the slot's track jumps from one person to another, and to the other team's player
    after an opponent swap). Every other run gets a random free slot of its team, as before."""
    P = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    team = np.repeat(np.array([0, 1], dtype=np.int8), SLOTS_PER_TEAM)
    free_at = np.zeros(2 * SLOTS_PER_TEAM, dtype=np.int64)
    last_stop = np.full(2 * SLOTS_PER_TEAM, -1, dtype=np.int64)
    slot_of = np.full(len(runs), -1, dtype=np.int64)
    n_take = 0
    for k in sorted(range(len(runs)), key=lambda r: (runs[r][0], runs[r][1])):
        s, e, tm, xy = runs[k]
        slot = -1
        if takes[k] >= 0 and slot_of[takes[k]] >= 0 and last_stop[slot_of[takes[k]]] == s:
            slot = int(slot_of[takes[k]])
            n_take += 1
        else:
            base = tm * SLOTS_PER_TEAM
            ok = np.flatnonzero(free_at[base:base + SLOTS_PER_TEAM] <= s)
            if len(ok) == 0:
                continue
            slot = base + int(rng.choice(ok))
        P[s:e, slot] = xy
        free_at[slot] = e + 1
        last_stop[slot] = e
        slot_of[k] = slot
    return P, team, n_take


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


def _runs_in_blocks(mask: np.ndarray, m: Match) -> tuple[np.ndarray, np.ndarray]:
    """Half-open (starts, stops) of the True runs of ``mask``, never crossing a half-block edge."""
    starts, stops = [], []
    for _, a, b in m.half_blocks:
        edges = np.diff(np.concatenate(([0], mask[a:b + 1].astype(np.int8), [0])))
        starts.append(np.flatnonzero(edges == 1) + a)
        stops.append(np.flatnonzero(edges == -1) + a)
    if not starts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(starts).astype(np.int64), np.concatenate(stops).astype(np.int64)


def ball_dropout_coupled(m: Match, cfg: DegradeCfg, rng: np.random.Generator, cal: dict) -> np.ndarray:
    """bool[n]: True where the ball is dropped (in addition to what is already NaN), with ball loss
    tied to dead ball (``cfg.dead_miss >= 0``; see the module docstring).

    Dead part: in each dead span the ball is lost from an exponential lag (mean ``dead_lag_s``) after
    the stoppage until the restart, on spans taken in a random order until P(missing | dead) reaches
    the target. Alive part: the bursts of ``ball_dropout`` (same hazard shape and durations), with
    onsets on alive frames only (a burst may run on into a stoppage), level solved so the missing
    fraction of in-play frames hits ``ball_missing``. Two alternating passes, since each part spills
    into the other's count. Without dead labels only the alive part runs."""
    n, dt = m.n, 1.0 / m.fps
    blk = block_mask(m)
    native_nan = ~np.isfinite(m.ball[:, 0])
    dead = blk & (m.y == D) if m.y is not None else np.zeros(n, dtype=bool)
    alive = blk & ~dead
    # alive-time bursts: the hazard of ball_dropout
    ball = _ffill(m.ball, int(10 / dt))
    d = np.linalg.norm(m.P - ball[:, None, :], axis=-1)
    cong = np.minimum(np.nansum(d < 4.0, axis=1), 8) / 4.0
    sp = np.zeros(n)
    sp[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=-1) / dt
    sp = np.nan_to_num(np.minimum(sp, 30.0)) / 10.0
    logit = cfg.cong_coef * (cong - cong[blk].mean()) + cfg.speed_coef * (sp - sp[blk].mean())
    g_long, g_short = cal["sc_burst"], cal["short_burst"]
    use_short = rng.random(n) < cfg.short_share
    secs = np.where(use_short, _lognormal_samples(rng, g_short["mu"], g_short["sigma"], n),
                    _lognormal_samples(rng, g_long["mu"], g_long["sigma"], n))
    lens = np.clip(np.ceil(secs / dt), 1, int(120 / dt)).astype(np.int64)
    u = rng.random(n)
    # dead spans: lag after the stoppage, lost until the restart; switched on in a random order
    s0, e0 = _runs_in_blocks(dead, m)
    lag = np.floor(rng.exponential(cfg.dead_lag_s / dt, len(s0))).astype(np.int64)
    ls = np.minimum(s0 + lag, e0)
    order = rng.permutation(len(s0))

    def alive_part(base: float) -> np.ndarray:
        h = np.minimum(base * np.exp(logit), 0.9)
        drop = np.zeros(n, dtype=bool)
        nxt = 0
        for i in np.flatnonzero(alive & (u < h)):           # same walk as ball_dropout's run()
            if i >= nxt:
                drop[i:i + lens[i]] = True
                nxt = i + lens[i]
        return drop

    def dead_part(k: int, late: int = -1) -> np.ndarray:
        """Spans order[:k] from their lag to the restart, plus span order[k] from sample ``late``."""
        sel = order[:k]
        inc = np.zeros(n + 1, dtype=np.int32)
        np.add.at(inc, ls[sel], 1)
        np.add.at(inc, e0[sel], -1)
        if late >= 0 and k < len(order):
            inc[late] += 1
            inc[e0[order[k]]] -= 1
        return np.cumsum(inc[:-1]) > 0

    r = cfg.ball_missing
    n_blk, n_dead = int(blk.sum()), int(dead.sum())
    if n_dead and n_blk:
        share = n_dead / n_blk
        floor = max(cfg.alive_floor, float(native_nan[alive].mean()) if alive.any() else 0.0)
        p_dead = min(cfg.dead_miss, (r - floor * (1.0 - share)) / share)
        p_dead = max(p_dead, float(native_nan[dead].mean()))
    A = np.zeros(n, dtype=bool)
    Dm = np.zeros(n, dtype=bool)
    for _ in range(2):
        if n_dead:
            # dead frames each span would add, given what is already missing; whole spans in the
            # random order while they fit, then the next span lost late enough to land on the target
            free = (dead & ~(A | native_nan)).astype(np.int64)
            cs = np.concatenate(([0], np.cumsum(free)))
            gain = cs[e0[order]] - cs[ls[order]]
            have = int((dead & (A | native_nan)).sum())
            reach = have + np.concatenate(([0], np.cumsum(gain)))      # dead frames lost after k spans
            want = p_dead * n_dead
            k, late = 0, -1
            if want > have:
                k = int(np.searchsorted(reach, want, side="right")) - 1
                if k < len(order):
                    j = order[k]
                    late = int(np.searchsorted(cs, cs[e0[j]] - (want - reach[k]), side="right")) - 1
                    late = int(np.clip(late, ls[j], e0[j]))
            Dm = dead_part(k, late)
        base_mask = Dm | native_nan
        if base_mask[blk].mean() >= r:
            A = np.zeros(n, dtype=bool)
            continue
        lo, hi = 1e-5, 0.5
        for _ in range(14):
            mid = np.sqrt(lo * hi)
            if (alive_part(mid) | base_mask)[blk].mean() < r:
                lo = mid
            else:
                hi = mid
        A = alive_part(np.sqrt(lo * hi))
    return (Dm | A) & ~native_nan


# ----------------------------------------------------------------------------------------------

def corrupt(m: Match, cfg: DegradeCfg, seed: int) -> Match:
    """Return a degraded copy of ``m`` (work-grid Match). Deterministic given ``seed``."""
    cal = _cal_cached()
    rng = np.random.default_rng(seed)
    n, S, _ = m.P.shape
    dt = 1.0 / m.fps
    blk = block_mask(m)
    pres = np.isfinite(m.P[..., 0])

    identity = identity_switches_on(cfg)
    id_info: dict = {}
    takes: list = []
    runs = []
    n_frag = 0
    if identity:
        runs, takes, n_frag, id_info = _identity_runs(rng, m, cfg, cal)
    else:
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
    ghosts = _ghost_runs(rng, n, dt, cfg, blk, m.P, m.team)
    runs += ghosts
    takes += [-1] * len(ghosts)

    # jitter, then slot assignment
    jittered = []
    for a, b, tm, xy in runs:
        jittered.append((a, b, tm, xy + rng.normal(0, cfg.sigma_player, xy.shape).astype(np.float32)))
    if identity:
        P_out, team, id_info["handovers"] = _assign_slots_take(rng, jittered, takes, n)
    else:
        P_out, team = _assign_slots(rng, jittered, n)
    P_out[~np.isfinite(P_out[..., 0])] = np.nan

    # ball
    coupled = dead_coupling_on(cfg)
    drop = ball_dropout_coupled(m, cfg, rng, cal) if coupled else ball_dropout(m, cfg, rng, cal)
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
    if coupled and m.y is not None:                        # D2 extras, only on the new paths
        miss = ~np.isfinite(ball[:, 0])
        for key, sel in (("ball_missing_dead", blk & (m.y == D)), ("ball_missing_alive", blk & (m.y < D))):
            meta["degrade"][key] = float(miss[sel].mean()) if sel.any() else float("nan")
    if identity:
        meta["degrade"].update(id_info)
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
