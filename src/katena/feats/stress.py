"""Cheap ATD-like degradations of a clean Match, for feature tests, benchmarks and smoke checks.

Not the D2 noise simulator (`katena.noise.corrupt`), which is fitted to the target. These are
fixed, hand-set stressors: identity fragmentation over 46 slots, dropouts, duplicates and ghosts,
and bursty ball loss biased towards crowded moments. Deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..match import A, H, Match


def swap_match_teams(m: Match) -> Match:
    """The same match with home and away relabelled (positions unchanged)."""
    y = None
    if m.y is not None:
        y = m.y.copy()
        y[m.y == H], y[m.y == A] = A, H
    return replace(m, team=(1 - m.team).astype(np.int8),
                   attack_dir={h: -d for h, d in m.attack_dir.items()}, y=y,
                   name=m.name + ":swapped", meta=dict(m.meta))


def fragment_slots(m: Match, n_slots: int = 46, mean_frag_s: float = 20.0,
                   drop_prob: float = 0.35, mean_drop_s: float = 1.5, dup_per_min: float = 1.0,
                   dup_s: float = 2.0, ghost_per_min: float = 0.5, ghost_s: float = 3.0,
                   seed: int = 0) -> Match:
    """Spread each player's track over short-lived anonymous slots, like ATD.

    Every player track is cut into fragments (exponential lengths, mean `mean_frag_s`); after a
    fragment the player vanishes with probability `drop_prob` for an exponential gap. Each fragment
    lands in a random free slot of the player's team, so slot identity means nothing over long
    windows. Duplicates copy a player into a second slot with a small offset; ghosts are short
    spurious tracks. Team pools are interleaved in slot order.
    """
    rng = np.random.default_rng(seed)
    n, fps = m.n, m.fps
    L, W = m.pitch
    perm = rng.permutation(n_slots)
    team_new = np.zeros(n_slots, dtype=np.int8)
    team_new[perm[n_slots // 2:]] = 1
    pools = {t: np.flatnonzero(team_new == t) for t in (0, 1)}
    busy = np.full(n_slots, -1, dtype=np.int64)  # last frame each slot is occupied
    P = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    S = np.full((n, n_slots), np.nan, dtype=np.float32)

    events = []  # (start, stop, kind, player)
    for p in range(m.n_slots):
        t = 0
        while t < n:
            ln = max(fps, int(rng.exponential(mean_frag_s * fps)))
            events.append((t, min(n, t + ln), "frag", p))
            t += ln
            if rng.random() < drop_prob:
                t += max(1, int(rng.exponential(mean_drop_s * fps)))
    minutes = n / fps / 60
    for _ in range(rng.poisson(dup_per_min * minutes)):
        t0 = int(rng.integers(0, n))
        events.append((t0, min(n, t0 + max(2, int(rng.exponential(dup_s * fps)))), "dup",
                       int(rng.integers(0, m.n_slots))))
    for _ in range(rng.poisson(ghost_per_min * minutes)):
        t0 = int(rng.integers(0, n))
        events.append((t0, min(n, t0 + max(2, int(rng.exponential(ghost_s * fps)))), "ghost",
                       int(rng.integers(0, 2))))
    events.sort(key=lambda e: (e[0], e[2], e[3]))

    for t0, t1, kind, p in events:  # one iteration per fragment / event, not per frame
        tm = p if kind == "ghost" else int(m.team[p])
        free = pools[tm][busy[pools[tm]] < t0]
        if len(free) == 0:
            continue
        s = int(rng.choice(free))
        busy[s] = t1 - 1
        if kind == "ghost":
            start = np.array([rng.uniform(-L / 2, L / 2), rng.uniform(-W / 2, W / 2)])
            walk = np.cumsum(rng.normal(0, 0.15, size=(t1 - t0, 2)), axis=0) + start
            P[t0:t1, s] = walk.astype(np.float32)
            S[t0:t1, s] = np.float32(1.5)
            continue
        P[t0:t1, s] = m.P[t0:t1, p]
        S[t0:t1, s] = m.speed[t0:t1, p]
        if kind == "dup":
            P[t0:t1, s] += rng.normal(0, 0.15, size=2).astype(np.float32)
    out = replace(m, P=P, speed=S, team=team_new, name=m.name + ":frag", meta=dict(m.meta))
    out.validate()
    return out


def drop_ball_bursts(m: Match, target: float = 0.41, mean_burst_s: float = 1.5,
                     crowd_weight: float = 1.0, fast_weight: float = 2.0, seed: int = 0) -> Match:
    """NaN out the ball in bursts until about `target` of graded frames have no ball.

    Burst starts are drawn with weight 1 + crowd_weight * (players within 5 m of the ball)
    + fast_weight * (ball faster than 10 m/s), so loss concentrates in duels, crowded areas and
    passes. Burst lengths are lognormal with mean about `mean_burst_s`.
    """
    rng = np.random.default_rng(seed)
    n = m.n
    ball = m.ball.copy()
    g = m.graded_mask()
    d = np.linalg.norm(m.P - ball[:, None, :], axis=-1)
    crowd = np.nansum(d < 5.0, axis=1)
    sp = np.zeros(n)
    sp[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=-1) * m.fps
    w = (1.0 + crowd_weight * crowd + fast_weight * (np.nan_to_num(sp) > 10.0)) * g
    w = w / w.sum()
    gone = ~np.isfinite(ball).all(axis=1)
    n_graded = max(1, int(g.sum()))
    sigma = 0.8
    mu = np.log(mean_burst_s * m.fps) - sigma ** 2 / 2
    while (deficit := target * n_graded - (gone & g).sum()) > 0:
        k = int(np.clip(deficit / (mean_burst_s * m.fps) / 2, 1, 64))
        starts = rng.choice(n, size=k, p=w)
        lens = np.maximum(1, rng.lognormal(mu, sigma, size=k).astype(np.int64))
        for s0, ln in zip(starts, lens):  # bursts, not frames
            gone[s0:s0 + ln] = True
    ball[gone] = np.nan
    return replace(m, ball=ball, name=m.name + ":balldrop", meta=dict(m.meta))


def duplicate_slot(m: Match, src: int, t0: int, t1: int, offset=(0.1, -0.1)) -> Match:
    """Copy slot `src` over frames [t0, t1) into one new slot of the same team (a tracker dup)."""
    n, S = m.n, m.n_slots
    P = np.concatenate([m.P, np.full((n, 1, 2), np.nan, np.float32)], axis=1)
    sp = np.concatenate([m.speed, np.full((n, 1), np.nan, np.float32)], axis=1)
    P[t0:t1, S] = m.P[t0:t1, src] + np.asarray(offset, np.float32)
    sp[t0:t1, S] = m.speed[t0:t1, src]
    team = np.concatenate([m.team, m.team[src:src + 1]]).astype(np.int8)
    return replace(m, P=P, speed=sp, team=team, meta=dict(m.meta))
