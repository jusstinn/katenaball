"""Synthetic clean match with labels: the shared fixture for tests and development.

It is not a football simulator. It is just plausible enough that features, decoders and
loaders have something with the right shapes, units, label structure and failure modes to run
against: spells of a few to tens of seconds, dead-ball stretches with a stationary ball, a ball
that moves between teammates, and two halves with switched attack directions.

Deterministic for a given seed. Contains nothing derived from the competition match.
"""
from __future__ import annotations

import numpy as np

from .match import A, D, FPS, H, PITCH, X, Match


def _spell_plan(rng, n_frames, fps):
    """List of (start, stop, label) covering [0, n_frames). Alternates alive spells and stoppages."""
    out, t = [], 0
    label = int(rng.integers(0, 2))
    # kickoff: a short dead stretch first, like a real half
    k = int(rng.integers(2, 5) * fps)
    out.append((0, k, D))
    t = k
    while t < n_frames:
        dur = int(max(1.0, rng.gamma(1.6, 8.0)) * fps)  # mean ~13 s, long right tail
        out.append((t, min(n_frames, t + dur), label))
        t += dur
        if t >= n_frames:
            break
        if rng.random() < 0.3:
            dead = int(rng.uniform(4, 40) * fps)
            out.append((t, min(n_frames, t + dead), D))
            t += dead
            label = int(rng.integers(0, 2))  # either side restarts
        else:
            label = 1 - label  # turnover
    return out


def synthetic_match(seed: int = 0, half_minutes: float = 6.0, pre: int = 250, gap: int = 500,
                    post: int = 250, fps: int = FPS, pitch=PITCH, name: str | None = None) -> Match:
    """Two halves of `half_minutes` each, with ungraded padding before, between and after."""
    rng = np.random.default_rng(seed)
    L, W = pitch
    hn = int(half_minutes * 60 * fps)
    n = pre + hn + gap + hn + post
    half_blocks = [(1, pre, pre + hn - 1), (2, pre + hn + gap, pre + hn + gap + hn - 1)]
    home_dir = 1 if rng.random() < 0.5 else -1
    attack_dir = {1: home_dir, 2: -home_dir}

    S = 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    y = np.full(n, X, dtype=np.int8)
    for _, first, last in half_blocks:
        for s, e, lab in _spell_plan(rng, last - first + 1, fps):
            y[first + s:first + e] = lab
    # outside halves: half-time and padding are labelled X, players still on the pitch

    # formation templates in "attacking towards +x" frame, metres
    base = np.array([[-45, 0], [-30, -20], [-32, -7], [-32, 7], [-30, 20], [-12, -18],
                     [-14, 0], [-12, 18], [5, -22], [8, 0], [5, 22]], dtype=np.float32)

    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    half_of = np.zeros(n, dtype=np.int8)
    for half, first, last in half_blocks:
        half_of[first:last + 1] = half

    # ball "phase" position along x drives the block shift; random walk with possession drift
    bx, by = 0.0, 0.0
    jitter = rng.normal(0, 1, size=(S, 2)).astype(np.float32)
    carrier = int(rng.integers(0, 11))
    flight_left, flight_from, flight_to, flight_len = 0, None, None, 1
    for f in range(n):
        half = half_of[f] or (1 if f < half_blocks[1][1] else 2)
        hd = attack_dir[half]
        lab = y[f]
        alive = lab in (H, A)
        poss = 0 if lab == H else 1
        if alive:
            drift = 0.15 * (hd if poss == 0 else -hd)
            bx = float(np.clip(bx + drift + rng.normal(0, 0.25), -L / 2 + 3, L / 2 - 3))
            by = float(np.clip(by + rng.normal(0, 0.25), -W / 2 + 2, W / 2 - 2))
        shift = np.clip(bx, -25, 25)
        for s in range(S):
            tm = team[s]
            d = hd if tm == 0 else -hd
            bpos = base[s % 11] * np.array([d, 1], dtype=np.float32)
            jitter[s] += rng.normal(0, 0.08, 2).astype(np.float32)
            jitter[s] *= 0.995
            P[f, s] = bpos + np.array([shift, by * 0.3], dtype=np.float32) + 3 * jitter[s]

        if alive:
            mates = np.flatnonzero(team == poss)
            if carrier not in mates:
                carrier = int(rng.choice(mates))
                flight_left = 0
            if flight_left == 0 and rng.random() < 1.0 / (1.5 * fps):
                nxt = int(rng.choice(mates[mates != carrier]))
                flight_from, flight_to = carrier, nxt
                flight_len = flight_left = int(rng.uniform(0.4, 1.6) * fps)
                carrier = nxt
            if flight_left > 0:
                a = 1 - flight_left / flight_len
                ball[f] = (1 - a) * P[f, flight_from] + a * P[f, flight_to]
                flight_left -= 1
            else:
                ball[f] = P[f, carrier] + rng.normal(0, 0.4, 2).astype(np.float32)
            # pull the block centre towards the ball so it stays consistent
            bx = 0.9 * bx + 0.1 * float(ball[f, 0])
        else:
            if f > 0 and not np.isnan(ball[f - 1, 0]):
                ball[f] = ball[f - 1]  # stationary dead ball
            else:
                ball[f] = (0.0, 0.0)
            flight_left = 0

    speed = np.full((n, S), np.nan, dtype=np.float32)
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1) * fps
    speed[1:] = v.astype(np.float32)
    speed[0] = speed[1]

    m = Match(P=P, team=team, speed=speed, ball=ball, half_blocks=half_blocks,
              attack_dir=attack_dir, y=y, name=name or f"synthetic-{seed}", fps=fps,
              pitch=tuple(pitch), meta={"source": "synthetic", "seed": seed})
    m.validate()
    return m
