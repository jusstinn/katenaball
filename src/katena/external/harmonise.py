"""Label harmonisation: touch-level possession labels -> "controlled" possession.

The competition ground truth changes possession only when the other team gains *controlled*
possession (CHALLENGE.md, "Possession rules"): a touch without control, a deflection, or an
aerial duel won without subsequent control is not a change, and while the ball is loose it stays
with the last team that had it. Its suggested proxies are "a player keeps the ball for a few
seconds" and "a team completes more than two consecutive passes". After a stoppage the
restarting team is a clear possessor.

DFL's BallPossession (and labels built from Metrica events) flip on single touches instead. The
rule here removes those flips:

    An opponent spell shorter than `min_control_s` that starts straight from the current
    owner's spell (no dead ball in between) is relabelled to that owner. Such a spell can only
    be followed by the owner again ("it came back") or by dead ball / the end of the half
    ("it went out"), so this is exactly: brief flips that return, or that die, are not changes.

    A spell that starts right after dead ball (or at the start of a half) is the restart
    possessor and is never relabelled, however short.

    Flips are removed shortest first, so a chain like H | A 1.0 s | H 0.3 s | A 10 s resolves by
    first discarding the 0.3 s H touch (A keeps the ball through it) rather than depending on
    scan direction. `order="causal"` scans left to right instead, for comparison.

Dead-ball frames are never changed, so the D class and every stoppage boundary are untouched.
How `min_control_s` was chosen is recorded in docs/external_data.md and docs/decision-log.md.
"""
from __future__ import annotations

import numpy as np

from ..match import A, D, FPS, H, X

DEFAULT_HARMONISE = {
    # Derived from external event data, not from the competition match: the duration that best
    # separates removable flips with and without a deliberate on-ball action by that team
    # (pooled 8 matches 3.7 s, leave-one-match-out 3.5-3.8 s, match bootstrap 90% [2.7, 4.0]),
    # rounded to 0.5 s. See docs/external_data.md, "Harmonisation".
    "min_control_s": 3.5,
    "fps": FPS,
    "protect_restart": True,  # the first spell after dead ball always survives
    "order": "shortest_first",
}

_TEAMS = (H, A)


def _runs(y: np.ndarray) -> list[tuple[int, int]]:
    """Half-open spans of non-X frames (the scorer's valid runs)."""
    graded = (y != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _segments(window: np.ndarray) -> list[list[int]]:
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)])).tolist()
    return [[bounds[i], bounds[i + 1], int(window[bounds[i]])] for i in range(len(bounds) - 1)]


def _candidate(segs, i, min_len, protect_restart):
    """Target code if segment i is a removable flip, else None."""
    a, b, k = segs[i]
    if k not in _TEAMS or b - a >= min_len:
        return None
    prev = segs[i - 1][2] if i > 0 else None
    if prev in _TEAMS:  # came straight from the other team's spell
        return prev
    if protect_restart:
        return None
    nxt = segs[i + 1][2] if i + 1 < len(segs) else None
    return nxt if nxt in _TEAMS else None


def _merge_at(segs, i):
    """Merge segment i with same-code neighbours in place; return the merged index."""
    if i + 1 < len(segs) and segs[i + 1][2] == segs[i][2]:
        segs[i][1] = segs[i + 1][1]
        del segs[i + 1]
    if i > 0 and segs[i - 1][2] == segs[i][2]:
        segs[i - 1][1] = segs[i][1]
        del segs[i]
        i -= 1
    return i


def _absorb(segs, min_len, protect_restart, order):
    if order == "causal":
        i = 0
        while i < len(segs):
            target = _candidate(segs, i, min_len, protect_restart)
            if target is None:
                i += 1
                continue
            segs[i][2] = target
            i = _merge_at(segs, i)
        return segs
    if order != "shortest_first":
        raise ValueError(f"unknown order {order!r}")
    while True:
        best = None
        for i in range(len(segs)):
            target = _candidate(segs, i, min_len, protect_restart)
            if target is None:
                continue
            key = (segs[i][1] - segs[i][0], segs[i][0])  # shortest, then earliest
            if best is None or key < best[0]:
                best = (key, i, target)
        if best is None:
            return segs
        _, i, target = best
        segs[i][2] = target
        _merge_at(segs, i)


def harmonise(y: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Copy of `y` (int8 codes 0 H 1 A 2 D 3 X) with brief opponent flips removed.

    Works inside each run of non-X frames separately, so nothing crosses halftime. D and X frames
    are never modified; only H/A frames can change team.
    """
    cfg = {**DEFAULT_HARMONISE, **(cfg or {})}
    min_len = int(round(float(cfg["min_control_s"]) * int(cfg["fps"])))
    y = np.asarray(y)
    out = y.astype(np.int8, copy=True)
    if min_len <= 1:
        return out
    for start, stop in _runs(out):
        segs = _absorb(_segments(out[start:stop]), min_len, bool(cfg["protect_restart"]),
                       cfg["order"])
        for a, b, k in segs:
            out[start + a:start + b] = k
    return out


def flip_table(y: np.ndarray, fps: int = FPS) -> dict:
    """Durations (s) of possession spells by context, for deriving `min_control_s`.

    - "return":  spell between two spells of the other team (T x T)
    - "to_dead": spell between the other team and dead ball / end of run (T x D)
    - "restart": spell right after dead ball or at the start of a run
    - "other":   everything else (a spell between two different contexts, e.g. T x T')
    """
    out = {"return": [], "to_dead": [], "restart": [], "other": []}
    for start, stop in _runs(np.asarray(y)):
        segs = _segments(np.asarray(y)[start:stop])
        for i, (a, b, k) in enumerate(segs):
            if k not in _TEAMS:
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev not in _TEAMS:
                kind = "restart"
            elif nxt == prev:
                kind = "return"
            elif nxt in (D, None):
                kind = "to_dead"
            else:
                kind = "other"
            out[kind].append((b - a) / fps)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
