"""Alternative Metrica dead-ball derivation: stoppages derived backwards from their restarts.

Port of David's ``metrica_open.labels_from_events`` (origin/david/foundation) onto
``katena.external.metrica`` events, kept in its own module so the corpus loader (and its code digest
in ``scripts/build_external.py``) is untouched. ``katena.external.metrica.derive_labels`` stays the
default; this is the D4 candidate from the D5 audit (finding A3). Results: docs/label-convention.md,
section 8.

Differences from the default rule (a dead trigger makes the ball dead until the next SET PIECE or a
PASS tagged GOAL KICK):

* Every restart ends exactly one stoppage: any SET PIECE, or any event with a GOAL KICK subtype
  (Metrica also tags goal kicks as BALL LOST, which the default does not treat as a restart).
* The stoppage starts after the first stop marker that follows the last in-play action before the
  restart (BALL OUT, a shot that goes out or in, offside and end of half from the event's end frame;
  fouls and hand ball from its start frame). Live play after a trigger is never swallowed.
* With no marker (an untagged stoppage) it starts when that last in-play action ends.
* Frames of a window before its first restart are dead.

Ownership is the default's (team of the last PASS, CARRY, RECOVERY or SET PIECE), plus the restarting
team at every restart. Windows (X outside the first and last event of a period) are the default's.
"""

from __future__ import annotations

import numpy as np

from katena.external.metrica import _OWNER_TYPES, FPS, Event
from katena.types import D, X

PLAY_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "CHALLENGE"}


def is_restart(e: Event) -> bool:
    return e.type == "SET PIECE" or "GOAL KICK" in e.sub


def stop_frame(e: Event) -> int | None:
    """0-based last in-play frame of an event that stops play, else None (David's stop markers)."""
    t, s = e.type, e.sub
    if t == "BALL OUT" or (t == "SHOT" and ("GOAL" in s or "OUT" in s)) \
            or (t in ("BALL LOST", "PASS") and "OFFSIDE" in s) or (t == "BALL LOST" and "END HALF" in s):
        return e.end - 1
    if (t == "BALL LOST" and "HAND BALL" in s) or t == "FAULT RECEIVED" \
            or (t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s):
        return e.start - 1
    return None


def dead_mask(pe: list[Event], first: int, last: int, stats: dict) -> np.ndarray:
    """bool[last - first + 1]: dead frames of one period window [first, last] (0-based, inclusive)."""
    n = last - first + 1
    dead = np.zeros(n, dtype=bool)
    restarts = sorted({e.start - 1 for e in pe if is_restart(e)})

    def start_of(lo: int, hi: int) -> int:
        win = [e for e in pe if lo <= e.start - 1 < hi]
        plays = [e for e in win if e.type in PLAY_TYPES and stop_frame(e) is None]
        last_play = max((e.start - 1 for e in plays), default=lo)
        marks = [stop_frame(e) for e in win if stop_frame(e) is not None and e.start - 1 >= last_play]
        if marks:
            stats["stoppages_with_marker"] += 1
            return min(max(min(marks) + 1, lo), hi)
        stats["stoppages_without_marker"] += 1
        return min(max(max((e.end - 1 for e in plays), default=lo) + 1, lo), hi)

    if not restarts:
        dead[:] = True
        return dead
    dead[:max(0, min(restarts[0], last + 1) - first)] = True
    for prev, nxt in zip(restarts[:-1], restarts[1:]):
        s = start_of(prev, nxt)
        dead[max(s - first, 0):max(nxt - first, 0)] = True
    dead[max(start_of(restarts[-1], last + 1) - first, 0):] = True
    return dead


def derive_labels(events: list[Event], n: int) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    """Same signature and windows as ``katena.external.metrica.derive_labels``."""
    ev = sorted(events, key=lambda e: (e.start, e.order))
    y = np.full(n, X, dtype=np.int8)
    windows = []
    stats = {"n_events": len(ev), "stoppages_with_marker": 0, "stoppages_without_marker": 0, "dead_spans_s": []}
    for period in sorted({e.period for e in ev}):
        pe = [e for e in ev if e.period == period]
        first = pe[0].start - 1
        last = min(max(max(e.end, e.start) for e in pe) - 1, n - 1)
        windows.append((period, first, last))
        dead = dead_mask(pe, first, last, stats)
        owner = np.full(last - first + 1, X, dtype=np.int8)
        for e in pe:
            f = e.start - 1 - first
            if (e.type in _OWNER_TYPES or is_restart(e)) and 0 <= f < len(owner):
                owner[f] = e.team
        at = np.maximum.accumulate(np.where(owner != X, np.arange(len(owner)), -1))
        filled = np.where(at >= 0, owner[np.maximum(at, 0)], X).astype(np.int8)
        y[first:last + 1] = np.where(dead, D, filled)
        edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
        stats["dead_spans_s"] += ((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / FPS).tolist()
    return y, windows, stats
