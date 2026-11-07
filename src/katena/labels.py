"""Label conventions: the sandwich merge that approximates "a simple touch does not change possession"."""

from __future__ import annotations

import dataclasses

import numpy as np

import possession_metrics as pm
from katena.types import A, D, H, Match

ORDERS = ("scan", "shortest")


def sandwich(y: np.ndarray, runs: list, fps: float, T: float, tail: bool = False,
             order: str = "scan") -> np.ndarray:
    """Relabel opponent flips shorter than ``T`` seconds that sit between two segments of the
    other team (no D between) to the surrounding team. Repeated to a fixed point. ``runs`` are
    half-open graded spans (``pm.valid_runs``); nothing crosses a run edge.

    Options (the defaults are the convention every cache was built with; docs/label-convention.md):

    * ``tail``: also relabel a short opponent flip that runs from the other team's spell straight
      into dead ball or the end of the run (touched, then out). CHALLENGE.md does not count a touch
      without control as a change of possession either way. A spell that follows dead ball or
      starts a run (the restart possessor) is never relabelled.
    * ``order``: ``"scan"`` (left to right, the original) or ``"shortest"`` (shortest flip first,
      ties by time, as in David's ``harmonise`` on origin/david/foundation). They differ only on
      chains of flips. ``tail=True, order="shortest"`` reproduces ``harmonise(min_control_s=T)``
      (exactly whenever ``T * fps`` is not a half-integer that Python's ``round`` takes down, which
      holds for 3.5 s at 25, 10 and 5 fps).
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    y = y.copy()
    if T <= 0:
        return y
    lim = T * fps
    for run in runs:
        if order == "shortest":
            _shortest_first(y, run, lim, tail)
            continue
        for _ in range(50):
            segs = pm.segments(y, [run])
            changed, i = False, 1
            stop = len(segs) if tail else len(segs) - 1
            while i < stop:
                a, b = segs[i - 1], segs[i]
                c = segs[i + 1] if i + 1 < len(segs) else None
                ends_ok = (c is not None and a[2] == c[2]) or (tail and (c is None or c[2] == D))
                if (b[2] in (H, A) and ends_ok and a[2] in (H, A) and a[2] != b[2]
                        and (b[1] - b[0]) < lim):
                    y[b[0]:b[1]] = a[2]
                    changed = True
                    i += 2                       # neighbours changed meaning; re-segment next pass
                else:
                    i += 1
            if not changed:
                break
    return y


def _flip_target(segs: list, i: int, lim: float, tail: bool):
    """Team that segment ``i`` merges into, or None if it is not a removable flip."""
    a, b, k = segs[i]
    if k not in (H, A) or (b - a) >= lim or i == 0 or segs[i - 1][2] not in (H, A):
        return None
    nxt = segs[i + 1][2] if i + 1 < len(segs) else None
    if nxt in (H, A) or tail:            # nxt is the previous team (two teams), D, or the run end
        return segs[i - 1][2]
    return None


def _shortest_first(y: np.ndarray, run: tuple, lim: float, tail: bool) -> None:
    """In place on ``y[run]``: remove removable flips one at a time, shortest first."""
    segs = [list(s) for s in pm.segments(y, [run])]
    while True:
        best = None
        for i in range(1, len(segs)):
            target = _flip_target(segs, i, lim, tail)
            if target is not None:
                key = (segs[i][1] - segs[i][0], segs[i][0])
                if best is None or key < best[0]:
                    best = (key, i, target)
        if best is None:
            break
        _, i, target = best
        segs[i][2] = target
        if i + 1 < len(segs) and segs[i + 1][2] == target:
            segs[i][1] = segs[i + 1][1]
            del segs[i + 1]
        if segs[i - 1][2] == target:
            segs[i - 1][1] = segs[i][1]
            del segs[i]
    for a, b, k in segs:
        y[a:b] = k


def relabel(m: Match, T: float, tail: bool = False, order: str = "scan") -> Match:
    """Copy of ``m`` (native grid) whose labels carry a sandwich merge of ``T`` seconds."""
    if m.y is None or T <= 0:
        return m
    return dataclasses.replace(m, y=sandwich(m.y, pm.valid_runs(m.y), m.fps, T, tail=tail, order=order))
