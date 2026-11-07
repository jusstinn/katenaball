"""Label conventions: the sandwich merge that approximates "a simple touch does not change possession"."""

from __future__ import annotations

import dataclasses

import numpy as np

import possession_metrics as pm
from katena.types import A, H, Match


def sandwich(y: np.ndarray, runs: list, fps: float, T: float) -> np.ndarray:
    """Relabel opponent flips shorter than ``T`` seconds that sit between two segments of the
    other team (no D between) to the surrounding team. Repeated to a fixed point. ``runs`` are
    half-open graded spans (``pm.valid_runs``); nothing crosses a run edge."""
    y = y.copy()
    if T <= 0:
        return y
    lim = T * fps
    for run in runs:
        for _ in range(50):
            segs = pm.segments(y, [run])
            changed, i = False, 1
            while i < len(segs) - 1:
                a, b, c = segs[i - 1], segs[i], segs[i + 1]
                if (b[2] in (H, A) and a[2] == c[2] and a[2] in (H, A) and a[2] != b[2]
                        and (b[1] - b[0]) < lim):
                    y[b[0]:b[1]] = a[2]
                    changed = True
                    i += 2                       # neighbours changed meaning; re-segment next pass
                else:
                    i += 1
            if not changed:
                break
    return y


def relabel(m: Match, T: float) -> Match:
    """Copy of ``m`` (native grid) whose labels carry a sandwich merge of ``T`` seconds."""
    if m.y is None or T <= 0:
        return m
    return dataclasses.replace(m, y=sandwich(m.y, pm.valid_runs(m.y), m.fps, T))
