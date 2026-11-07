"""Working time grid. The pipeline runs at 5 Hz; scoring is always on the native frame grid.

``downsample`` keeps every ``step``-th native frame. ``to_native`` expands per-sample predictions
back to native frames by nearest-sample hold, so the official scorer sees full-length arrays.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from katena.types import X, Match

WORK_HZ = 5.0


def step_for(fps: float, hz: float = WORK_HZ) -> int:
    step = int(round(fps / hz))
    if step < 1 or abs(fps / step - hz) > 1e-6:
        raise ValueError(f"fps {fps} is not an integer multiple of {hz} Hz")
    return step


def downsample(m: Match, hz: float = WORK_HZ) -> Match:
    """New Match on the ``hz`` grid. Native frame of sample i is ``i * step`` (kept in meta)."""
    step = step_for(m.fps, hz)
    idx = np.arange(0, m.n, step)
    blocks = []
    for h, a, b in m.half_blocks:
        a2, b2 = -(-a // step), b // step
        if b2 >= a2:
            blocks.append((h, a2, b2))
    meta = dict(m.meta)
    meta.update(step=step, n_native=m.n, native_fps=m.fps)
    return dataclasses.replace(
        m, P=m.P[idx].copy(), speed=m.speed[idx].copy(), ball=m.ball[idx].copy(),
        half_blocks=blocks, attack_dir={h: m.attack_dir[h] for h, _, _ in blocks},
        y=None if m.y is None else m.y[idx].copy(), fps=m.fps / step, meta=meta,
        extras={k: v[idx].copy() for k, v in m.extras.items()},
    )


def to_native(codes: np.ndarray, step: int, n_native: int) -> np.ndarray:
    """Expand an int8 label array on the work grid to ``n_native`` frames (nearest sample)."""
    src = np.clip(np.rint(np.arange(n_native) / step).astype(np.int64), 0, len(codes) - 1)
    return codes[src]


def block_mask(m: Match) -> np.ndarray:
    """bool[n]: frames inside the in-play half blocks."""
    out = np.zeros(m.n, dtype=bool)
    for _, a, b in m.half_blocks:
        out[a:b + 1] = True
    return out


def mask_outside_blocks(codes: np.ndarray, m: Match) -> np.ndarray:
    out = np.full_like(codes, X)
    bm = block_mask(m)
    out[bm] = codes[bm]
    return out
