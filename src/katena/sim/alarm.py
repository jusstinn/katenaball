"""Label-free alarm: which D2 world does a match's ball loss resemble? Aggregates only.

``signature(m)`` computes a few rates on the in-play samples of a 5 Hz Match without reading labels:
how much more often the ball is missing when the players stand still (stoppages) than when they run,
how long the gaps are, and how often a long gap starts with the ball last seen at a line. In the
coupled worlds the ball goes missing through stoppages; in the independent world it goes missing in
crowds and in flight instead.

``reference()`` (``scripts/d2_worlds.py alarm``) computes the same signature on the external development
matches in every world. ``nearest(sig, ref)`` ranks the worlds by a standardised distance and says
whether the signature lies inside the range they cover.

This is an alarm for humans. It never selects a world, a training mix or a parameter, and nothing is
fitted to the match it runs on. On the competition match a human runs ``scripts/d2_target_alarm.py``,
which prints only these aggregates. (``katena.noise.gap`` on ``david/foundation`` is a different tool,
a real-versus-simulated classifier; this module does not port it.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from katena.grid import block_mask
from katena.types import Match

REF_PATH = Path(__file__).with_name("alarm_reference.json")
# the statistics the distance uses (ball loss); the others are printed for context
BALL_KEYS = ("ball_missing", "still_vs_moving_logodds", "long_gap_share", "gap_median_s", "edge_gap_share")
# floors for the standardised distance, so a world whose matches agree closely is not infinitely sharp
SD_FLOOR = {"ball_missing": 0.01, "still_vs_moving_logodds": 0.15, "long_gap_share": 0.03, "gap_median_s": 0.3,
            "edge_gap_share": 0.03}


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _activity(m: Match) -> np.ndarray:
    """Mean tracked-object speed per sample from positions (m/s, capped at 12), centred 2 s mean."""
    P = m.P.astype(np.float64)
    v = np.full(P.shape[:2], np.nan)
    if m.n > 2:
        v[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * (m.fps / 2.0)
    with np.errstate(all="ignore"):
        v = np.where(v > 12.0, np.nan, v)
        cnt = np.isfinite(v).sum(1)
        mean = np.where(cnt > 0, np.nansum(np.nan_to_num(v), 1) / np.maximum(cnt, 1), np.nan)
    w = max(1, int(round(1.0 * m.fps)))
    ok = np.isfinite(mean)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, mean, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(m.n)
    lo, hi = np.maximum(i - w, 0), np.minimum(i + w + 1, m.n)
    c = cn[hi] - cn[lo]
    with np.errstate(all="ignore"):
        return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def signature(m: Match) -> dict:
    """Label-free aggregates of a 5 Hz match over its in-play samples (``m.y`` is never read)."""
    blk = block_mask(m)
    miss = ~np.isfinite(m.ball[:, 0])
    act = _activity(m)
    fin = blk & np.isfinite(act)
    q20, q50 = np.quantile(act[fin], [0.2, 0.5]) if fin.any() else (np.nan, np.nan)
    still, moving = fin & (act <= q20), fin & (act >= q50)
    p_still = float(miss[still].mean()) if still.any() else float("nan")
    p_moving = float(miss[moving].mean()) if moving.any() else float("nan")
    L, W = m.pitch
    lens, edge, n_long = [], 0, 0
    for _, a, b in m.half_blocks:
        s, e = _runs(miss[a:b + 1])
        for gs, ge in zip(s + a, e + a):
            lens.append((ge - gs) / m.fps)
            if (ge - gs) / m.fps >= 5.0 and gs > a:
                x, y = m.ball[gs - 1]
                n_long += 1
                edge += int(min(x, L - x, y, W - y) < 2.0)
    lens = np.array(lens) if lens else np.zeros(0)
    pres = np.isfinite(m.P[..., 0])
    starts = 0
    for _, a, b in m.half_blocks:
        p = pres[a:b + 1]
        starts += int((p[1:] & ~p[:-1]).sum())
    minutes = blk.sum() / m.fps / 60.0
    return {
        "ball_missing": float(miss[blk].mean()),
        "miss_still": p_still,
        "miss_moving": p_moving,
        "still_vs_moving_logodds": _logit(p_still) - _logit(p_moving),
        "long_gap_share": float(lens[lens >= 10.0].sum() / lens.sum()) if lens.sum() > 0 else float("nan"),
        "gap_median_s": float(np.median(lens)) if len(lens) else float("nan"),
        "edge_gap_share": edge / n_long if n_long else float("nan"),
        "objects_mean": float(pres[blk].sum(1).mean()),
        "slot_starts_per_min": starts / minutes if minutes else float("nan"),
        "minutes_in_play": float(minutes),
    }


def summarise(sigs: list) -> dict:
    """{stat: [mean, sd, min, max]} over a world's matches."""
    out = {}
    for k in sigs[0]:
        v = np.array([s[k] for s in sigs], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = [float(v.mean()), float(v.std()), float(v.min()), float(v.max())] if len(v) else [math.nan] * 4
    return out


def load_reference(path: Path = REF_PATH) -> dict:
    return json.loads(path.read_text())


def nearest(sig: dict, ref: dict) -> list:
    """Worlds sorted by RMS standardised distance on BALL_KEYS: [(world, distance, inside_range)].
    ``inside_range``: every ball statistic lies within the world's min-max over matches, widened by two
    floors."""
    out = []
    for w, st in ref["worlds"].items():
        z, inside = [], True
        for k in BALL_KEYS:
            mean, sd, lo, hi = st[k]
            x = sig.get(k, math.nan)
            if not (math.isfinite(x) and math.isfinite(mean)):
                continue
            z.append((x - mean) / max(sd, SD_FLOOR[k]))
            inside &= (lo - 2 * SD_FLOOR[k]) <= x <= (hi + 2 * SD_FLOOR[k])
        out.append((w, float(np.sqrt(np.mean(np.square(z)))) if z else math.nan, bool(inside)))
    return sorted(out, key=lambda t: t[1])
