#!/usr/bin/env python
"""Calibrate the D2 worlds on external data only.

    KATENA_EXTERNAL=... uv run python scripts/d2_calibrate.py            # dead-ball ball loss (Metrica)
    KATENA_EXTERNAL=... uv run python scripts/d2_calibrate.py --jitter   # positional jitter (SkillCorner)

Coupling: measures how Metrica's real ball loss is tied to dead ball, on the Metrica development games
only (Sample_Game_1 and Sample_Game_3; the locked Sample_Game_2 is skipped), on the 5 Hz grid inside the
in-play blocks, against the event-derived labels. Writes src/katena/sim/coupling_metrica.json.

Jitter (D5 audit A20): measures the per-slot and ball second differences, the D5 domain-gap statistic,
on real automatic tracking (the 20 SkillCorner development matches; Metrica's two are reported too).
Then picks the player and ball jitter at which the degrader, run on the six clean IDSSE development
matches at ``target`` settings, gives the SkillCorner medians. Writes src/katena/sim/jitter_external.json.

tests/test_d2_degrader.py checks both records against the constants in katena.sim.worlds.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.eval import d2  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.grid import block_mask, downsample  # noqa: E402
from katena.sim import degrade as dg  # noqa: E402
from katena.sim.degrade import _runs_in_blocks  # noqa: E402
from katena.types import D, load_match  # noqa: E402

OUT = REPO / "src" / "katena" / "sim" / "coupling_metrica.json"
OUT_JITTER = REPO / "src" / "katena" / "sim" / "jitter_external.json"
SIGMA_PLAYER = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
SIGMA_BALL = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def coupling() -> None:
    dev, locked = P.split(P.all_refs())
    refs = [r for r in dev if r.dataset == "metrica"]
    assert refs and not any(r.dataset == "metrica" and r.id == P.LOCKED_METRICA for r in refs)
    n_dead = n_alive = miss_dead = miss_alive = 0
    lags, leads, per_game = [], [], {}
    for r in refs:
        m5 = downsample(load_match(r.path))
        blk = block_mask(m5)
        dead = blk & (m5.y == D)
        alive = blk & (m5.y < D)
        miss = ~np.isfinite(m5.ball[:, 0])
        n_dead += int(dead.sum())
        n_alive += int(alive.sum())
        miss_dead += int((miss & dead).sum())
        miss_alive += int((miss & alive).sum())
        s0, e0 = _runs_in_blocks(dead, m5)
        lost = 0
        for s, e in zip(s0, e0):
            seg = miss[s:e]
            if seg.any():
                lost += 1
                lags.append(int(np.argmax(seg)) / m5.fps)
                leads.append(int(np.argmax(seg[::-1])) / m5.fps)
        per_game[r.id] = {"p_miss_dead": float(miss[dead].mean()), "p_miss_alive": float(miss[alive].mean()),
                          "overall": float(miss[blk].mean()), "dead_share": float(dead.sum() / blk.sum()),
                          "dead_spans": int(len(s0)), "spans_with_loss": int(lost)}
    lags, leads = np.array(lags), np.array(leads)
    out = {
        "games": [r.id for r in refs],
        "grid_hz": 5.0,
        "p_miss_dead": miss_dead / n_dead,
        "p_miss_alive": miss_alive / n_alive,
        "lag_s": {"mean": float(lags.mean()), "median": float(np.median(lags)),
                  "q10_q25_q75_q90": [float(v) for v in np.percentile(lags, [10, 25, 75, 90])]},
        "lead_zero_share": float((leads == 0).mean()),
        "per_game": per_game,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


def _solve(sigmas, stats, want) -> float:
    """Smallest sigma on the piecewise-linear curve stat(sigma) that reaches ``want`` (0 if already above)."""
    stats = np.maximum.accumulate(np.asarray(stats, dtype=float))
    if want <= stats[0]:
        return 0.0
    if want >= stats[-1]:
        return float(sigmas[-1])
    return float(np.interp(want, stats, sigmas))


def jitter() -> None:
    dev, _ = P.split(P.all_refs())
    real = {}
    for ds in ("skillcorner", "metrica"):
        vals = [d2.jitter_stats(downsample(load_match(r.path))) for r in dev if r.dataset == ds]
        real[ds] = {"matches": len(vals), "player_median": float(np.median([v[0] for v in vals])),
                    "ball_median": float(np.median([v[1] for v in vals])),
                    "player_range": [float(min(v[0] for v in vals)), float(max(v[0] for v in vals))],
                    "ball_range": [float(min(v[1] for v in vals)), float(max(v[1] for v in vals))]}
    ids = [r for r in dev if r.dataset == "idsse"]
    curve = {"sigma_player": list(SIGMA_PLAYER), "sigma_ball": list(SIGMA_BALL), "player": [], "ball": []}
    m5s = {r.key: downsample(load_match(r.path)) for r in ids}
    for sp, sb in zip(SIGMA_PLAYER, SIGMA_BALL):
        cfg = dataclasses.replace(dg.TARGET, name="target_jitter_probe", sigma_player=sp, sigma_ball=sb)
        vals = [d2.jitter_stats(dg.corrupt(m5s[r.key], cfg, P.seed_for(r.id, "target"))) for r in ids]
        curve["player"].append(float(np.median([v[0] for v in vals])))
        curve["ball"].append(float(np.median([v[1] for v in vals])))
    target_now = [d2.jitter_stats(dg.corrupt(m5s[r.key], dg.TARGET, P.seed_for(r.id, "target"))) for r in ids]
    out = {
        "statistic": "D5 domain-gap jitter: 2-D second difference at 5 Hz; player = slot mean per sample "
                     "(capped at 2 m), 10 s window mean; ball = 10 s window median; median over windows",
        "real": real,
        "degraded_idsse_target_now": {"player": float(np.median([v[0] for v in target_now])),
                                      "ball": float(np.median([v[1] for v in target_now]))},
        "curve_idsse_target_settings": curve,
        "sigma_player": _solve(SIGMA_PLAYER, curve["player"], real["skillcorner"]["player_median"]),
        "sigma_ball": _solve(SIGMA_BALL, curve["ball"], real["skillcorner"]["ball_median"]),
        "calibrated_on": "skillcorner development matches (real automatic tracking); IDSSE development matches as the clean base",
    }
    OUT_JITTER.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", action="store_true")
    jitter() if ap.parse_args().jitter else coupling()
