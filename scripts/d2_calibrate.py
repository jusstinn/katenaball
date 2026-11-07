#!/usr/bin/env python
"""Measure how Metrica's real ball loss is tied to dead ball, for the D2 coupling worlds.

    KATENA_EXTERNAL=... uv run python scripts/d2_calibrate.py

Uses the Metrica development games only (Sample_Game_1 and Sample_Game_3). The locked Sample_Game_2 is
skipped. Works on the 5 Hz grid inside the in-play blocks, against the event-derived labels. Prints the
aggregates and writes src/katena/sim/coupling_metrica.json, a record checked by tests/test_d2_degrader.py
against the constants in katena.sim.worlds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.eval import protocol as P  # noqa: E402
from katena.grid import block_mask, downsample  # noqa: E402
from katena.sim.degrade import _runs_in_blocks  # noqa: E402
from katena.types import D, load_match  # noqa: E402

OUT = REPO / "src" / "katena" / "sim" / "coupling_metrica.json"


def main() -> None:
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


if __name__ == "__main__":
    main()
