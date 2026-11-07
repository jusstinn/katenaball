#!/usr/bin/env python
"""HUMAN ONLY. Which D2 degrader world does the competition match's ball loss resemble?

    uv run python scripts/d2_target_alarm.py

Reads the tactical ATD in data/ (via katena.target.atd.load_target) and prints a dozen whole-match
aggregates: rates, a log-odds difference, a median gap length and a count per minute. It also prints
the D2 worlds ranked by distance from those aggregates (katena.sim.alarm). No frame, row, track or
label is printed, and exceptions print their type only. Do not paste anything except this output.

It is an alarm. It does not choose a world, a training mix or a parameter, and nothing is fitted to
the match. If the match falls outside every world, the simulated worlds do not cover it: tell the
team, and do not tune to it. Claude must never run this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))


def main() -> int:
    try:
        from katena.grid import downsample
        from katena.sim import alarm
        from katena.target.atd import load_target

        m5 = downsample(load_target())
        sig = alarm.signature(m5)
        ranked = alarm.nearest(sig, alarm.load_reference())
    except Exception as error:  # noqa: BLE001  the type alone, as messages can quote the file
        print(f"FAILED: {type(error).__name__}")
        return 1
    print("label-free aggregates over the graded halves (5 Hz):")
    for k, v in sig.items():
        print(f"  {k:26s} {v:.3f}")
    print("D2 worlds by distance (lower is closer; 'inside' = every ball statistic within that world's range):")
    for w, dist, inside in ranked:
        print(f"  {w:20s} {dist:6.2f}  {'inside' if inside else 'outside'}")
    if not any(inside for _, _, inside in ranked):
        print("ALARM: outside every simulated world")
    return 0


if __name__ == "__main__":
    sys.exit(main())
