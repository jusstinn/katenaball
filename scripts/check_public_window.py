#!/usr/bin/env python3
"""Pipeline check of a submission against the public 5-minute window. Booleans only.

    uv run python scripts/check_public_window.py submissions/<team>.csv

The public window is for checking the pipeline, never for training, tuning or choosing between versions
(docs/upstream/SUBMISSION.md). So this prints no score, only pass/fail flags for bugs that would wreck the
score silently:

* the file passes the official checks;
* home and away are the right way round: swapping H and A lowers team accuracy on the window;
* the frames line up: shifting the labels by 5 or 10 seconds either way lowers spellF1;
* the file beats the organisers' published smooth50 baseline on the window, as a sanity floor.

A FAIL means look for a bug (slot order, home team, frame offset), not tune. Log every run in
docs/decision-log.md, as the plan asks for every look at the public window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

import possession_metrics as pm  # noqa: E402

GT = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
BASELINE = REPO / "evaluation" / "results" / "baseline_smooth50.csv"
SHIFTS_S = (-10, -5, 5, 10)
FPS = 25


def swap_teams(codes: np.ndarray) -> np.ndarray:
    out = codes.copy()
    out[codes == 0], out[codes == 1] = 1, 0
    return out


def shift(codes: np.ndarray, k: int) -> np.ndarray:
    """Labels moved k frames later (k < 0: earlier), edges padded with the nearest label."""
    idx = np.clip(np.arange(len(codes)) - k, 0, len(codes) - 1)
    return codes[idx]


def checks(pred: np.ndarray, gt: np.ndarray, base: np.ndarray) -> dict:
    def ev(p):
        return pm.evaluate(gt, p, bootstrap_n=0)
    r = ev(pred)
    return {
        "home_away_orientation": bool(r["team_acc_gt_alive"] > ev(swap_teams(pred))["team_acc_gt_alive"]),
        "frame_alignment": bool(all(r["spell_f1_mean"] > ev(shift(pred, s * FPS))["spell_f1_mean"] for s in SHIFTS_S)),
        "beats_published_baseline": bool(r["spell_f1_mean"] > ev(base)["spell_f1_mean"]),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.splitlines()[2].strip())
        return 2
    gt = pm.label_codes(pm.read_label_csv(GT))
    frame = pm.read_label_csv(argv[0])
    reason = pm.validate_submission(frame, len(gt))
    print(f"valid_submission: {'PASS' if not reason else 'FAIL'}")
    if reason:
        return 1
    res = checks(pm.label_codes(frame), gt, pm.label_codes(pm.read_label_csv(BASELINE)))
    for k, v in res.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("Log this look in docs/decision-log.md.")
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
