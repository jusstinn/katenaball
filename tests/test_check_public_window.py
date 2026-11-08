"""scripts/check_public_window.py catches the bugs it is for, using only public files: the organisers'
published baseline (computed from the ATD, which every participant has) and the public ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "evaluation")]

import check_public_window as cpw  # noqa: E402
import possession_metrics as pm  # noqa: E402


def _codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def test_flags_a_home_away_swap_and_a_time_shift():
    gt, base = _codes(cpw.GT), _codes(cpw.BASELINE)
    smooth125 = _codes(REPO / "evaluation" / "results" / "baseline_smooth125.csv")
    ok = cpw.checks(smooth125, gt, base)
    assert ok["home_away_orientation"] and ok["frame_alignment"]
    swapped = cpw.checks(cpw.swap_teams(smooth125), gt, base)
    assert not swapped["home_away_orientation"]
    late = cpw.checks(cpw.shift(smooth125, 10 * cpw.FPS), gt, base)
    assert not late["frame_alignment"]


def test_shift_and_swap_helpers():
    import numpy as np
    c = np.array([0, 0, 1, 2, 3], dtype=np.int8)
    assert cpw.swap_teams(c).tolist() == [1, 1, 0, 2, 3]
    assert cpw.shift(c, 1).tolist() == [0, 0, 0, 1, 2]
    assert cpw.shift(c, -1).tolist() == [0, 1, 2, 3, 3]
