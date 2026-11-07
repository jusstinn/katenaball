import dataclasses
import math

import numpy as np
import pytest

from katena.match import Match
from katena.noise import corrupt, target_stats
from katena.synth import synthetic_match


@pytest.fixture(scope="module")
def clean():
    return synthetic_match(seed=4, half_minutes=1.0)


@pytest.fixture(scope="module")
def noisy(clean):
    return corrupt(clean, {}, seed=0)


def _same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def test_stats_are_plain_floats_and_label_free(noisy):
    s = target_stats(noisy)
    assert len(s) > 40
    assert all(type(v) is float for v in s.values())
    s2 = target_stats(dataclasses.replace(noisy, y=None))
    assert s.keys() == s2.keys() and all(_same(s[k], s2[k]) for k in s)


def test_clean_match_baseline(clean):
    s = target_stats(clean)
    assert s["ball_present_rate"] == 1.0 and s["ball_gaps_per_min"] == 0.0
    assert s["slots_present_mean"] == 22.0 and s["slots_used"] == 22.0
    assert s["slot_run_q50_s"] == pytest.approx(60.0)  # one run per slot per 1-minute half
    assert s["n_slots"] == 22.0 and s["home_slot_frac"] == 0.5


def test_stats_see_the_noise(clean, noisy):
    s, s0 = target_stats(noisy), target_stats(clean)
    assert abs(s["ball_present_rate"] - 0.593) < 0.02
    assert abs(s["slots_present_mean"] - 20.5) < 0.3
    assert s["slot_run_q50_s"] < s0["slot_run_q50_s"]
    assert s["slots_used"] > s0["slots_used"]
    assert s["ball_gaps_per_min"] > 0


def _toy(ball_missing, n=1500):
    """All 22 players present and still; ball NaN on the given frames."""
    P = np.zeros((n, 22, 2), dtype=np.float32)
    P[:, :, 0] = np.arange(22, dtype=np.float32)[None, :] * 3 - 30
    ball = np.zeros((n, 2), dtype=np.float32) + 40
    ball[ball_missing] = np.nan
    return Match(P=P, team=np.array([0] * 11 + [1] * 11, np.int8), speed=np.zeros((n, 22), np.float32),
                 ball=ball, half_blocks=[(1, 0, n - 1)], attack_dir={1: 1})


def test_gap_statistics_on_a_known_pattern():
    missing = np.zeros(1500, dtype=bool)
    missing[[100, 300]] = True          # two isolated 1-frame drops
    missing[500:510] = True             # one 10-frame burst
    s = target_stats(_toy(missing))
    assert s["ball_present_rate"] == pytest.approx(1 - 12 / 1500)
    assert s["ball_gaps_per_min"] == pytest.approx(3 / 1.0)
    assert s["ball_iso_share"] == pytest.approx(2 / 12)
    assert s["ball_bursts_per_min"] == pytest.approx(1.0)
    assert s["ball_burst_q50_s"] == pytest.approx(10 / 25)
    assert s["ball_gap_q50_s"] == pytest.approx(1 / 25)


def test_duplicate_pairs_are_counted():
    m = _toy(np.zeros(1500, dtype=bool))
    m.P[:, 1] = m.P[:, 0] + 0.2   # slot 1 shadows slot 0, same team
    m.P[:, 12] = m.P[:, 11] + 0.2  # away pair too
    s = target_stats(m)
    assert s["dup_pairs_per_frame"] == pytest.approx(2.0)
    assert s["dup_frame_frac"] == pytest.approx(1.0)
