"""B1's two pieces: the boundary target and the time-varying Viterbi. Synthetic data only."""

from __future__ import annotations

import numpy as np

from katena.decode import hmm
from katena.decode import tv
from katena.models import boundary as BD


def test_blocks5_matches_the_scorer_grid():
    assert BD.blocks5([(1, 0, 24), (2, 31, 62)], 5) == [(0, 5), (7, 13)]
    assert BD.blocks5([(1, 11, 13)], 5) == []                    # no work-grid sample inside the block


def test_owner_fills_unknown_with_the_last_label_inside_a_block():
    y = np.array([3, 0, 3, 3, 1, 3, 3, 0, 3], np.int8)
    np.testing.assert_array_equal(BD.owner(y, [(0, 6), (6, 9)]), [3, 0, 0, 0, 1, 1, 3, 0, 0])


def test_change_target_window_and_blocks():
    #            0  1  2  3  4  5  6  7  8  9 10 11 | 12 13 14 15
    y = np.array([0, 0, 0, 0, 3, 3, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0], np.int8)
    t, known = BD.change_target(y, [(0, 12), (12, 16)], tol=1)
    # changes at 6 (H -> A through the unknown gap) and at 10 (A -> D); none at 12 (a new block)
    np.testing.assert_array_equal(t, [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    assert known.all()
    t2, known2 = BD.change_target(np.array([3, 3, 0, 1], np.int8), [(0, 4)], tol=0)
    np.testing.assert_array_equal(t2, [0, 0, 0, 1])
    np.testing.assert_array_equal(known2, [False, False, True, True])


def _case(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = np.repeat(rng.integers(0, 3, 40), rng.integers(3, 20, 40)).astype(np.int8)[:n]
    pri = hmm.fit_priors([y])
    pa = np.clip(np.where(y == 2, 0.2, 0.8) + rng.normal(0, 0.2, len(y)), 0.01, 0.99).astype(np.float32)
    ph = np.clip(np.where(y == 0, 0.7, 0.3) + rng.normal(0, 0.25, len(y)), 0.01, 0.99).astype(np.float32)
    half = (np.arange(len(y)) >= len(y) // 2).astype(np.int8) + 1
    return y, pri, pa, ph, half


def test_lam_zero_is_the_static_viterbi():
    y, pri, pa, ph, half = _case()
    pc = np.random.default_rng(1).uniform(0.01, 0.99, len(y)).astype(np.float32)
    np.testing.assert_array_equal(tv.decode(pa, ph, pc, half, pri, 2.0, 0.0, 0.1),
                                  hmm.decode(pa, ph, half, pri, 2.0, "viterbi"))
    # a head that always says "base rate" changes nothing either
    np.testing.assert_array_equal(tv.decode(pa, ph, np.full(len(y), 0.1, np.float32), half, pri, 2.0, 1.0, 0.1),
                                  hmm.decode(pa, ph, half, pri, 2.0, "viterbi"))


def test_bonus_moves_the_number_of_switches_in_the_right_direction():
    y, pri, pa, ph, half = _case(seed=3)
    switches = lambda rows: int((np.diff(rows) != 0).sum())  # noqa: E731
    lo = tv.decode(pa, ph, np.full(len(y), 0.001, np.float32), half, pri, 2.0, 1.0, 0.1)
    mid = hmm.decode(pa, ph, half, pri, 2.0, "viterbi")
    hi = tv.decode(pa, ph, np.full(len(y), 0.9, np.float32), half, pri, 2.0, 1.0, 0.1)
    assert switches(lo) <= switches(mid) <= switches(hi) and switches(lo) < switches(hi)


def test_bonus_is_the_log_odds_ratio():
    b = tv.bonus(np.array([0.1, 0.5]), 0.1, 2.0)
    np.testing.assert_allclose(b, [0.0, 2.0 * np.log(9.0)], atol=1e-12)
