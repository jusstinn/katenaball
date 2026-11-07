import numpy as np

from katena.feats.temporal import (gap_diff, last_true_index, next_true_index, roll_max,
                                   roll_mean, roll_min, run_bounds, segments_from_lengths,
                                   shift_clip)


def _brute(x, h, lo, hi, fn):
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        w = x[max(t - h, lo[t]):min(t + h, hi[t]) + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[t] = fn(w)
    return out


def test_rolling_matches_brute_force_with_nans_and_segments():
    rng = np.random.default_rng(0)
    lengths = [37, 1, 50, 12]
    lo, hi = segments_from_lengths(lengths)
    x = rng.normal(size=sum(lengths))
    x[rng.random(len(x)) < 0.3] = np.nan
    x[40:52] = np.nan  # an all-NaN window inside a segment
    for h in (1, 3, 7, 30):
        assert np.allclose(roll_mean(x, h, lo, hi), _brute(x, h, lo, hi, np.mean), equal_nan=True,
                           atol=1e-6)
        assert np.allclose(roll_max(x, h, lo, hi), _brute(x, h, lo, hi, np.max), equal_nan=True)
        assert np.allclose(roll_min(x, h, lo, hi), _brute(x, h, lo, hi, np.min), equal_nan=True)


def test_last_next_index_and_runs_stop_at_segments():
    lo, hi = segments_from_lengths([5, 5])
    present = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=bool)
    assert last_true_index(present, lo).tolist() == [0, 0, 0, 3, 4, -1, 6, 7, 7, 7]
    assert next_true_index(present, hi).tolist() == [0, 3, 3, 3, 4, 6, 6, 7, -1, -1]
    rs, re = run_bounds(present, lo, hi)
    assert rs[[0, 3, 4, 6, 7]].tolist() == [0, 3, 3, 6, 6]
    assert re[[0, 3, 4, 6, 7]].tolist() == [0, 4, 4, 7, 7]
    # a run touching the boundary between segments is cut there
    full = np.ones(10, dtype=bool)
    rs, re = run_bounds(full, lo, hi)
    assert rs.tolist() == [0] * 5 + [5] * 5 and re.tolist() == [4] * 5 + [9] * 5


def test_gap_diff_never_crosses_a_gap():
    lo, hi = segments_from_lengths([12])
    t = np.arange(12, dtype=np.float32)
    X = np.stack([2.0 * t, -t], axis=1).astype(np.float32)  # constant velocity (2, -1) per frame
    present = np.ones(12, dtype=bool)
    present[[5, 9]] = False
    X[~present] = 1e6  # garbage in the gap must never leak into a difference
    rs, re = run_bounds(present, lo, hi)
    v = gap_diff(X, present, rs, re, 2, fps=25.0)
    # runs [0, 4], [6, 8], [10, 11]; the last has length 2 and uses one-sided differences
    assert np.isnan(v[~present]).all()
    assert np.allclose(v[present], [50.0, -25.0])
    # isolated single frame -> NaN
    p2 = np.zeros(12, dtype=bool)
    p2[3] = True
    rs, re = run_bounds(p2, lo, hi)
    assert np.isnan(gap_diff(X, p2, rs, re, 2, 25.0)).all()


def test_gap_diff_per_slot():
    lo, hi = segments_from_lengths([6, 6])
    t = np.arange(12, dtype=np.float32)
    X = np.stack([np.stack([t, 0 * t], 1), np.stack([0 * t, 3 * t], 1)], axis=1)  # [12, 2, 2]
    present = np.ones((12, 2), dtype=bool)
    present[2, 0] = False
    rs, re = run_bounds(present, lo, hi)
    v = gap_diff(X.astype(np.float32), present, rs, re, 2, 1.0)
    assert v.shape == (12, 2, 2)
    assert np.isnan(v[2, 0]).all()
    assert np.allclose(v[:, 0, 0][present[:, 0]], 1.0)
    assert np.allclose(v[:, 1, 1], 3.0)


def test_shift_clip():
    lo, hi = segments_from_lengths([3, 3])
    x = np.arange(6.0)
    assert shift_clip(x, 2, lo, hi).tolist() == [2, 2, 2, 5, 5, 5]
    assert shift_clip(x, -1, lo, hi).tolist() == [0, 0, 1, 3, 3, 4]
