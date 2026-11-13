"""Ball-free timing columns (katena.featurize.kin), on synthetic fixtures only."""

from __future__ import annotations

import dataclasses

import numpy as np
from synth import L, W, make_match

from katena.featurize import kin as K
from katena.featurize.build import featurize
from katena.grid import downsample
from katena.types import Match

FPS = 5.0


def _turning_match(turn_s: float = 30.0, seed: int = 0, sigma: float = 0.0) -> Match:
    """One 60 s half at 5 Hz, six players a side. Home attacks +x. Every player moves towards +x at 3 m/s until
    ``turn_s`` and towards -x at 3 m/s after it, as both teams do when the ball changes owner."""
    rng = np.random.default_rng(seed)
    n = int(60 * FPS)
    t = np.arange(n) / FPS
    dx = np.where(t <= turn_s, 3.0 * t, 3.0 * turn_s - 3.0 * (t - turn_s))
    team = np.array([0] * 6 + [1] * 6, dtype=np.int8)
    x0 = np.concatenate([np.linspace(5.0, 30.0, 6), np.linspace(40.0, 75.0, 6)])
    y0 = np.concatenate([np.linspace(8.0, 60.0, 6), np.linspace(10.0, 58.0, 6)])
    P = np.stack([x0[None, :] + dx[:, None], np.repeat(y0[None, :], n, 0)], -1)
    if sigma > 0:
        P = P + rng.normal(0, sigma, P.shape)
    return Match(P=P.astype(np.float32), team=team, speed=np.zeros((n, 12), np.float32),
                 ball=np.full((n, 2), np.nan, np.float32), half_blocks=[(1, 0, n - 1)], attack_dir={1: 1},
                 y=np.zeros(n, np.int8), pitch=(L, W), fps=FPS, name="turn", source="synthetic")


def _cols(m: Match, focal: int = 0, dirs: dict | None = None) -> dict:
    X, names, _ = K.kin_features(m, dirs=dirs if dirs is not None else {1: 1})
    return {n: X[focal][:, i] for i, n in enumerate(names)}


def test_names_are_unique_and_prefixed():
    X, names, idx = K.kin_features(_turning_match(), dirs={1: 1})
    assert len(names) == len(set(names)) == X[0].shape[1] == X[1].shape[1] == 35
    assert all(n.startswith("kin_") for n in names)
    assert X[0].dtype == np.float32 and len(idx) == len(X[0])


def test_one_sided_velocity_reverses_at_the_scripted_turn():
    m = _turning_match(turn_s=30.0)
    c = _cols(m)
    k = int(30 * FPS)
    # well before the turn both windows see +3 m/s, well after it both see -3 m/s
    for name in ("kin_vx_f_b1s", "kin_vx_f_a1s", "kin_vx_o_b1s", "kin_vx_o_a1s", "kin_vx_f_b2s", "kin_vx_f_a2s"):
        np.testing.assert_allclose(c[name][k - 50:k - 15], 3.0, atol=1e-3)
        np.testing.assert_allclose(c[name][k + 15:k + 50], -3.0, atol=1e-3)
    # at the turn the window behind still sees forward movement and the window ahead already sees the retreat
    assert c["kin_vx_f_b1s"][k] > 2.0 and c["kin_vx_f_a1s"][k] < -2.0
    assert c["kin_vx_f_rev1s"][k] < -4.0 and c["kin_vx_o_rev1s"][k] < -4.0 and c["kin_vx_all_rev1s"][k] < -4.0
    # the reversal is a peak at the turn and nothing away from it
    rev = c["kin_vx_all_rev1s"]
    assert abs(int(np.nanargmin(rev)) - k) <= 1
    np.testing.assert_allclose(rev[k - 50:k - 15], 0.0, atol=1e-3)
    np.testing.assert_allclose(rev[k + 15:k + 50], 0.0, atol=1e-3)
    # shares of slots moving forward
    np.testing.assert_allclose(c["kin_fwd_f_b1s"][k - 30:k - 10], 1.0)
    np.testing.assert_allclose(c["kin_fwd_f_a1s"][k + 10:k + 30], -1.0)


def test_the_other_focal_sees_the_opposite_reversal():
    m = _turning_match()
    c0, c1 = _cols(m, 0), _cols(m, 1)
    k = int(30 * FPS)
    np.testing.assert_allclose(c1["kin_vx_f_rev1s"][k], -c0["kin_vx_o_rev1s"][k], atol=1e-4)
    np.testing.assert_allclose(c1["kin_vx_all_rev1s"], -c0["kin_vx_all_rev1s"], atol=1e-4, equal_nan=True)


def test_a_fragment_that_starts_or_ends_is_not_movement():
    """Static players; one slot is unseen for a stretch and another jumps 40 m (a fragment taken over by another
    player). Neither may read as movement."""
    m = _turning_match()
    P = np.repeat(m.P[:1], m.n, 0).copy()                     # everyone stands still
    P[100:140, 2] = np.nan                                    # a home slot is lost and found again
    P[200:, 3, 0] += 40.0                                     # a home slot switches to a player 40 m away
    P[250:253, 4] = np.nan                                    # a home slot is lost for 0.6 s ...
    P[253:, 4, 0] += 30.0                                     # ... and comes back on a player 30 m away
    c = _cols(dataclasses.replace(m, P=P))
    for name in ("kin_vx_f_b1s", "kin_vx_f_a1s", "kin_vx_f_b2s", "kin_vx_f_a2s", "kin_vx_f_rev1s", "kin_vx_f_rev2s"):
        v = c[name][20:-20]
        assert np.isfinite(v).all() and np.abs(v).max() < 1e-3, name
    assert np.abs(c["kin_fwd_f_b1s"][20:-20]).max() == 0.0


def test_too_few_slots_give_no_value():
    m = _turning_match()
    P = m.P.copy()
    P[:, 2:6] = np.nan                                        # two home slots left
    c = _cols(dataclasses.replace(m, P=P))
    assert np.isnan(c["kin_vx_f_b1s"]).all() and np.isnan(c["kin_rear_f_b1s"]).all()
    assert np.isfinite(c["kin_vx_o_b1s"][20:-20]).all()


def test_side_symmetry():
    """Swapping the team labels and rotating the pitch gives the same columns for the other focal."""
    m = _turning_match(seed=3, sigma=0.2)
    sw = dataclasses.replace(m, P=np.stack([L - m.P[..., 0], W - m.P[..., 1]], -1).astype(np.float32),
                             team=(1 - m.team).astype(np.int8))
    X, _, _ = K.kin_features(m, dirs={1: 1})
    Xs, _, _ = K.kin_features(sw, dirs={1: 1})                # the new home team is the old away team: it attacks +x
    np.testing.assert_allclose(X[0], Xs[1], atol=2e-3, equal_nan=True)
    np.testing.assert_allclose(X[1], Xs[0], atol=2e-3, equal_nan=True)


def test_rows_line_up_with_featurize_and_halves_do_not_leak():
    m5 = downsample(make_match(seed=1))
    X, names, idx = K.kin_features(m5)
    f = featurize(m5, 0)
    np.testing.assert_array_equal(idx, f.idx)
    assert len(X[0]) == len(f.X)
    # each half by itself: the second half's columns do not change when the first half's players are moved
    (_, a, b), (_, a2, b2) = m5.half_blocks
    P = m5.P.copy()
    P[a:b + 1, :, 0] += 3.0
    X2, _, _ = K.kin_features(dataclasses.replace(m5, P=P), dirs={1: 1, 2: -1})
    X1, _, _ = K.kin_features(m5, dirs={1: 1, 2: -1})
    n1 = b - a + 1
    np.testing.assert_array_equal(X1[0][n1:], X2[0][n1:])


def test_no_column_reads_the_ball():
    m5 = downsample(make_match(seed=2))
    X, _, _ = K.kin_features(m5)
    gone = dataclasses.replace(m5, ball=np.full_like(m5.ball, np.nan))
    moved = dataclasses.replace(m5, ball=(m5.ball + 7.0).astype(np.float32))
    for other in (gone, moved):
        Xo, _, _ = K.kin_features(other, dirs={1: 1, 2: -1})
        Xr, _, _ = K.kin_features(m5, dirs={1: 1, 2: -1})
        np.testing.assert_array_equal(Xr[0], Xo[0])
        np.testing.assert_array_equal(Xr[1], Xo[1])
    assert len(X[0]) > 0


def test_lines_and_spread_rates():
    """The home team stretches along x at a constant rate: its span rate is that rate, its rear line stands."""
    m = _turning_match()
    n = m.n
    t = np.arange(n) / FPS
    P = np.repeat(m.P[:1], n, 0).astype(np.float64)
    x0 = P[0, :6, 0].copy()
    P[:, :6, 0] = x0[None, :] + (x0[None, :] - x0.min()) / (x0.max() - x0.min()) * 0.5 * t[:, None]
    c = _cols(dataclasses.replace(m, P=P.astype(np.float32)))
    q = (np.percentile(x0, 80) - np.percentile(x0, 20)) / (x0.max() - x0.min()) * 0.5
    np.testing.assert_allclose(c["kin_span_f_b2s"][30:-30], q, atol=5e-3)
    np.testing.assert_allclose(c["kin_span_f_a2s"][30:-30], q, atol=5e-3)
    np.testing.assert_allclose(c["kin_span_o_b2s"][30:-30], 0.0, atol=1e-3)
    np.testing.assert_allclose(c["kin_width_f_b2s"][30:-30], 0.0, atol=1e-3)


def test_closest_opposing_pair():
    m = _turning_match()
    P = np.repeat(m.P[:1], m.n, 0).copy()
    P[:, 0] = (20.0, 20.0)
    P[:, 6] = (23.0, 24.0)                                    # 5 m from home slot 0; everyone else is further
    P[:, 1:6, 1] = 60.0
    P[:, 7:, 1] = 40.0
    P[:, 7:, 0] = 90.0
    c = _cols(dataclasses.replace(m, P=P))
    np.testing.assert_allclose(c["kin_pair_min"], 5.0, atol=1e-4)
    np.testing.assert_allclose(c["kin_pair_min_b1s"], 5.0, atol=1e-4)
    np.testing.assert_allclose(c["kin_pair_min_a1s"], 5.0, atol=1e-4)


def test_fast_quantile_equals_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, (400, 14))
    x[rng.random(x.shape) < 0.3] = np.nan
    x[5] = np.nan
    x[6, 1:] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanpercentile(x, [20, 80], axis=1)
    np.testing.assert_allclose(K._nanq(x, 20.0), ref[0], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(K._nanq(x, 80.0), ref[1], atol=1e-9, equal_nan=True)
