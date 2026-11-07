from dataclasses import replace

import numpy as np
import pytest

from katena.feats import DEFAULT_CFG, Feats, ball_free, featurize, spec, swap_teams
from katena.feats.stress import (drop_ball_bursts, duplicate_slot, fragment_slots,
                                 swap_match_teams)
from katena.match import Match
from katena.synth import synthetic_match


@pytest.fixture(scope="module")
def clean():
    return synthetic_match(seed=3, half_minutes=1.0)


@pytest.fixture(scope="module")
def noisy(clean):
    """46 anonymous slots with fragments, dropouts, duplicates and ghosts; bursty ball loss."""
    return drop_ball_bursts(fragment_slots(clean, seed=1, dup_per_min=6, ghost_per_min=3), seed=2)


@pytest.fixture(scope="module")
def f_noisy(noisy):
    return featurize(noisy)


def _never_nan(names):
    return [n for n in names if not spec.nan_able(n)]


def test_shapes_dtypes_and_rows(clean):
    f = featurize(clean)
    assert isinstance(f, Feats)
    n_graded = int(clean.graded_mask().sum())
    assert f.X.dtype == np.float32 and f.X.shape == (n_graded, len(f.names))
    assert f.t_frame.dtype == np.int64 and f.mask.dtype == bool and f.mask.all()
    assert len(set(f.names)) == len(f.names)
    assert f.names[:len(spec.base_names())] == spec.base_names()
    assert np.array_equal(f.t_frame, np.flatnonzero(clean.graded_mask()))
    assert f.F == 191


def test_stride_rows_equal_full_rows(clean):
    full = featurize(clean)
    sub = featurize(clean, {"stride": 5, "stride_offset": 2})
    for _, first, last in clean.half_blocks:
        got = sub.t_frame[(sub.t_frame >= first) & (sub.t_frame <= last)]
        assert np.array_equal(got, np.arange(first + 2, last + 1, 5))
    rows = np.searchsorted(full.t_frame, sub.t_frame)
    assert np.array_equal(sub.X, full.X[rows], equal_nan=True)


def test_rows_all_masks_graded_frames(clean):
    f = featurize(clean, {"rows": "all"})
    assert np.array_equal(f.t_frame, np.arange(clean.n))
    assert np.array_equal(f.mask, clean.graded_mask())
    assert not np.isinf(f.X).any()
    # graded rows see the same per-frame base features as in the default (halves) layout
    halves = featurize(clean, {"context": False})
    base = spec.base_names()
    assert np.allclose(f.select(base).X[f.mask], halves.X, atol=1e-5, equal_nan=True)


def test_unknown_cfg_key_raises(clean):
    with pytest.raises(ValueError):
        featurize(clean, {"strid": 2})


def test_deterministic(noisy, f_noisy):
    again = featurize(noisy)
    assert again.names == f_noisy.names
    assert np.array_equal(again.X, f_noisy.X, equal_nan=True)


def test_nan_policy_on_noisy_input(f_noisy):
    X = f_noisy.X
    assert not np.isinf(X).any()
    for n in _never_nan(f_noisy.names):
        assert np.isfinite(f_noisy.col(n)).all(), n
    # the ball is present on ~60% of frames, so observed-ball columns are NaN about 40% of the time
    pres = f_noisy.col("ball_present") == 1
    assert 0.45 < pres.mean() < 0.75
    assert np.isnan(f_noisy.col("ball_x")[~pres]).all()
    assert np.isfinite(f_noisy.col("ball_x")[pres]).all()
    assert np.isfinite(f_noisy.col("home_nearest_dist")[pres]).all()
    # interpolation fills some of the gaps, and only short ones
    known = f_noisy.col("ball_known_i") == 1
    assert known.mean() > pres.mean()
    interp = f_noisy.col("ball_interp") == 1
    cap = DEFAULT_CFG["max_interp_s"] * 25
    assert (f_noisy.col("ball_gap_len")[interp] <= cap).all()
    assert np.isfinite(f_noisy.col("ball_x_i")[known]).all()
    # team shape stays defined whenever a team has at least two players
    for s in ("home", "away"):
        ok = f_noisy.col(f"{s}_n") >= 2
        assert ok.mean() > 0.99
        for c in ("cx", "sx", "line_x", "mark_dist"):
            assert np.isfinite(f_noisy.col(f"{s}_{c}")[ok]).mean() > 0.99, c


def test_clean_match_columns_mostly_defined(clean):
    f = featurize(clean)
    nanfrac = np.isnan(f.X).mean(axis=0)
    always = ["ball_x", "ball_x_i", "home_nearest_dist", "away_nearest_dist", "nearest_margin",
              "home_pressers", "home_cx", "away_line_x", "adv_diff", "all_mspd",
              "nearest_margin_i__mean2s"]
    for n in always:
        assert nanfrac[f.names.index(n)] == 0, n


def test_all_nan_ball(clean, noisy):
    m = replace(noisy, ball=np.full_like(noisy.ball, np.nan))
    f = featurize(m)
    ref = featurize(noisy)
    cap = DEFAULT_CFG["gap_cap_s"] * 25
    assert (f.col("ball_present") == 0).all() and (f.col("ball_since") == cap).all()
    assert (f.col("ball_until") == cap).all() and (f.col("ball_gap_len") == cap).all()
    for n in f.names:
        g = spec.group_of(n)
        if g in ("ball", "ball_rel", "press") and spec.nan_able(n):
            assert np.isnan(f.col(n)).all(), n
    # ball-free columns do not read the ball at all
    bf = ball_free(f.names)
    assert len(bf) > 60
    assert np.array_equal(f.select(bf).X, ref.select(bf).X, equal_nan=True)
    for n in _never_nan(f.names):
        assert np.isfinite(f.col(n)).all(), n


def test_ball_free_groups_exclude_ball_columns(f_noisy):
    bf = set(ball_free(f_noisy.names))
    assert "home_cx" in bf and "adv_diff__mean2s" in bf and "all_mspd" in bf
    assert not any(n.startswith("ball_") or "nearest" in n or "pressers" in n for n in bf)


def test_duplicate_slot_is_deduplicated(clean):
    src = 4  # a home player
    dup = duplicate_slot(clean, src, 0, clean.n)
    a, b = featurize(clean), featurize(dup)
    for core in ("n", "cx", "cy", "sx", "line_x", "top_x", "width", "mark_dist", "nearest_dist",
                 "second_dist", "n2", "n5", "pressers"):
        for s in ("home", "away"):
            name = f"{s}_{core}"
            assert np.allclose(a.col(name), b.col(name), atol=1e-4, equal_nan=True), name


def test_random_missing_and_duplicate_slots_are_robust(clean):
    rng = np.random.default_rng(7)
    m = fragment_slots(clean, seed=5, drop_prob=0.8, mean_drop_s=4.0, dup_per_min=20,
                       ghost_per_min=10)
    P = m.P.copy()
    P[rng.random(P.shape[:2]) < 0.2] = np.nan  # extra salt-and-pepper dropouts
    P[100:400, :, :] = np.nan                  # a stretch with nobody tracked
    m = replace(m, P=P)
    f = featurize(m)
    assert not np.isinf(f.X).any()
    for n in _never_nan(f.names):
        assert np.isfinite(f.col(n)).all(), n
    assert (f.col("n_present") >= 0).all()


def test_swap_is_an_involution(f_noisy):
    s = swap_teams(f_noisy)
    assert not np.array_equal(s.X, f_noisy.X, equal_nan=True)
    assert s.meta["swapped"] and not swap_teams(s).meta["swapped"]
    assert np.array_equal(swap_teams(s).X, f_noisy.X, equal_nan=True)


def test_swap_matches_featurizing_the_swapped_match(noisy, f_noisy):
    direct = featurize(swap_match_teams(noisy))
    assert direct.names == f_noisy.names
    assert np.allclose(swap_teams(f_noisy).X, direct.X, atol=1e-4, equal_nan=True)
    # the pairing really swaps: home's centroid becomes minus away's
    assert np.array_equal(direct.col("home_cx"), -f_noisy.col("away_cx"), equal_nan=True)
    assert np.array_equal(direct.col("nearest_margin"), -f_noisy.col("nearest_margin"),
                          equal_nan=True)


def test_swap_spec_covers_every_column(f_noisy):
    perm, sign = spec.swap_spec(f_noisy.names)
    assert np.array_equal(perm[perm], np.arange(len(perm)))
    assert np.array_equal(sign[perm] * sign, np.ones(len(sign), np.float32))


def test_extras_are_appended_and_subsampled(clean):
    n = clean.n
    extras = {"frame": np.arange(n, dtype=np.float32),
              "home_tag": np.full(n, 1.0, np.float32), "away_tag": np.full(n, 2.0, np.float32),
              "lead_signed": np.linspace(-1, 1, n).astype(np.float32)}
    f = featurize(clean, {"stride": 3}, extras=extras)
    assert np.array_equal(f.col("x_frame"), f.t_frame.astype(np.float32))
    s = swap_teams(f)
    assert (s.col("x_home_tag") == 2).all() and (s.col("x_away_tag") == 1).all()
    assert np.array_equal(s.col("x_lead_signed"), -f.col("x_lead_signed"))
    assert np.array_equal(s.col("x_frame"), f.col("x_frame"))
    with pytest.raises(ValueError):
        featurize(clean, extras={"bad": np.zeros(n - 1, np.float32)})


# ---------------------------------------------------------------------------------------------
# hand-computed values on a tiny constructed match
# ---------------------------------------------------------------------------------------------

def _tiny() -> Match:
    """One half of 10 frames in which HOME attacks -x, so features see every x and y negated.

    Raw positions (metres): home (10, 0), (20, 0), (0, 10); away (-5, 3), (-30, 0).
    Ball moves 0.2 m/frame along +x from (11.5, 0) on frames 0-3, is missing on 4-5, and sits at
    (14.5, 0) on frames 6-9.
    """
    n = 10
    home = np.array([[10, 0], [20, 0], [0, 10]], np.float32)
    away = np.array([[-5, 3], [-30, 0]], np.float32)
    P = np.repeat(np.concatenate([home, away])[None], n, axis=0)
    ball = np.full((n, 2), np.nan, np.float32)
    ball[0:4, 0] = 11.5 + 0.2 * np.arange(4)
    ball[0:4, 1] = 0
    ball[6:10] = (14.5, 0)
    return Match(P=P, team=np.array([0, 0, 0, 1, 1], np.int8),
                 speed=np.zeros((n, 5), np.float32), ball=ball, half_blocks=[(1, 0, n - 1)],
                 attack_dir={1: -1}, name="tiny")


def test_hand_computed_values():
    m = _tiny()
    m.validate()
    f = featurize(m)
    c = f.col
    # attack-normalised: home now attacks +x; home at (-10, 0), (-20, 0), (0, -10)
    assert np.isclose(c("ball_x")[0], -11.5) and np.isclose(c("ball_x")[3], -12.1)
    assert np.isclose(c("home_nearest_dist")[0], 1.5)
    assert np.isclose(c("home_second_dist")[0], 8.5)
    assert np.isclose(c("away_nearest_dist")[0], np.hypot(16.5, 3.0))
    assert np.isclose(c("nearest_margin")[0], np.hypot(16.5, 3.0) - 1.5)
    assert c("home_n2")[0] == 1 and c("home_n5")[0] == 1 and c("away_n5")[0] == 0
    L = m.pitch[0]
    assert np.isclose(c("home_ballgoal_dist")[0], L / 2 + 11.5)
    assert np.isclose(c("away_ballgoal_dist")[0], L / 2 - 11.5)
    # gap timers and interpolation over frames 4-5 (last seen 3, next seen 6)
    assert c("ball_since")[4:6].tolist() == [1, 2] and c("ball_until")[4:6].tolist() == [2, 1]
    assert c("ball_gap_len")[4:6].tolist() == [2, 2] and c("ball_gap_len")[3] == 0
    assert c("ball_interp")[4:6].tolist() == [1, 1] and np.isnan(c("ball_x")[4])
    assert np.allclose(c("ball_x_i")[4:6], [-12.9, -13.7])
    # constant velocity -0.2 m/frame along normalised x -> -5 m/s; static ball later
    assert np.allclose(c("ball_vx")[0:4], -5.0) and np.allclose(c("ball_speed")[0:4], 5.0)
    assert np.allclose(c("ball_speed")[6:10], 0.0) and np.isnan(c("ball_speed")[4])
    # team shape
    assert np.allclose(c("home_cx"), -10.0) and np.allclose(c("home_cy"), -10 / 3)
    assert np.allclose(c("home_gk_x"), -20) and np.allclose(c("home_line_x"), -10)
    assert np.allclose(c("home_top_x"), 0) and np.allclose(c("home_length"), 10)
    assert np.allclose(c("home_width"), 10) and np.allclose(c("home_area"), 100)
    assert np.allclose(c("away_gk_x"), 30) and np.allclose(c("away_line_x"), 5)
    assert np.allclose(c("adv_diff"), 7.5) and np.allclose(c("line_diff"), -5.0)
    d = np.hypot([0, 10, 10], np.array([10, 10, -20]) / 3)
    assert np.allclose(c("home_dcent"), d.mean(), atol=1e-5)
    assert np.allclose(c("home_mark_dist"), (np.hypot(15, 3) + np.hypot(5, 7)) / 2)
    assert np.allclose(c("away_mark_dist"), np.hypot(5, 7))
    # pressing: nobody within 5 m of either ball-nearest player
    assert c("away_pressers")[0] == 0 and np.isclose(c("home_carrier_space")[0], np.hypot(15, 3))
    assert np.isclose(c("away_carrier_space")[0], np.hypot(5, 7))
    # stillness: everyone has speed 0
    assert np.allclose(c("all_mspd"), 0) and np.allclose(c("frac_static"), 1)
    assert (c("n_present") == 5).all() and (c("home_n") == 3).all()


def test_hand_computed_context_windows():
    m = _tiny()
    f = featurize(m, {"windows_s": (0.2,), "step_s": (0.2,)})  # half-width round(2.5) = 2 frames
    pres = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], np.float32)
    expect = [pres[max(0, t - 2):t + 3].mean() for t in range(10)]
    assert np.allclose(f.col("ball_present__mean0.2s"), expect)
    spd = f.col("ball_speed")
    mx = f.col("ball_speed__max0.2s")
    assert np.isclose(mx[5], 5.0)  # window 3..7 holds 5, NaN, NaN, 0, 0
    assert np.isclose(mx[6], 0.0)  # window 4..8 holds NaN, NaN, 0, 0, 0
    # step = mean over the next window minus the previous one (clipped at the half's edges)
    bx = np.array([-11.5, -11.7, -11.9, -12.1, -12.9, -13.7, -14.5, -14.5, -14.5, -14.5])
    assert np.allclose(f.col("ball_x_i"), bx)
    st = f.col("ball_x_i__step0.2s")
    assert np.isclose(st[4], bx[4:9].mean() - bx[0:5].mean())
    assert np.isclose(st[0], bx[0:5].mean() - bx[0:3].mean())  # previous window clipped at t=0
    assert np.isfinite(spd).sum() == 8
