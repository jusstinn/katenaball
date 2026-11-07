"""Native-frame-rate touch features (katena.featurize.touch), on synthetic fixtures only."""

from __future__ import annotations

import dataclasses

import numpy as np
from synth import L, W, make_match

from katena.featurize import touch as T
from katena.featurize.build import featurize
from katena.grid import downsample
from katena.sim import degrade as dg
from katena.types import Match

FPS = 25.0


def _passing_match(seed: int = 0, sigma: float = 0.0) -> tuple[Match, list]:
    """One 60 s half at 25 fps: four static players (home 0 and 2, away 1 and 3) and a ball passed
    between them in straight lines. Returns the match and the scripted touches [(frame, team)]."""
    rng = np.random.default_rng(seed)
    n = int(60 * FPS)
    spots = np.array([[30.0, 30.0], [50.0, 30.0], [42.0, 50.0], [62.0, 48.0]])
    team = np.array([0, 1, 0, 1], dtype=np.int8)
    P = np.repeat(spots[None].astype(np.float32), n, 0)
    seq = [0, 2, 1, 3, 0, 1, 2, 3, 1, 0]               # carrier order; each holds 1.5 s, then passes at 15 m/s
    ball = np.full((n, 2), np.nan)
    touches = []
    t = 25
    ball[:t] = spots[seq[0]] + 0.5
    for a, b in zip(seq[:-1], seq[1:]):
        hold = int(1.5 * FPS)
        ball[t:t + hold] = spots[a] + 0.5                 # at the carrier's feet
        t += hold
        touches.append((t, int(team[a])))                 # the kick
        p0, p1 = spots[a] + 0.5, spots[b] + 0.5
        k = int(np.ceil(np.linalg.norm(p1 - p0) / 15.0 * FPS))
        w = np.arange(1, k + 1) / k
        ball[t:t + k] = p0 * (1 - w)[:, None] + p1 * w[:, None]
        t += k
        touches.append((t, int(team[b])))                 # the reception
    ball[t:] = spots[seq[-1]] + 0.5
    ball = ball + rng.normal(0, sigma, ball.shape) if sigma > 0 else ball
    y = np.zeros(n, dtype=np.int8)
    m = Match(P=P, team=team, speed=np.zeros((n, 4), np.float32), ball=ball.astype(np.float32),
              half_blocks=[(1, 0, n - 1)], attack_dir={1: 1}, y=y, pitch=(L, W), fps=FPS, name="passes",
              source="synthetic")
    return m, touches


def test_detects_scripted_kicks_and_receptions_and_attributes_them():
    m, touches = _passing_match()
    det = T.detect_touches(m.ball.astype(np.float64), FPS, lambda fr: m.P[fr], m.team)
    found = det["t"]
    hits = 0
    for t, tm in touches:
        k = np.flatnonzero(np.abs(found - t) <= 2)
        if len(k):
            hits += 1
            assert det["who"][k[0]] == tm
    assert hits >= len(touches) - 1
    assert len(found) <= len(touches) + 2                 # no spurious breaks on a clean path


def test_detection_survives_jitter_with_a_matching_noise_estimate():
    m, touches = _passing_match(seed=1, sigma=0.3)
    det = T.detect_touches(m.ball.astype(np.float64), FPS, lambda fr: m.P[fr], m.team)
    assert 0.2 < np.median(det["sigma"]) < 0.4             # jitter estimate finds sigma
    near = [np.any(np.abs(det["t"] - t) <= 4) for t, _ in touches]
    assert np.mean(near) >= 0.7
    assert len(det["t"]) <= 2 * len(touches)


def test_lift_ball_downsamples_to_the_5hz_corruption():
    m = make_match()
    m.ball[:] = np.where(np.isfinite(m.ball), m.ball, 0.0)
    m5 = downsample(m)
    mc = dg.corrupt(m5, dg.TARGET, 3)
    lift = T.lift_ball(m.ball, m5.ball, mc.ball, 5, dg.TARGET.sigma_ball, 11)
    np.testing.assert_array_equal(lift[::5], mc.ball)                     # NaN positions included
    dropped = np.flatnonzero(np.isfinite(m5.ball[:, 0]) & ~np.isfinite(mc.ball[:, 0]))
    for i in dropped[:50]:
        assert np.isnan(lift[max(0, 5 * i - 2):5 * i + 3, 0]).all()     # a burst covers its native frames
    off = np.isfinite(lift[:, 0]) & (np.arange(m.n) % 5 != 0)
    resid = (lift - m.ball)[off]
    resid = resid[np.hypot(resid[:, 0], resid[:, 1]) < 3.0]               # drop held false detections
    assert abs(resid.std() - dg.TARGET.sigma_ball) < 0.08
    same = T.lift_ball(m.ball, m5.ball, m5.ball, 5, 0.0, 0)               # no corruption: identity
    np.testing.assert_array_equal(same, m.ball)


def test_touch_rows_align_and_featurize_appends_the_group():
    m = make_match()
    m5 = downsample(m)
    nb = T.native_ball(m)
    base = featurize(m5, 0)
    full = featurize(m5, 0, native=nb)
    np.testing.assert_array_equal(full.idx, base.idx)
    np.testing.assert_array_equal(full.X[:, :base.X.shape[1]], base.X)
    assert list(full.groups[base.X.shape[1]:]) == [T.GROUP] * (full.X.shape[1] - base.X.shape[1])
    assert len(set(full.names)) == len(full.names)
    sel = full.select(drop=(T.GROUP,))
    np.testing.assert_array_equal(full.X[:, sel], base.X)


def test_touch_features_are_side_symmetric():
    """Swapping team labels and rotating the pitch gives the same focal features for the other focal."""
    m, _ = _passing_match(seed=2, sigma=0.2)
    m5 = downsample(m)
    nb = T.native_ball(m)
    sw = dataclasses.replace(m5, P=np.stack([L - m5.P[..., 0], W - m5.P[..., 1]], -1).astype(np.float32),
                             ball=np.stack([L - m5.ball[:, 0], W - m5.ball[:, 1]], -1).astype(np.float32),
                             team=(1 - m5.team).astype(np.int8))
    nb_sw = T.NativeBall(np.stack([L - nb.ball[:, 0], W - nb.ball[:, 1]], -1).astype(np.float32), nb.fps, nb.step)
    X, names, _ = T.touch_features(m5, nb)
    Xs, _, _ = T.touch_features(sw, nb_sw)
    np.testing.assert_allclose(X[0], Xs[1], atol=1e-3, equal_nan=True)
    assert X[0].shape[1] == len(names)


def test_touch_features_with_no_ball_are_defined():
    m = make_match()
    m.ball[:] = np.nan
    m5 = downsample(m)
    X, names, _ = T.touch_features(m5, T.native_ball(m))
    col = {n: X[0][:, i] for i, n in enumerate(names)}
    assert (col["t_since_f"] == T.CAP_S).all() and (col["t_last_team"] == 0).all()
    assert (col["t_vis_cell"] == 0).all() and np.isnan(col["t_dminf_cell"]).all()
