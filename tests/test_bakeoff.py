"""Unit tests for the bake-off components, on the tiny synthetic fixture (no external data needed)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from synth import L, W, make_match

import possession_metrics as pm
from katena.decode import hmm
from katena.eval import protocol as P
from katena.eval.bootstrap import adopted, paired_bootstrap
from katena.eval.score import score, to_native_codes
from katena.featurize.build import featurize, infer_dirs, interp_short, wmean
from katena.grid import downsample, to_native
from katena.labels import sandwich
from katena.models.baseline import m0_predict
from katena.models.rules import m1_predict
from katena.sim import degrade as dg
from katena.types import A, D, H, X


# --- protocol -------------------------------------------------------------------------------

def test_locked_idsse_is_fixed_by_hash():
    ids = ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
           "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"]
    assert P.locked_idsse(ids) == "DFL-MAT-J03WQQ"
    assert P.idsse_hash("DFL-MAT-J03WQQ").startswith("26bd839f8207")
    assert P.locked_idsse(ids[::-1]) == "DFL-MAT-J03WQQ"          # order independent


def test_seeds_and_groups_are_deterministic():
    assert P.seed_for("a", "target") == P.seed_for("a", "target") != P.seed_for("a", "mild")
    assert 0 <= P.group_of("x") < 6


# --- labels ---------------------------------------------------------------------------------

def _runs(y):
    return pm.valid_runs(y)


def test_sandwich_merges_short_flip_only():
    y = np.array([3] + [H] * 20 + [A] * 5 + [H] * 20 + [A] * 30 + [H] * 20 + [3], dtype=np.int8)
    out = sandwich(y, _runs(y), fps=5.0, T=2.0)             # 2 s = 10 samples
    assert (out[21:26] == H).all()                          # 5-sample A flip absorbed
    assert (out[46:76] == A).all()                          # 30-sample A kept
    assert sandwich(y, _runs(y), 5.0, 0.0).tolist() == y.tolist()


def test_sandwich_respects_dead_ball_and_run_edges():
    y = np.array([H] * 20 + [A] * 3 + [D] * 5 + [H] * 20, dtype=np.int8)
    assert (sandwich(y, _runs(y), 5.0, 5.0) == y).all()     # A is next to D, not sandwiched
    y2 = np.array([H] * 10 + [3] * 3 + [A] * 3 + [3] * 3 + [H] * 10, dtype=np.int8)
    assert (sandwich(y2, _runs(y2), 5.0, 5.0) == y2).all()  # X breaks the sandwich


# --- grid -----------------------------------------------------------------------------------

def test_downsample_and_back():
    m = make_match()
    m5 = downsample(m)
    assert m5.fps == 5.0 and m5.n == 400 and m5.meta["step"] == 5
    codes = np.arange(m5.n) % 3
    back = to_native(codes.astype(np.int8), 5, m.n)
    assert len(back) == m.n and back[0] == 0 and back[5] == 1 and back[7] == 1 and back[8] == 2


# --- degrader -------------------------------------------------------------------------------

def _long_match():
    """A longer clean match: the fixture tiled so burst statistics are not dominated by edges."""
    m = make_match()
    m.ball[:] = np.where(np.isfinite(m.ball), m.ball, 0.0)   # clean ball
    return m


def test_corrupt_is_deterministic_and_keeps_labels():
    m5 = downsample(_long_match())
    a, b, c = dg.corrupt(m5, dg.TARGET, 1), dg.corrupt(m5, dg.TARGET, 1), dg.corrupt(m5, dg.TARGET, 2)
    np.testing.assert_array_equal(a.P, b.P)
    np.testing.assert_array_equal(a.ball, b.ball)
    assert not np.array_equal(np.nan_to_num(a.P), np.nan_to_num(c.P))
    np.testing.assert_array_equal(a.y, m5.y)
    assert a.half_blocks == m5.half_blocks and a.attack_dir == m5.attack_dir


def test_corrupt_strips_identity_into_46_slots():
    m5 = downsample(_long_match())
    c = dg.corrupt(m5, dg.TARGET, 3)
    assert c.P.shape[1] == 46 and (c.team == 0).sum() == 23 and (c.team == 1).sum() == 23
    assert c.extras == {}


def test_ball_missing_orders_by_severity_and_never_below_native():
    m5 = downsample(_long_match())
    fr = []
    for cfg in (dg.MILD, dg.TARGET, dg.SEVERE):
        c = dg.corrupt(m5, cfg, 5)
        fr.append(c.meta["degrade"]["ball_missing"])
    assert fr[0] < fr[1] < fr[2]
    assert abs(fr[1] - 0.41) < 0.08 and abs(fr[0] - 0.20) < 0.08   # 560-sample fixture: coarse
    m_nat = downsample(make_match())            # native ball loss about 14 %
    nat = float((~np.isfinite(m_nat.ball[:, 0]))[[i for _, a, b in m_nat.half_blocks for i in range(a, b + 1)]].mean())
    assert dg.corrupt(m_nat, dg.MILD, 1).meta["degrade"]["ball_missing"] >= nat - 1e-9


def test_dead_coupling_concentrates_ball_loss_in_dead_frames():
    m5 = downsample(_long_match())
    plain = dg.corrupt(m5, dg.TARGET, 7)
    coupled = dg.corrupt(m5, dg.with_dead_odds(dg.TARGET, 15.0), 7)
    dead = m5.y == D
    miss = lambda c: (~np.isfinite(c.ball[:, 0]))[dead].mean()   # noqa: E731
    assert miss(coupled) >= miss(plain)


# --- features -------------------------------------------------------------------------------

def test_wmean_and_interp_helpers():
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    assert wmean(x, 1, 1)[1] == 2.0 and wmean(x, 0, 0)[0] == 1.0 and np.isnan(wmean(x, 0, 0)[1])
    b = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [8, 8.0]])
    out, interp, glen = interp_short(b, 2)
    assert np.allclose(out[1], [1, 1]) and interp[1] and not interp[4] and glen[4] == 3


def test_attack_direction_is_inferred_from_positions():
    m5 = downsample(make_match())
    assert infer_dirs(m5) == {1: 1, 2: -1}


def test_features_are_side_symmetric():
    """Swapping team labels and rotating the pitch 180 degrees must give the same focal features."""
    m5 = downsample(make_match())
    sw = dataclasses.replace(m5, P=np.stack([L - m5.P[..., 0], W - m5.P[..., 1]], -1).astype(np.float32),
                             ball=np.stack([L - m5.ball[:, 0], W - m5.ball[:, 1]], -1).astype(np.float32),
                             team=(1 - m5.team).astype(np.int8))
    f0, f1 = featurize(m5, 0), featurize(sw, 1)
    np.testing.assert_allclose(f0.X, f1.X, atol=1e-3, equal_nan=True)
    assert f0.X.shape[1] == len(f0.names) == len(f0.groups)
    assert set(f0.groups) == {"ball", "free", "dead_ball", "dead_player", "miss"}


def test_corrupted_features_have_same_columns():
    m5 = downsample(make_match())
    c = dg.corrupt(m5, dg.TARGET, 1)
    assert featurize(c, 0).names == featurize(m5, 0).names


# --- models ---------------------------------------------------------------------------------

def test_m0_uses_starter_smooth_and_labels_missing_ball_dead():
    m5 = downsample(make_match())
    out = m0_predict(m5)
    inside = np.zeros(m5.n, bool)
    for _, a, b in m5.half_blocks:
        inside[a:b + 1] = True
    assert (out[~inside] == X).all() and set(np.unique(out[inside])) <= {H, A, D}
    nb = m5.__class__(**{**m5.__dict__, "ball": np.full_like(m5.ball, np.nan)})
    assert (m0_predict(nb)[inside] == D).all()


def test_m1_rule_chain_on_scripted_features():
    names = ["d_f1", "d_o1", "ball_oob_dist", "frac_still_m2s", "ball_speed_mean1s"]
    n = 160
    X = np.zeros((n, 5), np.float32)
    X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4] = 6.0, 6.0, 5.0, 0.1, 8.0
    X[0:50, 0] = 1.0                       # home controls
    X[50:70, :2] = np.nan                  # ball lost: possession is held
    X[70:100, 1] = 1.0                     # away controls
    X[100:120, 2] = -2.0                   # ball out of the pitch -> dead
    X[120:160, 0] = 1.0                    # restart, home controls again
    pred = m1_predict(X, names, np.ones(n, np.int8))
    assert (pred[:50] == 0).all() and (pred[50:70] == 0).all()
    assert (pred[80:100] == 1).all()
    assert (pred[106:120] == 2).all()
    assert (pred[140:] == 0).all()


# --- decoders -------------------------------------------------------------------------------

def _seq(blocks):
    return np.concatenate([np.full(n, k, np.int8) for k, n in blocks])


def _probs(y, conf=0.9):
    pa = np.where(y == 2, 0.05, 0.95)
    ph = np.where(y == 0, conf, np.where(y == 1, 1 - conf, 0.5))
    return pa.astype(np.float32), ph.astype(np.float32)


def test_priors_fit_and_decoders_recover_clean_sequences():
    y = _seq([(0, 60), (1, 40), (2, 30), (1, 50), (0, 70), (2, 20), (0, 40)])
    pri = hmm.fit_priors([y, y])
    assert np.isfinite(pri.logA).all() and np.allclose(np.exp(pri.logA).sum(1), 1)
    assert pri.logT[0, 0] < -1e17 and pri.logdur.shape == (3, hmm.KMAX + 1)
    pa, ph = _probs(y)
    half = np.ones(len(y), np.int8)
    for kind in ("viterbi", "hsmm"):
        out = hmm.decode(pa, ph, half, pri, 1.0, kind)
        assert (out == y).mean() > 0.97, kind


def test_hsmm_removes_short_blips_viterbi_keeps_with_weak_prior():
    y = _seq([(0, 100), (1, 100)])
    pa, ph = _probs(y, 0.75)
    ph = ph.copy()
    ph[40:43] = 0.2                                        # a 3-sample contrary blip
    pri = hmm.fit_priors([_seq([(0, 120), (1, 150), (2, 40), (0, 130), (1, 110), (2, 60), (0, 90)])])
    half = np.ones(len(y), np.int8)
    out = hmm.decode(pa, ph, half, pri, 1.0, "hsmm")
    assert (out[40:43] == 0).all()
    assert len(hmm.runs_of(out)) == 1


def test_decode_handles_each_half_separately():
    y = np.concatenate([_seq([(0, 50), (1, 50)]), _seq([(1, 50), (0, 50)])])
    half = np.repeat(np.array([1, 2], np.int8), 100)
    pa, ph = _probs(y)
    pri = hmm.fit_priors([_seq([(0, 50), (1, 50), (2, 30), (1, 50), (0, 50), (2, 30), (0, 40)])])
    out = hmm.decode(pa, ph, half, pri, 1.0, "hsmm")
    assert (out == y).mean() > 0.97


# --- scoring --------------------------------------------------------------------------------

def test_score_matches_official_and_flags_reproduce_f1():
    m = make_match()
    gt = m.y.copy()
    pred = gt.copy()
    pred[200:230] = (pred[200:230] + 1) % 2 if (gt[200:230] < 2).all() else pred[200:230]
    s = score(gt, pred)
    off = pm.evaluate(gt, pred)
    assert s["spell_f1"] == off["spell_f1_mean"] and s["macro_f1"] == off["macro_f1"]
    tp = s["flags"].sum(1)
    f1 = np.mean([2 * t / (2 * t + max(0, s["n_pred"] - t) + (s["n_gt"] - t)) for t in tp])
    assert abs(f1 - s["spell_f1"]) < 1e-12


def test_score_sc_mode_replaces_dead_with_abstention():
    m = make_match()
    gt = np.where(m.y == D, X, m.y).astype(np.int8)          # SkillCorner-like: no D in the labels
    pred = m.y.copy()
    s = score(gt, pred, sc_mode=True)
    assert s["spell_f1"] > 0.99


def test_to_native_codes_places_rows():
    idx = np.array([0, 1, 2])
    out = to_native_codes(np.array([0, 1, 2], np.int8), idx, 5, 15)
    assert out[0] == 0 and out[5] == 1 and out[10] == 2


def test_paired_bootstrap_rule():
    m = make_match()
    gt = m.y
    good = score(gt, gt)
    bad_pred = gt.copy()
    bad_pred[::37] = 1 - np.minimum(bad_pred[::37], 1)
    bad = score(gt, bad_pred)
    same = paired_bootstrap([good] * 6, [good] * 6, draws=200)
    assert same["mean_diff"] == 0 and not adopted(same)
    win = paired_bootstrap([good] * 6, [bad] * 6, draws=200)
    assert win["win_rate"] == 1.0 and win["lo"] > 0 and adopted(win)


# --- spell rescoring and error census --------------------------------------------------------

def test_spell_features_targets_and_apply():
    from katena.models import spells as SP
    rows = _seq([(0, 60), (1, 8), (0, 60), (2, 30), (1, 60)])
    half = np.ones(len(rows), np.int8)
    pa = np.where(rows == 2, 0.05, 0.95).astype(np.float32)
    ph = np.where(rows == 0, 0.9, 0.1).astype(np.float32)
    names = ["ctrl_c3", "margin_c3", "tightf_c3", "tighto_c3", "d_f1", "d_o1", "ball_vis"]
    X0 = np.ones((len(rows), len(names)), np.float32)
    segs, F, which = SP.spell_features(rows, pa, ph, X0, names, half)
    assert F.shape == (len(which), len(SP.NAMES)) == (4, len(SP.NAMES))
    assert F[1, SP.NAMES.index("sandwich")] == 1.0           # the 8-sample A spell sits between two H spells
    gt = _seq([(0, 128), (2, 30), (1, 60)])                  # ground truth has no A blip
    tg = SP.spell_targets(segs, which, gt)
    assert tg.tolist() == [SP.KEEP, SP.FLIP, SP.KEEP, SP.KEEP]


def test_error_census_classifies_each_type():
    from katena.eval.errors import census
    gt = np.array([3] * 5 + [0] * 100 + [2] * 20 + [1] * 100 + [0] * 100 + [3] * 5, dtype=np.int8)
    pred = gt.copy()
    pred[125:225] = 0                               # second GT spell predicted as the wrong team
    pred[25:105] = 2                                # first GT spell mostly predicted dead
    c = census(gt, pred)
    assert c["gt_spells"]["wrong_team"] == 1 and c["gt_spells"]["dead_miss"] == 1
    assert c["gt_spells"]["ok"] == 1 and c["n_gt"] == 3
    assert c["dead_frame_err"]["gt_alive_pred_dead"] > 0


def test_heads_and_spell_model_train_when_lightgbm_loads():
    """Needs libomp. Run as: DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.11/site-packages/sklearn/.dylibs uv run pytest"""
    from katena.models import gbdt
    from katena.models import spells as SP
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = np.where(X[:, 0] > 0.5, 0, np.where(X[:, 0] < -0.5, 1, 2)).astype(np.int8)
    o = np.zeros(n, np.int8)
    heads = gbdt.fit_heads(X, o, y, np.ones(n, bool), np.arange(6), n_trees=20)
    pa, ph = heads.predict(X, X)
    assert pa.shape == (n,) and (pa[y == 2].mean() < pa[y < 2].mean())
    F = rng.normal(size=(200, len(SP.NAMES))).astype(np.float32)
    t = (F[:, 0] > 0.7).astype(np.int8)
    model = SP.fit(F, t)
    assert model.predict(F).shape == (200, 3)
