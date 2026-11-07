"""D4 training across label conventions (katena.eval.tmix) on synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

import possession_metrics as pm
from katena.eval import tmix as TM
from katena.types import A, D, H, X


def _random_labels(rng, n_segs=60):
    """Random H/A/D sequence with X gaps, short flips common."""
    codes, lens = [], []
    cur = int(rng.integers(0, 3))
    for _ in range(n_segs):
        cur = int(rng.choice([c for c in (H, A, D) if c != cur]))
        codes.append(cur)
        lens.append(int(rng.choice([1, 2, 3, 5, 8, 12, 20, 40])))
    y = np.repeat(np.array(codes, np.int8), lens)
    y[: int(rng.integers(0, 4))] = X
    cut = int(rng.integers(len(y) // 3, 2 * len(y) // 3))
    y[cut:cut + 3] = X
    return y



def test_team_design_single_mix_aug():
    o = np.array([0, 1, 0, 1, 0, 0], np.int8)
    ys = [np.array([0, 1, 2, 0, 3, 1], np.int8), np.array([0, 0, 2, 0, 3, 1], np.int8),
          np.array([0, 0, 2, 1, 3, 0], np.int8)]
    rows, t, w = TM.team_design("single", ys[:1], o)
    assert rows.tolist() == [0, 1, 3, 5] and t.tolist() == [1, 1, 0, 0] and w is None
    draw = np.array([2, 1, 0, 2, 0, 0])
    rows, t, w = TM.team_design("mix", ys, o, draw)
    assert rows.tolist() == [0, 1, 3, 5] and t.tolist() == [1, 0, 1, 0] and w is None
    rows, t, w = TM.team_design("aug", ys, o)
    # row 0: always focal (1); row 1: 1/3; row 3: 1/3; row 5: 1/3
    assert rows.tolist() == [0, 1, 3, 5, 1, 3, 5]
    assert t.tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert np.allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3])
    soft = np.bincount(rows, weights=w * t, minlength=6)[[0, 1, 3, 5]] / np.bincount(rows, weights=w, minlength=6)[[0, 1, 3, 5]]
    assert np.allclose(soft, [1, 1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(AssertionError):
        TM.team_design("aug", [ys[0], np.array([0, 1, 1, 0, 3, 1], np.int8)], o)   # D changed


def test_mix_draw_is_fixed():
    a, b = TM.mix_draw(1000), TM.mix_draw(1000)
    assert np.array_equal(a, b) and set(np.unique(a)) == set(range(5))


def test_spell_score_equals_official_spell_f1():
    rng = np.random.default_rng(3)
    for _ in range(60):
        gt = _random_labels(rng)
        pred = gt.copy()
        flip = rng.random(len(pred)) < 0.08
        pred[flip] = rng.integers(0, 4, int(flip.sum()))
        for sc in (False, True):
            p = np.where(pred == D, X, pred).astype(np.int8) if sc else pred
            assert TM.spell_score(gt, pred, sc_mode=sc)["spell_f1"] == pytest.approx(pm.evaluate(gt, p)["spell_f1_mean"], abs=1e-12)


def test_regret_picks_the_worst_convention():
    means = {("a", "S0"): 0.5, ("a", "S5"): 0.6, ("b", "S0"): 0.45, ("b", "S5"): 0.65}
    rg = TM.regret(means, ["a", "b"], ("S0", "S5"))
    assert rg["a"][:2] == (pytest.approx(0.05), "S5") and rg["b"][:2] == (pytest.approx(0.05), "S0")


def test_lightgbm_dataset_reuse_with_set_label_matches_a_fresh_dataset():
    from katena.models import gbdt
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load")
    rng = np.random.default_rng(4)
    Xa = rng.normal(size=(3000, 6)).astype(np.float32)
    y1 = (Xa[:, 0] + rng.normal(0, 1, 3000) > 0).astype(np.float32)
    y2 = (Xa[:, 1] + rng.normal(0, 1, 3000) > 0).astype(np.float32)
    params = dict(gbdt.PARAMS, num_threads=2)
    shared = gbdt.lgb.Dataset(Xa, y1, free_raw_data=False)
    gbdt.lgb.train(params, shared, 20)
    shared.set_label(y2)
    reused = gbdt.lgb.train(params, shared, 20).predict(Xa)
    fresh = gbdt.lgb.train(params, gbdt.lgb.Dataset(Xa, y2), 20).predict(Xa)
    assert np.array_equal(reused, fresh)
