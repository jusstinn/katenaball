"""D4 model on synthetic data only (offline, fast)."""
import dataclasses

import numpy as np
import pytest

from katena.feats import featurize, swap_teams
from katena.match import A, D, H, X
from katena.model import Model
from katena.model import data as Dm
from katena.model.baseline import nearest_slot_team, smooth_majority, starter_baseline
from katena.noise import corrupt
from katena.synth import synthetic_match

SMALL = {"num_leaves": 15, "min_data_in_leaf": 20, "num_threads": 2}


def _copy(seed, stride=5):
    m = corrupt(synthetic_match(seed=seed, half_minutes=1.0), {}, seed=seed + 100)
    f = featurize(m, {"stride": stride})
    return {"X": f.X, "t_frame": f.t_frame, "y": m.y[f.t_frame].astype(np.int8), "names": list(f.names)}, f, m


@pytest.fixture(scope="module")
def corpus():
    train = [_copy(s)[0] for s in (1, 2)]
    _, f_test, m_test = _copy(3)
    X, y, names = Dm.stack_training(train, swap=True)
    return X, y, names, f_test, m_test


@pytest.fixture(scope="module", params=["twohead", "3class"])
def fitted(request, corpus):
    X, y, names, _, _ = corpus
    return Model(kind=request.param, params=SMALL, rounds=15).fit(X, y, names)


def test_predict_logp_shape_dtype_and_normalisation(fitted, corpus):
    f = corpus[3]
    lp = fitted.predict_logp(f)
    assert lp.shape == (f.T, 3) and lp.dtype == np.float32
    assert np.isfinite(lp).all() and (lp <= 1e-6).all()
    assert np.allclose(np.exp(lp.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)


def test_predict_logp_is_deterministic_across_fits_and_save_load(fitted, corpus, tmp_path):
    X, y, names, f, _ = corpus
    again = Model(kind=fitted.kind, params=SMALL, rounds=15).fit(X, y, names)
    a, b = fitted.predict_logp(f), again.predict_logp(f)
    assert np.array_equal(a, b)
    assert np.array_equal(a, fitted.predict_logp(f))
    p = fitted.save(tmp_path / "m.json")
    loaded = Model.load(p)
    assert np.array_equal(loaded.predict_logp(f), a)
    assert loaded.columns == fitted.columns and loaded.kind == fitted.kind


def test_column_order_of_the_input_does_not_matter(fitted, corpus):
    f = corpus[3]
    rev = f.select(list(reversed(f.names)))
    assert np.array_equal(fitted.predict_logp(rev), fitted.predict_logp(f))


def test_temperature_fit_lowers_nll_and_keeps_normalisation(fitted, corpus):
    X, y, names, f, m = corpus
    yt = m.y[f.t_frame].astype(np.int64)
    raw = fitted.raw_scores(f.X, f.names)
    t = fitted.fit_temperature(raw, yt)
    assert all(0.2 <= v <= 5.0 for v in t.values())

    def nll(temp):
        lp = fitted.logp_from_raw(raw, temp).astype(np.float64)
        return -lp[np.arange(len(yt)), yt].mean()
    assert nll(t) <= nll(fitted._unit_temperature()) + 1e-9
    m2 = Model.from_dict(fitted.to_dict()).set_temperature(t)
    lp = m2.predict_logp(f)
    assert np.allclose(np.exp(lp.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)
    with pytest.raises(KeyError):
        m2.set_temperature({"nope": 1.0})


def test_twohead_combines_heads_as_documented():
    m = Model(kind="twohead")
    raw = np.array([[0.0, 0.0], [3.0, -2.0], [-4.0, 5.0]])
    lp = m.logp_from_raw(raw).astype(np.float64)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    pa, ph = sig(raw[:, 0]), sig(raw[:, 1])
    assert np.allclose(np.exp(lp), np.stack([pa * ph, pa * (1 - ph), 1 - pa], axis=1), atol=1e-6)


def test_ball_free_model_uses_no_ball_columns(corpus):
    X, y, names, f, _ = corpus
    mdl = Model(kind="twohead", features="ball_free", params=SMALL, rounds=5).fit(X, y, names)
    assert mdl.columns and not any(c.startswith("ball") or "nearest" in c for c in mdl.columns)
    assert mdl.predict_logp(f).shape == (f.T, 3)


def test_stack_training_swaps_labels_and_columns(corpus):
    c, f, _ = _copy(4)
    X, y, names = Dm.stack_training([c], swap=True)
    k = len(c["y"])
    assert X.shape == (2 * k, len(names))
    assert np.array_equal(y[:k], c["y"])
    sw = y[k:]
    assert np.array_equal(sw == H, c["y"] == A) and np.array_equal(sw == D, c["y"] == D)
    assert np.array_equal(X[k:], swap_teams(f).X, equal_nan=True)


def test_fill_missing_ball_is_label_free_and_complete():
    m = synthetic_match(seed=5, half_minutes=1.0)
    ball = m.ball.copy()
    dead = m.y == D
    ball[dead] = np.nan               # Metrica-style: ball removed during dead ball
    m2 = dataclasses.replace(m, ball=ball)
    filled, diag = Dm.fill_missing_ball(m2)
    g = m.graded_mask()
    assert np.isfinite(filled.ball[g]).all()
    assert diag["frames_filled"] == int((dead & g).sum())
    assert np.array_equal(filled.ball[g & ~dead], m.ball[g & ~dead])
    # label-free: same output when the labels are removed
    filled2, _ = Dm.fill_missing_ball(dataclasses.replace(m2, y=None))
    assert np.array_equal(filled.ball, filled2.ball, equal_nan=True)


def test_locked_matches_are_refused_and_seeds_are_disjoint():
    with pytest.raises(PermissionError):
        Dm.check_names([Dm.LOCKED_NAMES[0]])
    assert set(Dm.LOCKED_NAMES).isdisjoint(Dm.DEV_NAMES) and len(Dm.DEV_NAMES) == 6
    for n in Dm.DEV_NAMES:
        tr = {s.seed for a in ("severity", "default") for s in Dm.train_plan(n, a)}
        assert Dm.eval_spec(n).seed not in tr
    sev = Dm.train_plan(Dm.DEV_NAMES[0], "severity")
    dflt = Dm.train_plan(Dm.DEV_NAMES[0], "default")
    assert len({s.key() for s in sev}) == 5 and sev[2].key() == dflt[2].key()
    assert sorted(s.severity for s in sev) == pytest.approx([0.5, 0.7071, 1.0, 1.4142, 2.0], abs=1e-3)


def test_starter_baseline_on_a_match():
    m = synthetic_match(seed=6, half_minutes=1.0)
    codes = nearest_slot_team(m)
    assert codes.shape == (m.n,) and set(np.unique(codes)) <= {-1, 0, 1}
    out = starter_baseline(m, 50)
    g = m.graded_mask()
    assert (out[~g] == X).all() and np.isin(out[g], (H, A, D)).all()
    sm = smooth_majority(np.array([0, 0, 1, -1, -1, -1, -1, -1], dtype=np.int8), 1)
    assert sm.tolist() == [H, H, H, A, D, D, D, D]
