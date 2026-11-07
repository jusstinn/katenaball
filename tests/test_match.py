import numpy as np
import possession_metrics as pm

from katena.match import H, A, D, X, from_normalised
from katena.synth import synthetic_match


def test_synthetic_match_is_valid_and_deterministic():
    a = synthetic_match(seed=1, half_minutes=1.0)
    b = synthetic_match(seed=1, half_minutes=1.0)
    a.validate()
    assert np.array_equal(a.y, b.y)
    assert np.allclose(a.P, b.P, equal_nan=True)


def test_synthetic_labels_have_possession_structure():
    m = synthetic_match(seed=0, half_minutes=6.0)
    g = m.graded_mask()
    assert (m.y[~g] == X).all()
    assert set(np.unique(m.y[g])) <= {H, A, D}
    runs = pm.valid_runs(m.y)
    assert len(runs) == 2
    stats = pm.possession_stats(m.y, runs)
    assert 3.0 < stats["mean_duration_s"] < 40.0


def test_scorer_perfect_on_synthetic():
    m = synthetic_match(seed=2, half_minutes=2.0)
    res = pm.evaluate(m.y, m.y.copy())
    assert res["spell_f1_mean"] == 1.0


def test_from_normalised_centres_pitch():
    xy = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    out = from_normalised(xy, pitch=(100.0, 60.0))
    assert np.allclose(out, [[0, 0], [-50, 30]])
