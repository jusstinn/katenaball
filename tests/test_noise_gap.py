import numpy as np
import pytest

from katena.noise import DEFAULT_CFG, corrupt, domain_gap, window_features
from katena.noise.gap import FEATURE_NAMES
from katena.synth import synthetic_match

CFG_A = dict(DEFAULT_CFG)
CFG_B = dict(DEFAULT_CFG, ball_present_rate=0.85, players_present_mean=17.0, pos_jitter_sd_m=0.4,
             dup_rate_per_min=30.0, frag_median_s=2.0, ball_gap_median_s=3.0)


@pytest.fixture(scope="module")
def clean():
    return synthetic_match(seed=3, half_minutes=3.0)


@pytest.fixture(scope="module")
def sims(clean):
    return corrupt(clean, CFG_A, seed=10), corrupt(clean, CFG_A, seed=11), corrupt(clean, CFG_B, seed=12)


def test_window_features_shape(sims):
    X, names, groups = window_features(sims[0], 5.0)
    assert names == FEATURE_NAMES and X.shape == (72, len(names)) and groups.shape == (72,)
    assert np.isfinite(X).mean() > 0.95


@pytest.mark.parametrize("model", ["lgbm", "logreg"])
def test_same_cfg_other_seed_is_indistinguishable(sims, model):
    res = domain_gap(sims[0], sims[1], window_s=5.0, model=model)
    assert 0.3 < res["auc"] < 0.7


@pytest.mark.parametrize("model", ["lgbm", "logreg"])
def test_very_different_cfg_is_separated(sims, model):
    res = domain_gap(sims[0], sims[2], window_s=5.0, model=model)
    assert res["auc"] > 0.9


def test_result_holds_aggregates_only(sims):
    res = domain_gap(sims[0], sims[2], window_s=5.0, top_k=5)
    assert set(res) == {"auc", "auc_folds", "auc_std", "n_real_windows", "n_sim_windows", "n_features",
                        "model", "top_features"}
    assert len(res["top_features"]) == 5
    for name, imp in res["top_features"]:
        assert name in FEATURE_NAMES and isinstance(imp, float)
    assert all(isinstance(a, float) for a in res["auc_folds"])
    # deterministic
    assert domain_gap(sims[0], sims[2], window_s=5.0, top_k=5) == res
