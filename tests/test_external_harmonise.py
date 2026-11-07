"""harmonise(): brief opponent flips are not changes of possession. Offline, fast."""
import numpy as np
import pytest

import possession_metrics as pm
from katena.external.harmonise import DEFAULT_HARMONISE, flip_table, harmonise
from katena.match import A, D, H, X
from katena.synth import synthetic_match

CFG = {"min_control_s": 2.0, "fps": 25}  # 50 frames


def seq(*parts):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in parts])


def test_brief_flip_that_returns_is_absorbed():
    y = seq((H, 100), (A, 20), (H, 100))
    assert (harmonise(y, CFG) == H).all()


def test_brief_flip_into_dead_ball_is_absorbed():
    y = seq((H, 100), (A, 20), (D, 50), (A, 100))
    assert harmonise(y, CFG).tolist() == seq((H, 120), (D, 50), (A, 100)).tolist()


def test_flip_at_end_of_run_is_absorbed():
    y = seq((H, 100), (A, 20), (X, 10))
    assert harmonise(y, CFG).tolist() == seq((H, 120), (X, 10)).tolist()


def test_long_opponent_spell_is_a_real_change():
    y = seq((H, 100), (A, 50), (H, 100))  # exactly min_control_s: kept
    assert np.array_equal(harmonise(y, CFG), y)


def test_restart_possessor_is_protected():
    y = seq((D, 50), (A, 10), (H, 200))
    assert np.array_equal(harmonise(y, CFG), y)
    y2 = seq((A, 10), (H, 200))  # start of a run behaves like a restart
    assert np.array_equal(harmonise(y2, CFG), y2)
    got = harmonise(y, {**CFG, "protect_restart": False})
    assert got.tolist() == seq((D, 50), (H, 210)).tolist()


def test_dead_and_x_frames_never_change_and_runs_are_independent():
    y = seq((H, 100), (X, 10), (A, 20), (H, 100), (D, 30), (H, 5), (A, 80))
    out = harmonise(y, CFG)
    assert np.array_equal(out == D, y == D) and np.array_equal(out == X, y == X)
    # A 20 after X starts a new run, so it is a restart and survives
    assert out[110:130].tolist() == [A] * 20


def test_shortest_first_vs_causal_chain():
    # H | A 1.0 s | H 0.32 s | A 10 s: the 8-frame H touch goes first, A keeps it throughout
    y = seq((H, 100), (A, 25), (H, 8), (A, 250))
    sf = harmonise(y, CFG)
    assert sf.tolist() == seq((H, 100), (A, 283)).tolist()
    causal = harmonise(y, {**CFG, "order": "causal"})
    assert causal.tolist() == seq((H, 133), (A, 250)).tolist()


def test_chain_of_flips_that_dies_goes_back_to_owner():
    y = seq((H, 250), (A, 10), (H, 8), (A, 12), (D, 40))
    assert harmonise(y, CFG).tolist() == seq((H, 280), (D, 40)).tolist()


def test_identity_when_threshold_is_tiny():
    m = synthetic_match(seed=0, half_minutes=1.0)
    assert np.array_equal(harmonise(m.y, {"min_control_s": 0.0}), m.y)


@pytest.mark.parametrize("order", ["shortest_first", "causal"])
def test_idempotent_and_leaves_no_removable_flip(order):
    rng = np.random.default_rng(0)
    parts, k = [], H
    for _ in range(300):
        r = rng.random()
        if r < 0.1:
            parts.append((D, int(rng.integers(25, 500))))
        else:
            k = A if k == H else H
            parts.append((k, int(rng.integers(1, 300))))
    y = seq(*parts)
    cfg = {**CFG, "order": order}
    once = harmonise(y, cfg)
    assert np.array_equal(harmonise(once, cfg), once)
    ft = flip_table(once, fps=25)
    for kind in ("return", "to_dead"):
        if len(ft[kind]):
            assert ft[kind].min() >= CFG["min_control_s"] - 1e-9
    assert pm.possession_stats(once, pm.valid_runs(once))["n_spells"] <= \
        pm.possession_stats(y, pm.valid_runs(y))["n_spells"]
    assert once.dtype == np.int8 and once.shape == y.shape


def test_default_config_is_complete():
    assert set(DEFAULT_HARMONISE) == {"min_control_s", "fps", "protect_restart", "order"}
    m = synthetic_match(seed=1, half_minutes=2.0)
    out = harmonise(m.y)
    assert out.shape == m.y.shape and (out[m.y == D] == D).all()
