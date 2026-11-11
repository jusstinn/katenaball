"""Possession-length weights (S1, S2): the weight itself, and that an absent weight changes nothing in either
member. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from katena.models import gbdt
from katena.models import lenweight as LW


def test_run_seconds_and_clipping():
    y = np.array([0] * 50 + [1] * 5 + [2] * 100 + [0] * 2000, np.int8)
    s = LW.run_seconds(y, 25.0)
    assert s[0] == s[49] == 2.0 and s[50] == s[54] == 0.2 and s[55] == 4.0 and s[-1] == 80.0
    w = LW.inv_length(y, 25.0)
    assert w[0] == 0.5 and w[50] == 1.0 and w[55] == 0.25 and w[-1] == 1 / 30      # clipped to [1 s, 30 s]
    assert len(LW.run_seconds(np.zeros(0, np.int8), 25.0)) == 0
    np.testing.assert_array_equal(LW.run_seconds(np.array([3, 3, 3]), 1.0), [3.0, 3.0, 3.0])


def test_at_rows_reads_the_native_frame_of_each_row():
    w = np.arange(11, dtype=float)
    np.testing.assert_array_equal(LW.at_rows(w, np.array([0, 1, 2, 3]), 5), [0.0, 5.0, 10.0, 10.0])   # last row clamped


def _seqs(w=None):
    rng = np.random.default_rng(0)
    out = []
    for i in range(3):
        y = np.repeat(rng.integers(0, 3, 40), rng.integers(2, 30, 40)).astype(np.int8)
        s = {"X": rng.standard_normal((len(y), 6)).astype(np.float32), "y": y, "o": i % 2, "dead_ok": True}
        if w is not None:
            s["w"] = w(y)
        out.append(s)
    return out


def test_tcn_without_weights_is_unchanged_and_weights_change_it():
    pytest.importorskip("torch")
    from katena.models import ensemble as E

    state = lambda m: np.concatenate([p.detach().numpy().ravel() for p in m.parameters()])   # noqa: E731
    base = state(E.fit_tcn(_seqs(), seed=0, steps=4))
    ones = state(E.fit_tcn(_seqs(lambda y: np.ones(len(y), np.float32)), seed=0, steps=4))
    inv = state(E.fit_tcn(_seqs(lambda y: LW.inv_length(y, 5.0).astype(np.float32)), seed=0, steps=4))
    assert base.tobytes() == ones.tobytes()
    assert not np.array_equal(base, inv)


def test_gbdt_team_weight_touches_only_the_team_head():
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = np.repeat(rng.integers(0, 3, n // 20), 20).astype(np.int8)
    o = rng.integers(0, 2, n).astype(np.int8)
    ok, cols = np.ones(n, bool), np.arange(5)
    a = gbdt.fit_heads(X, o, y, ok, cols, n_trees=5)
    b = gbdt.fit_heads(X, o, y, ok, cols, n_trees=5, team_weight=LW.inv_length(y, 5.0) * (1 + (X[:, 0] > 0)))
    assert a.alive.model_to_string() == b.alive.model_to_string()
    assert a.team.model_to_string() != b.team.model_to_string()
