import itertools

import numpy as np
import pytest

from katena.match import A, D, H, X
from katena.model.decode import (argmax_frames, decode_simple, segment_viterbi, upsample_logp)


def _score(E, lab, pen):
    return E[np.arange(len(lab)), lab].sum() - pen * np.count_nonzero(np.diff(lab))


def _runs_ok(lab, d):
    edges = np.flatnonzero(np.diff(lab)) + 1
    bounds = np.r_[0, edges, len(lab)]
    return np.all(np.diff(bounds) >= d)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_segment_viterbi_matches_brute_force(d):
    rng = np.random.default_rng(d)
    T, K = 7, 3
    pens = [0.0, 0.7, 3.0]
    for trial in range(6):
        E = np.log(rng.dirichlet(np.ones(K) * 0.5, size=T))
        out = segment_viterbi(E, pens, d)
        assert out.shape == (len(pens), T) and out.dtype == np.int8
        for p, pen in enumerate(pens):
            best = max(_score(E, np.array(s), pen) for s in itertools.product(range(K), repeat=T)
                       if _runs_ok(np.array(s), d))
            assert _runs_ok(out[p], d)
            assert _score(E, out[p].astype(int), pen) == pytest.approx(best, abs=1e-9)


def test_min_dwell_removes_blips_and_keeps_real_changes():
    T = 400
    p = np.full((T, 3), 0.05)
    p[:, H] = 0.9
    p[200:, :] = 0.05
    p[200:, A] = 0.9                     # a real change at 200
    p[100:105, :] = 0.05
    p[100:105, D] = 0.9                  # a 5-frame blip of D
    E = np.log(p / p.sum(axis=1, keepdims=True))
    lab = segment_viterbi(E, [1.0], 25)[0]
    assert (lab[:200] == H).all() and (lab[200:] == A).all()
    # with no dwell and no penalty the blip survives
    lab0 = segment_viterbi(E, [0.0], 1)[0]
    assert (lab0[100:105] == D).all()
    # a huge penalty forbids any change
    lab_big = segment_viterbi(E, [1e6], 1)[0]
    assert len(np.unique(lab_big)) == 1


def test_upsample_linear_is_normalised_and_exact_at_rows():
    rng = np.random.default_rng(0)
    blocks = [(1, 10, 109), (2, 130, 229)]
    t = np.concatenate([np.arange(10, 110, 5), np.arange(130, 230, 5)])
    lp = np.log(rng.dirichlet(np.ones(3), size=len(t))).astype(np.float32)
    up = upsample_logp(lp, t, blocks, 240)
    assert up.shape == (240, 3)
    assert np.isnan(up[:10]).all() and np.isnan(up[110:130]).all() and np.isnan(up[230:]).all()
    inside = np.isfinite(up[:, 0])
    assert np.allclose(np.exp(up[inside]).sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(up[t], lp, atol=1e-5)
    near = upsample_logp(lp, t, blocks, 240, mode="nearest")
    assert np.allclose(near[t], lp, atol=1e-5)
    assert np.allclose(near[11], lp[0], atol=1e-5) and np.allclose(near[13], lp[1], atol=1e-5)


def test_decode_simple_per_half_and_x_outside():
    blocks = [(1, 5, 404), (2, 420, 819)]
    n = 830
    t = np.concatenate([np.arange(5, 405, 5), np.arange(420, 820, 5)])
    p = np.full((len(t), 3), 0.05)
    first = t < 405
    p[first, H] = 0.9
    p[~first, A] = 0.9
    p[(t >= 600) & (t < 700), :] = 0.05
    p[(t >= 600) & (t < 700), D] = 0.9
    lp = np.log(p / p.sum(axis=1, keepdims=True)).astype(np.float32)
    out = decode_simple(lp, t, blocks, n, {"switch_penalty": 4.0, "min_dwell_s": 1.0})
    assert out.dtype == np.int8 and out.shape == (n,)
    assert (out[:5] == X).all() and (out[405:420] == X).all() and (out[820:] == X).all()
    assert (out[5:405] == H).all()
    assert (out[420:598] == A).all() and (out[602:698] == D).all() and (out[702:820] == A).all()
    am = argmax_frames(lp, t, blocks, n)
    assert (am[5:405] == H).all() and (am[:5] == X).all()


def test_decoder_is_deterministic():
    rng = np.random.default_rng(5)
    blocks = [(1, 0, 999)]
    t = np.arange(0, 1000, 5)
    lp = np.log(rng.dirichlet(np.ones(3), size=len(t))).astype(np.float32)
    a = decode_simple(lp, t, blocks, 1000, {"switch_penalty": 2.0, "min_dwell_s": 0.5})
    b = decode_simple(lp.copy(), t.copy(), blocks, 1000, {"switch_penalty": 2.0, "min_dwell_s": 0.5})
    assert np.array_equal(a, b)
