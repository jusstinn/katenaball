"""D3 ensemble: blending, rounding, selection helpers, TCN determinism, and the gate pipeline end to end
on a tiny synthetic corpus (no external data, tiny TCN, few steps)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synth import make_match

from katena.decode import hmm
from katena.models import ensemble as E

REPO = Path(__file__).resolve().parents[1]


# --- blending and rounding --------------------------------------------------------------------

def test_blend_end_points_return_the_members_bit_for_bit():
    rng = np.random.default_rng(0)
    pg = rng.random(500).astype(np.float32)
    pt = rng.random(500).astype(np.float32)
    assert E.blend(pg, pt, 0.0).tobytes() == pg.tobytes()
    assert E.blend(pg, pt, 1.0).tobytes() == pt.tobytes()
    pa, ph = E.blend_heads((pg, pt), (pt, pg), 0.0)
    assert pa.tobytes() == pg.tobytes() and ph.tobytes() == pt.tobytes()


def test_blend_is_a_logit_average():
    p = np.array([0.1, 0.3, 0.5, 0.8, 0.95], np.float32)
    np.testing.assert_allclose(E.blend(p, p, 0.5), p, atol=1e-6)               # agreeing members
    np.testing.assert_allclose(E.blend(p, 1 - p, 0.5), 0.5, atol=1e-6)         # opposite members cancel
    z = E.blend(np.float32([0.9]), np.float32([0.6]), 0.25)[0]
    assert abs(np.log(z / (1 - z)) - (0.75 * np.log(9) + 0.25 * np.log(1.5))) < 1e-5
    lo, mid, hi = (E.blend(np.float32([0.2]), np.float32([0.9]), w)[0] for w in (0.25, 0.5, 0.75))
    assert 0.2 < lo < mid < hi < 0.9


def test_quantize_grids():
    rng = np.random.default_rng(1)
    p = rng.random(1000).astype(np.float32)
    q = E.quantize(p, 8)
    assert np.all(np.abs(q * 256 - np.rint(q * 256)) == 0) and np.abs(q - p).max() <= 1 / 512 + 1e-7
    assert E.quantize(q, 8).tobytes() == q.tobytes()                             # idempotent
    ql = E.quantize_logit(p, 0.25)
    odds = lambda x: np.clip(x, E.EPS, 1 - E.EPS) / (1 - np.clip(x, E.EPS, 1 - E.EPS))   # noqa: E731
    assert np.abs(np.log(odds(ql)) - np.log(odds(p))).max() <= 0.125 + 1e-5


# --- selection helpers ------------------------------------------------------------------------

def test_choose_takes_best_mean_and_first_on_ties():
    grid = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    assert E.choose(np.array([[0.5, 0.5], [0.6, 0.5], [0.7, 0.2]]), grid) == (0.5, 1.0)
    assert E.choose(np.array([[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]]), grid) == (0.0, 1.0)


def test_pairs_cover_each_unordered_pair_once():
    pr = E.pairs(6)
    assert len(pr) == 15 and len(set(pr)) == 15 and all(g < h for g, h in pr)


def test_hash_sees_dtype_shape_and_bytes():
    a = np.arange(6, dtype=np.int8)
    assert E.sha256_arrays(a) == E.sha256_arrays(a.copy())
    assert E.sha256_arrays(a) != E.sha256_arrays(a.astype(np.int16))
    assert E.sha256_arrays(a) != E.sha256_arrays(a.reshape(2, 3))
    assert E.sha256_arrays(a[:3], a[3:]) != E.sha256_arrays(a[:2], a[2:])


def test_blend_at_zero_decodes_exactly_like_the_lightgbm_member():
    y = np.concatenate([np.full(n, k, np.int8) for k, n in ((0, 60), (1, 40), (2, 30), (1, 50), (0, 70))])
    rng = np.random.default_rng(2)
    pa_g = np.clip(np.where(y == 2, 0.1, 0.9) + rng.normal(0, 0.15, len(y)), 0.01, 0.99).astype(np.float32)
    ph_g = np.clip(np.where(y == 0, 0.8, 0.2) + rng.normal(0, 0.3, len(y)), 0.01, 0.99).astype(np.float32)
    pa_t, ph_t = rng.random(len(y)).astype(np.float32), rng.random(len(y)).astype(np.float32)
    pri = hmm.fit_priors([y])
    half = np.ones(len(y), np.int8)
    ref = hmm.decode(pa_g, ph_g, half, pri, 2.0, "viterbi")
    pa, ph = E.blend_heads((pa_g, ph_g), (pa_t, ph_t), 0.0)
    assert hmm.decode(pa, ph, half, pri, 2.0, "viterbi").tobytes() == ref.tobytes()


# --- TCN determinism --------------------------------------------------------------------------

def _tiny_seqs(n_feat: int = 6, n: int = 300, k: int = 3) -> list:
    rng = np.random.default_rng(0)
    return [{"X": rng.normal(size=(n, n_feat)).astype(np.float32), "y": rng.integers(0, 4, n).astype(np.int8),
             "o": i % 2, "dead_ok": i != 1, "key": str(i)} for i in range(k)]


@pytest.fixture
def tiny_tcn(monkeypatch):
    """Shrink the TCN's batch and window so a few steps take milliseconds (the model is unchanged)."""
    pytest.importorskip("torch")
    from katena.models import tcn
    monkeypatch.setattr(tcn, "BATCH", 4)
    monkeypatch.setattr(tcn, "WIN", 64)
    return tcn


def test_tiny_tcn_is_repeatable_in_process_and_seed_sensitive(tiny_tcn):
    tcn = tiny_tcn
    seqs = _tiny_seqs()
    half = np.repeat(np.array([1, 2], np.int8), 150)
    m1, m2 = E.fit_tcn(seqs, seed=0, threads=2, steps=3), E.fit_tcn(seqs, seed=0, threads=2, steps=3)
    assert E.model_hash(m1) == E.model_hash(m2)
    p1 = tcn.predict(m1, seqs[0]["X"], seqs[1]["X"], half)
    p2 = tcn.predict(m2, seqs[0]["X"], seqs[1]["X"], half)
    assert E.sha256_arrays(*p1) == E.sha256_arrays(*p2)
    assert E.model_hash(E.fit_tcn(seqs, seed=1, threads=2, steps=3)) != E.model_hash(m1)


_CHILD = """
import sys
sys.path[:0] = [sys.argv[1]]
from katena.models import ensemble as E
E.pin_cpu_env()
import numpy as np
from katena.models import tcn
tcn.BATCH, tcn.WIN = 4, 64
rng = np.random.default_rng(0)
seqs = [{"X": rng.normal(size=(300, 6)).astype(np.float32), "y": rng.integers(0, 4, 300).astype(np.int8),
         "o": i % 2, "dead_ok": i != 1, "key": str(i)} for i in range(3)]
m = E.fit_tcn(seqs, seed=0, threads=2, steps=3)
pa, ph = tcn.predict(m, seqs[0]["X"], seqs[1]["X"], np.repeat(np.array([1, 2], np.int8), 150))
print(E.model_hash(m), E.sha256_arrays(pa, ph))
"""


def test_tiny_tcn_is_byte_identical_across_processes():
    """Two fresh interpreters (run side by side to halve the wall time: importing torch's deterministic
    switch alone takes seconds) must print the same weight and prediction hashes."""
    pytest.importorskip("torch")
    env = {k: v for k, v in os.environ.items() if k not in E.CPU_ENV}
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, str(REPO / "src")], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for _ in range(2)]
    outs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), outs[0][1][-2000:]
    a, b = (o[0].split() for o in outs)
    assert len(a) == 2 and a == b


# --- the gate pipeline on a synthetic corpus --------------------------------------------------

@pytest.fixture
def synth_corpus(tmp_path, monkeypatch):
    """Three synthetic 'metrica' matches whose ids fall in M6-style groups 0, 1, 2 of three; the bake-off
    cache (features, pools, M2 LOMO probabilities, and M2 scores decoded the bake-off way at tau 2) is built
    under a temporary work dir."""
    from katena.decode import hmm
    from katena.eval import bakeoff as B
    from katena.eval import corpus as C
    from katena.eval import lomo as Lm
    from katena.eval import protocol as P
    from katena.models import gbdt
    from katena.types import save_match
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    monkeypatch.setattr(P, "WORK", tmp_path / "work")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "work" / "oof")
    refs = []
    for seed, mid in ((0, "SYN1"), (1, "SYN0"), (2, "SYN7")):
        path = tmp_path / "canon" / "metrica" / f"{mid}.npz"
        save_match(make_match(seed=seed, invert_labels=seed == 1), path)
        refs.append(P.Ref("metrica", mid, path))
    assert sorted(P.group_of(r.id, 3) for r in refs) == [0, 1, 2]
    for r in refs:
        C.build_one(r)
    pools = Lm.Pools(refs)
    B.stage_lomo(pools, refs, "full", P.T_MAIN)
    base = Lm.Results(tmp_path / "work" / "results.pkl")
    for r in refs:
        oof, pri = Lm.load_oof("full", P.T_MAIN, r), B.priors_for(pools, {r.key}, P.T_MAIN)
        for c in C.conds_for(r):
            f = C.load_feat(r, c)
            rows = hmm.decode(oof[f"pa_{c}"], oof[f"ph_{c}"], f["half"], pri, 2.0, "viterbi")
            base.put(("M2", P.T_MAIN, c, r.key), Lm.score_rows(r, P.T_MAIN, c, rows, f["idx"]))
    return refs, pools, base


def test_gate_pipeline_end_to_end(synth_corpus, tiny_tcn):
    from katena.eval import d3
    from katena.eval import lomo as Lm
    from katena.eval import protocol as P
    refs, pools, base = synth_corpus
    T = P.T_MAIN
    d3.stage_tcn_lomo(pools, refs, refs, T, threads=2, steps=2)
    oof = Lm.load_oof(d3.VAR_TCN, T, refs[0])
    assert {f"{h}_{c}" for h in ("pa", "ph") for c in ("native", "mild", "target", "severe")} <= set(oof)
    res = d3.stage_score(pools, refs, T)
    for r in refs:                       # w = 0 must reproduce M2 exactly, every condition
        for c in ("native", "mild", "target", "severe"):
            assert res.get(("D3M2", T, c, r.key))["spell_f1"] == base.get(("M2", T, c, r.key))["spell_f1"]
    g = d3.gate(base, res, refs, "D3E50")
    assert set(g) == {"target", "severe"} and all(v["n"] == 3 and "adopted" in v for v in g.values())
    same = d3.gate(base, res, refs, "D3M2")
    assert same["target"]["mean_diff"] == 0 and not same["target"]["adopted"]
    # strictly nested choice on groups of three: one inner pair per group pair, then the outer apply
    d3.stage_inner(pools, refs, T, threads=2, n_groups=3, steps=2)
    choice = d3.stage_nested(pools, refs, T, res=res, n_groups=3)
    assert set(choice) == {0, 1, 2} and all(w in E.W_GRID for w in choice.values())
    assert all(res.get(("D3N", T, "target", r.key)) for r in refs)
