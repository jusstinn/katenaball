"""The D2 experiment harness (katena.eval.d2) on synthetic stand-ins: pool layout, worlds per dataset,
the simulated official baseline, and the registered keep rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from synth import make_match

from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import protocol as P
from katena.grid import downsample
from katena.models.baseline import m0_predict
from katena.sim import degrade as dg
from katena.sim import worlds as W


def _ref(dataset: str) -> P.Ref:
    return P.Ref(dataset, "fake", Path("fake.npz"))


def _fake_feats(worlds, n_rows=37, F=3, seed=0):
    rng = np.random.default_rng(seed)
    return {w: {"idx": np.arange(n_rows), "X0": rng.normal(size=(n_rows, F)).astype(np.float32),
                "X1": rng.normal(size=(n_rows, F)).astype(np.float32)} for w in worlds}


def _base_pool(feats, n_rows=37):
    """The pool exactly as katena.eval.corpus.build_one makes it."""
    rng = np.random.default_rng(P.seed_for("fake", "pool"))
    X, o_all, idx = [], [], []
    for c in C.POOL_CONDS:
        sel = np.flatnonzero((np.arange(n_rows) + C.POOL_CONDS.index(c)) % C.POOL_STRIDE == 0)
        o = rng.integers(0, 2, len(sel)).astype(np.int8)
        X.append(np.where(o[:, None] == 1, feats[c]["X1"][sel], feats[c]["X0"][sel]))
        o_all.append(o)
        idx.append(sel)
    return {"X": np.concatenate(X), "o": np.concatenate(o_all), "idx": np.concatenate(idx),
            "y_T2": np.zeros(sum(map(len, idx)), np.int8)}


def test_pool_layout_reproduces_the_bakeoff_pool_and_interleaves_worlds():
    worlds = list(C.POOL_CONDS) + [f"{c}_x" for c in C.POOL_CONDS]
    feats = _fake_feats(worlds)
    base = _base_pool(feats)
    load = lambda ref, w: feats[w]                                     # noqa: E731
    same = d2.assemble_pool(_ref("idsse"), {c: [c] for c in C.POOL_CONDS}, base, load=load)
    assert same["X"].tobytes() == base["X"].tobytes()
    assert (same["o"] == base["o"]).all() and (same["idx"] == base["idx"]).all()
    mixed = d2.assemble_pool(_ref("idsse"), {c: [c, f"{c}_x"] for c in C.POOL_CONDS}, base, load=load)
    off = 0
    for c, sel in d2.pool_rows(37).items():
        for j, row in enumerate(sel):
            w = c if j % 2 == 0 else f"{c}_x"
            want = feats[w]["X1" if base["o"][off + j] else "X0"][row]
            np.testing.assert_array_equal(mixed["X"][off + j], want)
        off += len(sel)
    assert (mixed["y_T2"] == base["y_T2"]).all()


def test_worlds_per_dataset():
    ids = d2.new_worlds(_ref("idsse"))
    sc = d2.new_worlds(_ref("skillcorner"))
    assert "target_cpl100" in ids and "severe_cpl50" in ids and "target_ids" in ids
    assert all("_cpl" not in w for w in sc) and set(sc) == {"mild_ids", "target_ids", "severe_ids"}
    assert not set(ids) & set(dg.CONFIGS)
    assert all(W.applies(w, "idsse") for w in d2.eval_worlds(_ref("metrica")))
    assert set(d2.eval_worlds(_ref("skillcorner"))) == {"mild", "target", "severe", "target_ids", "mild_ids", "severe_ids"}
    for name in W.WORLDS:                                       # new names never shadow cached conditions
        assert name not in C.CONFIG and name not in C.BASE_CONDS


def test_official_baseline_windows():
    m5 = dg.corrupt(downsample(make_match()), dg.TARGET, 1)
    mw = d2.m0_windows(m5)
    np.testing.assert_array_equal(mw[list(d2.M0_WINDOWS).index("smooth50")], m0_predict(m5))
    inside = mw[:, m5.y < 3]
    assert (inside < 3).all() and ((mw[0] == 2) >= (mw[3] == 2))[m5.y < 3].all()   # wider window, fewer D


def _fake_cache(tmp_path, monkeypatch, n5=600, keys=("a", "b", "c")):
    """A tiny bake-off cache: three IDSSE-like matches with ground truth, world rows and OOF probabilities.
    ``d2`` gets probabilities close to the truth, ``cur`` noisy ones."""
    monkeypatch.setattr(P, "WORK", tmp_path)
    rng = np.random.default_rng(0)
    refs = [P.Ref("idsse", k, tmp_path / f"{k}.npz") for k in keys]
    for r in refs:
        y5 = np.full(n5, 3, np.int8)
        blocks5 = [(1, 10, 289), (2, 310, 589)]
        for _, a, b in blocks5:
            seg = np.repeat(rng.choice([0, 1, 2], size=(b - a + 1) // 20 + 1), 20)[:b - a + 1]
            y5[a:b + 1] = seg
        gt = C.paths(r)["gt"]
        gt.parent.mkdir(parents=True, exist_ok=True)
        y_nat = np.repeat(y5, 5)
        np.savez(gt, **{f"y_T{t}": y_nat for t in P.T_SWEEP}, step=5, fps=25.0, n_native=len(y_nat),
                 blocks=np.array([(h, a * 5, b * 5 + 4) for h, a, b in blocks5]))
        idx = np.concatenate([np.arange(a, b + 1) for _, a, b in blocks5])
        half = np.concatenate([np.full(b - a + 1, h, np.int8) for h, a, b in blocks5])
        yr = y5[idx]
        for w in d2.eval_worlds(r):
            f = C.paths(r)["feat"](w)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, idx=idx, half=half)
        for v, noise in (("cur", 0.45), ("d2", 0.05)):
            arrays = {}
            for w in d2.eval_worlds(r):
                flip = rng.random(len(yr)) < noise
                arrays[f"pa_{w}"] = np.where(yr == 2, 0.1, 0.9).astype(np.float32)
                team = np.where(flip, 1 - np.minimum(yr, 1), np.minimum(yr, 1))
                arrays[f"ph_{w}"] = np.where(team == 0, 0.9, 0.1).astype(np.float32)
            p = d2.oof_path(v, r)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **arrays)
    return refs


def test_score_stage_end_to_end_on_a_fake_cache(tmp_path, monkeypatch):
    refs = _fake_cache(tmp_path, monkeypatch)
    res = d2.score_folds(refs, refs)
    for r in refs:
        for w in d2.eval_worlds(r):
            assert res[("d2", w, r.key)]["spell_f1"] >= res[("cur", w, r.key)]["spell_f1"]
        assert res[("tau", "d2", r.key)] in P.TAUS
    cmp = d2.compare_worlds(res, refs)
    assert set(cmp) == set(W.EVAL_WORLDS) | set(d2.SECONDARY) | set(d2.SECONDARY_IDSSE)
    assert all(c["n"] == 3 and c["mean_diff"] > 0 for c in cmp.values())
    v = d2.keep_verdict(cmp)
    assert v["keep"] and not v["worse"] and not v["missing"]


def _cmp(mean_diff, lo, hi, win_rate=0.5):
    return {"mean_diff": mean_diff, "lo": lo, "hi": hi, "win_rate": win_rate, "n": 8,
            "worse": mean_diff < 0 and hi < 0, "better": win_rate >= 0.7 and lo > 0}


def test_keep_rule():
    ok = {w: _cmp(0.001, -0.002, 0.004) for w in W.EVAL_WORLDS}
    assert d2.keep_verdict(ok)["keep"]
    bad = dict(ok, target=_cmp(-0.004, -0.007, -0.001))
    v = d2.keep_verdict(bad)
    assert not v["keep"] and v["worse"] == ["target"]
    sec = dict(ok, severe_ids=_cmp(-0.01, -0.02, -0.005))        # secondary worlds do not decide
    assert d2.keep_verdict(sec)["keep"]
    good = dict(ok, target_cpl100=_cmp(0.01, 0.004, 0.016, 0.875))
    assert d2.keep_verdict(good)["better"] == ["target_cpl100"]
