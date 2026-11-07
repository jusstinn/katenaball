"""Final candidate F (katena.eval.final): worlds, cache seeding, lazy TCN scaling, and the whole pipeline on a
tiny synthetic corpus (no external data, tiny TCN, few steps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synth import make_match

from katena.eval import corpus as C
from katena.eval import final as F
from katena.eval import protocol as P
from katena.models import ensemble as E
from katena.sim import worlds as W


def _ref(dataset: str, mid: str = "fake") -> P.Ref:
    return P.Ref(dataset, mid, Path(f"{mid}.npz"))


@pytest.fixture
def tiny_tcn(monkeypatch):
    """Shrink the TCN's batch and window so a few steps take milliseconds (the model is unchanged)."""
    pytest.importorskip("torch")
    from katena.models import tcn
    monkeypatch.setattr(tcn, "BATCH", 4)
    monkeypatch.setattr(tcn, "WIN", 64)
    return tcn


# --- worlds -----------------------------------------------------------------------------------

def test_training_and_evaluation_worlds():
    ids, sc = F.train_worlds("idsse"), F.train_worlds("skillcorner")
    assert len(ids) == 15 and len(sc) == 9 and F.train_worlds("metrica") == ids
    assert ids[0] == "mild" and ids[5] == "target" and ids[10] == "severe"          # severity blocks, mix order
    assert sc == ["mild", "mild_ids", "mild_smooth", "target", "target_ids", "target_smooth",
                  "severe", "severe_ids", "severe_smooth"]
    assert F.eval_worlds(_ref("idsse")) == list(W.EVAL_WORLDS)
    assert F.eval_worlds(_ref("skillcorner")) == ["mild", "target", "severe", "target_ids", "target_smooth"]
    assert F.feature_worlds(_ref("idsse")) == ids + ["target_cpl100_ids"]          # the one eval-only world
    assert F.feature_worlds(_ref("skillcorner")) == sc
    assert set(F.GATE_CONDS) <= set(sc)


def test_decision_rule():
    yes, no = {"adopted": True, "worse": False}, {"adopted": False, "worse": True}
    assert F.decision(yes, {"worse": False}) and F.decision(yes, {"worse": False, "adopted": False})
    assert not F.decision(yes, no) and not F.decision(no, {"worse": False})


def test_peak_memory_is_reported():
    assert 0.0 < F.peak_rss_gb() < 1024


def test_seed_copies_from_the_sources_and_refuses_its_own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "WORK", tmp_path / "work")
    monkeypatch.setattr(P, "CANON", tmp_path / "canon")
    ref = _ref("skillcorner", "123")
    base, d2r = tmp_path / "base", tmp_path / "d2"
    for src, _ in F.seed_plan([ref], base, d2r):
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(src.name.encode())
    info = F.seed_cache([ref], base, d2r)
    assert info["copied"] == info["planned"] == 2 + 9
    for src, dst in F.seed_plan([ref], base, d2r):
        assert dst.read_bytes() == src.read_bytes()
        assert (src.parent.parent == base) == (src.stem.endswith(("__mild", "__target", "__severe"))
                                               or src.parent.name in ("gt", "pool"))
    assert F.seed_cache([ref], base, d2r)["copied"] == 0                          # nothing twice
    with pytest.raises(RuntimeError):
        F.seed_cache([ref], P.WORK, d2r)


# --- lazy scaling -----------------------------------------------------------------------------

class _Pool:
    def __init__(self, keys, X):
        self.keys, self.X = keys, X


def test_scaled_sequences_train_the_same_tcn_as_eager_scaling(tiny_tcn):
    rng = np.random.default_rng(0)
    seqs = {}
    for i in range(4):
        X = rng.normal(3.0, 2.0, size=(300, 6)).astype(np.float16)
        X[rng.random(X.shape) < 0.05] = np.nan
        seqs[(f"k{i % 2}", i)] = {"X": X, "y": rng.integers(0, 4, 300).astype(np.int8), "o": i % 2,
                                  "dead_ok": i != 1, "key": f"k{i % 2}"}
    pool = _Pool(np.repeat(["k0", "k1"], 120), rng.normal(3.0, 2.0, size=(240, 6)).astype(np.float32))
    lazy, sc = F.fit_tcn(pool, seqs, {"k0"}, threads=2, steps=3, lazy=True)
    eager, _ = F.fit_tcn(pool, seqs, {"k0"}, threads=2, steps=3, lazy=False)
    assert E.model_hash(lazy) == E.model_hash(eager)
    s = next(iter(seqs.values()))["X"]
    v = F.ScaledSeq(s, sc)
    assert v.shape == s.shape and len(v) == len(s)
    assert v[17:250].tobytes() == sc(s.astype(np.float32))[17:250].tobytes()


# --- the pipeline on a synthetic corpus -------------------------------------------------------

@pytest.fixture
def synth_final(tmp_path, monkeypatch):
    """Three synthetic 'metrica' matches with the bake-off cache, D2's worlds, M2 LOMO probabilities, M2
    scores decoded at tau 2, and D2's ``cur`` probabilities, all under one temporary work dir."""
    from katena.decode import hmm
    from katena.eval import bakeoff as B
    from katena.eval import d2
    from katena.eval import lomo as Lm
    from katena.models import gbdt
    from katena.types import save_match
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    monkeypatch.setattr(P, "WORK", tmp_path / "work")
    monkeypatch.setattr(P, "CANON", tmp_path / "canon")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "work" / "oof")
    refs = []
    for seed, mid in ((0, "SYN1"), (1, "SYN0"), (2, "SYN7")):
        path = tmp_path / "canon" / "metrica" / f"{mid}.npz"
        save_match(make_match(seed=seed, invert_labels=seed == 1), path)
        refs.append(P.Ref("metrica", mid, path))
    for r in refs:
        C.build_one(r)
        d2.build_ref(r, [w for w in d2.new_worlds(r) if w not in C.CONFIG])
    pools = Lm.Pools(refs)
    B.stage_lomo(pools, refs, "full", P.T_MAIN)
    base = Lm.Results(tmp_path / "work" / "results.pkl")
    for r in refs:
        oof, pri = Lm.load_oof("full", P.T_MAIN, r), B.priors_for(pools, {r.key}, P.T_MAIN)
        for c in C.conds_for(r):
            f = C.load_feat(r, c)
            rows = hmm.decode(oof[f"pa_{c}"], oof[f"ph_{c}"], f["half"], pri, 2.0, "viterbi")
            base.put(("M2", P.T_MAIN, c, r.key), Lm.score_rows(r, P.T_MAIN, c, rows, f["idx"]))
    base.save()
    d2.run_lomo(refs, refs, variants=("cur",))
    return refs


def test_tcn_sequences_reduce_to_the_bakeoff_ones_for_the_current_mix(synth_final, monkeypatch):
    from katena.eval import stages as S
    refs = synth_final
    one = {s: [s] for s in C.POOL_CONDS}
    monkeypatch.setitem(W.TRAIN_MIXES, "cur", {"dead_labels": one, "no_dead_labels": one})
    mine = F.tcn_seqs(refs, P.T_MAIN, mix="cur")
    theirs = S._tcn_seqs(refs, P.T_MAIN, np.random.default_rng(P.SEED))
    assert list(mine) == list(theirs)
    for k in theirs:
        assert mine[k]["o"] == theirs[k]["o"] and mine[k]["X"].tobytes() == theirs[k]["X"].tobytes()
        assert np.array_equal(mine[k]["y"], theirs[k]["y"]) and mine[k]["dead_ok"] == theirs[k]["dead_ok"]
    d2j = F.tcn_seqs(refs, P.T_MAIN)
    assert len(d2j) == 15 * len(refs) and list(d2j)[:15] == [(refs[0].key, w) for w in F.train_worlds("metrica")]


def test_final_pipeline_end_to_end(synth_final, tiny_tcn):
    from katena.eval import d2
    from katena.eval import lomo as Lm
    from katena.eval import touch_gate as TG
    refs = synth_final
    T = P.T_MAIN
    for i, r in enumerate(refs):                       # every world recovered; the first one re-featurized
        st = F.build_touch(r, refeaturize=i == 0)
        assert st["recovered"] == st["worlds"] == 16 and not st["rebuilt"] and st["built"] == 16
        assert st["refeaturize_same"] == st["refeaturized"] == (16 if i == 0 else 0)
    assert F.build_touch(refs[1])["built"] == 0                                     # cached, keyed on touch.py
    info = F.build_pools(refs, other_root=P.WORK)
    assert info["identical"] == info["compared"] == 3
    pools = F.FinalPools(refs)
    assert pools.X.shape[1] == pools.n_base + int((pools.groups == TG.TF.GROUP).sum()) and pools.n_base > 50
    # a touch row comes from the same world, orientation and sample as its base row
    r = refs[2]
    base, rows = C.load_pool(r), d2.pool_rows(len(C.load_feat(r, "mild")["idx"]))
    off = len(rows["mild"]) + 1                        # second row of the target block: second world of the mix
    w = F.train_worlds("metrica")[5 + 1]
    t = TG.load_tfeat(r, w)
    j = rows["target"][1]
    want = t["A1" if base["o"][off] else "A0"][j]
    got = pools.X[pools.keys == r.key][off, pools.n_base:]
    assert np.array_equal(got, want, equal_nan=True)
    F.stage_gbdt(pools, refs)
    F.stage_tcn(F.FinalPools(refs, touch=False), refs, refs, threads=2, steps=2)
    for v in (F.VAR_G, F.VAR_T):
        oof = Lm.load_oof(v, T, refs[0])
        assert {f"pa_{c}" for c in W.EVAL_WORLDS} <= set(oof) and "model_sha" in oof
    res = F.results()
    chk = F.stage_incumbent(res, refs, refs, P.WORK, P.WORK)
    assert chk["m2_t2_equal"] == chk["m2w_equal"] == chk["cur_probs_equal"] == chk["n"] == 9
    F.stage_score(res, refs, refs)
    base_store = F.load_store(P.WORK / "results.pkl")
    get = F.getter(res, base_store)
    for r in refs:
        for c in W.EVAL_WORLDS:
            assert all(get(m, c, r) is not None for m in ("F", "FG", "FT", "M2w"))
    g = F.compare(get, refs, "F", "M2", "target")
    assert g["n"] == 3 and {"adopted", "worse", "wins", "cand_mean", "inc_mean"} <= set(g)
    same = F.compare(get, refs, "M2_t2", "M2", "severe")
    assert same["mean_diff"] == 0 and same["wins"] == 0 and not same["adopted"] and not same["worse"]
    assert F.costs() == []                                                           # stages log only via Cost
    F.Cost("x").done(n=1)
    assert F.costs()[-1]["stage"] == "x" and F.costs()[-1]["n"] == 1
