"""Gate G1 (katena.eval.g1): the byte-checked seeding, the 213-column sequences against F's, the prefitted
scaler against ``final.fit_tcn``, and the whole G1 path on a tiny synthetic corpus (no external data)."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import test_final as TFN

from katena.eval import corpus as C
from katena.eval import final as F
from katena.eval import g1 as G
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.models import ensemble as E
from katena.sim import worlds as W

synth_final = TFN.synth_final          # the F fixture: three synthetic metrica matches with every cache
tiny_tcn = TFN.tiny_tcn


def _ref(dataset: str, mid: str) -> P.Ref:
    return P.Ref(dataset, mid, Path(f"{mid}.npz"))


# --- seeding ------------------------------------------------------------------------------------

def test_seed_copies_byte_identical_and_refuses_bad_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "WORK", tmp_path / "work")
    monkeypatch.setattr(P, "CANON", tmp_path / "canon")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "work" / "oof")
    refs = [_ref("skillcorner", "123"), _ref("idsse", "DFL-X")]
    src_root = tmp_path / "final"
    plan = G.seed_plan(refs, src_root)
    assert len(plan) == (3 + 2 * 9 + 2) + (3 + 2 * 16 + 2) + 1                  # 9 and 16 feature worlds
    for src, dst in plan:
        assert src.relative_to(src_root) == dst.relative_to(P.WORK)
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(str(src.relative_to(src_root)).encode())
    info = G.seed_cache(refs, src_root)
    assert info["planned"] == info["copied"] == info["identical"] == len(plan)
    assert all(dst.read_bytes() == src.read_bytes() for src, dst in plan)
    assert G.seed_cache(refs, src_root)["copied"] == 0                            # nothing twice, still checked
    _, dst = plan[5]
    dst.write_bytes(bytes(reversed(dst.read_bytes())))                             # same size, other bytes
    with pytest.raises(RuntimeError, match="differ"):
        G.seed_cache(refs, src_root)
    with pytest.raises(RuntimeError):
        G.seed_cache(refs, P.WORK)


# --- the scaler and the fit ---------------------------------------------------------------------

def test_prefitted_scaler_trains_the_same_tcn_as_final_fit_tcn(tiny_tcn):
    rng = np.random.default_rng(1)
    seqs = {}
    for i in range(4):
        X = rng.normal(1.0, 3.0, size=(300, 9)).astype(np.float16)
        X[rng.random(X.shape) < 0.05] = np.nan
        seqs[(f"k{i % 2}", i)] = {"X": X, "y": rng.integers(0, 4, 300).astype(np.int8), "o": i % 2,
                                  "dead_ok": i != 1, "key": f"k{i % 2}"}
    pool = G._Rows(np.repeat(["k0", "k1"], 120), rng.normal(1.0, 3.0, size=(240, 9)).astype(np.float32))
    sc = G.scaler(pool, {"k0"})
    mine, _ = G.fit_tcn(sc, seqs, {"k0"}, threads=2, steps=3)
    theirs, sc_f = F.fit_tcn(pool, seqs, {"k0"}, threads=2, steps=3)
    assert sc.mu.tobytes() == sc_f.mu.tobytes() and sc.sd.tobytes() == sc_f.sd.tobytes()
    assert E.model_hash(mine) == E.model_hash(theirs) and mine.inp.in_channels == 9
    # F's member scales only a FinalPools' first n_base columns; G1's scaler takes every column
    pool.n_base = 5
    assert len(G.scaler(pool, {"k0"}).mu) == 9


# --- on the synthetic corpus --------------------------------------------------------------------

def _f_cache(refs: list, steps: int = 2) -> None:
    """F's run on the synthetic corpus, into the current P.WORK: touch, pools, both members, F's scores."""
    for r in refs:
        F.build_touch(r)
    F.build_pools(refs)
    F.stage_gbdt(F.FinalPools(refs), refs)
    F.stage_tcn(F.FinalPools(refs, touch=False), refs, refs, threads=2, steps=steps)
    F.stage_score(F.results(), refs, refs, weights=(F.W_F, 0.0, 1.0))


def test_g1_sequences_are_f_sequences_with_the_touch_columns(synth_final):
    refs = synth_final
    for r in refs:
        F.build_touch(r)
    mine, theirs = G.tcn_seqs(refs, P.T_MAIN), F.tcn_seqs(refs, P.T_MAIN)
    assert list(mine) == list(theirs) and len(mine) == 15 * len(refs)
    for (key, w), s in theirs.items():
        m = mine[(key, w)]
        n = s["X"].shape[1]
        assert m["o"] == s["o"] and m["dead_ok"] == s["dead_ok"] and np.array_equal(m["y"], s["y"])
        assert m["X"].dtype == np.float16 and m["X"].shape == (len(s["X"]), n + 53)
        assert m["X"][:, :n].tobytes() == s["X"].tobytes()
        r = next(x for x in refs if x.key == key)
        A = TG.load_tfeat(r, w)["A1" if s["o"] else "A0"].astype(np.float16)
        assert m["X"][:, n:].tobytes() == A.tobytes()
    assert {m["o"] for m in mine.values()} == {0, 1}


def test_g1_pipeline_end_to_end(synth_final, tiny_tcn, tmp_path, monkeypatch):
    refs = synth_final
    T = P.T_MAIN
    _f_cache(refs)
    f_root = tmp_path / "f_cache"                       # the F cache, read only from here on
    shutil.copytree(P.WORK, f_root)
    monkeypatch.setattr(P, "WORK", tmp_path / "g1")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "g1" / "oof")
    info = G.seed_cache(refs, f_root)
    assert info["planned"] == info["identical"] == info["copied"] == 3 * (3 + 2 * 16 + 2) + 1
    G.stage_tcn(refs, refs, threads=2, steps=2)
    n_base = C.load_feat(refs[0], "target")["X0"].shape[1]
    for r in refs:
        oof = Lm.load_oof(G.VAR_T, T, r)
        assert int(oof["n_in"]) == n_base + 53 and {f"pa_{c}" for c in W.EVAL_WORLDS} <= set(oof)
        assert str(oof["model_sha"]) != str(Lm.load_oof(F.VAR_T, T, r)["model_sha"])
    assert G.stage_tcn(refs, refs, threads=2, steps=2) == {"folds": 0}                # cached
    store = G.f_store()
    res = G.results()
    chk = G.stage_score(res, refs, refs, store)
    n = 3 * len(W.EVAL_WORLDS)
    assert chk == {"n": n, "F_equal": n, "FG_equal": n, "FT_equal": n}             # F re-decoded = F stored
    get = F.getter(res, store)
    for r in refs:
        for c in W.EVAL_WORLDS:
            assert all(get(m, c, r) is not None for m in ("G1", "G1T", "F", "FG", "FT"))
    g = F.compare(get, refs, "G1", "F", "target")
    assert g["n"] == 3 and {"adopted", "worse", "wins", "cand_mean", "inc_mean"} <= set(g)
    same = F.compare(get, refs, "F", "F", "severe")
    assert same["mean_diff"] == 0 and same["wins"] == 0 and not same["adopted"] and not same["worse"]
    # the report script on the synthetic results (its import pins CPU_ENV: restored afterwards)
    for k in E.CPU_ENV:
        old = os.environ.get(k)
        monkeypatch.setenv(k, old if old is not None else "")
        if old is None:
            monkeypatch.delenv(k)
    spec = importlib.util.spec_from_file_location("run_g1", Path(__file__).resolve().parents[1] / "scripts" / "run_g1.py")
    run_g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_g1)
    out = run_g1.gate_report(refs)
    assert out["g1_passes"] == F.decision(out["gate"]["28/target"], out["gate"]["28/severe"])
    assert out["gate"]["28/target"]["n"] == 3 and (P.WORK / "g1_gate.json").exists()
    assert {"world/target/G1", "world/target/G1T", "G1T-FT/28/target"} <= set(out["info"])
