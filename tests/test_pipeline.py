"""J2 final pipeline (katena.pipeline) on a tiny synthetic corpus: every stage, the saved model, and a
submission for a synthetic 'real match'. No external data, no data/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synth import make_match

from katena.eval import corpus as C
from katena.eval import protocol as P


@pytest.fixture
def synth_corpus(tmp_path, monkeypatch):
    from katena.models import gbdt
    from katena.types import save_match
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    pytest.importorskip("torch")
    from katena.models import tcn
    monkeypatch.setattr(tcn, "BATCH", 4)
    monkeypatch.setattr(tcn, "WIN", 64)
    monkeypatch.setattr(P, "CANON", tmp_path / "canon")
    for seed, (ds, mid) in enumerate((("metrica", "SYN1"), ("metrica", "SYN0"), ("idsse", "SYN7"),
                                      ("skillcorner", "SYN3"))):
        save_match(make_match(seed=seed, invert_labels=seed == 1), tmp_path / "canon" / ds / f"{mid}.npz")
    return tmp_path


def _use_work(monkeypatch, path: Path):
    from katena.eval import lomo as Lm
    monkeypatch.setattr(P, "WORK", path)
    monkeypatch.setattr(Lm, "OOF_DIR", path / "oof")


def _train(monkeypatch, work: Path, seed_from=None, spec=None):
    from katena import pipeline as PL
    _use_work(monkeypatch, work)
    refs = PL.training_refs()
    st = PL.stage_corpus(refs, seed_from)
    worlds = PL.stage_worlds(refs)
    man = PL.stage_train(refs, threads=2, steps=2, spec=spec)
    return refs, st, worlds, man


def test_pipeline_trains_on_every_match_and_labels_a_new_one(synth_corpus, monkeypatch):
    from katena import pipeline as PL
    from katena.eval import final as F
    refs, st, worlds, man = _train(monkeypatch, synth_corpus / "a")
    assert st == {"copied": 0, "built": 4, "present": 0}
    assert len(refs) == 4 and man["matches"] == [r.key for r in refs]               # nothing held out
    assert {w["key"]: w["worlds"] for w in worlds} == {r.key: len(F.train_worlds(r.dataset)) for r in refs}
    assert man["columns"] > man["rows"] * 0 + 160 and man["w_tcn"] == 0.5 and man["tau"] == 2.0

    model = PL.load_model()
    assert model.names[:model.n_base] == list(C.load_feat(refs[0], "target")["names"])
    # the reloaded LightGBM member predicts exactly what the trained one did
    from katena.models import gbdt
    pools = F.FinalPools(refs)
    heads = gbdt.fit_heads(pools.X, pools.o, pools.y[P.T_MAIN], pools.dead_ok, pools.cols(),
                           n_trees=PL.spec_of().gbdt_trees)                 # the submission spec's size
    X = pools.X[:500]
    a = heads.predict(X, X)
    b = model.heads().predict(X, X)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()

    target = make_match(seed=9)
    target.y = None
    codes, rep = PL.label_match(model, target, threads=2)
    inside = np.zeros(target.n, bool)
    for _, lo, hi in target.half_blocks:
        inside[lo:hi + 1] = True
    assert codes.shape == (target.n,) and (codes[~inside] == 3).all() and (codes[inside] < 3).all()
    assert rep["frames"] == target.n and abs(sum(rep["label_share_in_halves"].values()) - 1) < 1e-3

    sub = PL.write_submission(codes, synth_corpus / "out" / "s.csv", target.n)
    data = Path(sub["path"]).read_bytes()
    assert data.startswith(b"frame,label\n0,X\n") and b"\r" not in data and sub["rows"] == target.n
    with pytest.raises(RuntimeError):
        PL.write_submission(codes, synth_corpus / "out" / "bad.csv", target.n + 1)

    # a second cache, seeded from the first one's corpus files, trains a byte-identical model
    _, st2, _, man2 = _train(monkeypatch, synth_corpus / "b", seed_from=synth_corpus / "a")
    assert st2 == {"copied": 4, "built": 0, "present": 0}
    assert man2["sha256"] == man["sha256"]
    codes2, _ = PL.label_match(PL.load_model(), target, threads=2)
    assert codes2.tobytes() == codes.tobytes()


def test_corpus_refuses_its_own_cache_as_seed(synth_corpus, monkeypatch):
    from katena import pipeline as PL
    _use_work(monkeypatch, synth_corpus / "a")
    with pytest.raises(RuntimeError):
        PL.stage_corpus(PL.training_refs(), synth_corpus / "a")


def test_submission_bytes_are_lf_and_ascii():
    from katena import pipeline as PL
    assert PL.submission_bytes(["X", "H"]) == b"frame,label\n0,X\n1,H\n"


def test_spec_f3_trains_beside_f_with_the_gate_code(synth_corpus, monkeypatch):
    """F3 (more trees, a longer TCN that also reads the touch columns) trains into its own folder, leaves F's model
    untouched, and its held-out TCN equals what the gate's own code (scripts/run_g12.py, ``LT``) trains."""
    import importlib.util
    import os

    from katena import pipeline as PL
    from katena.eval import final as F
    from katena.models import ensemble as E
    refs, _, _, man_f = _train(monkeypatch, synth_corpus / "a", spec="F")
    assert PL.model_dir((), "F").name == "model"
    f_bytes = (PL.model_dir((), "F") / PL.BUNDLE).read_bytes()
    monkeypatch.setitem(PL.SPECS, "F3", PL.Spec("F3", 20, 4800, True))            # 20 trees keep the test short
    key = "idsse/SYN7"
    man = PL.stage_train(refs, threads=2, steps=2, spec="F3", hold_out=[key])
    assert PL.model_dir([key], "F3").name == "model_F3_without_idsse_SYN7"
    assert (PL.model_dir((), "F") / PL.BUNDLE).read_bytes() == f_bytes             # F's model is untouched
    assert man["spec"] == {"name": "F3", "gbdt_trees": 20, "tcn_steps": 2, "tcn_columns": man["columns"]}
    assert man_f["spec"]["name"] == "F" and man_f["spec"]["tcn_columns"] < man_f["columns"]
    assert man["sha256"]["gbdt"] != man_f["sha256"]["gbdt"] and man["sha256"]["tcn"] != man_f["sha256"]["tcn"]

    model = PL.load_model([key], "F3")
    assert model.tcn_n_in == len(model.names) == man["columns"]
    assert PL.load_model((), "F").tcn_n_in == model.n_base
    # the gate's code for the same fold: same weights, same scaler
    # run_g12.py sets environment defaults when imported; keep them out of the tests that follow
    for k, v in {**E.CPU_ENV, "KATENA_WORK": str(synth_corpus / "a")}.items():
        monkeypatch.setenv(k, os.environ.get(k, v))
    spec = importlib.util.spec_from_file_location("run_g12", P.REPO / "scripts" / "run_g12.py")
    G12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G12)
    monkeypatch.setattr(E, "TCN_THREADS", 2)
    pools = F.FinalPools(refs)
    seqs = F.tcn_seqs(refs, P.T_MAIN, 0)
    G12.add_touch(refs, seqs)
    net, sc = G12.fit_tcn_touch(pools, seqs, key, 2)
    assert E.model_hash(net) == man["sha256"]["tcn"]
    assert sc.mu.tobytes() == model.scaler_mu.tobytes() and sc.sd.tobytes() == model.scaler_sd.tobytes()

    target = make_match(seed=9)
    target.y = None
    codes, rep = PL.label_match(model, target, threads=2)
    assert codes.shape == (target.n,) and rep["features"][1] == man["columns"]
    codes_f, _ = PL.label_match(PL.load_model((), "F"), target, threads=2)
    assert codes_f.shape == codes.shape
    with pytest.raises(ValueError):
        PL.model_dir((), "nope")
    model.tcn_n_in = 7
    with pytest.raises(RuntimeError):
        PL.label_match(model, target, threads=2)


def test_specs_a_and_aw_train_on_action_labels(synth_corpus, monkeypatch):
    """A gives the SkillCorner rows their action labels and lets them train the alive head; AW adds the
    possession-length weights to both team heads. No other match's labels change, F3's training is untouched, and
    the LightGBM member is what the gate's call fits on those labels."""
    from katena import pipeline as PL
    from katena.eval import actlabels as AL
    from katena.eval import final as F
    from katena.eval import g1 as G1
    from katena.external import skillcorner_actions as SA
    from katena.models import gbdt

    def fake_actions(match_dir, m):                     # the standing label with H and A swapped
        assert Path(match_dir).name == "SYN3"
        return np.where(m.y == 0, 1, np.where(m.y == 1, 0, m.y)).astype(m.y.dtype)

    monkeypatch.setattr(SA, "build_labels", fake_actions)
    monkeypatch.setattr(P, "EXTERNAL", synth_corpus / "external")
    for name, sp in (("F3", PL.Spec("F3", 20, 4800, True)),
                     ("A", PL.Spec("A", 20, 4800, True, sc_actions=True, sc_dead=True)),
                     ("AW", PL.Spec("AW", 20, 4800, True, sc_actions=True, sc_dead=True, len_weights=True)),
                     ("AT", PL.Spec("AT", 20, 4800, True, sc_actions=True)),
                     ("ATW", PL.Spec("ATW", 20, 4800, True, sc_actions=True, len_weights=True))):
        monkeypatch.setitem(PL.SPECS, name, sp)
    refs, _, _, man3 = _train(monkeypatch, synth_corpus / "a", spec="F3")
    man_a = PL.stage_train(refs, threads=2, steps=2, spec="A")
    man_w = PL.stage_train(refs, threads=2, steps=2, spec="AW")
    assert [PL.model_dir((), s).name for s in ("F3", "A", "AW")] == ["model_F3", "model_A", "model_AW"]
    assert "labels" not in man3 and man_a["labels"]["length_weights"] is False and man_w["labels"]["length_weights"] is True

    pools = F.FinalPools(refs)
    y = AL.pool_y(refs, pools)
    sc_rows = np.char.startswith(pools.keys.astype(str), "skillcorner/")
    changed = y != pools.y[P.T_MAIN]
    assert changed[sc_rows].any() and not changed[~sc_rows].any()
    assert man_a["labels"]["alive_rows"] == int((y < 3).sum()) > int((pools.dead_ok & (pools.y[P.T_MAIN] < 3)).sum())
    assert man_a["labels"]["team_rows"] == int((y < 2).sum())

    # A's LightGBM member is the gate's call on the relabelled rows; AW changes the team head only
    heads = gbdt.fit_heads(pools.X, pools.o, y, np.ones(len(y), bool), pools.cols(), n_trees=20)
    a, w, f3 = (PL.load_model(spec=s) for s in ("A", "AW", "F3"))
    assert a.gbdt_team == heads.team.model_to_string() and a.gbdt_alive == heads.alive.model_to_string()
    assert w.gbdt_alive == a.gbdt_alive and w.gbdt_team != a.gbdt_team
    assert f3.gbdt_team != a.gbdt_team and f3.gbdt_alive != a.gbdt_alive
    assert len({man3["sha256"]["tcn"], man_a["sha256"]["tcn"], man_w["sha256"]["tcn"]}) == 3

    # AT and ATW: A's and AW's team heads with F3's alive head; their TCNs are their own
    man_t = PL.stage_train(refs, threads=2, steps=2, spec="AT")
    man_tw = PL.stage_train(refs, threads=2, steps=2, spec="ATW")
    at, atw = PL.load_model(spec="AT"), PL.load_model(spec="ATW")
    assert at.gbdt_team == a.gbdt_team and at.gbdt_alive == f3.gbdt_alive
    assert atw.gbdt_team == w.gbdt_team and atw.gbdt_alive == f3.gbdt_alive
    assert man_t["labels"]["skillcorner_trains_alive"] is False and man_a["labels"]["skillcorner_trains_alive"] is True
    assert man_t["labels"]["alive_rows"] == int((pools.dead_ok & (pools.y[P.T_MAIN] < 3)).sum())
    assert len({m["sha256"]["tcn"] for m in (man3, man_a, man_w, man_t, man_tw)}) == 5

    # A2 and AW2 are A and AW with twice the TCN steps and the same LightGBM member
    for base, longer in (("A", "A2"), ("AW", "AW2")):
        b, x = PL.spec_of(base), PL.SPECS[longer]
        assert (x.gbdt_trees, x.tcn_steps, x.tcn_touch) == (1200, 9600, True)
        assert (x.sc_actions, x.sc_dead, x.len_weights) == (b.sc_actions, b.sc_dead, b.len_weights)

    # the TCN sequences: SkillCorner ones relabelled and dead_ok, the others as they were
    base, seqs = G1.tcn_seqs(refs, P.T_MAIN, PL.SEED), G1.tcn_seqs(refs, P.T_MAIN, PL.SEED)
    assert AL.relabel_seqs(refs, seqs) == sum(k[0].startswith("skillcorner/") for k in seqs)
    for k, s in seqs.items():
        if k[0].startswith("skillcorner/"):
            assert s["dead_ok"] and not base[k]["dead_ok"] and (s["y"] != base[k]["y"]).any()
        else:
            assert s["dead_ok"] == base[k]["dead_ok"] and np.array_equal(s["y"], base[k]["y"])
    AL.weight_seqs(refs, seqs, AL.native_weights(refs))
    assert all(s["w"].dtype == np.float32 and len(s["w"]) == len(s["y"]) and (s["w"] > 0).all() for s in seqs.values())
