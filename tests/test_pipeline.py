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


def _train(monkeypatch, work: Path, seed_from=None):
    from katena import pipeline as PL
    _use_work(monkeypatch, work)
    refs = PL.training_refs()
    st = PL.stage_corpus(refs, seed_from)
    worlds = PL.stage_worlds(refs)
    man = PL.stage_train(refs, threads=2, steps=2)
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
    pools = F.FinalPools(refs)
    heads = pools.fit(set(), P.T_MAIN, pools.cols())
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
