"""katena.pipeline through the competition's file format, without any match data.

* A synthetic ATD bundle (EPTS metadata XML and raw text, ``katena.target.dress``) on the competition's own timeline
  (199,319 frames, games.py's graded halves, 105.3 x 68 m) goes through ``stage_predict``, which reads it with
  ``katena.target.atd.load_target``. The CSV must pass ``evaluation/validate_submission.py`` unchanged.
* The exporter is the exact inverse of ``load_target``.
* The dressed-match check in small: a synthetic external match, exported, labelled by the submission path with a
  model that held it out, gives the evaluation path's labels and score.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synth import make_match

from katena.eval import protocol as P
from katena.target import dress
from katena.target.atd import Timeline, load_target, official_timeline
from katena.types import Match

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                          # games.py


def synth_target(seed: int = 0, S: int = 6) -> Match:
    """A small-squad synthetic match on the competition timeline: players drift around a formation, home attacks +x
    in the first half and -x in the second, the ball follows the team in possession (cycles of H, A and a dead
    ball) and is missing on a third of the frames. Nothing outside the halves."""
    tl = official_timeline()
    n, (L, W) = tl.n_frames, tl.pitch
    rng = np.random.default_rng(seed)
    team = np.repeat(np.array([0, 1], dtype=np.int8), S // 2)
    Pp = np.full((n, S, 2), np.nan, dtype=np.float32)
    speed = np.full((n, S), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    base = np.stack([np.linspace(20, 45, S // 2), np.linspace(12, W - 12, S // 2)], 1)
    for half, a, b in tl.half_blocks:
        m = b - a + 1
        t = np.arange(m) / 25.0
        lab = ((np.arange(m) // 200) % 3).astype(np.int8)
        for s in range(S):
            bx, by = base[s % (S // 2)]
            x = bx if s < S // 2 else L - bx
            Pp[a:b + 1, s, 0] = (x if half == 1 else L - x) + 4 * np.sin(0.05 * t + s) + rng.normal(0, 0.05, m)
            Pp[a:b + 1, s, 1] = by + 3 * np.cos(0.04 * t + 2 * s) + rng.normal(0, 0.05, m)
        speed[a:b + 1] = 1.5
        ball[a:b + 1] = Pp[a + np.arange(m), np.where(lab == 0, 0, S // 2)] + 0.4
        ball[a:b + 1][lab == 2] = (L / 2, -0.5)
        ball[a:b + 1][rng.random(m) < 0.33] = np.nan
    return Match(P=Pp, team=team, speed=speed, ball=ball, half_blocks=[tuple(b) for b in tl.half_blocks],
                 attack_dir={1: 1, 2: -1}, y=None, pitch=tl.pitch, fps=tl.fps, name="synthetic_target",
                 source="synthetic")


def four_decimals(v: float) -> str:
    return f"{v:.4f}"


# --- the reader and the exporter ----------------------------------------------------------------

def test_official_timeline_is_the_published_one():
    tl = official_timeline()
    assert tl.n_frames == 199319 and tl.fps == 25.0 and tl.pitch == (105.3, 68.0)
    assert tl.half_blocks == ((1, 11185, 90651), (2, 110808, 190197))
    assert Timeline.from_json(json.loads(json.dumps(tl.to_json()))) == tl


def test_exported_bundle_reads_back_exactly(tmp_path):
    from katena.target.layout import assumptions_hold, layout_report
    m = make_match(seed=3)
    m.team[[2, 15]] = m.team[[15, 2]]                     # slots of the two teams interleaved
    m.ball[205:209, 0] = np.nan                           # x missing while y is present
    tl = dress.write_bundle(m, tmp_path / "bundle")
    back = load_target(tmp_path / "bundle", timeline=tl)
    same = dress.compare(m, back)
    assert all(same.values()), same
    paths = dress.bundle_paths(tmp_path / "bundle")
    rep = layout_report(paths["atd_meta"], n_frames=m.n)
    assert assumptions_hold(rep) and rep["specs_cover_1_to_n"]
    raw = paths["atd_raw"].read_bytes()
    lines = raw.split(b"\n")
    assert lines[0].startswith(b"1:") and lines[-2].startswith(f"{m.n}:".encode()) and lines[-1] == b""
    assert b"\r" not in raw and back.meta["read_report"]["contiguous_from_one"]


def test_exporter_refuses_the_competition_data_dir():
    import games
    with pytest.raises(ValueError):
        dress.write_bundle(make_match(seed=0), Path(games.DATA_DIR))


# --- the pipeline -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Justin's synthetic corpus (tests/test_pipeline.py), every stage, the submission model and one held out."""
    from katena.models import gbdt
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load libomp in this process")
    pytest.importorskip("torch")
    from katena import pipeline as PL
    from katena.eval import lomo as Lm
    from katena.models import tcn
    from katena.types import save_match
    tmp = tmp_path_factory.mktemp("pipeline_atd")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tcn, "BATCH", 4)
        mp.setattr(tcn, "WIN", 64)
        mp.setattr(P, "CANON", tmp / "canon")
        mp.setattr(P, "WORK", tmp / "work")
        mp.setattr(Lm, "OOF_DIR", tmp / "work" / "oof")
        for seed, (ds, mid) in enumerate((("metrica", "SYN1"), ("metrica", "SYN0"), ("idsse", "SYN7"),
                                          ("skillcorner", "SYN3"))):
            save_match(make_match(seed=seed, invert_labels=seed == 1), tmp / "canon" / ds / f"{mid}.npz")
        refs = PL.training_refs()
        PL.stage_corpus(refs)
        PL.stage_worlds(refs)
        full = PL.stage_train(refs, threads=2, steps=2)
        full_sha = (PL.model_dir() / PL.BUNDLE).read_bytes()
        held = PL.stage_train(refs, threads=2, steps=2, hold_out=["idsse/SYN7"])
        yield {"tmp": tmp, "refs": refs, "full": full, "held": held, "full_bytes": full_sha}


def test_predict_reads_an_atd_bundle_on_the_competition_timeline(trained):
    from katena import pipeline as PL
    tmp = trained["tmp"]
    dress.write_bundle(synth_target(), tmp / "bundle", fmt=four_decimals)
    out = tmp / "out" / "synthetic.csv"
    rep = PL.stage_predict(out, tmp / "bundle", threads=2)           # load_target with games.py's timeline
    data = out.read_bytes()
    assert b"\r" not in data and data.startswith(b"frame,label\n") and data.endswith(b"\n")
    rows = data.decode("ascii").splitlines()[1:]
    assert len(rows) == 199319
    assert np.array_equal(np.array([int(r.split(",")[0]) for r in rows]), np.arange(199319))
    labels = np.array([r.split(",")[1] for r in rows])
    graded = np.zeros(199319, bool)
    for _, a, b in official_timeline().half_blocks:
        graded[a:b + 1] = True
    assert set(labels[graded]) <= {"H", "A", "D"} and (labels[~graded] == "X").all()
    assert rep["read"]["contiguous_from_one"] and rep["read"]["n_malformed"] == 0
    assert rep["frames"] == 199319 and rep["frames_in_halves"] == 158857 and rep["features"][1] == 213
    assert rep["submission"]["sha256"] == hashlib.sha256(data).hexdigest()
    r = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout[-300:]


def test_hold_out_leaves_the_match_out_and_keeps_the_submission_model(trained):
    from katena import pipeline as PL
    full, held = trained["full"], trained["held"]
    assert "held_out" not in full and held["held_out"] == ["idsse/SYN7"]
    assert held["matches"] == [r.key for r in trained["refs"] if r.key != "idsse/SYN7"]
    assert held["rows_used"] < held["rows"] == full["rows"]
    assert held["sha256"]["gbdt"] != full["sha256"]["gbdt"] and held["sha256"]["tcn"] != full["sha256"]["tcn"]
    assert PL.model_dir(["idsse/SYN7"]) != PL.model_dir()
    assert (PL.model_dir() / PL.BUNDLE).read_bytes() == trained["full_bytes"]          # untouched
    with pytest.raises(ValueError):
        PL.stage_train(trained["refs"], threads=2, steps=2, hold_out=["idsse/NOPE"])
    with pytest.raises(ValueError, match="never the competition"):
        PL.stage_predict(trained["tmp"] / "x.csv", None, hold_out=["idsse/SYN7"])
    with pytest.raises(ValueError, match="never the competition"):
        PL.stage_predict(trained["tmp"] / "x.csv", None, timeline=official_timeline())


def test_dressed_match_scores_what_the_evaluation_path_scores(trained):
    """The key check in small: export a training match, hold it out, label it through the submission path, and
    compare with the evaluation path for the same held-out model at native."""
    import possession_metrics as pm
    from katena import pipeline as PL
    from katena.eval import dressed as Dr
    from katena.grid import mask_outside_blocks
    from katena.types import load_match
    tmp = trained["tmp"]
    ref = next(r for r in trained["refs"] if r.key == "idsse/SYN7")
    m = load_match(ref.path)
    tl = dress.write_bundle(m, tmp / "dressed")
    assert all(dress.compare(m, load_target(tmp / "dressed", timeline=tl)).values())
    csv = tmp / "dressed_out" / "dressed.csv"
    PL.stage_predict(csv, tmp / "dressed", threads=2, hold_out=["idsse/SYN7"], timeline=tl)
    codes = pm.label_codes(pm.read_label_csv(str(csv)))
    model = PL.load_model(["idsse/SYN7"])
    ev = Dr.eval_path(model, ref, threads=2)
    sub = Dr.submission_features(load_target(tmp / "dressed", timeline=tl))
    for k in Dr.KEYS:
        assert np.array_equal(ev["features"][k], sub[k], equal_nan=True), k
    assert np.array_equal(codes, mask_outside_blocks(ev["native"], m))
    gt = pm.label_codes(pm.read_label_csv(str(_gt_csv(tmp, ref))))
    res = pm.evaluate(gt, codes)
    assert res["spell_f1_mean"] == ev["score"]["spell_f1"] and res["macro_f1"] == ev["score"]["macro_f1"]


def _gt_csv(tmp: Path, ref) -> Path:
    from katena import pipeline as PL
    from katena.eval import corpus as C
    from katena.types import LABEL_NAMES
    path = tmp / "gt_T2.csv"
    path.write_bytes(PL.submission_bytes(np.array(LABEL_NAMES)[C.load_gt(ref, P.T_MAIN)["y"]]))
    return path


def test_a_file_that_fails_the_official_checks_is_not_left_behind(tmp_path):
    from katena import pipeline as PL
    codes = np.array([3, 0, 1, 2, 3], dtype=np.int8)
    with pytest.raises(RuntimeError):
        PL.write_submission(codes, tmp_path / "bad.csv", 6)
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    sub = PL.write_submission(codes, tmp_path / "ok.csv", 5)
    assert (tmp_path / "ok.csv").read_bytes() == b"frame,label\n0,X\n1,H\n2,A\n3,D\n4,X\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ok.csv"] and sub["rows"] == 5
