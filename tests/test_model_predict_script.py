"""scripts/predict_target.py end to end in --synthetic mode (never touches the competition data)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from katena.feats import featurize
from katena.model import Model
from katena.model import data as Dm
from katena.noise import corrupt
from katena.synth import synthetic_match

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_target.py"


def _load():
    spec = importlib.util.spec_from_file_location("predict_target", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    copies = []
    for s in (11, 12):
        m = corrupt(synthetic_match(seed=s, half_minutes=1.0), {}, seed=s)
        f = featurize(m, {"stride": 5})
        copies.append({"X": f.X, "t_frame": f.t_frame, "y": m.y[f.t_frame].astype(np.int8),
                       "names": list(f.names)})
    X, y, names = Dm.stack_training(copies)
    model = Model(kind="twohead", params={"num_leaves": 15, "min_data_in_leaf": 20, "num_threads": 2},
                  rounds=10).fit(X, y, names)
    model.info.update({"decode": {"switch_penalty": 8.0, "min_dwell_s": 1.0}, "predict_stride": 5})
    return model.save(tmp_path_factory.mktemp("model") / "tiny.json")


def test_synthetic_end_to_end_is_valid_and_deterministic(tiny_model, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "katena.io", None)   # the real-match loader must not be needed
    mod = _load()
    outs = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for out in outs:
        rc = mod.main(["--synthetic", "--model", str(tiny_model), "--out", str(out),
                       "--synthetic-half-minutes", "1"])
        assert rc == 0
    text = capsys.readouterr().out
    assert text.count("OK    ") == 2 and "possessions" in text and "sha256" in text
    assert outs[0].read_bytes() == outs[1].read_bytes()
    m = mod.synthetic_target(1.0)
    s = mod.summarise(str(outs[0]), m.n)
    assert s["valid"] and s["rows"] == m.n
    assert s["shares"]["X"] == pytest.approx(1 - m.graded_mask().mean())
    lines = outs[0].read_text().splitlines()
    assert lines[0] == "frame,label" and lines[1].startswith("0,")


def test_missing_model_fails_cleanly(tmp_path, capsys):
    rc = _load().main(["--synthetic", "--model", str(tmp_path / "none.json"),
                       "--out", str(tmp_path / "x.csv")])
    assert rc == 2 and "FAIL" in capsys.readouterr().out
    assert not (tmp_path / "x.csv").exists()
