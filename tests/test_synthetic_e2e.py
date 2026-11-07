"""The pieces of scripts/synthetic_e2e.py that CI's determinism job relies on (the full run is the
determinism job itself)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import possession_metrics as pm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import synthetic_e2e as e2e  # noqa: E402


def test_synthetic_match_is_valid_and_labelled_only_in_the_halves():
    m = e2e.synth_match(3, n_frames=9000)
    assert m.n == 9000 and m.P.shape == (9000, 22, 2)
    inside = np.zeros(m.n, dtype=bool)
    for _, a, b in m.half_blocks:
        inside[a:b + 1] = True
    assert (m.y[~inside] == 3).all() and set(np.unique(m.y[inside])) == {0, 1, 2}
    assert np.isnan(m.P[~inside]).all() and np.isfinite(m.P[inside]).all()
    np.testing.assert_array_equal(e2e.synth_match(3, n_frames=9000).ball, m.ball)     # seeded


def test_submission_bytes_are_lf_and_valid(tmp_path):
    labels = ["X", "H", "H", "A", "D", "X"] * 200
    data = e2e.submission_bytes(labels)
    assert data.startswith(b"frame,label\n0,X\n1,H\n") and b"\r" not in data and data.endswith(b"\n")
    path = tmp_path / "s.csv"
    path.write_bytes(data)
    assert pm.validate_submission(pm.read_label_csv(path), len(labels)) is None


def _hashes(tmp_path, name, **sha):
    record = {"sha256": dict({"submission_csv": "a" * 64, "models": "b" * 64}, **sha),
              "env": {"label": name, "os": "x", "machine": "y", "python": "3.11", "numpy": "2",
                      "lightgbm": "4", "numba": "0"},
              "summary": {"spell_f1_vs_synthetic_truth": 0.5}}
    path = tmp_path / "{}.json".format(name)
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


def test_compare_is_strict_unless_reporting(tmp_path, capsys):
    a, b = _hashes(tmp_path, "one"), _hashes(tmp_path, "two")
    c = _hashes(tmp_path, "three", models="c" * 64)
    assert e2e.compare([a, b]) == 0
    assert e2e.compare([a, c]) == 1
    summary = tmp_path / "summary.md"
    assert e2e.compare([a, b, c], report=True, summary_path=str(summary)) == 0
    assert "models" in capsys.readouterr().out and "| models |" in summary.read_text(encoding="utf-8")
