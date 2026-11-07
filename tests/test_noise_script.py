import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "noise_fit_target.py"


def _load():
    spec = importlib.util.spec_from_file_location("noise_fit_target", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_on_synthetic(tmp_path, capsys):
    out = tmp_path / "noise_cfg.json"
    rc = _load().main(["--synthetic", "--out", str(out), "--max-evals", "6", "--synthetic-half-minutes", "1",
                       "--max-minutes", "2", "--domain-gap", "--window-s", "5"])
    assert rc == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert 0 < cfg["ball_present_rate"] < 1 and "_fit" in cfg and "domain_gap" in cfg["_fit"]
    text = capsys.readouterr().out
    assert "target_stats" in text and "AUC" in text


def test_pending_loader_exits_cleanly(tmp_path, monkeypatch, capsys):
    # simulate the loader being absent, so this test can never touch real data even after J1/D1 land
    monkeypatch.setitem(sys.modules, "katena.io", None)
    with pytest.raises(SystemExit) as e:
        _load().main(["--out", str(tmp_path / "x.json")])
    assert e.value.code == 2
    assert "pending" in capsys.readouterr().out
    assert not (tmp_path / "x.json").exists()
