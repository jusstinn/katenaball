"""``make_submission.py bundle``: are the three files ``predict`` reads where it will look for them? A folder of
empty files with the bundle's names stands in for the data, as in tests/test_real_run.py; nothing here goes near
the competition data, and the script runs as a child so its environment pins stay out of the test process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

import games  # noqa: E402

FILES = ("atd_meta", "atd_raw", "smart_tagging")


def fake_bundle(root: Path, leave_out: tuple = ()) -> dict:
    paths = games.asset_paths(data_dir=str(root))
    for name in FILES:
        if name not in leave_out:
            Path(paths[name]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths[name]).write_bytes(b"")
    return paths


def run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("MTAH_DATA_DIR", "KATENA_EXTERNAL")}
    env.update(KATENA_WORK=str(tmp_path / "work"), KATENA_EXTERNAL=str(tmp_path / "external"),
               MTAH_DATA_DIR=str(tmp_path / "nowhere"), PYTHONUTF8="1")
    return subprocess.run([sys.executable, str(REPO / "scripts" / "make_submission.py"), *args], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)


@pytest.fixture(scope="module", autouse=True)
def _needs_the_models():
    from katena.models import gbdt
    if gbdt.lgb is None:
        pytest.skip("lightgbm cannot load in this process, so the script would stop before the check")
    pytest.importorskip("torch")


def test_bundle_passes_when_the_three_files_are_there(tmp_path):
    fake_bundle(tmp_path / "bundle")
    r = run(tmp_path, "bundle", "--data-dir", str(tmp_path / "bundle"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "bundle: 3 files found" in r.stdout


def test_bundle_names_a_missing_file_and_where_it_looked(tmp_path):
    paths = fake_bundle(tmp_path / "bundle", leave_out=("smart_tagging",))
    r = run(tmp_path, "bundle", "--data-dir", str(tmp_path / "bundle"))
    assert r.returncode == 1
    assert Path(paths["smart_tagging"]).name in r.stdout and "MTAH_DATA_DIR" in r.stdout
    assert Path(paths["atd_raw"]).name not in r.stdout          # only what is missing
    assert "docs/REPRODUCE.md" in r.stdout


def test_all_stops_at_a_misplaced_bundle_before_any_stage(tmp_path):
    r = run(tmp_path, "all")                                    # MTAH_DATA_DIR points at an empty place
    assert r.returncode == 1
    assert "match data not found" in r.stdout and "corpus" not in r.stdout.split("bundle:", 1)[1]
    assert not (tmp_path / "work").exists()


def test_predict_on_the_competition_match_needs_the_tagging_file(tmp_path):
    fake_bundle(tmp_path / "bundle", leave_out=("smart_tagging",))
    r = run(tmp_path, "predict", "--data-dir", str(tmp_path / "bundle"), "--out", str(tmp_path / "x.csv"))
    assert r.returncode == 1 and "smart_tagging" in r.stdout and not (tmp_path / "x.csv").exists()
