"""The data-firewall checker (scripts/firewall_check.py) on synthetic paths, tiny synthetic CSVs and
throwaway git repositories. Nothing here reads data/."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "scripts"):
    sys.path.insert(0, str(path))

import firewall_check as fc  # noqa: E402
import games  # noqa: E402

SENTINEL = "SENTINEL-CONTENT-MUST-NOT-BE-PRINTED"


# --- path rules -----------------------------------------------------------------------------------

@pytest.mark.parametrize("path, rules", [
    ("data/README.md", set()),
    ("data/tactical/BAR-ATH_tactical_atd.txt", {"match-data", "bundle-file"}),
    ("data/notes.md", {"match-data"}),
    ("Data/x.csv", {"match-data"}),
    ("evaluation/results/mine.csv", {"submission"}),
    ("evaluation/results/baseline_mine.csv", {"submission"}),
    ("evaluation/results/sub/dir.csv", {"submission"}),
    ("evaluation/results/baseline_smooth50.csv", set()),
    ("evaluation/results/README.md", set()),
    ("evaluation/gt/BAR-ATH_possession_gt.csv", {"private-gt"}),
    ("elsewhere/BAR-ATH_possession_gt_meta.json", {"private-gt"}),
    ("evaluation/gt/BAR-ATH_possession_gt_reference.csv", {"private-gt"}),
    ("evaluation/gt/BAR-ATH_possession_gt_public.csv", set()),
    ("evaluation/gt/BAR-ATH_possession_gt_public_meta.json", set()),
    ("x/possession_by_frame.pkl", {"private-gt", "binary"}),
    ("a/feats.npz", {"binary"}),
    ("a/feats.NPY", {"binary"}),
    ("a/t.parquet", {"binary"}),
    ("m.pkl", {"binary"}),
    ("bundle.zip", {"binary"}),
    ("video/clip.mp4", {"binary"}),
    ("raw.gzip", {"binary"}),
    ("x.pl3container", {"binary"}),
    ("docs/tech.pdf", {"binary"}),
    ("models/tcn.pt", {"binary"}),
    ("artifacts/run.json", {"generated-dir"}),
    ("src/cache/k.txt", {"generated-dir"}),
    ("features/a.txt", {"generated-dir"}),
    ("runs/1/log.txt", {"generated-dir"}),
    ("external_data/canonical/a.json", {"generated-dir"}),
    ("work/bakeoff/tables.md", {"generated-dir"}),
    ("submission.csv", {"root-csv"}),
    ("notes/BAR-ATH_tactical_preds.json", {"bundle-file"}),
    ("src/katena/featurize/build.py", set()),
    ("docs/workflow.md", set()),
    ("tests/test_cache.py", set()),
    ("docs/table.csv", set()),
    ("scripts/external_manifest.json", set()),
])
def test_path_rules(path, rules):
    assert {rule for rule, _ in fc.path_problems(path)} == rules


# --- content rules --------------------------------------------------------------------------------

def _csv(header: str, rows: int, eol: str = "\n", value=lambda i: "H") -> bytes:
    lines = [header] + ["{},{}".format(i, value(i)) for i in range(rows)]
    return (eol.join(lines) + eol).encode()


def test_submission_format_needs_enough_rows():
    assert {r for r, _ in fc.csv_problems(_csv("frame,label", 1000))} == {"submission-format"}
    assert fc.csv_problems(_csv("frame,label", 999)) == []
    crlf_bom = b"\xef\xbb\xbf" + _csv('"Frame" , "Label"', 1200, eol="\r\n")
    assert {r for r, _ in fc.csv_problems(crlf_bom)} == {"submission-format"}


def test_frame_export_by_row_count_at_native_and_work_grid_rates():
    lengths = (500,)
    assert {r for r, _ in fc.csv_problems(_csv("frame,p_home", 500, value=lambda i: "0.5"), lengths)} == {"frame-export"}
    assert {r for r, _ in fc.csv_problems(_csv("video_frame,x", 505, value=lambda i: "1"), lengths)} == {"frame-export"}
    assert {r for r, _ in fc.csv_problems(_csv("frame,x", 100, value=lambda i: "1"), lengths)} == {"frame-export"}
    assert fc.csv_problems(_csv("frame,p_home", 300, value=lambda i: "0.5"), lengths) == []
    assert fc.csv_problems(_csv("sample,p_home", 500, value=lambda i: "0.5"), lengths) == []


def test_frame_export_by_this_matchs_tracking_frame_numbers():
    lo, hi = fc.ID_FRAME_SPAN
    inside = "start_frame,team\n{},H\n".format(lo + 10).encode()
    tab = "IdFrame\tx\n{}\t1\n".format(hi).encode()
    outside = "start_frame,team\n{},H\n140000,A\n".format(lo - 1).encode()
    assert {r for r, _ in fc.csv_problems(inside)} == {"frame-export"}
    assert {r for r, _ in fc.csv_problems(tab)} == {"frame-export"}
    assert fc.csv_problems(outside) == []


def test_notebook_outputs():
    clean = {"cells": [{"cell_type": "code", "outputs": [], "source": "x"}], "nbformat": 4}
    dirty = {"cells": [{"cell_type": "code", "outputs": [{"text": SENTINEL}], "source": "x"}]}
    assert fc.notebook_problems(json.dumps(clean).encode()) == []
    assert {r for r, _ in fc.notebook_problems(json.dumps(dirty).encode())} == {"notebook-output"}
    assert {r for r, _ in fc.notebook_problems(b"not json")} == {"notebook"}


def test_blob_rules_size_and_pins():
    nothing = lambda: b""  # noqa: E731
    assert {r for r, _ in fc.blob_problems("a/b.bin", "0" * 40, fc.MAX_BLOB_BYTES + 1, nothing)} == {"large"}
    assert fc.blob_problems("a/b.bin", "0" * 40, fc.MAX_BLOB_BYTES, nothing) == []
    path, pinned = next(iter(fc.PUBLISHED.items()))
    assert fc.blob_problems(path, pinned, 10 ** 9, nothing) == []            # pinned: never read, never sized
    assert {r for r, _ in fc.blob_problems(path, "1" * 40, 10, nothing)} == {"published-modified"}


def test_config_rules():
    good = {fc.SETTINGS: json.dumps({"permissions": {"deny": ["Read(./data/**)"]}}).encode(),
            fc.CLAUDE_MD: b"# x\n\n## Data firewall (terms)\n"}
    assert fc.config_problems(good.get) == []
    no_deny = dict(good, **{fc.SETTINGS: json.dumps({"permissions": {"deny": ["Read(./cache/**)"]}}).encode()})
    assert {r for _, r, _ in fc.config_problems(no_deny.get)} == {"settings"}
    broken = dict(good, **{fc.SETTINGS: b"{"})
    assert {r for _, r, _ in fc.config_problems(broken.get)} == {"settings"}
    no_section = dict(good, **{fc.CLAUDE_MD: b"# x\nThe data firewall is gone.\n"})
    assert {r for _, r, _ in fc.config_problems(no_section.get)} == {"claude-md"}
    assert {r for _, r, _ in fc.config_problems({}.get)} == {"settings", "claude-md"}


def test_constants_match_games_py():
    assert set(fc.BUNDLE_STEMS) == {s["stem"] for s in games.SOURCES.values()}
    entry = games.GAMES[games.GAME]
    halves = entry["halves"]
    graded = sum(b - a + 1 for a, b in halves.values())
    assert set(fc.PER_FRAME_LENGTHS) == {entry["n_video_frames"], graded} | {
        s["n_video_frames"] for s in games.SOURCES.values()}
    assert fc.ID_FRAME_SPAN == (min(a for a, _ in halves.values()), max(b for _, b in halves.values()))


def test_published_pins_match_this_checkout():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "--"] + sorted(fc.PUBLISHED),
                            stdout=subprocess.PIPE, check=True).stdout.decode()
    blobs = {line.split("\t", 1)[1]: line.split()[1] for line in listed.splitlines()}
    assert blobs == fc.PUBLISHED


# --- end to end on throwaway repositories ---------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root)] + list(args), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout.decode()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    config = tmp_path / "gitconfig"                      # isolate from the developer's git config
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv("GIT_{}_NAME".format(who), "firewall test")
        monkeypatch.setenv("GIT_{}_EMAIL".format(who), "test@example.invalid")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".claude").mkdir()
    (root / fc.SETTINGS).write_text(json.dumps({"permissions": {"deny": ["Read(./data/**)"]}}), encoding="utf-8")
    (root / fc.CLAUDE_MD).write_text("# Project\n\n## Data firewall\n\nRules.\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / fc.DATA_README).write_text("Put the bundle here.\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _run(root: Path, *args: str, stdin: str = ""):
    out = io.StringIO()
    code = fc.main(["--repo", str(root)] + list(args), stdin=io.StringIO(stdin), out=out)
    return code, out.getvalue()


def test_clean_repository_passes(repo):
    assert _run(repo)[0] == 0
    assert _run(repo, "--history", "all")[0] == 0


def test_staged_match_data_fails_without_printing_it(repo):
    (repo / "data" / "BAR-ATH_tactical_atd.txt").write_text(SENTINEL, encoding="utf-8")
    _git(repo, "add", "data")
    code, text = _run(repo)
    assert code == 1 and "[match-data] data/BAR-ATH_tactical_atd.txt" in text
    assert SENTINEL not in text


def test_deleted_file_is_still_caught_in_history(repo):
    rows = "\n".join("{},{}".format(i, SENTINEL) for i in range(1500))
    (repo / "notes").mkdir()
    (repo / "notes" / "preds.csv").write_text("frame,label\n" + rows + "\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "oops")
    assert "[submission-format] notes/preds.csv" in _run(repo)[1]
    _git(repo, "rm", "-q", "--cached", "notes/preds.csv")
    _git(repo, "commit", "-q", "-m", "remove")
    assert _run(repo)[0] == 0                                       # gone from the index...
    code, text = _run(repo, "--history", "HEAD")                    # ...but not from history
    assert code == 1 and "notes/preds.csv  (history " in text
    assert SENTINEL not in text


def test_pre_push_checks_unpushed_commits(repo):
    (repo / "out.npz").write_bytes(b"\x00\x01")
    _git(repo, "add", "out.npz")
    _git(repo, "commit", "-q", "-m", "cache")
    _git(repo, "rm", "-q", "out.npz")
    _git(repo, "commit", "-q", "-m", "remove cache")
    head = _git(repo, "rev-parse", "HEAD").strip()
    line = "refs/heads/main {} refs/heads/main {}\n".format(head, fc.ZERO_SHA)
    code, text = _run(repo, "--pre-push", stdin=line)
    assert code == 1 and "[binary] out.npz" in text
    deleting = "(delete) {} refs/heads/old {}\n".format(fc.ZERO_SHA, head)
    assert _run(repo, "--pre-push", stdin=deleting)[0] == 0


def test_modified_published_file_and_guard_rails(repo):
    target = repo / "evaluation" / "results"
    target.mkdir(parents=True)
    (target / "baseline_smooth50.csv").write_text("frame,label\n0,H\n", encoding="utf-8")
    (repo / fc.CLAUDE_MD).write_text("# Project\n", encoding="utf-8")
    _git(repo, "add", "-A")
    code, text = _run(repo)
    assert code == 1
    assert "[published-modified] evaluation/results/baseline_smooth50.csv" in text
    assert "[claude-md] CLAUDE.md" in text


def test_hooks_call_the_checker():
    for name in ("pre-commit", "pre-push"):
        hook = REPO / ".githooks" / name
        text = hook.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh\n") and "firewall.sh" in text and "\r" not in text
    helper = (REPO / ".githooks" / "firewall.sh").read_text(encoding="utf-8")
    assert "scripts/firewall_check.py" in helper and "\r" not in helper
    if shutil.which("git") and (REPO / ".git").exists():
        modes = _git(REPO, "ls-files", "-s", ".githooks")
        for line in modes.splitlines():
            assert line.startswith("100755 "), line                   # executable on macOS and Linux
