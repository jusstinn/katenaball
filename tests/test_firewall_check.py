"""The data-firewall checker (scripts/firewall_check.py) on synthetic paths, synthetic file contents and
throwaway git repositories. Nothing here reads data/: every match-shaped file below is generated from
a seeded random number generator, at the match's lengths but with made-up values."""

from __future__ import annotations

import io
import json
import os
import random
import re
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
SHA = "0" * 40


def _rules(found) -> set:
    return {rule for rule, _ in found}


def _blob(path: str, data: bytes) -> set:
    return _rules(fc.blob_problems(path, SHA, len(data), lambda: data))


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
    # Frame images, crops, spreadsheets, databases and the remaining table formats.
    ("docs/frame_011185.jpg", {"binary"}),
    ("crops/p1.JPEG", {"binary"}),
    ("x.webp", {"binary"}),
    ("x.bmp", {"binary"}),
    ("x.tif", {"binary"}),
    ("x.TIFF", {"binary"}),
    ("docs/plot.png", {"binary"}),
    ("docs/anim.gif", {"binary"}),
    ("t.xlsx", {"binary"}),
    ("t.xls", {"binary"}),
    ("t.ods", {"binary"}),
    ("d.sqlite", {"binary"}),
    ("d.db", {"binary"}),
    ("m.mat", {"binary"}),
    ("t.feather", {"binary"}),
    ("t.h5", {"binary"}),
    ("artifacts/run.json", {"generated-dir"}),
    ("src/cache/k.txt", {"generated-dir"}),
    ("features/a.txt", {"generated-dir"}),
    ("runs/1/log.txt", {"generated-dir"}),
    ("external_data/canonical/a.json", {"generated-dir"}),
    ("work/bakeoff/tables.md", {"generated-dir"}),
    ("submission.csv", {"root-csv"}),
    ("notes/BAR-ATH_tactical_preds.json", {"bundle-file"}),
    ("src/katena/featurize/build.py", set()),
    ("src/katena/pipeline/run.py", set()),
    ("scripts/make_submission.py", set()),
    ("docs/technical-document.md", set()),
    ("docs/workflow.md", set()),
    ("tests/test_cache.py", set()),
    ("docs/table.csv", set()),
    ("scripts/external_manifest.json", set()),
    ("src/katena/format.py", set()),
])
def test_path_rules(path, rules):
    assert _rules(fc.path_problems(path)) == rules


def test_images_are_allowed_by_exact_path_only(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    monkeypatch.setattr(fc, "ALLOWED_IMAGES", ("docs/img/pipeline.png",))
    assert fc.path_problems("docs/img/pipeline.png") == []
    assert _blob("docs/img/pipeline.png", png) == set()
    assert _rules(fc.path_problems("docs/img/other.png")) == {"binary"}
    assert _rules(fc.path_problems("Docs/img/pipeline.png")) == {"binary"}       # exact, case included
    assert _rules(fc.path_problems("docs/img/pipeline.npz")) == {"binary"}       # images only
    big = fc.blob_problems("docs/img/pipeline.png", SHA, fc.MAX_BLOB_BYTES + 1, lambda: png)
    assert _rules(big) == {"large"}


def test_allowed_images_are_tracked():
    """An allow-listed path that is not tracked is a hole waiting for a frame image."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(_git(REPO, "ls-files").splitlines())
    assert set(fc.ALLOWED_IMAGES) <= tracked
    images = fc.FORBIDDEN_SUFFIXES[next(k for k in fc.FORBIDDEN_SUFFIXES if k.startswith("images"))]
    assert {p for p in tracked if p.lower().endswith(images)} == set(fc.ALLOWED_IMAGES)


def test_binary_content_under_any_name():
    assert _blob("notes/frames.txt", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") == {"binary-content"}
    assert _blob("src/katena/blob.bin", b"\x93NUMPY\x01\x00v\x00") == {"binary-content"}
    assert _blob("a/b.bin", b"") == set()


def test_git_lfs_pointers_and_filters():
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123456\n"
    assert _blob("docs/match.csv", pointer) == {"lfs"}
    assert _blob("src/katena/weights.txt", pointer) == {"lfs"}
    lfs = b"# big files\n*.csv filter=lfs diff=lfs merge=lfs -text\n"
    assert _blob(".gitattributes", lfs) == {"lfs"}
    assert _blob("docs/sub/.gitattributes", b"*.bin\tfilter=lfs -text\n") == {"lfs"}
    assert _blob(".gitattributes", b"* text=auto eol=lf\n# *.csv filter=lfs\n*.png binary\n") == set()
    assert _blob("docs/notes.md", lfs) == set()                   # only .gitattributes routes files
    assert _blob(".gitattributes", (REPO / ".gitattributes").read_bytes()) == set()


# --- content rules: row and line counts -----------------------------------------------------------

def _lines(rows, header=None) -> bytes:
    return "".join(line + "\n" for line in ([header] if header else []) + list(rows)).encode()


def _shuffled(i: int) -> int:
    return (i * 7919) % 100003          # distinct, but never an evenly stepped run of frame ids


def test_targets_cover_every_match_length_at_25_and_5_hz():
    assert {n for n, _ in fc._targets(fc.MATCH_LENGTHS)} == {
        199319, 164488, 158857, 79467, 79390, 39864, 32898, 31772, 15894, 15878}


def test_csv_row_count_whatever_the_columns_with_and_without_header():
    lengths = (500,)
    values = ["{},{:.3f}".format(_shuffled(i), (i % 7) / 7) for i in range(520)]
    for header in ("frame,p_home", "video_frame,x", "sample,p_home", "t,label", None):
        for n in (500, 505, 491, 100):
            found = fc.csv_problems(_lines(values[:n], header), lengths)
            assert _rules(found) == {"frame-export"}, (header, n)
        assert fc.csv_problems(_lines(values[:300], header), lengths) == [], header


def test_t_label_csv_at_the_graded_length():
    rng = random.Random(1)
    rows = ["{:.2f},{}".format(i / 25, rng.choice("HAD")) for i in range(158857)]
    assert "frame-export" in _rules(fc.csv_problems(_lines(rows, "t,label")))


def test_headerless_savetxt_of_probabilities():
    rng = random.Random(2)
    rows = ["{:.5e} {:.5e}".format(p, 1 - p) for p in (rng.random() for _ in range(199319))]
    data = _lines(rows)
    assert data.count(b"\n") == 199319 and len(data) < fc.MAX_BLOB_BYTES
    assert _blob("notes/probs.csv", data) == {"frame-export"}
    assert _blob("notes/probs.txt", data) == {"frame-export"}


@pytest.mark.parametrize("n", [79467, 79390])
def test_one_half_export(n):
    rng = random.Random(n)
    rows = ["{:.4f},{:.4f}".format(rng.random(), rng.random()) for _ in range(n)]
    assert _rules(fc.csv_problems(_lines(rows, "p_home,p_away"))) == {"frame-export"}
    assert _rules(fc.csv_problems(_lines(rows))) == {"frame-export"}


def test_labels_in_text_and_json_files():
    rng = random.Random(3)
    labels = [rng.choice("HAD") for _ in range(158857)]
    assert "frame-export" in _blob("notes/labels.txt", _lines(labels))
    assert "frame-export" in _blob("notes/labels.json", json.dumps(labels).encode())
    assert "frame-export" in _blob("notes/labels.json", json.dumps({"labels": labels}, indent=1).encode())
    assert "frame-export" in _blob("notes/labels.json", json.dumps({"labels": "".join(labels)}).encode())
    pairs = [[11185 + i, label] for i, label in enumerate(labels)]
    assert "frame-export" in _blob("notes/pairs.json", json.dumps(pairs).encode())
    halves = {"first": labels[:79467], "second": labels[79467:]}
    assert "frame-export" in _blob("notes/halves.json", json.dumps(halves, separators=(",", ":")).encode())
    assert "frame-export" in _blob("notes/labels.py", "LABELS = {!r}\n".format(labels).encode())


def test_submission_format_needs_enough_rows():
    assert "submission-format" in _rules(fc.csv_problems(_lines(("{},H".format(i) for i in range(1000)), "frame,label")))
    assert "submission-format" not in _rules(fc.csv_problems(_lines(("{},H".format(i) for i in range(999)), "frame,label")))
    crlf_bom = b"\xef\xbb\xbf" + "\r\n".join(['"Frame" , "Label"'] + ["{},H".format(i) for i in range(1200)]).encode()
    assert "submission-format" in _rules(fc.csv_problems(crlf_bom))


def test_frame_export_by_this_matchs_tracking_frame_numbers():
    lo, hi = fc.ID_FRAME_SPAN
    inside = "start_frame,team\n{},H\n".format(lo + 10).encode()
    tab = "IdFrame\tx\n{}\t1\n".format(hi).encode()
    outside = "start_frame,team\n{},H\n140000,A\n".format(lo - 1).encode()
    assert _rules(fc.csv_problems(inside)) == {"frame-export"}
    assert _rules(fc.csv_problems(tab)) == {"frame-export"}
    assert fc.csv_problems(outside) == []


def test_tracking_frame_numbers_in_order_in_any_text_file():
    lo = fc.ID_FRAME_SPAN[0]
    starts = [lo + 300 * i + (i * 37) % 200 for i in range(12)]
    spells = [{"start": a, "end": b, "team": "H"} for a, b in zip(starts, starts[1:])]
    assert _blob("notes/spells.json", json.dumps(spells, indent=2).encode()) == {"frame-export"}
    shuffled = [starts[(i * 5) % 12] for i in range(12)]                         # wheel sizes in uv.lock
    assert _blob("notes/sizes.toml", "".join("size = {}\n".format(s) for s in shuffled).encode()) == set()
    paths = "".join("data/matches/{0}/{0}_tracking.jsonl\n".format(s) for s in sorted(starts))
    assert _blob("notes/manifest.txt", paths.encode()) == set()                  # SkillCorner match ids


# --- content rules: the bundle's raw formats ------------------------------------------------------

def _atd_line(frame_id: int, slots: int, rng: random.Random) -> str:
    groups = ";".join("{:.4f},{:.4f},{:.2f}".format(rng.random(), rng.random(), rng.random() * 8)
                      if rng.random() > 0.3 else "NaN,NaN,NaN" for _ in range(slots))
    return "{}:{}:{:.4f},{:.4f},{:.2f}".format(frame_id, groups, rng.random(), rng.random(), rng.random())


def test_short_atd_fixtures():
    rng = random.Random(4)
    short = _lines(_atd_line(11185 + i, 46, rng) for i in range(5))
    assert _blob("tests/fixtures/atd_short.txt", short) == {"atd-raw"}
    assert _blob("tests/fixtures/one_frame.txt", _lines([_atd_line(11185, 46, rng)])) == {"atd-raw"}
    assert _blob("notes/one_frame.md", _lines([_atd_line(11185, 51, rng)])) == {"atd-raw"}     # broadcast
    three = [_atd_line(i + 1, 3, rng) for i in range(3)]
    assert _blob("tests/fixtures/tiny.txt", _lines(three)) == {"atd-raw"}
    assert _blob("tests/fixture_atd.py", "LINES = {!r}\n".format(three).encode()) == {"atd-raw"}
    assert _blob("docs/format.md", _lines(three[:2])) == set()                  # a docs example or two


def test_atd_metadata():
    players = "".join('<Player id="P{}" teamId="T{}"><Name>Track {}</Name></Player>\n'.format(i, i % 2, i)
                      for i in range(46))
    assert _blob("notes/meta.xml", "<Metadata><Players>\n{}</Players></Metadata>\n".format(players).encode()) == {"atd-meta"}
    few = "".join('<Player id="P{}" teamId="T1"/>\n'.format(i) for i in range(8))
    assert _blob("tests/meta.xml", few.encode()) == set()


def _homography_row(frame: int, rng: random.Random, sep: str = ",", valid: bool = True) -> str:
    matrix = ["{:.6g}".format(rng.uniform(-2000, 2000)) for _ in range(8)] + ["1"] if valid else ["NaN"] * 9
    return sep.join([str(frame)] + matrix + ["{:.4f}".format(rng.random()), "1" if valid else "0"])


def test_homography_rows():
    rng = random.Random(5)
    rows = [_homography_row(11185 + i, rng, valid=i != 1) for i in range(3)]
    header = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
    assert _blob("notes/h.csv", _lines(rows, header)) == {"homography"}
    assert _blob("notes/h.txt", _lines(_homography_row(i, rng, sep=" ") for i in range(3))) == {"homography"}
    no_frame = [row.split(",", 1)[1] for row in rows]
    assert _blob("notes/h.txt", _lines(no_frame)) == {"homography"}
    assert _blob("docs/format.md", _lines(rows[:1], header)) == set()           # DATA.md shows one row


def test_smart_tagging_records():
    def instance(i):
        return ("<instance><ID>{0}</ID><start>{1:.2f}</start><end>{2:.2f}</end><code>BUILD UP</code>"
                "<label><group>Team</group><text>Barcelona</text></label></instance>\n").format(i, 10.0 * i, 10.0 * i + 4)
    xml = "<file><ALL_INSTANCES>\n{}</ALL_INSTANCES></file>\n"
    assert _blob("notes/tags.xml", xml.format("".join(instance(i) for i in range(10))).encode()) == {"tagging"}
    assert _blob("tests/tags.xml", xml.format("".join(instance(i) for i in range(3))).encode()) == set()
    rows = ["0 days 00:{:02d}:27.440000,0 days 00:{:02d}:31.000000,SET PIECES,N/A,1st Half".format(i, i)
            for i in range(10)]
    assert _blob("notes/tags.csv", _lines(rows, "timestamp,end_timestamp,code,Team,Half")) == {"tagging"}


# --- content rules: runs of frame-level rows ------------------------------------------------------

def test_frame_label_rows_in_long_runs():
    lengths = fc.MATCH_LENGTHS
    rows = ["{},{}".format(11185 + 7 * i, "HAD"[i % 3]) for i in range(fc.LABEL_RUN_MIN)]
    assert _blob("docs/window.md", _lines(rows, "frame,label")) == {"frame-labels"}
    assert _blob("docs/window.md", _lines(rows[:-1], "frame,label")) == set()
    spells = ["{},{},{}".format(1000 * i, 1000 * i + 700, "HA"[i % 2]) for i in range(30)]
    assert _blob("notes/spells.txt", _lines(spells)) == {"frame-labels"}
    jsonl = ['{{"frame": {}, "label": "{}"}}'.format(_shuffled(i), "HAD"[i % 3]) for i in range(30)]
    assert _blob("notes/preds.jsonl", _lines(jsonl)) == {"frame-labels"}
    probs = ["{:.3f},{:.3f},{}".format(0.1 * (i % 9), 0.05, "HAD"[i % 3]) for i in range(30)]
    assert _rules(fc.csv_problems(_lines(probs, "p_home,p_away,label"), lengths)) == {"frame-labels"}


def test_monotone_runs_of_frame_ids():
    def rows(n, step):
        return _lines(("{},{:.3f},{:.3f}".format(11185 + step * i, (i % 13) / 13, (i % 7) / 7) for i in range(n)))
    assert _blob("notes/features.txt", rows(fc.FRAME_RUN_MIN, 1)) == {"frame-run"}
    assert _blob("notes/features.txt", rows(fc.FRAME_RUN_MIN, fc.WORK_GRID_STEP)) == {"frame-run"}
    assert _blob("notes/features.txt", rows(fc.FRAME_RUN_MIN - 1, 1)) == set()
    assert _blob("notes/features.txt", rows(fc.FRAME_RUN_MIN, fc.FRAME_RUN_MAX_STEP + 1)) == set()
    pretty = json.dumps({"frames": list(range(11185, 11185 + fc.FRAME_RUN_MIN))}, indent=1)
    assert _blob("notes/frames.json", pretty.encode()) == {"frame-run"}


def test_code_full_of_small_integers_is_not_flagged():
    """Integers in 1..199319 are everywhere in code. Only structure counts, never how many there are."""
    rng = random.Random(6)
    table = "".join("    {},\n".format(rng.randint(1, 199319)) for _ in range(5000))
    pairs = "".join("    {}: {},\n".format(rng.randint(1, 199319), rng.randint(1, 199319)) for _ in range(3000))
    module = ("TABLE = [\n" + table + "]\nPAIRS = {\n" + pairs + "}\n"
              + "x = a[0:10, 0:3]\nt = '12:30'\nN = {}\nHALVES = [(11185, 90651), (110808, 190197)]\n"
              .format(games.GAMES[games.GAME]["n_video_frames"]))
    assert _blob("src/katena/pipeline/tables.py", module.encode()) == set()
    numbered = "".join("{}. step {} of the plan\n".format(i, i) for i in range(1, 40))
    table_md = "| fold | spellF1 |\n|---|---|\n" + "".join("| {} | 0.{} |\n".format(i, 600 + i) for i in range(28))
    assert _blob("docs/technical-document.md", (numbered + table_md).encode()) == set()
    one_line = "SIZES = [{}]\n".format(", ".join(str(rng.randint(1, 199319)) for _ in range(20000)))
    assert _blob("src/katena/pipeline/sizes.py", one_line.encode()) == set()


def test_this_repository_passes():
    """Every tracked file, code, docs and tests included, passes every rule: no false positives."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    problems, n = fc.check_index(fc.Git(str(REPO)))
    assert n > 100 and problems == []


def test_notebook_outputs():
    clean = {"cells": [{"cell_type": "code", "outputs": [], "source": "x"}], "nbformat": 4}
    dirty = {"cells": [{"cell_type": "code", "outputs": [{"text": SENTINEL}], "source": "x"}]}
    assert fc.notebook_problems(json.dumps(clean).encode()) == []
    assert _rules(fc.notebook_problems(json.dumps(dirty).encode())) == {"notebook-output"}
    assert _rules(fc.notebook_problems(b"not json")) == {"notebook"}


def test_blob_rules_size_and_pins():
    nothing = lambda: b""  # noqa: E731
    assert _rules(fc.blob_problems("a/b.bin", SHA, fc.MAX_BLOB_BYTES + 1, nothing)) == {"large"}
    assert fc.blob_problems("a/b.bin", SHA, fc.MAX_BLOB_BYTES, nothing) == []
    path, pinned = next(iter(fc.PUBLISHED.items()))
    assert fc.blob_problems(path, pinned, 10 ** 9, nothing) == []            # pinned: never read, never sized
    assert _rules(fc.blob_problems(path, "1" * 40, 10, nothing)) == {"published-modified"}


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
    per_half = {b - a + 1 for a, b in halves.values()}
    assert set(fc.HALF_LENGTHS) == per_half
    assert set(fc.PER_FRAME_LENGTHS) == {entry["n_video_frames"], sum(per_half)} | {
        s["n_video_frames"] for s in games.SOURCES.values()}
    assert fc.ID_FRAME_SPAN == (min(a for a, _ in halves.values()), max(b for _, b in halves.values()))
    blocks = games.half_video_blocks(games.GAME)
    assert sorted(fc.HALF_LENGTHS, reverse=True) == [last - first + 1 for _, first, last in blocks]


def test_published_pins_match_this_checkout():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "--"] + sorted(fc.PUBLISHED),
                            stdout=subprocess.PIPE, check=True).stdout.decode()
    blobs = {line.split("\t", 1)[1]: line.split()[1] for line in listed.splitlines()}
    assert blobs == fc.PUBLISHED


# --- CI and release workflows ---------------------------------------------------------------------

def _jobs(workflow: str) -> dict:
    """{job name: its text} for a workflow file, split on two-space-indented keys under jobs:."""
    text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"(?m)^  ([\w-]+):\n", body)
    return dict(zip(parts[1::2], parts[2::2]))


def test_ci_runs_mains_copy_of_the_scanner_off_main():
    firewall = _jobs("ci.yml")["firewall"]
    assert "fetch-depth: 0" in firewall
    assert "git fetch" in firewall and "refs/heads/main:refs/remotes/origin/main" in firewall
    assert "git show origin/main:scripts/firewall_check.py" in firewall and "python3 -" in firewall
    assert "github.ref != 'refs/heads/main'" in firewall
    assert "python3 scripts/firewall_check.py" in firewall               # and the branch's own copy


def test_ci_grants_pull_request_read_to_the_plan_job_only():
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    top = text.split("\njobs:\n", 1)[0].split("\npermissions:\n", 1)[1].split("\n\n", 1)[0]
    assert "contents: read" in top and not re.search(r"(?m)^\s+pull-requests:", top)
    jobs = _jobs("ci.yml")
    assert [name for name, job in jobs.items() if "pull-requests: read" in job] == ["plan"]


def test_release_requires_the_tag_on_main():
    jobs = _jobs("release.yml")
    guard = [name for name, job in jobs.items() if "merge-base --is-ancestor" in job]
    assert len(guard) == 1
    assert "refs/heads/main:refs/remotes/origin/main" in jobs[guard[0]]
    assert re.search(r"needs: \[?.*\b{}\b".format(guard[0]), jobs["release-note"])
    assert re.search(r"needs: \[?.*\b{}\b".format(guard[0]), jobs["ci"])


def test_no_codeowners_file_is_committed():
    """The CODEOWNERS proposal lives in docs/ci.md until an admin enables code-owner review."""
    assert not any((REPO / where / "CODEOWNERS").exists() for where in (".", ".github", "docs"))
    assert "CODEOWNERS" in (REPO / "docs" / "ci.md").read_text(encoding="utf-8")


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


def test_content_rules_never_print_values(repo):
    rng = random.Random(7)
    lines = [_atd_line(11185 + i, 46, rng) for i in range(4)]
    (repo / "notes").mkdir()
    (repo / "notes" / "fixture.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (repo / "notes" / "labels.json").write_text(json.dumps(["H"] * 158857), encoding="utf-8")
    (repo / ".gitattributes").write_text("*.npz filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    _git(repo, "add", "-A")
    code, text = _run(repo)
    assert code == 1
    assert "[atd-raw] notes/fixture.txt" in text and "[frame-export] notes/labels.json" in text
    assert "[lfs] .gitattributes" in text
    assert not any(line.split(":", 1)[1][:40] in text for line in lines)


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


def test_scanner_runs_from_stdin_as_ci_runs_mains_copy(repo):
    """CI pipes main's copy into `python3 -`; it must work without its own file on disk."""
    script = (REPO / "scripts" / "firewall_check.py").read_bytes()
    command = [sys.executable, "-", "--repo", str(repo), "--history", "all"]
    clean = subprocess.run(command, input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert clean.returncode == 0, clean.stderr.decode()
    (repo / "notes.xlsx").write_bytes(b"PK\x03\x04")
    _git(repo, "add", "notes.xlsx")
    dirty = subprocess.run(command, input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert dirty.returncode == 1 and b"[binary] notes.xlsx" in dirty.stdout


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


def test_hooks_block_a_commit_and_a_push(repo, tmp_path, monkeypatch):
    """The real hooks, run by git, on a scratch repository with a copy of the checker."""
    shutil.copytree(str(REPO / ".githooks"), str(repo / ".githooks"))
    (repo / "scripts").mkdir()
    shutil.copy(str(REPO / "scripts" / "firewall_check.py"), str(repo / "scripts" / "firewall_check.py"))
    for hook in (repo / ".githooks").iterdir():
        hook.chmod(0o755)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    hooks = ["-c", "core.hooksPath=.githooks"]

    def git(*args):
        return subprocess.run(["git", "-C", str(repo)] + hooks + list(args), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    _git(repo, "add", "-A")
    assert git("commit", "-q", "-m", "hooks").returncode == 0
    (repo / "frame.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    _git(repo, "add", "frame.jpg")
    blocked = git("commit", "-q", "-m", "a frame image")
    assert blocked.returncode != 0 and b"[binary] frame.jpg" in blocked.stdout + blocked.stderr

    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    assert git("push", "-q", str(remote), "HEAD:refs/heads/main").returncode == 0
    assert git("commit", "-q", "--no-verify", "-m", "a frame image").returncode == 0
    refused = git("push", "-q", str(remote), "HEAD:refs/heads/main")
    assert refused.returncode != 0 and b"[binary] frame.jpg" in refused.stdout + refused.stderr
