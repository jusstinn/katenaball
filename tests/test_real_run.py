"""scripts/real_run.py without any match data and without training.

Every child script is a fake that prints what the real one prints, or a failure, so these tests check the order
of the steps, the stop at the first FAIL, the model-hash checks, the run report and what reaches the terminal.
The data bundle is a folder of empty files with the bundle's names. One test runs the real validator and the
real public-window check on a published baseline file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]

import games  # noqa: E402
import real_run as RR  # noqa: E402

EXPECTED = json.loads(RR.EXPECTED.read_text(encoding="utf-8"))["sha256"]
TEAM = "testteam"
# Decoys. The label shares are in the fake predict report, which is streamed to the terminal like the real
# one's, but must never reach the run report or the snippet. The other lines are the children's aggregates
# and exception messages, which must not reach the terminal at all.
SHARES = {"A": 0.398765, "D": 0.188889, "H": 0.412345, "X": 0.000123}
SHARE_DECOYS = ("0.398765", "0.188889", "0.412345", "0.000123", "label_share")
CHILD_DECOYS = ("secret_count", "SECRET-MESSAGE", "mtah2027", "label counts", "81234", "412 possessions",
                "touches/min", "jitter est", "n_players")

CHECK = """
    import pathlib, sys
    pathlib.Path(P["marker"]).write_text("ran", encoding="utf-8")
    print("\\n".join(P["lines"]), flush=True)
    if P["crash"]:
        raise ValueError(P["crash"])
    sys.exit(P["rc"])
"""
MAKE = """
    import hashlib, json, os, pathlib, sys, time
    pathlib.Path(P["marker"]).write_text("ran", encoding="utf-8")
    args = sys.argv[1:]
    assert args[:3] == ["all", "--workers", "2"], args
    out = pathlib.Path(args[args.index("--out") + 1])
    model = pathlib.Path(os.environ["KATENA_WORK"]) / "model"
    model.mkdir(parents=True, exist_ok=True)
    manifest = {"matches": ["m%d" % i for i in range(30)], "rows": 657133, "sha256": P["sha"]}
    (model / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print("[12:00:00] cache " + os.environ["KATENA_WORK"] + "; env {}", flush=True)
    print("[12:00:01] corpus {'matches': 30, 'up_to_date': 30}", flush=True)
    print("[12:00:02] worlds: 30 matches, 0 touch files built", flush=True)
    if P["train_line"]:
        print("[12:00:03] train " + json.dumps(manifest, sort_keys=True), flush=True)
    time.sleep(P["sleep"])
    if P["crash"]:
        raise RuntimeError(P["crash"])
    data = pathlib.Path(P["source"]).read_bytes() if P["source"] else b"frame,label\\n0,D\\n"
    out.write_bytes(data)
    rep = {"frames": 199319, "label_share_in_halves": P["shares"], "model_sha256": P["sha"]["bundle.pkl"],
           "submission": {"bytes": len(data), "path": str(out), "rows": 1, "sha256": hashlib.sha256(data).hexdigest()}}
    print(json.dumps(rep, indent=1, sort_keys=True), flush=True)
    print("[12:00:04] done in 4s", flush=True)
"""
TOUCH_LINE = ("target (real 25 Hz ball)                 rows  39728 cols  53   4.2s | ball visible 0.81 | "
              "touches/min  12.3 (attributed 0.91) | jitter est 0.052 m | median |D2| lag1 0.0123 m, lag5 0.0456 m | "
              "NaN share 0.02")
DEFAULT_LINES = {
    "check_target": ["Output holds no match values. Safe to paste.", "",
                     "PASS  one record per video frame, numbered from 1",
                     "PASS  reproduces published baseline_raw  0 frames differ", "secret_count: 4242", "",
                     "2 passed, 0 failed"],
    "check_target_layout": ["Output holds no match values. Safe to paste.", "", "tactical", "  n_players: 23",
                            "  PASS  katena.target.atd reads this layout as intended", "", "broadcast",
                            "  n_players: 23", "  PASS  katena.target.atd reads this layout as intended", ""],
    "check_touch_target": [TOUCH_LINE, "reference DFL-MAT-J03WMX clean          rows  29123 cols  53 | touches/min 9.9"],
    "validate_submission": ["OK    testteam.csv", "      199319 rows, frame 0..199318", "", "      label counts",
                            "        H    81234   40.8%", "", "      reports 412 possessions of mean 11.0 s", "",
                            "Valid. Note this checks the format only -- it says nothing about whether the labels "
                            "are right.", "Score it against the public ground truth with run_evaluation.py."],
    "check_public_window": ["valid_submission: PASS", "home_away_orientation: PASS", "frame_alignment: PASS",
                            "beats_published_baseline: PASS", "Log this look in docs/decision-log.md."],
}


class Box:
    """A sandbox: a fake bundle, cache and corpus, fake child scripts, and real_run pointed at them."""

    def __init__(self, tmp: Path, monkeypatch, capsys):
        self.tmp, self.mp, self.capsys = tmp, monkeypatch, capsys
        self.fakes = tmp / "fakes"
        self.fakes.mkdir()
        self.data = tmp / "bundle"
        for source, assets in (("tactical", ("atd_meta", "atd_raw", "smart_tagging", "homography")),
                               ("broadcast", ("atd_meta",))):
            paths = games.asset_paths(data_dir=str(self.data), source=source)
            for asset in assets:
                Path(paths[asset]).parent.mkdir(parents=True, exist_ok=True)
                Path(paths[asset]).write_bytes(b"")          # empty: the run checks existence only
        self.work = tmp / "work"
        self.work.mkdir()
        self.external = tmp / "external"
        for i in range(RR.N_EXTERNAL):
            p = self.external / "canonical" / ("idsse" if i < 7 else "metrica") / f"m{i}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        self.subs = tmp / "submissions"
        monkeypatch.setenv("MTAH_DATA_DIR", str(self.data))
        monkeypatch.setenv("KATENA_WORK", str(self.work))
        monkeypatch.setenv("KATENA_EXTERNAL", str(self.external))
        monkeypatch.setattr(RR, "SUBMISSIONS", self.subs)
        monkeypatch.setattr(RR, "git_info", lambda: {"commit": "c" * 40, "branch": "main", "dirty": 0, "on_main": True})
        monkeypatch.setattr(RR, "TOOLS", dict(RR.TOOLS))
        RR.TOOLS["check_kickoff_tags"] = self.fakes / "check_kickoff_tags.py"     # absent unless a test adds it
        for key, lines in DEFAULT_LINES.items():
            self.check(key, lines)
        self.make()

    def _write(self, key: str, body: str, **params) -> None:
        params["marker"] = str(self.fakes / f"{key}.ran")
        path = self.fakes / f"{key}.py"
        path.write_text(f"P = {params!r}\n" + textwrap.dedent(body), encoding="utf-8")
        RR.TOOLS[key] = path

    def check(self, key: str, lines, rc: int = 0, crash: str | None = None) -> None:
        self._write(key, CHECK, lines=list(lines), rc=rc, crash=crash)

    def make(self, sha: dict | None = None, train_line: bool = True, sleep: float = 0, crash: str | None = None,
             source: Path | None = None) -> None:
        self._write("make_submission", MAKE, sha=dict(EXPECTED, **(sha or {})), train_line=train_line, sleep=sleep,
                    crash=crash, source=str(source) if source else None, shares=SHARES)

    def ran(self, key: str) -> bool:
        return (self.fakes / f"{key}.ran").exists()

    def run(self, *extra: str) -> tuple[int, str, dict]:
        rc = RR.main(["--team", TEAM, *extra])
        out = self.capsys.readouterr().out
        path = self.subs / f"{TEAM}.run.json"
        return rc, out, json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@pytest.fixture
def box(tmp_path, monkeypatch, capsys):
    return Box(tmp_path, monkeypatch, capsys)


def snippet_of(out: str) -> str:
    """The pasteable lines: after the heading line, up to the end marker."""
    return out.split("== 6. Decision-log snippet", 1)[1].split("\n", 1)[1].split("== end of snippet", 1)[0]


def keys(obj) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [k for v in obj.values() for k in keys(v)]
    if isinstance(obj, list):
        return [k for v in obj for k in keys(v)]
    return []


# ---------------------------------------------------------------------------------------------

def test_a_full_run_passes_and_reports_commit_results_and_hashes(box):
    rc, out, rep = box.run()
    assert rc == 0, out
    csv = box.subs / f"{TEAM}.csv"
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert rep["result"] == "PASS" and rep["failed_step"] is None
    assert rep["commit"] == "c" * 40 and rep["tree_clean"] is True
    assert [(c["script"], c["result"]) for c in rep["checks"]] == [
        ("check_target.py", "PASS"), ("check_target_layout.py", "PASS"), ("check_touch_target.py", "PASS"),
        ("check_kickoff_tags.py", "SKIP")]
    assert rep["model_sha256"] == EXPECTED and set(rep["hash_check"].values()) == {"same"}
    assert rep["validate_submission"]["result"] == "PASS" and rep["public_window"]["result"] == "PASS"
    assert rep["csv"]["sha256"] == sha
    assert all(w.startswith(("libraries other than the pins", "python ")) for w in rep["warnings"])  # unpinned venv
    for step in ("environment", "checks", "make_submission", "validate_submission", "public_window", "total"):
        assert rep["seconds"][step] >= 0
    snip = snippet_of(out)
    assert f"Commit `{'c' * 40}` (tree clean). Result: **PASS**." in snip
    assert "`check_target.py` 2 PASS, 0 FAIL" in snip and "`check_kickoff_tags.py` not in this checkout" in snip
    assert all(h in snip for h in EXPECTED.values()) and sha in snip
    assert "`validate_submission.py`: Valid." in snip
    assert "beats_published_baseline: PASS" in snip
    assert "PASS  [tactical] katena.target.atd reads this layout as intended" in out
    assert "shape (39728, 53)" in out
    assert "PASS  every step passed" in out


def test_the_run_report_and_the_snippet_hold_no_label_shares(box):
    rc, out, rep = box.run()
    assert rc == 0, out
    assert "0.412345" in out                                  # streamed, as make_submission.py prints it
    text = (box.subs / f"{TEAM}.run.json").read_text(encoding="utf-8")
    snip = snippet_of(out)
    for decoy in SHARE_DECOYS:
        assert decoy not in text, decoy
        assert decoy not in snip, decoy
    assert "share" not in snip.lower()
    assert not [k for k in keys(rep) if "share" in k.lower() or "label" in k.lower()]
    assert "frames" not in keys(rep)


def test_children_print_only_verdicts_hashes_and_exception_types(box):
    box.check("check_target_layout", DEFAULT_LINES["check_target_layout"][:4]
              + ["  secret_count: 4242", "  PASS  katena.target.atd reads this layout as intended"])
    rc, out, _ = box.run()
    assert rc == 0, out
    for decoy in CHILD_DECOYS:
        assert decoy not in out, decoy
    assert "OK    testteam.csv" in out and "Valid." in out


def test_a_fail_line_stops_the_run_at_that_check(box):
    box.check("check_target", ["PASS  one record per video frame, numbered from 1",
                               "FAIL  slot count matches the published count", "secret_count: 4242",
                               "1 passed, 1 failed"], rc=1)
    rc, out, rep = box.run()
    assert rc == 1
    assert rep["result"] == "FAIL" and rep["failed_step"] == "checks"
    assert rep["checks"][0]["lines"] == ["PASS  one record per video frame, numbered from 1",
                                         "FAIL  slot count matches the published count"]
    assert len(rep["checks"]) == 1
    assert not box.ran("check_target_layout") and not box.ran("make_submission")
    assert not (box.subs / f"{TEAM}.csv").exists() and rep["csv"]["sha256"] is None
    assert "FAIL  slot count matches the published count" in snippet_of(out)
    assert "secret_count" not in out


def test_an_exception_line_stops_the_run(box):
    box.check("check_target_layout", ["tactical", "  n_players: 23", "  raised KeyError", "", "broadcast",
                                      "  PASS  katena.target.atd reads this layout as intended"], rc=1)
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "checks"
    assert "FAIL  [tactical] raised KeyError" in out
    assert not box.ran("check_touch_target") and "n_players" not in out


def test_a_traceback_in_a_check_is_reduced_to_its_type(box):
    box.check("check_target", ["PASS  one record per video frame, numbered from 1"],
              crash="SECRET-MESSAGE D:/mtah2027/tactical/BAR-ATH_tactical_atd.txt")
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "checks"
    assert "raised ValueError" in out
    assert "SECRET-MESSAGE" not in out and "mtah2027" not in out
    assert "SECRET-MESSAGE" not in json.dumps(rep)


def test_a_check_that_prints_no_verdict_fails(box):
    box.check("check_target", ["Output holds no match values. Safe to paste.", "", "0 passed, 0 failed"])
    rc, out, rep = box.run()
    assert rc == 1 and rep["checks"][0]["lines"] == ["FAIL  printed no PASS line"]


def test_the_touch_check_only_warns_when_the_external_reference_raises(box):
    # The reference worlds come from an external match, so their failure is an alarm, not a stop.
    box.check("check_touch_target", [TOUCH_LINE, "reference: skipped (FileNotFoundError; needs the external "
                                                 "corpus, KATENA_EXTERNAL)"])
    rc, out, rep = box.run()
    assert rc == 0, out
    touch = next(c for c in rep["checks"] if c["script"] == "check_touch_target.py")
    assert touch["result"] == "PASS"
    assert "WARN  reference: skipped (FileNotFoundError" in out and "touches/min" not in out
    assert any(w.startswith("check_touch_target.py: reference: skipped") for w in rep["warnings"])
    assert box.ran("make_submission")


def test_the_touch_check_fails_when_the_real_match_raises(box):
    box.check("check_touch_target", ["target: failed (ValueError)"])       # check_touch_target.py's own wording
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "checks"
    assert not box.ran("make_submission")


def test_a_kickoff_warn_line_is_shown_and_does_not_stop_the_run(box):
    # check_kickoff_tags.py's own wording: an early tag is safe, so it warns and exits 0
    box.check("check_kickoff_tags", [
        "WARN  1st half: the tag-derived kickoff lies within 2 s of the published kickoff  (more than 2 s away; "
        "the rule sets fewer frames to D)",
        "PASS  1st half: the tag-derived kickoff is not more than 2 s after the published kickoff",
        "PASS  2nd half: the tag-derived kickoff lies within 2 s of the published kickoff",
        "PASS  2nd half: the tag-derived kickoff is not more than 2 s after the published kickoff"])
    rc, out, rep = box.run()
    assert rc == 0, out
    assert "WARN  1st half: the tag-derived kickoff lies within 2 s" in out
    assert any(w.startswith("check_kickoff_tags.py: 1st half") for w in rep["warnings"])
    assert box.ran("make_submission")


@pytest.mark.parametrize("verdict, rc_expected", [("PASS", 0), ("FAIL", 1)])
def test_the_kickoff_check_runs_when_this_checkout_has_it(box, verdict, rc_expected):
    box.check("check_kickoff_tags", ["half 1 kickoff within 2 s of FAQ.md: " + verdict,
                                     "half 2 kickoff within 2 s of FAQ.md: PASS"], rc=rc_expected)
    rc, out, rep = box.run()
    assert rc == rc_expected, out
    assert rep["checks"][3]["script"] == "check_kickoff_tags.py" and rep["checks"][3]["result"] == verdict
    assert f"{verdict}  half 1 kickoff within 2 s of FAQ.md" in out
    assert box.ran("make_submission") is (verdict == "PASS")


def test_another_gbdt_hash_stops_make_submission_before_predict(box):
    box.make(sha={"gbdt": "0" * 64}, sleep=30)
    t = time.time()
    rc, out, rep = box.run()
    assert rc == 1 and time.time() - t < 20                 # stopped, not waited for
    assert rep["failed_step"] == "make_submission" and rep["hash_check"] == {"gbdt": "differs"}
    assert rep["make_submission"]["stopped_after_train"] is True
    assert "the training data changed" in rep["stop_reason"]
    assert not (box.subs / f"{TEAM}.csv").exists() and not box.ran("validate_submission")
    assert "gbdt `" + "0" * 64 + "` (differs)" in snippet_of(out)


def test_another_gbdt_hash_in_the_manifest_alone_is_fatal_too(box):
    box.make(sha={"gbdt": "0" * 64}, train_line=False)
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "make_submission"
    assert rep["hash_check"]["gbdt"] == "differs" and not box.ran("validate_submission")


def test_another_tcn_or_bundle_hash_alone_is_a_warning(box):
    box.make(sha={"tcn": "1" * 64, "bundle.pkl": "2" * 64})
    rc, out, rep = box.run()
    assert rc == 0, out
    assert rep["hash_check"] == {"gbdt": "same", "tcn": "differs", "bundle.pkl": "differs"}
    assert any("torch's CPU kernels changed" in w for w in rep["warnings"])
    assert "WARN  tcn" in out and rep["model_sha256"]["tcn"] == "1" * 64


def test_a_traceback_from_make_submission_is_reduced_to_its_type(box):
    box.make(crash="SECRET-MESSAGE D:/mtah2027/tactical/BAR-ATH_tactical_atd.txt")
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "make_submission"
    assert "[12:00:03] train" in out                         # streamed up to the crash
    assert "raised RuntimeError" in out and "SECRET-MESSAGE" not in out and "mtah2027" not in out


def test_a_validator_fail_stops_before_the_public_window(box):
    box.check("validate_submission", ["FAIL  testteam.csv", "      row count 5 != 199319", "",
                                      "See SUBMISSION.md for the contract."], rc=1)
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "validate_submission"
    assert "row count 5 != 199319" in out and not box.ran("check_public_window")
    assert "`validate_submission.py`: FAIL." in snippet_of(out)


def test_a_public_window_fail_stops_and_is_logged_with_the_files_hash(box):
    box.check("check_public_window", ["valid_submission: PASS", "home_away_orientation: PASS",
                                      "frame_alignment: FAIL", "beats_published_baseline: PASS"], rc=1)
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "public_window"
    sha = hashlib.sha256((box.subs / f"{TEAM}.csv").read_bytes()).hexdigest()
    snip = snippet_of(out)
    assert "frame_alignment: FAIL" in snip and sha in snip and rep["csv"]["sha256"] == sha


def test_every_environment_problem_is_named_before_anything_runs(box, monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR")
    monkeypatch.setenv("KATENA_WORK", str(box.tmp / "nowhere"))
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "environment"
    assert "MTAH_DATA_DIR is not set" in rep["stop_reason"] and "KATENA_WORK does not exist" in rep["stop_reason"]
    assert not box.ran("check_target") and not box.ran("make_submission")


def test_a_missing_bundle_file_is_named_by_asset(box):
    Path(games.asset_paths(data_dir=str(box.data))["atd_raw"]).unlink()
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "environment"
    assert "FAIL  bundle files missing: tactical atd_raw" in out
    assert rep["environment"]["bundle"]["needed_missing"] == ["tactical atd_raw"]


def test_skip_checks_and_the_work_and_external_flags(box, monkeypatch):
    Path(games.asset_paths(data_dir=str(box.data), source="broadcast")["atd_meta"]).unlink()  # checks only
    monkeypatch.setenv("KATENA_WORK", str(box.tmp / "nowhere"))
    monkeypatch.delenv("KATENA_EXTERNAL")
    rc, out, rep = box.run("--skip-checks", "--work", str(box.work), "--external", str(box.external))
    assert rc == 0, out
    assert rep["skip_checks"] is True and rep["checks"] == [] and not box.ran("check_target")
    assert rep["environment"]["KATENA_WORK"] == str(box.work.resolve())
    assert "skipped (`--skip-checks`)" in snippet_of(out)
    assert (box.work / "model" / "manifest.json").exists()    # the child got --work as KATENA_WORK


def test_the_real_validator_and_public_window_check_on_a_published_file(box):
    """The organisers' smooth50 baseline as the submission: the real validator says Valid, its label counts stay
    off the terminal, and the real public-window check fails "beats the published baseline" (it is that file)."""
    box.make(source=REPO / "evaluation" / "results" / "baseline_smooth50.csv")
    RR.TOOLS["validate_submission"] = REPO / "evaluation" / "validate_submission.py"
    RR.TOOLS["check_public_window"] = REPO / "scripts" / "check_public_window.py"
    rc, out, rep = box.run()
    assert rc == 1 and rep["failed_step"] == "public_window"
    assert rep["validate_submission"]["result"] == "PASS"
    assert rep["validate_submission"]["lines"][0] == f"OK    {TEAM}.csv"
    assert "label counts" not in out and "possessions of mean" not in out
    assert not re.search(r"^\s+[HADX]\s+\d+\s+[\d.]+%\s*$", out, re.M)       # the validator's label-count rows
    assert "beats_published_baseline: FAIL" in rep["public_window"]["lines"]
    assert "valid_submission: PASS" in rep["public_window"]["lines"]


def test_the_team_name_must_be_a_plain_file_name(capsys):
    for bad in ("../x", "a/b", "a\\b", "", " x", "x.", "a:b"):
        with pytest.raises(SystemExit) as e:
            RR.main(["--team", bad])
        assert e.value.code == 2
    capsys.readouterr()


# ---------------------------------------------------------------------------------------------

def test_the_traceback_filter_keeps_only_the_type():
    text = textwrap.dedent("""\
        [12:00:00] worlds: start
        Traceback (most recent call last):
          File "x.py", line 1, in <module>
            raise KeyError('D:/mtah2027/tactical/BAR-ATH_tactical_atd.txt')
        KeyError: 'D:/mtah2027/tactical/BAR-ATH_tactical_atd.txt'

        During handling of the above exception, another exception occurred:

        Traceback (most recent call last):
          File "y.py", line 2, in <module>
        concurrent.futures.process._RemoteTraceback: SECRET-MESSAGE
        multi line SECRET-MESSAGE
        [12:00:05] carried on
        Traceback (most recent call last):
        ValueError""")
    f = RR.TracebackFilter()
    out = [x for line in text.splitlines() for x in f.feed(line)] + f.close()
    assert out[0] == "[12:00:00] worlds: start"
    assert out[1].startswith("raised _RemoteTraceback (")
    assert out[2] == "[12:00:05] carried on"
    assert out[3].startswith("raised ValueError (") and len(out) == 4
    assert f.raised == ["_RemoteTraceback", "ValueError"]
    assert "SECRET" not in "\n".join(out) and "mtah2027" not in "\n".join(out)


def test_constants_and_the_train_line_agree_with_the_pipeline(capsys):
    from katena import pipeline as PL
    assert (RR.MODEL_DIR, RR.MANIFEST, RR.BUNDLE) == (PL.MODEL_DIR, PL.MANIFEST, PL.BUNDLE)
    PL.log("train " + json.dumps({"seconds": {"gbdt": 1.0}, "sha256": {"gbdt": "ab", "tcn": "cd"}}, sort_keys=True))
    line = capsys.readouterr().out.rstrip("\n")
    w = RR.MakeWatch(expected_gbdt="ab")
    assert w.feed(line) is True and w.train_gbdt == "ab"
    assert RR.MakeWatch(expected_gbdt="ef").feed(line) is False


def test_the_expected_hashes_are_the_documented_ones():
    real_run_md = (REPO / "docs" / "real-run.md").read_text(encoding="utf-8")
    log = (REPO / "docs" / "decision-log.md").read_text(encoding="utf-8")
    assert sorted(EXPECTED) == sorted(RR.HASH_KEYS)
    for key, label in (("gbdt", "gbdt"), ("tcn", "tcn"), ("bundle.pkl", "bundle")):
        h = EXPECTED[key]
        assert re.fullmatch(r"[0-9a-f]{64}", h)
        assert h in log, key                                  # the J3 Windows table
        assert h.startswith(re.search(rf"\b{label} `([0-9a-f]+)", real_run_md).group(1)), key
