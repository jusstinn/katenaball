"""W4a: the kickoff rule (``katena.pipeline.apply_kickoff_rule``) and its human check (``scripts/check_kickoff_tags.py``),
on synthetic Smart Tagging files in the documented SportsCode format. No match data.

The end-to-end run through ``stage_predict`` on a synthetic ATD bundle is in ``tests/test_pipeline_atd.py``, which
has the trained synthetic model."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from synth import tagging_xml, write_tagging

from katena import pipeline as PL
from katena.target.tagging import read_tags
from katena.types import D, X

ROOT = Path(__file__).resolve().parents[1]
FPS = 25.0
BLOCKS = [(1, 1000, 5000), (2, 8000, 20000)]            # a small synthetic timeline, not the match's
N = 21000


def codes_fixture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.full(N, X, dtype=np.int8)
    for _, a, b in BLOCKS:
        out[a:b + 1] = rng.integers(0, 3, b - a + 1)
    return out


def ko(frame: float, half: str, kind: str = "Kick Off (Start)", team: str = "Barcelona", length: int = 120):
    return (frame / FPS, (frame + length) / FPS, "SET PIECES", {"Team": team, "Half": half, "Type": kind})


def tags_of(tmp_path, instances, name="tags.xml"):
    p = tmp_path / name
    p.write_text(tagging_xml(instances), encoding="utf-8", newline="\n")
    return read_tags(p, fps=FPS)


def test_frames_before_the_tagged_kickoff_become_dead(tmp_path):
    codes = codes_fixture()
    tags = tags_of(tmp_path, [
        ko(999, "1st Half"),                                         # starts before its block: nothing to do
        ko(3000, "1st Half", "Kick Off (After Goal)"),              # ignored
        ko(8100, "2nd Half", "Kick Off (After Goal)", "Atletic Club"),
        ko(9500, "2nd Half"),
        (100.0, 110.0, "BUILD UP", {"Team": "Barcelona", "Half": "2nd Half"}),
    ])
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags, FPS)
    assert (out[8000:9500] == D).all()
    assert np.array_equal(out[:8000], codes[:8000]) and np.array_equal(out[9500:], codes[9500:])
    assert rep["half1"] == {"applied": False, "frames_changed": 0, "status": "before_block"}
    assert rep["half2"] == {"applied": True, "frames_changed": int((codes[8000:9500] != D).sum()), "status": "applied"}
    assert rep["bound_s"] == PL.KICKOFF_MAX_S
    assert rep["half2"]["frames_changed"] > 0 and codes is not out


def test_type_and_half_match_case_insensitively(tmp_path):
    tags = tags_of(tmp_path, [ko(1200, " 1ST HALF ", " KICK OFF (START) "), ko(8800, "2nd half", "kick off (start)")])
    out, rep = PL.apply_kickoff_rule(codes_fixture(1), BLOCKS, tags, FPS)
    assert rep["half1"]["applied"] and rep["half2"]["applied"]
    assert (out[1000:1200] == D).all() and (out[8000:8800] == D).all()


def test_start_time_maps_with_round_seconds_times_25(tmp_path):
    tags = tags_of(tmp_path, [(8800.75 / FPS, 9000 / FPS, "SET PIECES", {"Half": "2nd Half", "Type": "Kick Off (Start)"})])
    assert tags[0].start_frame == round(8800.75 / FPS * 25) == 8801
    out, _ = PL.apply_kickoff_rule(codes_fixture(), BLOCKS, tags, FPS)
    assert (out[8000:8801] == D).all()


@pytest.mark.parametrize("frame, status", [
    (7999, "before_block"),                                            # in the gap before the block
    (8000 + int(PL.KICKOFF_MAX_S * FPS) + 1, "beyond_bound"),          # more than the bound into the block
    (20001, "after_block"),
])
def test_out_of_bounds_tags_change_nothing(tmp_path, frame, status):
    codes = codes_fixture(2)
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags_of(tmp_path, [ko(frame, "2nd Half")]), FPS)
    assert np.array_equal(out, codes)
    assert rep["half2"] == {"applied": False, "frames_changed": 0, "status": status}
    assert rep["half1"]["status"] == "no_tag"


def test_bound_edges(tmp_path):
    codes = codes_fixture(3)
    at_bound = 8000 + int(PL.KICKOFF_MAX_S * FPS)
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags_of(tmp_path, [ko(at_bound, "2nd Half"), ko(1000, "1st Half")]),
                                     FPS)
    assert rep["half2"]["applied"] and (out[8000:at_bound] == D).all()
    assert rep["half1"] == {"applied": True, "frames_changed": 0, "status": "applied"}    # kickoff at the block start
    assert np.array_equal(out[at_bound:], codes[at_bound:])


def test_no_tag_several_tags_and_no_file_change_nothing(tmp_path):
    codes = codes_fixture(4)
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags_of(tmp_path, [ko(1500, "1st Half"), ko(1600, "1st Half")]), FPS)
    assert np.array_equal(out, codes)
    assert rep["half1"]["status"] == "several_tags" and rep["half2"]["status"] == "no_tag"
    assert not rep["half1"]["applied"] and not rep["half2"]["applied"]
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, None, FPS)
    assert np.array_equal(out, codes) and rep["half1"]["status"] == rep["half2"]["status"] == "no_tagging_file"
    # a tag with no Half cannot be placed, so it is ignored
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags_of(tmp_path, [(60.0, 61.0, "SET PIECES",
                                                                        {"Type": "Kick Off (Start)"})]), FPS)
    assert np.array_equal(out, codes) and rep["half1"]["status"] == "no_tag"


def test_reading_the_bundle_tagging_file(tmp_path):
    tags, why = PL.read_target_tags(tmp_path / "empty", FPS)
    assert tags is None and why == "no_tagging_file"
    write_tagging(tmp_path / "ok", [ko(9000, "2nd Half")])
    tags, why = PL.read_target_tags(tmp_path / "ok", FPS)
    assert why is None and [t.start_frame for t in tags] == [9000]
    path = write_tagging(tmp_path / "bad", [ko(9000, "2nd Half")])
    path.write_bytes(path.read_bytes()[:-30])                          # truncated: not XML any more
    tags, why = PL.read_target_tags(tmp_path / "bad", FPS)
    assert tags is None and why == "unreadable_ParseError"
    codes = codes_fixture()
    out, rep = PL.apply_kickoff_rule(codes, BLOCKS, tags, FPS, missing=why)
    assert np.array_equal(out, codes) and rep["half2"]["status"] == "unreadable_ParseError"


# --- the human check ----------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))
import check_kickoff_tags as CK  # noqa: E402


def _check(root, instances, capsys, write=True):
    CK.RESULTS.update({"pass": 0, "fail": 0})
    if write:
        write_tagging(root, instances)
    code = CK.main(["--data-dir", str(root)])
    out = capsys.readouterr().out
    assert not re.search(r"\d{3,}", out), "the check must not print frames or times"
    return code, out


def published(half: int, shift_s: float = 0.0, **kw):
    return ko(CK.PUBLISHED_KICKOFF[half] + shift_s * FPS, f"{'1st' if half == 1 else '2nd'} Half", **kw)


def test_check_passes_when_both_tags_sit_on_the_published_kickoffs(tmp_path, capsys):
    code, out = _check(tmp_path, [published(1, -1.5), published(2, 1.9),
                                  published(2, 30, kind="Kick Off (After Goal)")], capsys)
    lines = [x for x in out.splitlines() if x.startswith(("PASS", "FAIL"))]
    assert code == 0 and len(lines) == 4 and all(x.startswith("PASS") for x in lines), out


def test_check_fails_a_late_or_far_or_missing_tag(tmp_path, capsys):
    code, out = _check(tmp_path / "a", [published(1, -3.0), published(2, 2.5)], capsys)
    assert code == 1
    first = [x for x in out.splitlines() if "1st half" in x]
    second = [x for x in out.splitlines() if "2nd half" in x]
    assert first[0].startswith("FAIL") and first[1].startswith("PASS")           # early: far, but not late
    assert second[0].startswith("FAIL") and second[1].startswith("FAIL")         # late
    code, out = _check(tmp_path / "b", [published(1), published(1, 1.0)], capsys)
    assert code == 1 and "several Kick Off (Start) tags" in out and "no Kick Off (Start) tag" in out
    code, out = _check(tmp_path / "c", [], capsys, write=False)
    assert code == 1 and "raised OSError" in out
