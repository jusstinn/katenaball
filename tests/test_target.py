"""Tests for the competition-match loaders, on synthetic files in the documented format.

The official starter/baseline.py acts as the oracle: if our loader and its parser agree on a noisy
synthetic file, then the published baseline files can verify the loader on the real one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "starter"):
    sys.path.insert(0, str(path))

from baseline import nearest_player_labels, smooth, team_of_each_track  # noqa: E402

from katena.target.atd import nearest_team, read_raw, read_slot_teams  # noqa: E402
from katena.target.tagging import read_tags  # noqa: E402

PITCH = (105.3, 68.0)
N_FRAMES = 400
N_SLOTS = 8

META = """<?xml version="1.0"?>
<Metadata>
  <Teams>
    <Team id="T1"><Name>Home</Name></Team>
    <Team id="T2"><Name>Away</Name></Team>
  </Teams>
  <Players>
    {players}
  </Players>
</Metadata>
"""

TAGS = """<?xml version="1.0"?>
<file><ALL_INSTANCES>
  <instance><start>10.0</start><end>12.5</end><code>SET PIECES</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Half</group><text>1st Half</text></label>
    <label><group>Type</group><text>Throw-in</text></label></instance>
  <instance><start>20.04</start><end>30.0</end><code>BUILD UP</code>
    <label><group>Team</group><text>Barcelona</text></label>
    <label><group>Half</group><text>2nd Half</text></label></instance>
  <instance><start>40.0</start><end>41.0</end><code>LONG BALL</code>
    <label><group>Team</group><text>Atletic Club</text></label></instance>
</ALL_INSTANCES></file>
"""


def _value(v: float) -> str:
    return "NaN" if np.isnan(v) else repr(float(v))


@pytest.fixture()
def atd(tmp_path):
    """A noisy tracking file. Ball missing often, slots missing often, one frame absent."""
    rng = np.random.default_rng(7)
    teams = ["T1" if i < N_SLOTS // 2 else "T2" for i in range(N_SLOTS)]
    players = "\n".join(f'<Player id="P{i}" teamId="{t}"/>' for i, t in enumerate(teams))
    meta = tmp_path / "atd.xml"
    meta.write_text(META.format(players=players))

    xy = rng.random((N_FRAMES, N_SLOTS, 2))
    speed = rng.random((N_FRAMES, N_SLOTS)) * 8
    drop = rng.random((N_FRAMES, N_SLOTS)) < 0.3
    xy[drop] = np.nan
    speed[drop] = np.nan
    ball = rng.random((N_FRAMES, 3))
    ball[rng.random(N_FRAMES) < 0.4] = np.nan
    xy[50] = np.nan  # a frame with a ball and nobody around it
    speed[50] = np.nan
    ball[50] = [0.5, 0.5, 1.0]

    lines = []
    for f in range(N_FRAMES):
        groups = ";".join(f"{_value(x)},{_value(y)},{_value(s)}"
                          for (x, y), s in zip(xy[f], speed[f]))
        lines.append(f"{f + 1}:{groups}:{_value(ball[f, 0])},{_value(ball[f, 1])},{_value(ball[f, 2])}")
    raw = tmp_path / "atd.txt"
    raw.write_text("\n".join(lines) + "\n")
    return {"raw": raw, "meta": meta, "xy": xy, "speed": speed, "ball": ball, "teams": teams}


def test_slot_teams_match_baseline_reader(atd):
    teams = read_slot_teams(atd["meta"])
    assert teams.tolist() == team_of_each_track(str(atd["meta"])).tolist()
    assert teams.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_reads_positions_in_metres_with_frame_offset(atd):
    P, speed, ball, report = read_raw(atd["raw"], N_FRAMES, N_SLOTS, PITCH, dtype=np.float64)
    assert report["contiguous_from_one"]
    assert report["n_malformed"] == 0 and report["n_duplicate"] == 0
    # Row i of our arrays is frame_id i + 1 of the file.
    np.testing.assert_array_equal(P[..., 0], atd["xy"][..., 0] * PITCH[0])
    np.testing.assert_array_equal(P[..., 1], atd["xy"][..., 1] * PITCH[1])
    np.testing.assert_array_equal(speed, atd["speed"])
    np.testing.assert_array_equal(ball[:, 0], atd["ball"][:, 0] * PITCH[0])


def test_nearest_team_matches_official_baseline(atd):
    team = read_slot_teams(atd["meta"])
    official = nearest_player_labels(str(atd["raw"]), team_of_each_track(str(atd["meta"])),
                                     N_FRAMES, PITCH)
    P, _, ball, _ = read_raw(atd["raw"], N_FRAMES, N_SLOTS, PITCH, dtype=np.float64)
    ours = nearest_team(P, ball, team)
    np.testing.assert_array_equal(ours, official)
    assert ours[50] == -1  # ball visible, no player: no answer
    for window in (0, 25, 50, 125):
        np.testing.assert_array_equal(smooth(ours, window), smooth(official, window))


def test_malformed_and_absent_lines_are_counted_not_fatal(atd, tmp_path):
    lines = atd["raw"].read_text().splitlines()
    lines[3] = "not a record"
    lines[4] = "5:1,2,3:0,0,0"            # wrong number of player groups
    lines[5] = lines[5].replace("1", "x", 3)
    del lines[9]
    broken = tmp_path / "broken.txt"
    broken.write_text("\n".join(lines) + "\n")
    _, _, _, report = read_raw(broken, N_FRAMES, N_SLOTS, PITCH)
    assert report["n_malformed"] >= 2
    assert not report["contiguous_from_one"]


def test_tracking_values_do_not_appear_in_errors(tmp_path):
    secret = "0.123456789"
    raw = tmp_path / "atd.txt"
    raw.write_text(f"1:{secret},nonsense,1:{secret},0,0\n")
    _, _, _, report = read_raw(raw, 2, 1, PITCH)
    assert report["n_malformed"] == 1
    assert secret not in str(report)


def test_read_tags_maps_frames_and_sides(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(TAGS)
    tags = read_tags(path)
    assert [t.code for t in tags] == ["set pieces", "build up", "long ball"]
    assert (tags[0].start_frame, tags[0].end_frame) == (250, 312)
    assert tags[1].start_frame == 501
    assert [t.team for t in tags] == [None, 0, 1]
    assert [t.half for t in tags] == [1, 2, None]
    assert tags[0].kind == "throw-in" and tags[1].kind is None
