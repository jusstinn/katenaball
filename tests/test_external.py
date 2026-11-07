"""Invariants of the external-corpus loaders, on tiny synthetic fixtures.

Real-data checks at the bottom run only when ``external_data/canonical`` has been built.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from synth import L, W, make_match

import possession_metrics as pm
from katena.external import idsse, metrica, skillcorner
from katena.external.common import (derive_speed, infer_attack_dir, infer_attack_dir_gk,
                                    nearest_team_agreement, validate_canonical)
from katena.types import A, D, H, X, Match, load_match, save_match

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


# --- Match container -----------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    m = make_match()
    m.extras["flag"] = np.arange(m.n, dtype=np.int8)
    save_match(m, tmp_path / "a.npz")
    r = load_match(tmp_path / "a.npz")
    for k in ("P", "speed", "ball"):
        assert getattr(r, k).dtype == np.float32
        np.testing.assert_array_equal(getattr(m, k), getattr(r, k))
    np.testing.assert_array_equal(m.y, r.y)
    assert r.y.dtype == np.int8 and r.team.dtype == np.int8
    assert r.half_blocks == m.half_blocks and r.attack_dir == m.attack_dir
    assert (r.pitch, r.fps, r.name, r.source) == (m.pitch, m.fps, m.name, m.source)
    np.testing.assert_array_equal(r.extras["flag"], m.extras["flag"])


def test_target_match_without_labels(tmp_path):
    m = make_match()
    m.y = None
    save_match(m, tmp_path / "t.npz")
    assert load_match(tmp_path / "t.npz").y is None


def test_synthetic_passes_validation():
    validate_canonical(make_match())


def test_inverted_team_mapping_is_caught():
    m = make_match(invert_labels=True)
    assert nearest_team_agreement(m)["agree"] < 0.1
    with pytest.raises(AssertionError, match="team mapping"):
        validate_canonical(m)


def test_labels_outside_blocks_must_be_x():
    m = make_match()
    m.y[0] = H
    with pytest.raises(AssertionError):
        validate_canonical(m)


def test_unit_error_in_speed_is_caught():
    m = make_match()
    m.speed[300, 0] = 40.0          # km/h passed off as m/s
    with pytest.raises(AssertionError, match="speed"):
        validate_canonical(m)


# --- helpers --------------------------------------------------------------------------------

def test_derive_speed_constant_velocity():
    P = np.zeros((10, 2, 2), np.float32)
    P[:, 0, 0] = np.arange(10) * 0.2            # 0.2 m/frame at 25 fps = 5 m/s
    P[3, 1] = np.nan
    v = derive_speed(P, 25.0)
    np.testing.assert_allclose(v[1:-1, 0], 5.0, rtol=1e-5)
    assert np.isnan(v[0, 0]) and np.isnan(v[-1, 0]) and np.isnan(v[[2, 4], 1]).all() and v[3, 1] == 0


def test_attack_direction_from_positions():
    m = make_match()
    assert infer_attack_dir(m.P, m.team, 100, 799, L) == +1
    assert infer_attack_dir(m.P, m.team, 1100, 1799, L) == -1
    assert infer_attack_dir_gk(m.P, m.team, 100, 799, L) == +1
    P = m.P.copy()
    P[..., 0] = L / 2                                  # both teams on the centre line: ambiguous
    assert infer_attack_dir(P, m.team, 100, 799, L) == 0


# --- Metrica event-derived labels -----------------------------------------------------------

def _ev(start, end, team, typ, sub="", period=1, order=[0]):
    order[0] += 1
    return metrica.Event(period, start, end, team, typ, sub, order[0])


def test_metrica_derive_labels():
    ev = [
        _ev(11, 11, 0, "SET PIECE", "KICK OFF"),
        _ev(11, 20, 0, "PASS"),
        _ev(30, 35, 1, "RECOVERY"),            # possession flips to away
        _ev(50, 50, 1, "BALL OUT"),            # dead until a restart
        _ev(55, 56, 0, "PASS"),                # ignored while dead
        _ev(70, 70, 0, "SET PIECE", "THROW IN"),
        _ev(90, 90, 1, "SHOT", "ON TARGET GOAL"),
        _ev(110, 110, 0, "SET PIECE", "KICK OFF"),
        _ev(120, 120, 0, "CARRY"),
    ]
    y, windows, stats = metrica.derive_labels(ev, 200)
    assert windows == [(1, 10, 119)]
    assert (y[:10] == X).all() and (y[120:] == X).all()
    assert (y[10:29] == H).all()               # frames are 1-based in events
    assert (y[29:49] == A).all()
    assert (y[49:69] == D).all()               # BALL OUT .. throw-in
    assert (y[69:89] == H).all()
    assert (y[89:109] == D).all()              # goal: dead until kick-off
    assert y[109] == H
    assert stats["ignored_while_dead"] == 1


def test_metrica_foul_and_goal_kick_restart():
    ev = [_ev(1, 1, 0, "SET PIECE", "KICK OFF"), _ev(10, 10, 1, "FAULT RECEIVED"),
          _ev(40, 40, 0, "PASS", "GOAL KICK"), _ev(60, 60, 0, "PASS")]
    y, _, _ = metrica.derive_labels(ev, 80)
    assert (y[9:39] == D).all() and (y[39:60] == H).all()


# --- sandwich merge (profile script) --------------------------------------------------------

def test_profile_sandwich_and_official_spells():
    import profile_external as pe
    fps = 10.0
    y = np.array([3] * 5 + [0] * 30 + [1] * 5 + [0] * 30 + [2] * 10 + [1] * 20 + [3] * 5, np.int8)
    runs = pm.valid_runs(y)
    assert len(pm.possession_spells(y, runs)) == 4
    y2 = pe.sandwich(y, runs, fps, 1.0)                 # the 0.5 s A flip is inside H, T = 1 s
    sp = pm.possession_spells(y2, runs)
    assert [(s[2], s[1] - s[0]) for s in sp] == [(0, 65), (1, 20)]
    assert (pe.sandwich(y, runs, fps, 0.3) == y).all()  # flip longer than T: unchanged
    # synthetic match: per half H A D A H D H -> 5 spells, 2 direct flips, 2 D between same team
    st = pe.match_stats(make_match())
    assert st["n_spells"] == 10 and st["via_d_change"] == 0 and st["via_d_same"] == 4 \
        and st["direct"] == 4
    assert st["dead_share"] == pytest.approx(200 / 1400)
    assert st["merge"][1][0] == 10                      # nothing shorter than 1 s to merge


# --- IDSSE parser on a synthetic DFL-format match -------------------------------------------

def _write_dfl(tmp_path: Path):
    base = datetime(2023, 5, 27, 13, 31, 0, tzinfo=timezone.utc)
    info = f"""<?xml version="1.0"?><PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-TEST" MatchTitle="H:A"/>
<Environment PitchX="105.00" PitchY="68.00"/>
<Teams>
<Team TeamId="CLU-H" TeamName="Home" Role="home"><Players/></Team>
<Team TeamId="CLU-A" TeamName="Away" Role="guest"><Players/></Team></Teams>
<OtherGameInformation TotalTimeFirstHalf="4000" TotalTimeSecondHalf="4000" PlayingTimeFirstHalf="1600" PlayingTimeSecondHalf="1600"/>
</MatchInformation></PutDataRequest>"""
    ev = """<?xml version="1.0"?><PutDataRequest>
<Event EventTime="2023-05-27T15:31:07.000+02:00"><KickOff TeamLeft="CLU-H" GameSection="firstHalf" TeamRight="CLU-A"/></Event>
<Event EventTime="2023-05-27T15:31:09.000+02:00"><FinalWhistle GameSection="firstHalf"/></Event>
<Event EventTime="2023-05-27T15:32:06.000+02:00"><KickOff TeamLeft="CLU-A" GameSection="secondHalf" TeamRight="CLU-H"/></Event>
<Event EventTime="2023-05-27T15:32:08.000+02:00"><FinalWhistle GameSection="secondHalf"/></Event>
<Event EventTime="2023-05-27T15:31:08.000+02:00"><Caution CardColor="red" Team="CLU-A"/></Event>
</PutDataRequest>"""
    # each half: 100 frames (4 s); N0 = 10000 / 100000. Kick-off at frame 25, whistle at 75.
    def frames(n0, t0_s, fn):
        rows = []
        for i in range(100):
            t = t0_s + 0.04 * i
            rows.append(fn(n0 + i, i, (base + timedelta(seconds=t)).isoformat(timespec='milliseconds')))
        return "\n".join(rows)

    def person(section, team, pid, x_of, y0, n0, t0_s, speed_kmh, last=100):
        fr = frames(n0, t0_s, lambda N, i, T: f'<Frame N="{N}" T="{T}" X="{x_of(i):.2f}" Y="{y0}" D="0" S="{speed_kmh}" A="0" M="1"/>'
                    if i < last else "")
        return f'<FrameSet GameSection="{section}" MatchId="DFL-MAT-TEST" TeamId="{team}" PersonId="{pid}">\n{fr}\n</FrameSet>'

    def ball(section, n0, t0_s, home_first):
        def fn(N, i, T):
            alive = 1 if 10 <= i < 90 else 0
            dead = 0 if 40 <= i < 50 else alive
            poss = (1 if home_first else 2) if i < 60 else (2 if home_first else 1)
            return (f'<Frame N="{N}" T="{T}" X="{-10 + i * 0.1:.2f}" Y="1.00" Z="0.1" D="0" S="9" A="0" M="1" '
                    f'BallPossession="{poss}" BallStatus="{dead}"/>')
        return (f'<FrameSet GameSection="{section}" MatchId="DFL-MAT-TEST" TeamId="BALL" PersonId="OBJ-BALL">\n'
                f'{frames(n0, t0_s, fn)}\n</FrameSet>')

    parts = ['<?xml version="1.0"?><PutDataRequest><Positions><MetaData MatchId="DFL-MAT-TEST" Type="pitch-size"><PitchSize X="105.00" Y="68.00"/></MetaData>']
    for section, n0, t0_s, left_home, home_first in (("firstHalf", 10000, 6.0, True, True),
                                                     ("secondHalf", 100000, 65.0, False, False)):
        sgn = -1 if left_home else 1
        parts.append(person(section, "referee", "REF1", lambda i: 0.0, 5, n0, t0_s, 3.6))
        for k, (tid, side) in enumerate((("CLU-H", sgn), ("CLU-A", -sgn))):
            # goalkeeper (pid ..0) far on the defended side, two outfield players near the centre
            gx = side * 48.0
            parts.append(person(section, tid, f"{tid}-GK", lambda i, gx=gx: gx, 0, n0, t0_s, 0.0))
            for j in (1, 2):
                # 14.4 km/h = 4 m/s; moves 0.16 m/frame
                parts.append(person(section, tid, f"{tid}-P{j}", lambda i, j=j, side=side: side * (-5.0 * j) + 0.16 * i,
                                    j * 3, n0, t0_s, 14.4,
                                    last=(60 if (tid == "CLU-A" and j == 2 and section == "secondHalf") else 100)))
        parts.append(ball(section, n0, t0_s, home_first))
    parts.append("</Positions></PutDataRequest>")
    d = tmp_path / "raw"
    d.mkdir()
    (d / "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-TEST.xml").write_text(info)
    (d / "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-TEST.xml").write_text(ev)
    (d / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-TEST.xml").write_text("\n".join(parts))
    return d


def test_idsse_parser_on_synthetic_dfl(tmp_path):
    d = _write_dfl(tmp_path)
    f = idsse.match_files(d)["DFL-MAT-TEST"]
    m = idsse.build_match(f["info"], f["events"], f["positions"])
    m.validate()
    assert m.source == "idsse" and m.fps == 25.0 and m.pitch == (105.0, 68.0)
    assert m.n == 1575                                           # real-time grid incl. half-time gap
    # kick-off at +1 s (frame 25) .. whistle at +3 s (frame 75) of the first half
    assert m.half_blocks[0] == (1, 25, 75)
    assert m.attack_dir == {1: +1, 2: -1}                        # home on the left, then right
    assert (m.team == 0).sum() == 3 and (m.team == 1).sum() == 3  # referee excluded
    # centre-origin metres -> corner origin: ball starts at x = -10 + 52.5
    assert m.ball[0, 0] == pytest.approx(-10 + 52.5, abs=1e-4)
    # km/h -> m/s
    assert np.nanmax(m.speed) == pytest.approx(4.0, abs=1e-3)
    assert m.meta["speed_note"].startswith("source S / 3.6")
    # labels: X outside the block; BallStatus 0 -> D; BallPossession 1 -> H, 2 -> A
    y = m.y
    assert (y[:25] == X).all() and (y[76:1500] == X).all()
    assert y[30] == H and y[45] == D and y[65] == A
    # second half: possession codes flipped in the fixture (starts away)
    b2 = m.half_blocks[1][1]
    assert y[b2 + 10] == A
    # red card / dropout: away slot 2 vanishes after frame 60 of the second half
    act = np.isfinite(m.P[..., 0]).sum(1)
    assert act[b2 + 5] == 6 and act[b2 + 45] == 5
    assert m.meta["quality"]["nearest_team"]["n"] > 0
    assert m.meta["alive_frames_before_kickoff_event"]["1"] == 15


# --- SkillCorner parser on a synthetic match --------------------------------------------------

def _write_sc(tmp_path: Path) -> Path:
    mid = "999"
    d = tmp_path / mid
    d.mkdir()
    players = ([{"id": i, "team_id": 1, "trackable_object": 100 + i, "red_card": 0} for i in range(1, 4)]
               + [{"id": i, "team_id": 2, "trackable_object": 100 + i, "red_card": 0} for i in range(4, 7)])
    mj = {"id": 999, "pitch_length": 104, "pitch_width": 68, "home_team": {"id": 1, "short_name": "H"},
          "away_team": {"id": 2, "short_name": "A"}, "home_team_side": ["left_to_right", "right_to_left"],
          "home_team_playing_time": {"minutes_tip": 1.0, "minutes_otip": 1.0},
          "away_team_playing_time": {"minutes_tip": 1.0, "minutes_otip": 1.0},
          "match_periods": [{"period": 1, "start_frame": 2, "end_frame": 49},
                            {"period": 2, "start_frame": 60, "end_frame": 109}],
          "players": players}
    (d / f"{mid}_match.json").write_text(json.dumps(mj))
    lines = []
    for fr in range(110):
        per = 1 if 2 <= fr <= 49 else 2 if 60 <= fr <= 109 else None
        if per is None or fr in (20, 21):                       # no tracking
            lines.append({"frame": fr, "timestamp": None, "period": per,
                          "ball_data": {"x": None, "y": None, "z": None, "is_detected": None},
                          "possession": {"player_id": None, "group": None}, "player_data": []})
            continue
        sgn = -1 if per == 1 else 1                              # home keeper side
        pd_ = []
        for p in players:
            home = p["team_id"] == 1
            side = sgn if home else -sgn
            k = (p["id"] - 1) % 3
            x = side * (45.0 if k == 0 else 3.0 * k) + (0.1 * fr)
            pd_.append({"x": x, "y": 2.0 * k, "player_id": p["id"], "is_detected": fr % 2 == 0})
        grp = "home team" if fr < 30 else None if fr < 40 else "away team"
        lines.append({"frame": fr, "timestamp": "00:00:00.0", "period": per,
                      "ball_data": {"x": 1.0, "y": 0.5, "z": 0.0, "is_detected": fr % 3 != 0},
                      "possession": {"player_id": None, "group": grp}, "player_data": pd_})
    (d / f"{mid}_tracking_extrapolated.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return d


def test_skillcorner_parser_on_synthetic(tmp_path):
    m = skillcorner.build_match(_write_sc(tmp_path))
    m.validate()
    assert m.fps == 10.0 and m.pitch == (104.0, 68.0) and m.source == "skillcorner"
    assert m.half_blocks == [(1, 2, 49), (2, 60, 109)]
    assert m.attack_dir == {1: +1, 2: -1}
    assert m.meta["attack_dir_matches_positions"] == {"1": True, "2": True}
    assert m.y[5] == H and m.y[35] == X and m.y[45] == A            # null group -> X, never D
    assert (m.y == D).sum() == 0
    assert (m.y[:2] == X).all() and (m.y[50:60] == X).all()
    assert m.ball[5, 0] == pytest.approx(1.0 + 52.0)                  # centre -> corner
    assert (m.extras["ball_detected"][20:22] == -1).all()
    assert m.extras["player_detected"][4].max() == 1 and m.extras["player_detected"][5].max() == 0
    assert np.isnan(m.speed[20:22]).all()                             # no tracking, no speed
    assert (m.team == 0).sum() == 3


# --- real data (skipped unless built) ---------------------------------------------------------

CANON = Path(__import__("os").environ.get("KATENA_EXTERNAL") or REPO / "external_data") / "canonical"
REAL = sorted(CANON.glob("*/*.npz")) if CANON.exists() else []


@pytest.mark.skipif(not REAL, reason="external_data/canonical not built")
@pytest.mark.parametrize("path", REAL, ids=[f"{p.parent.name}/{p.stem}" for p in REAL])
def test_real_canonical_matches(path):
    m = load_match(path)
    validate_canonical(m)
    runs = pm.valid_runs(m.y)
    # official scorer segmentation never lets a spell straddle a half
    for a, b, _ in pm.possession_spells(m.y, runs):
        halves = {h for h, s, e in m.half_blocks if s <= a <= e or s <= b - 1 <= e}
        assert len(halves) == 1
    q = m.meta["quality"]
    assert q["nearest_team"]["agree"] > 0.6
    if m.source == "idsse":
        for h in ("1", "2"):
            alive = m.meta["alive_seconds_vs_source_PlayingTime"][h]
            assert abs(alive["ours"] - alive["source"]) < 1.0       # BallStatus 1 == alive
        assert m.meta["direction_from_positions_matches_events"]
