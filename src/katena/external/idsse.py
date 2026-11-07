"""IDSSE / DFL open data (Bassek et al. 2025, figshare 10.6084/m9.figshare.28196177, CC BY 4.0).

Raw files per match (all XML, positions at 25 Hz, metres, origin at the pitch centre):

* ``DFL_02_01_matchinformation_*``: teams (Role home/guest), pitch size, players, playing times.
* ``DFL_03_02_events_raw_*``: KickOff (TeamLeft/TeamRight, GameSection), FinalWhistle, cards...
* ``DFL_04_03_positions_raw_observed_*``: one FrameSet per person per half; ball FrameSet carries
  ``BallPossession`` (1 home / 2 away) and ``BallStatus`` (1 alive / 0 dead) on every frame.

Labels (kept raw, not harmonised): BallStatus==0 -> D, else BallPossession 1 -> H, 2 -> A;
frames outside [kick-off event, final-whistle event] of each half -> X.

Nothing in the raw files is executed. XML is read with the stdlib parser (``iterparse``), which
holds only one FrameSet worth of cleared elements at a time.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from array import array
from datetime import datetime
from pathlib import Path

import numpy as np

from katena.external.common import (blank_far_ball, derive_speed, infer_attack_dir,
                                    infer_attack_dir_gk, quality_report)
from katena.types import A, D, H, X, Match

FPS = 25.0
KMH_TO_MS = 1.0 / 3.6
HOME_CODE, AWAY_CODE = 1, 2           # BallPossession values (checked against nearest player)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def match_files(raw_dir: Path) -> dict[str, dict[str, Path]]:
    """Group the raw files of ``raw_dir`` by match id."""
    out: dict[str, dict[str, Path]] = {}
    for p in sorted(Path(raw_dir).glob("DFL_0*.xml")):
        mm = re.match(r"DFL_0(2_01_matchinformation|3_02_events_raw|4_03_positions_raw_observed)"
                      r"_DFL-COM-\d+_(DFL-MAT-\w+)\.xml$", p.name)
        if mm:
            kind = {"2_01_matchinformation": "info", "3_02_events_raw": "events",
                    "4_03_positions_raw_observed": "positions"}[mm.group(1)]
            out.setdefault(mm.group(2), {})[kind] = p
    return out


def parse_info(path: Path) -> dict:
    root = ET.parse(path).getroot()
    gen = root.find(".//General").attrib
    env = root.find(".//Environment").attrib
    teams = {}
    for t in root.iter("Team"):
        players = [p.attrib for p in t.iter("Player")]
        teams[t.get("Role")] = {"id": t.get("TeamId"), "name": t.get("TeamName"),
                                "players": players}
    other = root.find(".//OtherGameInformation").attrib
    return {"general": dict(gen), "pitch": (float(env["PitchX"]), float(env["PitchY"])),
            "teams": teams, "other": dict(other)}


def parse_events(path: Path) -> dict:
    """Kick-offs and final whistles per half, cards and substitutions (UTC datetimes)."""
    ev = {"kickoff": {}, "whistle": {}, "team_left": {}, "cards": [], "subs": []}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "Event":
            continue
        t = el.get("EventTime")
        for ch in el:
            if ch.tag == "KickOff" and ch.get("GameSection"):
                ev["kickoff"][ch.get("GameSection")] = _ts(t)
                ev["team_left"][ch.get("GameSection")] = ch.get("TeamLeft")
            elif ch.tag == "FinalWhistle" and ch.get("GameSection"):
                ev["whistle"][ch.get("GameSection")] = _ts(t)
            elif ch.tag == "Caution":
                ev["cards"].append((_ts(t), ch.get("CardColor"), ch.get("Team")))
            elif ch.tag == "Substitution":
                ev["subs"].append((_ts(t), ch.get("Team")))
        el.clear()
    return ev


def parse_positions(path: Path) -> dict:
    """Stream the positions file into compact arrays, grouped by (section, team, person)."""
    sets: dict[tuple[str, str, str], dict] = {}
    cur = None
    pitch = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag
        if ev == "start":
            if tag == "FrameSet":
                key = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
                cur = sets.setdefault(key, {"N": array("q"), "x": array("d"), "y": array("d"),
                                            "s": array("d"), "T0": None, "Tl": None})
                if key[1] == "BALL":
                    cur.update(poss=array("b"), stat=array("b"), z=array("d"),
                               T=[])
            continue
        if tag == "Frame":
            a = el.attrib
            cur["N"].append(int(a["N"]))
            cur["x"].append(float(a["X"]))
            cur["y"].append(float(a["Y"]))
            cur["s"].append(float(a.get("S", "nan")))
            if cur["T0"] is None:
                cur["T0"] = a["T"]
            cur["Tl"] = a["T"]
            if "BallStatus" in a:
                cur["poss"].append(int(a["BallPossession"]))
                cur["stat"].append(int(a["BallStatus"]))
                cur["z"].append(float(a.get("Z", "nan")))
                cur["T"].append(a["T"])
            el.clear()
        elif tag == "PitchSize":
            pitch = (float(el.get("X")), float(el.get("Y")))
    return {"sets": sets, "pitch": pitch}


def build_match(info_path: Path, events_path: Path, positions_path: Path) -> Match:
    info = parse_info(info_path)
    ev = parse_events(events_path)
    pos = parse_positions(positions_path)
    mid = info["general"]["MatchId"]
    L, W = pos["pitch"]
    assert (L, W) == info["pitch"], f"pitch mismatch {pos['pitch']} vs {info['pitch']}"
    sets = pos["sets"]
    home_id, away_id = info["teams"]["home"]["id"], info["teams"]["guest"]["id"]

    # --- frame grid from the ball FrameSets (one per half) ---------------------------------
    sections = ["firstHalf", "secondHalf"]
    ball_sets = {s: sets[(s, "BALL", k[2])] for s in sections for k in sets
                 if k[0] == s and k[1] == "BALL"}
    t_first = {s: _ts(ball_sets[s]["T0"]) for s in sections}
    t0 = t_first["firstHalf"]
    n0 = {s: int(ball_sets[s]["N"][0]) for s in sections}
    off = {s: int(round((t_first[s] - t0).total_seconds() * FPS)) for s in sections}
    n_frames = off["secondHalf"] + (int(ball_sets["secondHalf"]["N"][-1]) - n0["secondHalf"]) + 1

    def to_idx(section: str, N: np.ndarray) -> np.ndarray:
        return (N - n0[section]) + off[section]

    # Verify the 25 Hz clock: T must advance 40 ms per N for every ball frame.
    bad_clock = 0
    dup_frames = 0
    for s in sections:
        N = np.frombuffer(ball_sets[s]["N"], dtype=np.int64)
        T = np.array([(_ts(x) - t_first[s]).total_seconds() for x in ball_sets[s]["T"]])
        bad_clock += int((np.abs(T * FPS - (N - n0[s])) > 0.5).sum())
        dup_frames += int(len(N) - len(np.unique(N))) + int((np.diff(N) <= 0).sum())

    # --- slots ----------------------------------------------------------------------------
    persons = sorted({(k[1], k[2]) for k in sets if k[1] not in ("BALL", "referee")},
                     key=lambda tp: (0 if tp[0] == home_id else 1, tp[1]))
    slot = {p: i for i, p in enumerate(persons)}
    S = len(persons)
    team = np.array([0 if t == home_id else 1 for t, _ in persons], dtype=np.int8)
    assert {t for t, _ in persons} <= {home_id, away_id}
    P = np.full((n_frames, S, 2), np.nan, dtype=np.float32)
    speed_src = np.full((n_frames, S), np.nan, dtype=np.float32)
    dup_player = 0
    n_outside_clock = 0
    for (section, tid, pid), d in sets.items():
        if tid in ("BALL", "referee"):
            continue
        idx = to_idx(section, np.frombuffer(d["N"], dtype=np.int64))
        ok = (idx >= 0) & (idx < n_frames)
        n_outside_clock += int((~ok).sum())
        idx = idx[ok]
        dup_player += int(len(idx) - len(np.unique(idx)))
        j = slot[(tid, pid)]
        P[idx, j, 0] = np.frombuffer(d["x"], dtype=np.float64)[ok] + L / 2
        P[idx, j, 1] = np.frombuffer(d["y"], dtype=np.float64)[ok] + W / 2
        speed_src[idx, j] = (np.frombuffer(d["s"], dtype=np.float64)[ok] * KMH_TO_MS)

    # --- ball and raw labels --------------------------------------------------------------
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    poss = np.zeros(n_frames, dtype=np.int8)
    stat = np.full(n_frames, -1, dtype=np.int8)
    for s in sections:
        d = ball_sets[s]
        idx = to_idx(s, np.frombuffer(d["N"], dtype=np.int64))
        ball[idx, 0] = np.frombuffer(d["x"], dtype=np.float64) + L / 2
        ball[idx, 1] = np.frombuffer(d["y"], dtype=np.float64) + W / 2
        poss[idx] = np.frombuffer(d["poss"], dtype=np.int8)
        stat[idx] = np.frombuffer(d["stat"], dtype=np.int8)
    n_ball_far = blank_far_ball(ball, L, W)

    # --- in-play windows from the events (kick-off .. final whistle of each half) ---------
    def ev_idx(t: datetime) -> int:
        return int(round((t - t0).total_seconds() * FPS))

    blocks, window = [], {}
    for h, s in enumerate(sections, start=1):
        a, b = ev_idx(ev["kickoff"][s]), ev_idx(ev["whistle"][s])
        last = off[s] + (int(ball_sets[s]["N"][-1]) - n0[s])
        a, b = max(a, off[s]), min(b, last)
        blocks.append((h, a, b))
        window[h] = (off[s], last)

    y = np.full(n_frames, X, dtype=np.int8)
    for _, a, b in blocks:
        seg = slice(a, b + 1)
        lab = np.where(poss[seg] == HOME_CODE, H, np.where(poss[seg] == AWAY_CODE, A, X))
        y[seg] = np.where(stat[seg] == 0, D, lab)

    # --- speed: source S is km/h; verify against positions, fall back if unusable ----------
    v_pos = derive_speed(P, FPS)
    both = np.isfinite(v_pos) & np.isfinite(speed_src) & (v_pos > 3.0)
    ratio = float(np.median(speed_src[both] / v_pos[both])) if both.any() else float("nan")
    if not (0.8 < ratio < 1.2):
        speed, speed_note = v_pos, "derived_from_positions (source S failed unit check)"
    else:
        speed, speed_note = speed_src, "source S / 3.6 (S is km/h)"

    # --- attack direction: events (TeamLeft) cross-checked with goalkeeper positions -------
    attack, gk_dir, gk_alt = {}, {}, {}
    for h, s in enumerate(sections, start=1):
        left = ev["team_left"][s]
        attack[h] = +1 if left == home_id else -1
        a, b = blocks[h - 1][1:]
        gk_dir[h] = infer_attack_dir(P, team, a, b, L)           # team-centroid method
        assert gk_dir[h] == attack[h], f"{mid} half {h}: TeamLeft says {attack[h]}, data says {gk_dir[h]}"
        gk_alt[h] = infer_attack_dir_gk(P, team, a, b, L)         # second opinion, may be 0
    assert attack[1] == -attack[2]

    # --- checks of the source's own accounting --------------------------------------------
    pt = {1: float(info["other"]["PlayingTimeFirstHalf"]) / 1000,
          2: float(info["other"]["PlayingTimeSecondHalf"]) / 1000}
    alive_s = {h: float((stat[off[s]:window[h][1] + 1] == 1).sum() / FPS)
               for h, s in enumerate(sections, start=1)}
    pre_alive = {h: int(((stat[window[h][0]:blocks[h - 1][1]]) == 1).sum()) for h in (1, 2)}
    post_alive = {h: int(((stat[blocks[h - 1][2] + 1:window[h][1] + 1]) == 1).sum()) for h in (1, 2)}

    m = Match(
        P=P, team=team, speed=speed.astype(np.float32), ball=ball, half_blocks=blocks,
        attack_dir=attack, y=y, pitch=(L, W), fps=FPS, name=mid, source="idsse",
        meta={
            "label_source": "source_ball_status_and_possession",
            "match_title": info["general"]["MatchTitle"],
            "home_team": info["teams"]["home"]["name"], "away_team": info["teams"]["guest"]["name"],
            "t0_utc": t0.isoformat(),
            "tracking_extent": {str(h): list(window[h]) for h in window},
            "speed_note": speed_note, "speed_source_over_positions_median_ratio": ratio,
            "n_ball_far_outside_blanked": n_ball_far,
            "bad_clock_frames": bad_clock, "duplicate_or_nonmonotonic_frames": dup_frames,
            "duplicate_player_frames": dup_player, "player_frames_outside_grid": n_outside_clock,
            "referee_framesets": sum(1 for k in sets if k[1] == "referee"),
            "direction_from_positions_matches_events": True,
            "direction_goalkeeper_heuristic": {str(h): v for h, v in gk_alt.items()},
            "alive_seconds_vs_source_PlayingTime": {
                str(h): {"ours": alive_s[h], "source": pt[h]} for h in (1, 2)},
            "alive_frames_before_kickoff_event": {str(h): v for h, v in pre_alive.items()},
            "alive_frames_after_whistle_event": {str(h): v for h, v in post_alive.items()},
            "red_cards": [[t.isoformat(), c, tm] for t, c, tm in ev["cards"] if c == "red"],
            "n_subs": len(ev["subs"]),
        },
    )
    m.meta["quality"] = quality_report(m)
    return m
