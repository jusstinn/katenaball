"""SkillCorner open data (github.com/SkillCorner/opendata, MIT).

Broadcast-derived tracking at 10 fps, metres, origin at the pitch centre (converted here to a
corner origin). One JSON line per video frame with ball (x, y, z, ``is_detected``), a
``possession`` dict (``player_id``, ``group`` in {"home team", "away team", null}) and
``player_data`` (22 players whenever the frame has tracking; ``is_detected`` false = extrapolated).

Labels: group "home team" -> H, "away team" -> A, null -> X (undetermined). SkillCorner does not
document what a null group means and offers no dead-ball label, so none is fabricated: see
docs/external-data.md for the evidence on what null looks like.

Detection flags are kept in ``Match.extras``: ``ball_detected`` int8[n] (1 detected, 0 extrapolated,
-1 no data) and ``player_detected`` int8[n, S] (same coding). Positions in ``P``/``ball`` include
the extrapolated values exactly as supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from katena.external.common import count_speed_blanked, derive_speed, infer_attack_dir, infer_attack_dir_gk, quality_report
from katena.types import A, H, X, Match

FPS = 10.0
GROUP = {"home team": H, "away team": A}


def match_ids(root: Path) -> list[str]:
    """Match ids whose tracking file is the real file (not an LFS pointer)."""
    out = []
    for d in sorted(Path(root).iterdir()):
        t = d / f"{d.name}_tracking_extrapolated.jsonl"
        if t.is_file() and t.stat().st_size > 1000:
            out.append(d.name)
    return out


def build_match(match_dir: Path) -> Match:
    mid = match_dir.name
    mj = json.loads((match_dir / f"{mid}_match.json").read_text(encoding="utf-8"))
    L, W = float(mj["pitch_length"]), float(mj["pitch_width"])
    home_id = mj["home_team"]["id"]
    # tracking player_id corresponds to ``trackable_object`` in match.json (checked below)
    team_by_obj = {p["trackable_object"]: (0 if p["team_id"] == home_id else 1) for p in mj["players"]}
    team_by_id = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in mj["players"]}
    periods = {p["period"]: (p["start_frame"], p["end_frame"]) for p in mj["match_periods"]}

    frames, rows = [], []
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            frames.append(d["frame"])
            rows.append(d)
    frames_a = np.array(frames)
    n = int(frames_a.max()) + 1
    gaps = int((np.diff(frames_a) != 1).sum()) + int(len(frames_a) - len(np.unique(frames_a)))

    seen = {pl["player_id"] for d in rows for pl in d["player_data"]}
    use_obj = len(seen & set(team_by_obj)) >= len(seen & set(team_by_id))
    team_map = team_by_obj if use_obj else team_by_id
    unknown = sorted(seen - set(team_map))
    pids = sorted(seen & set(team_map), key=lambda p: (team_map[p], p))
    slot = {p: i for i, p in enumerate(pids)}
    S = len(pids)
    team = np.array([team_map[p] for p in pids], dtype=np.int8)

    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    pdet = np.full((n, S), -1, dtype=np.int8)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    bdet = np.full(n, -1, dtype=np.int8)
    g = np.full(n, -1, dtype=np.int8)           # -1 null, 0 H, 1 A
    period = np.zeros(n, dtype=np.int8)
    for d in rows:
        i = d["frame"]
        period[i] = d["period"] or 0
        pos = d["possession"] or {}
        g[i] = GROUP.get(pos.get("group"), -1)
        b = d["ball_data"]
        if b["x"] is not None:
            ball[i] = (b["x"] + L / 2, b["y"] + W / 2)
            bdet[i] = 1 if b["is_detected"] else 0
        for pl in d["player_data"]:
            j = slot.get(pl["player_id"])
            if j is None:
                continue
            P[i, j] = (pl["x"] + L / 2, pl["y"] + W / 2)
            pdet[i, j] = 1 if pl["is_detected"] else 0

    # match.json can end a period one frame after the last tracked frame (seen in 1899585)
    clipped = {p: b - (n - 1) for p, (a, b) in periods.items() if b > n - 1}
    blocks = [(p, a, min(b, n - 1)) for p, (a, b) in sorted(periods.items())]
    y = np.full(n, X, dtype=np.int8)
    for _, a, b in blocks:
        seg = slice(a, b + 1)
        y[seg] = np.where(g[seg] >= 0, g[seg], X)

    sides = mj["home_team_side"]                # attack direction of home per period
    attack = {p: (+1 if sides[p - 1] == "left_to_right" else -1) for p, _, _ in blocks}
    gk = {p: infer_attack_dir(P, team, a, b, L) for p, a, b in blocks}
    for p in gk:
        assert gk[p] == attack[p], f"{mid} period {p}: match.json says {attack[p]}, positions say {gk[p]}"
    team_total = {"home_minutes_tip": mj["home_team_playing_time"]["minutes_tip"],
                  "away_minutes_tip": mj["away_team_playing_time"]["minutes_tip"]}
    labelled = {k: int((y == c).sum()) / FPS / 60 for k, c in (("home", H), ("away", A))}
    n_reds = sum(1 for p in mj["players"] if p.get("red_card"))
    sp = derive_speed(P, FPS)

    m = Match(
        P=P, team=team, speed=sp, ball=ball, half_blocks=blocks, attack_dir=attack, y=y,
        pitch=(L, W), fps=FPS, name=mid, source="skillcorner",
        extras={"ball_detected": bdet, "player_detected": pdet},
        meta={
            "label_source": "source_possession_group (null -> X)",
            "home_team": mj["home_team"]["short_name"], "away_team": mj["away_team"]["short_name"],
            "speed_note": "derived_from_positions at 10 fps (source has no speed); extrapolated "
                          "positions included, so noisy",
            "speed_blanked_above_12.5ms": count_speed_blanked(P, FPS),
            "period_end_beyond_tracking_frames": {str(k): v for k, v in clipped.items()},
            "frame_gaps_or_duplicates": gaps, "player_ids_matched_via":
                "trackable_object" if use_obj else "id",
            "unknown_player_ids": unknown, "attack_dir_matches_positions":
                {str(p): gk[p] == attack[p] for p in gk},
            "direction_goalkeeper_heuristic": {str(p): infer_attack_dir_gk(P, team, a, b, L)
                                                for p, a, b in blocks},
            "source_possession_minutes": team_total, "our_labelled_minutes": labelled,
            "red_cards_in_match_json": n_reds,
            "frames_without_tracking_inplay": int(sum(
                (bdet[a:b + 1] < 0).sum() for _, a, b in blocks)),
        },
    )
    m.meta["quality"] = quality_report(m)
    q = m.meta["quality"]
    inb = np.zeros(n, bool)
    for _, a, b in blocks:
        inb[a:b + 1] = True
    q["share_players_detected"] = float((pdet[inb] == 1).sum() / max((pdet[inb] >= 0).sum(), 1))
    q["share_ball_detected_of_tracked_frames"] = float((bdet[inb] == 1).sum() / max((bdet[inb] >= 0).sum(), 1))
    return m
