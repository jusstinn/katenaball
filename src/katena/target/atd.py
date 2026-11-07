"""Read the competition's automatic tracking (ATD) into a Match.

Written from the public format description in docs/upstream/DATA.md and starter/baseline.py:

    <frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>

Coordinates are 0-1 of the pitch, speed is m/s, missing values are NaN, and the slot order follows
the Player entries in the metadata XML. frame_id counts from 1, submission rows from 0.

Errors never quote file contents, so a traceback cannot carry match data anywhere (see CLAUDE.md).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from katena.external.common import infer_attack_dir
from katena.types import Match

ROOT = Path(__file__).resolve().parents[3]
FIELDS_PER_GROUP = 3
HOME, AWAY = 0, 1


def _games():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import games

    return games


def read_slot_teams(meta_path: str | Path) -> np.ndarray:
    """int8[S]: 0 for a home slot, 1 for an away slot. The first Team in the file is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams in the ATD metadata, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in the ATD metadata")
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def read_raw(raw_path: str | Path, n_frames: int, n_slots: int, pitch: tuple[float, float],
             dtype=np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(P[n,S,2] metres, speed[n,S], ball[n,2] metres, report). Absent frames stay NaN.

    The report holds counters only. Malformed lines are skipped and counted, as the reference
    baseline skips them. Pass dtype=np.float64 to reproduce the baseline's arithmetic exactly.
    """
    length, width = pitch
    P = np.full((n_frames, n_slots, 2), np.nan, dtype=dtype)
    speed = np.full((n_frames, n_slots), np.nan, dtype=dtype)
    ball = np.full((n_frames, 2), np.nan, dtype=dtype)
    expected = n_slots * FIELDS_PER_GROUP
    seen = np.zeros(n_frames, dtype=bool)
    report = {"n_lines": 0, "n_malformed": 0, "n_out_of_range": 0, "n_duplicate": 0,
              "first_frame_id": None, "last_frame_id": None}

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            report["n_lines"] += 1
            parts = line.strip().split(":")
            if len(parts) != 3:
                report["n_malformed"] += 1
                continue
            try:
                frame_id = int(parts[0])
                players = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
                ball_fields = np.array(parts[2].split(","), dtype=np.float64)
            except ValueError:
                report["n_malformed"] += 1
                continue
            if players.size != expected or ball_fields.size < 2:
                report["n_malformed"] += 1
                continue

            if report["first_frame_id"] is None:
                report["first_frame_id"] = frame_id
            report["last_frame_id"] = frame_id

            frame = frame_id - 1
            if not 0 <= frame < n_frames:
                report["n_out_of_range"] += 1
                continue
            if seen[frame]:
                report["n_duplicate"] += 1
            seen[frame] = True

            groups = players.reshape(n_slots, FIELDS_PER_GROUP)
            P[frame, :, 0] = groups[:, 0] * length
            P[frame, :, 1] = groups[:, 1] * width
            speed[frame] = groups[:, 2]
            ball[frame, 0] = ball_fields[0] * length
            ball[frame, 1] = ball_fields[1] * width

    report["n_frames_present"] = int(seen.sum())
    report["contiguous_from_one"] = (report["first_frame_id"] == 1
                                     and report["last_frame_id"] == n_frames
                                     and bool(seen.all()))
    return P, speed, ball, report


def load_target(data_dir: str | Path | None = None, source: str = "tactical") -> Match:
    """The provided match as a Match, with no labels. Half blocks exist for tactical only."""
    games = _games()
    paths = games.asset_paths(data_dir=str(data_dir) if data_dir else None, source=source)
    pitch = tuple(float(v) for v in games.GAMES[games.GAME]["pitch"])
    n_frames = games.source_frames(source)

    team = read_slot_teams(paths["atd_meta"])
    P, speed, ball, report = read_raw(paths["atd_raw"], n_frames, len(team), pitch)

    blocks = games.half_video_blocks(games.GAME) if source == "tactical" else []
    attack_dir = {h: infer_attack_dir(P, team, a, b, pitch[0]) for h, a, b in blocks}

    match = Match(P=P, team=team, speed=speed, ball=ball, half_blocks=blocks,
                  attack_dir=attack_dir, y=None, pitch=pitch, fps=float(games.GAMES[games.GAME]["fps"]),
                  name=f"{games.GAME}_{source}", source="competition_atd",
                  meta={"read_report": report})
    match.validate()
    return match


def nearest_team(P: np.ndarray, ball: np.ndarray, team: np.ndarray,
                 chunk: int = 20000) -> np.ndarray:
    """int8[n]: the team of the player nearest the ball, -1 where the ball or all players are absent.

    Same rule as the reference baseline, so that reproducing the published baseline files proves
    this loader reads the file the way the organisers do. Ties go to the lowest slot.
    """
    n = P.shape[0]
    out = np.full(n, -1, dtype=np.int8)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        delta = P[start:stop] - ball[start:stop, None, :]
        dist = delta[..., 0] ** 2 + delta[..., 1] ** 2
        has_player = np.isfinite(dist).any(axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        nearest = dist.argmin(axis=1)
        block = team[nearest].astype(np.int8)
        block[~has_player] = -1
        out[start:stop] = block
    return out
