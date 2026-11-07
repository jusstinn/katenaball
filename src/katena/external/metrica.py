"""Metrica Sports sample data (github.com/metrica-sports/sample-data).

* Games 1 and 2: CSV tracking (home and away files, 25 Hz, coordinates 0-1 with origin top-left)
  and a CSV event file. Anonymised.
* Game 3: FIFA/EPTS tracking text plus XML metadata (the same layout as the competition ATD, with
  one DataFormatSpecification per roster change) and JSON events.

None of the three has possession labels or speed. Labels are *derived from events* (see
:func:`derive_labels`) and are therefore approximate; ``meta['label_source'] == 'derived_events'``.
Speed is derived from positions by central differences.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from katena.external.common import count_speed_blanked, derive_speed, infer_attack_dir, infer_attack_dir_gk, quality_report
from katena.types import D, X, Match

FPS = 25.0
PITCH = (105.0, 68.0)

# Events after which the ball is dead until a restart event.
_DEAD_TYPES = {"BALL OUT", "FAULT RECEIVED"}
_DEAD_IN_SUBTYPE = {"CHALLENGE": ("FAULT",), "BALL LOST": ("OFFSIDE", "HAND BALL"),
                    "PASS": ("OFFSIDE",), "SHOT": ("GOAL", "OUT")}
_OWNER_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE"}


@dataclass
class Event:
    period: int
    start: int          # 1-based tracking frame
    end: int
    team: int           # 0 home, 1 away
    type: str
    sub: str
    order: int


def _is_restart(e: Event) -> bool:
    return e.type == "SET PIECE" or (e.type == "PASS" and "GOAL KICK" in e.sub)


def _is_dead_trigger(e: Event) -> bool:
    if e.type in _DEAD_TYPES:
        return True
    return any(k in e.sub for k in _DEAD_IN_SUBTYPE.get(e.type, ()))


def derive_labels(events: list[Event], n: int) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    """H/A/D/X per frame (0-based grid of length n) from an event list.

    * A *dead trigger* (BALL OUT, fouls, offside, hand ball, goals, shots out) makes the ball D
      from the event's start frame until the next *restart* (any SET PIECE, or a goal kick).
    * Otherwise the possession is the team of the last PASS, CARRY, RECOVERY or SET PIECE.
    * Frames outside [first event of a period, last event end of that period] are X.

    Returns (y, windows [(period, first_idx, last_idx)], stats).
    """
    ev = sorted(events, key=lambda e: (e.start, e.order))
    y = np.full(n, X, dtype=np.int8)
    windows = []
    stats = {"n_events": len(ev), "ignored_while_dead": 0, "dead_spans_s": []}
    for period in sorted({e.period for e in ev}):
        pe = [e for e in ev if e.period == period]
        first = pe[0].start - 1
        last = min(max(max(e.end, e.start) for e in pe) - 1, n - 1)
        windows.append((period, first, last))
        state, dead_from = X, None
        changes: list[tuple[int, int]] = []
        for e in pe:
            f = e.start - 1
            if _is_restart(e):
                new = e.team
            elif _is_dead_trigger(e):
                new = D
            elif e.type in _OWNER_TYPES and state != D:
                new = e.team
            else:
                if state == D and e.type in _OWNER_TYPES:
                    stats["ignored_while_dead"] += 1
                continue
            if new == D and state != D:
                dead_from = f
            if new != D and state == D and dead_from is not None:
                stats["dead_spans_s"].append((f - dead_from) / FPS)
            if new != state:
                changes.append((f, new))
                state = new
        # paint
        fr = np.array([c[0] for c in changes] + [last + 1])
        for i, (f, s) in enumerate(changes):
            a, b = max(f, first), min(fr[i + 1] - 1, last)
            if a <= b:
                y[a:b + 1] = s
    return y, windows, stats


# --- Games 1 and 2 (CSV) ------------------------------------------------------------------

def _read_csv_tracking(path: Path):
    raw = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64)
    names = pd.read_csv(path, header=None, skiprows=2, nrows=1).iloc[0].tolist()
    period = raw.iloc[:, 0].to_numpy(np.int64)
    frame = raw.iloc[:, 1].to_numpy(np.int64)
    xy = raw.iloc[:, 3:].to_numpy(np.float64)
    ids = [str(names[3 + 2 * i]) for i in range((xy.shape[1]) // 2)]
    return period, frame, xy.reshape(len(raw), -1, 2), ids


def _read_csv_events(path: Path) -> list[Event]:
    df = pd.read_csv(path)
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        sub = "" if pd.isna(r.Subtype) else str(r.Subtype).upper()
        out.append(Event(int(r.Period), int(r[4]), int(r[6]), 0 if r.Team == "Home" else 1,
                         str(r.Type).upper(), sub, i))
    for e in out:                       # KICK OFF rows carry end=0
        if e.end < e.start:
            e.end = e.start
    return out


def build_csv_game(game_dir: Path, k: int) -> Match:
    stem = f"Sample_Game_{k}"
    per_h, fr_h, xy_h, id_h = _read_csv_tracking(game_dir / f"{stem}_RawTrackingData_Home_Team.csv")
    per_a, fr_a, xy_a, id_a = _read_csv_tracking(game_dir / f"{stem}_RawTrackingData_Away_Team.csv")
    assert np.array_equal(fr_h, fr_a) and np.array_equal(per_h, per_a)
    ball_h, ball_a = xy_h[:, -1], xy_a[:, -1]
    both = np.isfinite(ball_h[:, 0]) & np.isfinite(ball_a[:, 0])
    ball_equal = float(np.mean(np.abs(ball_h[both] - ball_a[both]) < 1e-9))
    pl_h, pl_a = xy_h[:, :-1], xy_a[:, :-1]
    return _assemble(
        name=stem, frames=fr_h, period=per_h, players=[pl_h, pl_a], ids=[id_h[:-1], id_a[:-1]],
        ball=np.where(np.isfinite(ball_h), ball_h, ball_a), events=_read_csv_events(
            game_dir / f"{stem}_RawEventsData.csv"),
        attack_from_meta=None, extra={"ball_home_away_files_identical_share": ball_equal,
                                      "format": "metrica_csv"})


# --- Game 3 (EPTS) ------------------------------------------------------------------------

def _parse_epts_meta(path: Path):
    root = ET.parse(path).getroot()
    chan = {pc.get("id"): pc.get("playerId") for pc in root.iter("PlayerChannel")}
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    local = root.find(".//Score").get("idLocalTeam")
    segs = []
    for dfs in root.iter("DataFormatSpecification"):
        top = dfs.find("SplitRegister")
        pids = []
        for sr in top.findall("SplitRegister"):
            refs = [r.get("playerChannelId") for r in sr.findall("PlayerChannelRef")]
            pids.append(chan[refs[0]])
        segs.append((int(dfs.get("startFrame")), int(dfs.get("endFrame")), pids))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.find(".//GlobalConfig").iter("ProviderParameter")}
    attack = {t.get("id"): t.find(".//Value").text for t in root.iter("Team")}
    return {"team_of": team_of, "local": local, "segs": segs, "params": params, "attack": attack,
            "fps": float(root.findtext(".//FrameRate")),
            "field": (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))}


def _read_json_events(path: Path, local: str) -> list[Event]:
    out = []
    for i, e in enumerate(json.loads(path.read_text(encoding="utf-8"))["data"]):
        sub = e["subtypes"]
        if isinstance(sub, dict):
            sub = [sub]
        subs = " ".join(s["name"] for s in sub) if sub else ""
        out.append(Event(int(e["period"]), int(e["start"]["frame"]), int(e["end"]["frame"]),
                         0 if e["team"]["id"] == local else 1, e["type"]["name"].upper(),
                         subs.upper(), i))
    return out


def build_game3(game_dir: Path) -> Match:
    meta = _parse_epts_meta(game_dir / "Sample_Game_3_metadata.xml")
    assert meta["field"] == PITCH and meta["fps"] == FPS
    segs = meta["segs"]
    all_pids = sorted({p for _, _, ps in segs for p in ps},
                      key=lambda p: (0 if meta["team_of"][p] == meta["local"] else 1, int(p[1:])))
    col = {p: i for i, p in enumerate(all_pids)}
    n_lines = segs[-1][1]
    P = np.full((n_lines, len(all_pids), 2), np.nan)
    ball = np.full((n_lines, 2), np.nan)
    bad_lines = 0
    with open(game_dir / "Sample_Game_3_tracking.txt", encoding="utf-8") as f:
        seg_i = 0
        for i, line in enumerate(f):
            fr, pl, bl = line.rstrip("\n").split(":")[:3]
            fr = int(fr)
            if fr != i + 1:
                bad_lines += 1
            while fr > segs[seg_i][1]:
                seg_i += 1
            pids = segs[seg_i][2]
            cells = pl.split(";")
            if len(cells) != len(pids):
                bad_lines += 1
                continue
            for p, c in zip(pids, cells):
                x, y = c.split(",")
                P[i, col[p]] = (float(x), float(y))
            bx, by = bl.split(",")
            ball[i] = (float(bx), float(by))
    team_ids = [meta["team_of"][p] for p in all_pids]
    home_mask = np.array([t == meta["local"] for t in team_ids])
    frames = np.arange(1, n_lines + 1)
    period = np.where(frames <= int(meta["params"]["first_half_end"]), 1, 2)
    flipped = {t: v for t, v in meta["attack"].items()}
    home_dir = +1 if flipped[meta["local"]] == "left_to_right" else -1
    return _assemble(
        name="Sample_Game_3", frames=frames, period=period,
        players=[P[:, home_mask], P[:, ~home_mask]],
        ids=[[p for p, h in zip(all_pids, home_mask) if h], [p for p, h in zip(all_pids, home_mask) if not h]],
        ball=ball, events=_read_json_events(game_dir / "Sample_Game_3_events.json", meta["local"]),
        attack_from_meta={1: home_dir},
        extra={"format": "epts", "tracking_line_problems": bad_lines,
               "n_roster_segments": len(segs),
               "provider_half_bounds": {k: v for k, v in meta["params"].items() if v}})


# --- shared assembly ------------------------------------------------------------------------

def _assemble(*, name, frames, period, players, ids, ball, events, attack_from_meta, extra) -> Match:
    L, W = PITCH
    n = int(frames.max())
    gaps = int(len(frames) - len(np.unique(frames))) + int((np.diff(frames) != 1).sum())
    keep, team, slot_ids = [], [], []
    for t, (arr, idl) in enumerate(zip(players, ids)):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[:, j, 0]).any():
                keep.append((t, j))
    P = np.full((n, len(keep), 2), np.nan, dtype=np.float32)
    for s, (t, j) in enumerate(keep):
        P[frames - 1, s] = players[t][:, j] * (L, W)
        team.append(t)
        slot_ids.append(f"{'HA'[t]}:{ids[t][j]}")
    B = np.full((n, 2), np.nan, dtype=np.float32)
    B[frames - 1] = ball * (L, W)
    # (a) out-of-pitch ball: Metrica coordinates can leave 0-1 slightly; kept (real, dead ball).
    y, windows, st = derive_labels(events, n)
    blocks = [(p, a, b) for p, a, b in windows]
    team = np.array(team, dtype=np.int8)
    attack = {}
    gk_dir = {p: infer_attack_dir(P, team, a, b, L) for p, a, b in blocks}
    for p, a, b in blocks:
        attack[p] = gk_dir[p]
    gk_alt = {p: infer_attack_dir_gk(P, team, a, b, L) for p, a, b in blocks}
    meta_dir_ok = None
    if attack_from_meta:
        meta_dir_ok = attack_from_meta[1] == gk_dir[1]
        assert meta_dir_ok, f"{name}: metadata attack direction disagrees with positions"
    # Games where the goalkeeper heuristic is ambiguous fall back to metadata
    for p in attack:
        if attack[p] == 0 and attack_from_meta:
            attack[p] = attack_from_meta[1] * (1 if p == 1 else -1)
    assert all(v != 0 for v in attack.values()), f"{name}: attack direction ambiguous"
    assert all(gk_alt[p] in (0, attack[p]) for p in attack), f"{name}: centroid and goalkeeper methods disagree"
    assert len(attack) < 2 or attack[1] == -attack[2], f"{name}: halves do not switch ends"
    sp = derive_speed(P, FPS)
    m = Match(P=P, team=team, speed=sp, ball=B, half_blocks=blocks, attack_dir=attack, y=y,
              pitch=PITCH, fps=FPS, name=name, source="metrica",
              meta={"label_source": "derived_events", "speed_note": "derived_from_positions "
                    "(central difference); the source has no speed", "slot_ids": slot_ids,
                    "speed_blanked_above_12.5ms": count_speed_blanked(P, FPS),
                    "frame_gaps_or_duplicates": gaps, "direction_from_positions": gk_dir, "direction_goalkeeper_heuristic": gk_alt,
                    "direction_matches_metadata": meta_dir_ok,
                    "label_stats": {"n_events": st["n_events"],
                                    "ignored_while_dead": st["ignored_while_dead"],
                                    "dead_span_s_quantiles": (np.percentile(st["dead_spans_s"], [50, 90, 99, 100]).round(1).tolist()
                                                              if st["dead_spans_s"] else [])},
                    **extra})
    m.meta["quality"] = quality_report(m)
    return m
