"""Metrica Sports open sample Game 3 (FIFA/EPTS tracking + Metrica JSON events) -> `Match`.

The tracking has no possession flags, so labels are derived from the (manually tagged) events:

Dead ball
    Every restart (`SET PIECE` of any subtype, and the `GOAL KICK` subtype, which Metrica tags on
    a PASS / BALL LOST rather than as a set piece) ends a stoppage. The stoppage starts at the
    earliest stop marker after the last in-play action before the restart:
        BALL OUT, SHOT ...GOAL/OUT, BALL LOST/PASS ...OFFSIDE, BALL LOST ...END HALF -> end frame
        FAULT RECEIVED, CHALLENGE ...FAULT (not ADVANTAGE), BALL LOST ...HAND BALL  -> start frame
    and, when no marker is present (e.g. an untagged stoppage), at the end frame of that last
    in-play action. Frames of a half before its first restart (pre-kickoff) and after its last
    stoppage are dead too.
Possession (alive frames)
    The team of the most recent on-ball event (PASS, CARRY, SHOT, RECOVERY, BALL LOST, SET PIECE)
    at or before the frame. CHALLENGE, CARD, FAULT RECEIVED and BALL OUT do not set the owner.
    So a loose ball stays with the last team that had it until the other side's RECOVERY.

These labels are touch-level like DFL's (a RECOVERY immediately followed by a BALL LOST is a
first-time clearance, not control), so they go through the same `harmonise()`.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from ..match import A, D, H, X, Match
from .common import HALF_GAP
from .epts import load_epts

GAME3_FILES = {"meta": "Sample_Game_3_metadata.xml", "tracking": "Sample_Game_3_tracking.txt",
               "events": "Sample_Game_3_events.json"}
GAME3_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
             "Sample_Game_3/{name}")

PLAY_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "CHALLENGE"}
OWNER_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"}


def game3_paths(root) -> dict:
    folder = Path(root) / "raw" / "metrica" / "game3"
    return {k: folder / v for k, v in GAME3_FILES.items()}


def download_game3(root, log=print) -> dict:
    paths = game3_paths(root)
    for kind, dst in paths.items():
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"  downloading Metrica Game 3 {kind} ...")
        tmp = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(GAME3_URL.format(name=dst.name), tmp)
        os.replace(tmp, dst)
    return paths


def _subtypes(e) -> set:
    s = e.get("subtypes")
    if not s:
        return set()
    if isinstance(s, dict):
        s = [s]
    return {x["name"] for x in s}


def _type(e) -> str:
    return e["type"]["name"]


def is_restart(e) -> bool:
    return _type(e) == "SET PIECE" or "GOAL KICK" in _subtypes(e)


def stop_marker_frame(e) -> int | None:
    """Frame at which this event stops play, or None if it does not."""
    t, sub = _type(e), _subtypes(e)
    if t == "BALL OUT":
        return int(e["end"]["frame"])
    if t == "SHOT" and sub & {"GOAL", "OUT"}:
        return int(e["end"]["frame"])
    if t in ("BALL LOST", "PASS") and "OFFSIDE" in sub:
        return int(e["end"]["frame"])
    if t == "BALL LOST" and "END HALF" in sub:
        return int(e["end"]["frame"])
    if t == "BALL LOST" and "HAND BALL" in sub:
        return int(e["start"]["frame"])
    if t == "FAULT RECEIVED":
        return int(e["start"]["frame"])
    if t == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return int(e["start"]["frame"])
    return None


def _is_play(e) -> bool:
    return _type(e) in PLAY_TYPES and stop_marker_frame(e) is None


def labels_from_events(events: list, team_code: dict, periods: list, f0: int, n_src: int,
                       ) -> tuple[np.ndarray, dict]:
    """int8[n_src] labels on the source timeline (index = frame - f0); X outside `periods`.

    periods: [(half, first_frame, last_frame)] in source frame numbers, inclusive.
    team_code: {event team id: 0 home / 1 away}.
    """
    y = np.full(n_src, X, dtype=np.int8)
    diag = {"restarts": 0, "stoppages_with_marker": 0, "stoppages_without_marker": 0,
            "alive_without_owner": 0}
    for half, a, b in periods:
        evs = sorted((e for e in events if int(e["period"]) == int(half)),
                     key=lambda e: (int(e["start"]["frame"]), int(e["index"])))
        n = b - a + 1
        dead = np.zeros(n, dtype=bool)
        restarts = sorted({int(e["start"]["frame"]) for e in evs if is_restart(e)})
        diag["restarts"] += len(restarts)

        def stoppage_start(lo, hi):
            """First dead frame of the stoppage that ends at restart frame `hi` (lo = prev
            restart). The event frame is the last frame in play, hence the +1: with it, the
            stoppage starts on the same frame as the ball disappears from Game 3's tracking
            (median offset 0 over all 103 stoppages; see docs/external_data.md)."""
            win = [e for e in evs if lo <= int(e["start"]["frame"]) < hi]
            plays = [e for e in win if _is_play(e)]
            last_play = max((int(e["start"]["frame"]) for e in plays), default=lo)
            marks = [stop_marker_frame(e) for e in win
                     if stop_marker_frame(e) is not None and int(e["start"]["frame"]) >= last_play]
            if marks:
                diag["stoppages_with_marker"] += 1
                return min(max(min(marks) + 1, lo), hi)
            diag["stoppages_without_marker"] += 1
            end = max((int(e["end"]["frame"]) for e in plays), default=lo)
            return min(max(end + 1, lo), hi)

        if restarts:
            dead[:max(0, min(restarts[0], b + 1) - a)] = True  # before the first restart
            for prev, nxt in zip(restarts[:-1], restarts[1:]):
                s = stoppage_start(prev, nxt)
                dead[max(s - a, 0):max(nxt - a, 0)] = True
            s = stoppage_start(restarts[-1], b + 1)
            dead[max(s - a, 0):] = True
        else:
            dead[:] = True

        owner = np.full(n, -1, dtype=np.int8)
        for e in evs:
            tid = (e.get("team") or {}).get("id")
            if _type(e) in OWNER_TYPES and tid in team_code:
                f = int(e["start"]["frame"]) - a
                if 0 <= f < n:
                    owner[f] = team_code[tid]
        idx = np.maximum.accumulate(np.where(owner >= 0, np.arange(n), -1))
        filled = np.where(idx >= 0, owner[np.maximum(idx, 0)], -1)
        lab = np.full(n, D, dtype=np.int8)
        alive = ~dead
        lab[alive & (filled == 0)] = H
        lab[alive & (filled == 1)] = A
        diag["alive_without_owner"] += int((alive & (filled < 0)).sum())
        y[a - f0:b - f0 + 1] = lab
    return y, diag


def load_metrica_game3(root, half_gap: int = HALF_GAP, download: bool = True) -> Match:
    paths = download_game3(root) if download else game3_paths(root)
    m = load_epts(paths["meta"], paths["tracking"], half_gap=half_gap, name="metrica-game3")
    with open(paths["events"], encoding="utf-8") as fh:
        ev = json.load(fh)["data"]
    from .epts import read_epts_metadata
    md = read_epts_metadata(paths["meta"])
    team_code = {md.teams[0].team_id: 0, md.teams[1].team_id: 1}
    frame_id = m.meta["frame_id"]
    f0 = int(m.meta["first_source_frame"])
    n_src = int(m.meta["n_source_frames"])
    periods = [(h, int(frame_id[a]), int(frame_id[b])) for h, a, b in m.half_blocks]
    y_src, diag = labels_from_events(ev, team_code, periods, f0, n_src)
    y = np.full(m.n, X, dtype=np.int8)
    ok = frame_id >= 0
    y[ok] = y_src[frame_id[ok] - f0]
    m.y = y
    m.meta.update(source="metrica-open", match_id="Sample_Game_3",
                  label_source="derived from Metrica events (see metrica_open.py)",
                  label_diag=diag)
    m.validate()
    return m
