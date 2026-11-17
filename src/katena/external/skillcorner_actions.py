"""SkillCorner labels from its own action events (candidate A, ``scripts/run_a.py``).

SkillCorner's open data carries, beside the per-frame ``possession.group`` that ``skillcorner.py`` turns into
labels, a ``<id>_dynamic_events.csv`` per match. Its ``player_possession`` rows give every spell of one player on
the ball: first and last frame, team, how it started and ended, and whether a game interruption (throw-in, goal
kick, corner, free kick, goal) lies before or after it. ``scripts/probe_sc_actions.py`` (decision log,
2026-11-12) showed that the frame label is a phase-level one: it marks dead ball as unknown, gives the other
team's short possessions to the team around them, and changes before the other team touches the ball.

This module builds the label the target's ground truth is described as (``docs/upstream/CHALLENGE.md``): an owner
that changes at the other team's first touch, and dead ball as a label.

* **Owner.** From the first frame of a player possession to the first frame of the next one in the same period,
  the ball is that player's team's. So it stays with the last team while it is loose or in flight.
* **Dead ball.** A gap between two player possessions holds a stoppage when the earlier one is flagged
  ``game_interruption_after`` or the later one ``game_interruption_before``. Inside it:

  - the ball goes dead at the first gap frame where it is outside the pitch or the frame has no tracking (the
    broadcast cut away), or ``OUT_FALLBACK_S`` after the last touch when neither happens; at once when the
    earlier possession ended in a foul;
  - play restarts ``RESTART_FLIGHT_S`` before the next possession when that one starts as the reception or the
    interception of the restart (the taker has no possession row of his own), and at its first frame otherwise;
  - from the restart to that first frame the ball is the taking team's: the next possession's team when the
    flag ends in ``_for`` (or the start type in ``_reception``), the other team when it ends in ``_against``
    (``_interception``).

* **Unknown (X).** Frames before the first player possession of a period, and dead-ball frames with no tracking
  at all: nothing can be read from them, and "no tracking" as a sign of dead ball belongs to a broadcast feed,
  not to the target's tactical camera.

The two times are approximations fixed here, before any score, not fitted to anything. The events are produced
automatically by SkillCorner from the same broadcast tracking; no person labelled them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from katena.types import A, D, H, X

FPS = 10.0
RESTART_FLIGHT_S = 1.0      # restart taken this long before its reception
OUT_FALLBACK_S = 1.0        # the ball is dead this long after the last touch when nothing else says when
FOUL_ENDS = ("foul_committed", "foul_suffered")
LABEL_SOURCE = "skillcorner_player_possession_events (owner = last touch, interruption gaps = D)"


@dataclass(frozen=True)
class Possession:
    """One ``player_possession`` row. ``team``: 0 home, 1 away. Frames are inclusive tracking frame numbers."""

    start: int
    end: int
    period: int
    team: int
    start_type: str = ""
    end_type: str = ""
    before: str = ""        # game_interruption_before
    after: str = ""         # game_interruption_after


def read_possessions(csv_path: Path, home_team_id) -> list:
    """The ``player_possession`` rows of a ``*_dynamic_events.csv``, sorted by period and first frame."""
    out = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["event_type"] != "player_possession":
                continue
            out.append(Possession(
                start=int(r["frame_start"]), end=int(r["frame_end"]), period=int(r["period"]),
                team=0 if str(r["team_id"]) == str(home_team_id) else 1,
                start_type=r.get("start_type") or "", end_type=r.get("end_type") or "",
                before=r.get("game_interruption_before") or "", after=r.get("game_interruption_after") or ""))
    out.sort(key=lambda p: (p.period, p.start, p.end))
    return out


def taker_team(prev: Possession, nxt: Possession) -> int:
    """Team that takes the restart between ``prev`` and ``nxt``."""
    if nxt.before.endswith("_for") or (not nxt.before and nxt.start_type.endswith("_reception")):
        return nxt.team
    if nxt.before.endswith("_against") or (not nxt.before and nxt.start_type.endswith("_interception")):
        return 1 - nxt.team
    if prev.after.endswith("_for"):
        return prev.team
    if prev.after.endswith("_against"):
        return 1 - prev.team
    return nxt.team


def restart_received(nxt: Possession) -> bool:
    """``nxt`` starts as the reception or interception of a restart, so the restart was taken before it."""
    st = nxt.start_type
    return bool(nxt.before) and (st.endswith("_reception") or st.endswith("_interception"))


def action_labels(poss: list, periods: list, n: int, ball: np.ndarray, tracked: np.ndarray, pitch: tuple,
                  fps: float = FPS) -> np.ndarray:
    """int8[n] labels (H, A, D, X) on the native grid.

    ``poss``: ``Possession`` rows. ``periods``: [(period, first, last)] inclusive frames. ``ball``: float[n, 2] in
    metres from a pitch corner, NaN where missing. ``tracked``: bool[n], the frame has tracking data.
    ``pitch``: (length, width)."""
    L, W = pitch
    y = np.full(n, X, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out_of_pitch = (ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W)
    gone = out_of_pitch | ~np.asarray(tracked, dtype=bool)
    flight, fallback = int(round(RESTART_FLIGHT_S * fps)), int(round(OUT_FALLBACK_S * fps))
    for period, first, last in periods:
        last = min(int(last), n - 1)
        ps = [p for p in poss if p.period == period and first <= p.start <= last]
        for k, p in enumerate(ps):
            nxt = ps[k + 1] if k + 1 < len(ps) else None
            stop = nxt.start if nxt is not None else last + 1          # half-open end of this owner's stretch
            if stop <= p.start:
                continue
            y[p.start:stop] = H if p.team == 0 else A
            if not (p.after or (nxt is not None and nxt.before)):
                continue
            g0 = min(max(p.end + 1, p.start + 1), stop)                 # first frame after the last touch
            if p.end_type in FOUL_ENDS:
                d0 = g0
            else:
                hit = np.flatnonzero(gone[g0:stop])
                d0 = g0 + int(hit[0]) if len(hit) else min(g0 + fallback, stop)
            d1 = stop
            if nxt is not None and restart_received(nxt):
                d1 = max(stop - flight, d0)
            seg = slice(d0, d1)
            y[seg] = np.where(np.asarray(tracked[seg], dtype=bool), D, X)
            if nxt is not None and d1 < stop:
                y[d1:stop] = H if taker_team(p, nxt) == 0 else A
    return y


def build_labels(match_dir: Path, m) -> np.ndarray:
    """Action labels of the canonical SkillCorner match ``m`` (native grid), from the raw match folder."""
    mid = Path(match_dir).name
    mj = json.loads((Path(match_dir) / f"{mid}_match.json").read_text(encoding="utf-8"))
    poss = read_possessions(Path(match_dir) / f"{mid}_dynamic_events.csv", mj["home_team"]["id"])
    tracked = (np.asarray(m.extras["ball_detected"]) >= 0) | (np.asarray(m.extras["player_detected"]) >= 0).any(axis=1)
    return action_labels(poss, list(m.half_blocks), m.n, np.asarray(m.ball), tracked, m.pitch, m.fps)
