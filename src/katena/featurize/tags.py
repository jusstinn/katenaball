"""Set-piece tag columns (G2, "J5-lite"): where each 5 Hz sample sits relative to the tagged set pieces.

The real match carries Metrica's automatic Smart Tagging (docs/upstream/DATA.md). Its ``SET PIECES`` intervals each
have a ``Type`` (``Throw-in``, ``Corner Kick``, ``Goal Kick``, ``Free Kick``, ``Kick Off (After Goal)``,
``Kick Off (Start)``) and a ``Team`` (a club, or ``N/A``, which every throw-in is). On external matches the same
intervals are imitated from real event data (``katena.external.setpieces``). Either way the input here is a list of
``SetPiece`` on the match's native frame grid, and the columns are the same code.

Columns, for a focal team (0 home, 1 away), fixed in the G2 pre-registration (docs/decision-log.md, 2026-11-10):

    tag_since_out, tag_until_out    seconds since the start of the latest / until the start of the next set piece of
    tag_since_fk,  tag_until_fk     the group: ``out`` = throw-in, corner kick, goal kick (restarts after the ball
    tag_since_ko,  tag_until_ko     went out), ``fk`` = free kick, ``ko`` = kick-off (start or after a goal).
                                    "Latest" starts at or before the sample, "next" strictly after it. NaN when there
                                    is none in the match, capped at ``CAP_S``.
    tag_last_team, tag_next_team    the tagged team of the latest / next set piece of any group: +1 the focal team,
                                    -1 the other team, 0 no team (N/A); NaN when there is none.
    tag_inside                      1 when the sample lies inside a tagged interval [start, end), else 0.

A match with no tagging at all (SkillCorner, which has no events) gets NaN in every column, ``tag_inside`` included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP = "tags"
KIND_GROUP = {"throw-in": "out", "corner kick": "out", "goal kick": "out", "free kick": "fk",
              "kick off (after goal)": "ko", "kick off (start)": "ko"}
KINDS = tuple(KIND_GROUP)                      # DATA.md's Type values, lower case, in its order
TYPE_GROUPS = ("out", "fk", "ko")
CAP_S = 300.0                                  # times beyond 5 minutes are all "long ago"
CODE = "set pieces"
NAMES = ([f"tag_{w}_{g}" for g in TYPE_GROUPS for w in ("since", "until")]
         + ["tag_last_team", "tag_next_team", "tag_inside"])


@dataclass(frozen=True)
class SetPiece:
    start: int              # native frame of the interval's start
    end: int                # first native frame after it
    kind: str               # one of KINDS
    team: int | None        # 0 home, 1 away, None for N/A


def from_tagging(tags) -> list:
    """``SetPiece`` list from ``katena.target.tagging.read_tags`` output: every ``SET PIECES`` instance whose Type is
    one of ``KINDS`` (code and Type compared lower case and stripped). Other instances are ignored."""
    out = []
    for t in tags:
        kind = (t.kind or "").strip().lower()
        if (t.code or "").strip().lower() == CODE and kind in KIND_GROUP:
            out.append(SetPiece(int(t.start_frame), int(t.end_frame), kind, t.team))
    return sorted(out, key=lambda s: (s.start, s.end, KINDS.index(s.kind)))


def counts(sps) -> dict:
    """Set pieces per Type (aggregate counts)."""
    return {k: int(sum(s.kind == k for s in sps)) for k in KINDS}


def _since_until(starts: np.ndarray, f: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    j = np.searchsorted(starts, f, side="right")            # starts[:j] <= f < starts[j:]
    since = np.full(len(f), np.nan)
    until = np.full(len(f), np.nan)
    has_last, has_next = j > 0, j < len(starts)
    since[has_last] = (f[has_last] - starts[j[has_last] - 1]) / fps
    until[has_next] = (starts[j[has_next]] - f[has_next]) / fps
    return np.minimum(since, CAP_S), np.minimum(until, CAP_S)


def tag_columns(sps, idx: np.ndarray, step: int, fps: float, focals=(0, 1)) -> tuple[dict, list]:
    """({focal: float32[len(idx), 9]}, NAMES) for the work-grid samples ``idx`` (native frame ``idx * step``, native
    frame rate ``fps``). ``sps`` is a ``SetPiece`` list, or None for a match without tagging (all NaN)."""
    f = np.asarray(idx, dtype=np.int64) * int(step)
    n = len(f)
    if sps is None:
        nan = np.full((n, len(NAMES)), np.nan, dtype=np.float32)
        return {fo: nan.copy() for fo in focals}, list(NAMES)
    sps = sorted(sps, key=lambda s: (s.start, s.end, KINDS.index(s.kind)))
    cols = {}
    for g in TYPE_GROUPS:
        cols[f"tag_since_{g}"], cols[f"tag_until_{g}"] = _since_until(
            [s.start for s in sps if KIND_GROUP[s.kind] == g], f, fps)
    starts = np.array([s.start for s in sps], dtype=np.int64)
    team = np.array([-1 if s.team is None else int(s.team) for s in sps], dtype=np.int64)
    j = np.searchsorted(starts, f, side="right")
    inside = np.zeros(n)
    for s in sps:
        inside[(f >= s.start) & (f < s.end)] = 1.0
    cols["tag_inside"] = inside
    out = {}
    for fo in focals:
        rel = np.where(team < 0, 0.0, np.where(team == fo, 1.0, -1.0))
        last = np.full(n, np.nan)
        nxt = np.full(n, np.nan)
        ok = j > 0
        last[ok] = rel[j[ok] - 1]
        ok = j < len(starts)
        nxt[ok] = rel[j[ok]]
        c = dict(cols, tag_last_team=last, tag_next_team=nxt)
        out[fo] = np.stack([c[k] for k in NAMES], 1).astype(np.float32)
    return out, list(NAMES)
