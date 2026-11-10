"""Imitation Smart Tagging for external matches (G2): set-piece intervals from real external event data.

The real match's ``SET PIECES`` tags are Metrica's automatic output (docs/upstream/DATA.md). External matches have
no such tagging, so it is imitated from each match's own set-piece events, never from possession labels:

* IDSSE / DFL (``DFL_03_02_events_raw``): ``ThrowIn``, ``CornerKick``, ``GoalKick``, ``FreeKick`` and ``KickOff``
  (with a ``GameSection``: the start of a half; without: after a goal), at the event's ``EventTime`` on the match
  grid (row ``i`` is ``t0 + i / fps``, the canonical match's ``t0_utc``). Penalties and referee balls have no
  Smart Tagging Type and are left out.
* Metrica sample games: ``SET PIECE`` events (``THROW IN``, ``CORNER KICK``, ``FREE KICK``, ``KICK OFF``; the first
  kick-off of a period is the start, later ones follow a goal) and goal kicks (any other event whose subtype
  contains ``GOAL KICK``), at the event's start frame.
* SkillCorner has no events: no tagging, so its tag columns are NaN.

Then the real tags' character is simulated, as fixed in the G2 pre-registration (docs/decision-log.md, 2026-11-10):

* **Recall per Type.** The real file has ``REAL_COUNTS`` instances of each Type. Of an external match's ``n`` events
  of a Type, ``min(REAL_COUNTS[type], n)`` are kept, drawn uniformly without replacement: the real share of each
  Type, with the external match's own count standing in for the real match's unknown true count.
* **Timing.** The real intervals are windows around the restart (about 5 s, ``starter/load_smart_tagging.py``),
  and where the restart sits in them is unknown. Assumed: the interval starts ``U(0, 5)`` s before the event and ends
  ``U(0, 5)`` s after it (at least one frame after), independently per event.
* **Team.** Every throw-in is ``N/A`` (DATA.md). Every other Type carries the team that takes it.
* No false positives are added.

Every draw comes from ``default_rng(protocol.seed_for(match_id, "g2_tags"))``, so the tags depend on the match id
only, the same in every world and every fold.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from katena.featurize.tags import KINDS, SetPiece

# docs/upstream/DATA.md, the Type tag group of the real match's Smart Tagging
REAL_COUNTS = {"throw-in": 26, "corner kick": 12, "goal kick": 10, "free kick": 5, "kick off (after goal)": 4,
               "kick off (start)": 2}
PRE_S = (0.0, 5.0)              # the interval starts U(0, 5) s before the restart
POST_S = (0.0, 5.0)             # and ends U(0, 5) s after it
SEED_TAG = "g2_tags"
NO_TEAM_KINDS = ("throw-in",)   # DATA.md: every throw-in is tagged N/A

DFL_KIND = {"ThrowIn": "throw-in", "CornerKick": "corner kick", "GoalKick": "goal kick", "FreeKick": "free kick"}


# ---------------------------------------------------------------------------------------------
# events: (row on the match grid, kind, team 0/1)

def _dfl_team(el, home_id: str) -> int | None:
    team = el.get("Team")
    if not team:                                   # a KickOff names its team on the nested Play
        team = next((d.get("Team") for d in el.iter() if d.get("Team")), None)
    if not team:
        return None
    return 0 if team == home_id else 1


def idsse_events(events_path, home_id: str, t0_utc: str, fps: float) -> list:
    """``(row, kind, team)`` of every set piece in a DFL raw event file, rows on the canonical grid."""
    t0 = datetime.fromisoformat(t0_utc)
    out = []
    for ev in ET.parse(str(events_path)).getroot().iter("Event"):
        when = ev.get("EventTime")
        if not when:
            continue
        row = int(round((datetime.fromisoformat(when) - t0).total_seconds() * fps))
        for el in ev:
            if el.tag in DFL_KIND:
                kind = DFL_KIND[el.tag]
            elif el.tag == "KickOff":
                kind = "kick off (start)" if el.get("GameSection") else "kick off (after goal)"
            else:
                continue
            out.append((row, kind, _dfl_team(el, home_id)))
    return sorted(out, key=lambda e: (e[0], KINDS.index(e[1])))


def metrica_events(events) -> list:
    """``(row, kind, team)`` from ``katena.external.metrica.Event`` objects (1-based frames, so row = start - 1)."""
    out = []
    first_ko = {}
    for e in sorted(events, key=lambda e: (e.start, e.order)):
        sub = e.sub or ""
        kind = None
        if e.type == "SET PIECE":
            if "THROW IN" in sub:
                kind = "throw-in"
            elif "CORNER KICK" in sub:
                kind = "corner kick"
            elif "FREE KICK" in sub:
                kind = "free kick"
            elif "KICK OFF" in sub:
                kind = "kick off (after goal)" if first_ko.setdefault(e.period, e.start) != e.start else "kick off (start)"
        elif "GOAL KICK" in sub:
            kind = "goal kick"
        if kind is not None:
            out.append((int(e.start) - 1, kind, int(e.team)))
    return sorted(set(out), key=lambda e: (e[0], KINDS.index(e[1])))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def events_for(ref, external_root: Path) -> list | None:
    """Set-piece events of an external match, from the raw files its canonical record names (their SHA-256 is
    checked against the record). None for SkillCorner, which has no events."""
    if ref.dataset == "skillcorner":
        return None
    rec = json.loads(Path(ref.path).with_suffix(".json").read_text(encoding="utf-8"))
    inputs = {Path(k).name: (Path(external_root) / k, v) for k, v in rec["inputs"].items()}
    for name, (p, sha) in inputs.items():
        if (("events" in name.lower()) or ("matchinformation" in name.lower()) or name.endswith("metadata.xml")) \
                and _sha256(p) != sha:
            raise RuntimeError(f"{ref.key}: {name} differs from the file the canonical match was built from")
    fps = float(rec["fps"])
    if ref.dataset == "idsse":
        from katena.external.idsse import parse_info
        info = next(p for n, (p, _) in inputs.items() if "matchinformation" in n)
        evs = next(p for n, (p, _) in inputs.items() if "events_raw" in n)
        return idsse_events(evs, parse_info(info)["teams"]["home"]["id"], rec["meta"]["t0_utc"], fps)
    if ref.dataset == "metrica":
        from katena.external import metrica as M
        csv = [p for n, (p, _) in inputs.items() if n.endswith("RawEventsData.csv")]
        if csv:
            return metrica_events(M._read_csv_events(csv[0]))
        meta = next(p for n, (p, _) in inputs.items() if n.endswith("metadata.xml"))
        js = next(p for n, (p, _) in inputs.items() if n.endswith("events.json"))
        return metrica_events(M._read_json_events(js, M._parse_epts_meta(meta)["local"]))
    raise ValueError(f"no set-piece events for dataset {ref.dataset}")


# ---------------------------------------------------------------------------------------------
# imitation

def imitate(events: list | None, match_id: str, fps: float) -> tuple[list | None, dict]:
    """(``SetPiece`` list, aggregate counts) imitating the real tags from ``events`` (``(row, kind, team)``).
    ``events`` None (no event data) gives (None, {}). Every event counts, also one a few frames before the tracking
    grid starts (a DFL first-half kickoff often is), since the columns only compare frame numbers."""
    if events is None:
        return None, {}
    rng = np.random.default_rng(_seed(match_id))
    kept = []
    stats = {"events": {}, "kept": {}}
    for kind in KINDS:
        evs = sorted((e for e in events if e[1] == kind), key=lambda e: (e[0], -1 if e[2] is None else e[2]))
        keep = min(REAL_COUNTS[kind], len(evs))
        pick = np.sort(rng.choice(len(evs), size=keep, replace=False)) if keep else np.array([], dtype=np.int64)
        kept += [evs[i] for i in pick]
        stats["events"][kind], stats["kept"][kind] = len(evs), int(keep)
    kept.sort(key=lambda e: (e[0], KINDS.index(e[1])))
    out = []
    for row, kind, team in kept:
        pre = int(round(rng.uniform(*PRE_S) * fps))
        post = max(1, int(round(rng.uniform(*POST_S) * fps)))
        out.append(SetPiece(start=int(row) - pre, end=int(row) + post, kind=kind,
                            team=None if kind in NO_TEAM_KINDS else team))
    return out, stats


def _seed(match_id: str) -> int:
    from katena.eval.protocol import seed_for
    return seed_for(match_id, SEED_TAG)
