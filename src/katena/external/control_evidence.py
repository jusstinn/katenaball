"""Event evidence of *controlled* possession for brief opponent flips.

Port of David's ``katena.external.control_evidence`` (origin/david/foundation), adapted to
``katena.types.Match`` and the loaders in ``katena.external.idsse`` and ``katena.external.metrica``.
It reads external event files only, never the competition match. docs/label-convention.md has the
results and what they can and cannot say about the target's convention.

Definitions (codes 0 H, 1 A, 2 D, 3 X; spells are the scorer's maximal H or A runs inside graded runs):

* A *flip* is a spell whose previous segment is the other team's spell. With three labels it can
  end only two ways: ``return`` (back to the other team, the case ``katena.labels.sandwich``
  merges) or ``to_dead`` (dead ball or the end of the run, which ``sandwich(..., tail=True)`` and
  David's ``harmonise`` also merge). A spell after dead ball or at a run start is a ``restart``
  spell and never a flip (CHALLENGE.md: after a stoppage there is a clear possessor).
* A flip *has control evidence* when its own team has at least one deliberate on-ball action within
  ``[start - tol, stop + tol)``: a DFL ``Play`` (pass or cross attempt), ``ShotAtGoal`` or
  ``BallClaiming``. ``OtherBallAction`` (clearances, touches) and duels do not count. ``strict``
  keeps only completed plays (plus shots and claims). For Metrica: a PASS, a SHOT, or a CARRY of
  at least ``min_carry_s``.
* DFL event clocks are offset from the tracking clock by up to about 2 s per match, so each match's
  offset is estimated first: the whole-frame shift (within +-``max_shift_s``) that maximises the
  agreement between every Play's team and the label on that frame (alive frames only).

Metrica's labels are derived from the same events (``katena.external.metrica.derive_labels``), so
the evidence is partly circular there: it is reported for comparison with David's table but is
kept out of the pooled estimate.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

import possession_metrics as pm
from katena.types import A, D, H

CONTROL_TAGS = ("Play", "ShotAtGoal", "BallClaiming")
KINDS = ("return", "to_dead", "restart")


# ---------------------------------------------------------------------------------------------
# spells and flips

def spell_kinds(y: np.ndarray) -> list[tuple[int, int, int, str]]:
    """``(start, stop, code, kind)`` for every possession spell, in time order."""
    out = []
    for start, stop in pm.valid_runs(y):
        segs = pm.segments(y, [(start, stop)])
        for i, (a, b, k) in enumerate(segs):
            if k not in (H, A):
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev not in (H, A):
                kind = "restart"
            elif nxt == prev:
                kind = "return"
            else:
                kind = "to_dead"            # next is D or the end of the run
            out.append((a, b, k, kind))
    return out


# ---------------------------------------------------------------------------------------------
# reading actions

def read_dfl_actions(events_path, home_id: str) -> dict:
    """Every Play / ShotAtGoal / BallClaiming / OtherBallAction with a team, nested ones included
    (a set piece wraps its Play). Returns arrays ``t`` (POSIX seconds), ``team`` (0 home, 1 away),
    ``tag`` and ``done`` (completed: Play ``successfullyCompleted``/``successful``; shots and
    claims always)."""
    t, team, tag, done = [], [], [], []
    root = ET.parse(str(events_path)).getroot()
    for ev in root.iter("Event"):
        when = ev.get("EventTime")
        if not when:
            continue
        ts = datetime.fromisoformat(when).timestamp()
        for el in ev.iter():
            if el.tag not in CONTROL_TAGS + ("OtherBallAction",) or not el.get("Team"):
                continue
            t.append(ts)
            team.append(0 if el.get("Team") == home_id else 1)
            tag.append(el.tag)
            done.append(el.tag != "Play" or el.get("Evaluation", "") in ("successfullyCompleted", "successful"))
    order = np.argsort(np.asarray(t), kind="stable")
    return {"t": np.asarray(t, dtype=np.float64)[order], "team": np.asarray(team, dtype=np.int8)[order],
            "tag": np.asarray(tag)[order], "done": np.asarray(done, dtype=bool)[order]}


def align_dfl(y: np.ndarray, t0: float, fps: float, acts: dict, max_shift_s: float = 3.0) -> tuple:
    """Frame row of every action on the Match grid (row ``i`` is ``t0 + i / fps``), with the clock
    shift that maximises Play-team agreement with ``y``. Returns ``(rows, diag)``; rows off the grid
    are -1. Ties go to the most negative shift, so the result is deterministic."""
    n = len(y)
    base = np.rint((acts["t"] - t0) * fps).astype(np.int64)
    plays = acts["tag"] == "Play"
    alive = (y == H) | (y == A)
    k = int(round(max_shift_s * fps))
    best = None
    for s in range(-k, k + 1):
        r = base + s
        ok = plays & (r >= 0) & (r < n)
        ok[ok] = alive[r[ok]]
        agree = float((y[r[ok]] == acts["team"][ok]).mean()) if ok.any() else 0.0
        if best is None or agree > best[0] + 1e-12:
            best = (agree, s)
    rows = base + best[1]
    rows = np.where((rows >= 0) & (rows < n), rows, -1)
    return rows, {"shift_s": round(best[1] / fps, 3), "play_team_agreement": round(best[0], 4),
                  "n_actions": int(len(rows)), "n_on_grid": int((rows >= 0).sum())}


def metrica_actions(events: list, fps: float, min_carry_s: float = 1.0) -> dict:
    """Actions from ``katena.external.metrica.Event`` objects (1-based tracking frames): every PASS
    (Metrica logs failed passes as BALL LOST), SHOT, and CARRY of at least ``min_carry_s``."""
    rows, team, tag = [], [], []
    for e in events:
        if e.type in ("PASS", "SHOT") or (e.type == "CARRY" and (e.end - e.start) >= min_carry_s * fps):
            rows.append(e.start - 1)
            team.append(e.team)
            tag.append(e.type)
    order = np.argsort(np.asarray(rows, dtype=np.int64), kind="stable")
    return {"rows": np.asarray(rows, dtype=np.int64)[order], "team": np.asarray(team, dtype=np.int8)[order],
            "tag": np.asarray(tag)[order], "done": np.ones(len(rows), dtype=bool)}


# ---------------------------------------------------------------------------------------------
# evidence per flip

def _action_counter(rows: np.ndarray, team: np.ndarray, keep: np.ndarray | None = None):
    """``count(code, lo, hi)``: actions of team ``code`` with row in ``[lo, hi)``."""
    ok = rows >= 0 if keep is None else (rows >= 0) & keep
    by_team = {c: np.sort(rows[ok & (team == c)]) for c in (0, 1)}

    def count(code: int, lo: int, hi: int) -> int:
        r = by_team[code]
        return int(np.searchsorted(r, hi, "left") - np.searchsorted(r, lo, "left"))
    return count


def spell_evidence(y: np.ndarray, rows: np.ndarray, team: np.ndarray, fps: float, tol_s: float = 0.4,
                   kinds=("return", "to_dead"), keep: np.ndarray | None = None) -> dict:
    """Per flip of the given kinds: ``dur`` (s), ``has`` (own-team action in the spell +- tol),
    ``n`` (number of such actions) and ``kind``. ``keep`` masks the actions that count."""
    tol = int(round(tol_s * fps))
    count = _action_counter(np.asarray(rows), np.asarray(team), keep)
    dur, n, kind = [], [], []
    for a, b, k, kd in spell_kinds(y):
        if kd not in kinds:
            continue
        dur.append((b - a) / fps)
        n.append(count(0 if k == H else 1, a - tol, b + tol))
        kind.append(kd)
    n = np.asarray(n, dtype=np.int64)
    return {"dur": np.asarray(dur, dtype=np.float64), "has": n > 0, "n": n, "kind": np.asarray(kind)}


def control_labels(y: np.ndarray, rows: np.ndarray, team: np.ndarray, fps: float, tol_s: float = 0.4,
                   max_s: float = 10.0, tail: bool = True, keep: np.ndarray | None = None) -> np.ndarray:
    """The event-defined convention: every flip shorter than ``max_s`` whose team has no control
    action in it (+- tol) is relabelled to the previous owner, shortest first, until none is left.
    Restart spells, D and X are never changed. No duration threshold decides control, only the
    events; ``max_s`` is a safety cap (flips of 10 s and more carry an action 99% of the time)."""
    tol = int(round(tol_s * fps))
    count = _action_counter(np.asarray(rows), np.asarray(team), keep)
    out = y.copy()
    lim = max_s * fps
    for run in pm.valid_runs(y):
        segs = [list(s) for s in pm.segments(out, [run])]
        while True:
            best = None
            for i in range(1, len(segs)):
                a, b, k = segs[i]
                if k not in (H, A) or segs[i - 1][2] not in (H, A) or (b - a) >= lim:
                    continue
                nxt = segs[i + 1][2] if i + 1 < len(segs) else None
                if nxt not in (H, A) and not tail:
                    continue
                if count(0 if k == H else 1, a - tol, b + tol):
                    continue
                key = (b - a, a)
                if best is None or key < best[0]:
                    best = (key, i)
            if best is None:
                break
            i = best[1]
            target = segs[i - 1][2]
            segs[i][2] = target
            if i + 1 < len(segs) and segs[i + 1][2] == target:
                segs[i][1] = segs[i + 1][1]
                del segs[i + 1]
            segs[i - 1][1] = segs[i][1]
            del segs[i]
        for a, b, k in segs:
            out[a:b] = k
    return out


# ---------------------------------------------------------------------------------------------
# thresholds

GRID = np.round(np.arange(0.2, 8.01, 0.1), 2)


def best_threshold(dur: np.ndarray, has: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Threshold ``t`` maximising the accuracy of 'controlled iff duration >= t' (and the balanced
    accuracy, which ignores the base rate)."""
    grid = GRID if grid is None else np.asarray(grid)
    acc = np.array([((dur >= t) == has).mean() for t in grid])
    bal = np.array([0.5 * ((dur[has] >= t).mean() + (dur[~has] < t).mean())
                    if has.any() and (~has).any() else np.nan for t in grid])
    i, j = int(np.argmax(acc)), int(np.nanargmax(bal))
    return {"accuracy_best_t": float(grid[i]), "accuracy": float(acc[i]),
            "balanced_best_t": float(grid[j]), "balanced_accuracy": float(bal[j]),
            "grid": grid, "acc_curve": acc, "bal_curve": bal}


def evidence_by_bin(dur, has, bins=(0, 0.5, 1, 1.5, 2, 3, 4, 6, 10, np.inf)) -> list[tuple]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (dur >= lo) & (dur < hi)
        out.append((lo, hi, int(sel.sum()), float(has[sel].mean()) if sel.any() else float("nan")))
    return out


def derive_threshold(per: dict, n_boot: int = 500, seed: int = 0) -> dict:
    """Pooled, per-match, leave-one-match-out and match-bootstrap estimates of the best threshold.
    ``per``: {match: (dur, has)}."""
    names = sorted(per)

    def pooled(keys):
        return (np.concatenate([per[k][0] for k in keys]), np.concatenate([per[k][1] for k in keys]))

    dur, has = pooled(names)
    best = best_threshold(dur, has)
    per_match = {n: best_threshold(*per[n])["accuracy_best_t"] for n in names}
    lomo = {}
    for n in names:
        t = best_threshold(*pooled([k for k in names if k != n]))["accuracy_best_t"]
        lomo[n] = (t, float(((per[n][0] >= t) == per[n][1]).mean()))
    rng = np.random.default_rng(seed)
    boot = [best_threshold(*pooled(rng.choice(names, size=len(names), replace=True)))["accuracy_best_t"]
            for _ in range(n_boot)]
    return {"pooled": best, "n_flips": int(len(dur)), "per_match": per_match, "lomo": lomo,
            "bootstrap_median": float(np.median(boot)),
            "bootstrap_p5_p95": tuple(float(v) for v in np.percentile(boot, [5, 95])),
            "by_bin": evidence_by_bin(dur, has)}
