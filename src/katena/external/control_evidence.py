"""Event-based evidence of *controlled* possession, used to derive `harmonise`'s threshold.

For every possession spell that `harmonise` could remove (an opponent spell that starts straight
from the owner's spell: kinds "return" and "to_dead" in `flip_table`), ask the provider's event
data whether that team made a deliberate on-ball action during it:

- DFL: a `Play` (pass or cross attempt), `ShotAtGoal` or `BallClaiming` by the spell's team.
  `OtherBallAction` (clearances, touches) and duels do not count.
- Metrica: a completed `PASS`, a `SHOT`, or a `CARRY` of at least `min_carry_s` by the team.

The duration threshold that best separates spells with and without such an action is an
external, definition-based estimate of "a player keeps the ball for a few seconds". It never
looks at the competition match.

DFL event clocks are not frame-synchronised with tracking, so a per-match offset is estimated
first: the shift (within +-`max_shift_s`) that maximises agreement between each Play's team and
the tracking's BallPossession at that frame.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from lxml import etree

from ..match import A, D, H
from .harmonise import _runs, _segments

DFL_CONTROL_TAGS = {"Play", "ShotAtGoal", "BallClaiming"}


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def read_dfl_actions(events_path) -> list[tuple[float, str, str]]:
    """[(unix time, tag, team id)] for every Play / ShotAtGoal / BallClaiming / OtherBallAction."""
    out = []
    root = etree.parse(str(events_path)).getroot()
    for ev in root.iter("Event"):
        t = ev.get("EventTime")
        if not t:
            continue
        for el in ev.iter():
            if el.tag in DFL_CONTROL_TAGS | {"OtherBallAction"} and el.get("Team"):
                out.append((_ts(t), el.tag, el.get("Team")))
    out.sort()
    return out


def dfl_action_frames(m, actions, home_id, max_shift_s=3.0, step_s=0.04):
    """Map DFL actions to Match rows with the best clock offset. Returns (rows, code, tag, diag)."""
    fps = m.fps
    fid = m.meta["frame_id"]
    t0 = {int(h): _ts(v["first_frame_time"]) for h, v in m.meta["diag"]["halves"].items()}
    first = {int(h): int(v["first_N"]) for h, v in m.meta["diag"]["halves"].items()}
    last = {int(h): int(v["last_N"]) for h, v in m.meta["diag"]["halves"].items()}
    row_of = {int(f): r for r, f in enumerate(fid.tolist()) if f >= 0}
    times = np.array([a[0] for a in actions])
    code = np.array([0 if a[2] == home_id else 1 for a in actions], dtype=np.int8)
    tags = np.array([a[1] for a in actions])
    owner = m.meta["owner_raw"]
    alive = np.isin(m.y, (H, A))

    def rows_for(shift):
        r = np.full(len(times), -1, dtype=np.int64)
        for h in t0:
            n = first[h] + np.round((times + shift - t0[h]) * fps).astype(np.int64)
            inside = (n >= first[h]) & (n <= last[h])
            r[inside] = [row_of[int(v)] for v in n[inside]]
        return r

    best = None
    for shift in np.arange(-max_shift_s, max_shift_s + 1e-9, step_s):
        r = rows_for(shift)
        ok = (r >= 0) & (tags == "Play")
        ok &= alive[np.maximum(r, 0)]
        agree = float((owner[r[ok]] == code[ok]).mean()) if ok.any() else 0.0
        if best is None or agree > best[0] + 1e-12:
            best = (agree, float(shift))
    rows = rows_for(best[1])
    return rows, code, tags, {"shift_s": round(best[1], 3), "play_owner_agreement": best[0],
                              "n_actions": int(len(times)), "n_mapped": int((rows >= 0).sum())}


def read_metrica_actions(events_json, min_carry_s=1.0, fps=25) -> list[tuple[int, int, str]]:
    """[(source frame, team id, kind)] for completed passes, shots and carries >= min_carry_s."""
    with open(events_json, encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    out = []
    for e in data:
        t = e["type"]["name"]
        team = (e.get("team") or {}).get("id")
        if team is None:
            continue
        a, b = int(e["start"]["frame"]), int(e["end"]["frame"])
        if t in ("PASS", "SHOT") or (t == "CARRY" and (b - a) >= min_carry_s * fps):
            out.append((a, team, t))
    return out


def spell_evidence(y, action_rows, action_code, fps=25, tol_s=0.4,
                   kinds=("return", "to_dead")) -> tuple[np.ndarray, np.ndarray]:
    """(durations_s, has_action) for spells of the given flip kinds.

    has_action: the spell's own team has at least one control action with row in
    [start - tol, stop + tol).
    """
    tol = int(round(tol_s * fps))
    order = np.argsort(action_rows, kind="stable")
    ar, ac = np.asarray(action_rows)[order], np.asarray(action_code)[order]
    keep = ar >= 0
    ar, ac = ar[keep], ac[keep]
    dur, has = [], []
    for start, stop in _runs(np.asarray(y)):
        segs = _segments(np.asarray(y)[start:stop])
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
                kind = "to_dead" if nxt in (D, None) else "other"
            if kind not in kinds:
                continue
            lo = np.searchsorted(ar, start + a - tol)
            hi = np.searchsorted(ar, start + b + tol)
            team = 0 if k == H else 1
            dur.append((b - a) / fps)
            has.append(bool((ac[lo:hi] == team).any()))
    return np.asarray(dur), np.asarray(has)


def best_threshold(dur, has, grid=None) -> dict:
    """Threshold t maximising accuracy of the rule 'controlled iff duration >= t'."""
    grid = np.round(np.arange(0.2, 8.01, 0.1), 2) if grid is None else np.asarray(grid)
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
        out.append((lo, hi, int(sel.sum()), float(has[sel].mean()) if sel.any() else np.nan))
    return out


def corpus_evidence(matches, root) -> dict:
    """{match name: (durations_s, has_action)} for the cached corpus (needs the event files)."""
    from .epts import read_epts_metadata, rows_for_frames
    from .idsse import idsse_paths, read_dfl_meta
    from .metrica_open import game3_paths

    out, align = {}, {}
    for m in matches:
        if m.meta["source"] == "idsse":
            p = idsse_paths(m.meta["match_id"].split("-")[-1], root)
            teams = read_dfl_meta(p["meta"])["teams"]
            home_id = next(t for t, v in teams.items() if v["code"] == 0)
            acts = [a for a in read_dfl_actions(p["event"]) if a[1] in DFL_CONTROL_TAGS]
            rows, code, _, diag = dfl_action_frames(m, acts, home_id)
            align[m.name] = diag
        elif m.meta["source"] == "metrica-open":
            p = game3_paths(root)
            home_id = read_epts_metadata(p["meta"]).teams[0].team_id
            acts = read_metrica_actions(p["events"], fps=m.fps)
            rows = rows_for_frames(m.meta["frame_id"], [a[0] for a in acts])
            code = np.array([0 if a[1] == home_id else 1 for a in acts], dtype=np.int8)
        else:
            continue
        out[m.name] = spell_evidence(m.y, rows, code, fps=m.fps)
    return {"evidence": out, "alignment": align}


def derive_threshold(per: dict, n_boot: int = 500, seed: int = 0) -> dict:
    """Pooled, leave-one-match-out and match-bootstrap estimates of the best threshold."""
    names = sorted(per)

    def pooled(keys):
        return (np.concatenate([per[k][0] for k in keys]),
                np.concatenate([per[k][1] for k in keys]))

    dur, has = pooled(names)
    best = best_threshold(dur, has)
    per_match = {n: best_threshold(*per[n])["accuracy_best_t"] for n in names}
    lomo = {}
    for n in names:
        t = best_threshold(*pooled([k for k in names if k != n]))["accuracy_best_t"]
        lomo[n] = (t, float(((per[n][0] >= t) == per[n][1]).mean()))
    rng = np.random.default_rng(seed)
    boot = [best_threshold(*pooled(rng.choice(names, size=len(names), replace=True)))
            ["accuracy_best_t"] for _ in range(n_boot)]
    return {"pooled": best, "n_flips": int(len(dur)), "per_match": per_match, "lomo": lomo,
            "bootstrap_median": float(np.median(boot)),
            "bootstrap_p5_p95": tuple(float(v) for v in np.percentile(boot, [5, 95])),
            "by_bin": evidence_by_bin(dur, has)}
