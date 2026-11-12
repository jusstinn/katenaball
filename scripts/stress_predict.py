#!/usr/bin/env python
"""Does the submission path survive tracking it has never seen? An exported external match, damaged on purpose,
through ``katena.pipeline.label_match`` with the trained model.

    export KATENA_WORK=D:/katena/work-submit
    python scripts/stress_predict.py --dir D:/katena/dressed/J03WMX --spec F3 --out runs/stress_F3.json
    python scripts/stress_predict.py --dir D:/katena/dressed/J03WMX --spec F --only as_exported,target_like

Why. Until the real run, the provided match has been read by the loader check only: no feature, model or decoder
code has run on it, and nobody may look at it to debug (CLAUDE.md). The training worlds corrupt external matches
on the 5 Hz grid, in the ways the degrader was built for. This script instead damages the native-rate match that
``predict`` reads, in ways the degrader does not produce (no ball for a half, nothing tracked for minutes, one
team gone, positions far off the pitch, infinities, ghosts, frozen players, teams that do not change ends, ...),
and asks one question of each case: does ``label_match`` finish and write a valid file?

What a case must do to pass: ``label_match`` returns; every frame inside the graded halves is H, A or D and every
other frame is X; the CSV passes the official checks (``pipeline.write_submission``); labelling the same case
twice gives the same labels when ``--twice`` is given. It says nothing about accuracy: the labels of a damaged
match are not scored. One case also goes through the file format (``dress.write_bundle`` and ``load_target``) and
must come back equal.

The match is the dressed external match of ``scripts/dressed_check.py export`` (IDSSE ``DFL-MAT-J03WMX``); the
model is the one ``make_submission.py train`` saved for ``--spec`` under ``KATENA_WORK``, read and never
written. External data only: the competition data directory is refused. Output is aggregates only.

Exit status: 0 every case passed, 1 a case failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.external.common import infer_attack_dir  # noqa: E402
from katena.types import Match  # noqa: E402

SEED = 20261112
SLOTS_PER_TEAM = 23            # DATA.md: 46 anonymous slots


# ----------------------------------------------------------------------------------------------
# helpers

def clone(m: Match) -> Match:
    return copy.deepcopy(m)


def finish(m: Match) -> Match:
    """What ``load_target`` would have made of these arrays: the attack directions come from the positions."""
    m.P = np.ascontiguousarray(m.P, dtype=np.float32)
    m.speed = np.ascontiguousarray(m.speed, dtype=np.float32)
    m.ball = np.ascontiguousarray(m.ball, dtype=np.float32)
    m.team = np.asarray(m.team, dtype=np.int8)
    m.attack_dir = {h: infer_attack_dir(m.P, m.team, a, b, m.pitch[0]) for h, a, b in m.half_blocks}
    m.y = None
    m.validate()
    return m


def inside(m: Match) -> np.ndarray:
    out = np.zeros(m.n, dtype=bool)
    for _, a, b in m.half_blocks:
        out[a:b + 1] = True
    return out


def bursts(n: int, share: float, mean_len: float, rng) -> np.ndarray:
    """bool[n]: True in bursts of geometric length (mean ``mean_len`` frames) covering about ``share``."""
    if share <= 0:
        return np.zeros(n, dtype=bool)
    if share >= 1:
        return np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    mean_gap = mean_len * (1 - share) / share
    i = int(rng.geometric(1.0 / max(mean_gap, 1.0)))
    while i < n:
        k = int(rng.geometric(1.0 / mean_len))
        out[i:i + k] = True
        i += k + int(rng.geometric(1.0 / max(mean_gap, 1.0)))
    return out


def span(m: Match, half: int, start: float = 0.0, length_s: float | None = None) -> slice:
    """Frames of a half block from the fraction ``start`` of its length on: ``length_s`` seconds of them (at most
    a quarter of the half, so a short test match keeps its shape), or the rest of the half when None."""
    a, b = next((a, b) for h, a, b in m.half_blocks if h == half)
    n = b - a + 1
    s = a + int(start * n)
    if length_s is None:
        return slice(s, b + 1)
    return slice(s, min(b + 1, s + max(1, min(int(length_s * m.fps), n // 4))))


def anonymous_slots(m: Match, rng, mean_s: float = 30.0, gap_share: float = 0.3) -> Match:
    """The match on 46 anonymous slots, 23 per team: each player's track is cut into fragments (mean ``mean_s``
    seconds), some followed by a short gap, and each fragment goes to a free slot of its team, chosen at random."""
    n, fps = m.n, m.fps
    S = 2 * SLOTS_PER_TEAM
    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    speed = np.full((n, S), np.nan, dtype=np.float32)
    team = np.repeat(np.array([0, 1], dtype=np.int8), SLOTS_PER_TEAM)
    for t in (0, 1):
        segs = []
        for s in np.flatnonzero(m.team == t):
            seen = np.flatnonzero(np.isfinite(m.P[:, s, 0]))
            if len(seen) == 0:
                continue
            i, last = int(seen[0]), int(seen[-1])
            while i <= last:
                k = max(int(2 * fps), int(rng.exponential(mean_s * fps)))
                segs.append((i, min(last + 1, i + k), int(s)))
                i += k
                if rng.random() < gap_share:
                    i += int(rng.exponential(2 * fps))
        segs.sort()
        free_at = np.zeros(SLOTS_PER_TEAM, dtype=np.int64)
        for a, b, s in segs:
            free = np.flatnonzero(free_at <= a)
            if len(free) == 0:
                continue                          # more fragments than slots at this instant: dropped
            j = int(rng.choice(free))
            col = t * SLOTS_PER_TEAM + j
            P[a:b, col] = m.P[a:b, s]
            speed[a:b, col] = m.speed[a:b, s]
            free_at[j] = b
    out = clone(m)
    out.P, out.speed, out.team = P, speed, team
    return out


def target_like(m: Match, rng) -> Match:
    """46 anonymous slots and the ball missing on about 41% of frames, in bursts of about 2 s (DATA.md's
    aggregates). Every other case starts from this."""
    out = anonymous_slots(m, rng)
    out.ball[bursts(out.n, 0.41, 2 * out.fps, rng)] = np.nan
    return out


# ----------------------------------------------------------------------------------------------
# the cases: each takes the target-like match (a private copy) and a generator, and returns a Match

def c_ball_absent_first_half(m, rng):
    m.ball[span(m, 1)] = np.nan
    return m


def c_ball_absent_everywhere(m, rng):
    m.ball[:] = np.nan
    return m


def c_ball_seen_10_percent(m, rng):
    m.ball[~bursts(m.n, 0.10, 1 * m.fps, rng)] = np.nan
    return m


def c_ball_flickers(m, rng):
    m.ball[rng.random(m.n) < 0.5] = np.nan          # single frames, no bursts
    return m


def c_players_absent_5_minutes(m, rng):
    for h, _, _ in m.half_blocks:
        s = span(m, h, 0.2, 300)
        m.P[s], m.speed[s] = np.nan, np.nan
    return m


def c_everything_absent_5_minutes(m, rng):
    for h, _, _ in m.half_blocks:
        s = span(m, h, 0.2, 300)
        m.P[s], m.speed[s], m.ball[s] = np.nan, np.nan, np.nan
    return m


def c_half_edges_empty(m, rng):
    for h, a, b in m.half_blocks:
        first = span(m, h, 0.0, 120)
        k = first.stop - first.start
        for s in (first, slice(b + 1 - k, b + 1)):
            m.P[s], m.speed[s], m.ball[s] = np.nan, np.nan, np.nan
    return m


def c_away_team_absent_second_half(m, rng):
    s = span(m, 2)
    m.P[s, m.team == 1], m.speed[s, m.team == 1] = np.nan, np.nan
    return m


def c_home_team_absent_everywhere(m, rng):
    m.P[:, m.team == 0], m.speed[:, m.team == 0] = np.nan, np.nan
    return m


def c_three_slots_per_team(m, rng):
    keep = np.concatenate([np.flatnonzero(m.team == t)[:3] for t in (0, 1)])
    drop = np.setdiff1d(np.arange(m.n_slots), keep)
    m.P[:, drop], m.speed[:, drop] = np.nan, np.nan
    return m


def c_tracks_flicker(m, rng):
    off = rng.random(m.speed.shape) < 0.7             # every slot seen on 30% of frames, independently
    m.P[off], m.speed[off] = np.nan, np.nan
    return m


def c_ball_off_pitch(m, rng):
    L, W = m.pitch
    seen = np.flatnonzero(np.isfinite(m.ball[:, 0]))
    near = rng.choice(seen, size=len(seen) // 50, replace=False)
    m.ball[near] = np.stack([rng.uniform(-30, L + 30, len(near)), rng.uniform(-30, W + 30, len(near))], 1)
    far = rng.choice(seen, size=len(seen) // 200, replace=False)
    m.ball[far] = rng.uniform(-1e4, 1e4, (len(far), 2))
    return m


def c_players_off_pitch(m, rng):
    hit = np.isfinite(m.P[..., 0]) & (rng.random(m.speed.shape) < 0.01)
    m.P[hit] = rng.uniform(-500, 500, (int(hit.sum()), 2)).astype(np.float32)
    return m


def c_ghost_duplicates(m, rng):
    for j in range(8):                                # eight slots repeat another slot's samples exactly
        src, dst = int(rng.integers(m.n_slots)), int(rng.integers(m.n_slots))
        a = int(rng.integers(0, m.n // 2))
        m.P[a:a + m.n // 3, dst], m.speed[a:a + m.n // 3, dst] = m.P[a:a + m.n // 3, src], m.speed[a:a + m.n // 3, src]
    return m


def c_everyone_on_one_spot(m, rng):
    s = span(m, 1, 0.35, 300)
    m.P[s] = np.where(np.isfinite(m.P[s]), np.float32(40.0), np.nan)
    return m


def c_speed_absent(m, rng):
    m.speed[:] = np.nan
    return m


def c_speed_garbage(m, rng):
    m.speed[:] = np.where(np.isfinite(m.speed), rng.uniform(-50, 500, m.speed.shape), np.nan)
    return m


def c_players_frozen_10_minutes(m, rng):
    s = span(m, 1, 0.5, 600)
    m.P[s] = m.P[s.start]
    return m


def c_ball_frozen_everywhere(m, rng):
    m.ball[:] = np.where(np.isfinite(m.ball), np.array([m.pitch[0] / 2, m.pitch[1] / 2], dtype=np.float32), np.nan)
    return m


def c_ball_random_every_frame(m, rng):
    seen = np.isfinite(m.ball[:, 0])
    m.ball[seen] = np.stack([rng.uniform(0, m.pitch[0], int(seen.sum())), rng.uniform(0, m.pitch[1], int(seen.sum()))], 1)
    return m


def c_heavy_jitter(m, rng):
    m.P += rng.normal(0, 1.0, m.P.shape).astype(np.float32)
    m.ball += rng.normal(0, 1.0, m.ball.shape).astype(np.float32)
    return m


def c_teams_do_not_change_ends(m, rng):
    s = span(m, 2)
    m.P[s, :, 0] = m.pitch[0] - m.P[s, :, 0]
    m.ball[s, 0] = m.pitch[0] - m.ball[s, 0]
    return m


def c_teams_overlap(m, rng):
    m.P[:, :, 0] = m.pitch[0] / 2 + (m.P[:, :, 0] - m.pitch[0] / 2) * 0.01     # both centroids within 2 m
    return m


def c_tracked_outside_the_halves(m, rng):
    ins = inside(m)
    out_idx, in_idx = np.flatnonzero(~ins), np.flatnonzero(ins)
    src = in_idx[np.arange(len(out_idx)) % len(in_idx)]
    m.P[out_idx], m.speed[out_idx], m.ball[out_idx] = m.P[src], m.speed[src], m.ball[src]
    return m


def c_ball_one_coordinate(m, rng):
    m.ball[rng.random(m.n) < 0.1, 1] = np.nan         # x present, y missing
    m.ball[rng.random(m.n) < 0.1, 0] = np.nan         # y present, x missing
    return m


def c_infinite_values(m, rng):
    seen = np.flatnonzero(np.isfinite(m.ball[:, 0]))
    m.ball[rng.choice(seen, 200, replace=False)] = np.inf
    m.ball[rng.choice(seen, 200, replace=False), 0] = -np.inf
    hit = np.isfinite(m.P[..., 0]) & (rng.random(m.speed.shape) < 1e-4)
    m.P[hit] = np.inf
    m.speed[hit] = np.inf
    return m


CASES = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("c_") and callable(fn)}
SPECIAL = ("as_exported", "target_like", "target_like_through_the_file")


# ----------------------------------------------------------------------------------------------
# running one case

def check_codes(m: Match, codes: np.ndarray) -> list:
    bad = []
    if codes.shape != (m.n,) or codes.dtype != np.int8:
        return [f"codes are {codes.dtype}{codes.shape}, expected int8({m.n},)"]
    ins = inside(m)
    if not np.isin(codes[ins], (0, 1, 2)).all():
        bad.append("a graded frame is not H, A or D")
    if not (codes[~ins] == 3).all():
        bad.append("a frame outside the halves is not X")
    return bad


def run_case(name: str, m: Match, model, PL, twice: bool) -> dict:
    row = {"case": name, "ok": False, "slots": int(m.n_slots),
           "ball_missing_in_halves": round(float(np.isnan(m.ball[inside(m), 0]).mean()), 3),
           "attack_dir": {str(k): int(v) for k, v in m.attack_dir.items()}}
    t0 = time.time()
    try:
        with np.errstate(all="ignore"):
            codes, rep = PL.label_match(model, m)
        problems = check_codes(m, codes)
        with tempfile.TemporaryDirectory(prefix="katena-stress-") as tmp:
            sub = PL.write_submission(codes, Path(tmp) / "stress.csv", m.n)     # raises if the official checks fail
        if twice:
            codes2, _ = PL.label_match(model, m)
            if not np.array_equal(codes, codes2):
                problems.append("a second run gave other labels")
        shares = rep["label_share_in_halves"]
        row.update(ok=not problems, problems=problems, rows=sub["rows"], spells_5hz=rep["spells_5hz"],
                   share_H=shares["H"], share_A=shares["A"], share_D=shares["D"],
                   touches=rep.get("touch_events"), seconds=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001  an external match: the message may be shown
        tb = traceback.extract_tb(e.__traceback__)[-1]
        row.update(problems=[f"{type(e).__name__}: {str(e)[:300]}"],
                   where=f"{Path(tb.filename).name}:{tb.lineno} in {tb.name}", seconds=round(time.time() - t0, 1))
    return row


def through_the_file(m: Match, model, PL) -> dict:
    """The target-like match written as an ATD bundle and read back: every field equal, and the same labels."""
    from katena.target import dress
    from katena.target.atd import load_target
    row = {"case": "target_like_through_the_file", "ok": False, "slots": int(m.n_slots)}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="katena-stress-") as tmp:
            tl = dress.write_bundle(m, Path(tmp) / "bundle")
            back = load_target(Path(tmp) / "bundle", timeline=tl)
        same = dress.compare(m, back)
        direct, _ = PL.label_match(model, m)
        via, _ = PL.label_match(model, back)
        problems = [f"field differs after the round trip: {k}" for k, v in same.items() if not v]
        if not np.array_equal(direct, via):
            problems.append(f"{int((direct != via).sum())} labels differ between the arrays and the file")
        row.update(ok=not problems, problems=problems, seconds=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        row.update(problems=[f"{type(e).__name__}: {str(e)[:300]}"],
                   where=f"{Path(tb.filename).name}:{tb.lineno} in {tb.name}", seconds=round(time.time() - t0, 1))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, type=Path, help="the folder dressed_check.py export wrote")
    ap.add_argument("--spec", default=None, help="the model spec (default: the submission model)")
    ap.add_argument("--only", default="", help="comma-separated case names (default: all)")
    ap.add_argument("--twice", action="store_true", help="label each case twice and compare")
    ap.add_argument("--out", type=Path, default=None, help="write the table as JSON")
    ap.add_argument("--list", action="store_true", help="print the case names and stop")
    a = ap.parse_args(argv)
    names = list(SPECIAL) + sorted(CASES)
    if a.list:
        print("\n".join(names))
        return 0
    only = [s for s in a.only.split(",") if s]
    unknown = sorted(set(only) - set(names))
    if unknown:
        print(f"unknown case(s): {unknown}")
        return 2

    from katena import pipeline as PL
    from katena.target.atd import Timeline, _games, load_target
    bundle = (a.dir / "bundle").resolve()
    if bundle in {Path(_games().DATA_DIR).resolve(), (REPO / "data").resolve()} or REPO / "data" in bundle.parents:
        print("refused: this check is for an exported external match, never the competition data")
        return 2
    tl = Timeline.from_json(json.loads((a.dir / "timeline.json").read_text(encoding="utf-8")))
    sp = PL.spec_of(a.spec)
    model = PL.load_model(spec=sp.name)
    base = load_target(bundle, timeline=tl)
    print(f"[stress] spec {sp.name}, match {tl.name}: {base.n} frames, {base.n_slots} slots; "
          f"cache {os.environ.get('KATENA_WORK')}", flush=True)
    tlike = finish(target_like(clone(base), np.random.default_rng(SEED)))

    rows = []
    for i, name in enumerate(names):
        if only and name not in only:
            continue
        if name == "as_exported":
            row = run_case(name, base, model, PL, a.twice)
        elif name == "target_like":
            row = run_case(name, tlike, model, PL, a.twice)
        elif name == "target_like_through_the_file":
            row = through_the_file(tlike, model, PL)
        else:
            try:
                m = finish(CASES[name](clone(tlike), np.random.default_rng(SEED + 1 + i)))
            except Exception as e:  # noqa: BLE001
                row = {"case": name, "ok": False, "problems": [f"building the case: {type(e).__name__}: {str(e)[:200]}"]}
            else:
                row = run_case(name, m, model, PL, a.twice)
        rows.append(row)
        tail = "" if row["ok"] else "  " + "; ".join(row.get("problems", [])) + ("  @ " + row["where"] if "where" in row else "")
        print(f"  {'PASS' if row['ok'] else 'FAIL'}  {name:34s} {row.get('seconds', 0):6.1f}s  "
              f"D {row.get('share_D', float('nan')):.3f}  spells {row.get('spells_5hz', '-')}{tail}", flush=True)

    failed = [r["case"] for r in rows if not r["ok"]]
    print(f"[stress] {len(rows) - len(failed)} of {len(rows)} cases passed" + (f"; failed: {failed}" if failed else ""))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"spec": sp.name, "match": tl.name, "seed": SEED, "cases": rows}, indent=1) + "\n",
                         encoding="utf-8", newline="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
