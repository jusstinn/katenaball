"""D4 label-convention tools on synthetic fixtures: sandwich options and control evidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import possession_metrics as pm
from katena.external import control_evidence as ce
from katena.labels import sandwich
from katena.types import A, D, H, X


# --- references ------------------------------------------------------------------------------

def _sandwich_original(y, runs, fps, T):
    """katena.labels.sandwich as of origin/david/infra 6bd2807, verbatim: the default must not move."""
    y = y.copy()
    if T <= 0:
        return y
    lim = T * fps
    for run in runs:
        for _ in range(50):
            segs = pm.segments(y, [run])
            changed, i = False, 1
            while i < len(segs) - 1:
                a, b, c = segs[i - 1], segs[i], segs[i + 1]
                if (b[2] in (H, A) and a[2] == c[2] and a[2] in (H, A) and a[2] != b[2]
                        and (b[1] - b[0]) < lim):
                    y[b[0]:b[1]] = a[2]
                    changed = True
                    i += 2
                else:
                    i += 1
            if not changed:
                break
    return y


def _harmonise_david(y, min_control_s, fps):
    """David's harmonise (origin/david/foundation, shortest_first, protect_restart), condensed."""
    min_len = int(round(float(min_control_s) * int(fps)))
    out = np.asarray(y).astype(np.int8, copy=True)
    if min_len <= 1:
        return out
    for start, stop in pm.valid_runs(out):
        w = out[start:stop]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bd = np.concatenate(([0], br, [len(w)])).tolist()
        segs = [[bd[i], bd[i + 1], int(w[bd[i]])] for i in range(len(bd) - 1)]
        while True:
            best = None
            for i in range(len(segs)):
                a, b, k = segs[i]
                if k not in (H, A) or b - a >= min_len:
                    continue
                prev = segs[i - 1][2] if i > 0 else None
                if prev not in (H, A):
                    continue
                key = (b - a, a)
                if best is None or key < best[0]:
                    best = (key, i, prev)
            if best is None:
                break
            _, i, target = best
            segs[i][2] = target
            if i + 1 < len(segs) and segs[i + 1][2] == segs[i][2]:
                segs[i][1] = segs[i + 1][1]
                del segs[i + 1]
            if i > 0 and segs[i - 1][2] == segs[i][2]:
                segs[i - 1][1] = segs[i][1]
                del segs[i]
        for a, b, k in segs:
            out[start + a:start + b] = k
    return out


def _random_labels(rng, n_segs=60):
    """Random H/A/D sequence with X gaps, short flips common."""
    codes, lens = [], []
    cur = int(rng.integers(0, 3))
    for _ in range(n_segs):
        cur = int(rng.choice([c for c in (H, A, D) if c != cur]))
        codes.append(cur)
        lens.append(int(rng.choice([1, 2, 3, 5, 8, 12, 20, 40])))
    y = np.repeat(np.array(codes, np.int8), lens)
    y[: int(rng.integers(0, 4))] = X
    cut = int(rng.integers(len(y) // 3, 2 * len(y) // 3))
    y[cut:cut + 3] = X
    return y


def _seq(parts):
    return np.concatenate([np.full(n, c, np.int8) for c, n in parts])


# --- sandwich options ------------------------------------------------------------------------

def test_sandwich_default_is_unchanged():
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = _random_labels(rng)
        for T in (0, 0.5, 1, 2, 3, 5):
            assert np.array_equal(sandwich(y, pm.valid_runs(y), 5.0, T), _sandwich_original(y, pm.valid_runs(y), 5.0, T))


def test_tail_merges_touch_then_out_but_never_a_restart():
    y = _seq([(H, 20), (A, 3), (D, 5), (A, 2), (H, 20), (A, 3)])
    out = sandwich(y, pm.valid_runs(y), 5.0, 1.0, tail=True)
    assert (out[20:23] == H).all()                # H then a 3-sample A touch, then dead: stays H
    assert (out[28:30] == A).all()                # the restart possessor after D is kept, however short
    assert (out[50:53] == H).all()                # a flip running into the end of the run is a tail too
    assert np.array_equal(sandwich(y, pm.valid_runs(y), 5.0, 1.0), y)   # default: no sandwich here


def test_shortest_first_with_tail_reproduces_david_harmonise():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = _random_labels(rng)
        for t in (0.04, 0.2, 1.0, 2.0, 3.5):      # not 0.5: round(12.5) == 12 in David's frame count
            got = sandwich(y, pm.valid_runs(y), 25.0, t, tail=True, order="shortest")
            assert np.array_equal(got, _harmonise_david(y, t, 25.0)), t


def test_scan_and_shortest_differ_only_on_chains():
    # H | A 1.0 s | H 0.3 s | A 10 s at 10 fps, T = 2 s
    y = _seq([(H, 50), (A, 10), (H, 3), (A, 100), (D, 5)])
    scan = sandwich(y, pm.valid_runs(y), 10.0, 2.0)
    short = sandwich(y, pm.valid_runs(y), 10.0, 2.0, order="shortest")
    assert (scan[50:63] == H).all() and (scan[63:163] == A).all()    # A owns from its long spell
    assert (short[50:163] == A).all()                                 # A owns from its first touch
    with pytest.raises(ValueError):
        sandwich(y, pm.valid_runs(y), 10.0, 2.0, order="random")


def test_conventions_never_touch_dead_or_unlabelled_frames():
    rng = np.random.default_rng(2)
    for _ in range(50):
        y = _random_labels(rng)
        for kw in ({}, {"tail": True}, {"tail": True, "order": "shortest"}):
            out = sandwich(y, pm.valid_runs(y), 5.0, 3.0, **kw)
            assert np.array_equal(out >= 2, y >= 2) and np.array_equal(out[y >= 2], y[y >= 2])


# --- control evidence ------------------------------------------------------------------------

def test_spell_kinds():
    y = _seq([(X, 2), (H, 10), (A, 3), (H, 10), (A, 4), (D, 5), (H, 6), (A, 7), (X, 1), (A, 3)])
    kinds = [(k, kd) for _, _, k, kd in ce.spell_kinds(y)]
    assert kinds == [(H, "restart"), (A, "return"), (H, "return"), (A, "to_dead"), (H, "restart"),
                     (A, "to_dead"), (A, "restart")]


def test_align_dfl_recovers_a_clock_shift():
    fps = 25.0
    y = _seq([(H, 300), (A, 300), (H, 300), (A, 300)])
    t0 = 1000.0
    true_rows = np.array([150, 299, 300, 450, 599, 600, 899, 900])     # either side of every change
    team = np.array([H, H, A, A, A, H, H, A], np.int8)
    shift = 37                                              # the event clock runs 37 frames early
    acts = {"t": t0 + (true_rows - shift) / fps, "team": team, "tag": np.array(["Play"] * 8),
            "done": np.ones(8, bool)}
    rows, diag = ce.align_dfl(y, t0, fps, acts, max_shift_s=3.0)
    assert diag["play_team_agreement"] == 1.0 and diag["shift_s"] == pytest.approx(37 / fps)
    assert rows.tolist() == true_rows.tolist()


def test_spell_evidence_and_keep_mask():
    fps = 10.0
    y = _seq([(H, 50), (A, 10), (H, 50), (A, 12), (D, 10)])
    rows = np.array([55, 115, 30])                          # A action in each A flip, an H action
    team = np.array([A, A, H], np.int8)
    ev = ce.spell_evidence(y, rows, team, fps, tol_s=0.0)
    assert ev["kind"].tolist() == ["return", "return", "to_dead"]     # A, then H (between two A), then A
    assert ev["has"].tolist() == [True, False, True] and ev["dur"].tolist() == [1.0, 5.0, 1.2]
    ev2 = ce.spell_evidence(y, rows, team, fps, tol_s=0.0, keep=np.array([False, True, True]))
    assert ev2["has"].tolist() == [False, False, True]
    ev3 = ce.spell_evidence(y, np.array([49]), np.array([A], np.int8), fps, tol_s=0.2)   # tol reaches back
    assert ev3["has"].tolist() == [True, False, False]
    assert ce.spell_evidence(y, rows, team, fps, kinds=("to_dead",))["dur"].tolist() == [1.2]


def test_control_labels_remove_flips_without_action_only():
    fps = 10.0
    # D 0-4 | A 5-7 restart | H 8-57 | A 58-67 | H 68-117 | A 118-129 | H 130-169 | A 170-369 | H 370-389 | A 390-393 | D
    y = _seq([(D, 5), (A, 3), (H, 50), (A, 10), (H, 50), (A, 12), (H, 40), (A, 200), (H, 20), (A, 4), (D, 5)])
    rows = np.array([30, 90, 120, 150, 380])
    team = np.array([H, H, A, H, H], np.int8)
    out = ce.control_labels(y, rows, team, fps, tol_s=0.0, max_s=10.0)
    assert (out[5:8] == A).all()                             # restart spell kept without any action
    assert (out[8:118] == H).all()                           # A 58-67 had no action: a touch, not a possession
    assert (out[118:130] == A).all()                         # A acted: kept
    assert (out[170:370] == A).all()                         # longer than the cap: kept without an action
    assert (out[370:394] == H).all()                         # tail flip without action merged
    assert np.array_equal(out >= 2, y >= 2)


def test_best_threshold_and_derive():
    dur = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 8.0, 12.0])
    has = np.array([False, False, False, False, True, True, True, True])
    b = ce.best_threshold(dur, has)
    assert 2.5 < b["accuracy_best_t"] <= 3.5 and b["accuracy"] == 1.0
    per = {"a": (dur, has), "b": (dur + 0.1, has), "c": (dur - 0.1, has)}
    d = ce.derive_threshold(per, n_boot=20)
    assert set(d["lomo"]) == {"a", "b", "c"} and d["n_flips"] == 24
    assert d["bootstrap_p5_p95"][0] <= d["bootstrap_median"] <= d["bootstrap_p5_p95"][1]


def test_read_dfl_actions(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?><PutDataRequest>
    <Event EventTime="2023-05-27T15:30:15.000+02:00"><Play Team="HOME" Player="p1" Evaluation="successfullyCompleted"><Pass/></Play></Event>
    <Event EventTime="2023-05-27T15:30:16.000+02:00"><ThrowIn Team="AWAY"><Play Team="AWAY" Evaluation="unsuccessful"/></ThrowIn></Event>
    <Event EventTime="2023-05-27T15:30:17.000+02:00"><OtherBallAction Team="AWAY" DefensiveClearance="true"/></Event>
    <Event EventTime="2023-05-27T15:30:18.000+02:00"><TacklingGame WinnerTeam="HOME" PossessionChange="true"/></Event>
    <Event EventTime="2023-05-27T15:30:19.000+02:00"><BallClaiming Team="HOME" Type="InterceptedBall"/></Event>
    </PutDataRequest>"""
    p = tmp_path / "events.xml"
    p.write_text(xml, encoding="utf-8")
    acts = ce.read_dfl_actions(p, "HOME")
    assert acts["tag"].tolist() == ["Play", "Play", "OtherBallAction", "BallClaiming"]
    assert acts["team"].tolist() == [0, 1, 1, 0] and acts["done"].tolist() == [True, False, True, True]
    assert acts["t"][1] - acts["t"][0] == pytest.approx(1.0)


def test_metrica_actions_from_events():
    E = SimpleNamespace
    ev = [E(type="PASS", start=11, end=20, team=0), E(type="CARRY", start=30, end=40, team=1),
          E(type="CARRY", start=50, end=80, team=1), E(type="RECOVERY", start=90, end=90, team=0),
          E(type="SHOT", start=95, end=99, team=0)]
    acts = ce.metrica_actions(ev, 25.0, min_carry_s=1.0)
    assert acts["rows"].tolist() == [10, 49, 94] and acts["team"].tolist() == [0, 1, 0]
