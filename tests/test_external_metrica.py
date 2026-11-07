"""Event-derived labels for Metrica open data, on a hand-made event list. Offline, fast."""
import numpy as np

from katena.external.metrica_open import is_restart, labels_from_events, stop_marker_frame
from katena.match import A, D, H, X

TEAMS = {"TA": 0, "TB": 1}


def ev(i, team, typ, start, end=None, sub=(), period=1):
    return {"index": i, "team": {"id": team} if team else None, "type": {"name": typ},
            "subtypes": [{"name": s} for s in sub] or None, "period": period,
            "start": {"frame": start}, "end": {"frame": end if end is not None else start}}


EVENTS = [
    ev(1, "TA", "SET PIECE", 10, sub=("KICK OFF",)),
    ev(2, "TA", "PASS", 10, 30),
    ev(3, "TA", "CARRY", 30, 50),
    ev(4, "TA", "BALL OUT", 50, 60),                 # out at 60 -> dead from 61
    ev(5, "TB", "SET PIECE", 100, sub=("THROW IN",)),
    ev(6, "TB", "PASS", 100, 120),
    ev(7, "TB", "BALL LOST", 120, 140),               # loose ball stays with TB ...
    ev(8, "TA", "RECOVERY", 135),                     # ... until TA recovers at 135
    ev(9, "TA", "CARRY", 135, 160),
    ev(10, "TB", "CHALLENGE", 160, sub=("GROUND", "FAULT", "LOST")),
    ev(11, "TA", "FAULT RECEIVED", 160),              # foul at 160 -> dead from 161
    ev(12, "TB", "CARD", 170, sub=("YELLOW",)),
    ev(13, "TA", "SET PIECE", 200, sub=("FREE KICK",)),
    ev(14, "TA", "PASS", 200, 230),
    ev(15, "TB", "PASS", 260, 280, sub=("GOAL KICK",)),  # untagged stoppage: dead from 231
    ev(16, "TB", "CARRY", 280, 290),
    ev(17, "TB", "BALL LOST", 290, 295, sub=("END HALF",)),  # whistle at 295 -> dead from 296
]


def test_restart_and_marker_classification():
    assert is_restart(EVENTS[0]) and is_restart(EVENTS[14]) and not is_restart(EVENTS[1])
    assert stop_marker_frame(EVENTS[3]) == 60
    assert stop_marker_frame(EVENTS[10]) == 160
    assert stop_marker_frame(EVENTS[9]) == 160
    assert stop_marker_frame(EVENTS[16]) == 295
    assert stop_marker_frame(EVENTS[6]) is None


def test_labels_from_events():
    y, diag = labels_from_events(EVENTS, TEAMS, periods=[(1, 1, 320)], f0=1, n_src=330)
    f = lambda frame: y[frame - 1]  # noqa: E731  (index = frame - f0)
    assert all(f(t) == D for t in range(1, 10))           # before kickoff
    assert all(f(t) == H for t in range(10, 61))          # TA from kickoff to the line
    assert all(f(t) == D for t in range(61, 100))         # out of play until the throw-in
    assert all(f(t) == A for t in range(100, 135))        # TB incl. the loose ball
    assert all(f(t) == H for t in range(135, 161))        # TA recovered
    assert all(f(t) == D for t in range(161, 200))        # foul to free kick (card ignored)
    assert all(f(t) == H for t in range(200, 231))
    assert all(f(t) == D for t in range(231, 260))        # no marker: last action end + 1
    assert all(f(t) == A for t in range(260, 296))
    assert all(f(t) == D for t in range(296, 321))        # after the final whistle
    assert all(f(t) == X for t in range(321, 331))        # outside the period
    assert diag["restarts"] == 4
    assert diag["stoppages_with_marker"] == 3 and diag["stoppages_without_marker"] == 1
    assert diag["alive_without_owner"] == 0


def test_second_period_is_independent():
    evs = [dict(e) for e in EVENTS[:4]] + [
        ev(30, "TB", "SET PIECE", 400, sub=("KICK OFF",), period=2),
        ev(31, "TB", "PASS", 400, 420, period=2)]
    y, _ = labels_from_events(evs, TEAMS, periods=[(1, 1, 100), (2, 350, 450)], f0=1,
                              n_src=460)
    assert (y[349:399] == D).all() and (y[399:420] == A).all()  # frames 350-399, 400-420
    assert (y[420:450] == D).all()  # nothing after the pass ended at 420
    assert (y[100:349] == X).all()
    assert np.isin(y[:100], (H, D)).all()
