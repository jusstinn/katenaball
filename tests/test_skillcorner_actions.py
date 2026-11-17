"""Candidate A's label rule (``katena.external.skillcorner_actions``). Synthetic events and tracking only."""

from __future__ import annotations

import numpy as np

from katena.external import skillcorner_actions as SA
from katena.external.skillcorner_actions import Possession as Po
from katena.types import A, D, H, X

PITCH = (100.0, 60.0)


def _labels(poss, n=400, ball=None, tracked=None, periods=((1, 0, 399),)):
    ball = np.full((n, 2), 50.0, np.float32) if ball is None else ball
    tracked = np.ones(n, bool) if tracked is None else tracked
    return SA.action_labels(list(poss), list(periods), n, ball, tracked, PITCH, fps=10.0)


def test_owner_changes_at_the_other_teams_first_touch_and_holds_through_flight():
    y = _labels([Po(10, 20, 1, 0), Po(35, 40, 1, 0), Po(60, 62, 1, 1), Po(80, 90, 1, 0)])
    assert (y[:10] == X).all()                       # before the first possession of the period
    assert (y[10:60] == H).all()                     # the pass between two home players is home's
    assert (y[60:80] == A).all()                     # a 0.3 s away touch owns the ball until home touches it
    assert (y[80:] == H).all()                       # to the end of the period


def test_interruption_gap_ball_out_then_restart_reception():
    ball = np.full((400, 2), 50.0, np.float32)
    ball[130:170, 1] = 61.0                          # out over the touchline from frame 130
    poss = [Po(100, 120, 1, 0, end_type="pass", after="throw_in_against"),
            Po(200, 210, 1, 1, start_type="throw_in_reception", before="throw_in_for")]
    y = _labels(poss, ball=ball)
    assert (y[100:130] == H).all()                   # still home's while the ball rolls out
    assert (y[130:190] == D).all()                   # dead from the line to the throw, one second before the reception
    assert (y[190:200] == A).all()                   # the thrower's team
    assert (y[200:] == A).all()


def test_restart_intercepted_belongs_to_the_taker_until_the_interception():
    poss = [Po(100, 120, 1, 0, after="goal_kick_against"),
            Po(200, 210, 1, 0, start_type="goal_kick_interception", before="goal_kick_against")]
    y = _labels(poss)
    assert (y[121:131] == H).all() and (y[131:190] == D).all()     # no ball-out frame: dead 1 s after the last touch
    assert (y[190:200] == A).all() and (y[200:210] == H).all()     # away took the goal kick, home cut it out


def test_foul_is_dead_at_once_and_a_taker_with_his_own_possession_restarts_it():
    poss = [Po(100, 120, 1, 0, end_type="foul_suffered", after="free_kick_for"),
            Po(180, 200, 1, 0, start_type="unknown", before="free_kick_for")]
    y = _labels(poss)
    assert y[120] == H and (y[121:180] == D).all() and (y[180:201] == H).all()


def test_untracked_frames_of_a_stoppage_are_unknown_and_start_it():
    tracked = np.ones(400, bool)
    tracked[125:160] = False                         # the broadcast cuts away
    poss = [Po(100, 120, 1, 0, after="corner_for"), Po(200, 205, 1, 0, start_type="corner_reception", before="corner_for")]
    y = _labels(poss, tracked=tracked)
    assert (y[121:125] == H).all() and (y[125:160] == X).all() and (y[160:190] == D).all() and (y[190:200] == H).all()


def test_untracked_frames_in_open_play_keep_their_owner():
    tracked = np.ones(400, bool)
    tracked[30:40] = False
    y = _labels([Po(10, 20, 1, 1), Po(60, 70, 1, 0)], tracked=tracked)
    assert (y[10:60] == A).all()


def test_periods_do_not_leak_and_the_last_stoppage_runs_to_the_whistle():
    poss = [Po(10, 20, 1, 0, after="goal_for"), Po(210, 220, 2, 1)]
    y = _labels(poss, periods=((1, 0, 99), (2, 200, 399)))
    assert (y[10:31] == H).all() and (y[31:100] == D).all()        # dead 1 s after the shot, to the end of the half
    assert (y[100:210] == X).all() and (y[210:] == A).all()


def test_read_possessions_maps_teams_and_sorts(tmp_path):
    p = tmp_path / "1_dynamic_events.csv"
    p.write_text(
        "event_type,frame_start,frame_end,period,team_id,start_type,end_type,game_interruption_before,game_interruption_after\n"
        "player_possession,50,60,1,7,pass_reception,pass,,\n"
        "passing_option,1,2,1,7,,,,\n"
        "player_possession,10,20,1,9,recovery,possession_loss,throw_in_for,\n", encoding="utf-8")
    got = SA.read_possessions(p, 7)
    assert [(g.start, g.team, g.before) for g in got] == [(10, 1, "throw_in_for"), (50, 0, "")]
