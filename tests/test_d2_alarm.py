"""The label-free D2 alarm (katena.sim.alarm) on synthetic matches."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
from test_d2_degrader import dead_span_match

from katena.sim import alarm
from katena.sim import degrade as dg
from katena.sim import worlds as W


def test_signature_reads_no_labels():
    m = dg.corrupt(dead_span_match(seed=1, minutes=10.0, dead_speed=0.2), W.WORLDS["target_cpl100"], 2)
    a = alarm.signature(m)
    b = alarm.signature(dataclasses.replace(m, y=None))
    assert a == b
    assert all(isinstance(v, float) for v in a.values())        # aggregates only


def test_coupled_world_shows_up_as_ball_loss_when_players_stand_still():
    m = dead_span_match(seed=3, dead_speed=0.15)
    indep = alarm.signature(dg.corrupt(m, dg.TARGET, 4))
    coupled = alarm.signature(dg.corrupt(m, W.WORLDS["target_cpl100"], 4))
    assert abs(indep["ball_missing"] - coupled["ball_missing"]) < 0.03
    assert coupled["still_vs_moving_logodds"] > indep["still_vs_moving_logodds"] + 1.0
    assert coupled["miss_still"] > 0.6 > indep["miss_still"]


def test_nearest_ranks_the_world_a_match_came_from(tmp_path):
    worlds = {"target": dg.TARGET, "target_cpl100": W.WORLDS["target_cpl100"]}
    sigs = {w: [alarm.signature(dg.corrupt(dead_span_match(seed=s, minutes=12.0, dead_speed=0.15), cfg, 10 + s))
                for s in range(4)] for w, cfg in worlds.items()}
    ref = {"worlds": {w: alarm.summarise(v) for w, v in sigs.items()}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    ref = alarm.load_reference(p)
    for w, cfg in worlds.items():
        probe = alarm.signature(dg.corrupt(dead_span_match(seed=9, minutes=12.0, dead_speed=0.15), cfg, 99))
        ranked = alarm.nearest(probe, ref)
        assert ranked[0][0] == w and math.isfinite(ranked[0][1])
    far = dict(probe, ball_missing=0.9)
    assert not any(inside for _, _, inside in alarm.nearest(far, ref))


def test_committed_reference_covers_the_decision_worlds():
    if not alarm.REF_PATH.exists():
        return                                                   # written by scripts/d2_worlds.py alarm
    ref = alarm.load_reference()
    assert {"target", "target_cpl50", "target_cpl100", "target_ids", "metrica_native"} <= set(ref["worlds"])
    assert all(k in ref["worlds"]["target"] for k in alarm.BALL_KEYS)
    assert not any("J03WQQ" in k or "Sample_Game_2" in k for k in ref["matches"])     # no locked bench
    assert np.isfinite(ref["worlds"]["target_cpl100"]["still_vs_moving_logodds"][0])
