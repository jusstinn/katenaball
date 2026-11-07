"""D2 degrader options: backward compatibility, dead-span ball loss, encounter identity switches.

Backward compatibility is proved against ``_degrade_frozen_47a3f4b.py``, a verbatim copy of
``src/katena/sim/degrade.py`` at commit 47a3f4b (the last version before the D2 options). Every
existing config must give byte-identical output for the same seed, so the corpus feature caches,
which are keyed by condition name, stay valid.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
from synth import make_match

import _degrade_frozen_47a3f4b as frozen
from katena.eval import protocol as P
from katena.grid import block_mask, downsample
from katena.sim import degrade as dg
from katena.sim import worlds as W
from katena.types import A, D, H, X, Match

FROZEN = Path(__file__).with_name("_degrade_frozen_47a3f4b.py")
FROZEN_SHA256 = "986400c2492e061debeb67f6a454c0412ae2b4e007f61aff0d78d222522d7cd7"   # LF-normalised


@pytest.fixture(autouse=True)
def _frozen_reads_the_same_calibration():
    """The frozen copy lives in tests/, so point it at the package's calibration.json."""
    frozen.CAL_PATH = dg.CAL_PATH
    frozen._CAL_CACHE.clear()
    yield


def _existing_configs():
    """Every condition the corpus builds today (katena.eval.corpus.CONFIG), rebuilt here from the
    frozen module so the test does not depend on the new code to define them."""
    t = frozen.TARGET
    return [frozen.MILD, t, frozen.SEVERE,
            dataclasses.replace(t, name="target_cb", ball_missing=0.0, sigma_ball=0.0, false_ball_rate=0.0),
            frozen.with_dead_odds(t, 3.0), frozen.with_dead_odds(t, 15.0)]


def _as_new(cfg) -> dg.DegradeCfg:
    """The same config as a new-module DegradeCfg (all D2 fields left at their defaults)."""
    return dg.DegradeCfg(**dataclasses.asdict(cfg))


def _assert_same_match(a: Match, b: Match) -> None:
    for f in ("P", "team", "speed", "ball", "y"):
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and x.shape == y.shape, f
        assert x.tobytes() == y.tobytes(), f                     # bytes: NaN payloads included
    assert a.half_blocks == b.half_blocks and a.attack_dir == b.attack_dir
    assert a.meta == b.meta and a.extras == b.extras and a.fps == b.fps and a.pitch == b.pitch


def _clean_ball(m: Match) -> Match:
    return dataclasses.replace(m, ball=np.where(np.isfinite(m.ball), m.ball, 0.0).astype(np.float32))


def test_frozen_copy_is_the_47a3f4b_module():
    text = FROZEN.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(text).hexdigest() == FROZEN_SHA256


def test_existing_configs_are_byte_identical_to_the_frozen_module():
    fixtures = [downsample(make_match()), downsample(_clean_ball(make_match())),
                downsample(make_match(seed=3)), _clean_ball(make_match(seed=5))]   # last one at 25 Hz
    seeds = [0, 1, 7, P.seed_for("DFL-MAT-J03WOH", "target")]
    for m in fixtures:
        for cfg in _existing_configs():
            for seed in seeds:
                _assert_same_match(dg.corrupt(m, _as_new(cfg), seed), frozen.corrupt(m, cfg, seed))


def test_existing_named_configs_have_the_d2_options_off():
    for new, old in ((dg.MILD, frozen.MILD), (dg.TARGET, frozen.TARGET), (dg.SEVERE, frozen.SEVERE)):
        assert not dg.dead_coupling_on(new) and not dg.identity_switches_on(new)
        assert new == _as_new(old)                    # same values, D2 fields at their defaults
    assert not dg.dead_coupling_on(dg.with_dead_odds(dg.TARGET, 15.0))
    assert set(dg.CONFIGS) == {"mild", "target", "severe"}


# --- a longer synthetic match with real dead spans ----------------------------------------------

def dead_span_match(seed: int = 0, minutes: float = 24.0, fps: float = 5.0) -> Match:
    """Two halves of alternating possessions and stoppages (dead share about 0.4), 22 moving players,
    a ball at the carrier's feet while alive and still (or off the pitch) while dead."""
    rng = np.random.default_rng(seed)
    half = int(minutes / 2 * 60 * fps)
    n = 2 * half + int(60 * fps)
    blocks = [(1, 0, half - 1), (2, half + int(60 * fps), n - 1)]
    S = 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    y = np.full(n, X, dtype=np.int8)
    for h, a, b in blocks:
        t = np.arange(a, b + 1)
        base = np.c_[rng.uniform(10, 95, S), rng.uniform(5, 63, S)]
        walk = np.cumsum(rng.normal(0, 0.4, (len(t), S, 2)), axis=0)
        P[t] = np.clip(base[None] + 6 * np.tanh(walk / 6), [0, 0], [105, 68])
        f = a
        while f <= b:
            alive_len = int(rng.uniform(5, 50) * fps)
            dead_len = int(rng.uniform(8, 40) * fps)
            code = H if rng.random() < 0.5 else A
            e = min(b + 1, f + alive_len)
            y[f:e] = code
            carrier = int(rng.integers(0, 11)) + (0 if code == H else 11)
            ball[f:e] = P[f:e, carrier] + 0.5
            f, e = e, min(b + 1, e + dead_len)
            y[f:e] = D
            ball[f:e] = ball[f - 1] if f > a else (52.5, 34.0)
            f = e
    speed = np.where(np.isfinite(P[..., 0]), 2.0, np.nan).astype(np.float32)
    return Match(P=P, team=team, speed=speed, ball=ball, half_blocks=blocks, attack_dir={1: 1, 2: -1},
                 y=y, pitch=(105.0, 68.0), fps=fps, name="dead_span_synth", source="synthetic", meta={})


def _miss_rates(c: Match, m: Match):
    """(overall, dead, alive) ball-missing rates on in-play frames; NaN where a class is absent."""
    blk = block_mask(m)
    miss = ~np.isfinite(c.ball[:, 0])
    rate = lambda sel: float(miss[sel].mean()) if sel.any() else float("nan")   # noqa: E731
    return rate(blk), rate(blk & (m.y == D)), rate(blk & (m.y < D))


def test_dead_coupling_hits_both_targets():
    m = dead_span_match()
    share = float((m.y[block_mask(m)] == D).mean())
    assert 0.3 < share < 0.6
    for p_dead in (0.60, 0.79):
        cfg = dataclasses.replace(dg.TARGET, name="t", dead_miss=p_dead)
        c = dg.corrupt(m, cfg, 11)
        overall, dead, alive = _miss_rates(c, m)
        want = min(p_dead, (0.41 - 0.015 * (1 - share)) / share)
        assert abs(overall - 0.41) < 0.02, overall
        assert abs(dead - want) < 0.015, (dead, want)
        assert alive >= 0.015 - 1e-9 and alive < overall
        assert abs(c.meta["degrade"]["ball_missing_dead"] - dead) < 1e-12


def test_dead_coupling_is_stronger_than_the_legacy_worlds():
    m = dead_span_match(seed=2)
    legacy = _miss_rates(dg.corrupt(m, dg.TARGET, 3), m)
    odds15 = _miss_rates(dg.corrupt(m, dg.with_dead_odds(dg.TARGET, 15.0), 3), m)
    metrica = _miss_rates(dg.corrupt(m, W.WORLDS["target_cpl100"], 3), m)
    assert metrica[1] > odds15[1] > legacy[1] and metrica[2] < odds15[2]


def test_dead_span_loss_starts_after_a_lag_and_lasts_to_the_restart():
    m = dead_span_match(seed=4)
    c = dg.corrupt(m, dataclasses.replace(dg.MILD, name="m", ball_missing=0.30, dead_miss=0.9, dead_lag_s=1.0), 5)
    miss = ~np.isfinite(c.ball[:, 0])
    s0, e0 = dg._runs_in_blocks(block_mask(m) & (m.y == D), m)
    lost = [k for k in range(len(s0)) if miss[e0[k] - 1]]
    assert len(lost) > 0.5 * len(s0)
    # a lost span stays lost from its first missing sample to the restart (alive bursts only add)
    for k in lost:
        seg = miss[s0[k]:e0[k]]
        first = int(np.argmax(seg))
        assert seg[first:].all()
    lags = [int(np.argmax(miss[s0[k]:e0[k]])) / m.fps for k in lost]
    assert 0.3 < float(np.mean(lags)) < 2.5                 # exponential, mean 1 s, plus alive spill-over


def test_dead_coupling_never_reduces_native_ball_loss():
    m = dead_span_match(seed=6)
    ball = m.ball.copy()
    dead = m.y == D
    ball[dead & (np.arange(m.n) % 2 == 0)] = np.nan             # half of every dead span already lost
    ball[(m.y < D) & (np.arange(m.n) % 50 == 0)] = np.nan
    m2 = dataclasses.replace(m, ball=ball)
    c = dg.corrupt(m2, W.WORLDS["target_cpl50"], 9)
    native = ~np.isfinite(m2.ball[:, 0])
    assert (~np.isfinite(c.ball[:, 0]))[native].all()


def test_dead_coupling_without_dead_labels_only_tops_up():
    m = dead_span_match(seed=8)
    no_dead = dataclasses.replace(m, y=np.where(m.y == D, H, m.y).astype(np.int8))
    overall, _, _ = _miss_rates(dg.corrupt(no_dead, W.WORLDS["target_cpl100"], 1), no_dead)
    assert abs(overall - 0.41) < 0.02


# --- identity switches ---------------------------------------------------------------------------

def crossing_match(fps: float = 5.0) -> Match:
    """Four players on straight lines: home 0 and 1 cross at t = 20 s, home 2 and away 3 cross at
    t = 40 s, and nobody else is ever within 3 m. One half of 60 s."""
    n = int(60 * fps)
    t = np.arange(n) / fps
    P = np.full((n, 4, 2), np.nan, dtype=np.float32)
    P[:, 0] = np.c_[10 + 1.0 * t, np.full(n, 20.0)]
    P[:, 1] = np.c_[50 - 1.0 * t, np.full(n, 20.0)]            # meets player 0 at t = 20 s, x = 30
    P[:, 2] = np.c_[20 + 1.5 * t, np.full(n, 50.0)]
    P[:, 3] = np.c_[100 - 0.5 * t, np.full(n, 50.0)]           # meets player 2 at t = 40 s, x = 80
    team = np.array([0, 0, 0, 1], dtype=np.int8)
    y = np.full(n, H, dtype=np.int8)
    ball = (P[:, 0] + 0.3).astype(np.float32)
    return Match(P=P, team=team, speed=np.full((n, 4), 1.0, np.float32), ball=ball,
                 half_blocks=[(1, 0, n - 1)], attack_dir={1: 1}, y=y, pitch=(105.0, 68.0), fps=fps,
                 name="crossing", source="synthetic", meta={})


def _quiet(cfg: dg.DegradeCfg, **kw) -> dg.DegradeCfg:
    """No gaps, ghosts, jitter, flips or long-life cuts: only the mechanism under test acts."""
    return dataclasses.replace(cfg, name="quiet", player_avail=0.999999, ghost_mean=0.0, sigma_player=0.0,
                               false_ball_rate=0.0, team_flip=0.0, frag_life_s=1e6, **kw)


def _slot_tracks(c: Match):
    return [np.flatnonzero(np.isfinite(c.P[:, k, 0])) for k in range(c.P.shape[1])]


def test_teammate_swap_exchanges_slots_at_the_encounter():
    m = crossing_match()
    c = dg.corrupt(m, _quiet(dg.TARGET, swap_prob=1.0), 0)
    assert c.meta["degrade"]["swaps_same"] == 1 and c.meta["degrade"]["swaps_opp"] == 0
    home = np.flatnonzero(c.team == 0)
    # the per-team set of positions is unchanged at every sample (anonymous features cannot see it)
    for t in range(m.n):
        got = np.sort(c.P[t, home][np.isfinite(c.P[t, home, 0])], axis=0)
        want = np.sort(m.P[t, :3], axis=0)
        np.testing.assert_allclose(got, want, atol=1e-5)
    # the slot that followed player 0 continues on player 1 after t = 20 s, seamlessly
    t0 = int(20 * m.fps)
    k = int(np.flatnonzero(np.isclose(c.P[t0 - 5, :, 0], m.P[t0 - 5, 0, 0]) & (c.team == 0))[0])
    np.testing.assert_allclose(c.P[t0 + 5, k], m.P[t0 + 5, 1], atol=1e-5)
    assert np.isfinite(c.P[t0 - 5:t0 + 5, k, 0]).all()


def test_opponent_swap_puts_players_in_the_wrong_team_for_the_hold():
    m = crossing_match()
    c = dg.corrupt(m, _quiet(dg.TARGET, xswap_prob=1.0, xswap_hold_s=4.0), 0)
    assert c.meta["degrade"]["swaps_opp"] == 1 and c.meta["degrade"]["swaps_same"] == 0
    t0 = int(40 * m.fps)
    away = np.flatnonzero(c.team == 1)
    # just after the crossing an away slot carries home player 2, and a home slot carries away player 3
    hit_away = [k for k in away if np.isfinite(c.P[t0 + 1, k, 0]) and np.allclose(c.P[t0 + 1, k], m.P[t0 + 1, 2], atol=1e-5)]
    hit_home = [k for k in np.flatnonzero(c.team == 0)
                if np.isfinite(c.P[t0 + 1, k, 0]) and np.allclose(c.P[t0 + 1, k], m.P[t0 + 1, 3], atol=1e-5)]
    assert hit_away and hit_home
    # the confusion ends: long after the hold every player is back in a slot of its own team
    t1 = int(58 * m.fps)
    for p in range(4):
        own = np.flatnonzero(c.team == m.team[p])
        assert any(np.isfinite(c.P[t1, k, 0]) and np.allclose(c.P[t1, k], m.P[t1, p], atol=1e-5) for k in own)


def test_identity_world_keeps_presence_and_slot_count():
    m = downsample(_clean_ball(make_match()))
    base = dg.corrupt(m, dg.TARGET, 3)
    ids = dg.corrupt(m, W.WORLDS["target_ids"], 3)
    assert ids.P.shape == base.P.shape and (ids.team == base.team).all()
    assert abs(ids.meta["degrade"]["mean_objects"] - base.meta["degrade"]["mean_objects"]) < 1.5
    assert ids.meta["degrade"]["slots_used"] <= 46


def test_crowding_clock_moves_cuts_into_crowds():
    m = crossing_match()
    near = np.zeros(m.n, dtype=bool)
    near[int(15 * m.fps):int(25 * m.fps)] = True             # players 0 and 1 within 1.5 m of each other
    counts = {}
    for coef in (0.0, 2.0):
        cfg = dataclasses.replace(_quiet(dg.TARGET, frag_cong_coef=coef if coef else 1e-12), frag_life_s=5.0,
                                  occl_radius_m=6.0)
        cuts_near = cuts_far = 0
        for seed in range(20):
            c = dg.corrupt(m, cfg, seed)
            starts = np.flatnonzero(np.diff(np.isfinite(c.P[..., 0]).astype(np.int8), axis=0).clip(0).any(1)) + 1
            cuts_near += int(near[starts].sum())
            cuts_far += int((~near[starts]).sum())
        counts[coef] = cuts_near / max(1, cuts_near + cuts_far)
    assert counts[2.0] > counts[0.0] + 0.05


def test_new_worlds_are_deterministic_and_seed_dependent():
    m = dead_span_match(minutes=8.0)
    for name in ("target_cpl50", "target_cpl100", "target_ids", "target_cpl100_ids"):
        a, b, c = (dg.corrupt(m, W.WORLDS[name], s) for s in (1, 1, 2))
        _assert_same_match(a, b)
        assert a.ball.tobytes() != c.ball.tobytes() or a.P.tobytes() != c.P.tobytes()
        assert a.meta["degrade"]["cfg"] == name
