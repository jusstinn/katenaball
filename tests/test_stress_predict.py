"""scripts/stress_predict.py on a small synthetic match, without a model: the damage it does is the damage it
says it does, every case still makes a valid Match, and the label check catches a wrong label."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from synth import make_match

from katena.models import ensemble as E

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def SP():
    """The script as a module. Importing it pins the CPU environment and the cache folder, as every script does:
    both are put back afterwards."""
    keys = list(E.CPU_ENV) + ["KATENA_WORK"]
    before = {k: os.environ.get(k) for k in keys}
    spec = importlib.util.spec_from_file_location("stress_predict", REPO / "scripts" / "stress_predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_bursts_cover_about_the_asked_share(SP):
    rng = np.random.default_rng(0)
    out = SP.bursts(200_000, 0.41, 50, rng)
    assert out.dtype == bool and abs(out.mean() - 0.41) < 0.03
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], out.astype(np.int8), [0]]))))[::2]
    assert 35 < runs.mean() < 65                                   # bursts, not single frames
    assert not SP.bursts(100, 0.0, 5, rng).any() and SP.bursts(100, 1.0, 5, rng).all()


def test_anonymous_slots_only_move_samples(SP):
    m = make_match(seed=1)
    out = SP.finish(SP.anonymous_slots(SP.clone(m), np.random.default_rng(2), mean_s=2.0))
    assert out.n_slots == 46 and (out.team == 0).sum() == 23 and (out.team == 1).sum() == 23
    for t in (0, 1):
        src, dst = m.P[:, m.team == t], out.P[:, out.team == t]
        seen_src, seen_dst = np.isfinite(src[..., 0]).sum(1), np.isfinite(dst[..., 0]).sum(1)
        assert (seen_dst <= seen_src).all() and seen_dst.sum() > 0.5 * seen_src.sum()
        for i in (150, 400, 1300, 1700):                           # every sample comes from a player of that team
            got = dst[i][np.isfinite(dst[i, :, 0])]
            assert all((np.abs(src[i] - g).sum(1) == 0).any() for g in got)
    cuts = (np.isfinite(out.P[1:, :, 0]) != np.isfinite(out.P[:-1, :, 0])).sum()
    assert cuts > 2 * 22                                           # tracks are fragments now
    assert np.array_equal(out.ball, m.ball, equal_nan=True)


def test_every_case_gives_a_valid_match(SP):
    base = SP.finish(SP.target_like(SP.clone(make_match(seed=3)), np.random.default_rng(4)))
    assert 0.25 < np.isnan(base.ball[SP.inside(base), 0]).mean() < 0.75
    assert len(SP.CASES) >= 25
    for i, (name, fn) in enumerate(sorted(SP.CASES.items())):
        m = SP.finish(fn(SP.clone(base), np.random.default_rng(10 + i)))        # finish() validates
        assert m.n == base.n and m.n_slots == 46 and m.y is None, name
        changed = not (np.array_equal(m.P, base.P, equal_nan=True) and np.array_equal(m.ball, base.ball, equal_nan=True)
                       and np.array_equal(m.speed, base.speed, equal_nan=True))
        assert changed, f"{name} changed nothing"
    assert np.array_equal(base.P, SP.finish(SP.clone(base)).P, equal_nan=True)   # the cases worked on copies


def test_named_cases_do_what_their_names_say(SP):
    base = SP.finish(SP.target_like(SP.clone(make_match(seed=5)), np.random.default_rng(6)))
    rng = np.random.default_rng(7)
    ins = SP.inside(base)
    assert np.isnan(SP.c_ball_absent_everywhere(SP.clone(base), rng).ball).all()
    m = SP.c_ball_absent_first_half(SP.clone(base), rng)
    (_, a1, b1), (_, a2, b2) = base.half_blocks
    assert np.isnan(m.ball[a1:b1 + 1]).all() and np.isfinite(m.ball[a2:b2 + 1, 0]).any()
    m = SP.c_home_team_absent_everywhere(SP.clone(base), rng)
    assert np.isnan(m.P[:, m.team == 0]).all() and np.isfinite(m.P[:, m.team == 1, 0]).any()
    assert SP.finish(m).attack_dir == {1: 0, 2: 0}                  # no direction can be read: the pipeline falls back
    m = SP.c_infinite_values(SP.clone(base), rng)
    assert np.isinf(m.ball).any() and np.isinf(m.P).any()
    m = SP.c_tracked_outside_the_halves(SP.clone(base), rng)
    assert np.isfinite(m.P[~ins][..., 0]).any() and not np.isfinite(base.P[~ins][..., 0]).any()
    flipped = SP.finish(SP.c_teams_do_not_change_ends(SP.clone(base), rng))
    assert flipped.attack_dir[2] == -base.attack_dir[2] and flipped.attack_dir[1] == base.attack_dir[1]


def test_check_codes(SP):
    m = make_match(seed=8)
    ins = SP.inside(m)
    good = np.where(ins, 0, 3).astype(np.int8)
    assert SP.check_codes(m, good) == []
    bad = good.copy()
    bad[np.flatnonzero(ins)[5]] = 3
    assert SP.check_codes(m, bad) == ["a graded frame is not H, A or D"]
    bad = good.copy()
    bad[0] = 2
    assert SP.check_codes(m, bad) == ["a frame outside the halves is not X"]
    assert "expected int8" in SP.check_codes(m, good.astype(np.int64))[0]
    assert "expected int8" in SP.check_codes(m, good[:-1])[0]
