import dataclasses

import numpy as np
import pytest

from katena.match import Match
from katena.noise import DEFAULT_CFG, corrupt, target_stats
from katena.noise._util import count_within
from katena.synth import synthetic_match


@pytest.fixture(scope="module")
def clean():
    return synthetic_match(seed=3, half_minutes=1.0)


def _graded_rate(m):
    return float(np.isfinite(m.ball[m.graded_mask(), 0]).mean())


def test_output_is_a_valid_atd_like_match(clean):
    m = corrupt(clean, {}, seed=0)
    m.validate()
    n = clean.n
    assert m.P.shape == (n, 46, 2) and m.P.dtype == np.float32
    assert m.speed.shape == (n, 46) and m.speed.dtype == np.float32
    assert m.ball.shape == (n, 2) and m.ball.dtype == np.float32
    assert m.team.shape == (46,) and m.team.dtype == np.int8 and set(np.unique(m.team)) == {0, 1}
    assert np.array_equal(m.y, clean.y) and m.y is not clean.y
    assert m.half_blocks == clean.half_blocks and m.attack_dir == clean.attack_dir
    pres = np.isfinite(m.P[..., 0])
    assert np.array_equal(pres, np.isfinite(m.P[..., 1]))
    assert not np.isfinite(m.speed[~pres]).any()          # no speed without a position
    assert np.isfinite(m.speed[pres]).mean() > 0.98        # and speed almost wherever there is one
    assert np.nanmedian(m.speed) < 15
    assert "noise" in m.meta and m.meta["noise"]["seed"] == 0


def test_input_is_not_modified(clean):
    before = (clean.P.copy(), clean.ball.copy(), clean.speed.copy(), clean.y.copy())
    corrupt(clean, {}, seed=1)
    for a, b in zip(before, (clean.P, clean.ball, clean.speed, clean.y)):
        np.testing.assert_array_equal(a, b)


def test_deterministic_given_seed(clean):
    a = corrupt(clean, {}, seed=5)
    b = corrupt(clean, {}, seed=5)
    c = corrupt(clean, {}, seed=6)
    for x, y in ((a.P, b.P), (a.speed, b.speed), (a.ball, b.ball)):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(np.isfinite(a.ball[:, 0]), np.isfinite(c.ball[:, 0]))
    assert not np.allclose(a.P, c.P, equal_nan=True)


@pytest.mark.parametrize("target", [0.45, 0.593, 0.8])
def test_ball_present_rate_lands_on_target(clean, target):
    m = corrupt(clean, {"ball_present_rate": target}, seed=1)
    assert abs(_graded_rate(m) - target) < 0.02


def test_default_ball_present_rate_is_the_public_number(clean):
    assert DEFAULT_CFG["ball_present_rate"] == 0.593
    assert abs(_graded_rate(corrupt(clean, None, seed=2)) - 0.593) < 0.02


@pytest.mark.parametrize("target", [18.0, 20.5])
def test_present_objects_land_on_target(clean, target):
    m = corrupt(clean, {"players_present_mean": target}, seed=2)
    cnt = np.isfinite(m.P[m.graded_mask(), :, 0]).sum(1).mean()
    assert abs(cnt - target) < 0.3


def test_ball_loss_is_bursty(clean):
    bursty = corrupt(clean, {"ball_iso_drop_share": 0.0, "ball_gap_median_s": 1.0}, seed=3)
    iid = corrupt(clean, {"ball_iso_drop_share": 1.0}, seed=3)
    sb, si = target_stats(bursty), target_stats(iid)
    assert sb["ball_gap_q50_s"] > 0.3 and si["ball_gap_q50_s"] <= 0.08
    assert sb["ball_gaps_per_min"] < 0.2 * si["ball_gaps_per_min"]


def test_ball_loss_rises_with_congestion_near_the_ball(clean):
    cfg = {"ball_iso_drop_share": 0.0, "ball_cong_radius_m": 10.0, "ball_beta_speed": 0.0,
           "ball_beta_dead": 0.0, "ball_gap_median_s": 0.3, "ball_gap_cong_gamma": 0.0}
    c = count_within(clean.P, clean.ball, 10.0)
    hi, lo = clean.graded_mask() & (c >= 3), clean.graded_mask() & (c <= 1)
    assert hi.sum() > 200 and lo.sum() > 200
    miss = {}
    for beta in (0.0, 1.5):
        _, truth = corrupt(clean, {**cfg, "ball_beta_cong": beta}, seed=4, return_truth=True)
        obs = truth["ball_observed"]
        miss[beta] = (1 - obs[hi].mean()) - (1 - obs[lo].mean())
    assert miss[1.5] > miss[0.0] + 0.1


def test_ball_loss_rises_when_the_ball_is_in_the_air(clean):
    z = np.zeros(clean.n, dtype=np.float32)
    z[(np.arange(clean.n) // 100) % 4 == 0] = 5.0   # a quarter of the time aloft
    m0 = dataclasses.replace(clean, meta={**clean.meta, "ball_z": z})
    cfg = {"ball_iso_drop_share": 0.0, "ball_beta_cong": 0.0, "ball_beta_speed": 0.0,
           "ball_beta_dead": 0.0, "ball_gap_median_s": 0.3, "ball_gap_cong_gamma": 0.0}
    g, air = clean.graded_mask(), z > 1.8
    gap = {}
    for beta in (0.0, 2.0):
        _, truth = corrupt(m0, {**cfg, "ball_beta_air": beta}, seed=4, return_truth=True)
        obs = truth["ball_observed"]
        gap[beta] = obs[g & ~air].mean() - obs[g & air].mean()
    assert gap[2.0] > gap[0.0] + 0.15


def test_identity_meta_does_not_survive(clean):
    meta = {**clean.meta, "slot_ids": [f"p{i}" for i in range(clean.n_slots)],
            "frame_id": np.arange(clean.n), "per_player": np.zeros((clean.n, clean.n_slots))}
    m = corrupt(dataclasses.replace(clean, meta=meta), {}, seed=0)
    assert "slot_ids" not in m.meta and "per_player" not in m.meta
    assert "frame_id" in m.meta
    assert set(m.meta["noise"]["dropped_meta"]) == {"slot_ids", "per_player"}


def test_identities_are_stripped_and_fragmented(clean):
    m, truth = corrupt(clean, {}, seed=3, return_truth=True)
    sp, kind = truth["slot_player"], truth["slot_kind"]
    hosts = [np.unique(sp[:, s][sp[:, s] >= 0]).size for s in range(m.n_slots)]
    assert max(hosts) >= 3                      # slots change occupant over the match
    slots_per_player = [np.unique(np.nonzero(sp == p)[1]).size for p in range(clean.n_slots)]
    assert min(slots_per_player) >= 2           # and every player moves between slots
    st = target_stats(m)
    assert st["slot_run_q50_s"] < 30 and st["slots_used"] > 22
    # positions stay close to the truth they came from
    t, s = np.nonzero(kind == 0)
    err = np.linalg.norm(m.P[t, s] - clean.P[t, sp[t, s]], axis=-1)
    assert np.median(err) < 0.6


def test_slots_respect_team_membership_without_flips(clean):
    m, truth = corrupt(clean, {"team_flip_rate": 0.0}, seed=4, return_truth=True)
    sp = truth["slot_player"]
    t, s = np.nonzero(sp >= 0)
    assert np.array_equal(m.team[s], clean.team[sp[t, s]])
    m2, truth2 = corrupt(clean, {"team_flip_rate": 0.3}, seed=4, return_truth=True)
    sp2 = truth2["slot_player"]
    t, s = np.nonzero(sp2 >= 0)
    assert (m2.team[s] != clean.team[sp2[t, s]]).mean() > 0.1


def test_duplicates_and_ghosts(clean):
    m, truth = corrupt(clean, {"dup_rate_per_min": 40.0, "ghost_rate_per_min": 20.0,
                               "players_present_mean": 21.5}, seed=5, return_truth=True)
    assert (truth["slot_kind"] == 1).any() and (truth["slot_kind"] == 2).any()
    assert (np.isfinite(m.P[..., 0]).sum(1) > 22).any()
    assert target_stats(m)["dup_pairs_per_frame"] > 0
    none = corrupt(clean, {"dup_rate_per_min": 0.0, "ghost_rate_per_min": 0.0}, seed=5)
    assert none.meta["noise"]["summary"]["n_duplicates"] == 0
    assert none.meta["noise"]["summary"]["n_ghosts"] == 0


def test_swaps_happen_when_teammates_meet(clean):
    m = corrupt(clean, {"swap_dist_m": 8.0, "swap_prob": 1.0}, seed=6)
    assert m.meta["noise"]["summary"]["n_swaps"] > 0
    m.validate()


def test_jitter_shows_in_high_frequency_residual(clean):
    lo = target_stats(corrupt(clean, {"pos_jitter_sd_m": 0.0}, seed=7))["player_d2_mad"]
    hi = target_stats(corrupt(clean, {"pos_jitter_sd_m": 0.5}, seed=7))["player_d2_mad"]
    assert hi > lo * 1.5


def test_cfg_keys_are_checked(clean):
    with pytest.raises(KeyError):
        corrupt(clean, {"bal_present_rate": 0.5}, seed=0)
    with pytest.raises(ValueError):
        corrupt(clean, {"slot_policy": "nope"}, seed=0)
    corrupt(clean, {"_fit": {"anything": 1}, "_severity": 2.0}, seed=0).validate()


@pytest.mark.parametrize("policy", ["lowest", "lru", "random"])
def test_slot_policies_and_slot_counts(clean, policy):
    m = corrupt(clean, {"slot_policy": policy, "n_slots": 30, "home_slot_frac": 0.4}, seed=8)
    m.validate()
    assert m.n_slots == 30 and (m.team == 0).sum() == 12


def test_unlabelled_clean_match_and_extra_slots():
    base = synthetic_match(seed=9, half_minutes=0.5)
    # four extra players who are on the pitch for only part of the match, like substitutes
    extra = np.full((base.n, 4, 2), np.nan, dtype=np.float32)
    extra[: base.n // 2] = base.P[: base.n // 2, :4] + 1.0
    P = np.concatenate([base.P, extra], axis=1)
    sp = np.concatenate([base.speed, np.full((base.n, 4), np.nan, np.float32)], axis=1)
    team = np.concatenate([base.team, np.array([0, 0, 1, 1], np.int8)])
    m0 = dataclasses.replace(base, P=P, speed=sp, team=team, y=None)
    m0.validate()
    m = corrupt(m0, {}, seed=1)
    m.validate()
    assert m.y is None and m.n_slots == 46


def test_clean_ball_gaps_pass_through(clean):
    ball = clean.ball.copy()
    ball[1000:1100] = np.nan
    m0 = dataclasses.replace(clean, ball=ball)
    m = corrupt(m0, {"ball_present_rate": 0.9}, seed=2)
    assert not np.isfinite(m.ball[1000:1100]).any()
    assert abs(_graded_rate(m) - 0.9) < 0.02


def test_single_block_match_without_halves():
    base = synthetic_match(seed=10, half_minutes=0.5)
    m0 = Match(P=base.P, team=base.team, speed=base.speed, ball=base.ball, half_blocks=[],
               attack_dir={}, y=base.y)
    m = corrupt(m0, {"ball_present_rate": 0.6}, seed=0)
    m.validate()
    assert abs(np.isfinite(m.ball[:, 0]).mean() - 0.6) < 0.03
