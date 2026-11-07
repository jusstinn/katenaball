import dataclasses
import json

import numpy as np
import pytest

from katena.noise import DEFAULT_CFG, corrupt, crop_match, fit_noise_cfg, severity_grid, target_stats
from katena.noise.fit import stats_loss
from katena.synth import synthetic_match

TRUTH = dict(ball_present_rate=0.7, players_present_mean=19.5, frag_median_s=4.0, pos_jitter_sd_m=0.15,
             ball_gap_median_s=1.5, ball_iso_drop_share=0.25)


@pytest.fixture(scope="module")
def clean():
    return synthetic_match(seed=8, half_minutes=1.5)


@pytest.fixture(scope="module")
def target(clean):
    return corrupt(clean, TRUTH, seed=21)


@pytest.fixture(scope="module")
def fitted(clean, target):
    return fit_noise_cfg(target, clean, max_evals=30, n_noise_seeds=2)


def test_fit_matches_stated_moments_and_improves(target, fitted, clean):
    tgt = target_stats(target)
    assert fitted["ball_present_rate"] == pytest.approx(tgt["ball_present_rate"])
    assert fitted["players_present_mean"] == pytest.approx(tgt["slots_present_mean"])
    assert fitted["n_slots"] == 46
    fit = fitted["_fit"]
    assert fit["loss_final"] < fit["loss_init"] and fit["n_evals"] <= 30
    st = target_stats(corrupt(clean, fitted, seed=99))
    assert abs(st["ball_present_rate"] - tgt["ball_present_rate"]) < 0.03
    assert abs(st["slots_present_mean"] - tgt["slots_present_mean"]) < 0.4
    # moved towards the truth on the parameters it could see
    assert abs(np.log(fitted["frag_median_s"] / TRUTH["frag_median_s"])) < \
        abs(np.log(DEFAULT_CFG["frag_median_s"] / TRUTH["frag_median_s"]))
    assert abs(np.log(fitted["pos_jitter_sd_m"] / TRUTH["pos_jitter_sd_m"])) < \
        abs(np.log(DEFAULT_CFG["pos_jitter_sd_m"] / TRUTH["pos_jitter_sd_m"]))


def test_fitted_cfg_is_json_and_reusable(fitted, clean):
    text = json.dumps(fitted)
    cfg = json.loads(text)
    corrupt(clean, cfg, seed=0).validate()


def test_fit_never_reads_labels(target, clean):
    a = fit_noise_cfg(target, clean, max_evals=6, n_noise_seeds=0)
    b = fit_noise_cfg(dataclasses.replace(target, y=None), clean, max_evals=6, n_noise_seeds=0)
    a.pop("_fit"), b.pop("_fit")
    assert a == b


def test_stats_loss_zero_on_itself(target):
    s = target_stats(target)
    assert stats_loss(s, s) == 0.0
    s2 = dict(s, ball_burst_q50_s=s["ball_burst_q50_s"] * 2)
    assert stats_loss(s2, s) > 0


def test_crop_keeps_halves_apart(clean):
    c = crop_match(clean, 1000)
    c.validate()
    assert len(c.half_blocks) == 2
    assert sum(l - f + 1 for _, f, l in c.half_blocks) == 1000
    gap = slice(c.half_blocks[0][2] + 1, c.half_blocks[1][1])
    assert not np.isfinite(c.P[gap]).any()


def test_severity_grid(clean):
    grid = severity_grid(DEFAULT_CFG, 5)
    assert len(grid) == 5
    sev = [g["_severity"] for g in grid]
    assert sev == sorted(sev) and sev[2] == pytest.approx(1.0)
    mid = grid[2]
    for k in ("ball_present_rate", "players_present_mean", "frag_median_s", "pos_jitter_sd_m"):
        assert mid[k] == pytest.approx(DEFAULT_CFG[k])
    bp = [g["ball_present_rate"] for g in grid]
    pp = [g["players_present_mean"] for g in grid]
    assert bp == sorted(bp, reverse=True) and pp == sorted(pp, reverse=True)
    for g in (grid[0], grid[-1]):
        m = corrupt(clean, g, seed=1)
        m.validate()
        assert abs(np.isfinite(m.ball[m.graded_mask(), 0]).mean() - g["ball_present_rate"]) < 0.03
    assert len(severity_grid(DEFAULT_CFG, 1)) == 1
