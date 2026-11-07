"""Corpus statistics and cache handling. The real-corpus checks skip when it is not built."""
import numpy as np
import pytest

from katena.external.common import save_match_npz
from katena.external.corpus import (CORPUS_NAMES, EXTERNAL_DATA, cache_path, load_corpus,
                                    match_report, pooled_markdown, report_markdown, spell_stats)
from katena.match import D, H, X
from katena.synth import synthetic_match


def test_spell_stats_on_known_sequence():
    y = np.concatenate([np.full(50, H), np.full(25, 1), np.full(25, D), np.full(10, X),
                        np.full(100, H)]).astype(np.int8)
    s = spell_stats(y, fps=25)
    assert s["n_spells"] == 3
    assert s["mean_s"] == pytest.approx((2 + 1 + 4) / 3)
    assert s["median_s"] == pytest.approx(2.0)
    assert s["dead_share"] == pytest.approx(25 / 200)
    assert s["n_dead_stretches"] == 1


def test_report_and_load_corpus_on_synthetic(tmp_path):
    m = synthetic_match(seed=0, half_minutes=1.0, name="synthetic-0")
    save_match_npz(m, tmp_path / "cache" / "synthetic-0.npz")
    ms = load_corpus(tmp_path, names=["synthetic-0", "missing"], harmonise_cfg={})
    assert len(ms) == 1
    assert np.array_equal(ms[0].meta["y_raw"], m.y)
    row = match_report(ms[0], {"min_control_s": 2.0})
    assert row["harmonised"]["n_spells"] <= row["raw"]["n_spells"]
    md = report_markdown([row])
    assert md.count("\n") == 4 and "synthetic-0" in md
    assert pooled_markdown([row]).startswith("| source")


CACHED = [n for n in CORPUS_NAMES if cache_path(n, EXTERNAL_DATA).exists()]


@pytest.mark.skipif(not CACHED, reason="external corpus not built on this machine")
@pytest.mark.parametrize("name", CACHED)
def test_cached_corpus_match_is_sane(name):
    (m,) = load_corpus(EXTERNAL_DATA, names=[name])
    m.validate()
    assert len(m.half_blocks) == 2 and m.n_slots >= 22
    assert m.meta["source"] in ("idsse", "metrica-open")
    for d in m.meta["attack_dir_diag"].values():
        assert d.get("provider_agrees", True)
    g = m.graded_mask()
    s = spell_stats(m.y)
    assert 0.2 < s["dead_share"] < 0.6
    assert 150 < s["n_spells"] < 800
    assert (m.y[~g] == X).all() and (m.y[g] != X).all()
    if m.meta["source"] == "metrica-open":
        # Metrica's sample removes the ball exactly while it is dead: our event-derived D agrees
        dead = m.y[g] == D
        nan = np.isnan(m.ball[g, 0])
        assert (dead == nan).mean() > 0.97
