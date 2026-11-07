"""Control-evidence helpers that derive min_control_s. Offline, fast."""
import numpy as np

from katena.external.control_evidence import best_threshold, evidence_by_bin, spell_evidence
from katena.match import A, D, H


def seq(*parts):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in parts])


def test_spell_evidence_selects_removable_flips_and_matches_team():
    # H 100 (run start: restart, ignored) | A 20 (return) | H 100 (return) | A 150 (to_dead)
    # | D 50 | A 30 (restart, ignored)
    y = seq((H, 100), (A, 20), (H, 100), (A, 150), (D, 50), (A, 30))
    rows = np.array([105, 260, 50, 360])   # A in the A 20 and A 150 flips; H in the first H
    code = np.array([1, 1, 0, 1])
    dur, has = spell_evidence(y, rows, code, fps=25, tol_s=0.0)
    assert dur.tolist() == [20 / 25, 100 / 25, 150 / 25]
    assert has.tolist() == [True, False, True]
    dur, has = spell_evidence(y, rows, np.array([0, 0, 0, 0]), fps=25, tol_s=0.0)
    assert has.tolist() == [False, False, False]   # wrong team: no evidence


def test_best_threshold_recovers_a_clean_split():
    dur = np.array([0.5, 1.0, 1.5, 2.0, 4.0, 5.0, 6.0, 8.0])
    has = dur >= 3.0
    res = best_threshold(dur, has)
    assert 2.0 < res["accuracy_best_t"] <= 4.0 and res["accuracy"] == 1.0
    bins = evidence_by_bin(dur, has, bins=(0, 3, np.inf))
    assert bins == [(0, 3, 4, 0.0), (3, np.inf, 4, 1.0)]
