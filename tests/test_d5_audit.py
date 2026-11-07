"""Regression tests for the D5 audit findings (docs/audit/d5-bakeoff-audit.md). Synthetic only."""

from __future__ import annotations

import numpy as np

from katena.eval.score import to_native_codes
from katena.types import X


# --- F: work grid -> native frames must never put X on a graded frame -----------------------

def test_to_native_codes_keeps_block_edges_graded():
    step, n = 5, 1000
    blocks = [(1, 101, 498), (2, 601, 899)]      # edges not on the 5-frame grid
    idx = np.concatenate([np.arange(-(-a // step), b // step + 1) for _, a, b in blocks])
    rows = (np.arange(len(idx)) % 3).astype(np.int8)
    old = to_native_codes(rows, idx, step, n)
    new = to_native_codes(rows, idx, step, n, blocks)
    inside = np.zeros(n, bool)
    for _, a, b in blocks:
        inside[a:b + 1] = True
    assert (old[inside] == X).sum() == 7           # frames 101, 102, 498, 601, 602, 898, 899
    assert not (new[inside] == X).any()
    assert (new[~inside] == X).all()               # nothing written outside the blocks
    assert new[101] == new[105] and new[498] == new[495]       # nearest sample inside the block
    same = inside & (old != X)
    np.testing.assert_array_equal(new[same], old[same])        # interior frames unchanged


def test_to_native_codes_target_blocks_never_leaked():
    """The competition's graded halves (games.py) happen to sit well on the 5-frame grid."""
    step, n = 5, 199319
    blocks = [(1, 11185, 90651), (2, 110808, 190197)]
    idx = np.concatenate([np.arange(-(-a // step), b // step + 1) for _, a, b in blocks])
    out = to_native_codes(np.zeros(len(idx), np.int8), idx, step, n)
    for _, a, b in blocks:
        assert not (out[a:b + 1] == X).any()
