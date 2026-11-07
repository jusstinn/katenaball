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


# --- F: locked benches are pinned by id, not recomputed from whatever is on disk -------------

def test_split_pins_the_locked_benches():
    from pathlib import Path

    from katena.eval import protocol as P
    assert P.locked_idsse(list(P.IDSSE_IDS)) == P.LOCKED_IDSSE
    refs = ([P.Ref("idsse", i, Path(i)) for i in P.IDSSE_IDS]
            + [P.Ref("metrica", f"Sample_Game_{k}", Path("m")) for k in (1, 2, 3)]
            + [P.Ref("skillcorner", "1874553", Path("s"))])
    dev, locked = P.split(refs)
    assert {r.key for r in locked} == {"idsse/DFL-MAT-J03WQQ", "metrica/Sample_Game_2"}
    assert len(dev) == 9
    # a corpus missing the IDSSE bench must not promote the next-smallest hash (J03WPY) to bench
    partial = [r for r in refs if r.id != P.LOCKED_IDSSE]
    dev, locked = P.split(partial)
    assert {r.key for r in locked} == {"metrica/Sample_Game_2"}
    assert "idsse/DFL-MAT-J03WPY" in {r.key for r in dev}


# --- F: the corpus code digest must not depend on the checkout's line endings ----------------

def test_code_digest_ignores_crlf(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_external as bx

    def write(eol: bytes) -> None:
        for i, f in enumerate(bx.CODE_FILES):
            p = tmp_path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(eol.join([b"# loader", b"x = %d" % i, b""]))

    monkeypatch.setattr(bx, "REPO", tmp_path)
    write(b"\n")
    lf, lf_raw = bx.code_digest("idsse"), bx.code_digest("idsse", normalise_eol=False)
    write(b"\r\n")                                   # what Git for Windows checks out
    assert bx.code_digest("idsse") == lf == lf_raw
    assert bx.code_digest("idsse", normalise_eol=False) != lf
    assert bx.code_digest("skillcorner") != lf       # still per dataset
