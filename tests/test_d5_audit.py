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


# --- F: the schema probe must not print a data row when a CSV has no header ------------------

def test_probe_schema_never_prints_a_data_row(tmp_path, capsys):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_schema

    with_header = tmp_path / "h.csv"
    with_header.write_text("frame,h00,h01,valid\n0,1822.663,-511.2005,1\n", encoding="utf-8")
    probe_schema.probe_csv_header(str(with_header))
    assert "columns: frame,h00,h01,valid" in capsys.readouterr().out
    no_header = tmp_path / "n.csv"
    no_header.write_text("11185,1822.663,-511.2005,1\n", encoding="utf-8")
    probe_schema.probe_csv_header(str(no_header))
    out = capsys.readouterr().out
    assert "1822.663" not in out and "11185" not in out and "4 fields" in out


# --- F: the ATD reader's layout assumptions, checked from the EPTS metadata itself ----------

def _epts_meta(order, n_players=6, n_frames=100) -> str:
    """FIFA/EPTS metadata whose raw groups follow ``order`` (indices into the <Player> list)."""
    half = n_players // 2
    players = "".join(f'<Player id="P{i}" teamId="{"T1" if i < half else "T2"}"><Name>Track {i + 1}</Name>'
                      f'</Player>' for i in range(n_players))
    chans = "".join(f'<PlayerChannel id="p{i}_{c}" playerId="P{i}" channelId="{c}"/>'
                    for i in range(n_players) for c in ("x", "y", "speed"))
    groups = "".join('<SplitRegister separator=",">' + "".join(
        f'<PlayerChannelRef playerChannelId="p{i}_{c}"/>' for c in ("x", "y", "speed")) + "</SplitRegister>"
        for i in order)
    return (f'<?xml version="1.0"?><main><Metadata><Sessions><Session><MatchParameters>'
            f'<Score idLocalTeam="T1" idVisitingTeam="T2"/></MatchParameters></Session></Sessions>'
            f'<Teams><Team id="T1"/><Team id="T2"/></Teams><Players>{players}</Players>'
            f'<PlayerChannels>{chans}</PlayerChannels></Metadata><DataFormatSpecifications>'
            f'<DataFormatSpecification startFrame="1" endFrame="{n_frames}" separator=":">'
            f'<StringRegister name="frameCount"/><SplitRegister separator=";">{groups}</SplitRegister>'
            f'<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
            f'<BallChannelRef channelId="z"/></SplitRegister></DataFormatSpecification>'
            f'</DataFormatSpecifications></main>')


def test_layout_report_detects_group_order_the_reader_cannot_see(tmp_path):
    from katena.target.atd import read_slot_teams
    from katena.target.layout import assumptions_hold, layout_report

    aligned, shuffled = tmp_path / "a.xml", tmp_path / "s.xml"
    aligned.write_text(_epts_meta([0, 1, 2, 3, 4, 5]), encoding="utf-8")
    shuffled.write_text(_epts_meta([3, 0, 4, 1, 5, 2]), encoding="utf-8")
    ra, rs = layout_report(aligned, n_frames=100), layout_report(shuffled, n_frames=100)
    assert assumptions_hold(ra) and ra["specs_cover_1_to_n"] and ra["first_team_is_local_team"]
    assert ra["channels_per_group"] == {"x,y,speed": 6} and ra["ball_channels"] == {"x,y,z": 1}
    assert not assumptions_hold(rs) and rs["group_order_matches_player_elements"] is False
    # the reader takes teams from <Player> order in both files: right for one, wrong for the other
    truth_shuffled = [1, 0, 1, 0, 1, 0]                  # teams of raw groups 3, 0, 4, 1, 5, 2
    assert read_slot_teams(aligned).tolist() == [0, 0, 0, 1, 1, 1]
    assert read_slot_teams(shuffled).tolist() != truth_shuffled
    for rep in (ra, rs):                                 # nothing but structure in the report
        assert not any(s in str(rep) for s in ("P0", "T1", "Track"))


def test_layout_report_on_metadata_without_a_format_spec(tmp_path):
    from katena.target.layout import assumptions_hold, layout_report

    p = tmp_path / "m.xml"
    p.write_text('<Metadata><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>'
                 '<Player id="P0" teamId="T1"/><Player id="P1" teamId="T2"/></Players></Metadata>',
                 encoding="utf-8")
    rep = layout_report(p, n_frames=10)
    assert rep["n_format_specs"] == 0 and not assumptions_hold(rep)


# --- ported from david/foundation: the domain-gap alarm (aggregates only, fits nothing) ------

def _gap_fixture():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_domain_gap as cdg
    from katena.sim import degrade as dg
    clean = [cdg.synthetic_clean(s, half_s=240.0) for s in (1, 2, 3)]
    return cdg, dg, clean


def test_domain_gap_separates_worlds_and_not_seeds():
    import dataclasses

    from katena.diag import domain_gap as gapmod
    cdg, dg, clean = _gap_fixture()
    real = dg.corrupt(clean[0], dg.TARGET, seed=5)
    same = [dg.corrupt(clean[0], dg.TARGET, seed=6)]               # same match and world, other seed
    clean_side = [clean[0]]                                         # same match, no corruption
    r_same = gapmod.domain_gap(real, same)
    r_far = gapmod.domain_gap(real, clean_side)
    assert 0.25 < r_same["auc"] < 0.75
    assert r_far["auc"] > 0.95
    assert tuple(r_far) == gapmod.RESULT_KEYS
    assert all(name in gapmod.FEATURE_NAMES for name, _ in r_far["top_features"])
    # deterministic, and blind to labels
    assert gapmod.domain_gap(real, clean_side) == r_far
    assert gapmod.domain_gap(dataclasses.replace(real, y=None), clean_side) == r_far


def test_domain_gap_module_fits_nothing_to_its_input():
    from katena.diag import domain_gap as gapmod
    public = [n for n in dir(gapmod) if not n.startswith("_") and callable(getattr(gapmod, n))]
    assert set(public) <= {"domain_gap", "window_features", "Match", "annotations"}
    cdg, dg, clean = _gap_fixture()
    with __import__("pytest").raises(ValueError):                    # grids must match
        gapmod.domain_gap(clean[0], [dataclasses_replace_fps(clean[1], 25.0)])


def dataclasses_replace_fps(m, fps):
    import dataclasses
    return dataclasses.replace(m, fps=fps)


# --- cross-check ported from david/foundation's decoder tests: Viterbi is the exact argmax ---

def test_viterbi_matches_brute_force():
    import itertools

    from katena.decode import hmm
    rng = np.random.default_rng(0)

    def path_score(path, le, logA, log_init):
        return log_init[path[0]] + le[0, path[0]] + sum(
            logA[path[t - 1], path[t]] + le[t, path[t]] for t in range(1, len(path)))

    for _ in range(25):
        T = int(rng.integers(1, 8))
        le = rng.normal(0, 2, (T, 3))
        logA = np.log(rng.dirichlet(np.ones(3), size=3))
        log_init = np.log(rng.dirichlet(np.ones(3)))
        best = max(path_score(p, le, logA, log_init) for p in itertools.product(range(3), repeat=T))
        out = hmm._viterbi(np.ascontiguousarray(le), logA, log_init)
        assert out.dtype == np.int8 and len(out) == T
        assert abs(path_score(tuple(out), le, logA, log_init) - best) < 1e-9
