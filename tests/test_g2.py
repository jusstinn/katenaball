"""G2 (W4b): set-piece tag columns (``katena.featurize.tags``), imitation tags from external events
(``katena.external.setpieces``), the real-tag reader behind the pipeline's off-by-default option, and the gate code
(``katena.eval.g2``) end to end on the tiny synthetic corpus of ``tests/test_final.py``. No external or match data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synth import tagging_xml
from test_final import synth_final, tiny_tcn  # noqa: F401  (fixtures)

from katena.external import setpieces as SP
from katena.featurize import tags as TAGS
from katena.featurize.tags import SetPiece

FPS = 25.0


# --- the columns -------------------------------------------------------------------------------

def test_column_names_are_the_registered_ones():
    assert TAGS.NAMES == ["tag_since_out", "tag_until_out", "tag_since_fk", "tag_until_fk", "tag_since_ko",
                          "tag_until_ko", "tag_last_team", "tag_next_team", "tag_inside"]
    assert TAGS.CAP_S == 300.0 and TAGS.GROUP == "tags"
    assert {TAGS.KIND_GROUP[k] for k in ("throw-in", "corner kick", "goal kick")} == {"out"}
    assert TAGS.KIND_GROUP["free kick"] == "fk"
    assert TAGS.KIND_GROUP["kick off (start)"] == TAGS.KIND_GROUP["kick off (after goal)"] == "ko"


def test_columns_by_hand():
    sps = [SetPiece(0, 100, "kick off (start)", 0), SetPiece(1000, 1100, "throw-in", None),
           SetPiece(2000, 2150, "free kick", 1), SetPiece(2500, 2600, "corner kick", 0)]
    idx = np.array([0, 10, 199, 220, 400, 500, 9000])               # native frames 0, 50, 995, 1100, 2000, 2500, 45000
    A, names = TAGS.tag_columns(sps, idx, 5, FPS)
    col = {n: i for i, n in enumerate(names)}
    a0, a1 = A[0], A[1]
    assert a0.dtype == np.float32 and a0.shape == (7, 9)
    np.testing.assert_allclose(a0[:, col["tag_since_ko"]], [0, 2, 39.8, 44, 80, 100, 300])          # capped
    assert np.isnan(a0[:3, col["tag_since_out"]]).all()
    np.testing.assert_allclose(a0[3:, col["tag_since_out"]], [4, 40, 0, 300])
    np.testing.assert_allclose(a0[:3, col["tag_until_out"]], [40, 38, 0.2])
    np.testing.assert_allclose(a0[3:6, col["tag_until_out"]], [56, 20, np.nan])
    assert np.isnan(a0[:, col["tag_until_ko"]]).all()
    np.testing.assert_allclose(a0[:, col["tag_until_fk"]], [80, 78, 40.2, 36, np.nan, np.nan, np.nan])
    # the latest set piece starts at or before the sample, the next strictly after it
    np.testing.assert_allclose(a0[:, col["tag_last_team"]], [1, 1, 1, 0, -1, 1, 1])
    np.testing.assert_allclose(a1[:, col["tag_last_team"]], [-1, -1, -1, 0, 1, -1, -1])
    np.testing.assert_allclose(a0[:, col["tag_next_team"]], [0, 0, 0, -1, 1, np.nan, np.nan])
    np.testing.assert_allclose(a0[:, col["tag_inside"]], [1, 1, 0, 0, 1, 1, 0])
    team_cols = [col["tag_last_team"], col["tag_next_team"]]
    other = [i for i in range(9) if i not in team_cols]
    assert np.array_equal(a0[:, other], a1[:, other], equal_nan=True)           # only the team columns depend on focal


def test_no_tagging_is_all_nan_and_empty_tagging_is_not():
    A, _ = TAGS.tag_columns(None, np.arange(10), 5, FPS)
    assert np.isnan(A[0]).all() and np.isnan(A[1]).all()
    A, names = TAGS.tag_columns([], np.arange(10), 5, FPS)
    assert (A[0][:, names.index("tag_inside")] == 0).all()
    assert np.isnan(np.delete(A[0], names.index("tag_inside"), 1)).all()


def test_real_tags_read_from_sportscode_xml(tmp_path):
    from katena.target.tagging import read_tags
    p = tmp_path / "tags.xml"
    p.write_text(tagging_xml([
        (10.0, 14.0, "SET PIECES", {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}),
        (20.0, 25.0, "Set Pieces", {"Team": "Barcelona", "Half": "1st Half", "Type": " CORNER KICK "}),
        (30.0, 35.0, "SET PIECES", {"Team": "Atletic Club", "Half": "2nd Half", "Type": "Kick Off (After Goal)"}),
        (5.0, 9.0, "SET PIECES", {"Team": "Barcelona", "Half": "1st Half", "Type": "Penalty"}),      # unknown Type
        (40.0, 50.0, "BUILD UP", {"Team": "Barcelona", "Half": "2nd Half", "Type": "Free Kick"}),   # not a set piece
    ]), encoding="utf-8", newline="\n")
    sps = TAGS.from_tagging(read_tags(p, fps=FPS))
    assert sps == [SetPiece(250, 350, "throw-in", None), SetPiece(500, 625, "corner kick", 0),
                   SetPiece(750, 875, "kick off (after goal)", 1)]
    assert TAGS.counts(sps)["corner kick"] == 1 and sum(TAGS.counts(sps).values()) == 3


def test_pipeline_tag_option_reads_the_bundle_file_and_f_refuses_it(tmp_path):
    from synth import write_tagging

    from katena import pipeline as PL
    write_tagging(tmp_path / "b", [(10.0, 14.0, "SET PIECES", {"Team": "N/A", "Type": "Throw-in"})])
    tags, why = PL.read_target_tags(tmp_path / "b", FPS)
    assert why is None and TAGS.from_tagging(tags) == [SetPiece(250, 350, "throw-in", None)]
    import inspect
    assert inspect.signature(PL.stage_predict).parameters["tag_columns"].default is False
    assert inspect.signature(PL.label_match).parameters["set_pieces"].default is None


# --- imitation -----------------------------------------------------------------------------------

def _events(n_per_kind: dict, seed: int = 0) -> list:
    """Synthetic events at least 40 s apart, so each imitated interval contains exactly one of them."""
    rng = np.random.default_rng(seed)
    kinds = [k for k, n in n_per_kind.items() for _ in range(n)]
    rows = rng.choice(np.arange(1, 400), size=len(kinds), replace=False) * 1000
    return sorted((int(r), k, int(rng.integers(0, 2))) for r, k in zip(rows, kinds))


def test_imitation_keeps_the_real_share_per_type_and_simulates_timing():
    ext = {"throw-in": 40, "corner kick": 9, "goal kick": 15, "free kick": 25, "kick off (after goal)": 6,
           "kick off (start)": 2}
    ev = _events(ext)
    sps, st = SP.imitate(ev, "MATCH-A", FPS)
    want = {k: min(SP.REAL_COUNTS[k], n) for k, n in ext.items()}
    assert st["kept"] == want and st["events"] == ext and TAGS.counts(sps) == want
    assert want == {"throw-in": 26, "corner kick": 9, "goal kick": 10, "free kick": 5, "kick off (after goal)": 4,
                    "kick off (start)": 2}
    rows = {(r, k): t for r, k, t in ev}
    for s in sps:
        hits = [(r, k) for (r, k) in rows if k == s.kind and s.start <= r < s.end]
        assert hits, s
        r = hits[0][0]
        assert 0 <= r - s.start <= SP.PRE_S[1] * FPS and 1 <= s.end - r <= SP.POST_S[1] * FPS
        assert s.team is None if s.kind == "throw-in" else s.team == rows[hits[0]]
    again, _ = SP.imitate(ev, "MATCH-A", FPS)
    other, _ = SP.imitate(ev, "MATCH-B", FPS)
    assert again == sps and other != sps                          # seeded by the match id, and only by it
    assert SP.imitate(None, "MATCH-A", FPS) == (None, {})


def test_dfl_events_parse_set_pieces_and_skip_the_rest(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest>
 <Event EventId="1" EventTime="2023-05-27T15:30:12.230+02:00"><KickOff GameSection="firstHalf" TeamLeft="H" TeamRight="G"><Play Team="H" Player="p"/></KickOff></Event>
 <Event EventId="2" EventTime="2023-05-27T15:30:33.800+02:00"><ThrowIn Team="G" Side="right"><Play Team="G"/></ThrowIn></Event>
 <Event EventId="3" EventTime="2023-05-27T15:31:00.000+02:00"><Delete/></Event>
 <Event EventId="4" EventTime="2023-05-27T15:32:08.070+02:00"><GoalKick Team="H"><Play Team="H"/></GoalKick></Event>
 <Event EventId="5" EventTime="2023-05-27T15:33:00.000+02:00"><FreeKick Team="G" ExecutionMode="direct"><ShotAtGoal Team="G"/></FreeKick></Event>
 <Event EventId="6" EventTime="2023-05-27T15:34:00.000+02:00"><CornerKick Team="H"><Play Team="H"/></CornerKick></Event>
 <Event EventId="7" EventTime="2023-05-27T15:35:00.000+02:00"><Penalty Team="G"><ShotAtGoal Team="G"/></Penalty></Event>
 <Event EventId="8" EventTime="2023-05-27T15:36:00.000+02:00"><KickOff TeamLeft="H" TeamRight="G"><Play Team="G"/></KickOff></Event>
 <Event EventId="9" EventTime="2023-05-27T15:37:00.000+02:00"><Play Team="H"/></Event>
</PutDataRequest>
"""
    p = tmp_path / "events.xml"
    p.write_text(xml, encoding="utf-8")
    ev = SP.idsse_events(p, "H", "2023-05-27T13:30:12.600000+00:00", FPS)
    # t0 is 13:30:12.600 UTC; the events are at +02:00. The first kickoff falls 0.37 s before the grid starts.
    assert ev == [(-9, "kick off (start)", 0), (530, "throw-in", 1), (2887, "goal kick", 0), (4185, "free kick", 1),
                  (5685, "corner kick", 0), (8685, "kick off (after goal)", 1)]


def test_metrica_events_map_subtypes():
    from katena.external.metrica import Event
    evs = [Event(1, 1, 1, 0, "SET PIECE", "KICK OFF", 0), Event(1, 101, 120, 1, "SET PIECE", "THROW IN", 1),
           Event(1, 201, 230, 0, "PASS", "GOAL KICK", 2), Event(1, 301, 330, 1, "BALL LOST", "GOAL KICK-INTERCEPTION", 3),
           Event(1, 401, 430, 0, "SET PIECE", "FREE KICK INDIRECT", 4), Event(1, 501, 530, 1, "SET PIECE", "CORNER KICK", 5),
           Event(1, 601, 601, 1, "SET PIECE", "KICK OFF", 6), Event(2, 1001, 1001, 1, "SET PIECE", "KICK OFF", 7),
           Event(2, 1101, 1130, 0, "PASS", "", 8), Event(2, 1201, 1230, 0, "SET PIECE", "THROW IN RETAKEN", 9)]
    assert SP.metrica_events(evs) == [
        (0, "kick off (start)", 0), (100, "throw-in", 1), (200, "goal kick", 0), (300, "goal kick", 1),
        (400, "free kick", 0), (500, "corner kick", 1), (600, "kick off (after goal)", 1), (1000, "kick off (start)", 1),
        (1200, "throw-in", 0)]


def test_skillcorner_has_no_events():
    from katena.eval import protocol as P
    assert SP.events_for(P.Ref("skillcorner", "1", Path("1.npz")), Path(".")) is None


# --- the gate code on a synthetic corpus ------------------------------------------------------------

def test_g2_gate_end_to_end(synth_final, tiny_tcn, tmp_path, monkeypatch):  # noqa: F811
    """F's stages build a synthetic F cache; G2 copies from it, imitates tags (one match without event data),
    reproduces F's LightGBM member byte for byte without its tag columns, trains its own on 222 columns, and re-decodes
    F from the stored members to exactly F's stored scores."""
    from katena.eval import final as F
    from katena.eval import g2 as G
    from katena.eval import lomo as Lm
    from katena.eval import protocol as P
    refs = synth_final
    T = P.T_MAIN
    for r in refs:
        F.build_touch(r)
    F.build_pools(refs)
    F.stage_gbdt(F.FinalPools(refs), refs)
    F.stage_tcn(F.FinalPools(refs, touch=False), refs, refs, threads=2, steps=2)
    F.stage_score(F.results(), refs, refs)
    f_root = P.WORK
    monkeypatch.setattr(P, "WORK", tmp_path / "work-g2")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "work-g2" / "oof")
    with pytest.raises(RuntimeError):
        G.seed_cache(refs, P.WORK)
    info = G.seed_cache(refs, f_root)
    assert info["copied"] == info["planned"] == 3 * (3 + 2 * 2) and G.seed_cache(refs, f_root)["copied"] == 0

    def fake_events(ref, root):
        if ref.id == "SYN7":
            return None                                          # stands in for a match without events
        return [(100, "kick off (start)", 0), (400, "throw-in", 1), (700, "free kick", 0), (1100, "kick off (start)", 1),
                (1400, "corner kick", 1), (1700, "goal kick", 0)]
    monkeypatch.setattr(SP, "events_for", fake_events)
    sts = [G.build_tags(r) for r in refs]
    assert [s["has_tags"] for s in sts] == [True, True, False]
    assert sts[2]["nan_share_pool"] == 1.0 and sts[0]["set_pieces"] > 2 and sts[0]["nan_share_pool"] < 1.0
    pools = G.G2Pools(refs)
    assert pools.X.shape[1] == pools.n_f + 9 and list(pools.names[-9:]) == TAGS.NAMES
    assert (pools.groups == TAGS.GROUP).sum() == 9
    chk = G.check_fg(pools, refs[0], f_root)
    assert chk["model_equal"] and chk["probs_equal_target"] and chk["probs_equal_severe"], chk
    G.stage_gbdt(pools, refs)
    oof = Lm.load_oof(G.VAR, T, refs[0])
    assert {"pa_target", "ph_target", "pa_severe", "ph_severe", "model_sha"} == set(oof)
    X0, _ = G.features(refs[0], "target")
    assert X0.shape[1] == pools.X.shape[1] and len(oof["pa_target"]) == len(X0)
    res = G.results()
    out = G.stage_score(res, refs, refs, f_root, F.load_store(f_root / F.RES_NAME))
    assert out["f_check_equal"] == out["n"] == 6
    get = F.getter(res, F.load_store(f_root / F.RES_NAME))
    g = F.compare(get, refs, "G2", "F", "target")
    assert g["n"] == 3 and {"adopted", "worse", "wins"} <= set(g)
    same = F.compare(get, refs, "F_check", "F", "severe")
    assert same["mean_diff"] == 0 and same["wins"] == 0
    assert all(get("G2G", c, r) is not None for c in G.CONDS for r in refs)
