"""Round-trip tests for the IDSSE and EPTS loaders on tiny generated files. Offline, fast."""
import numpy as np
import pytest

import possession_metrics as pm
from external_fixtures import tiny_match, write_dfl, write_epts
from katena.external.common import (concat_halves, derive_speed, detect_attack_dir,
                                    load_match_npz, save_match_npz)
from katena.external.epts import load_epts, rows_for_frames
from katena.external.idsse import labels_from_dfl, load_idsse
from katena.match import A, D, H, X


def _half_rows(m):
    return np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])


def test_dfl_labels_from_flags():
    status = np.array([0, 1, 1, 1, 0, 1, 1], dtype=np.int8)
    poss = np.array([1, 1, 2, -1, 2, 2, 1], dtype=np.int8)
    y, diag = labels_from_dfl(status, poss)
    assert y.tolist() == [D, H, A, A, D, A, H]  # unknown owner is carried forward
    assert diag["alive_owner_unknown"] == 1


def test_idsse_roundtrip(tmp_path):
    src = tiny_match(seed=3)
    paths = write_dfl(src, tmp_path / "dfl")
    m = load_idsse(paths["meta"], paths["tracking"], paths["event"], gap=7)
    m.validate()
    # home/away from Role, not from file order (guest is listed first)
    assert m.team.tolist() == src.team.tolist()
    assert m.meta["home"] == "Home FC" and m.meta["away"] == "Away FC"
    rows = _half_rows(m)
    srows = _half_rows(src)
    assert np.allclose(m.P[rows], src.P[srows], atol=0.006)          # y flip undone
    assert np.allclose(m.ball[rows], src.ball[srows], atol=0.006)
    assert np.allclose(m.speed[rows], src.speed[srows], atol=0.005)   # km/h -> m/s
    assert np.array_equal(m.y[rows], src.y[srows])
    # 7 X rows between the halves, scorer sees exactly two runs
    (h1, a1, b1), (h2, a2, b2) = m.half_blocks
    assert a2 - b1 - 1 == 7 and (m.y[b1 + 1:a2] == X).all()
    assert len(pm.valid_runs(m.y)) == 2
    assert m.attack_dir == src.attack_dir
    assert all(d["provider_agrees"] for d in m.meta["attack_dir_diag"].values())
    assert m.meta["frame_id"][a1] == 10000 and m.meta["frame_id"][a2] == 100000
    # the provider's owner on dead frames is kept, but never leaks into y
    dead = m.y == D
    assert (m.meta["owner_raw"][dead] >= 0).all()


def test_epts_roundtrip_with_layout_switch(tmp_path):
    src = tiny_match(seed=4)
    paths, rows_written = write_epts(src, tmp_path / "epts")
    m = load_epts(paths["meta"], paths["raw"], half_gap=5, name="tiny")
    m.validate()
    assert m.y is None
    assert m.team.tolist() == src.team.tolist()
    fid = m.meta["frame_id"]
    rows = np.flatnonzero(fid >= 0)
    assert len(rows) == len(rows_written)
    assert np.allclose(m.P[rows], src.P[rows_written], atol=0.01)
    alive = src.y[rows_written] != D
    assert np.allclose(m.ball[rows][alive], src.ball[rows_written][alive], atol=0.01)
    assert np.isnan(m.ball[rows][~alive]).all()
    (h1, a1, b1), (h2, a2, b2) = m.half_blocks
    assert a2 - b1 - 1 == 5
    assert b1 - a1 + 1 == src.half_blocks[0][2] - src.half_blocks[0][1] + 1
    assert m.attack_dir == src.attack_dir
    assert all(d["provider_agrees"] for d in m.meta["attack_dir_diag"].values())
    assert m.meta["speed_source"].startswith("derived")
    assert rows_for_frames(fid, [1, int(fid[a2]), 10 ** 9]).tolist() == [0, a2, -1]


def test_concat_halves_pads_with_x_and_nan():
    parts = [dict(half=1, y=np.array([H, A], np.int8), P=np.zeros((2, 1, 2), np.float32),
                  fid=np.array([5, 6]), ok=np.array([True, True])),
             dict(half=2, y=np.array([D], np.int8), P=np.ones((1, 1, 2), np.float32),
                  fid=np.array([9]), ok=np.array([True]))]
    arr, blocks = concat_halves(parts, gap=2)
    assert blocks == [(1, 0, 1), (2, 4, 4)]
    assert arr["y"].tolist() == [H, A, X, X, D]
    assert np.isnan(arr["P"][2:4]).all()
    assert arr["fid"].tolist() == [5, 6, -1, -1, 9]
    assert arr["ok"].tolist() == [True, True, False, False, True]


def test_detect_attack_dir_on_synthetic():
    for seed in range(4):
        m = tiny_match(seed=seed, half_minutes=0.5)
        got, diag = detect_attack_dir(m.P, m.team, m.half_blocks, m.fps)
        assert got == m.attack_dir
        assert all(d["agree"] for d in diag.values())


def test_derive_speed_constant_velocity():
    t = np.arange(10, dtype=np.float32)
    P = np.stack([t * 0.2, np.zeros_like(t)], axis=-1)[:, None, :]  # 0.2 m/frame = 5 m/s
    s = derive_speed(P, fps=25)
    assert np.allclose(s, 5.0)


def test_npz_roundtrip(tmp_path):
    src = tiny_match(seed=5)
    src.meta["frame_id"] = np.arange(src.n, dtype=np.int64)
    src.meta["nested"] = {"a": 1}
    p = save_match_npz(src, tmp_path / "m.npz")
    m = load_match_npz(p)
    assert np.array_equal(m.y, src.y) and np.allclose(m.P, src.P, equal_nan=True)
    assert m.half_blocks == src.half_blocks and m.attack_dir == src.attack_dir
    assert m.pitch == tuple(src.pitch) and m.name == src.name
    assert np.array_equal(m.meta["frame_id"], src.meta["frame_id"])
    assert m.meta["nested"] == {"a": 1}


@pytest.mark.parametrize("bad", ["P", "team"])
def test_npz_roundtrip_validates(tmp_path, bad):
    src = tiny_match(seed=6)
    if bad == "P":
        src.P = src.P.astype(np.float64)
    else:
        src.team = src.team.copy()
        src.team[0] = 5
    p = tmp_path / "bad.npz"
    save_match_npz(src, p)
    with pytest.raises(ValueError):
        load_match_npz(p)
