"""Target-match loader: the competition tactical ATD -> `Match`, row i = video frame i.

Thin wrapper over the generic EPTS loader (`katena.external.epts.load_epts`). Provisional until
J1 lands; J1 owns the competition loader and this should be reconciled with it.

FIREWALL: this reads the gated match data. Only a human runs code that calls it, and that code
must log aggregates only. Errors raised here state shapes and counts, never values.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .external.epts import load_epts
from .match import Match

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import games  # noqa: E402  (repo-root module from the upstream starter)

N_VIDEO_FRAMES = 199_319


def load_target_match(data_dir=None, source: str = "tactical", game: str = games.GAME,
                      pitch: tuple = (105.3, 68.0)) -> Match:
    """Load the competition match so that row i is submission frame i.

    ATD frame_id runs 1..N and video_frame = frame_id - 1 (docs/upstream/DATA.md, "Frame
    alignment"), so the EPTS source row r is video frame r exactly when the first source frame is
    1. That is checked, as is the row count, before anything is returned.
    """
    if source != "tactical":
        raise ValueError("only the tactical recording is aligned to submission rows")
    paths = games.asset_paths(game=game, data_dir=data_dir, source=source)
    for key in ("atd_meta", "atd_raw"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"{key} not found; unpack the bundle into data/ (see data/README.md)")
    blocks = [tuple(int(v) for v in b) for b in games.half_video_blocks(game)]
    m = load_epts(paths["atd_meta"], paths["atd_raw"], half_blocks=blocks, half_gap=0,
                  pitch=pitch, name=game)
    f0 = int(m.meta.get("first_source_frame", -1))
    if f0 != 1:
        raise ValueError(f"expected the first ATD frame_id to be 1, got a different start (offset {f0 - 1})")
    if m.n != N_VIDEO_FRAMES:
        raise ValueError(f"expected {N_VIDEO_FRAMES} rows, got {m.n}")
    if not np.array_equal(m.meta["frame_id"], np.arange(1, m.n + 1)):
        raise ValueError("ATD frame ids are not contiguous 1..N")
    m.meta["home_is"] = "first team in the ATD metadata (Barcelona per DATA.md)"
    return m
