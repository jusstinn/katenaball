"""Write a Match as a competition-style ATD bundle: the inverse of ``katena.target.atd.load_target``.

    <root>/tactical/BAR-ATH_tactical_atd.xml   FIFA/EPTS metadata: two Teams, home first; one Player per slot,
                                               in slot order, with its team; the DataFormatSpecification chain
    <root>/tactical/BAR-ATH_tactical_atd.txt   one line per frame, frame_id 1..n:
                                               id:x,y,speed;...;x,y,speed:ball_x,ball_y,ball_z

Positions are fractions of the pitch (origin at a corner, as the canonical matches and the ATD both use) and NaN
where missing. The file names are the competition's, so ``load_target`` finds them through ``games.asset_paths``
unchanged; ``timeline_of`` gives the frame count, half blocks, pitch and frame rate that ``load_target`` would
otherwise take from ``games.py``.

With the default ``fmt=repr`` every value is the shortest decimal that round-trips its float64, so the reader's
``float64(text) * pitch`` rounds back to the original float32 and the round trip is exact. Used for external
matches (``scripts/dressed_check.py``) and synthetic test bundles, never for the competition match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from katena.target.atd import ROOT, Timeline, _games
from katena.types import Match

CHUNK = 20000


def bundle_paths(root: Path) -> dict:
    """The tactical ATD pair under ``root``, named as ``games.asset_paths`` names them."""
    p = _games().asset_paths(data_dir=str(root), source="tactical")
    return {"atd_meta": Path(p["atd_meta"]), "atd_raw": Path(p["atd_raw"])}


def timeline_of(m: Match, name: str | None = None) -> Timeline:
    return Timeline(n_frames=int(m.n), half_blocks=tuple((int(h), int(a), int(b)) for h, a, b in m.half_blocks),
                    pitch=(float(m.pitch[0]), float(m.pitch[1])), fps=float(m.fps), name=name or f"dressed_{m.name}")


def epts_metadata(team: np.ndarray, n_frames: int, fps: float, pitch: tuple) -> str:
    """Metadata in the layout ``katena.target.layout.assumptions_hold`` accepts: raw groups in ``<Player>`` order,
    x, y, speed per group, the first ``<Team>`` the local (home) team, one specification covering 1..n."""
    ids = {0: "TH", 1: "TA"}
    players = "".join(f'<Player id="P{s + 1}" teamId="{ids[int(t)]}"><Name>Track {s + 1}</Name>'
                      f'<ShirtNumber>{s + 1}</ShirtNumber></Player>' for s, t in enumerate(team))
    chans = "".join(f'<PlayerChannel id="P{s + 1}_{c}" playerId="P{s + 1}" channelId="{c}"/>'
                    for s in range(len(team)) for c in ("x", "y", "speed"))
    groups = "".join('<SplitRegister separator=",">' + "".join(
        f'<PlayerChannelRef playerChannelId="P{s + 1}_{c}"/>' for c in ("x", "y", "speed")) + "</SplitRegister>"
        for s in range(len(team)))
    return ('<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata>'
            f'<GlobalConfig><FrameRate>{fps:g}</FrameRate><TrackingType>dressed</TrackingType></GlobalConfig>'
            f'<Sessions><Session id="S1"><MatchParameters><FieldSize><Width>{pitch[0]:g}</Width>'
            f'<Height>{pitch[1]:g}</Height></FieldSize><Score idLocalTeam="TH" idVisitingTeam="TA"/>'
            '</MatchParameters></Session></Sessions>'
            '<Teams><Team id="TH"><Name>Home</Name></Team><Team id="TA"><Name>Away</Name></Team>'
            f'</Teams><Players>{players}</Players>'
            '<Devices><Device id="D1"><Sensors><Sensor id="position"><Channels>'
            '<Channel id="x"/><Channel id="y"/><Channel id="z"/><Channel id="speed"/></Channels></Sensor></Sensors>'
            f'</Device></Devices><PlayerChannels>{chans}</PlayerChannels></Metadata>'
            '<DataFormatSpecifications>'
            f'<DataFormatSpecification id="DFS1" startFrame="1" endFrame="{int(n_frames)}" separator=":">'
            f'<StringRegister id="frameCount" name="frameCount"/><SplitRegister separator=";">{groups}</SplitRegister>'
            '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
            '<BallChannelRef channelId="z"/></SplitRegister></DataFormatSpecification>'
            '</DataFormatSpecifications></main>\n')


def _strings(a: np.ndarray, fmt) -> list:
    return ["NaN" if v != v else fmt(v) for v in a.ravel().tolist()]


def raw_lines(m: Match, fmt=repr, chunk: int = CHUNK):
    """Yield the raw file in chunks of lines, frame_id = row + 1."""
    L, W = (float(v) for v in m.pitch)
    S = m.n_slots
    for s0 in range(0, m.n, chunk):
        s1 = min(m.n, s0 + chunk)
        V = np.empty((s1 - s0, S, 3), dtype=np.float64)
        V[..., 0] = m.P[s0:s1, :, 0].astype(np.float64) / L
        V[..., 1] = m.P[s0:s1, :, 1].astype(np.float64) / W
        V[..., 2] = m.speed[s0:s1].astype(np.float64)
        B = np.empty((s1 - s0, 3), dtype=np.float64)
        B[:, 0] = m.ball[s0:s1, 0].astype(np.float64) / L
        B[:, 1] = m.ball[s0:s1, 1].astype(np.float64) / W
        B[:, 2] = np.where(np.isfinite(m.ball[s0:s1, 0]), 0.0, np.nan)
        vs, bs = _strings(V, fmt), _strings(B, fmt)
        out = []
        for i in range(s1 - s0):
            row = vs[i * 3 * S:(i + 1) * 3 * S]
            groups = ";".join(",".join(row[3 * k:3 * k + 3]) for k in range(S))
            out.append(f"{s0 + i + 1}:{groups}:{','.join(bs[3 * i:3 * i + 3])}\n")
        yield "".join(out)


def write_bundle(m: Match, root: Path, fmt=repr, name: str | None = None) -> Timeline:
    """Write ``m`` as the tactical ATD pair under ``root``; returns the timeline ``load_target`` needs to read it
    back. ``root`` must not be the competition data directory."""
    games = _games()
    root = Path(root)
    if root.resolve() in {Path(games.DATA_DIR).resolve(), (ROOT / "data").resolve()}:
        raise ValueError("refusing to write a bundle into the competition data directory")
    paths = bundle_paths(root)
    paths["atd_meta"].parent.mkdir(parents=True, exist_ok=True)
    paths["atd_meta"].write_text(epts_metadata(m.team, m.n, m.fps, m.pitch), encoding="utf-8", newline="\n")
    tmp = paths["atd_raw"].with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for block in raw_lines(m, fmt):
            fh.write(block)
    tmp.replace(paths["atd_raw"])
    return timeline_of(m, name)


def compare(canon: Match, read: Match) -> dict:
    """Field by field: is what ``load_target`` read back the canonical match? (NaN equals NaN.)"""
    def same(a, b):
        return bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True))
    return {"n_frames": canon.n == read.n, "n_slots": canon.n_slots == read.n_slots,
            "team": same(canon.team, read.team), "P": same(canon.P, read.P), "speed": same(canon.speed, read.speed),
            "ball": same(canon.ball, read.ball),
            "half_blocks": [tuple(int(v) for v in b) for b in canon.half_blocks]
            == [tuple(int(v) for v in b) for b in read.half_blocks],
            "pitch": tuple(float(v) for v in canon.pitch) == tuple(float(v) for v in read.pitch),
            "fps": float(canon.fps) == float(read.fps),
            "attack_dir": {int(k): int(v) for k, v in canon.attack_dir.items()}
            == {int(k): int(v) for k, v in read.attack_dir.items()}}
