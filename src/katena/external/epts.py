"""Generic FIFA/EPTS (Metrica) metadata XML + raw TXT pair -> `Match`.

The same format as the competition ATD, so this loader is written to be source-agnostic: it
reads the data layout from the metadata's DataFormatSpecifications (several of them when the
channel layout changes mid-match, e.g. at substitutions) through kloppy's EPTS reader, and
writes straight into numpy arrays instead of building kloppy Frame objects.

NOTE: this overlaps Justin's J1 ATD loader. J1 owns the competition loader; this one exists so
the external Metrica sample can be loaded now, and should be reconciled with J1 when it lands.

Conventions applied:
- Home is the metadata's local team (Score@idLocalTeam), else the first Team listed, as
  kloppy's `load_metadata` does and as `starter/baseline.py` assumes for the competition.
- Coordinates: EPTS positions are normalised 0..1 (x left->right, y top->bottom) and are turned
  into centred metres with `katena.match.from_normalised` and the pitch size (metadata FieldSize
  unless `pitch` is given, e.g. games.py's 105.3 x 68.0).
- Row r corresponds to source frame `first_frame + r` within the source timeline; when
  `half_gap > 0`, `half_gap` X rows are inserted between consecutive halves (needed when the
  source halves are adjacent, otherwise the scorer would see one run). `meta["frame_id"]` keeps
  the source frame of every row (-1 in inserted gaps).
- Speed: the provider's speed channel if the layout has one, else derived from positions.
- `y` is None: EPTS tracking carries no possession labels.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from ..match import FPS, Match, from_normalised
from .common import concat_halves, derive_speed, detect_attack_dir


def _kloppy_epts():
    from kloppy.domain import Orientation
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
    from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data
    return load_metadata, read_raw_data, Orientation


def read_epts_metadata(meta_path):
    load_metadata, _, _ = _kloppy_epts()
    with open(meta_path, "rb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_metadata(fh)


def _speed_key(md) -> str | None:
    """Channel id of a per-player speed channel, if the layout has one."""
    for sensor in md.sensors:
        sid = sensor.sensor_id.lower()
        if "speed" in sid or sid in ("s", "velocity"):
            return sensor.channels[0].channel_id
    return None


def load_epts(meta_path, raw_path, half_blocks: list | None = None, half_gap: int = 0,
              pitch: tuple | None = None, name: str | None = None,
              derive_missing_speed: bool = True) -> Match:
    """Load one EPTS pair.

    half_blocks: [(half, first_frame, last_frame)] in SOURCE ROW indices (row 0 = first source
        frame, inclusive last), e.g. `games.half_video_blocks(...)` for the competition ATD.
        Default: the metadata periods (ProviderGlobalParameters first_half_start/end ...).
    half_gap: X rows to insert between halves. Rows outside every half are kept and are X in
        `half_of_frame`; with half_gap > 0 the source frames between two halves are kept too and
        the gap is added after them.
    """
    _, read_raw_data, Orientation = _kloppy_epts()
    md = read_epts_metadata(meta_path)
    fps = int(md.frame_rate or FPS)
    specs = md.data_format_specifications
    f0 = min(s.start_frame for s in specs)
    f1 = max(s.end_frame for s in specs)
    n_src = f1 - f0 + 1

    players = [(0 if ti == 0 else 1, p) for ti, team in enumerate(md.teams) for p in team.players]
    team = np.array([t for t, _ in players], dtype=np.int8)
    S = len(players)
    keys_x = [f"player_{p.player_id}_x" for _, p in players]
    keys_y = [f"player_{p.player_id}_y" for _, p in players]
    skey = _speed_key(md)
    keys_s = [f"player_{p.player_id}_{skey}" for _, p in players] if skey else None

    P01 = np.full((n_src, S, 2), np.nan, dtype=np.float32)
    ball01 = np.full((n_src, 2), np.nan, dtype=np.float32)
    speed = np.full((n_src, S), np.nan, dtype=np.float32)
    seen = np.zeros(n_src, dtype=bool)
    with open(raw_path, "rb") as fh:
        for row in read_raw_data(fh, md, sensor_ids=[s.sensor_id for s in md.sensors]):
            r = int(row["frame_id"]) - f0
            if not (0 <= r < n_src):
                continue
            seen[r] = True
            P01[r, :, 0] = [row.get(k, np.nan) for k in keys_x]
            P01[r, :, 1] = [row.get(k, np.nan) for k in keys_y]
            if keys_s:
                speed[r] = [row.get(k, np.nan) for k in keys_s]
            ball01[r] = (row.get("ball_x", np.nan), row.get("ball_y", np.nan))

    if pitch is None:
        pd_ = md.pitch_dimensions
        pitch = (float(pd_.pitch_length), float(pd_.pitch_width)) if pd_ else (105.0, 68.0)
    P = from_normalised(P01, pitch)
    ball = from_normalised(ball01, pitch)
    del P01, ball01
    if not keys_s and derive_missing_speed:
        speed = derive_speed(P, fps)
        speed_source = "derived from positions (central difference)"
    else:
        speed_source = f"provider channel '{skey}'" if keys_s else "none"
    frame_id = np.arange(f0, f1 + 1, dtype=np.int64)

    if half_blocks is None:
        half_blocks = []
        for period in md.periods:
            a = int(round(period.start_timestamp.total_seconds() * fps)) - f0
            b = int(round(period.end_timestamp.total_seconds() * fps)) - f0
            half_blocks.append((int(period.id), max(a, 0), min(b, n_src - 1)))
    half_blocks = [tuple(int(v) for v in hb) for hb in half_blocks]

    if half_gap > 0 and len(half_blocks) > 1:
        # split the source timeline right after each half's last frame and insert X rows there
        cuts = [hb[2] + 1 for hb in half_blocks[:-1]]
        bounds = [0] + cuts + [n_src]
        parts = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            parts.append(dict(half=i + 1, P=P[a:b], speed=speed[a:b], ball=ball[a:b],
                              frame_id=frame_id[a:b], seen=seen[a:b]))
        arr, _ = concat_halves(parts, gap=half_gap)
        shift = [i * half_gap for i in range(len(half_blocks))]
        half_blocks = [(h, a + s, b + s) for (h, a, b), s in zip(half_blocks, shift)]
        P, speed, ball, frame_id = arr["P"], arr["speed"], arr["ball"], arr["frame_id"]
        seen = arr["seen"].astype(bool)
        seen[frame_id < 0] = False

    attack_dir, ad_diag = detect_attack_dir(P, team, half_blocks, fps)
    if md.orientation in (Orientation.HOME_AWAY, Orientation.AWAY_HOME):
        first = 1 if md.orientation == Orientation.HOME_AWAY else -1
        for half in attack_dir:
            provider = first if half % 2 == 1 else -first
            ad_diag[half]["provider"] = provider
            ad_diag[half]["provider_agrees"] = bool(provider == attack_dir[half])

    m = Match(P=P, team=team, speed=speed.astype(np.float32), ball=ball,
              half_blocks=half_blocks, attack_dir=attack_dir, y=None,
              name=name or os.path.splitext(os.path.basename(str(raw_path)))[0], fps=fps,
              pitch=tuple(pitch),
              meta={"source": "epts", "home": md.teams[0].name, "away": md.teams[1].name,
                    "slot_ids": [p.player_id for _, p in players],
                    "coords": "centred metres from normalised EPTS (y down)",
                    "speed_source": speed_source, "attack_dir_diag": ad_diag,
                    "first_source_frame": int(f0), "n_source_frames": int(n_src),
                    "source_frames_missing": int((~seen[frame_id >= 0]).sum()),
                    "frame_id": frame_id})
    m.validate()
    return m


def rows_for_frames(frame_id: np.ndarray, frames) -> np.ndarray:
    """Row index of each source frame number (-1 where the frame is not in the Match)."""
    frame_id = np.asarray(frame_id)
    valid = frame_id >= 0
    order = np.argsort(frame_id[valid], kind="stable")
    src = frame_id[valid][order]
    rows = np.flatnonzero(valid)[order]
    frames = np.asarray(frames, dtype=np.int64)
    pos = np.searchsorted(src, frames)
    pos = np.clip(pos, 0, max(len(src) - 1, 0))
    out = np.where((len(src) > 0) & (src[pos] == frames), rows[pos], -1)
    return out.astype(np.int64)


__all__ = ["load_epts", "read_epts_metadata", "rows_for_frames"]
