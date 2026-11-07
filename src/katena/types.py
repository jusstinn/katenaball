"""Shared data types (docs/PLAN.md, "Shared interfaces").

Coordinate convention: metres, origin at a pitch corner, x in [0, length] along the long axis and
y in [0, width] along the short axis. This matches the competition ATD's 0-1 x 0-1 convention
multiplied by (length, width). Home's attacking direction per half is stored in ``attack_dir``
(+1 = home attacks towards increasing x) and refers to these canonical coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_NAMES = ("H", "A", "D", "X")


@dataclass
class Match:
    """One match on a uniform frame grid.

    Frame index ``i`` of every array is the same instant. Frames outside the halves (before
    kick-off, half-time, after the final whistle) are kept so that frame indices stay aligned
    with the source, and are labelled X in ``y``.
    """

    P: np.ndarray                  # float32[n, S, 2] metres, NaN = missing
    team: np.ndarray               # int8[S] 0 home, 1 away
    speed: np.ndarray              # float32[n, S] m/s, NaN if unavailable
    ball: np.ndarray               # float32[n, 2] metres, NaN = missing
    half_blocks: list              # [(half, first_frame, last_frame)] inclusive
    attack_dir: dict               # {half: +1 or -1} for home
    y: np.ndarray | None           # int8[n] 0 H, 1 A, 2 D, 3 X; None on the target match
    pitch: tuple[float, float] = (105.0, 68.0)
    fps: float = 25.0
    name: str = ""
    source: str = ""
    # Free-form provenance. label_source says where y came from ("source_ball_possession",
    # "derived_events", "source_possession_group", ...). Also holds data-quality counters.
    meta: dict = field(default_factory=dict)
    # Optional extra per-frame arrays (first axis n), e.g. SkillCorner's detection flags.
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.P.shape[1])

    def validate(self) -> None:
        n, s = self.P.shape[:2]
        assert self.P.shape == (n, s, 2) and self.P.dtype == np.float32
        assert self.team.shape == (s,) and self.team.dtype == np.int8
        assert self.speed.shape == (n, s) and self.speed.dtype == np.float32
        assert self.ball.shape == (n, 2) and self.ball.dtype == np.float32
        if self.y is not None:
            assert self.y.shape == (n,) and self.y.dtype == np.int8
            assert self.y.min() >= 0 and self.y.max() <= 3
        for half, a, b in self.half_blocks:
            assert 0 <= a <= b < n
        for k, v in self.extras.items():
            assert v.shape[0] == n, k
        assert set(self.attack_dir) == {h for h, _, _ in self.half_blocks}


def _meta_json(m: Match) -> str:
    return json.dumps({
        "half_blocks": [[int(h), int(a), int(b)] for h, a, b in m.half_blocks],
        "attack_dir": {str(k): int(v) for k, v in m.attack_dir.items()},
        "pitch": [float(m.pitch[0]), float(m.pitch[1])],
        "fps": float(m.fps),
        "name": m.name,
        "source": m.source,
        "has_y": m.y is not None,
        "meta": m.meta,
    }, sort_keys=True, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_match(m: Match, path: str | Path) -> None:
    """Write compressed npz (float32 arrays) with the JSON meta stored inside the same file."""
    m.validate()
    arrays = {
        "P": m.P, "team": m.team, "speed": m.speed, "ball": m.ball,
        "meta_json": np.frombuffer(_meta_json(m).encode(), dtype=np.uint8),
    }
    if m.y is not None:
        arrays["y"] = m.y
    for k, v in m.extras.items():
        arrays[f"extra__{k}"] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_match(path: str | Path) -> Match:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode())
        y = z["y"] if meta["has_y"] else None
        m = Match(
            P=z["P"], team=z["team"], speed=z["speed"], ball=z["ball"],
            half_blocks=[tuple(b) for b in meta["half_blocks"]],
            attack_dir={int(k): v for k, v in meta["attack_dir"].items()},
            y=y, pitch=tuple(meta["pitch"]), fps=meta["fps"], name=meta["name"],
            source=meta["source"], meta=meta["meta"],
            extras={k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")},
        )
    m.validate()
    return m
