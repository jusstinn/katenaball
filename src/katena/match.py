"""The shared in-memory match format. Every module reads and writes `Match`.

Provisional copy of the interface in docs/PLAN.md ("Shared interfaces"), written on the D-side
so D1-D3 can start before J1 lands. If J1 defines `Match` differently, J1 wins and this file is
reconciled to it.

Conventions (frozen):
- Frame index 0..n-1 is the row of the submission for the target match.
- Coordinates in metres, origin at the pitch centre: x along the length in [-L/2, L/2],
  y across in [-W/2, W/2]. NaN = missing.
- `team[s]` is 0 for home, 1 for away. Slots that are not players (referees) are excluded.
- `half_blocks` uses inclusive last frames, like `games.half_video_blocks`.
- `attack_dir[half]` is +1 if the HOME team attacks towards +x in that half, else -1.
- Labels `y` use the scorer's codes: 0 H, 1 A, 2 D, 3 X.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
FPS = 25
PITCH = (105.3, 68.0)  # metres, from games.py


@dataclass
class Match:
    P: np.ndarray            # float32[n, S, 2], metres, NaN = missing
    team: np.ndarray         # int8[S], 0 home, 1 away
    speed: np.ndarray        # float32[n, S], m/s, NaN if unavailable
    ball: np.ndarray         # float32[n, 2], metres, NaN = missing
    half_blocks: list        # [(half, first_frame, last_frame)], last inclusive
    attack_dir: dict         # {half: +1 or -1} for home
    y: np.ndarray | None = None   # int8[n], 0 H 1 A 2 D 3 X; None on the target match
    name: str = ""
    fps: int = FPS
    pitch: tuple = PITCH
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.P.shape[1])

    def half_of_frame(self) -> np.ndarray:
        """int8[n]: half number for each frame, 0 outside every half."""
        out = np.zeros(self.n, dtype=np.int8)
        for half, first, last in self.half_blocks:
            out[first:last + 1] = half
        return out

    def graded_mask(self) -> np.ndarray:
        return self.half_of_frame() > 0

    def validate(self) -> None:
        """Raise ValueError if shapes, dtypes or conventions are violated."""
        n, S = self.n, self.n_slots
        if self.P.shape != (n, S, 2) or self.P.dtype != np.float32:
            raise ValueError(f"P must be float32[n,S,2], got {self.P.dtype}{self.P.shape}")
        if self.team.shape != (S,) or not np.isin(self.team, (0, 1)).all():
            raise ValueError("team must be int8[S] with values in {0, 1}")
        if self.speed.shape != (n, S):
            raise ValueError(f"speed must be [n,S], got {self.speed.shape}")
        if self.ball.shape != (n, 2) or self.ball.dtype != np.float32:
            raise ValueError(f"ball must be float32[n,2], got {self.ball.dtype}{self.ball.shape}")
        for half, first, last in self.half_blocks:
            if not (0 <= first <= last < n):
                raise ValueError(f"half block {(half, first, last)} outside 0..{n - 1}")
            if self.attack_dir.get(half) not in (1, -1):
                raise ValueError(f"attack_dir missing or invalid for half {half}")
        if self.y is not None:
            if self.y.shape != (n,) or not np.isin(self.y, (H, A, D, X)).all():
                raise ValueError("y must be int8[n] with values in {0,1,2,3}")


def from_normalised(xy01: np.ndarray, pitch=PITCH) -> np.ndarray:
    """ATD-style 0..1 coordinates to centred metres. NaN passes through."""
    out = np.asarray(xy01, dtype=np.float32).copy()
    out[..., 0] = (out[..., 0] - 0.5) * pitch[0]
    out[..., 1] = (out[..., 1] - 0.5) * pitch[1]
    return out


def labels_to_str(codes: np.ndarray) -> np.ndarray:
    return np.array(LABELS, dtype=object)[np.asarray(codes, dtype=np.int64)]
