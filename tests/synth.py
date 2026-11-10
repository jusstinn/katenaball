"""Tiny synthetic fixtures (no external data needed)."""

from __future__ import annotations

import numpy as np

from katena.types import Match

H_, A_ = 0, 1

L, W = 105.0, 68.0


def make_match(seed: int = 0, invert_labels: bool = False) -> Match:
    """Two 700-frame halves at 25 fps, 11+11 players, a ball that follows the possessing team.

    Home defends the left goal in half 1 (attack_dir[1] = +1), the right goal in half 2.
    Labels per half: H(150) A(100) D(50) A(150) H(100) D(50) H(100).
    """
    rng = np.random.default_rng(seed)
    n = 2000
    blocks = [(1, 100, 799), (2, 1100, 1799)]
    S = 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    y = np.full(n, 3, dtype=np.int8)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    pattern = [(0, 150), (1, 100), (2, 50), (1, 150), (0, 100), (2, 50), (0, 100)]
    base = np.stack([np.linspace(10, 60, 11), np.linspace(10, 58, 11)], 1)
    base[0] = (4.0, W / 2)                                  # goalkeeper
    for half, a, b in blocks:
        home_dir = 1 if half == 1 else -1
        t = np.arange(a, b + 1)
        for s in range(S):
            if s < 11:
                px = base[s, 0] if home_dir == 1 else L - base[s, 0]
            else:
                px = L - base[s - 11, 0] if home_dir == 1 else base[s - 11, 0]
            P[t, s, 0] = px + rng.normal(0, 0.05, len(t))
            P[t, s, 1] = base[s % 11, 1] + rng.normal(0, 0.05, len(t))
        f = a
        for code, length in pattern:
            seg = slice(f, f + length)
            y[seg] = code
            if code == 2:
                ball[seg] = (L / 2, W / 2)
            else:
                carrier = 5 if code == 0 else 16
                ball[seg] = P[seg, carrier] + 0.4
            f += length
    y_out = y.copy()
    if invert_labels:                       # swap H and A, leave D and X
        y_out[y == H_] = A_
        y_out[y == A_] = H_
    speed = np.where(np.isfinite(P[..., 0]), 1.5, np.nan).astype(np.float32)
    ball[::7] = np.nan                                       # some missing ball frames
    return Match(P=P, team=team, speed=speed, ball=ball, half_blocks=blocks,
                 attack_dir={1: +1, 2: -1}, y=y_out, pitch=(L, W), fps=25.0,
                 name="synthetic", source="synthetic", meta={"label_source": "synthetic"})


def tagging_xml(instances) -> str:
    """A Smart Tagging file in the documented SportsCode layout (docs/upstream/DATA.md, starter/load_smart_tagging.py).

    ``instances``: ``(start_s, end_s, code, {group: text})``, written in the order given. Seconds are written with
    ``repr``, so ``round(seconds * 25)`` gives back the frame the caller meant."""
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<file><ALL_INSTANCES>"]
    for i, (start, end, code, labels) in enumerate(instances, start=1):
        body = "".join(f"<label><group>{g}</group><text>{t}</text></label>" for g, t in labels.items())
        out.append(f"<instance><ID>{i}</ID><start>{float(start)!r}</start><end>{float(end)!r}</end>"
                   f"<code>{code}</code>{body}</instance>")
    out.append("</ALL_INSTANCES></file>")
    return "\n".join(out) + "\n"


def write_tagging(root, instances):
    """Write ``tagging_xml(instances)`` where ``games.asset_paths(data_dir=root)`` looks for the Smart Tagging file."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import games
    path = Path(games.asset_paths(data_dir=str(root))["smart_tagging"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tagging_xml(instances).encode("utf-8"))
    return path
