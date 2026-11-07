"""Helpers shared by the external loaders: half assembly, attack direction, speed, npz cache.

Layout of an external `Match` (frozen conventions from `katena.match`):
- rows are frames at the source rate (25 fps for every source here);
- halves are separated by `gap` rows of X with NaN positions, so the scorer's `valid_runs`
  sees one run per half and nothing temporal crosses the break;
- `meta["frame_id"]` (int64[n], -1 in gaps) maps each row back to the source frame counter, so
  events and raw files can be joined later.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from ..match import X, Match

HALF_GAP = 250  # rows of X inserted between halves (10 s at 25 fps); any value >= 1 works


def possession_metrics():
    """The official scorer module, importable from a script as well as from pytest."""
    try:
        import possession_metrics as pm  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "evaluation"))
        import possession_metrics as pm  # noqa: F401
    return pm


def concat_halves(parts: list[dict], gap: int = HALF_GAP) -> tuple[dict, list]:
    """Stack per-half arrays in play order with `gap` X rows between consecutive halves.

    Each part is {"half": int, <name>: array with leading axis = frames of that half}. Every part
    must carry the same array names. Float arrays are padded with NaN, `y` with X, integer
    arrays with -1. Returns (arrays, half_blocks) with inclusive last frames.
    """
    names = [k for k in parts[0] if k != "half"]
    out = {k: [] for k in names}
    blocks, row = [], 0
    for i, part in enumerate(parts):
        n_half = len(part[names[0]])
        if i > 0 and gap > 0:
            for k in names:
                ref = parts[0][k]
                out[k].append(_pad_like(ref, gap, k))
            row += gap
        for k in names:
            if len(part[k]) != n_half:
                raise ValueError(f"half {part['half']}: '{k}' has {len(part[k])} rows, "
                                 f"expected {n_half}")
            out[k].append(part[k])
        blocks.append((int(part["half"]), row, row + n_half - 1))
        row += n_half
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}, blocks


def _pad_like(ref: np.ndarray, n: int, name: str) -> np.ndarray:
    shape = (n,) + ref.shape[1:]
    if name == "y":
        return np.full(shape, X, dtype=ref.dtype)
    if ref.dtype == bool:
        return np.zeros(shape, dtype=bool)
    if np.issubdtype(ref.dtype, np.floating):
        return np.full(shape, np.nan, dtype=ref.dtype)
    return np.full(shape, -1, dtype=ref.dtype)


def derive_speed(P: np.ndarray, fps: int) -> np.ndarray:
    """float32[n, S] m/s from positions: central difference, one-sided at the ends, NaN-aware."""
    n = P.shape[0]
    speed = np.full(P.shape[:2], np.nan, dtype=np.float32)
    if n < 2:
        return speed
    d = np.linalg.norm(P[2:] - P[:-2], axis=-1) * (fps / 2.0)
    speed[1:-1] = d
    speed[0] = np.linalg.norm(P[1] - P[0], axis=-1) * fps
    speed[-1] = np.linalg.norm(P[-1] - P[-2], axis=-1) * fps
    return speed.astype(np.float32)


def detect_attack_dir(P: np.ndarray, team: np.ndarray, half_blocks: list, fps: int,
                      kickoff_s: float = 10.0, min_players: int = 7) -> tuple[dict, dict]:
    """{half: +1/-1} for the HOME team, from player positions alone.

    Primary signal: median over the half of (home centroid x - away centroid x). Each centroid
    includes that team's goalkeeper at its own goal, so the home side defending -x (attacking +x)
    sits behind the away side on average. Secondary signal: the same difference over the first
    `kickoff_s` seconds of the half, when both teams line up in their own halves. Both are
    returned in the diagnostics; the primary decides, and any disagreement is flagged.
    """
    home = team == 0
    away = team == 1
    out, diag = {}, {}
    for half, first, last in half_blocks:
        x = P[first:last + 1, :, 0]
        ok_h = np.isfinite(x[:, home]).sum(axis=1) >= min_players
        ok_a = np.isfinite(x[:, away]).sum(axis=1) >= min_players
        ok = ok_h & ok_a
        with np.errstate(all="ignore"):
            diff = np.nanmean(x[:, home], axis=1) - np.nanmean(x[:, away], axis=1)
        full = float(np.median(diff[ok])) if ok.any() else float("nan")
        k = min(len(diff), int(kickoff_s * fps))
        early_ok = ok[:k]
        early = float(np.median(diff[:k][early_ok])) if early_ok.any() else float("nan")
        decide = full if np.isfinite(full) else early
        if not np.isfinite(decide) or decide == 0:
            raise ValueError(f"cannot determine attack direction for half {half}")
        out[int(half)] = 1 if decide < 0 else -1
        diag[int(half)] = {"median_diff_full_m": round(full, 3),
                           "median_diff_kickoff_m": round(early, 3),
                           "agree": bool(np.sign(full) == np.sign(early))
                           if np.isfinite(full) and np.isfinite(early) else None}
    return out, diag


# -- cache -------------------------------------------------------------------------------------

_ARR_PREFIX = "meta_arr__"


def save_match_npz(m: Match, path: str | os.PathLike) -> Path:
    """Compressed cache of a Match. Array-valued meta entries are stored as their own arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json, arrays = {}, {}
    for k, v in (m.meta or {}).items():
        if isinstance(v, np.ndarray):
            arrays[_ARR_PREFIX + k] = v
        else:
            meta_json[k] = v
    hb = np.asarray(m.half_blocks, dtype=np.int64).reshape(-1, 3)
    ad = np.asarray(sorted(m.attack_dir.items()), dtype=np.int64).reshape(-1, 2)
    payload = dict(P=m.P, team=m.team, speed=m.speed, ball=m.ball, half_blocks=hb,
                   attack_dir=ad, name=np.array(m.name), fps=np.array(m.fps),
                   pitch=np.asarray(m.pitch, dtype=np.float64),
                   meta_json=np.array(json.dumps(meta_json, sort_keys=True, default=_jsonable)),
                   **arrays)
    if m.y is not None:
        payload["y"] = m.y
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path


def load_match_npz(path: str | os.PathLike) -> Match:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        for k in z.files:
            if k.startswith(_ARR_PREFIX):
                meta[k[len(_ARR_PREFIX):]] = z[k]
        m = Match(P=z["P"], team=z["team"], speed=z["speed"], ball=z["ball"],
                  half_blocks=[tuple(int(v) for v in row) for row in z["half_blocks"]],
                  attack_dir={int(h): int(d) for h, d in z["attack_dir"]},
                  y=z["y"] if "y" in z.files else None, name=str(z["name"]),
                  fps=int(z["fps"]), pitch=tuple(float(v) for v in z["pitch"]), meta=meta)
    m.validate()
    return m


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(f"not JSON serialisable: {type(v)}")
