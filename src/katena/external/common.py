"""Shared checks and helpers for the external loaders (no I/O, no dataset specifics)."""

from __future__ import annotations

import warnings

import numpy as np

from katena.types import A, D, H, Match, X

# Ball further than this outside the touchlines/goal lines is treated as a tracking glitch and
# blanked; anything between 0 and this margin is real (throw-ins, goal kicks) and kept.
BALL_GLITCH_MARGIN_M = 15.0


SPEED_MAX_MS = 12.5          # faster than any footballer: a tracking jump, not motion


def derive_speed(P: np.ndarray, fps: float, vmax: float = SPEED_MAX_MS) -> np.ndarray:
    """Central-difference speed in m/s, NaN wherever either neighbour is missing or the value
    exceeds ``vmax`` (position jumps). Use :func:`count_speed_blanked` for the number blanked."""
    v = _central_speed(P, fps)
    v[v > vmax] = np.nan
    return v


def _central_speed(P: np.ndarray, fps: float) -> np.ndarray:
    v = np.full(P.shape[:2], np.nan, dtype=np.float32)
    d = P[2:] - P[:-2]
    v[1:-1] = np.sqrt((d.astype(np.float64) ** 2).sum(-1)) * (fps / 2.0)
    return v


def count_speed_blanked(P: np.ndarray, fps: float, vmax: float = SPEED_MAX_MS) -> int:
    with np.errstate(invalid="ignore"):
        return int((_central_speed(P, fps) > vmax).sum())


def infer_attack_dir_gk(P: np.ndarray, team: np.ndarray, first: int, last: int,
                        length: float) -> int:
    """Goalkeeper heuristic: the slot of each team with the largest mean distance from the centre
    spot is taken as its goalkeeper and stands on the side that team defends. Returns 0 if the
    two teams' keepers are on the same side. Fragile: a high-standing keeper (seen in IDSSE
    J03WR9, median 23 m off his line) defeats it; kept as a second opinion only."""
    seg = P[first:last + 1, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN slots (never on pitch)
        mean_x = np.nanmean(seg, axis=0) - length / 2.0
    sides = []
    for t in (0, 1):
        idx = np.flatnonzero((team == t) & np.isfinite(mean_x))
        if len(idx) == 0:
            return 0
        gk = idx[np.argmax(np.abs(mean_x[idx]))]
        sides.append(np.sign(mean_x[gk]))
    if sides[0] == sides[1] or 0 in sides:
        return 0
    return int(-sides[0])  # home keeper on the left (negative) defends left, attacks right (+1)


def infer_attack_dir(P: np.ndarray, team: np.ndarray, first: int, last: int,
                     length: float, min_gap_m: float = 2.0) -> int:
    """Home's attacking direction (+1 towards increasing x) in frames [first, last] from the
    two teams' mean x over the block: the team defending the left goal has the smaller mean x.
    Returns 0 when the centroids are closer than ``min_gap_m`` (ambiguous)."""
    seg = P[first:last + 1, :, 0]
    mx = []
    for t in (0, 1):
        v = seg[:, team == t]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0
        mx.append(float(v.mean()))
    if abs(mx[0] - mx[1]) < min_gap_m:
        return 0
    return +1 if mx[0] < mx[1] else -1


def nearest_team_agreement(m: Match) -> dict:
    """For alive frames with a ball and a team label, how often is the nearest player of the
    labelled team. An inverted H/A mapping would give a value well below 0.5."""
    assert m.y is not None
    alive = np.isin(m.y, (H, A)) & np.isfinite(m.ball[:, 0])
    idx = np.flatnonzero(alive)
    agree = 0
    dist_lab, dist_opp = [], []
    for s in range(0, len(idx), 20000):
        ii = idx[s:s + 20000]
        d = np.sqrt(((m.P[ii] - m.ball[ii, None, :]) ** 2).sum(-1))       # [k, S]
        d = np.where(np.isfinite(d), d, np.inf)
        lab = m.y[ii].astype(np.int8)
        own = (m.team[None, :] == lab[:, None])
        d_own = np.where(own, d, np.inf).min(1)
        d_opp = np.where(~own, d, np.inf).min(1)
        ok = np.isfinite(d_own) & np.isfinite(d_opp)
        agree += int((d_own[ok] < d_opp[ok]).sum())
        dist_lab.append(d_own[ok])
        dist_opp.append(d_opp[ok])
    n_ok = sum(len(a) for a in dist_lab)
    if n_ok == 0:
        return {"n": 0, "agree": float("nan"), "mean_d_labelled": float("nan"),
                "mean_d_other": float("nan")}
    dl, do = np.concatenate(dist_lab), np.concatenate(dist_opp)
    return {"n": n_ok, "agree": agree / n_ok, "mean_d_labelled": float(dl.mean()),
            "mean_d_other": float(do.mean())}


def quality_report(m: Match) -> dict:
    """Aggregate data-quality figures (stored in ``Match.meta['quality']``)."""
    L, W = m.pitch
    inplay = np.zeros(m.n, bool)
    for _, a, b in m.half_blocks:
        inplay[a:b + 1] = True
    active = np.isfinite(m.P[..., 0])
    act_in = active[inplay].sum(1)
    ball_ok = np.isfinite(m.ball[:, 0])
    bx, by = m.ball[:, 0], m.ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = ball_ok & ((bx < 0) | (bx > L) | (by < 0) | (by > W))
    with np.errstate(invalid="ignore"):
        px, py = m.P[..., 0], m.P[..., 1]
        far = active & ((px < -5) | (px > L + 5) | (py < -5) | (py > W + 5))
    q = {
        "players_far_outside_pitch_share": float(far[inplay].sum() / max(active[inplay].sum(), 1)),
        "n_frames": m.n,
        "n_inplay": int(inplay.sum()),
        "n_slots": m.n_slots,
        "active_slots_min": int(act_in.min()) if len(act_in) else 0,
        "active_slots_mean": float(act_in.mean()) if len(act_in) else 0.0,
        "active_slots_max": int(act_in.max()) if len(act_in) else 0,
        "ball_nan_share_inplay": float((~ball_ok[inplay]).mean()) if inplay.any() else float("nan"),
        "ball_outside_pitch_inplay": int(outside[inplay].sum()),
        "ball_outside_pitch_alive": int((outside & np.isin(m.y, (H, A))).sum())
        if m.y is not None else None,
        "ball_outside_pitch_dead": int((outside & (m.y == D)).sum()) if m.y is not None else None,
        "speed_nan_share_inplay": float(np.isnan(m.speed[inplay][active[inplay]]).mean())
        if inplay.any() else float("nan"),
    }
    if m.y is not None:
        q["share"] = {n: float((m.y[inplay] == c).mean()) for c, n in enumerate("HADX")} \
            if inplay.any() else {}
        q["y_x_inside_blocks"] = int((m.y[inplay] == X).sum())
        q["nearest_team"] = nearest_team_agreement(m)
    return q


def blank_far_ball(ball: np.ndarray, length: float, width: float) -> int:
    """NaN-out ball positions more than BALL_GLITCH_MARGIN_M beyond the pitch; returns count."""
    mg = BALL_GLITCH_MARGIN_M
    with np.errstate(invalid="ignore"):
        bad = (ball[:, 0] < -mg) | (ball[:, 0] > length + mg) | \
              (ball[:, 1] < -mg) | (ball[:, 1] > width + mg)
    ball[bad] = np.nan
    return int(bad.sum())


def validate_canonical(m: Match, min_team_agreement: float | None = 0.6) -> None:
    """Hard invariants every canonical Match must satisfy; raises AssertionError otherwise.

    ``min_team_agreement`` is the floor for :func:`nearest_team_agreement` (None skips it, for
    synthetic fixtures). An inverted H/A mapping would score about 1 - 0.75 = 0.25.
    """
    m.validate()
    L, W = m.pitch
    assert m.fps > 0 and L > 90 and W > 55
    assert set(np.unique(m.team)) <= {0, 1} and (m.team == 0).any() and (m.team == 1).any()
    assert len(m.half_blocks) == 2
    (_, a1, b1), (_, a2, b2) = m.half_blocks
    assert a1 <= b1 < a2 <= b2, "halves overlap or are out of order"
    assert m.attack_dir[1] == -m.attack_dir[2] and abs(m.attack_dir[1]) == 1
    finite = np.isfinite(m.P[..., 0])
    assert (np.isfinite(m.P[..., 1]) == finite).all(), "x/y NaN pattern differs"
    assert np.nanmin(m.P) > -30 and np.nanmax(m.P[..., 0]) < L + 30 and np.nanmax(m.P[..., 1]) < W + 30
    fb = np.isfinite(m.ball[:, 0])
    assert (np.isfinite(m.ball[:, 1]) == fb).all()
    assert (m.speed[finite][np.isfinite(m.speed[finite])] >= 0).all()
    assert np.nanmax(m.speed) < 14.0, "speed above 14 m/s: unit error?"
    if m.y is not None:
        inb = np.zeros(m.n, bool)
        for _, a, b in m.half_blocks:
            inb[a:b + 1] = True
        assert (m.y[~inb] == X).all(), "labels outside the half blocks must be X"
        if min_team_agreement is not None:
            ag = nearest_team_agreement(m)
            assert ag["agree"] >= min_team_agreement, f"team mapping suspect: {ag}"
            assert ag["mean_d_labelled"] < ag["mean_d_other"]


def runs_from_blocks(m: Match) -> list[tuple[int, int]]:
    """Half-open [start, stop) spans of the in-play blocks."""
    return [(a, b + 1) for _, a, b in m.half_blocks]


__all__ = ["derive_speed", "infer_attack_dir", "nearest_team_agreement", "quality_report",
           "blank_far_ball", "runs_from_blocks", "validate_canonical", "infer_attack_dir_gk", "D"]
