"""Label-free fit of the corruption config to a target match, and severity grids around it.

The fit is plain moment matching:

1. Read off what the target states directly: slot count and team split, graded ball-present rate,
   mean present objects. `corrupt` solves its internal rates to hit the last two exactly.
2. Coordinate search over the shape parameters (gap and fragment lengths, congestion and speed
   effects, duplicates, ghosts, jitter, slot policy) to minimise a weighted distance between
   `target_stats(corrupt(clean, cfg))` and `target_stats(target)`, with common random numbers.

Only `target_stats` of the target is used, so the target's labels are never read.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..match import X, Match
from .sim import DEFAULT_CFG, SLOT_POLICIES, corrupt, resolve_cfg
from .stats import target_stats

# (stat, weight, kind, scale):
#   "log": ln((sim + scale) / (tgt + scale)) / ln(2), so a factor of 2 off costs 1 (scale = floor)
#   "abs": (sim - tgt) / scale, for shares and log-ratios
LOSS_SPEC = [
    ("ball_gap_q50_s", 1.0, "log", 0.04),
    ("ball_gap_q90_s", 1.0, "log", 0.04),
    ("ball_gap_q99_s", 0.5, "log", 0.2),
    ("ball_gaps_per_min", 1.0, "log", 0.5),
    ("ball_iso_share", 1.0, "abs", 0.05),
    ("ball_bursts_per_min", 1.0, "log", 0.5),
    ("ball_burst_q50_s", 1.0, "log", 0.04),
    ("ball_burst_q90_s", 1.0, "log", 0.2),
    ("ball_vis_q50_s", 0.5, "log", 0.04),
    ("ball_vis_q90_s", 0.5, "log", 0.2),
    ("ball_loss_logratio_contest", 1.0, "abs", 0.25),
    ("ball_loss_logratio_fast", 1.0, "abs", 0.25),
    ("ball_gap_logratio_near", 0.5, "abs", 0.25),
    ("ball_present_lowact_ratio", 1.0, "abs", 0.05),
    ("ball_d2_mad", 1.0, "log", 0.01),
    ("ball_jump_rate", 0.5, "log", 0.002),
    ("slot_run_q10_s", 1.0, "log", 0.04),
    ("slot_run_q50_s", 1.5, "log", 0.1),
    ("slot_run_q90_s", 1.0, "log", 0.5),
    ("slot_onsets_per_min", 1.5, "log", 1.0),
    ("slot_gap_q50_s", 0.5, "log", 0.04),
    ("slots_present_q05", 0.5, "log", 1.0),
    ("slots_present_std", 1.0, "log", 0.1),
    ("slots_present_std_5s", 1.0, "log", 0.05),
    ("team_count_absdiff_mean", 1.0, "log", 0.1),
    ("slots_present_frac_gt22", 1.0, "log", 0.005),
    ("dup_pairs_per_frame", 1.0, "log", 0.005),
    ("opp_close_pairs_per_frame", 0.5, "log", 0.005),
    ("slot_jump_rate", 1.0, "log", 0.0005),
    ("player_d2_mad", 1.0, "log", 0.005),
    ("speed_pos_consistency", 0.5, "log", 0.05),
    ("slots_used", 0.5, "log", 1.0),
    ("slot_occ_q10", 0.5, "abs", 0.05),
    ("slot_occ_q50", 0.5, "abs", 0.05),
    ("slot_occ_q90", 0.5, "abs", 0.05),
    ("oob_frac", 0.5, "log", 0.002),
]
_LN2 = math.log(2.0)
_HUBER = 3.0  # errors beyond this many tolerance units grow linearly, not quadratically

# (param, lo, hi, scale), searched in this order
SEARCH_SPACE = [
    ("ball_gap_median_s", 0.04, 20.0, "log"),
    ("ball_iso_drop_share", 0.0, 0.9, "lin"),
    ("frag_median_s", 0.5, 300.0, "log"),
    ("p_gap", 0.05, 1.0, "lin"),
    ("ball_gap_sigma", 0.1, 2.5, "lin"),
    ("frag_sigma", 0.1, 2.5, "lin"),
    ("dup_rate_per_min", 0.01, 100.0, "log"),
    ("pos_jitter_sd_m", 0.002, 1.0, "log"),
    ("ball_jitter_sd_m", 0.01, 3.0, "log"),
    ("ball_beta_cong", -1.0, 3.0, "lin"),
    ("ball_beta_speed", -1.5, 3.0, "lin"),
    ("ball_beta_air", -1.5, 3.0, "lin"),
    ("ball_beta_dead", -2.0, 3.0, "lin"),
    ("gap_sigma", 0.1, 2.5, "lin"),
    ("ghost_rate_per_min", 0.01, 100.0, "log"),
    ("swap_prob", 0.0, 1.0, "lin"),
    ("ball_gap_cong_gamma", -0.5, 1.0, "lin"),
    ("ball_outlier_rate", 1e-5, 0.05, "log"),
    ("pos_wobble_sd_m", 0.01, 2.0, "log"),
    ("dup_median_s", 0.1, 30.0, "log"),
    ("speed_win_s", 0.0, 1.0, "lin"),
    ("team_flip_rate", 0.0, 0.2, "lin"),
    ("swap_dist_m", 0.5, 5.0, "lin"),
]


def stats_loss(sim: dict, tgt: dict, spec=LOSS_SPEC, noise: dict | None = None,
               tol: float = 0.05) -> float:
    """Weighted Huber distance between two `target_stats` dicts; stats NaN on either side are skipped.

    Each stat is put in loss units (log2 for "log" stats, so 0.1 units is about 7%) and its error
    divided by max(seed-to-seed noise of that stat, tol): precise stats count fully, noisy ones less.
    """
    tot, wsum = 0.0, 0.0
    noise = noise or {}
    for key, w, kind, scale in spec:
        a, b = sim.get(key, math.nan), tgt.get(key, math.nan)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        err = (_unit(a, kind, scale) - _unit(b, kind, scale)) / max(noise.get(key, 0.0), tol)
        e = abs(err)
        tot += w * (e * e if e <= _HUBER else _HUBER * (2 * e - _HUBER))
        wsum += w
    return tot / wsum if wsum else math.nan


def _unit(v: float, kind: str, scale: float) -> float:
    """A stat in loss units: log2(v + floor) for "log" stats, v / scale for "abs" stats."""
    return math.log(max(v, 0.0) + scale) / _LN2 if kind == "log" else v / scale


def stat_noise(stats: list[dict], spec=LOSS_SPEC) -> dict:
    """Seed-to-seed standard deviation of each stat in loss units, from repeated corruptions."""
    out = {}
    for key, _, kind, scale in spec:
        v = [_unit(s[key], kind, scale) for s in stats if math.isfinite(s.get(key, math.nan))]
        out[key] = float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0
    return out


def crop_match(m: Match, max_graded_frames: int, spacer: int = 25) -> Match:
    """Up to `max_graded_frames` graded frames, taken evenly from the start of each half, joined by
    short empty spacers so no track runs across a cut. Ungraded frames are dropped."""
    blocks = [(h, f, l) for h, f, l in m.half_blocks if l >= f] or [(1, 0, m.n - 1)]
    per = max(1, int(max_graded_frames) // len(blocks))
    pieces, hb, t = [], [], 0
    for i, (h, f, l) in enumerate(blocks):
        e = min(l + 1, f + per)
        pieces.append((f, e))
        hb.append((h, t, t + (e - f) - 1))
        t += (e - f) + (spacer if i < len(blocks) - 1 else 0)
    n = t
    S = m.n_slots
    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    sp = np.full((n, S), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    y = None if m.y is None else np.full(n, X, dtype=np.int8)
    for (f, e), (_, a, b) in zip(pieces, hb):
        P[a:b + 1], sp[a:b + 1], ball[a:b + 1] = m.P[f:e], m.speed[f:e], m.ball[f:e]
        if y is not None:
            y[a:b + 1] = m.y[f:e]
    attack = {h: m.attack_dir.get(h, 1) for h, _, _ in hb}
    return Match(P=P, team=m.team.copy(), speed=sp, ball=ball, half_blocks=hb, attack_dir=attack,
                 y=y, name=f"{m.name}[crop]", fps=m.fps, pitch=m.pitch, meta=dict(m.meta))


def _direct(cfg: dict, tgt: dict) -> dict:
    """Parameters the target states directly."""
    cfg = dict(cfg)

    def ok(k):
        return k in tgt and math.isfinite(tgt[k])

    if ok("n_slots") and tgt["n_slots"] >= 2:
        cfg["n_slots"] = int(round(tgt["n_slots"]))
    if ok("home_slot_frac") and 0 < tgt["home_slot_frac"] < 1:
        cfg["home_slot_frac"] = float(tgt["home_slot_frac"])
    if ok("ball_present_rate"):
        cfg["ball_present_rate"] = float(np.clip(tgt["ball_present_rate"], 0.02, 1.0))
    if ok("slots_present_mean"):
        cfg["players_present_mean"] = float(np.clip(tgt["slots_present_mean"], 1.0, 44.0))
    if ok("ball_iso_share"):
        cfg["ball_iso_drop_share"] = float(np.clip(tgt["ball_iso_share"], 0.0, 0.9))
    if ok("slot_run_q50_s") and tgt["slot_run_q50_s"] > 0:
        cfg["frag_median_s"] = float(np.clip(tgt["slot_run_q50_s"], 0.5, 300.0))
    return cfg


def fit_noise_cfg(m_target: Match, m_clean: Match | None = None, *, base_cfg: dict | None = None,
                  max_evals: int = 150, max_graded_minutes: float = 20.0, seed: int = 0,
                  search_policy: bool = True, n_noise_seeds: int = 3,
                  verbose: bool = False) -> dict:
    """Fit a corruption config so that target_stats(corrupt(m_clean, cfg)) approximates
    target_stats(m_target). Uses no labels of m_target.

    `m_clean` is the clean base to corrupt. Use an external clean match (D1) for a real fit; the
    default is a synthetic match, whose own kinematics bias the jitter and speed parameters.
    The returned cfg carries a `_fit` entry with aggregate diagnostics; `corrupt` ignores it.
    """
    t0 = time.time()
    tgt = target_stats(m_target)
    if m_clean is None:
        from ..synth import synthetic_match
        m_clean = synthetic_match(seed=0, half_minutes=min(10.0, max_graded_minutes / 2))
        base_name = "synthetic"
    else:
        base_name = m_clean.name or "clean"
    base = crop_match(m_clean, int(max_graded_minutes * 60 * m_clean.fps))

    cfg = _direct(resolve_cfg(base_cfg), tgt)
    cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    cache: dict = {}
    n_evals = 0

    # seed-to-seed noise of each stat at the starting point sets how much each stat is trusted
    reps = [target_stats(corrupt(base, cfg, seed + 1000 + i)) for i in range(max(0, n_noise_seeds))]
    noise = stat_noise(reps) if len(reps) >= 2 else {}

    def evaluate(c):
        nonlocal n_evals
        k = tuple(sorted((a, (round(b, 9) if isinstance(b, float) else b)) for a, b in c.items()))
        if k not in cache:
            st = target_stats(corrupt(base, c, seed))
            cache[k] = (stats_loss(st, tgt, noise=noise), st)
            n_evals += 1
        return cache[k]

    best, best_st = evaluate(cfg)
    loss_init = best
    if verbose:
        print(f"  fit: start loss {best:.4f}")

    def try_policies():
        nonlocal best, best_st, cfg
        for pol in SLOT_POLICIES if search_policy else ():
            if pol == cfg["slot_policy"] or n_evals >= max_evals:
                continue
            c = {**cfg, "slot_policy": pol}
            l, st = evaluate(c)
            if l < best:
                best, best_st, cfg = l, st, c

    def move(v, name, lo, hi, scale, step, sign):
        nv = max(v, lo) * step ** sign if scale == "log" else v + sign * step * (hi - lo)
        return float(np.clip(nv, lo, hi))

    # coordinate search with shrinking steps; a step that helps is repeated while it keeps helping
    steps = [(2.0, 0.2), (1.41, 0.1), (1.19, 0.05), (1.09, 0.025)]
    for log_step, lin_step in steps:
        try_policies()
        improved = False
        for name, lo, hi, scale in SEARCH_SPACE:
            step = log_step if scale == "log" else lin_step
            for sign in (1, -1):
                if n_evals >= max_evals:
                    break
                v = float(cfg[name])
                nv = move(v, name, lo, hi, scale, step, sign)
                if nv == v:
                    continue
                l, st = evaluate({**cfg, name: nv})
                if l >= best - 1e-9:
                    continue
                best, best_st, cfg = l, st, {**cfg, name: nv}
                improved = True
                for _ in range(4):  # keep going while it helps
                    v = nv
                    nv = move(v, name, lo, hi, scale, step, sign)
                    if nv == v or n_evals >= max_evals:
                        break
                    l, st = evaluate({**cfg, name: nv})
                    if l >= best - 1e-9:
                        break
                    best, best_st, cfg = l, st, {**cfg, name: nv}
                break
        if verbose:
            print(f"  fit: step {log_step:.2f} loss {best:.4f} after {n_evals} evals")
        if n_evals >= max_evals:
            break
        if not improved and log_step < 1.1:
            break

    keys = [k for k, *_ in LOSS_SPEC] + ["ball_present_rate", "slots_present_mean"]
    cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    cfg["_fit"] = {
        "loss_init": float(loss_init),
        "loss_final": float(best),
        "n_evals": int(n_evals),
        "base": base_name,
        "base_graded_frames": int(sum(l - f + 1 for _, f, l in base.half_blocks)),
        "seconds": round(time.time() - t0, 2),
        "stats": {k: [float(tgt.get(k, math.nan)), float(best_st.get(k, math.nan))] for k in keys},
    }
    return cfg


# parameters scaled by severity s, and the power of s applied (missing rates are handled apart)
_SEVERITY_POW = {
    "ball_gap_median_s": 0.5, "frag_median_s": -1.0, "dup_rate_per_min": 1.0,
    "ghost_rate_per_min": 1.0, "pos_jitter_sd_m": 1.0, "pos_wobble_sd_m": 1.0,
    "frag_bias_sd_m": 1.0, "ball_jitter_sd_m": 1.0, "ball_outlier_rate": 1.0, "swap_prob": 1.0,
    "team_flip_rate": 1.0,
}


def severity_grid(cfg: dict, k: int = 5, lo: float = 0.5, hi: float = 2.0) -> list[dict]:
    """k configs from mild (severity lo) to harsh (severity hi) around `cfg` (severity 1).

    Severity s multiplies the ball-missing rate and the player deficit (22 minus present objects),
    and scales fragment breaks, duplicates, ghosts, swaps, team flips and positional error. Log-spaced,
    so an odd k with lo * hi == 1 has `cfg` itself in the middle.
    """
    base = resolve_cfg(cfg)
    base = {a: b for a, b in base.items() if not str(a).startswith("_")}
    sev = [1.0] if k <= 1 else list(np.geomspace(lo, hi, k))
    out = []
    for s in sev:
        c = dict(base)
        miss = (1.0 - base["ball_present_rate"]) * s
        c["ball_present_rate"] = float(np.clip(1.0 - miss, 0.05, 1.0))
        deficit = (22.0 - base["players_present_mean"]) * s
        c["players_present_mean"] = float(np.clip(22.0 - deficit, 5.0, 30.0))
        for name, pw in _SEVERITY_POW.items():
            c[name] = float(base[name] * s ** pw)
        c["swap_prob"] = float(min(1.0, c["swap_prob"]))
        c["team_flip_rate"] = float(min(0.5, c["team_flip_rate"]))
        c["_severity"] = float(s)
        out.append(c)
    return out


__all__ = ["fit_noise_cfg", "severity_grid", "stats_loss", "crop_match", "LOSS_SPEC", "SEARCH_SPACE",
           "DEFAULT_CFG"]
