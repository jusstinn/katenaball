"""Column registry: names, team-swap parity, feature group, and NaN policy.

Naming grammar (everything `swap_teams` and the group / NaN helpers need is in the name):

    [home_|away_]<core>[__<op><window>]       base features and temporal context
    x_[home_|away_]<name>                     extras from other modules

- `home_<core>` and `away_<core>` are a pair. A team swap exchanges them.
- A core is *odd* if it changes sign under a team swap (signed positions / velocities in the
  attack-normalised frame, home-minus-away differences). Odd cores are listed in `CORES` below.
- Temporal ops: `mean`, `min`, `max`, `step`. Under a swap an odd core's `min` becomes minus the
  partner's `max` and vice versa; `mean` and `step` keep their op and just take the sign.
- Extras are even unless their name ends in `_signed`; `x_home_*` / `x_away_*` are paired.

NaN policy: NaN means "undefined or not observed" (no ball known, no players of that team present,
direction undefined at low speed, empty window). LightGBM routes NaN natively, so there is no
imputation and no sentinel inside NaN-able columns. Columns in the never-NaN set (presence flags,
gap timers, and counts over present players) are always finite. Gap timers are capped at
`gap_cap_s`: "never seen in this half" is reported as the cap, and `ball_present` / `ball_known_i`
disambiguate. No column is ever +-inf.
"""
from __future__ import annotations

import re

import numpy as np

# core -> (paired, odd, group, nan_able)
# groups: ball (ball state and kinematics), ball_rel (players vs ball), press (pressing around the
# ball-nearest players; needs a known or last-known ball), shape (team shape and kinematics, no
# ball), relative (team-vs-team shape, no ball), still (stillness and set-piece layout, no ball).
CORES: dict[str, tuple[bool, bool, str, bool]] = {
    # ball presence and gap timing
    "ball_present": (False, False, "ball", False),
    "ball_since": (False, False, "ball", False),
    "ball_until": (False, False, "ball", False),
    "ball_gap_len": (False, False, "ball", False),
    "ball_interp": (False, False, "ball", False),
    "ball_known_i": (False, False, "ball", False),
    # observed ball position (attack-normalised: home attacks +x)
    "ball_x": (False, True, "ball", True),
    "ball_absy": (False, False, "ball", True),
    "ballgoal_dist": (True, False, "ball", True),
    "ball_out_x": (False, False, "ball", True),
    "ball_out_y": (False, False, "ball", True),
    "ball_edge_dist": (False, False, "ball", True),
    # interpolated ball position (short gaps filled)
    "ball_x_i": (False, True, "ball", True),
    "ball_absy_i": (False, False, "ball", True),
    # ball kinematics (observed, gap-aware)
    "ball_vx": (False, True, "ball", True),
    "ball_speed": (False, False, "ball", True),
    "ball_accel": (False, False, "ball", True),
    "ball_turn": (False, False, "ball", True),
    "ball_straight": (False, False, "ball", True),
    "ball_inflight": (False, False, "ball", True),
    # players vs observed ball
    "nearest_dist": (True, False, "ball_rel", True),
    "second_dist": (True, False, "ball_rel", True),
    "n2": (True, False, "ball_rel", True),
    "n5": (True, False, "ball_rel", True),
    "relspeed": (True, False, "ball_rel", True),
    "nearest_margin": (False, True, "ball_rel", True),
    # players vs interpolated ball
    "nearest_dist_i": (True, False, "ball_rel", True),
    "n5_i": (True, False, "ball_rel", True),
    "nearest_margin_i": (False, True, "ball_rel", True),
    "ball_dx_i": (True, True, "ball_rel", True),
    # observed margin held over gaps (offline: forward and backward fill within the half)
    "nearest_margin_ffill": (False, True, "ball_rel", True),
    "nearest_margin_bfill": (False, True, "ball_rel", True),
    # pressing around each team's player nearest the interpolated / last-known ball
    "pressers": (True, False, "press", True),
    "carrier_space": (True, False, "press", True),
    # team shape and kinematics (ball-free)
    "n": (True, False, "shape", False),
    "cx": (True, True, "shape", True),
    "cy": (True, True, "shape", True),
    "sx": (True, False, "shape", True),
    "sy": (True, False, "shape", True),
    "dcent": (True, False, "shape", True),
    "gk_x": (True, True, "shape", True),
    "line_x": (True, True, "shape", True),
    "top_x": (True, True, "shape", True),
    "length": (True, False, "shape", True),
    "width": (True, False, "shape", True),
    "area": (True, False, "shape", True),
    "cvx": (True, True, "shape", True),
    "mspd": (True, False, "shape", True),
    "nsprint": (True, False, "shape", False),
    "nopp_half": (True, False, "shape", False),
    "box_att": (True, False, "shape", False),
    "box_n": (True, False, "shape", False),
    "mark_dist": (True, False, "shape", True),
    # team vs team (ball-free)
    "adv_diff": (False, True, "relative", True),
    "cx_gap": (False, False, "relative", True),
    "line_diff": (False, True, "relative", True),
    "width_diff": (False, True, "relative", True),
    "length_diff": (False, True, "relative", True),
    "dcent_diff": (False, True, "relative", True),
    # stillness and layout (ball-free)
    "all_mspd": (False, False, "still", True),
    "frac_static": (False, False, "still", True),
    "all_dcent": (False, False, "still", True),
    "n_present": (False, False, "still", False),
}

BALL_FREE_GROUPS = frozenset({"shape", "relative", "still"})
GROUPS = ("ball", "ball_rel", "press", "shape", "relative", "still", "extra")

# Temporal context: which signals get which centred aggregates. Windows / steps come from cfg.
CONTEXT_MEAN = (
    "ball_present", "ball_speed", "ball_inflight", "ball_x_i", "nearest_margin_i",
    "home_nearest_dist_i", "away_nearest_dist_i", "home_pressers", "away_pressers",
    "home_carrier_space", "away_carrier_space", "adv_diff", "line_diff", "home_cvx", "away_cvx",
    "home_mark_dist", "away_mark_dist", "all_mspd", "frac_static", "home_cx", "away_cx",
    "width_diff",
)
CONTEXT_MIN = ("home_nearest_dist_i", "away_nearest_dist_i", "nearest_margin_i", "all_mspd")
CONTEXT_MAX = ("ball_speed", "nearest_margin_i")
CONTEXT_STEP = ("adv_diff", "home_cx", "away_cx", "all_mspd", "frac_static", "ball_x_i",
                "nearest_margin_i")

_OPS = ("mean", "min", "max", "step")
_SUFFIX = re.compile(r"^(?P<base>.+?)__(?P<op>mean|min|max|step)(?P<win>[0-9.]+s)$")


def win_tag(seconds: float) -> str:
    """0.5 -> '0.5s', 2.0 -> '2s'."""
    return f"{seconds:g}s"


def base_names() -> list[str]:
    """Per-frame base columns, in output order."""
    out = []
    for core, (paired, *_rest) in CORES.items():
        out += [f"home_{core}", f"away_{core}"] if paired else [core]
    return out


def context_names(windows_s, step_s) -> list[str]:
    out = []
    for w in windows_s:
        t = win_tag(w)
        out += [f"{b}__mean{t}" for b in CONTEXT_MEAN]
        out += [f"{b}__min{t}" for b in CONTEXT_MIN]
        out += [f"{b}__max{t}" for b in CONTEXT_MAX]
    for w in step_s:
        out += [f"{b}__step{win_tag(w)}" for b in CONTEXT_STEP]
    return out


def parse(name: str) -> dict:
    """Split a column name into its parts: side ('home', 'away' or None), core, op, win, extra."""
    extra = name.startswith("x_")
    rest = name[2:] if extra else name
    op = win = None
    m = _SUFFIX.match(rest)
    if m and not extra:
        rest, op, win = m["base"], m["op"], m["win"]
    side = None
    for s in ("home", "away"):
        if rest.startswith(s + "_"):
            side, rest = s, rest[len(s) + 1:]
            break
    return {"extra": extra, "side": side, "core": rest, "op": op, "win": win}


def _compose(p: dict) -> str:
    s = (p["side"] + "_" if p["side"] else "") + p["core"]
    if p["op"]:
        s += f"__{p['op']}{p['win']}"
    return ("x_" + s) if p["extra"] else s


def is_odd(name: str) -> bool:
    p = parse(name)
    if p["extra"]:
        return p["core"].endswith("_signed")
    if p["core"] not in CORES:
        raise KeyError(f"unknown feature core in {name!r}")
    return CORES[p["core"]][1]


def group_of(name: str) -> str:
    p = parse(name)
    if p["extra"]:
        return "extra"
    return CORES[p["core"]][2]


def nan_able(name: str) -> bool:
    """True if the column may hold NaN under the documented policy."""
    p = parse(name)
    if p["extra"]:
        return True
    return CORES[p["core"]][3]


def swap_partner(name: str) -> tuple[str, float]:
    """(column whose negated / plain value this column takes after a team swap, sign)."""
    p = dict(parse(name))
    odd = is_odd(name)
    if p["side"] is not None:
        p["side"] = "away" if p["side"] == "home" else "home"
    if odd and p["op"] in ("min", "max"):
        p["op"] = "max" if p["op"] == "min" else "min"
    return _compose(p), (-1.0 if odd else 1.0)


def swap_spec(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """perm int64[F], sign float32[F] such that X_swapped = X[:, perm] * sign."""
    pos = {n: i for i, n in enumerate(names)}
    perm = np.empty(len(names), dtype=np.int64)
    sign = np.empty(len(names), dtype=np.float32)
    for i, n in enumerate(names):
        partner, s = swap_partner(n)
        if partner not in pos:
            raise KeyError(f"{n!r} has no swap partner {partner!r} in this feature set")
        perm[i] = pos[partner]
        sign[i] = s
    return perm, sign


def ball_free(names: list[str]) -> list[str]:
    """Columns that never read the ball (team shape, team-vs-team, stillness)."""
    return [n for n in names if group_of(n) in BALL_FREE_GROUPS]
