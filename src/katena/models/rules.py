"""M1: rule chain. Constants are fixed a priori and never tuned.

control detection -> team state with a dwell before a hand-over -> held through flight and ball gaps;
dead ball when the ball has been outside the pitch for 1 s or the whole pitch has been still for 3 s,
alive again once the ball is back in and players move.
"""

from __future__ import annotations

import numpy as np

CTRL_R = 2.5          # m: a player this close to the ball controls it ...
CTRL_MARGIN = 0.5     # ... if the nearest opponent is at least this much further away
HANDOVER_S = 1.0      # opponent must control this long before possession flips
OUT_S = 1.0           # ball out of the pitch this long -> dead
STILL_S = 3.0         # everything still this long -> dead
STILL_FRAC = 0.85
RESTART_FRAC = 0.6


def m1_predict(X0: np.ndarray, names: list, half: np.ndarray, fps: float = 5.0) -> np.ndarray:
    """int8[T] labels (0 H, 1 A, 2 D) from focal=home features of one match."""
    col = {n: X0[:, i] for i, n in enumerate(names)}
    d_f1, d_o1 = col["d_f1"], col["d_o1"]
    have = np.isfinite(d_f1) & np.isfinite(d_o1)
    with np.errstate(invalid="ignore"):
        ctrl = np.where(have & (d_f1 < CTRL_R) & (d_f1 + CTRL_MARGIN < d_o1), 0,
                        np.where(have & (d_o1 < CTRL_R) & (d_o1 + CTRL_MARGIN < d_f1), 1, -1))
        out_ball = np.nan_to_num(col["ball_oob_dist"], nan=1.0) < -0.3
    seen = np.isfinite(col["ball_oob_dist"])
    still = np.nan_to_num(col["frac_still_m2s"], nan=0.0)
    bsp = np.nan_to_num(col["ball_speed_mean1s"], nan=0.0)
    n_ho, n_out, n_still = int(HANDOVER_S * fps), int(OUT_S * fps), int(STILL_S * fps)
    out = np.empty(len(d_f1), dtype=np.int8)
    for h in np.unique(half):
        team, opp_run, dead, out_run, in_run, still_run = -1, 0, False, 0, 0, 0
        for i in np.flatnonzero(half == h):
            if dead:
                in_run = in_run + 1 if (seen[i] and not out_ball[i]) or not seen[i] else 0
                if in_run >= n_out and still[i] < RESTART_FRAC:
                    dead, team, opp_run, out_run, still_run = False, -1, 0, 0, 0
            else:
                out_run = out_run + 1 if out_ball[i] else 0
                still_run = still_run + 1 if (still[i] > STILL_FRAC and bsp[i] < 1.0) else 0
                if out_run >= n_out or still_run >= n_still:
                    dead, in_run = True, 0
            if not dead and ctrl[i] >= 0:
                if team < 0:
                    team = int(ctrl[i])
                elif ctrl[i] == team:
                    opp_run = 0
                else:
                    opp_run += 1
                    if opp_run >= n_ho:
                        team, opp_run = int(ctrl[i]), 0
            if dead:
                out[i] = 2
            elif team >= 0:
                out[i] = team
            else:
                out[i] = 0 if (have[i] and d_f1[i] <= d_o1[i]) else 1
    return out
