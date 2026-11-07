"""D3 feature store: per-frame features for the H / A / D possession model.

    from katena.feats import featurize, swap_teams
    f = featurize(m)                       # one row per graded frame
    f = featurize(m, {"stride": 5})        # every 5th frame, for training
    f2 = swap_teams(f)                     # home <-> away augmentation (labels: swap H and A)

(The package is `katena.feats`, not `katena.features`: the repo's firewall rules deny reads of any
`features/` directory, which is where computed feature exports live.)

Frame of reference
------------------
Positions are rotated per half by `m.attack_dir` so that the HOME team always attacks +x (both x
and y are multiplied by the direction, a 180-degree rotation, so handedness is preserved). Paired
columns `home_*` / `away_*` describe each team in that same frame; `swap_teams` exchanges them and
negates odd columns (signed positions and velocities, home-minus-away differences). Slots are never
treated as people: every team feature is a per-frame statistic over the present slots of that team,
after compaction and de-duplication (same-team slots closer than `dedup_m` count once). Per-slot
history is used only for a few-frame velocity, with a jump guard.

Columns (v1, default cfg: 93 base + 98 temporal context = 191)
---------------------------------------------------------------
ball (21, uses the ball)
    ball_present, ball_since / ball_until / ball_gap_len (frames, capped at gap_cap_s),
    ball_interp (frame filled by short-gap interpolation), ball_known_i (observed or interpolated);
    ball_x, ball_absy, {home,away}_ballgoal_dist (to the goal that team attacks), ball_out_x,
    ball_out_y (signed distance beyond goal line / touchline), ball_edge_dist; ball_x_i,
    ball_absy_i (interpolated); ball_vx, ball_speed, ball_accel, ball_turn (radians),
    ball_straight (displacement / path), ball_inflight (fast or straight-and-moving)
ball_rel (20, players against the ball)
    per team: nearest_dist, second_dist, n2, n5 (counts within 2 / 5 m), relspeed (nearest
    player's velocity minus the ball's); nearest_margin = away - home nearest distance;
    interpolated versions nearest_dist_i, n5_i, nearest_margin_i, ball_dx_i (ball x minus team
    centroid x); nearest_margin_ffill / _bfill (last / next observed margin, held over gaps)
press (4, needs a known or last-known ball)
    {home,away}_pressers (players within press_r of the opponent's ball-nearest player),
    {home,away}_carrier_space (that team's ball-nearest player to the nearest opponent)
shape (38, ball-free)
    per team: n, cx, cy, sx, sy, dcent (mean distance to centroid), gk_x (deepest), line_x
    (second deepest = defensive line), top_x (highest), length, width (outfield y-range),
    area (length x width), cvx (mean player velocity along the attack axis), mspd, nsprint
    (> sprint_speed), nopp_half, box_att (own players in the box it attacks), box_n (all players
    in that box), mark_dist (mean outfield distance to the nearest opponent: pressing intensity)
relative (6, ball-free)
    adv_diff = home_cx + away_cx (who is higher up the pitch, in each team's own direction),
    cx_gap, line_diff, width_diff, length_diff, dcent_diff
still (4, ball-free)
    all_mspd, frac_static (< static_speed), all_dcent (gathering), n_present
temporal context (98)
    `<col>__mean{0.5s,2s,6s}` of 22 key signals, `__min` of 4 and `__max` of 2 at the same
    windows, and `<col>__step{1s,4s}` (mean over the next w seconds minus the previous w) of 7.
    Centred, offline, NaN-aware, clipped to the half. Lists in `spec.CONTEXT_*`.
extras
    `x_<name>` per-frame columns from other modules, sampled at t_frame. Treated as even under a
    swap unless named `*_signed`; `x_home_*` / `x_away_*` are swapped as pairs.

NaN policy (LightGBM routes NaN natively, so nothing is imputed)
----------------------------------------------------------------
NaN means undefined or unobserved: ball unknown (observed-ball columns), ball neither observed nor
interpolated (`_i` columns), no ball seen anywhere in the half (press), too few players of a team,
direction undefined at low speed (ball_turn), or an empty window. Never-NaN columns: ball_present,
ball_since, ball_until, ball_gap_len, ball_interp, ball_known_i, {home,away}_n, nsprint,
nopp_half, box_att, box_n, n_present, and their temporal aggregates. Timers report "not seen in
this half" as the cap. No column is ever +-inf. `spec.nan_able(name)` is the machine-readable form.
"""
from . import spec
from .core import DEFAULT_CFG, FEATURE_VERSION, Feats, featurize, swap_teams
from .spec import ball_free

__all__ = ["featurize", "swap_teams", "Feats", "DEFAULT_CFG", "FEATURE_VERSION", "ball_free",
           "spec"]
