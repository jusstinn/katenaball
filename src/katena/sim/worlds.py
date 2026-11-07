"""Named degrader worlds for D2: ball loss tied to dead ball at several levels, and encounter identity
switches. New names only; ``mild``, ``target`` and ``severe`` (``katena.sim.degrade``) are unchanged.

Corpus caches are keyed by condition name, so a world's definition must never change once its features
are cached. Add a new name instead.

Sources, published or external only. Nothing here is fitted to the competition match.

* **Overall ball loss.** Every world keeps its severity's ``ball_missing``. For ``target`` that is 0.41,
  from DATA.md: the ball is present on 59.3 % of graded frames.
* **Coupling level kappa.** P(ball missing | dead) = r + kappa * (0.79 - r), where r is the severity's
  overall rate. Then kappa = 0 is independence, the "none" level and the existing worlds' neighbourhood:
  ``target`` itself measured 0.35 dead / 0.47 alive on IDSSE in the bake-off. kappa = 1 is the Metrica
  sample data: the ball is NaN on 0.79 of D frames and 0.015 of alive frames, pooled at 5 Hz over the two
  development games, Sample_Game_1 and Sample_Game_3 (bake-off Table 7). Alive frames take the rest of the
  overall rate, never less than 0.015. The lag from the stoppage to the loss is exponential with mean
  1.4 s and median 1.0 s, and the ball returns at the restart (lead 0 on 96 % of spans). These come from
  ``scripts/d2_calibrate.py`` on the same two games. The locked Sample_Game_2 is not used.
* **Identity switches.** The 46 slots (23 per team) and the mean tracked-object count stay as in the
  severity's config (DATA.md: 46 slots, 20.5 objects), because swaps and cuts do not change who is
  present. No swap rate is published, and none can be measured on the external corpus, whose identities
  are clean or hand-corrected. These values are therefore assumptions. At ``target`` level, 0.1 per
  team-mate encounter under 1 m and a crowding coefficient of 0.3 per player within 1.5 m are the
  defaults of the D2 simulator on ``david/foundation`` (``katena.noise.sim``). Opponent swaps at 0.05,
  held for 3 s, are our own guess. ``mild`` uses half of each, ``severe`` twice.

* **Smooth positions** (amendment A1, after the D5 audit A20). The degrader's i.i.d. jitter is about 10x
  rougher than real automatic tracking. On D5's statistic (5 Hz second difference), the SkillCorner
  development matches give 0.046 m per slot and 0.21 m for the ball, against 0.92 m and 1.80 m for
  ``target``. ``*_smooth`` sets player jitter 0, which already gives 0.050 m on clean IDSSE because of
  the ghosts, and ball jitter 0.058 m, which gives the SkillCorner ball value. Recorded by
  ``scripts/d2_calibrate.py --jitter`` in ``jitter_external.json``.

Coupled worlds need dead labels, so they exist for IDSSE and Metrica only. SkillCorner has no D, so it
would only be topped up, and it is not built in those worlds. The D5 audit (A19) adds that SkillCorner's
``target`` world has 15.3 tracked objects, not 20.5. The degrader removes players on top of SkillCorner's
own broadcast gaps, and these worlds leave that as it is.
"""

from __future__ import annotations

import dataclasses

from katena.sim import degrade as dg

METRICA_DEAD_MISS = 0.79       # P(ball NaN | D), Metrica development games, 5 Hz, pooled
METRICA_ALIVE_MISS = 0.015     # P(ball NaN | H or A), same games
METRICA_LAG_S = 1.4            # mean stoppage-to-loss delay, same games (exponential fit)
COUPLING_LEVELS = {"cpl50": 0.5, "cpl100": 1.0}
ID_LEVELS = {                  # (frag_cong_coef, swap_prob, xswap_prob) per severity
    "mild": (0.15, 0.05, 0.025),
    "target": (0.3, 0.1, 0.05),
    "severe": (0.6, 0.2, 0.1),
}
DEAD_LABEL_DATASETS = ("idsse", "metrica")
SMOOTH_SIGMA_PLAYER = 0.0      # m; SkillCorner-matched (jitter_external.json)
SMOOTH_SIGMA_BALL = 0.058      # m; SkillCorner-matched (jitter_external.json)


def coupled(cfg: dg.DegradeCfg, kappa: float, tag: str) -> dg.DegradeCfg:
    """``cfg`` with dead-span ball loss at coupling level ``kappa`` (0 independent, 1 Metrica)."""
    p_dead = cfg.ball_missing + kappa * (METRICA_DEAD_MISS - cfg.ball_missing)
    return dataclasses.replace(cfg, name=f"{cfg.name}_{tag}", dead_miss=float(p_dead),
                               dead_lag_s=METRICA_LAG_S, alive_floor=METRICA_ALIVE_MISS)


def with_ids(cfg: dg.DegradeCfg, severity: str) -> dg.DegradeCfg:
    """``cfg`` with encounter identity switches at the level of ``severity``."""
    coef, same, opp = ID_LEVELS[severity]
    return dataclasses.replace(cfg, name=f"{cfg.name}_ids", frag_cong_coef=coef, swap_prob=same,
                               xswap_prob=opp, swap_dist_m=1.0, occl_radius_m=1.5, xswap_hold_s=3.0)


def with_smooth(cfg: dg.DegradeCfg) -> dg.DegradeCfg:
    """``cfg`` with positional jitter at the level of real automatic tracking."""
    return dataclasses.replace(cfg, name=f"{cfg.name}_smooth", sigma_player=SMOOTH_SIGMA_PLAYER,
                               sigma_ball=SMOOTH_SIGMA_BALL)


def _build() -> dict:
    out = {}
    for sev, base in dg.CONFIGS.items():
        for tag, kappa in COUPLING_LEVELS.items():
            w = coupled(base, kappa, tag)
            out[w.name] = w
        for w in (with_ids(base, sev), with_smooth(base)):
            out[w.name] = w
    combo = with_ids(out["target_cpl100"], "target")       # evaluation only: both effects at once
    out[combo.name] = combo
    return out


WORLDS: dict[str, dg.DegradeCfg] = _build()

# Training mixes under test: each severity's training rows are split evenly over these worlds. The current
# mix is the first entry of each list alone. ``d2`` was registered first; ``d2j`` adds the smooth worlds
# (amendment A1).
TRAIN_MIXES = {
    "d2": {"dead_labels": {s: [s, f"{s}_cpl50", f"{s}_cpl100", f"{s}_ids"] for s in dg.CONFIGS},
           "no_dead_labels": {s: [s, f"{s}_ids"] for s in dg.CONFIGS}},
    "d2j": {"dead_labels": {s: [s, f"{s}_cpl50", f"{s}_cpl100", f"{s}_ids", f"{s}_smooth"] for s in dg.CONFIGS},
            "no_dead_labels": {s: [s, f"{s}_ids", f"{s}_smooth"] for s in dg.CONFIGS}},
}
TRAIN_MIX = TRAIN_MIXES["d2"]
# Worlds every candidate is scored in and gated on. Coupled worlds run on IDSSE and Metrica only.
# target_smooth was added by amendment A1, before any gate result.
EVAL_WORLDS = ("mild", "target", "severe", "target_cpl50", "target_cpl100", "target_ids", "target_cpl100_ids",
               "target_smooth")


def needs_dead_labels(name: str) -> bool:
    return "_cpl" in name


def applies(name: str, dataset: str) -> bool:
    """Whether world ``name`` is built for a match of ``dataset``."""
    return dataset in DEAD_LABEL_DATASETS or not needs_dead_labels(name)


def config(name: str) -> dg.DegradeCfg:
    """Any condition name: the existing ``mild``/``target``/``severe`` or a D2 world."""
    return dg.CONFIGS[name] if name in dg.CONFIGS else WORLDS[name]
