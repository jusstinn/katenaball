# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-07

### D5: audit and reconciliation (David's side, `david/d5-audit`)

**M2 reproduces on the Windows machine: 0.5861 on 28 folds**, every per-dataset, per-severity and fixed-tau mean
equal to `bakeoff.md` at its precision. Shared base cache: `D:/katena/work-base` (READY, code `8537084`). No public
window was looked at.

**Audit: no leakage that moves a reported number.** Tau is chosen from folds whose models and priors saw the held-out
match, but all 28 folds chose tau = 2, so the effect is nil. Full findings: [audit/d5-bakeoff-audit.md](audit/d5-bakeoff-audit.md).

| Decision | Why | Owner |
|---|---|---|
| Fixed: block-edge X on graded frames, locked bench pinned by id, CRLF-proof corpus digest, no data row from `probe_schema.py` | Bugs, each with a test. Score-neutral on the dev folds (M2 re-scored 0.586133 before and after) | done |
| Added `scripts/check_target_layout.py` | The ATD reader's slot-order assumption is invisible to the baseline check | J1 to run |
| Recommend gates fix tau = 2 a priori | Removes the second-order leak and five decodes per fold; the plateau is known | coordinator |
| Recommend gates report the 8 IDSSE + Metrica folds beside the 28 (M2 0.569) | 20 of 28 folds are SkillCorner, which has no dead-ball label | coordinator |
| Metrica dead-ball labels run long (P(ball missing given D) 0.88 vs 0.99 for David's derivation on Game 3) | Teaches "dead" on live frames to the alive head (25% of its rows) | D4, through the gate |
| Submission written with LF line ends; final model trained and hashed on one reference machine | Byte identity across macOS and Windows | J2, J3 |

**Reconcile ([reconcile.md](reconcile.md)).** Ported the domain-gap alarm (`katena.diag.domain_gap`,
`scripts/check_domain_gap.py`; on the target it prints one alarm bit) and a brute-force Viterbi test. Not ported: `fit_noise_cfg`
and `noise_fit_target.py`, which fit the degrader to the target. Everything else is superseded by main or is a candidate
for its owner (D1 features, D2 identity-switch noise, D4 harmonisation and control evidence).

**Domain-gap alarm on external data:** real SkillCorner and Metrica tracking are separated from every degraded world
with AUC 1.00, mainly by positional jitter (median second difference 0.04-0.05 m against 0.47-1.38 m). A D2 candidate:
calibrate jitter on SkillCorner, through the gate. Not run on the target.

**Final training (D5 second half): blocked.** `origin/main` has no J2 training script yet.

### D2: degrader realism (David + Claude)

**Registered before any gate result.** Branch `david/d2-degrader`. External development matches only.
No `data/`, no `evaluation/gt/`, no locked bench.

**New worlds** in `katena.sim.worlds`. They have new names only. The existing `mild`, `target`, `severe`
and `target_dead*` configs give byte-identical output for the same seed, which
`tests/test_d2_degrader.py` checks against a frozen copy of the 47a3f4b degrader. The new options in
`katena.sim.degrade` default to off.

| World | What changes | Source |
|---|---|---|
| `{mild,target,severe}_cpl50`, `_cpl100` | P(ball missing \| dead) = r + kappa (0.79 - r), kappa 0.5 or 1. r is the severity's overall rate, which stays put (`target`: 0.41, DATA.md). Alive frames take the rest, at least 0.015. The ball is lost an exponential lag after the stoppage (mean 1.4 s) and stays lost until the restart. | Metrica development games 1 and 3, pooled at 5 Hz: dead 0.79, alive 0.015. Lag median 1.0 s, mean 1.4 s. The ball is back at the restart on 96% of spans. IDSSE and Metrica only, because SkillCorner has no D. |
| `{mild,target,severe}_ids` | At `target` level: fragment cuts on a crowding clock (0.3 per player within 1.5 m), team-mate slot swaps 0.1 and opponent swaps 0.05 (held 3 s) per encounter under 1 m. Half at `mild`, double at `severe`. Presence, the 46 slots and the object count are unchanged. | Slots and objects: DATA.md. Swap rates are not published and cannot be measured externally. Crowding and team-mate rates are the defaults of David's `katena.noise.sim` (`david/foundation`). The opponent rate is ours. |
| `target_cpl100_ids` | Both of the above. Evaluation only. | As above |

**Candidates.** `cur` is the bake-off pool: M2 trained on `mild`/`target`/`severe`, stride 4. `d2` uses the
same rows, focal orientations and labels, but each severity's rows are split evenly, interleaved, over
{s, s_cpl50, s_cpl100, s_ids} for IDSSE and Metrica, and over {s, s_ids} for SkillCorner. LightGBM
parameters, Viterbi, T = 2 and the tau rule are unchanged.

**Decision worlds.** `mild`, `target`, `severe` and `target_ids` on 28 folds. `target_cpl50`,
`target_cpl100` and `target_cpl100_ids` on the 8 IDSSE + Metrica folds. The mild and severe coupled and
`_ids` worlds are secondary: reported, not decisive.

**Rule** (PLAN bet #2). Keep the d2 mix if and only if no decision world gets worse. A world gets worse
when mean(d2 - cur) < 0 and the upper bound of the paired 95% bootstrap interval is below 0
(`katena.eval.bootstrap`: possessions resampled, 2000 draws). For every world we report folds won, mean
difference and CI, and whether it passes the adoption rule (wins >= 70% and CI > 0). We screen on the 8
folds first, and the 28-fold gate decides.

**Official-baseline check.** We score the official baseline algorithm in every world with all four
published windows, on the IDSSE and Metrica development matches, and set it beside the published
full-match scores (smooth50: macroF1 0.6724, spellF1 0.2433). This is a report only. The kappa levels
above were fixed before it was computed, and it chooses no world.

**Amendment A1**, also before any gate result. It follows the D5 audit (`docs/audit/d5-bakeoff-audit.md`
on `david/d5-audit`: A2, A18, A20), relayed by the coordinator.

- **Tau.** Tau is fixed at 2 for every variant. Per-fold tau selection would use folds whose models saw the
  held-out match.
- **Folds.** The 8 IDSSE + Metrica folds are reported beside the 28. A decision world gets worse if it is
  significantly worse on either set. SkillCorner, which has no D, cannot show a dead-ball change.
- **Smooth worlds.** `{mild,target,severe}_smooth` are the severity's config with player jitter 0 and ball
  jitter 0.058 m. On D5's 5 Hz second-difference statistic this gives the SkillCorner development
  matches' 0.046 m per slot and 0.21 m for the ball, against 0.92 m and 1.80 m in `target`
  (`scripts/d2_calibrate.py --jitter`, `jitter_external.json`). `target_smooth` is a decision world on
  28 folds.
- **Second candidate.** `d2j` is the d2 mix with s_smooth added to each severity's list (IDSSE and
  Metrica 5 worlds, SkillCorner 3). The same rule applies.
- **Alarm.** There is no D2 alarm. D5 ported David's domain-gap alarm (`katena.diag.domain_gap`), so the
  D2 one was removed unused.

**Result 1: the official-baseline gap.** This is a report only, and it chooses nothing. The kappa levels
were fixed above before it was run. It is the starter algorithm (nearest player, ball missing = D,
majority window) at 5 Hz, scored at T = 2 on the 6 IDSSE development matches
(`scripts/d2_worlds.py m0`). Cells are macroF1 / spellF1.

| World | P(miss \| dead) / P(miss \| alive) | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|---|
| `target` (independent) | 0.35 / 0.47 | 0.369 / 0.025 | 0.394 / 0.112 | 0.389 / 0.173 | 0.366 / 0.206 |
| `target_dead15` (bake-off) | 0.64 / 0.22 | 0.577 / 0.042 | 0.522 / 0.156 | 0.463 / 0.199 | 0.378 / 0.201 |
| `target_cpl50` | 0.60 / 0.25 | 0.552 / 0.046 | 0.599 / 0.184 | 0.614 / 0.228 | 0.582 / 0.212 |
| `target_cpl100` (Metrica-like) | 0.79 / 0.09 | 0.686 / 0.058 | 0.705 / 0.226 | 0.699 / 0.259 | 0.645 / 0.229 |
| `target_ids` | 0.33 / 0.48 | 0.355 / 0.026 | 0.380 / 0.112 | 0.378 / 0.165 | 0.356 / 0.184 |
| `target_smooth` | 0.34 / 0.47 | 0.371 / 0.039 | 0.397 / 0.127 | 0.392 / 0.178 | 0.363 / 0.211 |
| **Published, real target** | 0.407 overall | 0.653 / 0.049 | 0.675 / 0.200 | 0.672 / 0.243 | 0.636 / 0.224 |

- **Coupling closes the gap; identity and jitter do not.** The independent world misses the published
  smooth50 macroF1 by 0.28. `target_cpl50` misses it by 0.06 and `target_cpl100` by 0.03, the other way.
- **The Metrica-like world matches the whole fingerprint.** `target_cpl100` comes within 0.035 of all
  eight published numbers, always slightly above them. It also gets the published order of the windows:
  macroF1 peaks at smooth25 and spellF1 at smooth50. The independent world peaks spellF1 at smooth125.
- **Caveats.** This is consistent with the real ATD dropping the ball through stoppages, but it is not a
  measurement.
  - It is not a label-convention effect. MacroF1 barely moves with T: at smooth50 the independent world
    gives 0.386 to 0.389 and `target_cpl100` 0.694 to 0.699, for evaluation T = 0, 2 and 5. SpellF1 does
    move with T.
  - It still depends on the target's dead share, and on 5 Hz against 25 Hz.
  - On Metrica's two development games every world already looks like this (smooth50 about 0.72 / 0.30),
    because their own ball loss is coupled.
- `target` and `target_dead15` reproduce bake-off Table 4 exactly.

**The worlds as realised**, means over development matches, from each feature file's degrader record.

- **Overall ball loss.** It stays at 0.41 in every `target` world.
- **Coupling.** On IDSSE, P(miss | dead) / P(miss | alive) is 0.60 / 0.25 in `target_cpl50` and
  0.79 / 0.09 in `target_cpl100`. Metrica is already coupled natively (0.81 to 0.84 dead) in every world,
  because native loss is never reduced. `mild_cpl50` and `mild_cpl100` are the same world: a 20% overall
  loss caps the dead share at 0.42 / 0.015.
- **Identity.** `target_ids` has about 50 team-mate and 60 opponent swaps per IDSSE match, with 10% more
  fragments and the same object count (20.3) and slots (46).
- **SkillCorner.** It keeps 15.3 objects (audit A19).

**Result 2: screen, the 8 IDSSE + Metrica folds.** These are LOMO folds, Viterbi at tau 2, scored with the
official `evaluate`. CI is the paired 95% bootstrap over possessions.

- **Sanity checks.** `cur` reproduces the base cache's M2 exactly: its out-of-fold probabilities are
  byte-identical, and it scores target 0.5689 (D5's 0.569), mild 0.6122 and severe 0.5149.
- **Fold sets.** Under A1 these 8 folds are final for both fold sets. Tau is fixed, and the priors leave
  out only the held-out match, so the 28-fold run gives the same scores on them.

| Decision world | cur | d2 - cur [CI] | wins | d2j - cur [CI] | wins |
|---|---|---|---|---|---|
| `mild` | 0.6122 | +0.0007 [-0.0052, +0.0067] | 5/8 | +0.0031 [-0.0029, +0.0092] | 4/8 |
| `target` | 0.5689 | +0.0004 [-0.0061, +0.0067] | 2/8 | -0.0025 [-0.0084, +0.0034] | 4/8 |
| `severe` | 0.5149 | -0.0002 [-0.0072, +0.0069] | 4/8 | -0.0019 [-0.0093, +0.0050] | 3/8 |
| `target_ids` | 0.5791 | +0.0009 [-0.0059, +0.0075] | 5/8 | -0.0013 [-0.0083, +0.0054] | 3/8 |
| `target_smooth` | 0.5679 | +0.0032 [-0.0040, +0.0105] | 4/8 | **+0.0078 [+0.0005, +0.0152]** | 6/8 |
| `target_cpl50` | 0.5873 | +0.0086 [+0.0017, +0.0155] | 5/8 | **+0.0160 [+0.0094, +0.0234]** | 8/8 |
| `target_cpl100` | 0.5951 | **+0.0097 [+0.0032, +0.0163]** | 6/8 | +0.0123 [+0.0055, +0.0195] | 5/8 |
| `target_cpl100_ids` | 0.5900 | **+0.0133 [+0.0070, +0.0195]** | 7/8 | **+0.0140 [+0.0076, +0.0205]** | 7/8 |

Bold marks a world that passes the adoption rule (wins >= 70% and CI > 0). No decision world is worse for
either candidate, so both go on to 28 folds.

- **Secondary worlds.** Both candidates gain in every coupled secondary world, by +0.008 to +0.020.
- **The jitter gap.** `d2j` gains most where positions are smooth: `severe_smooth` +0.024 (8/8) and
  `mild_smooth` +0.011 (7/8). `cur` itself scores lower on smooth input than on rough
  (`severe_smooth` 0.504 against `severe` 0.515). The models depend on the jitter they were trained on,
  and the D5 audit found real tracking far smoother.
- **The bake-off's dead-coupled worlds.** `target_dead3` and `target_dead15` (IDSSE, 6 folds) do not
  move.

**Amendment A2**, written after the screen and before any 28-fold result. If both candidates are kept, we
recommend the one with the larger worst-case mean difference over the decision worlds. Each world counts
on the 28 folds where it runs there, otherwise on the 8. This follows PLAN.md: "prefer changes that help
in every world". The 8-fold numbers above were known when this was written.

**Result 3: the gate, 28 folds.** `cur` scores target 0.5861, equal to M2. The coupled worlds have no
SkillCorner folds, so their 8-fold rows above stand.

| Decision world (28 folds) | cur | d2 - cur [CI] | wins | d2j - cur [CI] | wins |
|---|---|---|---|---|---|
| `mild` | 0.6116 | -0.0001 [-0.0035, +0.0032] | 14/28 | +0.0024 [-0.0010, +0.0059] | 15/28 |
| `target` | 0.5861 | +0.0007 [-0.0028, +0.0042] | 13/28 | +0.0019 [-0.0014, +0.0051] | 15/28 |
| `severe` | 0.5475 | +0.0031 [-0.0008, +0.0071] | 19/28 | +0.0036 [-0.0001, +0.0076] | 18/28 |
| `target_ids` | 0.5844 | +0.0016 [-0.0022, +0.0050] | 17/28 | +0.0022 [-0.0015, +0.0056] | 15/28 |
| `target_smooth` | 0.5796 | +0.0010 [-0.0029, +0.0050] | 15/28 | +0.0055 [+0.0014, +0.0097] | 19/28 |

**Decision.** Both mixes are kept: no decision world is worse, on the 28 folds or on the 8.

- **Recommendation: `d2j`** (amendment A2). Its worst decision world is +0.0019 (`target`), against
  -0.0001 (`mild`) for `d2`. `d2j` is ahead of `d2` in every decision world.
- **What it buys.** Coupled worlds gain +0.012 to +0.016 (CI > 0). `target_smooth` gains +0.0055, and the
  secondary `severe_smooth` +0.013 (22/28).
- **What it does not buy.** The headline independent `target` gains only +0.002, which is not
  significant.
- **The real cost: IDSSE.** By dataset, `d2j` costs IDSSE -0.006 in `target` (2/6 wins) and -0.011 in
  `severe` (1/6). IDSSE is the only dataset whose ball is never missing natively, so it is the one where
  the independent world is pure. The pooled rule does not see this, because Metrica (+0.008, +0.025),
  whose own loss is already coupled, and SkillCorner (+0.004, +0.006) gain. On IDSSE, `d2j` gains in the
  smooth world (+0.005) and the coupled worlds (+0.007 to +0.016).
- **So the mix is a bet** that the real ATD's ball loss is tied to stoppages and that its positions are
  smooth. The official-baseline fingerprint (Result 1) and the D5 jitter audit both point that way.
  Neither was used to choose.
- **Next for J2.** The final model should train on the `d2j` mix:
  - `katena.sim.worlds.TRAIN_MIXES["d2j"]`, built with `katena.eval.d2.build_ref` and `build_pool`.
  - The D2 worlds are built for development matches only so far, so the final 30-match training must
    also build them for the two benches.
  - `katena.eval.corpus` and every cached condition are unchanged.
- **Calibration is pooled, not left out per fold (review note).** The Metrica coupling constants
  come from Sample_Game_1 and Sample_Game_3 and the smooth-jitter scale from the 20 SkillCorner
  matches, all development folds. So a held-out match contributes to a few pooled scalars that
  define its own fold's training worlds. That is not strict leave-one-match-out; with a handful of
  pooled scalars the effect is probably small. Recalibrate per fold before relying on the coupled-world
  gains.
- **Not done.**
  - The D2 worlds were not added to D5's domain-gap alarm.
  - A 25 Hz identity world for D1 was not built; the options work at any frame rate.

## 2026-11-06

**Baseline architecture: M2.** LightGBM alive/dead and team heads, 5 Hz features, Viterbi, trained at T = 2 s. 0.586 on
28 folds. Evidence: [bakeoff.md](bakeoff.md).

| Candidate | Result | Decision |
|---|---|---|
| Ball-free features | +0.036, wins 14/14 | Keep |
| HSMM | -0.0007 | Drop, Viterbi is equal and cheaper |
| Spell rescoring | +0.001, CI includes 0 | Drop |
| TCN | +0.013 seed 0, +0.010 seed 1 on 8 folds | Not adopted alone, test as ensemble member |
| Dead-ball cues | +0.005, CI includes 0 | Keep, harmless and they serve the tie-break |
| Missing-ball features | +0.001 | Keep, could matter if the target couples ball loss to dead ball |
| PathCRF | Not run: needs 22 identified players, weights trained on our IDSSE matches | Drop |
| Simulated team tags in the decoder | At best +0.023 at 90% accuracy, can be negative | Timeboxed experiment only |

**Main bet: possessions under 8 s.** They are 75-85% of unmatched true possessions, and even clean input leaves
0.71 predicted possessions per true one on IDSSE. Test native-frame-rate touch features first.

**Target match is inference-only for now.** We have a second-hand report of a Nov 3 organiser email saying training and tuning must
use external data. The public rules allow algorithmic pseudo-labels. Until the wording is confirmed we use the strict reading:
no self-training, and no fitting the degrader to the target.

**Locked benches are spent for selection.** `DFL-MAT-J03WQQ` and `Sample_Game_2` have been scored once. Use them for
one final sanity read before the freeze, and make no choice between versions from them.

**Compliance note.** A research agent opened Zenodo record 20185375 (`dataposesion.xlsx`). It includes statistics from the
banned match, and a fragment of one row was printed in that agent's session. The file was deleted and is not in this repo
or any pipeline, and nothing from it was used. Do not download that record or DataHub LaLiga.

**Not decided: StatsBomb.** Its agreement asks for a name and email and bans redistribution. 35 matches were fetched to a
temporary folder before the agreement was read. They are not in this repo or any pipeline. The team decides.
