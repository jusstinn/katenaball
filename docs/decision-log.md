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

### D4 label convention (David + Claude)

Evidence: [label-convention.md](label-convention.md). External development matches only; the locked benches were not
loaded, and the public 5 minutes were not looked at.

**Review.** The sandwich merge matches the written definition for brief opponent touches that return to the owner, and
keeps restart possessors. It never merges a brief opponent touch that runs straight into dead ball: at T = 2 about 35
per IDSSE match are left (11% of spells), and a larger T does not remove them. Added `sandwich(..., tail=True)` and
`order="shortest"` as options; the defaults and every cache are unchanged.

**3.5 s vs T = 2.** David's control evidence, rerun on 6 IDSSE development matches without the bench: pooled best
threshold 3.5 s (was 3.7 s), leave-one-match-out 3.5-3.8 s, bootstrap 90% [3.5, 4.0] s. It estimates what "controlled"
means in DFL data; T = 2 is the training choice under an unknown convention. Neither identifies the target's convention.
As training labels the 3.5 s convention loses (H3.5 row below), even against its own ground truth on the 8 folds.

**Metrica dead ball (D5 audit A3).** Confirmed: main's trigger rule keeps D through live play and does not end a
stoppage at goal kicks tagged BALL LOST. David's backward-from-restart rule, ported as `katena.external.metrica_stoppages`
(the corpus loader is untouched), puts the ball missing on 99.1% of Game 3's dead frames (main: 88.0%) and 85.0% of Game
1's (73.1%), at a cost on Game 1's alive frames (ball present 96.3%, was 99.2%). Gate: neutral (MR row).

**Training across T** (target-like, tau fixed at 2 per D5 audit A2; gate against M2 at T = 2 ground truth; the 8 IDSSE +
Metrica folds reported beside all 28 per audit A18; M2 = 0.569 / 0.586). Regret: worst case over evaluation T = 0-5.

| Candidate | Gate, 8 folds | Gate, 28 folds | Worst-case regret, 8 / 28 (M2: 0.005 / 0.009) | Decision |
|---|---|---|---|---|
| MIX: one random T per training row | -0.0058, 1 of 8 wins | -0.0001, 43% | 0.012 / 0.006 | Drop |
| AUG: soft labels over all T | -0.0111, 0 of 8 | -0.0050 [-0.0080, -0.0020], 21% | 0.017 / 0.010 | Drop |
| H3.5: David's 3.5 s labels | -0.0160, 0 of 8 | -0.0078 [-0.0118, -0.0038], 29% | 0.024 / 0.016 | Drop |
| S2tail: T = 2 plus `tail=True` | +0.0007, 75% | +0.0024 [-0.0003, +0.0051], 68% | 0.007 / 0.004 | Not adopted (narrow miss); keep ready |
| MR: Metrica stoppages from restarts | +0.0026 (own labels), 62% | +0.0010, 57% | - | Not adopted; a label-quality call for the team |
| S1: train at T = 1 (reference) | +0.0055, 8 of 8, CI above 0 | +0.0034 [+0.0005, +0.0062], 79% | 0.010 / 0.011 | Passes the rule. Proposed, not applied: keep T_MAIN = 2 (regret, loses at T = 5 and the event-defined convention) |

Decoder priors from another convention (M2's heads, no retraining) move every score by less than 0.004: no lever.
`protocol.T_MAIN` unchanged.

**Correction (review, after the gate).** The S2tail and H3.5 numbers above were computed with `tail=True` treating every end of a graded run as "touched, then out". On SkillCorner a run also ends at every X gap inside a half (camera cutaways), after which the same team may still have the ball, so those labels merged some flips they should have kept. `relabel`, `tmix.conv_labels` and `label_evidence` now pass the half blocks, and only the end of the last graded run of each half counts as out (`katena.labels.run_ends_out`; IDSSE and Metrica are unaffected). Neither convention is adopted, so the gate was not re-run; re-run it before adopting either.

**Compliance note.** David's earlier evidence for `min_control_s = 3.5 s` (origin/david/foundation) pooled all 8 of its
matches, including `DFL-MAT-J03WQQ`, which is now the locked IDSSE bench. Only label statistics were computed from it;
no model was trained, scored or selected on it. The rerun without it moves the pooled estimate from 3.7 s to 3.5 s and
changes no conclusion.

### D1: native-frame-rate touch features (David)

**What was built.** A new feature group `touch` (53 columns, `src/katena/featurize/touch.py`) computed at the source
frame rate (25 Hz IDSSE and Metrica, 10 Hz SkillCorner) and aggregated onto the 5 Hz grid. The ball's velocity breaks are
found by fitting least-squares lines over 0.4 s before and after every native frame, and a break must beat a z-score of
3.5 against a jitter estimate taken from second differences. Each break goes to the nearest player of either team
within 3 m. Per-sample features follow from that: time since and until each team's last and next touch, the team of the
last, previous and next touch, turnovers between touches, touch counts and balance in 0.5-2 s windows, the strongest break
near the sample, each team's closest approach and control margin from a smoothed native ball in 0.2-1 s windows, ball
speed, flight, visibility and the jitter estimate. The hook is `featurize(m, focal, dirs, native=NativeBall)`, and
`drop=("touch",)` returns M2's 160 columns unchanged. Gate code: `katena.eval.touch_gate`, `scripts/run_touch_gate.py`.

**Noise consistency.** The corpus corrupts the 5 Hz match. The gate re-runs that corruption with the corpus seeds and
checks it against the cache for every match and condition (identical M0 predictions and degrader statistics, pool rows
aligned). It then lifts the ball corruption to native frames: a dropped sample drops its native frames, a false detection
is held, every other native frame gets the clean position plus i.i.d. N(0, sigma^2) jitter (the degrader's own model, per
native frame), and frame `i * step` is set to the 5 Hz value exactly. The lifted ball therefore downsamples to the cached
5 Hz ball, and M2's features do not change. Players are the corrupted 5 Hz slots interpolated to native frames. Corrupting
at 25 Hz and then downsampling was rejected: the degrader is not frame-rate invariant (the ghost random-walk step, the
false-ball rate and length and the minimum fragment length are all per sample), and it would move M2 off the cached world,
so the comparison would no longer be paired.

**Hypothesis test first** (6 IDSSE folds, IDSSE-only training, tau 2, `scripts/probe_touch.py` and the first feature
version as of commit f0a5925; the later probe script adds smooth-ball and mixed-jitter settings that were not run).

| setting | touch group vs base | same features from the 5 Hz ball vs base | native vs 5 Hz |
|---|---|---|---|
| clean train, clean test | +0.051 [+0.037, +0.066], 5/6 | +0.034 [+0.020, +0.049], 6/6 | +0.017 [+0.005, +0.028], 5/6 |
| noise-trained, target-like | +0.008 [-0.004, +0.019], 5/6 | +0.001 | +0.007, 6/6 |
| noise-trained, severe | +0.001 | -0.003 | +0.004 |

On clean input touches carry much of the missing short-spell signal: predicted possessions per true one rise from 0.885 to
0.943, and recall at IoU 0.5 for spells under 2 s rises from 0.34 to 0.47. About a third of that gain needs the native
rate. Under the degrader's noise most of it goes. After this probe, one revision went in before the screen: distances from
a smoothed native ball, plus native-rate control-margin windows (46 to 53 columns).

**Screen and gate** (M2 + touch against M2, all 27 other development matches train, M2's heads and pool, Viterbi at a
fixed tau = 2 as the D5 audit recommends; `run_touch_gate.py build screen` then `gate`). Fixed-tau M2 equals the cached
M2 on every fold, and an M2 refit from the joined pool reproduces the cached M2 on all 8 screen folds.

| condition | folds | M2 | M2 + touch | diff [95% paired CI] | wins |
|---|---|---|---|---|---|
| target-like | 28 | 0.5861 | 0.5930 | +0.0069 [+0.0023, +0.0114] | 19/28 (68%) |
| severe | 28 | 0.5475 | 0.5526 | +0.0051 [+0.0005, +0.0097] | 18/28 (64%) |
| mild | 28 | 0.6116 | 0.6215 | +0.0099 [+0.0057, +0.0143] | 20/28 (71%) |
| native (near-clean, not for decisions) | 28 | 0.6210 | 0.6352 | +0.0141 [+0.0098, +0.0188] | 19/28 (68%) |
| target-like | 8 IDSSE + Metrica | 0.5689 | 0.5800 | +0.0111 [+0.0030, +0.0189] | 7/8 |
| severe | 8 IDSSE + Metrica | 0.5149 | 0.5268 | +0.0119 [+0.0035, +0.0203] | 7/8 |

At target-like, by dataset: IDSSE +0.0105 (5/6), Metrica +0.0129 (2/2), SkillCorner +0.0052 (12/20, median +0.0026). The
9 losing folds lose at most 0.0084, and 8 of them are SkillCorner. Its 10 Hz source gives only two native frames per
5 Hz sample.

Spell counts and length bins, target-like, 28 folds. Predicted possessions per true one go from 0.737 to 0.745 (IDSSE
0.648 to 0.646, Metrica 0.837 to 0.884, SkillCorner 0.760 to 0.768). Recall at IoU 0.5 by true length:

| model | 0-2 s | 2-4 s | 4-8 s | 8-16 s | 16-32 s | 32+ s |
|---|---|---|---|---|---|---|
| M2 | 0.167 | 0.228 | 0.491 | 0.676 | 0.817 | 0.918 |
| M2 + touch | 0.169 | 0.242 | 0.494 | 0.691 | 0.819 | 0.932 |

At severe, predicted possessions per true one go down (0.747 to 0.737), and recall stays within 0.01 of M2's in every bin
(0-2 s 0.152 to 0.144, 2-4 s 0.187 to 0.195, 32+ s 0.897 to 0.900).

Evaluation-only worlds, never trained on. The smooth-ball world was declared before the screen: the target-like
corruption with the same seed and no ball jitter, +0.0167 [+0.0083, +0.0251], 8/8. The IDSSE oracle and coupled worlds give
clean ball +0.029 (6/6), dead_odds 3 +0.015 (5/6) and dead_odds 15 +0.014 (5/6).

D2's evaluation worlds were added after the gate, on request. They ran on 8 folds with `run_touch_gate.py worlds`, using
`degrade.py` and `worlds.py` from `origin/david/d2-degrader` 9e55800 placed in the tree and not committed here. M2 + touch
is ahead in all three: `target_cpl50` +0.0132 [+0.0052, +0.0217] 8/8,
`target_cpl100` +0.0212 [+0.0121, +0.0304] 8/8, and `target_smooth` (SkillCorner-matched jitter) +0.0268 [+0.0187, +0.0352]
7/8. With D2's degrader the touch build's recovery check passed on all 28 matches, which confirms on real data that it
reproduces the old worlds byte for byte.

**Is it the native rate?** `run_touch_gate.py rate` runs the same 53 columns from the corrupted 5 Hz ball, trained and
decoded the same way (8 folds). The 5 Hz version gains +0.0060 [-0.0007, +0.0131] over M2 at target-like and +0.0118
at severe. The native rate adds +0.0051 [-0.0019, +0.0121] (6/8) at target-like, +0.0001 at severe, +0.0034 in the
smooth-ball world, +0.016 at mild and +0.020 on clean input. About half of the target-like gain is the feature design and
half the frame rate. The frame rate's share grows as the noise falls, and the CI includes 0 at target-like.

**Decision: not adopted under the pre-registered rule.** The paired interval excludes zero at target-like and at severe,
and nothing gets worse in any world or severity. But the win rate is 19/28 (68%), one fold short of 70%. On the eight
25 Hz folds alone it passes (7/8). The group stays out of the baseline. Whether to take it anyway is a team call, not
this gate's: the target is 25 Hz, 8 of the 9 losses sit on 10 Hz SkillCorner folds, and on the 8 folds the effect is
larger in every smoother or coupled world than at target-like. Nothing was tuned after the gate.

**Caveat on noise.** Real native ball tracks are smooth frame to frame. The median absolute second difference at 25 Hz is
0.001 m on Metrica sample games 1 and 3, and consecutive increments correlate +0.6 to +0.9. The degrader's target-like
jitter gives about 1 m. D5 (audit A20) and D2 measure the same gap at 5 Hz. Touch detection loses most where the jitter is
rough, so the target-like numbers are probably pessimistic for this group. We did not tune toward clean data because of
it.

**On the real match (human-run, aggregates only).** `uv run python scripts/check_touch_target.py` prints counts, rates,
a runtime, the feature shape and two jitter statistics of the real ball, next to the same numbers for one external
development match in the target-like, smooth-ball and severe worlds. It is an alarm for which jitter world the target
resembles, never a selector.

**Correction (review, after the gate).** `touch.lift_ball` let the ball reappear inside a simulated loss burst on the native frames of a sample that was already missing at the source (Metrica and SkillCorner; IDSSE never misses the ball natively). Such a sample now counts as dropped when the nearest source-visible samples on both sides were dropped. The numbers above predate the fix and were not re-run, since the group is not adopted; the touch cache is keyed on the hash of `touch.py`, so any new run rebuilds it.

### D3: TCN + LightGBM ensemble (`david/d3-ensemble`)

**What.** `katena.models.ensemble` averages the M2 LightGBM heads and the TCN in logit space, per head, with TCN
weight 0.5 fixed before any result (`W_MAIN`), and decodes with the unchanged Viterbi at fixed tau = 2 (D5 audit A2;
M2 chose 2 on all 28 folds, so M2 is unchanged). The TCN is M5 unchanged (`katena.models.tcn`), trained with seed 0,
4 torch threads and deterministic algorithms. Both members of fold k are trained on the same 27 development matches.
Code: `katena.eval.d3`, `scripts/run_d3.py`, `scripts/d3_determinism.py`. External matches only; the public window
was not looked at; the locked benches were not run.

**Gate.** Adopt if the ensemble wins on at least 70% of folds and the paired 95% interval excludes 0, at target-like
and at severe noise. `D3M2` (w = 0 through the same code) equals the stored M2 on every fold and condition.

| comparison | folds | noise | candidate | M2 | diff | 95% paired CI | wins | rule |
|---|---|---|---|---|---|---|---|---|
| ensemble vs M2 | 28 | target | 0.5926 | 0.5861 | +0.0065 | [+0.0027, +0.0103] | 18/28 (64%) | **fail** (wins) |
| ensemble vs M2 | 28 | severe | 0.5602 | 0.5475 | +0.0127 | [+0.0084, +0.0169] | 24/28 (86%) | pass |
| ensemble vs M2 | 8 IDSSE + Metrica (the screen) | target | 0.5809 | 0.5689 | +0.0120 | [+0.0053, +0.0183] | 7/8 | pass |
| ensemble vs M2 | 8 IDSSE + Metrica (the screen) | severe | 0.5263 | 0.5149 | +0.0114 | [+0.0033, +0.0189] | 7/8 | pass |
| ensemble vs M2 | 20 SkillCorner | target | 0.5973 | 0.5930 | +0.0042 | | 11/20 | |
| TCN alone vs M2 | 28 | target | 0.5897 | 0.5861 | +0.0036 | [-0.0021, +0.0092] | 13/28 | fail |
| TCN alone vs M2 | 28 | severe | 0.5620 | 0.5475 | +0.0145 | [+0.0090, +0.0203] | 25/28 | pass |
| TCN alone vs M2 | 8 | target | 0.5787 | 0.5689 | +0.0098 | [-0.0008, +0.0201] | 6/8 | fail |

macroF1 (IDSSE + Metrica, target-like): ensemble 0.8736, M2 0.8677. The TCN alone now has a SkillCorner score for the
first time: 0.5941 against M2's 0.5930.

**Second TCN seed (8 IDSSE + Metrica folds, a stability check).** With seed 1 the ensemble still passes: +0.0084
[+0.0017, +0.0151], 7/8 at target-like; +0.0104 [+0.0033, +0.0172], 8/8 at severe. The TCN alone does not (+0.0032,
4/8 at target-like), as in the bake-off: averaging with LightGBM absorbs most of the TCN's seed variance.

**Strictly nested weight (secondary, information).** For each M6 group, w in {0, 0.25, 0.5, 0.75, 1} was chosen on
inner predictions from LightGBM heads and TCNs trained without that group and without the scored match's own group
(15 group pairs, 39 min of one slot). The groups chose 0.5, 0.5, 0.75, 0.25, 0.75, 0.5, never w = 0. Applied to the
LOMO predictions: target-like +0.0044 [+0.0002, +0.0084], 14/28 wins, fail; severe +0.0139 [+0.0095, +0.0183],
25/28, pass. Choosing the weight inside the folds does not rescue the target-like gate.

**Where the gain is, and where it is not (information).** The TCN makes the ensemble smoother. It predicts fewer
possessions per true one (IDSSE + Metrica 0.67 against M2's 0.69 at target-like, 0.65 against 0.70 at severe;
SkillCorner 0.735 against 0.760) and, at target-like, matches more of the possessions over 8 s: it removes false
flips. It finds no more short ones: under 4 s it matches the same share on IDSSE + Metrica at target-like, fewer at
severe, and fewer on SkillCorner (0-2 s 0.206 against 0.221, 2-4 s 0.222 against 0.239, matched at IoU 0.5), which
is why SkillCorner is a coin flip at target-like (11/20, +0.004)
while severe noise, where false flips dominate, is a clear win (17/20). D1 targets exactly the short spells, so the
two are complementary, but the ensemble should be re-gated on top of any D1 change. Fixed-weight curve over 28 folds,
target-like / severe: w 0.25 +0.0050 (71% wins) / +0.0095; 0.5 +0.0065 (64%) / +0.0127; 0.75 +0.0062 (61%) /
+0.0163; 1 +0.0036 (46%) / +0.0145. This is a sensitivity table only: picking w = 0.25 from it because it clears
70% would be choosing on the gate folds.

**Determinism.** Two fresh processes that train both members on the 27 matches of fold `DFL-MAT-J03WMX` and predict it
give byte-identical TCN weights, LightGBM, TCN and blended probabilities, and decoded labels (sha256 of the
target-like blend `35f9c2a5...`, labels `64691ace...`). The TCN probabilities also equal, byte for byte, those of the
LOMO process that trained eight folds in a row, and the LightGBM probabilities equal D5's bake-off cache. This covers
one machine: Windows 10, i7-7700 (AVX2), torch 2.15.0+cpu (MSVC build, oneDNN 3.13.3), LightGBM 4.7.0 with 8 threads,
the locked environment. It does not cover macOS, Linux, ARM or another torch build.

**Cross-machine outlook (probed on this machine, fold J03WMX with the five other IDSSE matches as training set).**
Changing what another machine would change (3 torch threads instead of 4, ATen's scalar kernels instead of AVX2,
oneDNN capped at SSE4.1, or the same weights run for inference under both) changes the TCN weight and probability
bytes, by at most 1.2e-6 in probability, but no decoded label (29,123 samples, target-like and severe). Training did
not amplify last-bit differences. Moving every probability by one float32 ulp changes no label; rounding to 2^-16
before decoding is label-neutral, and coarser rounding changes the result itself, so rounding is not needed and does
not help. Expect identical labels but not identical model or probability bytes across machines: the two-machine test
must compare the submission bytes (D5 A11). Not tested on another OS or on ARM; `scripts/d3_determinism.py synthetic`
gives CI a data-free TCN hash to compare across operating systems.

**CPU.** One LOMO TCN fold (27 training matches, 1200 steps, 4 threads, i7-7700): training 126-277 s (median 155 s)
with another heavy job running, prediction of four conditions 2-6 s, peak RSS 3.3 GB. All 28 folds took 79 min of
one slot, so no cheaper configuration was needed. The final pipeline trains it once.

**Decision: not adopted; M2 stays.** The pre-declared candidate fails the gate at target-like on 28 folds: the paired
interval excludes 0 (+0.0065 [+0.0027, +0.0103]) but it wins on only 18 of 28 folds (64%, the rule needs 70%),
because SkillCorner is a coin flip (11/20). It passes at severe (24/28) and on the 8 IDSSE + Metrica folds at both
noise levels. Determinism is not the blocker: same-machine regeneration is byte-identical. The code stays as the
tested second member. Re-gate it, with w = 0.5 declared in advance, on top of any D1 or D2 change. Weighting severe
noise or the dead-labelled folds more heavily would be a rule change, to be made before looking again, not after.

### Final candidate F (pre-registration)

**Registered before any score of F was computed.** Branch `david/final-candidate`, cut from `origin/main` at
`be107d1` (PRs #1 to #7, including the `lift_ball` review fix). External development matches only: no `data/`, no
`evaluation/gt/`, no public window, no locked bench. Code: `katena.eval.final`, `scripts/run_final.py`; cache
`D:/katena/work-final`. Nothing below changes after a result.

**Candidate.** `F` = `ensemble.blend_heads(lgbm, tcn, w=0.5)`, the logit average per head, decoded by the existing
3-state Viterbi at fixed tau = 2 with priors fit on the training matches' labels (`bakeoff.priors_for`: the IDSSE and
Metrica label sequences without the held-out match, as `d3.stage_score`), T = 2 (`protocol.T_MAIN`). In fold k both
members train on the same 27 development matches. The held-out match is in no model, scaler or prior.

| Member | Input columns | Training data | Model |
|---|---|---|---|
| LightGBM (`FG`, F at w = 0) | M2's 160 plus D1's 53 `touch` (213) | The `d2j` pool, built as `d2.build_pool` does: the bake-off pool's rows, orientations and T = 2 labels (stride 4, one residue class per severity), each severity's rows split evenly, interleaved, over `TRAIN_MIXES["d2j"]` for the match's dataset. The touch columns of each row come from the same world (`d2.assemble_pool` run over the touch files). | M2's heads and parameters (`gbdt.fit_heads`, `PARAMS`, 150 trees, through `Pools.fit`) |
| TCN (`FT`, F at w = 1) | M2's 160 only | One whole-match sequence per (training match, world) for every world in the match's `d2j` mix: 15 per IDSSE or Metrica match (5 worlds x 3 severities), 9 per SkillCorner match. Orientation drawn per sequence as `stages._tcn_seqs` does, from `default_rng(0)`, in the order development match (corpus order), severity (mild, target, severe), world (mix order), over all 28 matches, so the draws do not depend on the fold. Scaler: `tcn.Scaler` on every 6th `d2j` pool row (M2 columns) of the 27 training matches, as `d3.fit_tcn_excluding`. | `katena.models.tcn` unchanged, through `ensemble.fit_tcn`: seed 0, 4 threads, deterministic algorithms, `pin_cpu_env` before torch loads, 1200 steps |

- **Touch columns in every world.** Computed the way `touch_gate` computes its conditions. The 5 Hz world is
  re-corrupted with the corpus seed (`seed_for(id, world)`) and checked against that world's cached base features:
  identical in-play rows, M0 predictions and degrader statistics. Its ball corruption is then lifted to the native
  frames with `lift_ball`, using seed `seed_for(id, world + "|native")` and the world's ball jitter. No world needs an
  exception. Every D2 world corrupts the ball only by drops, held false detections and i.i.d. jitter, which is what
  `lift_ball` lifts. Identity switches and fragment cuts move players only.
- **Evaluation worlds.** `target` and `severe` on 28 folds decide. F and both members are also predicted in every
  `worlds.EVAL_WORLDS` world that applies to the fold. `target_cpl100_ids` is the one world outside the training mix,
  so its touch columns are built for the 8 IDSSE + Metrica matches as well.
- **A consequence of the TCN design, recorded now.** `tcn.train` draws windows in proportion to sequence length.
  An IDSSE or Metrica match now has 15 sequences against 9 for a SkillCorner match, so per match the TCN sees the 8
  dead-labelled matches 5/3 as often as D3's TCN did (3 sequences each). By pooled length they give 39% of the
  windows instead of 28%. The LightGBM pool keeps each match's row count. This is the definition as written, and it stays.
- **Memory, not a change.** The sequences are held in float16 (as `_tcn_seqs` holds them) and scaled when a window is
  drawn, instead of all 285 to 291 training sequences up front (about 5 GB of float32). The scaler is elementwise, so
  the training input is bit-identical. `tests/test_final.py` checks the TCN weight hash both ways.

**Incumbent.** The stored M2 scores in `D:/katena/work-base` (read only; target-like 0.5861 on 28 folds). This code
re-decodes `work-base/oof/full` at tau 2, and the result must equal the stored scores on every fold and condition.
`FG` is not M2: it is the LightGBM member with touch and `d2j`.

**Rule** (`katena.eval.bootstrap`, unchanged). Adopt F if and only if both hold on the 28 development folds:

- at `target`, it wins on at least 70% of folds (at least 20 of 28, a win being a strictly higher spellF1) and the
  paired 95% bootstrap interval of the mean difference (possessions resampled, 2000 draws, seed 0) lies above 0
  (`adopted`);
- at `severe`, it is not worse. Worse means, as in D2, a mean difference below 0 with the interval's upper bound also
  below 0.

The 8 IDSSE + Metrica folds (`stages.FOLDS8`) and the per-dataset means are reported beside the 28 and do not decide.
Nothing is tuned after the gate. If F fails, M2 stays.

**Information only, never a reason to change F.**

- F in `worlds.EVAL_WORLDS` (the coupled worlds on the 8 folds only), against M2's heads in the same worlds. These
  are D2's `cur` probabilities re-decoded at tau 2; they must equal the stored M2 at `target` and `severe`.
- `FG` and `FT` alone, taken from the same runs.
- macroF1 on IDSSE + Metrica.
- CPU time and wall time of each stage, and peak memory.
- Same-machine byte identity of one fold, `DFL-MAT-J03WMX`: two fresh processes train both members and decode, and
  we compare the SHA-256 of the TCN weights, of both members' and the blended probabilities, and of the labels at
  `target` and `severe`, also against the LOMO run's files for that fold.

**Reused caches.** These are read only and copied into `work-final`, development matches only.

- From `work-base` (code `8537084`): ground truth, base pools and the `mild`/`target`/`severe` features. `featurize`
  and `corpus` have changed since only in imports, comments and the optional native-ball hook. The degrader's default
  path is pinned by D2's frozen-copy test.
- From `work-d2`: the D2-world features, built 11:04 to 11:49 on 2026-11-07. `degrade.py` has not changed since
  `8067e59` (10:42), and the world definitions have not changed since they were built.

Every copied (match, world) is checked in the touch build (rows, M0, degrader statistics). The `d2j` pools are rebuilt
from the copies and compared with `work-d2/pool_d2j`. The first match of each dataset is re-featurized in all of its
worlds and compared byte for byte. A file that fails a check is rebuilt in `work-final` with the current code, never
dropped.

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
