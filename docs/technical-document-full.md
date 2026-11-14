# Technical document, long version

More than a Hack 2027, Phase 1. Repository: https://github.com/jusstinn/katenaball. Version of 2026-11-13.

The document we submit is the short one, [`technical-document.md`](technical-document.md). This is the same
account with every table and every change we tested. Every number comes from `docs/decision-log.md` and the
evidence pages it links. What only the real run can fill in (its public-window result, the tagged commit and
the file's SHA-256) is filled in there, not here.

## 1. Summary

- We label each frame H, A or D from the automatic tracking (ATD). A LightGBM model and a temporal
  convolutional network (TCN) each predict whether the ball is alive and which team has it. We average
  them in logit space and decode with a 3-state Viterbi. The only other input is one Smart Tagging kickoff
  tag per half (section 6).
- Models are trained and selected on public external matches only (IDSSE/DFL, Metrica, SkillCorner): 28
  for development, and all 30 for the final model. A degrader corrupts their tracking to look like
  automatic tracking.
- No model was trained, tuned or selected on the provided match. Nobody labelled any frame by hand.
- The model before the submitted one, F3, came from two gated steps, each under a rule registered before
  its score, on 28 leave-one-match-out development folds at target-like noise. F beat the previous
  baseline, M2: spellF1 0.6019 against 0.5861, +0.0158 [+0.0111, +0.0205], 22 of 28 folds won. F3 is F with
  a larger LightGBM model and a longer-trained TCN, and it beat F: 0.6180 against 0.6019, +0.0161 [+0.0118,
  +0.0207], 21 of 28 folds won.
- The submitted model, AW2, is F3 with three changes to its training and none to its inference path. They
  came in two gated steps (section 3.5).
- **AW: new labels and weights.** The 20 SkillCorner matches are relabelled from SkillCorner's own action
  events, and both team heads weigh a training frame by the inverse length of its possession. The standing
  rule scores 20 of its 28 folds against the label AW stops training on, so AW was judged under a second
  rule, declared before any score. On the 8 IDSSE and Metrica folds, whose truth is untouched, AW scores
  0.6923 against F3's 0.6374: +0.0548 [+0.0399, +0.0697], 8 of 8 folds won. On the 20 SkillCorner folds
  scored against the new labels it scores 0.6586 against 0.6094. Under the standing rule, with SkillCorner
  scored on the old label, it loses 0.0224 on the 28 folds and wins 8 of them.
- **AW2: a longer-trained TCN.** AW2 is AW with a TCN trained 9,600 steps instead of 4,800. It was
  registered before its score under its own rule and beat AW on the 28 folds with the new labels on
  SkillCorner: +0.0071 [+0.0046, +0.0096], 20 of 28 folds won. AW2 scores 0.6993 on the 8 IDSSE and Metrica
  folds (F3: 0.6374; +0.0618, 8 of 8), 0.6657 on the 20 SkillCorner folds against the new labels (F3:
  0.6094; +0.0563, 18 of 20), and 0.6753 against 0.6174 on all 28.
- AW2 is a bet on how the organisers describe the ground truth (section 7). These are development
  estimates, not a prediction of the hidden score.

## 2. Approach

### 2.1 Pipeline

1. Read the tactical ATD: 46 anonymous slots with a team each, plus the ball, in metres.
2. Downsample to 5 Hz (every 5th frame).
3. Compute 160 feature columns per 5 Hz sample, plus 53 `touch` columns from the ball at the native 25 Hz.
4. Predict p(alive) and p(home) per sample with each of the two members.
5. Average the members in logit space, per head, with weight 0.5.
6. Decode each half with a 3-state Viterbi.
7. Expand to frames: each frame takes the label of the nearest sample in its own half. Graded frames get
   H, A or D; frames outside the graded halves get X. The submission path (`katena.pipeline.label_match`)
   calls the same function as the development folds (`katena.eval.score.to_native_codes`, with the half
   blocks), and writes the CSV as bytes with LF line ends on every OS.

### 2.2 Features

Features describe a *focal* team attacking towards +x. The attack direction is inferred from player
positions. Every row is computed twice, once with each team as focal, and the predictions are averaged, so
the model treats both sides the same.

| Group | Columns | Content |
|---|---|---|
| `ball` | 62 | Gap-aware ball kinematics, nearest player of each team, persistence of control |
| `free` | 61 | No ball needed: team shape, line heights, velocity along the attack axis, pressing, a contact hot-spot, tracked-object counts |
| `dead_ball` | 13 | Dead-ball cues from the ball: out of bounds, stillness, restart geometry |
| `dead_player` | 16 | Dead-ball cues from players only: stillness, restart layouts, player counts in the box and the centre |
| `miss` | 8 | Ball visibility fractions, time since the ball was seen |
| `touch` | 53 | From the native-rate ball (D1), below |

**Touch features.** At every native frame we fit a straight line to the visible ball over the 0.4 s before
it and another over the 0.4 s after. A touch is a velocity break of at least 2 m/s that also exceeds 3.5
times a local jitter estimate. It goes to the nearest player within 3 m who is clearly nearer than the
nearest opponent. Per sample we then compute time since and until each team's touches, who touched last,
turnovers, touch counts and balance in 0.5-2 s windows, each team's closest approach and control margin in
0.2-1 s windows, ball speed and flight. Both members get these columns (213 in total); in F only the
LightGBM member did.

### 2.3 Models and decoder

- **LightGBM heads** (`katena.models.gbdt`). The *alive* head trains on every row labelled H, A or D, and
  the *team* head ("the focal team has the ball") on the H and A rows, from all three sources. Up to F3
  the alive head saw IDSSE and Metrica rows only, because the SkillCorner label then in use has no dead
  ball; the labels AW and AW2 train on have one (sections 3.5 and 6). The parameters were fixed before the
  bake-off and never tuned, apart from the number of trees: 1,200 (F had 150; section 3.4). Learning rate
  0.1, 31 leaves, at least 50 rows per leaf, feature and bagging fractions 0.5, `max_bin` 63, seed 0,
  deterministic, 8 threads.
- **TCN** (`katena.models.tcn`). Six residual blocks of 48 channels with dilations 1 to 32, looking both
  ways in time, with the same two heads. It sees all 213 columns and trains for 9,600 AdamW steps on
  random 256-sample windows, with seed 0, 4 threads and deterministic algorithms. The one-cycle schedule
  spans the steps. F trained the 160 base columns for 1,200 steps (section 3.4); F3 and AW trained 213
  columns for 4,800 (sections 3.4 and 3.5).
- **Possession-length weights** (`katena.models.lenweight`; new in AW, section 3.5). In both team heads a
  training frame weighs 1 / L, with L the length in seconds of the constant-label run it sits in, in the
  T = 2 labels it trains on, clipped to [1, 30]. spellF1 counts every possession once, however long it
  lasts, and an unweighted per-frame loss counts a 60 s possession thirty times as much as a 2 s one. The
  LightGBM team head scales the weights to mean 1 over its rows; the TCN's team loss divides by their sum.
  The alive heads carry no weight. The constants were fixed before their first probe and never tuned.
- **Blend** (`katena.models.ensemble`). `logit p = 0.5 logit p_lgbm + 0.5 logit p_tcn` per head. The
  weight was fixed before any result.
- **Decoder** (`katena.decode.hmm`). With pa = p(alive) and ph = p(home), the class probabilities are
  pa·ph for H, pa·(1 - ph) for A and 1 - pa for D. The emission score is tau (log p(class) - log
  prior(class)) with tau = 2. Transitions and class priors are smoothed counts from the IDSSE and Metrica
  training labels. Tau sits on a plateau (M2: 0.583, 0.586 and 0.582 at tau 1, 2 and 4). All 28 bake-off
  folds chose 2, so every later gate fixed it in advance.

### 2.4 Training under simulated noise

The provided ATD has no identities, 46 slots for 22 players, and no ball on 41% of graded frames
(DATA.md). IDSSE's tracking is clean, and Metrica's and SkillCorner's keep identities and lose the ball
less often. `katena.sim.degrade` tops up their ball loss to the severity's level, cuts players into anonymous
fragments in 23 slots per team, and adds gaps, ghosts and duplicates. It also drops the ball in bursts that
start more often in crowds and in flight, and adds jitter, false ball detections and wrong-team slots. Gap
lengths and the ball-loss hazard are calibrated on SkillCorner and Metrica. The target-like severity uses
DATA.md's published aggregates, and mild and severe bracket it:

| Severity | Ball missing | Mean tracked objects (realised on IDSSE) |
|---|---|---|
| `mild` | 20% | 21.3 |
| `target` | 41% | 20.2 |
| `severe` | 55% | 18.2 |

D2 added worlds that keep each severity's overall ball loss (`katena.sim.worlds`):

- **`_cpl50`, `_cpl100`** tie ball loss to stoppages. P(ball missing | dead) = r + kappa (0.79 - r), where
  r is the overall rate, kappa is 0.5 or 1, and 0.79 is the rate in Metrica's two development games.
- **`_ids`** adds identity switches in crowds, at assumed rates.
- **`_smooth`** cuts jitter to the level of SkillCorner's development matches at 5 Hz.

F, F3, AW and AW2 all train on the `d2j` mix. Each severity's rows are split evenly over five worlds for IDSSE and
Metrica (`s`, `s_cpl50`, `s_cpl100`, `s_ids`, `s_smooth`) and over three for SkillCorner (`s`, `s_ids`,
`s_smooth`): the mix was built when the SkillCorner label had no D, and AW and AW2 keep F3's rows and
sequences as they are. LightGBM sees every 4th 5 Hz sample, with a random focal orientation per row. The
TCN sees one whole-match sequence per match and world. For the `touch` columns, each world's ball
corruption is carried over to the native frames (`touch.lift_ball`), consistently with the 5 Hz ball.

## 3. Why this design

### 3.1 How we validated

- **Leave-one-match-out on 28 external development matches** (6 IDSSE, 2 Metrica, 20 SkillCorner). In
  fold k, no model, scaler or prior sees match k. Scores come from the official
  `possession_metrics.evaluate`, unchanged. A SkillCorner fold scored against SkillCorner's per-frame
  label, which has no D, has a predicted D scored as X. Scored against the action labels of section 3.5,
  it is scored as an IDSSE fold is.
- **A pre-registered gate (rule 1).** A change is adopted if, at target-like noise, it wins on at least
  70% of folds and the paired 95% bootstrap interval of the mean difference lies above 0. The bootstrap
  resamples ground-truth possessions in each match, 2,000 draws, seed 0. The rule comes from
  `docs/PLAN.md`. D2 and F were registered in the decision log before any result, and D3's blend weight
  was fixed before any result. We report the 8 IDSSE and Metrica folds beside the 28, because the 20
  SkillCorner folds, scored on the per-frame label, cannot show a dead-ball change.
- **A second rule for a change of training labels (rule 2).** AW replaces the SkillCorner training labels.
  Rule 1 scores 20 of its 28 folds against the label AW stops learning, so it cannot pass such a change.
  Rule 2, declared before any score of the new labels, lets the 8 IDSSE and Metrica folds decide, with
  the 20 SkillCorner folds scored against the new labels as a guard. It is a change of protocol and weaker
  evidence than rule 1. Section 3.5 gives both rules in full, and the rule of the last step, the longer
  TCN: rule 1's thresholds on the 28 folds with the new labels on SkillCorner.
- **Two locked benches**, `DFL-MAT-J03WQQ` (chosen by hash) and Metrica `Sample_Game_2`, were scored once
  in the bake-off and no choice was made from them. After F was adopted, F was read on them once, trained on
  the 28 development matches only. The read was registered before any score, as a sanity check that
  changes no decision. F beat M2 on both benches in both registered conditions:

  | Bench | Noise | F | M2 | Difference |
  |---|---|---|---|---|
  | IDSSE `DFL-MAT-J03WQQ` | target-like | 0.6092 | 0.5669 | +0.0422 |
  | IDSSE `DFL-MAT-J03WQQ` | severe | 0.5605 | 0.5478 | +0.0126 |
  | Metrica `Sample_Game_2` | target-like | 0.6820 | 0.6311 | +0.0509 |
  | Metrica `Sample_Game_2` | severe | 0.5945 | 0.5371 | +0.0574 |

  F is ahead in 15 of the 16 pairs of bench and evaluation world (2 benches, 8 worlds; the table shows the 4
  registered ones); the exception loses 0.003. M2 retrained here
  reproduces the bake-off's recorded bench scores to the 3 decimals printed. The final model also trains
  on these two matches, so this read is of F trained on 28 matches, not of the submitted model. F3, AW and
  AW2 have no bench read: the benches were spent by then.
- **The provided match is inference only.** No training, tuning, self-training or fitting of the degrader
  on it, which is also what the organisers' clarification of 2026-11-03 requires. We used only DATA.md's
  published aggregates (46 slots, 20.5 tracked objects, 41% ball missing) and the published half
  boundaries. Human-run checks on the real file (the loader, the slot layout, a touch-feature alarm and
  the kickoff tags) print pass or fail and aggregates only, and never choose between versions. Nobody compared our output with the video.
- **The public 5 minutes are a pipeline check only.** CI reads them only to check that the official scorer
  still reproduces the published scores of the published baseline files. The final file is checked on the
  window by `scripts/check_public_window.py`, which prints pass or fail only, never a score. It checks four
  things: the official file checks pass; swapping H and A lowers team accuracy (home and away are the right
  way round); shifting the labels by 5 or 10 s lowers spellF1 (the frames line up); and the file beats the
  published smooth50 baseline. A fail means a bug to find, not something to tune, and every run is logged.
  The result of the real run is in the short document.
- **One look at the public labels' structure, with no prediction and no score.** On Nov 9
  `scripts/probe_label_structure.py` counted the window's possessions and their lengths (17 possessions,
  median 7.6 s) and placed them among 5-minute windows of IDSSE labels under each label convention. The
  question was whether T = 2 looks implausible. It does not: the counts sit in the middle of the T = 2
  windows, and 17 possessions cannot separate T = 2 from T = 3. Nothing was changed because of this look.
- **One look at the public window beside the tactical video, labels only.** On Nov 11 one team member watched
  the public 5 minutes of the tactical video next to the public label file, to understand what counts as
  possession, as the organisers' clarification of Nov 3 allows. No prediction of ours was on screen and no
  score was computed. What he saw agrees with the IDSSE convention on loose balls and on dead ball; the two
  possessions under 2 s in the window are too few to choose T by, and the window may not choose it anyway.
  Nothing was changed because of this look.
- **The submission path equals the evaluation path.** We exported an external development match
  (IDSSE `DFL-MAT-J03WMX`) in the provided ATD file format and read it back bit-identical through the real
  loader (`katena.target.atd.load_target`). We trained F without it and labelled it with
  `make_submission.py predict`. Scored with the official scorer, the CSV gives spellF1 0.650611 and
  macroF1 0.930641, exactly what the evaluation code gives for the same model. Every probability and graded
  label is bit-identical on both paths. The same check with F3 gives 0.722130 and 0.941533 on both paths.
  It was not repeated with AW or AW2, which change the training and no line of either path.
  A SkillCorner match (10 Hz) also agrees, once the evaluation's
  "predicted D counts as X" rule for SkillCorner is applied (`scripts/dressed_check.py`). A synthetic
  bundle on the provided match's own timeline (199,319 frames, the graded halves of `games.py`) runs
  through the same path in CI, and the official validator accepts its CSV (`tests/test_pipeline_atd.py`).
- **The submission path does not break on damaged tracking.** No feature, model or decoder code had run on
  the provided match before the real run. `scripts/stress_predict.py` puts the same exported external match
  on 46 anonymous slots with 41% of the ball missing, then damages it in 27 further ways that the training
  worlds do not produce: no ball at all, nothing tracked for five minutes, one team missing, positions far
  off the pitch, infinite values, slots that change team. In every case, with F and with F3, labelling
  finished and the file passed the official checks. This shows the path does not crash, not that the labels
  are good. It was not repeated with AW or AW2, whose inference code is F3's.

### 3.2 Bake-off (2026-11-06)

Development folds, target-like noise, T = 2 s labels (section 6).

| Model | spellF1 | Outcome |
|---|---|---|
| M0: starter algorithm (nearest player, missing ball = D, majority window) | 0.280 | Reference |
| M1: fixed rules (1 s control dwell, hold through gaps, simple dead rule) | 0.432 | +0.152 over M0: adopted |
| **M2: LightGBM heads + Viterbi** | **0.586** | +0.154 over M1, 28/28 wins: adopted |
| M3: M2 + explicit-duration HSMM | 0.585 | -0.0007, 25% wins: dropped |
| M6: spell rescoring after decoding (28 grouped folds) | 0.584 | +0.0012, interval includes 0: dropped |
| M5: TCN alone (8 IDSSE + Metrica folds; M2 0.569 there) | 0.582 | +0.013 seed 0, +0.010 seed 1 (interval touches 0): kept for an ensemble |

Removing the ball-free features cost 0.036 (14/14 folds), so they stay. The dead-ball cues (+0.005,
interval includes 0) and the missing-ball features (+0.0007) are harmless and stay too. The oracles showed
where the points are. For M3 on the 8 folds, perfect team labels raise spellF1 from 0.568 to 0.898, while perfect
alive/dead raises it only to 0.639. Spells under 8 s are 75-85% of the true possessions we miss, so short
possessions became the main bet.

### 3.3 Work packages and the final candidate (2026-11-07)

Each package was gated against M2 (0.5861) on the 28 folds at target-like noise, with tau fixed at 2.

| Package | Change | Difference [95% CI] | Wins | Decision |
|---|---|---|---|---|
| D1 | `touch` columns on M2 | +0.0069 [+0.0023, +0.0114] | 19/28 (68%) | Missed by one fold |
| D2 | `d2j` training mix on M2 | +0.0019 [-0.0014, +0.0051] | 15/28 | Kept: its rule was "no decision world gets worse" |
| D3 | TCN + LightGBM blend, w = 0.5 | +0.0065 [+0.0027, +0.0103] | 18/28 (64%) | Missed on wins |
| D4 | Training-label conventions | Section 4 | | None adopted |
| **F** | **D1 + D2 + D3** | **+0.0158 [+0.0111, +0.0205]** | **22/28 (79%)** | **Adopted** |

**D2.** Both D2 mixes passed D2's rule. Amendment A2, written before the 28-fold run, prefers the mix with
the better worst decision world: +0.0019 for `d2j` against -0.0001 for `d2`. `d2j` gains +0.0160
[+0.0094, +0.0234] (8/8) in the coupled world `target_cpl50` and +0.0055 [+0.0014, +0.0097] (19/28) in
`target_smooth`. It costs IDSSE, the only dataset that never loses the ball natively, 0.006 at target-like
(2/6 wins). The mix is a bet that the real ATD loses the ball through stoppages and has smooth positions.

**F.** F, its parts and its rule were registered before any score of F (commit `3921643`). F had to pass
the gate at target-like and not be worse at severe.

| Folds | Noise | F | M2 | Difference [95% CI] | Wins |
|---|---|---|---|---|---|
| 28 | target-like | 0.6019 | 0.5861 | +0.0158 [+0.0111, +0.0205] | 22/28 |
| 28 | severe | 0.5696 | 0.5475 | +0.0221 [+0.0173, +0.0271] | 26/28 |
| 8 IDSSE + Metrica | target-like | 0.5991 | 0.5689 | +0.0302 [+0.0218, +0.0388] | 7/8 |
| 8 IDSSE + Metrica | severe | 0.5426 | 0.5149 | +0.0277 [+0.0186, +0.0366] | 8/8 |

At target-like, F scores 0.5670 on IDSSE (+0.0229, 5/6), 0.6955 on Metrica (+0.0521, 2/2) and 0.6030 on
SkillCorner (+0.0100, 15/20). Its 8-fold macroF1 is 0.8781, against 0.8677 for M2. F passes the adoption
rule in every D2 evaluation world. Among the target-level worlds the independent one gains least; the
coupled worlds gain +0.035 to +0.042 on the 8 folds. Neither member passes alone (LightGBM +0.0073, 17/28;
TCN +0.0100, 17/28). They err on different folds, so the blend passes.

These are means over external matches under simulated noise, not a forecast of the hidden score. The
label convention alone moves the level by about 0.15 (section 6), and the simulated noise is not the real
tracking (section 7).

### 3.4 After F: a larger tree model and a longer-trained TCN (F3, 2026-11-09 to 11)

After F, nine more packages were gated on the same 28 folds under rule 1, each against the model before it
and each registered before its score. Seven failed (section 4). Two passed:

| Package | Change, everything else as F | Against | Difference [95% CI] | Wins |
|---|---|---|---|---|
| `T1200` | The LightGBM member has 1,200 trees instead of 150 | F | +0.0041 [+0.0015, +0.0070] | 21/28 |
| `LT` | The TCN trains 4,800 steps instead of 1,200 and also reads the 53 touch columns | F with `T1200` | +0.0120 [+0.0082, +0.0160] | 21/28 |

- **Where the sizes came from.** An unregistered screen on the 8 IDSSE and Metrica folds read one LightGBM
  model at 150, 300, 600 and 1,200 trees, and the score was still rising at 1,200. Because those 8 folds chose
  the size, `T1200`'s rule added a condition: not worse on the 20 SkillCorner folds alone. The TCN's 4,800
  steps were fixed without a screen of step counts. Two variants were screened on the 8 folds, with and without
  the touch columns, and the better one went to the 28.
- **F3 is F with both changes.** One teammate's laptop (Apple silicon) ran the two gates. The pair was then
  run again on the other teammate's PC (Windows, Intel), under a registration written before any score on
  that machine:

| Folds | Noise | F3 | F | Difference [95% CI] | Wins |
|---|---|---|---|---|---|
| 28 | target-like | 0.6180 | 0.6019 | +0.0161 [+0.0118, +0.0207] | 21/28 |
| 28 | severe | 0.5840 | 0.5696 | +0.0144 [+0.0095, +0.0191] | 19/28 |
| 20 SkillCorner | target-like | 0.6103 | 0.6030 | +0.0073 [+0.0021, +0.0122] | 13/20 |
| 20 SkillCorner | severe | 0.5816 | 0.5804 | +0.0012 [-0.0044, +0.0064] | 11/20 |
| 8 IDSSE + Metrica | target-like | 0.6374 | 0.5991 | +0.0383 [+0.0288, +0.0469] | 8/8 |
| 8 IDSSE + Metrica | severe | 0.5900 | 0.5426 | +0.0474 [+0.0373, +0.0573] | 8/8 |

- **The two machines agree.** Both give 0.6180 and 0.5840 on the 28 folds, although their TCN weights differ
  in the last bits. The second run is a replication, not new evidence: the folds are the same.
- **Where the gain is.** Mostly on the 8 folds with a dead-ball label, where every fold gains. On SkillCorner
  F3 gains 0.007 at target-like noise and nothing at severe.
- **Every evaluation world gains.** F3 meets the adoption rule in the mild, target-like, smooth and coupled
  worlds. With identity switches (`target_ids`) it gains +0.0127 on 18 of 28 folds.
- **The TCN alone now scores above the blend** (0.6205 against 0.6180). We kept the blend at w = 0.5, fixed
  before any result, and tried no other weight.
- **Reproduction.** F3 trained twice on the PC gives the same model file. A dressed external match gets a
  byte-identical CSV on six OS and CPU combinations and with other torch kernels (section 7).
- **Adopted on Nov 12.** F3 passed its gates and its reproduction checks. It was the submission model
  until AW2 replaced it (section 3.5), and it is the model AW and AW2 are measured against.

### 3.5 After F3: SkillCorner labels from its action events, length weights, and a longer TCN (AW and AW2, 2026-11-12 to 13)

AW is F3 with other training labels on the 20 SkillCorner matches and with possession-length weights in
both team heads. AW2, the submitted model, is AW with a TCN trained 9,600 steps instead of 4,800. Features,
members, the LightGBM size, seeds, blend, decoder and priors are F3's, and no line of the inference path
changes. This section takes AW first, as it was chosen, and the longer TCN at its end.

**What led to it.**

- **Four changes in a row split the same way.** After F3, every package that helped at all (the length
  weights S1 and S2, the label-kind column K1 and K; section 4) helped the 8 IDSSE and Metrica folds, did
  nothing on the 20 SkillCorner folds, and failed rule 1 on folds won.
- **The points are in short possessions.** An error analysis of F3's cached folds found that short
  possessions after an opponent's are 26% of the true possessions on the IDSSE and Metrica folds, and F3
  finds 11% of them. Putting in the missed possessions under 4 s would add 0.194 there and 0.181 on
  SkillCorner. The true alive/dead pattern would add 0.024, and boundary timing, the decoder and the blend
  weight are each worth under 0.02.
- **A probe of the SkillCorner label (Nov 12; nothing fitted or scored).** SkillCorner's pinned repository
  ships, beside the per-frame `possession.group` that every model so far had trained on, a
  `*_dynamic_events.csv` per match with about 936 `player_possession` events (first touch to last, with
  team, start type and end type) and flags for game interruptions. Set side by side, pooled over the 20
  matches:

| Question | Answer |
|---|---|
| What is the null frame label (46% of in-play frames)? | 96% of it lies between a player possession flagged as followed by an interruption (throw-in, goal kick, corner, free kick, goal) and the next player possession. Half of those frames have no tracking: the broadcast cut away. Under 3% is a gap between possessions in open play |
| Does the frame label follow the last touch? | No. On labelled frames it agrees with the team of the most recent player possession on 82% |
| What does it do with a short possession of the other team? | It gives it to the team around it: 153 player possessions a match (17%), median 0.7 s, are labelled as the other team's |
| When does it switch? | A median of 1.0 s before the new team's first touch (25th to 75th percentile: 0.3 to 2.3 s) |

So the frame label is a phase-level one. It marks dead ball as unknown, swallows the other team's short
possessions, and changes before the other team touches the ball. Those rows were 72% of the team head's
training rows and 20 of the 28 gate folds, and on those folds the scorer cuts possessions at X, so they do
not test where a possession starts or ends, or dead ball. CHALLENGE.md describes another kind of ground
truth: dead ball is a label, every live frame has an owner, a loose ball stays with the last team, and the
owner changes when the other team gains control.

**The label change (candidate A).** The 20 SkillCorner matches are relabelled from the events
(`katena.external.skillcorner_actions`). No IDSSE or Metrica row changes.

- **Owner.** From the first frame of a player possession to the first frame of the next one in the same
  period, the ball is that player's team's. So it stays with the last team while it is loose or in flight.
- **Dead ball.** A gap between two player possessions holds a stoppage when SkillCorner flags the earlier
  one `game_interruption_after` or the later one `game_interruption_before`. The ball goes dead at the first
  gap frame where it is outside the pitch or the frame has no tracking, or 1.0 s after the last touch when
  neither happens, and at once after a foul. Play restarts 1.0 s before the next possession when that one
  starts as the reception or interception of the restart, and at its first frame otherwise. From the
  restart on, the ball is the taking team's.
- **Unknown (X).** Frames before a period's first player possession, and dead-ball frames with no tracking
  at all.
- **The two times of 1.0 s** were fixed in the registration, from what a throw-in or a pass takes. They
  were fitted to nothing.
- **Then as every other label:** the sandwich merge at T = 2 (section 6).
- **Training.** Both members exactly as F3's (rows, sequences, features, 1,200 trees, 4,800 steps, seeds),
  with two differences on SkillCorner rows only: the label, and that those rows now also train the alive
  head and the TCN's alive output. The decoder's priors are F3's.

| SkillCorner label, T = 2 | Home | Away | Dead | Unknown | Possessions a match | Median | Under 2 s | Under 4 s |
|---|---|---|---|---|---|---|---|---|
| Frame label (up to F3) | 0.275 | 0.266 | 0 | 0.459 | 272 | 7.2 s | 13.9% | 31.6% |
| Action label (A, AW) | 0.280 | 0.273 | 0.206 | 0.242 | 239 | 8.7 s | 9.3% | 25.0% |

The two labels agree on the team on 84.7% of the frames where both name one. In the development pool,
148,637 of 443,350 SkillCorner rows change label, the alive head's rows go from 171,452 to 507,549, and
the team head's from 334,270 to 339,490. A second implementation, written from the registration's text
by a reader who saw no model output, gives the same labels on all 20 matches, frame for frame.

**The weights (candidate AW).** AW is A plus the possession-length weights of section 2.3 in the LightGBM
team head and in the TCN's team loss, read from the label each frame trains on. These are the weights of
S1 and S2, with the same constants. On the frame label each had failed rule 1 on folds won, the two
together were worse than either on 28 folds, and the weights were closed (section 4). A's registration
said no other weighting would be tried. One teammate asked for this one before any score of A existed,
and AW's registration amends that sentence and says so. Its reason: the frame label has no short
possession of the kind the weight is meant to protect, and the action labels do. The alive heads carry no
weight, so AW's LightGBM alive probabilities must equal A's: checked on 28 of 28 folds.

**Two truths and two rules.** A SkillCorner fold now has two truths. The *frame* truth is the label every
gate so far scored on, with a predicted D read as no call. The *action* truth is the label above, scored
as an IDSSE fold is. An IDSSE or Metrica fold has one truth, and A does not change it.

- **Rule 1, the standing rule, unchanged.** Against F3 on the 28 folds, every SkillCorner fold on its
  frame truth: at target-like noise at least 70% of folds won and the paired 95% interval above 0; not
  worse at severe; not worse at target-like on the 20 SkillCorner folds. A's registration expected A to
  fail it, because on 20 folds it scores A against a convention A no longer learns.
- **Rule 2, declared in A's registration before any score of A.** The same thresholds on the 8 IDSSE and
  Metrica folds, whose truth A does not touch: at target-like at least 6 of 8 folds won and the interval
  above 0; not worse at severe on those 8; and on the 20 SkillCorner folds against the action truth, not
  worse than F3 at target-like.
- **Why a second rule.** The reasons are public documents, not scores. CHALLENGE.md describes the IDSSE and
  Metrica kind of label. EVALUATION.md says a prediction that is perfect but 1 s late scores 0.65, and the
  frame label changes a median of 1.0 s before the first touch. And five candidates in a row (S1, S2, K1,
  K, and an unregistered probe without SkillCorner in the team head) gained 0.017 to 0.038 on the 8 folds,
  each on 8 of 8, while the four that were gated failed rule 1 on folds won. A rule that cannot pass a
  change of that kind cannot judge one.
- **Rule 2 is a change of protocol and weaker evidence than rule 1.** Every package since Nov 6 has seen
  the 8 folds, and four changes passed this kind of test before and were not adopted (S1, S2, K1, K). A
  SkillCorner score against the action truth favours a model trained on it by construction: it is a guard
  and a description, not evidence that the model is better. The registration made rule 2 count only with
  both team members' names. Both gave them (PENDING_JUSTIN_NAME). One side agreed while no score of A or AW
  was known to it. The other had by then seen an interim score of A (its LightGBM member blended with F3's
  TCN: +0.0232 [+0.0144, +0.0323] on the 8 folds, 8 of 8).

**Four candidates.** Two more were registered on Nov 13, before any score of the four was known on the
side that registered them. A's registration had said no other variant would be tried; the new
registration says so and does not hide it. Its reason was a review of A's labels, which found no bug and
found their dead-ball ends weaker than their owner changes (section 7). AT and ATW keep the SkillCorner
rows out of the alive heads:

| Candidate | Team heads | Alive/dead training |
|---|---|---|
| A | Action labels on SkillCorner rows | SkillCorner rows included |
| AW | As A, with the length weights | As A |
| AT | As A | As F3: IDSSE and Metrica rows only |
| ATW | As AW | As F3 |

The registrations fix how one of them is chosen. Only candidates that pass the rule in force against F3
are eligible. Two paired tests follow, at target-like noise on that rule's folds, and each needs a 95%
interval above 0 to leave the simpler choice: first the weights inside each alive/dead design (AW over A,
ATW over AT), then the alive/dead training (the AT side over the A side). If none is eligible, F3 stands.

**The run.** The fold models of all four were fitted on GitHub's Ubuntu runners (`gate.yml`, four jobs of
seven folds per candidate, on AMD and Intel CPUs). No score is computed on a runner: the gates ran on the
Windows PC against F3's cached members. Two checks were made before a score was read. A's LightGBM member
from the runners equals the PC's own fit on 27 of 28 folds, hash and every probability; section 7 has the
28th. And A's LightGBM member with F3's TCN gives the other machine's interim number on the 8 folds to the
digit.

**A's gate on a second machine.** The PC then ran A's whole gate itself, as registered before any score of
A was known there. It gives the runners' verdicts under both rules and every mean within 0.0002, against
a registered limit of 0.002:

| A against F3, target-like | The PC (Windows, Intel i7-7700) | GitHub runners (Ubuntu, AMD EPYC 7763) |
|---|---|---|
| Rule 1: 28 folds, frame truth on SkillCorner | 0.6093, -0.0087 [-0.0149, -0.0024], 10/28 | 0.6092, -0.0088 [-0.0150, -0.0025], 10/28 |
| Rule 2: 8 IDSSE + Metrica folds | 0.6763, +0.0389 [+0.0276, +0.0498], 8/8 | the same |
| Rule 2: the same at severe noise | 0.6263, +0.0363 [+0.0254, +0.0470], 8/8 | the same |
| Rule 2: 20 SkillCorner folds, action truth | 0.6645, +0.0551 [+0.0461, +0.0639], 20/20 | 0.6647, +0.0553 [+0.0461, +0.0641], 20/20 |
| All 28 folds, action truth on SkillCorner | 0.6679 | 0.6680 |

27 of the 28 folds score the same on the two machines to the last digit, at both severities and on both
truths. The one that differs is the fold whose LightGBM team head grew other trees (section 7). This is a
replication, not new evidence: the folds are the same. AW, AT, ATW and the two 9,600-step candidates
below were fitted on the runners only. The results that follow are the runners'.

| spellF1 at target-like noise unless stated | F3 | A | AW | AT | ATW |
|---|---|---|---|---|---|
| Rule 1: 28 folds, frame truth on SkillCorner | 0.6180 | -0.0088, 10/28 | -0.0224, 8/28 | -0.0084, 10/28 | -0.0191, 8/28 |
| Rule 2: 8 IDSSE + Metrica folds | 0.6374 | 0.6763, +0.0389 [+0.0276, +0.0498], 8/8 | **0.6923, +0.0548 [+0.0399, +0.0697], 8/8** | 0.6725, +0.0351 [+0.0253, +0.0448], 8/8 | 0.6964, +0.0590 [+0.0448, +0.0733], 8/8 |
| Rule 2: the same at severe noise | 0.5900 | +0.0363, 8/8 | +0.0613, 6/8 | +0.0287, 7/8 | +0.0625, 6/8 |
| Rule 2: 20 SkillCorner folds, action truth | 0.6094 | 0.6647, +0.0553 [+0.0461, +0.0641], 20/20 | 0.6586, +0.0491 [+0.0387, +0.0597], 19/20 | 0.6669, +0.0575 [+0.0490, +0.0660], 20/20 | 0.6577, +0.0483 [+0.0386, +0.0585], 19/20 |
| All 28 folds, action truth on SkillCorner | 0.6174 | 0.6680 | 0.6682 | 0.6685 | 0.6688 |
| Meets rule 2 | | yes | yes | yes | yes |

All four fail rule 1, as A's registration expected: on the frame truth of SkillCorner they lose 0.026 to
0.053. The tests that say which one stands, on the 8 folds of rule 2:

| Step | Test | Difference [95% CI] | Wins | Result |
|---|---|---|---|---|
| 1. Weights, alive/dead as A | AW minus A | +0.0160 [+0.0041, +0.0282] | 5/8 | Interval above 0: AW |
| 1. Weights, alive/dead as F3 | ATW minus AT | +0.0239 [+0.0115, +0.0362] | 5/8 | Interval above 0: ATW |
| 2. Alive/dead training | ATW minus AW | +0.0041 [-0.0017, +0.0099] | 6/8 | Not above 0: AW stays |

So the registered procedure names AW. Leaving the SkillCorner rows out of the alive heads neither helps
nor hurts: AT minus A is -0.0038 [-0.0100, +0.0023] on the 8 folds and +0.0022 on SkillCorner against the
action truth, and alive/dead F1 on the 8 folds is 0.962 for F3 and for all four.

**By the process that made the truth** (target-like noise; predicted possessions per true one in brackets).
This is not part of any registered test.

| Truth | F3 | A | AW | AT | ATW |
|---|---|---|---|---|---|
| IDSSE (6 folds) | 0.6024 (0.68) | 0.6453 (0.73) | 0.6725 (0.90) | 0.6402 (0.73) | 0.6739 (0.90) |
| Metrica (2 folds) | 0.7427 (0.91) | 0.7692 (0.97) | 0.7517 (1.17) | 0.7695 (0.96) | 0.7638 (1.17) |
| SkillCorner, action truth (20 folds) | 0.6094 (0.89) | 0.6647 (1.02) | 0.6586 (1.24) | 0.6669 (1.00) | 0.6577 (1.19) |

| Labels at T = 2 | Mean possession | Under 2 s | Under 4 s |
|---|---|---|---|
| IDSSE | 9.6 s | 15.2% | 38.2% |
| Metrica | 12.7 s | 7.6% | 28.6% |
| SkillCorner, action labels | 13.7 s | 9.3% | 25.0% |

- **The step from F3 to any of the four is large on every truth with these mechanics:** +0.035 to +0.059
  on the 8 folds (every fold, for every candidate), +0.048 to +0.058 on SkillCorner against the action
  truth, +0.009 to +0.027 on the two Metrica folds.
- **Among the four there is nothing to choose on all 28 folds:** 0.6680 to 0.6688.
- **The weights move score from two kinds of truth to the third.** They gain on IDSSE (+0.027 for AW over
  A, +0.034 for ATW over AT) and lose on Metrica (-0.018 and -0.006) and on SkillCorner against the action
  truth (-0.006 and -0.009; at severe noise AW is worse than A there, -0.0106 [-0.0185, -0.0028]). IDSSE's
  labels hold about twice the share of possessions under 2 s. With the weights the models predict 0.90
  possessions per true one on IDSSE and 1.17 to 1.24 on the other two.
- **So the test that names AW rests on the six IDSSE folds.** AW minus A by fold: +0.060, +0.035, +0.033,
  +0.027, +0.010 and -0.001 on IDSSE; -0.009 and -0.026 on Metrica.

**Evidence from models that never trained on an action label.** Rule 2 rests on 8 folds that every
package has seen and whose noise is simulated. Two reads ask whether the label mechanics of IDSSE and
Metrica also do better on real broadcast tracking, once the truth has those mechanics. Both score the 20
SkillCorner folds against the action truth, which none of these models trained on. Each was written down
before it was read, as information that decides nothing.

| Model against F3, 20 SkillCorner folds | Action truth, target-like | Action truth, severe | Frame truth, target-like |
|---|---|---|---|
| K at kind 0: the two members with the label-kind column (section 4), read with the value of the IDSSE and Metrica rows | 0.6359 against 0.6094: +0.0265 [+0.0178, +0.0349], 18/20 | +0.0243 [+0.0165, +0.0320], 17/20 | -0.0326 [-0.0396, -0.0255], 0/20 |
| N: F3 with a LightGBM team head fitted on IDSSE and Metrica rows only, so it never saw a SkillCorner label of either kind | 0.6209 against 0.6094: +0.0115 [+0.0037, +0.0193], 15/20 | +0.0144 [+0.0079, +0.0212], 16/20 | -0.0328 [-0.0395, -0.0261], 0/20 |

- **Both are ahead of F3 on the action truth and behind it on every fold of the frame truth.** So on
  SkillCorner the choice of truth decides the verdict, and rule 1 scores 20 of its 28 folds on the truth
  the documentation does not describe.
- **K at kind 0 finds more of the short possessions.** At IoU 0.5 on the action truth it finds 12% of the
  true possessions under 2 s and 39% of those of 2 to 4 s, where F3 finds 5% and 22%.
- **For N it is the blend that gains.** Alone, N's LightGBM member is level with F3's (-0.0050 [-0.0136,
  +0.0036], 8/20), having lost 72% of its team rows.
- **K at kind 0 does not reopen K.** K failed the rule it was registered under (section 4), and this
  reading of it was chosen after K's scores were known.
- **The old weights split by member on this truth.** S1 and S2 (section 4), trained with weights taken
  from the frame label, were read the same way: S1 is ahead of F3 (+0.0083 [+0.0022, +0.0150], 14/20), S2
  is not (-0.0057 [-0.0123, +0.0011], 5/20). Neither is AW's member: AW's weights come from the action
  labels.

**What none of this shows.** That the action labels are right: SkillCorner makes its events automatically
from the same tracking, and a stoppage's two ends are approximations. That the provided match's ground
truth follows them. Section 7 lists what follows from that.

**The longer TCN (A2 and AW2).** The one change since F that had passed on both kinds of fold was more
fitting: 1,200 trees for 150, and a TCN at 4,800 steps for 1,200 (section 3.4). The step count had never
been screened beyond 4,800. A2 is A, and AW2 is AW, with 9,600 TCN steps in place of 4,800; the one-cycle
schedule spans the steps, and labels, weights, seed, threads and the LightGBM member are the base
candidate's. They were registered on Nov 13, before any score of either, as the 18th and 19th packages on
the same 28 folds. The registration tried no other step count and said none would follow.

- **Their rule, each against its own base.** F's rule on the 28 folds with the action truth on SkillCorner:
  at target-like noise at least 70% of folds won and the paired 95% interval above 0; not worse at severe.
  And not worse at target-like on the 8 IDSSE and Metrica folds, nor on the 20 SkillCorner folds. A2 can
  only replace A, and AW2 only AW.
- **Why this rule and not rule 2's.** The change is of the kind rule 1 was written for: it should help every
  kind of fold, so it has to win folds. The frame truth is not used, because the candidates no longer
  train on it.

| The longer TCN against its own base | A2 minus A | AW2 minus AW |
|---|---|---|
| **28 folds, action truth on SkillCorner, target-like (rule)** | +0.0067 [+0.0040, +0.0093], 23/28 | **+0.0071 [+0.0046, +0.0096], 20/28** |
| The same at severe noise (rule: not worse) | +0.0082 [+0.0050, +0.0114], 24/28 | +0.0077 [+0.0047, +0.0106], 24/28 |
| 8 IDSSE + Metrica folds, target-like (rule: not worse) | +0.0075 [+0.0034, +0.0120], 7/8 | +0.0070 [+0.0018, +0.0118], 6/8 |
| 20 SkillCorner folds, action truth, target-like (rule: not worse) | +0.0064 [+0.0031, +0.0095], 16/20 | +0.0071 [+0.0044, +0.0099], 14/20 |
| Verdict | Passes | Passes, with exactly the 20 folds the rule asks for |

| spellF1, target-like | F3 | A | A2 | AW | AW2 |
|---|---|---|---|---|---|
| 8 IDSSE + Metrica folds | 0.6374 | 0.6763 | 0.6838 | 0.6923 | **0.6993** |
| 20 SkillCorner folds, action truth | 0.6094 | 0.6647 | 0.6712 | 0.6586 | **0.6657** |
| All 28 folds, action truth on SkillCorner | 0.6174 | 0.6680 | 0.6748 | 0.6682 | **0.6753** |
| IDSSE (6 folds) | 0.6024 | 0.6453 | 0.6571 | 0.6725 | 0.6814 |
| Metrica (2 folds) | 0.7427 | 0.7692 | 0.7642 | 0.7517 | 0.7530 |

- **The gain is about 0.007 everywhere**, on the 8 folds and on SkillCorner alike, at both severities. It is
  the first change since the longer TCN of F3 to win folds of both kinds, and it is the same kind of change.
- **Against F3, AW2 stands at** +0.0618 on the 8 folds (8 of 8) and +0.0563 on SkillCorner against the
  action truth (18 of 20). A2 stands at +0.0464 (8 of 8) and +0.0617 (20 of 20).
- **Rule 1 was read for AW, not for AW2.** AW is -0.0224 against F3 on the 28 folds with the frame truth
  on SkillCorner (8 of 28). The longer TCN's rule does not read the frame truth.
- **The weights read the same at 9,600 steps** (information; the registered test was at 4,800). AW2 minus
  A2 is +0.0155 [+0.0032, +0.0275] on the 8 folds (6 of 8), -0.0055 [-0.0131, +0.0024] on SkillCorner
  against the action truth, and -0.0103 [-0.0183, -0.0025] there at severe noise. They gain on IDSSE and
  lose on the other two.
- **The longer TCN leaves the count of possessions alone:** 0.98 predicted per true one on the 8 folds and
  1.23 on SkillCorner for AW2, 0.80 and 1.02 for A2.

**Adopted.** AW met rule 2 and the registered tests chose it among the four. AW2 passed its own rule
against AW, so the longer TCN is part of it. Both team members put their names to rule 2 and to AW2 before
the cut-off of Nov 14, 18:00 CET (PENDING_JUSTIN_NAME). AW2 is the submitted model.

**Reproduction.** The all-30 models of A and of AW were each trained from an empty cache on macOS, Ubuntu
and Windows runners and on the reference PC: one LightGBM member and one CSV for the dressed external
match on the four machines. AW2's all-30 model was checked the same way. Section 7
has the hashes.

## 4. Assets, and what we tried and dropped

| Provided asset | Used? | How |
|---|---|---|
| Tactical ATD | Yes, the only model input | 5 Hz features, plus the native-rate ball for `touch` |
| Half boundaries (DATA.md, `games.half_video_blocks`) | Yes | Which frames are graded |
| Smart Tagging | Kickoff only | The `Kick Off (Start)` tags set the pre-kickoff frames of each half to D (section 6). Set-piece tag features (G2) were gated and not adopted (below) |
| Homography | No | Only the schema probe reads its header |
| Tactical video | No | We built no detector |
| Broadcast video, ATD and homography | No | Only the slot-layout check reads the broadcast ATD metadata. A "cutaways as a dead-ball prior" package was planned and has no gate result |
| Public 5-minute ground truth | Pipeline check only | Section 3.1 |

| Method tried | Result (28 folds unless stated) | Decision |
|---|---|---|
| HSMM decoder | -0.0007 vs Viterbi | Dropped |
| Spell rescoring | +0.0012, interval includes 0 | Dropped |
| TCN alone (D3) | +0.0036 [-0.0021, +0.0092], 13/28 | Ensemble member only |
| D3 blend weight chosen inside the folds | +0.0044, 14/28 | Did not rescue the gate |
| MIX: a random T per training row | -0.0001, 43% wins | Dropped |
| AUG: soft labels over T = 0-5 | -0.0050 [-0.0080, -0.0020], 21% | Dropped |
| H3.5: 3.5 s control-threshold labels | -0.0078 [-0.0118, -0.0038], 29% | Dropped |
| S2tail: T = 2, also merging touch-then-out flips | +0.0024 [-0.0003, +0.0051], 68% | Narrow miss; an option, not re-run after a later SkillCorner label fix |
| MR: Metrica stoppages derived back from restarts | +0.0010, 57% | Not adopted |
| S1: training at T = 1 | +0.0034 [+0.0005, +0.0062], 79% | Passed; not applied (below) |
| G1: the TCN member also reads the 53 touch columns | +0.0017 [-0.0014, +0.0047], 16/28 against F | Failed; not adopted. The TCN alone gains (+0.0041), but the blend barely moves |
| G2: set-piece tag features in the LightGBM member, imitation tags from external events | +0.0046 [+0.0014, +0.0079], 16/28 (57%) against F | Failed on wins and not adopted. The 8 tagged IDSSE + Metrica folds all gain (+0.023, 8/8); the 20 SkillCorner folds have no events, so no tags, and lose slightly |
| S1: possession-length weights (1 / length) in the LightGBM team head, on the SkillCorner frame label | +0.0053 [+0.0004, +0.0101], 17/28 against F3 | Failed on wins. All of the gain is on the 8 IDSSE + Metrica folds, which had been seen; the 20 SkillCorner folds lose slightly |
| S2: the same weights in the TCN's team loss | +0.0108 [+0.0051, +0.0165], 17/28 against F3 | Failed on wins, and worse at severe on SkillCorner. With S1 as well, -0.0140 against S1 on 28 folds (6/28). The weights were closed, and reopened once, in AW's registration, for the action labels (section 3.5) |
| K1: a label-kind column (1 on SkillCorner rows, 0 on IDSSE and Metrica rows) in the LightGBM team head | +0.0045 [+0.0013, +0.0079], 18/28 against F3 | Failed on wins. +0.0175 on the 8 IDSSE + Metrica folds (8/8), -0.0007 on the 20 SkillCorner folds. Read with the wrong value it loses 0.006 to 0.009 against F3 |
| K: the same column in the TCN too | +0.0086 [+0.0043, +0.0131], 17/28 against F3 | Failed on wins. +0.0345 on the 8 IDSSE + Metrica folds (8/8), -0.0017 on SkillCorner. Read with the wrong value it loses 0.033 to 0.045 against F3. Read at kind 0 on SkillCorner it is behind F3 on the frame truth and ahead of it on the action truth (section 3.5) |
| A: SkillCorner training labels from its action events (section 3.5) | Rule 2: 0.6763 on the 8 IDSSE + Metrica folds, +0.0389 [+0.0276, +0.0498], 8/8; 0.6647 on SkillCorner against the action truth, +0.0553 [+0.0461, +0.0641], 20/20. Rule 1: -0.0088, 10/28 | Met rule 2, failed rule 1. Not used: the registered test prefers AW (+0.0160 [+0.0041, +0.0282] on the 8 folds, 5/8) |
| AT: A's labels in the team heads only; alive/dead trained as in F3 | Rule 2: 0.6725, +0.0351 [+0.0253, +0.0448], 8/8; 0.6669 on SkillCorner against the action truth. Rule 1: -0.0084, 10/28 | Met rule 2, failed rule 1. Not ahead of A (-0.0038 [-0.0100, +0.0023] on the 8 folds) |
| ATW: AT with the length weights | Rule 2: 0.6964, +0.0590 [+0.0448, +0.0733], 8/8; 0.6577 on SkillCorner against the action truth. Rule 1: -0.0191, 8/28 | Met rule 2, failed rule 1. Not ahead of AW by the registered test (+0.0041 [-0.0017, +0.0099], 6/8) |
| A2: A with a TCN trained 9,600 steps instead of 4,800 | Against A, 28 folds with the action truth on SkillCorner: +0.0067 [+0.0040, +0.0093], 23/28. 0.6838 on the 8 IDSSE + Metrica folds, 0.6712 on SkillCorner against the action truth, 0.6748 on all 28 | Passed its rule. Not used: it can only replace A, and the registered test had chosen AW. The same step on AW is AW2, the submitted model (section 3.5) |
| B1: a LightGBM model of owner changes within 0.4 s, whose odds scale the Viterbi switch cost frame by frame | +0.0003 [-0.0020, +0.0028], 13/28 against F3 | Failed. The model is weak (mean average precision 0.22 at a base rate of 0.04) |
| No SkillCorner match in the LightGBM team head | 8 folds, not registered: +0.035, 8/8 | Not gated and not used. Its only folds had been seen, and it would drop the only training matches whose input is automatic broadcast tracking. Read later on the 20 SkillCorner folds as information (N, section 3.5) |
| A smooth-only training mix | 8 folds, not registered: +0.02 in the smooth worlds, -0.09 to -0.13 in the rough ones | Not used. Choosing it would need a statistic of the provided match |
| Decoder priors from another convention | Under 0.004 | No lever |
| PathCRF | Not run | Needs 22 identified players; its weights were trained on our IDSSE matches |
| StatsBomb open data | Not used | Its agreement asks for a name and email and bans redistribution |

T = 1 passed the gate, but we kept T = 2. T = 1 has a higher worst-case regret across conventions (0.011
against 0.009), and it loses against the 5 s and event-defined conventions, the two closest to the written
"controlled" definition. Its gain is about a tenth of the scorer's noise band.

## 5. Training data

| Dataset | Source and licence | Who made the labels, and how | Matches (dev + bench) | In-play frames (dev / bench) | Rate |
|---|---|---|---|---|---|
| IDSSE / DFL open data | figshare 10.6084/m9.figshare.28196177 (Bassek, Rein, Weber, Memmert, *Sci Data* 2025). CC BY 4.0; credit the authors, DFL and Sportec Solutions | DFL/TRACAB optical tracking, checked by live operators. Per-frame `BallStatus` 0 gives D; otherwise `BallPossession` gives H or A. The paper does not say how the possession field is derived | 7 (6 + `DFL-MAT-J03WQQ`); 2022/23 Bundesliga and 2. Bundesliga | 858,240 / 142,091 | 25 Hz |
| Metrica sample data | github.com/metrica-sports/sample-data, commit `e706dd5`. No licence stated; the README asks users to acknowledge the source. Not redistributed | None supplied. We derive them from Metrica's own event tagging. Owner: team of the last PASS, CARRY, RECOVERY or SET PIECE. D: from a dead trigger (ball out, foul, offside, hand ball, goal, shot out) to the next set piece or goal kick | 3 (Games 1 and 3 + `Sample_Game_2`); anonymised | 284,772 / 140,114 | 25 Hz |
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27`. MIT, (c) 2020 SkillCorner | SkillCorner's broadcast-video tracking. We derive the labels from the `player_possession` events of its `*_dynamic_events.csv`, which SkillCorner makes automatically from that tracking; no person labelled them. Owner: the team of the current or last player possession. D: inside a gap that SkillCorner flags as a game interruption (rule in sections 3.5 and 6). After the T = 2 merge: H 28.0%, A 27.3%, D 20.6%, X 24.2% of in-play frames. Up to F3 we trained on its per-frame `possession.group` (H or A; null, 46% of in-play frames, gives X; no D; how it is produced is not documented), and rule 1 still scores on it | 20, all development; A-League 2024/25 | 1,182,244 / none | 10 Hz |

- **No hand labels.** Nobody on the team labelled or corrected any frame of any game. Every label is a
  provider field or an algorithm over provider data, and none comes from the provided match.
- **Use.** The alive heads and the team heads use all three sources; the decoder priors use IDSSE and
  Metrica. In the development folds each model trains on the 27 other development matches. The final model
  trains once on all 30, benches included (`katena.pipeline.training_refs`): 657,133 LightGBM rows of 213
  columns, of which 549,880 train the alive head and 363,838 the team head.
- **Integrity.** All 111 raw files are SHA-256 pinned in `scripts/external_manifest.json`, the 20
  SkillCorner event files among them. `scripts/build_external.py` downloads and rebuilds the corpus, and
  the action labels are built from those raw files at training time (`katena.eval.actlabels`).
- **Frame counts** are frames inside each half's in-play block at the native rate, from the corpus
  sidecars (`n_inplay`).

## 6. Possession rules

We train on a per-frame owner from each provider: its own field (IDSSE), or a rule of ours over its events
(Metrica, SkillCorner; section 5). One harmonisation follows, the **T = 2 s sandwich merge**
(`katena.labels.sandwich`). An H or A segment shorter than 2 s, with the other team on both sides, is
relabelled to the surrounding team. It never crosses D or X. The scan runs left to right until nothing
changes.

| Situation | CHALLENGE.md | Our labels |
|---|---|---|
| Brief opponent touch, deflection, block or duel; the ball returns to the owner | Not a change | Merged if under 2 s; a change if 2 s or longer |
| Brief opponent touch, then the ball goes straight out | Not a change | **Kept as a flip.** The `tail` option would merge it; not adopted |
| Pass in flight, loose ball | Stays with the last team | The last team, in all three sources. (SkillCorner's frame label, used up to F3, gave X here, which split the spell) |
| Change of possession | When the other team gains control | IDSSE: when the provider's field changes. Metrica: at the other team's first PASS, CARRY, RECOVERY or SET PIECE. SkillCorner: at the first frame of the other team's first player possession, its first touch |
| Restart | There is a clear possessor | The restart spell is never merged, however short. On SkillCorner the taking team owns the ball from 1.0 s before the restart is received |
| Chain: H, then A 1 s, H 0.3 s, A 10 s | A gains control at the 10 s spell | Change placed at the 10 s spell |
| Ball out of play | D | D, from `BallStatus` (IDSSE), events (Metrica) or interruption flags (SkillCorner); never merged |
| Who owns a dead ball | Not asked | Not predicted |

**Why 2 s.** CHALLENGE.md calls "controlled" ambiguous, and external data cannot tell what the corrected
tracking does. We trained at T = 0, 1, 2, 3 and 5 s and evaluated against each, and T = 2 s had the
smallest worst-case regret (0.017 on IDSSE). DFL's events suggest an opponent flip becomes more likely
than not to be deliberate at about 3.5 s. That describes control in DFL data, but labels built on it lost
at the gate (H3.5). The convention matters a lot: the same M2 predictions score 0.495 to 0.653 across the
conventions.

**SkillCorner labels from its action events (new in AW).** Up to F3 the SkillCorner matches trained on
SkillCorner's per-frame `possession.group`. That label has no dead ball, is null on 46% of in-play frames,
gives 17% of player possessions to the team around them, and changes owner a median of 1.0 s before the
new team's first touch (section 3.5). None of that is in CHALLENGE.md's description. AW derives the label
from SkillCorner's `player_possession` events instead (`katena.external.skillcorner_actions`):

- **Owner.** From the first frame of a player possession to the first frame of the next one in the same
  period, the ball is that player's team's. A possession needs control: SkillCorner's specification says
  it runs from the player's first touch to his last, and that a block does not count.
- **Dead ball.** A gap that SkillCorner flags as a game interruption (throw-in, goal kick, corner, free
  kick, goal) is D from the first frame where the ball is outside the pitch or the frame has no tracking,
  else from 1.0 s after the last touch, and at once after a foul. It ends 1.0 s before the next possession
  when that one receives or intercepts the restart, and at its first frame otherwise.
- **Unknown.** Frames before a period's first player possession, and dead-ball frames with no tracking at
  all, are X. "No tracking" as a sign of dead ball belongs to a broadcast feed, not to the provided
  match's tactical camera.
- **Then the T = 2 merge**, as for every other label. The two 1.0 s constants were fixed before any score.

**The output.** The decoder applies no explicit 2 s rule; it learns transitions from the labels. F3
under-segmented, with 0.751 predicted possessions per true one at target-like noise on the 28 folds of
rule 1 (F: 0.738). AW2 predicts 0.98 possessions per true one on the 8 IDSSE and Metrica folds and 1.23 on
SkillCorner against the action truth, where F3 predicts 0.73 and 0.89. By dataset the log has AW's
figures, at 4,800 steps: 0.90 on IDSSE, 1.17 on Metrica and 1.24 on SkillCorner, where F3 predicts 0.68,
0.91 and 0.89 (section 3.5). The output never has X on a graded frame: the decoder has three states, H, A
and D.

**Second-half kickoff.** The second half is graded from frame 110,808 but kicks off later (FAQ.md), and
every frame in between is D. We take the kickoff from the provided Smart Tagging, Metrica's automatic
annotation, never from a frame constant. In each graded half, the frames before that half's single
`Kick Off (Start)` tag are set to D (`katena.pipeline.apply_kickoff_rule`). The tag's start time maps to a
frame by DATA.md's `round(seconds * 25)`. Nothing changes when there is no such tag, when there are several,
or when the tag lies outside the half or more than 180 s into it.

`scripts/check_kickoff_tags.py`, run by a human, prints pass, warn or fail only. It fails only when a tag
lies more than 2 s after the published kickoff, because then the rule would turn play into D. An early tag
only warns: the rule then sets fewer frames to D, which is safe. `predict`'s report also prints the share of
D over the first 60 and 120 s of each half, as an aggregate.

## 7. Limitations

- **The new labels are a bet on the documentation.** Rule 2 was chosen for what public documents say, not
  for a score.
  CHALLENGE.md describes a ground truth with a dead-ball label, an owner on every live frame, a loose ball
  that stays with the last team, and a change of owner when the other team gains control. Metrica's
  published event definitions and SkillCorner's event specification agree with it on when the owner
  changes. That is the IDSSE and Metrica kind of label, and the action labels are built to have it;
  SkillCorner's frame label is the one that does not. But no document says how the corrected tracking
  behind the ground truth times a change to the frame. If the provided match's ground truth behaves as
  SkillCorner's frame label does, the bet loses. Under rule 1 AW, at 4,800 TCN steps, is -0.0224 against
  F3 on the 28 folds with 8 folds won, and on the frame truth of SkillCorner the four candidates lose 0.026
  to 0.053. AW2 was not read under rule 1: the longer TCN's rule does not use the frame truth.
- **The action labels are our construction, and their dead-ball ends are approximate.** SkillCorner makes
  its events automatically from the same broadcast tracking; no person labelled them. Of 1,851 flagged
  stoppages, 228 take their start from the broadcast cutting away, a median of 6.5 to 9.1 s after the last
  touch, depending on the kind of stoppage; 138 of them are free kicks, where the ball never leaves the
  pitch. Placing those starts 1.0 s after the last touch instead would turn 18,267 frames (2.8% of the
  team-labelled ones) from a team into dead ball. Restarts sit 1.0 s before the reception, by a constant;
  that differs by more than 0.5 s from the start of SkillCorner's own following phase
  (`*_phases_of_play.csv`) on 51% of received restarts. In AW and AW2 these rows are 66% of the alive
  head's rows in the development pool (336,097 of 507,549). The two
  candidates that keep them out of the alive heads (AT, ATW) were not ahead, so the late starts did not
  show in the score. And a SkillCorner score against the action truth favours a model trained on it: it is
  a guard and a description, not a second estimate.
- **The weights help one kind of truth.** AW is ahead of A, the same model without the weights, on the 8
  folds (+0.0160 [+0.0041, +0.0282], 5 of 8). All of that comes from the six IDSSE folds (0.6725 against
  0.6453). On the two Metrica folds AW is behind A (0.7517 against 0.7692), and on SkillCorner against the
  action truth too (0.6586 against 0.6647; at severe noise -0.0106 [-0.0185, -0.0028]). On all 28 folds the
  two are level (0.6682 and 0.6680). With the weights the model predicts more possessions than there are
  on two of the three truths (1.17 and 1.24 per true one). The provided match's ground truth comes from
  Metrica's corrected tracking, and A is ahead of AW on both kinds of truth that are not IDSSE's. Two
  Metrica folds are thin evidence, and the action truth is our own construction. The registered test
  names AW. With the longer TCN the picture is the same (information; the registered test was at 4,800
  steps). AW2 minus A2 is +0.0155 [+0.0032, +0.0275] on the 8 folds (6 of 8): 0.6814 against 0.6571 on
  IDSSE, 0.7530 against 0.7642 on Metrica. On SkillCorner against the action truth it is -0.0055 [-0.0131,
  +0.0024] (0.6657 against 0.6712), and -0.0103 [-0.0183, -0.0025] at severe noise. On all 28 folds AW2
  scores 0.6753 and A2 0.6748. AW2 predicts 0.98 possessions per true one on the 8 folds and 1.23 on
  SkillCorner, A2 0.80 and 1.02.
- **Rule 2 is weaker evidence than rule 1.** It decides on 8 folds that every package since Nov 6 has
  seen, and whose noise is simulated. Four earlier changes passed a test of this kind and were not adopted
  (S1, S2, K1, K). AW was also chosen among four related candidates on those 8 folds; the two registered
  tests are there to keep that choice honest. The two reads of section 3.5 on real broadcast tracking
  support the label mechanics, not AW itself: neither model in them trained on an action label. And AW2
  passed its own rule with exactly the 20 of 28 folds it asks for.
- **AW and AW2 were fitted on one kind of machine.** The fold models of AW, AT, ATW, A2 and AW2 were
  fitted on GitHub's Ubuntu runners only, and gated on the Windows PC against F3's cached members. F3's
  gate was repeated on a second machine before its adoption; theirs were not. A's was (section 3.5): the
  PC gives the runners' verdicts and every mean within 0.0002, and 27 of 28 folds score the same to the
  last digit. The one fold whose LightGBM team head differs (below) moves by 0.0018 at target-like noise
  on the frame truth and by 0.0036 on the action truth. So we read the runners' numbers as stable to the
  fourth decimal unless a LightGBM fold flips.
- **Simulated noise is not the real ATD.** A classifier separates real SkillCorner and Metrica tracking
  from every degraded world with AUC 1.00, mostly on jitter. At 5 Hz the per-slot second difference is
  0.04-0.05 m in real data and 0.47-1.38 m in the degrader. At 25 Hz the ball's median absolute second
  difference is 0.001 m on Metrica's samples and about 1 m in the target-like world. The `_smooth` worlds
  are one world in five of the mix (one in three for SkillCorner). At inference, `touch` reads the real
  25 Hz ball, which no development world reproduces.
- **SkillCorner is 20 of the 28 folds, and our only real broadcast tracking.** Under rule 1 its truth is
  the frame label, which has no dead ball and whose null possession splits its spells. Under rule 2 it is
  the action truth, which is ours (above). At 10 Hz the touch features get two native frames per sample.
  Its "target-like" world keeps 15.3 tracked objects, not 20.5. Against M2, five of F's six losing folds
  are SkillCorner, and the worst loses 0.0181. Against F, all seven of F3's losing folds are SkillCorner.
- **Metrica's dead spans run long.** The event rule keeps D through live play after an untagged restart.
  Metrica gave about a quarter of the alive head's rows up to F3; in AW and AW2 the SkillCorner rows are
  the larger part (above). The cleaner rule (MR) was neutral.
- **Pooled calibration.** The D2 coupling constants come from Metrica Games 1 and 3, and the smooth jitter
  from the 20 SkillCorner matches, all development folds. So a held-out match helps set a few scalars of
  its own training worlds. The first degrader calibration also used every SkillCorner and Metrica match,
  `Sample_Game_2` included. The effect is probably small but unmeasured.
- **The development estimate is optimistic.** D1, D2 and D3 were gated on the same 28 folds before F
  combined them. F was registered before its own score, but its parts were chosen knowing their results.
  Their gains (+0.0069, +0.0065, +0.0019) sum to +0.0153, against F's +0.0158. After F, nine more packages
  were gated on the same folds (sections 3.4 and 4), each registered before its score: thirteen in all, besides
  the bake-off's models and the six label-convention variants of section 4, which were scored on these folds
  too. A, AW, AT and ATW were the 14th to 17th packages on them, and A2 and AW2 the 18th and 19th
  (section 3.5): nineteen gated packages in all. The two
  locked benches are the only matches no choice was made on. F's one read there gains about as much as on
  the development folds (section 3.1). With two matches, that read is a sanity check, not a second
  estimate, and nothing after F was read on them.
- **The TCN seed matters.** F's TCN uses seed 0, fixed in advance. Averaging in a second seed lowered the
  8-fold score by 0.005 (1 of 8 folds won), so seed 0 does unusually well on those folds. With seed 1, F is
  still about +0.02 above M2 there, so roughly a quarter of the 8-fold gain depends on the seed. The 28-fold
  gate is the better guide. Both tests were run on F's 1,200-step TCN and were not repeated for F3's,
  AW's or AW2's, which keep seed 0.
- **Unknown label convention.** The absolute score is uncertain by about 0.15. About 35 touch-then-out
  flips per IDSSE match (11% of spells) remain in our labels.
- **Assumed identity-switch rates.** No external source measures them.
- **Short possessions.** At IoU 0.5 and target-like noise, M2 matched 0.167 of true possessions under
  2 s, and M2 with touch features 0.169. F3 finds 7% of the true possessions under 2 s on the IDSSE and
  Metrica folds and 24% on the SkillCorner folds (frame truth), and 91% and 85% of those over 16 s. It
  finds 11% of the short possessions that follow an opponent's on the IDSSE and Metrica folds
  (section 3.5). The count by length was not repeated for AW or AW2.
- **Model bytes differ across machines; the labels did not.** On the reference machine the rebuild is exact.
  A fresh clone, with a fresh download of the external corpus and an empty cache, reproduced every stage byte
  for byte: corpus, features, worlds, pools, both models and a dressed-match CSV (`docs/decision-log.md`, J3 W1).
  Across machines we trained F 11 times on 7 OS and CPU combinations (`.github/workflows/repro.yml`):
  - Windows 10 on Intel (the reference machine), Windows Server on Intel and on AMD, Ubuntu on two AMD models
    and on Intel, macOS on Apple M1.
  - Every training input is equal in content everywhere. On macOS only NaN sign bits differ, and nothing reads them.
  - The LightGBM model, the scaler and the decoder priors are identical everywhere.
  - The TCN's weights differ in their last bits whenever the OS or the CPU vendor differs, because torch picks
    different kernels. Windows on another Intel CPU reproduced the reference machine bit for bit, in two runs.
  - The dressed external match (IDSSE `DFL-MAT-J03WMX`, 173,528 frames, in the provided ATD format) got a
    byte-identical CSV in all 11 trainings.

  F3 (section 3.4) was checked the same way: seven combinations (Windows 10 and Windows Server on Intel,
  Windows Server on AMD, Ubuntu on three AMD models, macOS on Apple M1), four different sets of TCN weights,
  one CSV; and the same LightGBM model on an eighth machine. GitHub's Ubuntu runners never drew an Intel CPU
  for F3, so that pair was run under WSL on the reference PC, with Ubuntu's libraries on its Intel CPU: a
  fifth set of TCN weights, and the same CSV.

  AW (section 3.5) was trained on all 30 matches on four machines. Three were GitHub runners, each from an
  empty corpus and cache: macOS 15 on Apple M1, Ubuntu 24.04 on Intel Xeon 6973P-C, and Windows Server
  2025 on Intel Xeon Platinum 8573C. The fourth was the reference PC.
  - The action labels are built from the pinned raw files on each runner, and the label counts in the three
    manifests agree.
  - The LightGBM member is `55de693c7ea2...` on all four. Its alive head (`59bfcd31dc1e...`) is also the
    alive head of A: the weights touch the team head only.
  - The TCN's weights are three sets, as for F and F3. The PC's (`7f21c128504d...`) are the Windows
    runner's, whose CPU is also Intel, and so is its `bundle.pkl` (`33bae152771c...`), byte for byte.
  - The dressed external match got one CSV, `280060af276f1665...`, on all four. That match is one of the
    30 training matches here, so its score says nothing and is not quoted.
  - A, AW without the weights, was checked the same way, on three runners of its own (macOS 15 on Apple M1,
    Ubuntu 24.04 on AMD EPYC 9V45, Windows Server 2025 on AMD EPYC 7763) and on the PC: LightGBM
    `eb1030c8ec94...` and dressed CSV `98b2919621653c99...` on all four.

  AW2, the submitted model, differs from AW in the TCN's steps only. Its all-30 model was checked on the
  same four kinds of machine, and they gave one LightGBM member and one dressed CSV.
  Expected on the reference PC (`docs/repro/expected-windows.json`): LightGBM `55de693c7ea2...`, TCN
  `78a241f571ad...`, bundle `9b4c2cfaa553...`; the dressed match's CSV
  `4e0ab1d06ba7...`.

  **LightGBM is not byte-identical on every machine for every training set.** In the gate of section 3.5,
  A's LightGBM member fitted on the Linux runners equals the PC's own fit on 27 of 28 folds, hash and every
  probability. On the fold that holds out `skillcorner/2010085`, the alive head is identical and the team
  head is not: the two machines grew different trees, and its probabilities differ by 0.008 on average and
  by up to 0.15. That moves the fold's score by 0.0018 at target-like noise on the frame truth and by
  0.0036 on the action truth, which is the size to keep in mind for a reproduction of the final file on
  another machine. F and F3 had not shown this on any machine we tried. The fold models of AW and AW2 were
  fitted on the runners only, so they have no such comparison. The all-30 models of A and of AW did not
  show it on the four machines each was trained on.

  So expect the same file on another machine. That has been shown on an external match in the provided format,
  not on the provided match itself, where a near-tie could still flip a label; and a fold has shown that
  LightGBM itself can differ between two machines.

**With more time** we would recalibrate the D2 worlds inside each fold, and make native-ball noise as
smooth and correlated as real tracking. We would check the action labels' dead-ball ends against a second
source and repeat AW2's gate on a second machine. We would also re-run the `tail` gate after its label
fix, and test more of Smart Tagging than the set-piece tags, which failed their gate (section 4).

## 8. Reproduction

One command, from a clone and the data bundle (`docs/REPRODUCE.md` has the details):

```bash
git clone https://github.com/jusstinn/katenaball && cd katenaball
git checkout <submission_tag>                    # the tag named in the short document
uv sync --frozen                                 # Python 3.11, dependencies pinned in uv.lock
export MTAH_DATA_DIR=/path/to/data/bundle        # the unpacked competition bundle (default: data/)
make submit TEAM=<team_name>                     # writes submissions/<team_name>.csv, validates it, prints its sha256
```

`make submit` runs six steps:
- `scripts/make_submission.py bundle` checks that the bundle's three files exist, before anything else;
- `scripts/build_external.py` downloads the 30 external matches and checks every file's SHA-256;
- `scripts/make_submission.py corpus` builds the base features;
- `scripts/make_submission.py worlds` builds the `d2j` worlds, touch columns and pools;
- `scripts/make_submission.py train` builds the action labels of the 20 SkillCorner matches from their
  pinned event files, trains the two members and saves `bundle.pkl` with its hashes;
- `scripts/make_submission.py predict` labels the provided match, writes the CSV and runs the official checks
  on the exact bytes written.

Without `make` (the team's Windows PC), run the same steps one by one; `docs/REPRODUCE.md` lists them.
`predict` is the only step that opens the provided match, and it prints aggregates only.

- **Pinned stack:** Python 3.11, LightGBM 4.7.0, torch 2.15.0 (run on the CPU on every OS), numpy 2.2.6,
  pandas 2.3.3, numba 0.68.1, scipy 1.17.1, scikit-learn 1.9.1, locked in `uv.lock`.
- **Needed on Linux:** the system's OpenMP runtime, `libgomp.so.1` (`apt-get install libgomp1`). Slim container
  images lack it; `make submit` then stops at its first step and names the package.
- **Checked:** on the reference machine, a fresh clone with an empty cache rebuilt every file and F's
  model byte for byte in 1 h 39 min, including the download. F3 changes only the training step, and F3
  trained twice there gives the same model file. AW and AW2 also change only the training step. AW's
  all-30 model gave one LightGBM member and one dressed CSV on three GitHub runners, each from an empty
  cache, and on the reference machine. Section 7 has that check, AW2's, and the hashes expected on the
  reference machine.
- **Determinism:** fixed seeds, 8 LightGBM threads, 4 torch threads, and torch's CPU kernels pinned before
  import (`ensemble.pin_cpu_env`). The CSV is built as bytes with LF line ends on every OS.
- **Cost:**

  | Machine | Cache build (corpus, worlds) | Train | Predict | Disk and memory |
  |---|---|---|---|---|
  | Windows 10, i7-7700, CPU only, 2 workers | 18 + 34 min | 23 min with another job running (LightGBM 3, TCN 20) | about a minute | 19 GB cache |
  | Apple-silicon laptop, 3 workers | 14 min | 9 min (LightGBM 0.5, TCN 8) | under a minute | |
  | GitHub runners: Ubuntu, Windows, macOS | | 8, 13 and 37 min | under a minute | 5.7 GB peak in training (Ubuntu) |

  The external download (about 5 GB) is extra. The training times in the table are F3's. On the reference
  PC `train` took 15 minutes for AW, at 4,800 TCN steps. AW2's TCN trains twice as long: `train` took 26 minutes
  there.
- **The submitted file:** its SHA-256 and its commit are in the short document.
- **No match data in the repository.** CI fails any commit that holds it (`docs/ci.md`).

## 9. How this was built

| Part | Where |
|---|---|
| Evidence and decisions | `docs/decision-log.md`, `docs/bakeoff.md`, `docs/bakeoff-protocol.md`, `docs/label-convention.md`, `docs/external-data.md`, `docs/audit/d5-bakeoff-audit.md` |
| External loaders and corpus | `src/katena/external/`, `scripts/build_external.py`, `scripts/external_manifest.json` |
| Label convention | `src/katena/labels.py`; `T_MAIN` in `src/katena/eval/protocol.py` |
| SkillCorner action labels and length weights (AW, AW2) | `src/katena/external/skillcorner_actions.py`, `src/katena/eval/actlabels.py`, `src/katena/models/lenweight.py`; spec `AW2` in `src/katena/pipeline.py`; `tests/test_skillcorner_actions.py`, `tests/test_lenweight.py` |
| Degrader and worlds | `src/katena/sim/degrade.py`, `src/katena/sim/worlds.py`, `scripts/calibrate_degrader.py`, `scripts/d2_calibrate.py` |
| Features | `src/katena/featurize/build.py`, `src/katena/featurize/touch.py` |
| Models and blend | `src/katena/models/gbdt.py`, `tcn.py`, `ensemble.py` |
| Decoder | `src/katena/decode/hmm.py` |
| Gates | `src/katena/eval/` (`lomo`, `bootstrap`, `score`, `d2`, `d3`, `touch_gate`, `tmix`, `final`), `scripts/run_*.py`; AW and its sister candidates: `scripts/run_a.py`, `run_aw.py`, `run_at.py`; the longer TCN: `scripts/run_x2.py`; all fitted on runners by `.github/workflows/gate.yml`; the label probe and the two reads of section 3.5: `scripts/probe_sc_actions.py`, `scripts/probe_transfer.py`, `scripts/read_act.py` |
| Target reader and checks | `src/katena/target/`, `scripts/check_target.py`, `check_target_layout.py`, `check_touch_target.py`, `check_kickoff_tags.py`, `check_domain_gap.py` |
| Submission | `src/katena/pipeline.py`, `scripts/make_submission.py`, `Makefile`, `docs/REPRODUCE.md`; the real run: `scripts/real_run.py`, `docs/real-run.md`; checks: `scripts/dressed_check.py`, `scripts/stress_predict.py`, `scripts/check_public_window.py`, `tests/test_pipeline_atd.py`; this document as a PDF: `scripts/build_document.py` |
| Firewall and CI | `scripts/firewall_check.py`, `.github/workflows/`, `docs/ci.md` |

## 10. Disclosures

- **AI assistance.** Most of the code and of this document was written with a hosted AI coding assistant
  (Claude Code), working under the team's direction. The competition's data was kept away from it. The
  repository's rules (`CLAUDE.md`) forbid it to open the data folder, only team members ran anything on the
  provided match, and what those runs print is pass or fail and aggregates: counts, shares and hashes, never
  a file, a frame or a per-frame value. One exception in words only: the team member who watched the public
  5 minutes (section 3.1) described to it what he saw, in general statements and with estimated times. It
  saw the public label file, which is in the repository, and no image and no other provided file.
- **A public file we opened and did not use.** While searching the web for external datasets, an automated
  agent opened a public record that turned out to hold statistics about the provided game. The file was
  deleted. Nothing from it entered the code, the training data or any decision.
- **StatsBomb open data.** 35 matches were downloaded to a temporary folder before we read the user
  agreement. We did not use them: they are in no training set, result or commit.
