# <team_name>: technical document

More than a Hack 2027, Phase 1. Repository: https://github.com/jusstinn/katenaball. Draft of 2026-11-11.
Every number comes from `docs/decision-log.md` and the evidence pages it
links. `TODO(after the real run)` marks what only the final run on the provided match can fill in.

## 1. Summary

- We label each frame H, A or D from the automatic tracking (ATD) alone. A LightGBM model and a temporal
  convolutional network (TCN) each predict whether the ball is alive and which team has it. We average
  them in logit space and decode with a 3-state Viterbi.
- Models are trained and selected on public external matches only (IDSSE/DFL, Metrica, SkillCorner): 28
  for development, and all 30 for the final model. A degrader corrupts their tracking to look like
  automatic tracking.
- No model was trained, tuned or selected on the provided match. Nobody labelled any frame by hand.
- The submitted model, F3, came from two gated steps, each under a rule registered before its score, on
  28 leave-one-match-out development folds at target-like noise. F beat the previous baseline, M2: spellF1
  0.6019 against 0.5861, +0.0158 [+0.0111, +0.0205], 22 of 28 folds won. F3 is F with a larger LightGBM
  model and a longer-trained TCN, and it beat F: 0.6180 against 0.6019, +0.0161 [+0.0118, +0.0207], 21 of
  28 folds won. These are development estimates, not a prediction of the hidden score.

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
0.2-1 s windows, ball speed and flight. Only the LightGBM member gets these columns (213 in total).

### 2.3 Models and decoder

- **LightGBM heads** (`katena.models.gbdt`). The *alive* head trains on IDSSE and Metrica rows only,
  because SkillCorner has no dead-ball label. The *team* head ("the focal team has the ball") trains on H
  and A rows from all three sources. The parameters were fixed before the bake-off and never tuned, apart
  from the number of trees: 1,200 (F had 150; section 3.4). Learning rate 0.1, 31 leaves, at least 50 rows
  per leaf, feature and bagging fractions 0.5, `max_bin` 63, seed 0, deterministic, 8 threads.
- **TCN** (`katena.models.tcn`). Six residual blocks of 48 channels with dilations 1 to 32, looking both
  ways in time, with the same two heads. It sees all 213 columns and trains for 4,800 AdamW steps on
  random 256-sample windows, with seed 0, 4 threads and deterministic algorithms (F: the 160 base columns
  and 1,200 steps; section 3.4).
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

F trains on the `d2j` mix. Each severity's rows are split evenly over five worlds for IDSSE and Metrica
(`s`, `s_cpl50`, `s_cpl100`, `s_ids`, `s_smooth`) and over three for SkillCorner, which has no D (`s`,
`s_ids`, `s_smooth`). LightGBM sees every 4th 5 Hz sample, with a random focal orientation per row. The
TCN sees one whole-match sequence per match and world. For the `touch` columns, each world's ball
corruption is carried over to the native frames (`touch.lift_ball`), consistently with the 5 Hz ball.

## 3. Why this design

### 3.1 How we validated

- **Leave-one-match-out on 28 external development matches** (6 IDSSE, 2 Metrica, 20 SkillCorner). In
  fold k, no model, scaler or prior sees match k. Scores come from the official
  `possession_metrics.evaluate`, unchanged. On SkillCorner, which has no D, a predicted D is scored as X.
- **A pre-registered gate.** A change is adopted if, at target-like noise, it wins on at least 70% of
  folds and the paired 95% bootstrap interval of the mean difference lies above 0. The bootstrap resamples
  ground-truth possessions in each match, 2,000 draws, seed 0. The rule comes from `docs/PLAN.md`. D2 and
  F were registered in the decision log before any result, and D3's blend weight was fixed before any
  result. We report the 8 IDSSE and Metrica folds beside the 28, because the 20 SkillCorner folds cannot
  show a dead-ball change.
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

  F is ahead in 15 of the 16 bench-and-world pairs; the exception loses 0.003. M2 retrained here
  reproduces the bake-off's recorded bench scores to the 3 decimals printed. The final model also trains
  on these two matches, so this read is of the 28-match model, not the submitted one.
- **The provided match is inference only.** No training, tuning, self-training or fitting of the degrader
  on it, which is also what the organisers' clarification of 2026-11-03 requires. We used only DATA.md's
  published aggregates (46 slots, 20.5 tracked objects, 41% ball missing) and the published half
  boundaries. Human-run checks on the real file (loader, slot layout, domain-gap alarms) print aggregates
  only and never choose between versions. Nobody compared our output with the video.
- **The public 5 minutes are a pipeline check only.** CI reads them only to check that the official scorer
  still reproduces the published scores of the published baseline files. The final file is checked on the
  window by `scripts/check_public_window.py`, which prints pass or fail only, never a score. It checks four
  things: the official file checks pass; swapping H and A lowers team accuracy (home and away are the right
  way round); shifting the labels by 5 or 10 s lowers spellF1 (the frames line up); and the file beats the
  published smooth50 baseline. A fail means a bug to find, not something to tune, and every run is logged.
  `TODO(after the real run)`: the result of that run.
- **One look at the public labels' structure, with no prediction and no score.** On Nov 9
  `scripts/probe_label_structure.py` counted the window's possessions and their lengths (17 possessions,
  median 7.6 s) and placed them among 5-minute windows of IDSSE labels under each label convention. The
  question was whether T = 2 looks implausible. It does not: the counts sit in the middle of the T = 2
  windows, and 17 possessions cannot separate T = 2 from T = 3. Nothing was changed because of this look.
- **The submission path equals the evaluation path.** We exported an external development match
  (IDSSE `DFL-MAT-J03WMX`) in the provided ATD file format and read it back bit-identical through the real
  loader (`katena.target.atd.load_target`). We trained F without it and labelled it with
  `make_submission.py predict`. Scored with the official scorer, the CSV gives spellF1 0.650611 and
  macroF1 0.930641, exactly what the evaluation code gives for the same model. Every probability and graded
  label is bit-identical on both paths. The same check with F3 gives 0.722130 and 0.941533 on both paths. A SkillCorner match (10 Hz) also agrees, once the evaluation's
  "predicted D counts as X" rule for SkillCorner is applied (`scripts/dressed_check.py`). A synthetic
  bundle on the provided match's own timeline (199,319 frames, the graded halves of `games.py`) runs
  through the same path in CI, and the official validator accepts its CSV (`tests/test_pipeline_atd.py`).

### 3.2 Bake-off (2026-11-06)

Development folds, target-like noise, T = 2 s labels.

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

After F, six more packages were gated on the same 28 folds, each against the model before it and each
registered before its score. Four failed (section 4). Two passed:

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
- **Adopted.** F3 passed its gates and its reproduction checks, and it is the submitted model.

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
| Possession-length weights (1 / length) in the LightGBM team head | +0.0053 [+0.0004, +0.0101], 17/28 against F3 | Failed on wins. All of the gain is on the 8 IDSSE + Metrica folds, which had been seen; the 20 SkillCorner folds lose slightly |
| The same weights in the TCN's team loss | +0.0108 [+0.0051, +0.0165], 17/28 against F3 | Failed on wins, and worse at severe on SkillCorner |
| No SkillCorner match in the LightGBM team head | 8 folds, not registered: +0.035, 8/8 | Not gated and not used. Its only folds had been seen, and it would drop the only training matches whose input is automatic broadcast tracking |
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
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27`. MIT, (c) 2020 SkillCorner | SkillCorner's broadcast-video tracking. Per-frame `possession.group` gives H or A; null gives X (46% of in-play frames). No D. How the group is produced is not documented | 20, all development; A-League 2024/25 | 1,182,244 / none | 10 Hz |

- **No hand labels.** Nobody on the team labelled or corrected any frame of any game. Every label is a
  provider field or an algorithm over provider data, and none comes from the provided match.
- **Use.** The alive head and the decoder priors use IDSSE and Metrica. The team head uses all three. In
  the development folds each model trains on the 27 other development matches. The final model trains
  once on all 30, benches included (`katena.pipeline.training_refs`): 657,133 LightGBM rows of 213 columns.
- **Integrity.** All 111 raw files are SHA-256 pinned in `scripts/external_manifest.json`.
  `scripts/build_external.py` downloads and rebuilds the corpus.
- **Frame counts** are frames inside each half's in-play block at the native rate, from the corpus
  sidecars (`n_inplay`).

## 6. Possession rules

We train on each provider's per-frame owner after one harmonisation, the **T = 2 s sandwich merge**
(`katena.labels.sandwich`). An H or A segment shorter than 2 s, with the other team on both sides, is
relabelled to the surrounding team. It never crosses D or X. The scan runs left to right until nothing
changes.

| Situation | CHALLENGE.md | Our labels |
|---|---|---|
| Brief opponent touch, deflection, block or duel; the ball returns to the owner | Not a change | Merged if under 2 s; a change if 2 s or longer |
| Brief opponent touch, then the ball goes straight out | Not a change | **Kept as a flip.** The `tail` option would merge it; not adopted |
| Pass in flight, loose ball | Stays with the last team | The last team (IDSSE, Metrica). SkillCorner gives X, which splits the spell |
| Restart | There is a clear possessor | The restart spell is never merged, however short |
| Chain: H, then A 1 s, H 0.3 s, A 10 s | A gains control at the 10 s spell | Change placed at the 10 s spell |
| Ball out of play | D | D, from `BallStatus` (IDSSE) or events (Metrica); never merged |
| Who owns a dead ball | Not asked | Not predicted |

**Why 2 s.** CHALLENGE.md calls "controlled" ambiguous, and external data cannot tell what the corrected
tracking does. We trained at T = 0, 1, 2, 3 and 5 s and evaluated against each, and T = 2 s had the
smallest worst-case regret (0.017 on IDSSE). DFL's events suggest an opponent flip becomes more likely
than not to be deliberate at about 3.5 s. That describes control in DFL data, but labels built on it lost
at the gate (H3.5). The convention matters a lot: the same M2 predictions score 0.495 to 0.653 across the
conventions.

**The output.** The decoder applies no explicit 2 s rule; it learns transitions from the labels. F
under-segments, with 0.738 predicted possessions per true one at target-like. On the development folds it
never writes X on a graded frame.

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

- **Simulated noise is not the real ATD.** A classifier separates real SkillCorner and Metrica tracking
  from every degraded world with AUC 1.00, mostly on jitter. At 5 Hz the per-slot second difference is
  0.04-0.05 m in real data and 0.47-1.38 m in the degrader. At 25 Hz the ball's median absolute second
  difference is 0.001 m on Metrica's samples and about 1 m in the target-like world. The `_smooth` worlds
  are one world in five of the mix (one in three for SkillCorner). At inference, `touch` reads the real
  25 Hz ball, which no development world reproduces.
- **SkillCorner is 20 of the 28 folds.** It has no dead ball, and null possession splits its spells. At
  10 Hz the touch features get two native frames per sample. Its "target-like" world keeps 15.3 tracked
  objects, not 20.5. Five of F's six losing folds are SkillCorner; the worst fold loses 0.0181.
- **Metrica's dead spans run long.** The event rule keeps D through live play after an untagged restart,
  and Metrica gives about a quarter of the alive head's rows. The cleaner rule (MR) was neutral.
- **Pooled calibration.** The D2 coupling constants come from Metrica Games 1 and 3, and the smooth jitter
  from the 20 SkillCorner matches, all development folds. So a held-out match helps set a few scalars of
  its own training worlds. The first degrader calibration also used every SkillCorner and Metrica match,
  `Sample_Game_2` included. The effect is probably small but unmeasured.
- **The development estimate is optimistic.** D1, D2 and D3 were gated on the same 28 folds before F
  combined them. F was registered before its own score, but its parts were chosen knowing their results.
  Their gains (+0.0069, +0.0065, +0.0019) sum to +0.0153, against F's +0.0158. After F, six more packages
  were gated on the same folds (sections 3.4 and 4), each registered before its score: ten in all. The two
  locked benches are the only matches no choice was made on. F's one read there gains about as much as on
  the development folds (section 3.1). With two matches, that read is a sanity check, not a second
  estimate, and nothing after F was read on them.
- **The TCN seed matters.** F's TCN uses seed 0, fixed in advance. Averaging in a second seed lowered the
  8-fold score by 0.005 (1 of 8 folds won), so seed 0 does unusually well on those folds. With seed 1, F is
  still about +0.02 above M2 there, so roughly a quarter of the 8-fold gain depends on the seed. The 28-fold
  gate is the better guide.
- **Unknown label convention.** The absolute score is uncertain by about 0.15. About 35 touch-then-out
  flips per IDSSE match (11% of spells) remain in our labels.
- **Assumed identity-switch rates.** No external source measures them.
- **Short possessions.** At IoU 0.5 and target-like noise, M2 matched 0.167 of true possessions under
  2 s, and M2 with touch features 0.169.
- **Model bytes differ across machines; the labels did not.** On the reference machine the rebuild is exact.
  A fresh clone, with a fresh download of the external corpus and an empty cache, reproduced every stage byte
  for byte: corpus, features, worlds, pools, both models and a dressed-match CSV (`docs/decision-log.md`, J3 W1).
  Across machines we trained F 11 times on 7 OS and CPU combinations (`.github/workflows/repro.yml`):
  - Windows 10 and Windows Server on Intel, Windows on AMD, Ubuntu on AMD and on Intel, macOS on Apple M1.
  - Every training input is equal in content everywhere. On macOS only NaN sign bits differ, and nothing reads them.
  - The LightGBM model, the scaler and the decoder priors are identical everywhere.
  - The TCN's weights differ in their last bits whenever the OS or the CPU vendor differs, because torch picks
    different kernels. Windows on two other Intel CPUs reproduced the reference machine bit for bit.
  - The dressed external match (IDSSE `DFL-MAT-J03WMX`, 173,528 frames, in the provided ATD format) got a
    byte-identical CSV in all 11 trainings.

  F3 (section 3.4) was checked the same way: six combinations (Windows 10 and Windows Server on Intel,
  Windows and Ubuntu on AMD, macOS on Apple M1), four different sets of TCN weights, one CSV; and the same
  LightGBM model on a seventh machine. Ubuntu on Intel was not drawn for F3.

  So expect the same file on another machine. That has been shown on an external match in the provided format,
  not on the provided match itself, where a near-tie could still flip a label.

**With more time** we would recalibrate the D2 worlds inside each fold, and make native-ball noise as
smooth and correlated as real tracking. We would also re-run the `tail` gate after its label fix, add a
head aimed at the team label (where the oracle puts the points), and test Smart Tagging as features.

## 8. Reproduction

One command, from a clone and the data bundle (`docs/REPRODUCE.md` has the details):

```bash
git clone https://github.com/jusstinn/katenaball && cd katenaball
git checkout <submission_tag>                    # TODO(after the real run): the tagged commit
uv sync --frozen                                 # Python 3.11, dependencies pinned in uv.lock
export MTAH_DATA_DIR=/path/to/data/bundle        # the unpacked competition bundle (default: data/)
make submit TEAM=<team_name>                     # writes submissions/<team_name>.csv, validates it, prints its sha256
```

`make submit` runs five steps:
- `scripts/build_external.py` downloads the 30 external matches and checks every file's SHA-256;
- `scripts/make_submission.py corpus` builds the base features;
- `scripts/make_submission.py worlds` builds the `d2j` worlds, touch columns and pools;
- `scripts/make_submission.py train` trains the two members and saves `bundle.pkl` with its hashes;
- `scripts/make_submission.py predict` labels the provided match, writes the CSV and runs the official checks
  on the exact bytes written.

Without `make` (the team's Windows PC), run the same steps one by one; `docs/REPRODUCE.md` lists them.
`predict` is the only step that opens the provided match, and it prints aggregates only.

- **Pinned stack:** Python 3.11, LightGBM 4.7.0, torch 2.15.0 (CPU), numpy 2.2.6, pandas 2.3.3, numba
  0.68.1, scipy 1.17.1, scikit-learn 1.9.1, locked in `uv.lock`.
- **Checked:** on the reference machine, a fresh clone with an empty cache rebuilt every file and F's
  model byte for byte in 1 h 39 min, including the download. F3 changes only the training step, and F3
  trained twice there gives the same model file.
- **Determinism:** fixed seeds, 8 LightGBM threads, 4 torch threads, and torch's CPU kernels pinned before
  import (`ensemble.pin_cpu_env`). The CSV is built as bytes with LF line ends on every OS.
- **Cost:**

  | Machine | Cache build (corpus, worlds) | Train | Predict | Disk and memory |
  |---|---|---|---|---|
  | Windows 10, i7-7700, CPU only, 2 workers | 18 + 34 min | 23 min with another job running (LightGBM 3, TCN 20) | about a minute | 19 GB cache |
  | Apple-silicon laptop, 3 workers | 14 min | 9 min (LightGBM 0.5, TCN 8) | under a minute | |
  | GitHub runners: Ubuntu, Windows, macOS | | 8, 13 and 37 min | under a minute | 5.7 GB peak in training (Ubuntu) |

  The external download (about 5 GB) is extra.
- **The submitted file:** `TODO(after the real run)`: SHA-256 of `<team_name>.csv` and its commit.
- **No match data in the repository.** CI fails any commit that holds it (`docs/ci.md`).

## 9. How this was built

| Part | Where |
|---|---|
| Evidence and decisions | `docs/decision-log.md`, `docs/bakeoff.md`, `docs/bakeoff-protocol.md`, `docs/label-convention.md`, `docs/external-data.md`, `docs/audit/d5-bakeoff-audit.md` |
| External loaders and corpus | `src/katena/external/`, `scripts/build_external.py`, `scripts/external_manifest.json` |
| Label convention | `src/katena/labels.py`; `T_MAIN` in `src/katena/eval/protocol.py` |
| Degrader and worlds | `src/katena/sim/degrade.py`, `src/katena/sim/worlds.py`, `scripts/calibrate_degrader.py`, `scripts/d2_calibrate.py` |
| Features | `src/katena/featurize/build.py`, `src/katena/featurize/touch.py` |
| Models and blend | `src/katena/models/gbdt.py`, `tcn.py`, `ensemble.py` |
| Decoder | `src/katena/decode/hmm.py` |
| Gates | `src/katena/eval/` (`lomo`, `bootstrap`, `score`, `d2`, `d3`, `touch_gate`, `tmix`, `final`), `scripts/run_*.py` |
| Target reader and checks | `src/katena/target/`, `scripts/check_target.py`, `check_target_layout.py`, `check_domain_gap.py`, `check_touch_target.py` |
| Submission | `src/katena/pipeline.py`, `scripts/make_submission.py`, `Makefile`, `docs/REPRODUCE.md`; checks: `scripts/dressed_check.py`, `scripts/check_public_window.py`, `tests/test_pipeline_atd.py` |
| Firewall and CI | `scripts/firewall_check.py`, `.github/workflows/`, `docs/ci.md` |
