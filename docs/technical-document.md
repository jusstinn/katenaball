# <team_name>: technical document

More than a Hack 2027, Phase 1. Repository: https://github.com/jusstinn/katenaball. Version of 2026-11-13.

This is the short version. `docs/technical-document-full.md` has every table and every change we tested, and
every number in both comes from `docs/decision-log.md`.
`TODO(after the real run)` marks what only the final run on the provided match can fill in.

## 1. Summary

- We label each frame H, A or D from the automatic tracking (ATD). A LightGBM model and a temporal
  convolutional network (TCN) each predict whether the ball is alive and which team has it. We average
  them in logit space and decode with a 3-state Viterbi. One Smart Tagging kickoff tag per half is also read.
- Models are trained and selected on public external matches only (IDSSE/DFL, Metrica, SkillCorner): 28
  for development, and all 30 for the final model. A degrader corrupts their tracking to look like
  automatic tracking. No model was trained, tuned or selected on the provided match. Nobody labelled any
  frame by hand, and nobody compared our output with the video.
- Every change was adopted under a rule written down before its score. On 28 leave-one-match-out folds at
  target-like noise, spellF1 rose from 0.280 (the starter algorithm) to 0.586 (LightGBM and Viterbi), 0.602
  (model F) and 0.618 (model F3). These are development estimates, not a prediction of the hidden score.
- The submitted model, AW2, is F3 trained on other labels for the 20 SkillCorner matches, with weights by
  possession length and a TCN trained twice as long. The old rule scores 20 of its 28 folds against the
  labels AW2 no longer learns, so the new labels were judged under a second rule, declared before any
  score. On the 8 folds whose truth they leave alone AW2 scores 0.6993 against F3's 0.6374; on the old
  SkillCorner labels the relabelled model loses. It is a bet on how the organisers describe the ground
  truth (sections 3 and 7).

## 2. Approach

1. Read the tactical ATD: 46 anonymous slots with a team each, plus the ball, in metres.
2. Downsample to 5 Hz. Compute 160 feature columns per sample, plus 53 `touch` columns from the 25 Hz ball.
3. Each model predicts p(alive) and p(home) per sample. Average the two models in logit space, weight 0.5.
4. Decode each half with a 3-state Viterbi. Each frame takes the label of the nearest sample in its half;
   frames outside the graded halves get X. Frames before a half's tagged kickoff become D (section 6).

**Features.** They describe a *focal* team attacking towards +x; the attack direction is inferred from player
positions. Every row is computed twice, once with each team as focal, and the two predictions are averaged.
The 160 base columns are: gap-aware ball kinematics and the nearest player of each team (62); team shape,
line heights, pressing and tracked-object counts, which need no ball (61); dead-ball cues from the ball and
from the players (29); ball visibility (8).

**Touch features.** From the native 25 Hz ball we detect touches: breaks of at least 2 m/s in its velocity,
between lines fitted over the 0.4 s before and after a frame, that also exceed 3.5 times a local jitter
estimate. A touch goes to the nearest player within 3 m who is clearly nearer than the nearest opponent.
The columns are times since and until each team's touches, who touched last, turnovers and counts.

- **LightGBM heads.** The *alive* head trains on every labelled row and the *team* head on the H and A
  rows, from all three sources. In the team head a training frame weighs 1 / L, where L is the length in
  seconds of its possession, clipped to [1, 30]: spellF1 counts a possession once, however long it lasts.
  1,200 trees, learning rate 0.1, 31 leaves, seed 0, deterministic. The parameters were fixed before the
  first comparison and never tuned, apart from the number of trees.
- **TCN.** Six residual blocks of 48 channels with dilations 1 to 32, looking both ways in time, with the
  same two heads and the same weights in its team loss. 9,600 AdamW steps on 256-sample windows, seed 0.
- **Decoder.** With pa = p(alive) and ph = p(home), the class probabilities are pa·ph for H, pa·(1 - ph) for
  A and 1 - pa for D. The emission score is 2 (log p(class) - log prior(class)). Transitions and class
  priors are smoothed counts from the IDSSE and Metrica training labels.

**Training under simulated noise.** The provided ATD has no identities, 46 slots for 22 players, and no ball
on 41% of graded frames (DATA.md). A degrader makes the cleaner external tracking look like that: it tops
up the ball loss, cuts players into anonymous fragments in 23 slots per team, and adds gaps, ghosts,
duplicates, jitter, false ball detections and wrong-team slots, at three severities: mild (20% of the ball
missing), target-like (41%) and severe (55%). Variants tie ball loss to stoppages, add identity switches in
crowds, or cut the jitter to the level of real broadcast tracking. Both models train on an even mix.

## 3. Why this design

**How we validated.**

- **Leave-one-match-out on 28 external development matches** (6 IDSSE, 2 Metrica, 20 SkillCorner), scored
  with the official `possession_metrics.evaluate`, unchanged. In fold k, nothing fitted sees match k.
- **Rule 1, registered before each score.** A change is adopted if, at target-like noise, it wins at least
  70% of the folds and the paired 95% bootstrap interval of the mean difference lies above 0. From F on, it
  must also not be worse at severe noise. M2, F and F3 passed it.
- **Rule 2, a change of protocol for the new labels.** AW is F3 with the new SkillCorner labels (section 6)
  and the weights. Rule 1 scores the 20 SkillCorner folds against the per-frame label AW stops training on,
  so it cannot pass AW. Rule 2 was written down before any score of the new labels: rule 1's thresholds on
  the 8 IDSSE and Metrica folds, whose truth does not change, and not worse than F3 on the 20 SkillCorner
  folds scored on the new labels. It is weaker evidence (section 7). Both of us put our names to it
  (PENDING_JUSTIN_NAME); one had by then seen a first score of the new labels.
- **The longer TCN.** AW2 is AW with 9,600 TCN steps for 4,800, registered before its score against AW:
  rule 1's thresholds on the 28 folds with SkillCorner on the new labels, and not worse on the 8 or the 20.
- **The provided match is inference only.** No training, tuning, self-training or fitting of the degrader
  on it. We used only DATA.md's published aggregates and the published half boundaries. The checks a team
  member runs on the real file print pass or fail and aggregates only, and never choose between versions.
- **The public 5 minutes are a pipeline check only.** `scripts/check_public_window.py` prints pass or fail,
  never a score: the official file checks pass, home and away are the right way round, the frames line up,
  and the file beats the published smooth50 baseline. A fail means a bug to find, not something to tune.
  `TODO(after the real run)`: the result of that run. Twice we looked at the window with no prediction and
  no score, and changed nothing: we counted its possessions and their lengths, and one team member watched
  its tactical video beside the public labels, as the organisers' clarification of Nov 3 allows.
- **The submission path equals the evaluation path, and does not break on damaged tracking.** An external
  match exported in the provided ATD format and labelled by the submission code gets, from the official
  scorer, exactly the score the evaluation code gives the same model. Damaged in 27 ways the training
  worlds do not produce, it was labelled every time and passed the official checks: a crash test, not a
  test of the labels. Both ran with F and F3; AW2 changes training only.

**Results** at target-like noise, each model against the one before it (F against M2, LightGBM heads and
Viterbi, at 0.586). "Old label" and "new labels" say how the SkillCorner folds are scored (section 6):

| Model | Folds | spellF1 | Difference [95% CI] | Folds won |
|---|---|---|---|---|
| F: M2 with the touch features, the mix of noise worlds and the TCN blend | 28, old label | 0.6019 | +0.0158 [+0.0111, +0.0205] | 22/28 |
| F3: F with 1,200 trees (was 150) and a TCN trained 4,800 steps (was 1,200) on the touch columns too | 28, old label | 0.6180 | +0.0161 [+0.0118, +0.0207] | 21/28 |
| AW: F3 with the new SkillCorner labels and the weights | 8 IDSSE and Metrica (rule 2) | 0.6923 | +0.0548 [+0.0399, +0.0697] | 8/8 |
| | 20 SkillCorner, new labels | 0.6586 | +0.0491 [+0.0387, +0.0597] | 19/20 |
| | 28, old label (rule 1: fails) | | -0.0224 | 8/28 |
| **AW2: AW with a TCN trained 9,600 steps** | 28, new labels (its rule) | **0.6753** | **+0.0071 [+0.0046, +0.0096]** | **20/28** |
| | 8 IDSSE and Metrica | 0.6993 | +0.0070 [+0.0018, +0.0118] | 6/8 |
| | 20 SkillCorner, new labels | 0.6657 | +0.0071 [+0.0044, +0.0099] | 14/20 |

- **Against F3, AW2 is +0.0618 on the 8 folds (8 of 8)** and +0.0563 on the 20 SkillCorner folds on the new
  labels (18 of 20), where F3 scores 0.6374 and 0.6094. Neither step is worse at severe noise. AW2 passed
  its rule with exactly the 20 folds it asks for; that rule does not read the old label.
- **How they were run.** The fold models of AW and AW2 were fitted on GitHub's Ubuntu runners only, and
  scored on our PC against F3's cached fold models. The gate of A, AW without the weights, was repeated on
  our PC: the same verdicts, every mean within 0.0002, 27 of 28 folds identical to the last digit.
- **Evidence that does not rest on AW's labels.** On the 20 SkillCorner folds scored on the new labels, two
  models that never trained on them also beat F3: one with a column naming each training row's dataset,
  read as IDSSE and Metrica (+0.0265 [+0.0178, +0.0349], 18 of 20; it had failed rule 1), and F3 with a
  LightGBM team head that saw no SkillCorner match (+0.0115 [+0.0037, +0.0193], 15 of 20). On the old label
  both lose: on SkillCorner the truth decides the verdict.

## 4. Assets, and what we tried and dropped

| Provided asset | Used? | How |
|---|---|---|
| Tactical ATD | Yes, the only model input | 5 Hz features, plus the native-rate ball for `touch` |
| Half boundaries (DATA.md, `games.half_video_blocks`) | Yes | Which frames are graded |
| Smart Tagging | Kickoff only | The `Kick Off (Start)` tags set the frames of each half before its kickoff to D (section 6). Set-piece tags as features were tested and dropped (below) |
| Homography; tactical video | No | Only the schema probe reads the homography's header. We built no detector |
| Broadcast video, ATD and homography | No | Only the slot-layout check reads the broadcast ATD metadata |
| Public 5-minute ground truth | Pipeline check only | Section 3 |

| Tried | Result | Decision |
|---|---|---|
| Six other training-label conventions | 28 folds: -0.008 to +0.003 | None adopted (section 6) |
| Set-piece features from Smart Tagging, with imitation tags built from external events | 28 folds: +0.0046, 16/28 | Failed: the gain is all on the 8 folds that have events |
| Weights by possession length, in either model, on the old SkillCorner label | 28 folds: +0.005 and +0.011, 17/28 each | Failed rule 1 on folds won. Adopted with the new labels (AW) |
| A: AW without the weights. AT, ATW: A and AW with F3's alive/dead training | 8 folds: A 0.6763, +0.0389 [+0.0276, +0.0498], 8/8; AT 0.6725; ATW 0.6964 | All met rule 2 and failed rule 1. The registered tests chose AW: ahead of A by +0.0160 [+0.0041, +0.0282], 5/8; ATW not ahead of AW, +0.0041 [-0.0017, +0.0099] |

## 5. Training data

| Dataset | Source and licence | Who made the labels, and how | Matches | In-play frames | Rate |
|---|---|---|---|---|---|
| IDSSE / DFL open data | figshare 10.6084/m9.figshare.28196177 (Bassek, Rein, Weber, Memmert, *Sci Data* 2025). CC BY 4.0; credit the authors, DFL and Sportec Solutions | DFL/TRACAB optical tracking, checked by live operators. Per-frame `BallStatus` 0 gives D; otherwise `BallPossession` gives H or A. The paper does not say how the possession field is derived | 7; 2022/23 Bundesliga and 2. Bundesliga | 1,000,331 | 25 Hz |
| Metrica sample data | github.com/metrica-sports/sample-data, commit `e706dd5`. No licence stated; the README asks users to acknowledge the source. Not redistributed | None supplied. We derive them from Metrica's own event tagging. Owner: the team of the last PASS, CARRY, RECOVERY or SET PIECE. D: from a dead trigger (ball out, foul, offside, hand ball, goal, shot out) to the next set piece or goal kick | 3; anonymised | 424,886 | 25 Hz |
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27`. MIT, (c) 2020 SkillCorner | SkillCorner's broadcast-video tracking. We derive them from the `player_possession` events of its `*_dynamic_events.csv`, made automatically from that tracking (section 6). D on 20.6% of in-play frames, X on 24.2% | 20; A-League 2024/25 | 1,182,244 | 10 Hz |

- **No hand labels.** Nobody on the team labelled or corrected any frame of any game. Every label is a
  provider field or an algorithm over provider data, and none comes from the provided match.
- **Use.** Both heads train on all three sources; the decoder priors use IDSSE and Metrica. The final model
  trains once on all 30: 657,133 LightGBM rows of 213 columns, 549,880 of them in the alive head and 363,838
  in the team head.
- **Integrity.** All 111 raw files, the SkillCorner event files among them, are SHA-256 pinned in
  `scripts/external_manifest.json`, and `scripts/build_external.py` downloads them and rebuilds the corpus.

## 6. Possession rules

We train on a per-frame owner from each provider: its own field (IDSSE) or a rule of ours over its events
(Metrica, SkillCorner). One harmonisation follows, the **2 s sandwich merge**. An H or A segment shorter
than 2 s, with the other team on both sides, is relabelled to the surrounding team. It never crosses D or X.

| Situation | CHALLENGE.md | Our labels |
|---|---|---|
| Brief opponent touch, deflection, block or duel; the ball returns to the owner | Not a change | Merged if under 2 s; a change if 2 s or longer |
| Brief opponent touch, then the ball goes straight out | Not a change | Kept as a change. Merging it was tested and missed the gate by a little |
| Pass in flight, loose ball | Stays with the last team | The last team, in all three sources |
| Restart | There is a clear possessor | The restart possession is never merged, however short |
| Ball out of play | D | D, from `BallStatus` (IDSSE), events (Metrica) or interruption flags (SkillCorner); never merged |

- **The SkillCorner labels, new in AW.** F3 trained on SkillCorner's per-frame `possession.group`, which
  has no dead ball, is null on 46% of in-play frames, and changes owner a median of 1.0 s before the new
  team's first touch. CHALLENGE.md describes a truth with dead ball as a label, an owner on every live
  frame, and a change when the other team gains control. So AW builds the label from SkillCorner's
  `player_possession` events: from the first frame of one event to the first frame of the next, the ball
  is that player's team's. A gap that SkillCorner flags as a game interruption is dead ball, from the first
  frame where the ball is outside the pitch or untracked (else 1.0 s after the last touch; at once after a
  foul) until 1.0 s before the restart is received. Both constants were fixed before any score.
- **Why 2 s.** CHALLENGE.md calls "controlled" ambiguous. We trained with merge thresholds of 0, 1, 2, 3 and
  5 s and scored each against each: 2 s had the smallest worst-case loss. A 1 s threshold later passed
  rule 1 (+0.003); we kept 2 s, which loses less against the conventions closest to the written definition.
- **The output.** The decoder applies no explicit 2 s rule; it learns transitions from the labels. AW2
  predicts 0.98 possessions per true one on the 8 IDSSE and Metrica folds and 1.23 on SkillCorner's new
  labels (F3: 0.73 and 0.89). It never writes X on a graded frame.
- **Kickoff.** The second half is graded from before its kickoff (FAQ.md), and those frames are D. In each
  graded half, the frames before that half's single `Kick Off (Start)` tag in the provided Smart Tagging
  become D; no frame constant is used. Nothing changes when the tag is absent, repeated, outside the half
  or more than 180 s into it.

## 7. Limitations

- **The new labels are a bet on the documentation.** They follow CHALLENGE.md's description; SkillCorner's
  per-frame label does not. If the hidden truth behaves like that label after all, the bet loses: under
  rule 1 AW, at 4,800 TCN steps, is -0.0224 against F3 on the 28 folds, with 8 folds won.
- **The new SkillCorner labels are our construction.** SkillCorner makes its events automatically from the
  same tracking. In 228 of 1,851 stoppages the dead ball starts late, only when the broadcast cuts away, and
  a restart sits a fixed 1.0 s before its reception. A score against these labels favours a model trained
  on them: it is a guard, not proof.
- **The weights help one kind of truth.** The test that chose AW over A rests on the six IDSSE folds, and
  the longer TCN does not change that: on IDSSE AW2 scores 0.6814 and A2, the same model without the
  weights, 0.6571; on the two Metrica folds 0.7530 against 0.7642; on SkillCorner's new labels 0.6657
  against 0.6712. The provided match's ground truth comes from Metrica's corrected tracking.
- **Simulated noise is not the real ATD.** A classifier tells real tracking (SkillCorner, Metrica) from
  every degraded world with AUC 1.00, mostly on jitter, which the degrader makes 10 to 30 times rougher;
  one world in five of the training mix has realistic jitter. At inference the touch features read the
  real 25 Hz ball, which no development world reproduces.
- **The development estimate is optimistic.** Nineteen changes were gated one after another on the same 28
  folds, besides the first comparison of models and the label conventions. Rule 2 reads 8 folds that all
  of them saw, and four earlier changes passed a test like it and were not adopted. Two locked bench
  matches (one IDSSE, one Metrica) are the only ones no choice was made on. Only F was read on them, once,
  after its adoption: it beat the model before it on both (+0.013 to +0.057).
- **Other unknowns.** The unknown label convention moves the absolute score by about 0.15. With another TCN
  seed, F's gain on 8 folds was about a quarter smaller; seed 0 was kept, and the test was not repeated since.
- **Model bytes differ across machines; the labels did not.** The TCN's weights differ in their last bits
  with the OS and the CPU vendor. LightGBM was byte-identical on every machine for F and F3; in A's gate,
  one of the 28 LightGBM fold models grew other trees on a Linux runner than on our PC. The final models of
  A and of AW did not show that: each gave one LightGBM model and one CSV, byte for byte, for an external
  match in the provided format, on macOS, Ubuntu and Windows runners and on our PC. AW2's final model did
  the same (PENDING_AW2_REPRO). So expect the same file on another machine; that was shown on an external
  match, not on the provided one, where a near-tie could flip a label.

## 8. Reproduction

One command, from a clone and the data bundle (`docs/REPRODUCE.md` has the details):

```bash
git clone https://github.com/jusstinn/katenaball && cd katenaball
git checkout <submission_tag>                    # TODO(after the real run): the tagged commit
uv sync --frozen                                 # Python 3.11, dependencies pinned in uv.lock
export MTAH_DATA_DIR=/path/to/data/bundle        # the unpacked competition bundle (default: data/)
make submit TEAM=<team_name>                     # writes submissions/<team_name>.csv, validates it, prints its sha256
```

- **What it runs.** It checks that the bundle's three files exist, downloads the 30 external matches and
  checks every file's SHA-256, builds the features, the noise worlds and the SkillCorner labels, trains the
  two models, labels the provided match, and runs the official checks on the exact bytes written. Only the
  last step opens the provided match, and it prints aggregates only.
- **Needs.** [uv](https://docs.astral.sh/uv/), git, about 30 GB of disk and 16 GB of RAM, no GPU. On Linux,
  the system's OpenMP runtime (`apt-get install libgomp1`); the first step names it when it is missing.
- **Pinned stack.** Python 3.11, LightGBM 4.7.0, torch 2.15.0 (on the CPU), numpy 2.2.6, pandas 2.3.3,
  numba 0.68.1, scipy 1.17.1, scikit-learn 1.9.1, locked in `uv.lock`.
- **Determinism.** Fixed seeds, 8 LightGBM threads, 4 torch threads, torch's CPU kernels pinned before
  import. The CSV is built as bytes with LF line ends on every OS. Section 7 has the cross-machine checks.
- **Cost.** On a 4-core PC from 2017: about an hour to build the features and worlds, about 30 minutes to
  train (PENDING_AW2_TRAIN_TIME), a minute to label. The external download is about 5 GB.
- **The submitted file.** `TODO(after the real run)`: SHA-256 of `<team_name>.csv` and its commit.
- **No match data in the repository.** CI fails any commit that holds it.

## 9. Disclosures

- **AI assistance.** Most of the code and of this document was written with a hosted AI coding assistant
  (Claude Code), working under the team's direction. The competition's data was kept away from it. The
  repository's rules (`CLAUDE.md`) forbid it to open the data folder, only team members ran anything on the
  provided match, and what those runs print is pass or fail and aggregates: counts, shares and hashes, never
  a file, a frame or a per-frame value. One exception in words only: the team member who watched the public
  5 minutes (section 3) described to it what he saw, with estimated times. It saw no image and no file.
- **A public file we opened and did not use.** While searching the web for external datasets, an automated
  agent opened a public record that turned out to hold statistics about the provided game. The file was
  deleted. Nothing from it entered the code, the training data or any decision.
- **StatsBomb open data.** 35 matches were downloaded to a temporary folder before we read the user
  agreement. We did not use them: they are in no training set, result or commit.
