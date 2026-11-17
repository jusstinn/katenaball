# <team_name>: technical document

More than a Hack 2027, Phase 1. Repository: https://github.com/jusstinn/katenaball. Version of 2026-11-17.

This is the short version. `docs/technical-document-full.md` has every table and every change we tested, and
every number in both comes from `docs/decision-log.md` or the evidence pages it links.

## 1. Summary

- We label each frame H, A or D from the automatic tracking (ATD). A LightGBM model and a temporal
  convolutional network (TCN) each predict whether the ball is alive and which team has it. We average
  them in logit space and decode with a 3-state Viterbi. One Smart Tagging kickoff tag per half is also read.
- Models are trained and selected on public external matches only (IDSSE/DFL, Metrica, SkillCorner): 28
  for development, and all 30 for the final model. A degrader corrupts their tracking to look like
  automatic tracking. No model was trained, tuned or selected on the provided match. Nobody labelled any
  frame by hand, and nobody compared our output with the video.
- Each change up to F3 passed a rule fixed before its score. On 28 leave-one-match-out folds of external
  matches, with simulated noise like the provided tracking's, spellF1 rose from 0.280 (the starter
  algorithm) and 0.432 (fixed rules) to 0.586 (LightGBM and Viterbi), 0.602 (F: adds touch features, a noise
  mix and a TCN) and 0.618 (F3: more trees, longer TCN training). These are development estimates, not a forecast.
- The submitted model, AW2, is F3 with three training changes: the 20 SkillCorner matches relabelled from
  SkillCorner's own on-ball events (with dead ball, and a change of owner at the new team's first touch),
  weights by possession length, and a TCN trained twice as long. Our first rule scores the SkillCorner folds
  on the old label, so it cannot judge the new one. We adopted the change under a second rule, whose text
  predates its scores but which we agreed to use only after seeing them.
- AW2 scores 0.699 on the 8 IDSSE and Metrica folds, whose labels did not change (F3: 0.637), and 0.675 on
  all 28, of which 20 are scored on labels we built ourselves. Scored on SkillCorner's old label it loses
  0.018 against F3. It is a bet on how the organisers describe the ground truth (sections 3 and 7).

## 2. Approach

1. We read the tactical ATD: 46 anonymous slots with a team each, plus the ball, in metres.
2. We downsample to 5 Hz and compute 160 feature columns per sample, plus 53 `touch` columns from the 25 Hz ball.
3. Each model predicts p(alive) and p(home) per sample. We average the two models in logit space, weight 0.5.
4. We decode each half with a 3-state Viterbi. Each frame takes the label of the nearest sample in its half;
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
missing), target-like (41%) and severe (55%). Variants tie ball loss to stoppages, add identity switches
in crowds, or cut the jitter to the level of real broadcast tracking. Both models train on an even mix.

## 3. Why this design

**Why learned models, on the tracking alone.** With the ball missing on 41% of graded frames, most frames
have to be decided from the players' shape and from what happens before and after. Fixed rules on the
nearest player reach 0.432 on our development folds; a model learned from external labels reaches 0.586.
We did not use the video: our external matches come without video, so a detector could not have been
trained or tested under the same rules as everything else.

**The models, in order.** M1: fixed rules. M2: LightGBM heads and a Viterbi decoder. F: M2 with the touch
features, a mix of noise variants and the TCN blend. F3: F with more trees and a longer-trained TCN. AW: F3
trained on the new SkillCorner labels (section 6) with possession-length weights. AW2, submitted: AW with
the TCN trained twice as long.

**How we validated.**

- **Leave-one-match-out on 28 external development matches** (6 IDSSE, 2 Metrica, 20 SkillCorner), scored
  with the official `possession_metrics.evaluate`, unchanged. In fold k, nothing fitted sees match k.
- **Rule 1, fixed before each score.** A change is adopted if, at target-like noise, it wins at least
  70% of the folds and the paired 95% bootstrap interval of the mean difference lies above 0. From F on, it
  must also not be worse at severe noise. M2, F and F3 passed it.
- **Rule 2, for the new labels.** Rule 1 scores the 20 SkillCorner folds against the old label, which AW no
  longer learns, so it cannot pass AW. Rule 2 applies rule 1's thresholds to the 8 IDSSE and Metrica folds,
  whose labels do not change, and asks for no loss against F3 on the 20 SkillCorner folds scored on the new
  labels. Its text was fixed before any score of the new labels, but we agreed to adopt under it only after
  we had seen the scores, so it is weaker evidence (section 7).
- **The longer TCN's rule.** AW2 is AW with 9,600 TCN steps for 4,800. Fixed before its score against AW:
  rule 1's thresholds on the 28 folds with SkillCorner on the new labels, and not worse on the 8 or the 20.
- **The provided match is inference only.** No training, tuning, self-training or fitting of the noise
  simulation on it. We used only DATA.md's published aggregates and the published half boundaries.
- **The public 5 minutes are a pipeline check only.** `scripts/check_public_window.py` prints pass or fail,
  never a score: the official file checks pass, home and away are the right way round, the frames line up,
  and the file beats the published smooth50 baseline. A fail means a bug to find, not something to tune,
  and the command that writes the submission stops on one. On the submitted file all four passed
  (valid_submission, home_away_orientation, frame_alignment, beats_published_baseline). We also studied the window itself, with no
  prediction and no score: we counted its possessions and their lengths, and one of us watched its tactical
  video beside the public labels to understand what counts as possession, as the organisers' clarification
  of Nov 3 allows.
- **The submission path equals the evaluation path.** An external match exported in the provided ATD format
  gets, through the submission code and the official scorer, exactly the evaluation code's score. Damaged in
  27 further ways that our noise simulation does not produce, it was still labelled every time and the file
  passed the official checks. That shows the code does not break on bad tracking, not that the labels stay good.

**Results** at target-like noise, each model against the one before it. "Old label" and "new labels" say
how the SkillCorner folds are scored (section 6):

| Model | Folds | spellF1 | Difference [95% CI] | Folds won |
|---|---|---|---|---|
| F against M2 | 28, old label | 0.6019 | +0.0158 [+0.0111, +0.0205] | 22/28 |
| F3 against F | 28, old label | 0.6180 | +0.0161 [+0.0118, +0.0207] | 21/28 |
| AW against F3 | 8 IDSSE and Metrica (rule 2) | 0.6923 | +0.0548 [+0.0399, +0.0697] | 8/8 |
| **AW2 against AW** | 28, new labels (the longer TCN's rule) | **0.6753** | **+0.0071 [+0.0046, +0.0096]** | **20/28** |

- **Against F3, AW2 is +0.062 on the 8 folds (8 of 8)**, 0.699 against 0.637, and +0.056 on the 20
  SkillCorner folds on the new labels (18 of 20), 0.666 against 0.609. Neither step is worse at severe
  noise. Against AW it won 20 of the 28 folds, the minimum that 70% requires.
- **Evidence that does not rest on the new labels.** Two models never trained on them also beat F3 on the
  20 SkillCorner folds scored on them: one with a column marking each row's dataset (+0.0265, 18 of 20), and
  F3 with no SkillCorner match in its LightGBM team head (+0.0115, 15 of 20). Both lose on the old label.

## 4. Assets, and what we tried and dropped

| Provided asset | Used? | How |
|---|---|---|
| Tactical ATD | Yes, the only model input | 5 Hz features, plus the native-rate ball for `touch` |
| Half boundaries (DATA.md, `games.half_video_blocks`) | Yes | Which frames are graded |
| Smart Tagging | Kickoff only | The `Kick Off (Start)` tags set the frames of each half before its kickoff to D (section 6). Set-piece tags as features were tested and dropped (below) |
| Homography; tactical video | No | We read only the homography file's header, to check its format. We built no detector (section 3) |
| Broadcast video, ATD and homography | No | One check reads the broadcast ATD's metadata, to confirm how the tracking slots are laid out |
| Public 5-minute ground truth | Pipeline check only | Section 3 |

| Tried | Result | Decision |
|---|---|---|
| Six other training-label conventions | 28 folds: -0.008 to +0.003 | None adopted (section 6) |
| Set-piece features from Smart Tagging, with imitation tags built from external events | 28 folds: +0.0046, 16/28 | Failed: the gain is all on the 8 folds that have events |
| Weights by possession length, in either model, on the old SkillCorner label | 28 folds: +0.005 and +0.011, 17/28 each | Failed rule 1 on folds won. Adopted with the new labels (AW) |

## 5. Training data

| Dataset | Source and licence | Who made the labels, and how | Matches | Frames in the halves | Rate |
|---|---|---|---|---|---|
| IDSSE / DFL open data | figshare 10.6084/m9.figshare.28196177 (Bassek, Rein, Weber, Memmert, *Sci Data* 2025). CC BY 4.0; credit the authors, DFL and Sportec Solutions | DFL/TRACAB optical tracking, checked by live operators. Per-frame `BallStatus` 0 gives D; otherwise `BallPossession` gives H or A. The paper does not say how the possession field is derived | 7; 2022/23 Bundesliga and 2. Bundesliga | 1,000,331 | 25 Hz |
| Metrica sample data | github.com/metrica-sports/sample-data, commit `e706dd5`. No licence stated; the README asks users to acknowledge the source. Not redistributed | None supplied. We derive them from Metrica's own event tagging. Owner: the team of the last PASS, CARRY, RECOVERY or SET PIECE. D: from a dead trigger (ball out, foul, offside, hand ball, goal, shot out) to the next set piece or goal kick | 3; anonymised | 424,886 | 25 Hz |
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27`. MIT, (c) 2020 SkillCorner | SkillCorner's broadcast-video tracking. We derive them from the `player_possession` events of its `*_dynamic_events.csv`, made automatically from that tracking (section 6). D on 20.6% of those frames, no label (X) on 24.2% | 20; A-League 2024/25 | 1,182,244 | 10 Hz |

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
| Brief opponent touch, then the ball goes straight out | Not a change | Kept as a change. Merging it narrowly missed rule 1 |
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
  or more than 180 s into it. External matches have no graded frames before a kickoff, so this rule could
  not be scored and was adopted directly; the 180 s bound was fixed before the rule ran on the provided
  match, from no frame of it.

## 7. Limitations

- **The new labels are a bet on the documentation.** They follow CHALLENGE.md's description; SkillCorner's
  per-frame label does not. If the hidden truth behaves like that label after all, the bet loses: scored
  that way (rule 1), AW2 is -0.0184 [-0.0264, -0.0108] against F3 on the 28 folds and wins 8.
- **The new SkillCorner labels are our construction.** SkillCorner makes its events automatically from the
  same tracking. In 228 of 1,851 stoppages the dead ball starts late, only when the broadcast cuts away, and
  a restart sits a fixed 1.0 s before its reception. A score against these labels favours a model trained
  on them, so it can show that a change does no harm there, not that the change is right.
- **The length weights help one kind of truth.** With them AW2 scores 0.681 on the six IDSSE folds, and
  the same model without them 0.657. Without them it is slightly ahead on the two Metrica folds (0.764
  against 0.753) and on SkillCorner's new labels (0.671 against 0.666). The provided match's ground truth
  comes from Metrica's corrected tracking.
- **Simulated noise is not the real ATD.** A classifier tells external real tracking (SkillCorner, Metrica) from
  every degraded variant with AUC 1.00, mostly on jitter, which the degrader makes 10 to 30 times rougher;
  one variant in five of the training mix has realistic jitter. At inference the touch features read the
  real 25 Hz ball, which no development variant reproduces.
- **The development estimate is optimistic.** Twenty-one changes were tested one after another on the same
  28 folds, besides the first comparison of models and the label conventions. Rule 2 reads 8 folds that all
  of them saw, and four earlier changes passed a test like it and were not adopted. Two held-back matches
  (one IDSSE, one Metrica) are the only ones no choice was made on. Only F was read on them, once, after
  its adoption: it beat the model before it on both (+0.013 to +0.057).
- **Other unknowns.** The unknown label convention moves the absolute score by about 0.15. With another TCN
  seed, F's gain on 8 folds was about a quarter smaller; seed 0 was kept, and the test was not repeated since.
- **Model bytes differ across machines; the labels did not.** The TCN's last bits vary with the OS and CPU
  vendor, and one LightGBM fold model of another candidate grew other trees on Linux than on our PC.
  AW2's final model gave one LightGBM model and one CSV, byte for byte, for an external match in the
  provided format, on GitHub's macOS, Ubuntu and Windows machines and on our PC. We expect the same file on
  other machines, but on the provided match a near-tie could still flip a label. If your file differs,
  compare `model_AW2/manifest.json` with `docs/repro/expected-windows.json`: it shows which model changed.
- **Where it fails, and what we would do with more time.** Most lost points are short possessions: one team
  wins the ball and loses it again within seconds. In F3's error analysis these are 26% of the true
  possessions on the 8 IDSSE and Metrica folds and F3 finds 11% of them, against 91% of the possessions
  over 16 s. Recovering the missed ones under 4 s would be worth +0.19 there; the true alive/dead pattern
  +0.02. We did not repeat this count for AW2. Next: check the new labels' dead-ball starts against a
  second source, repeat AW2's test on a second machine, and calibrate the noise simulation inside each fold.

## 8. Reproduction

One command, from a clone and the data bundle (`docs/REPRODUCE.md` has the details):

```bash
git clone https://github.com/jusstinn/katenaball && cd katenaball
git checkout submission-v1                       # the tagged commit that made the submitted file
uv sync --frozen                                 # Python 3.11, dependencies pinned in uv.lock
export MTAH_DATA_DIR=/path/to/data/bundle        # the unpacked competition bundle (default: data/)
make submit TEAM=<team_name>                     # writes submissions/<team_name>.csv, validates it, prints its sha256
```

- **What it runs.** It checks that the bundle's three files exist, downloads the 30 external matches and
  checks every file's SHA-256, builds the features, the noise variants and the SkillCorner labels, trains the
  two models, labels the provided match, and runs the official checks on the exact bytes written. Only the
  last step opens the provided match.
- **Needs.** [uv](https://docs.astral.sh/uv/), git, about 30 GB of disk and 16 GB of RAM, no GPU. On Linux,
  the system's OpenMP runtime (`apt-get install libgomp1`); the first step names it when it is missing.
- **Pinned stack.** Python 3.11 and every dependency are locked in `uv.lock` (LightGBM 4.7.0, torch 2.15.0
  on the CPU).
- **Determinism.** Fixed seeds, 8 LightGBM threads, 4 torch threads, torch's CPU kernels pinned before
  import. The CSV is built as bytes with LF line ends on every OS. Section 7 has the cross-machine checks.
- **Cost.** On a 4-core PC from 2017: about an hour to build the features and noise variants, about 26
  minutes to train, a minute to label. The external download is about 5 GB.
- **The submitted file.** `<team_name>.csv`, SHA-256
  `478211821e25e71c8e3ce93ae5587d2ef89bb5279c4c1b36e9d747a8b164d109`, made on our PC on 2026-11-17 by commit
  `4f9e40b7e4bb` (tag `submission-v1`). In that run the three model files had the hashes recorded beforehand
  in `docs/repro/expected-windows.json`, and the official validator printed Valid. The run's entry is in
  `docs/decision-log.md`.
- **No match data in the repository.** CI fails any commit that holds it.

## 9. Disclosures

- **LLMs.** We used a hosted coding assistant to write code and text, as the rules allow. No video, tracking
  file or other restricted file of the provided match was given to it, and what we run on the match prints
  pass or fail and aggregates only. Of the organisers' files it saw only the public 5-minute labels. One of
  us also described to it, in words, what he saw in the public 5 minutes of video; no image or file was shared.
  It started the final run on our PC at the request of one of us, who was away from the PC: the run's output
  went to a local file it never read, and of that run it read only the pass or fail lines and the hashes.
- **A public file we opened and did not use.** While searching the web for external datasets, an automated
  search agent we ran opened a public record (Zenodo 20185375) that turned out to hold statistics about the
  provided game, and printed a fragment of one row. We deleted the file. Nothing from it entered the code,
  the training data or any decision.
