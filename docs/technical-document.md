# <team_name>: technical document

More than a Hack 2027, Phase 1. Repository: https://github.com/jusstinn/katenaball. Version of 2026-11-12.

This is the short version. `docs/technical-document-full.md` has every table and every change we tested, and
every number in both comes from `docs/decision-log.md`.
`TODO(after the real run)` marks what only the final run on the provided match can fill in.

## 1. Summary

- We label each frame H, A or D from the automatic tracking (ATD). A LightGBM model and a temporal
  convolutional network (TCN) each predict whether the ball is alive and which team has it. We average
  them in logit space and decode with a 3-state Viterbi. The only other input is one Smart Tagging kickoff
  tag per half.
- Models are trained and selected on public external matches only (IDSSE/DFL, Metrica, SkillCorner): 28
  for development, and all 30 for the final model. A degrader corrupts their tracking to look like
  automatic tracking.
- No model was trained, tuned or selected on the provided match. Nobody labelled any frame by hand, and
  nobody compared our output with the video.
- Every change was adopted under a rule written down before its score. On 28 leave-one-match-out folds at
  target-like noise, spellF1 rose from 0.280 (the starter algorithm) to 0.586 (LightGBM and Viterbi), 0.602
  (model F) and 0.618 (model F3, submitted). These are development estimates, not a prediction of the hidden
  score.

## 2. Approach

**Pipeline.**

1. Read the tactical ATD: 46 anonymous slots with a team each, plus the ball, in metres.
2. Downsample to 5 Hz. Compute 160 feature columns per sample, plus 53 `touch` columns from the ball at the
   native 25 Hz.
3. Each model predicts p(alive) and p(home) per sample. Average the two models in logit space, weight 0.5.
4. Decode each half with a 3-state Viterbi. Each frame takes the label of the nearest sample in its half;
   frames outside the graded halves get X.
5. In each graded half, set the frames before the tagged kickoff to D (section 6).

**Features.** They describe a *focal* team attacking towards +x; the attack direction is inferred from player
positions. Every row is computed twice, once with each team as focal, and the two predictions are averaged,
so both sides are treated the same. The 160 base columns are: gap-aware ball kinematics and the nearest
player of each team (62); team shape, line heights, pressing and tracked-object counts, which need no ball
(61); dead-ball cues from the ball and from the players (29); ball visibility (8).

**Touch features.** At every native frame we fit a straight line to the visible ball over the 0.4 s before
it and another over the 0.4 s after. A touch is a velocity break of at least 2 m/s that also exceeds 3.5
times a local jitter estimate. It goes to the nearest player within 3 m who is clearly nearer than the
nearest opponent. The columns are the time since and until each team's touches, who touched last,
turnovers, touch counts, and each team's closest approach to the ball.

**Models and decoder.**

- **LightGBM heads.** The *alive* head trains on IDSSE and Metrica rows only, because SkillCorner has no
  dead-ball label. The *team* head trains on H and A rows from all three sources. 1,200 trees, learning
  rate 0.1, 31 leaves, seed 0, deterministic. The parameters were fixed before the first comparison and
  never tuned, apart from the number of trees (section 3).
- **TCN.** Six residual blocks of 48 channels with dilations 1 to 32, looking both ways in time, with the
  same two heads. 4,800 AdamW steps on random 256-sample windows, seed 0.
- **Decoder.** With pa = p(alive) and ph = p(home), the class probabilities are pa·ph for H, pa·(1 - ph) for
  A and 1 - pa for D. The emission score is 2 (log p(class) - log prior(class)). Transitions and class
  priors are smoothed counts from the IDSSE and Metrica training labels.

**Training under simulated noise.** The provided ATD has no identities, 46 slots for 22 players, and no ball
on 41% of graded frames (DATA.md). The external tracking is cleaner. A degrader tops up the ball loss, cuts
players into anonymous fragments in 23 slots per team, and adds gaps, ghosts, duplicates, jitter, false ball
detections and wrong-team slots. It has three severities: mild (20% of the ball missing), target-like (41%,
DATA.md's figure) and severe (55%). Variants tie ball loss to stoppages, add identity switches in crowds, or
cut the jitter to the level of real broadcast tracking. Both models train on an even mix of these worlds.

## 3. Why this design

**How we validated.**

- **Leave-one-match-out on 28 external development matches** (6 IDSSE, 2 Metrica, 20 SkillCorner), scored
  with the official `possession_metrics.evaluate`, unchanged. In fold k, no model, scaler or prior sees
  match k.
- **A gate registered before each score.** A change is adopted if, at target-like noise, it wins at least
  70% of the folds and the paired 95% bootstrap interval of the mean difference lies above 0. From F on, it
  must also not be worse at severe noise.
- **Two locked bench matches** stayed out of every choice. F was read on them once, after its adoption, and
  beat the model before it on both (+0.013 to +0.057). F3 has no bench read.
- **The provided match is inference only.** No training, tuning, self-training or fitting of the degrader
  on it. We used only DATA.md's published aggregates and the published half boundaries. The checks a team
  member runs on the real file print pass or fail and aggregates only, and never choose between versions.
- **The public 5 minutes are a pipeline check only.** `scripts/check_public_window.py` prints pass or fail,
  never a score: the official file checks pass, home and away are the right way round, the frames line up,
  and the file beats the published smooth50 baseline. A fail means a bug to find, not something to tune.
  `TODO(after the real run)`: the result of that run. Once, we also counted the public window's possessions
  and their lengths, with no prediction and no score, to ask whether our label convention looks
  implausible. It does not, and nothing was changed. One team member also watched the window's tactical
  video beside the public labels, to understand what counts as possession, as the organisers' clarification
  of Nov 3 allows. No prediction of ours was on screen, and nothing was changed.
- **The submission path equals the evaluation path.** An external match exported in the provided ATD format
  and labelled by the submission code gets, from the official scorer, exactly the score the evaluation code
  gives the same model, with every probability and label bit-identical.
- **The submission path does not break on damaged tracking.** The same match on 46 anonymous slots, with 41%
  of the ball missing, was damaged in 27 further ways the training worlds do not produce: no ball at all,
  nothing tracked for five minutes, one team missing, infinite values, slots that change team. Every case
  was labelled and passed the official checks. This shows the path does not crash, not that the labels
  are good.

**Results** on the 28 development folds at target-like noise:

| Model | spellF1 | Against the row above [95% CI] | Folds won |
|---|---|---|---|
| M0: the starter algorithm (nearest player, missing ball = D, majority window) | 0.280 | | |
| M1: fixed rules (1 s control dwell, hold through gaps, simple dead rule) | 0.432 | +0.152 | |
| M2: LightGBM heads and Viterbi | 0.586 | +0.154 | 28/28 |
| F: M2 with the touch features, the mix of noise worlds and the TCN blend | 0.6019 | +0.0158 [+0.0111, +0.0205] | 22/28 |
| **F3: F with 1,200 trees (was 150) and a TCN trained 4,800 steps (was 1,200) that also reads the touch columns** | **0.6180** | **+0.0161 [+0.0118, +0.0207]** | **21/28** |

- **At severe noise** F3 scores 0.5840 against F's 0.5696: +0.0144 [+0.0095, +0.0191], 19 of 28 folds.
- **Where F3's gain is.** On the 8 IDSSE and Metrica folds, +0.0383 with every fold won. On the 20
  SkillCorner folds, +0.0073 [+0.0021, +0.0122] with 13 won. SkillCorner is our only source of broadcast
  tracking, the kind the provided ATD is.
- **Two machines.** F3's two changes were gated on one teammate's laptop and again on the other's PC, under
  a registration written before any score there. Both give 0.6180.
- **What the bake-off showed.** Removing the features that need no ball costs 0.036. With perfect team
  labels a model scores 0.898, and with perfect alive/dead only 0.639, so the points are in the team label.
  Possessions under 8 s are 75 to 85% of the ones we miss.

## 4. Assets, and what we tried and dropped

| Provided asset | Used? | How |
|---|---|---|
| Tactical ATD | Yes, the only model input | 5 Hz features, plus the native-rate ball for `touch` |
| Half boundaries (DATA.md, `games.half_video_blocks`) | Yes | Which frames are graded |
| Smart Tagging | Kickoff only | The `Kick Off (Start)` tags set the frames of each half before its kickoff to D (section 6). Set-piece tags as features were tested and dropped (below) |
| Homography | No | Only the schema probe reads its header |
| Tactical video | No | We built no detector |
| Broadcast video, ATD and homography | No | Only the slot-layout check reads the broadcast ATD metadata |
| Public 5-minute ground truth | Pipeline check only | Section 3 |

| Tried | Result on the 28 folds | Decision |
|---|---|---|
| An explicit-duration HSMM decoder; rescoring spells after decoding | -0.0007; +0.0012 | Dropped |
| Six other training-label conventions | -0.008 to +0.003 | None adopted (section 6) |
| The TCN reading the touch columns at 1,200 steps | +0.0017, 16/28 | Failed alone; adopted with the longer training (F3) |
| Set-piece features from Smart Tagging, with imitation tags built from external events | +0.0046, 16/28 | Failed: the gain is all on the 8 folds that have events |
| Weights by possession length, in either model | +0.005 and +0.011, 17/28 each | Failed on folds won |
| A column that tells the models which dataset's labelling a training row follows | +0.0045, 18/28 (one model); +0.0086, 17/28 (both) | Failed on folds won: +0.035 on the 8 IDSSE and Metrica folds, nothing on SkillCorner |
| A model of possession changes that moves the decoder's switch cost | +0.0003, 13/28 | Failed |
| A second TCN seed averaged in | -0.005 on 8 folds | Dropped |
| Training on smooth positions only | 8 folds, not registered: +0.02 if the real positions are smooth, -0.09 to -0.13 if not | Not used: choosing it would need a statistic of the provided match |
| No SkillCorner match in the team head | 8 folds, not registered: +0.035 | Not used: no unseen fold was left to test it on |
| PathCRF; StatsBomb open data | Not run; not used | Needs 22 identified players; its agreement bans redistribution |

## 5. Training data

| Dataset | Source and licence | Who made the labels, and how | Matches | In-play frames | Rate |
|---|---|---|---|---|---|
| IDSSE / DFL open data | figshare 10.6084/m9.figshare.28196177 (Bassek, Rein, Weber, Memmert, *Sci Data* 2025). CC BY 4.0; credit the authors, DFL and Sportec Solutions | DFL/TRACAB optical tracking, checked by live operators. Per-frame `BallStatus` 0 gives D; otherwise `BallPossession` gives H or A. The paper does not say how the possession field is derived | 7; 2022/23 Bundesliga and 2. Bundesliga | 1,000,331 | 25 Hz |
| Metrica sample data | github.com/metrica-sports/sample-data, commit `e706dd5`. No licence stated; the README asks users to acknowledge the source. Not redistributed | None supplied. We derive them from Metrica's own event tagging. Owner: the team of the last PASS, CARRY, RECOVERY or SET PIECE. D: from a dead trigger (ball out, foul, offside, hand ball, goal, shot out) to the next set piece or goal kick | 3; anonymised | 424,886 | 25 Hz |
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27`. MIT, (c) 2020 SkillCorner | SkillCorner's broadcast-video tracking. Per-frame `possession.group` gives H or A; null gives X (46% of in-play frames). No D. How the group is produced is not documented | 20; A-League 2024/25 | 1,182,244 | 10 Hz |

- **No hand labels.** Nobody on the team labelled or corrected any frame of any game. Every label is a
  provider field or an algorithm over provider data, and none comes from the provided match.
- **Use.** The alive head and the decoder priors use IDSSE and Metrica; the team head uses all three. In the
  development folds each model trains on the 27 other development matches. Two matches (one IDSSE, one
  Metrica) were the locked benches. The final model trains once on all 30: 657,133 LightGBM rows of 213
  columns.
- **Integrity.** All 111 raw files are SHA-256 pinned in `scripts/external_manifest.json`, and
  `scripts/build_external.py` downloads them and rebuilds the corpus.

## 6. Possession rules

We train on each provider's per-frame owner after one harmonisation, the **2 s sandwich merge**. An H or A
segment shorter than 2 s, with the other team on both sides, is relabelled to the surrounding team. It never
crosses D or X.

| Situation | CHALLENGE.md | Our labels |
|---|---|---|
| Brief opponent touch, deflection, block or duel; the ball returns to the owner | Not a change | Merged if under 2 s; a change if 2 s or longer |
| Brief opponent touch, then the ball goes straight out | Not a change | Kept as a change. Merging it was tested and missed the gate by a little |
| Pass in flight, loose ball | Stays with the last team | The last team (IDSSE, Metrica). SkillCorner gives X, which splits the possession |
| Restart | There is a clear possessor | The restart possession is never merged, however short |
| Ball out of play | D | D, from `BallStatus` (IDSSE) or events (Metrica); never merged |
| Who owns a dead ball | Not asked | Not predicted |

- **Why 2 s.** CHALLENGE.md calls "controlled" ambiguous. We trained with merge thresholds of 0, 1, 2, 3 and
  5 s and scored each against each: 2 s had the smallest worst-case loss. A 1 s threshold later passed our
  gate (+0.003) and we still kept 2 s, because 1 s loses more against the conventions closest to the written
  definition. The convention matters a lot: the same predictions score 0.495 to 0.653 across the conventions.
- **The output.** The decoder applies no explicit 2 s rule; it learns transitions from the labels. F3
  under-segments: 0.75 predicted possessions per true one. It never writes X on a graded frame.
- **Kickoff.** The second half is graded from before its kickoff (FAQ.md), and those frames are D. We take
  the kickoff from the provided Smart Tagging, never from a frame constant: in each graded half, the frames
  before that half's single `Kick Off (Start)` tag become D. Nothing changes when the tag is absent, repeated,
  outside the half or more than 180 s into it.

## 7. Limitations

- **Simulated noise is not the real ATD.** A classifier tells real tracking (SkillCorner, Metrica) from
  every degraded world with AUC 1.00, mostly on jitter, which the degrader makes 10 to 30 times rougher. One
  world in five of the training mix has realistic jitter. At inference the touch features read the real
  25 Hz ball, which no development world reproduces.
- **SkillCorner is 20 of the 28 folds**, it has no dead ball, and F3 gains least there.
- **The development estimate is optimistic.** The same 28 folds gated thirteen changes one after another, besides
  the first comparison of models and the label conventions. The two bench matches are the only ones no
  choice was made on, and only F was read on them.
- **The TCN seed matters.** With another seed, F's gain on 8 folds was about a quarter smaller. Seed 0 was
  fixed in advance and kept; the test was not repeated for F3.
- **Unknown label convention.** The absolute score is uncertain by about 0.15.
- **Short possessions.** At IoU 0.5, F3 finds 7% (IDSSE and Metrica) to 24% (SkillCorner) of the true
  possessions under 2 s, and 85 to 91% of those over 16 s.
- **Model bytes differ across machines; the labels did not.** The LightGBM model is byte-identical
  everywhere. The TCN's weights differ in their last bits with the OS and the CPU vendor. Trained from an
  empty cache on seven OS and CPU combinations (Windows, Ubuntu and macOS; Intel, AMD and Apple), and once
  more under Linux on an Intel CPU, F3 gave an external match in the provided format the same CSV, byte for
  byte. So expect the same file on another machine; that was shown on an external match, not on the provided
  one, where a near-tie could flip a label.

**With more time** we would make the simulated positions as smooth and correlated as real tracking, add a
model aimed at the team label, and test more of Smart Tagging than the set-piece tags.

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
  checks every file's SHA-256, builds the features and the noise worlds, trains the two models, labels the
  provided match, and runs the official checks on the exact bytes written. Only the last step opens the
  provided match, and it prints aggregates only. Without `make`, `docs/REPRODUCE.md` lists the same steps.
- **Needs.** [uv](https://docs.astral.sh/uv/), git, about 30 GB of disk and 16 GB of RAM, no GPU. On Linux,
  the system's OpenMP runtime (`apt-get install libgomp1`); the first step names it when it is missing.
- **Pinned stack.** Python 3.11, LightGBM 4.7.0, torch 2.15.0 (on the CPU), numpy 2.2.6, pandas 2.3.3,
  numba 0.68.1, scipy 1.17.1, scikit-learn 1.9.1, locked in `uv.lock`.
- **Determinism.** Fixed seeds, 8 LightGBM threads, 4 torch threads, torch's CPU kernels pinned before
  import. The CSV is built as bytes with LF line ends on every OS. On the reference machine a fresh clone
  with an empty cache rebuilt every file byte for byte.
- **Cost.** On a 4-core PC from 2017: about an hour to build the features and worlds, 20 minutes to train,
  a minute to label. The external download is about 5 GB.
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
