# katenaball: technical document (DRAFT)

> Draft for the submission form's technical document. TODO markers are facts that can only be filled in
> after the human-run steps on the real match: running time, the submission hash and the aggregate checks.
> Every number below comes from external matches.

## 1. The approach

We label every frame of the tactical video H, A or D in four steps. We read the automatic tracking (ATD)
at 25 Hz and compute features on a 5 Hz grid. Two models give, per sample, the probability that the ball
is alive and the probability that the home team owns it. A three-state Viterbi decoder turns those
probabilities into possession spells. We then expand the spells back to video frames.

**The models.** We submit candidate F (`docs/decision-log.md`), an equal blend in logit space of two
members:

- **LightGBM.** Two gradient-boosted heads, alive/dead and home/away, on 213 columns per sample. 160
  describe the ball, both teams' shape and pressure, dead-ball cues and missing data, computed twice with
  each team as the focal team and averaged. The other 53 are touch and turnover features computed at the
  ATD's native 25 Hz, from breaks in the ball's velocity and the nearest player at each break.
- **TCN.** A temporal convolutional network (6 dilated blocks, dilations 1-32, 48 channels) on the 160
  base columns, run over each half as one sequence.

**The decoder.** Viterbi over {H, A, D} at temperature 2. Its transition priors come from the external
labels.

**Why this, rather than something else.** We tested every choice before adopting it, on 28 external
matches with leave-one-match-out folds. We degraded the tracking to look like the real ATD (§3) and scored
with the official `spellF1`.

| Model | spellF1 (28 folds, target-like noise) |
|---|---|
| Official baseline algorithm (nearest player, smoothed) | 0.280 |
| Hand-written rules | 0.432 |
| LightGBM + Viterbi (M2) | 0.586 |
| **F: M2 + touch features + TCN blend + realistic noise mix** | **0.602** (+0.016 [+0.011, +0.021], wins 22 of 28) |

**What we dropped:**

- A semi-Markov decoder (−0.001).
- Spell rescoring (+0.001, CI includes 0).
- Training labels with a 3.5 s control threshold (−0.008).
- PathCRF, which needs identified players.

The rule fixed before each test was to adopt only with wins on at least 70% of folds and a paired 95%
bootstrap CI above 0, and only if the change was not worse at severe noise.

## 2. Which assets we used, and how

| Asset | Used | How |
|---|---|---|
| ATD, tactical source (metadata XML and raw tracking) | Yes | The only input to the model |
| Half blocks from `games.py` | Yes | Which frames are inside a half (X outside) |
| Homography | No | The ATD is already in pitch coordinates |
| Videos (tactical and broadcast) | No | TODO: a broadcast cut-away dead-ball prior was considered (J6) |
| Smart Tagging | No | TODO: J5 tested learned tag features |
| Public 5-minute ground truth | Pipeline check only | Never used to train, tune or choose between versions |

## 3. Training data

None of it comes from the provided match. Every label is from a third party; nobody on the team labelled
anything.

| Dataset | Matches | Labels | Who labelled | Licence |
|---|---|---|---|---|
| IDSSE / DFL open data (Bassek et al., Sci Data 2025) | 7 (Bundesliga and 2. Bundesliga 2022/23) | Per-frame `BallPossession` and `BallStatus` at 25 Hz | DFL / Sportec Solutions operators; TRACAB optical tracking | CC BY 4.0 |
| Metrica Sports sample data | 3 | H/A/D derived by us from Metrica's event tagging, by a fixed rule (`docs/external-data.md` §2) | Metrica event taggers | None stated; credited, not redistributed |
| SkillCorner open data | 20 (A-League 2024/25), 10 Hz broadcast tracking | `possession.group` (H/A; no dead ball) | SkillCorner computer vision | MIT |

**Training worlds.** Clean tracking would not look like the real ATD, so we corrupt the external tracking
with a degrader before training:

- **Defects added:** ball loss, false ball detections, position jitter, fragmented and swapped player
  tracks.
- **Severities:** mild, target-like and severe.
- **Calibration:** set from the published ATD statistics and the external matches only, never from the
  provided match.
- **The `d2j` mix:** also trains on ball loss tied to stoppages, identity swaps and smooth positions.
  These came from what external tracking looks like, again never from the provided match.

The final model trains on all 30 external matches, 657,133 rows.

## 4. The possession rules we implemented

The ground truth is mechanical: D where the corrected tracking marks the ball dead, otherwise the team the
tracking names as owning it. Our training labels follow the sources and add one rule:

- **Dead ball.** D exactly where the source says the ball is out of play (IDSSE `BallStatus`). For Metrica,
  D runs from a stoppage event to the next restart.
- **Brief opponent touches.** An opponent spell shorter than 2 s that sits between two spells of the same
  team, with no dead ball between, is relabelled to the surrounding team. This applies the
  "controlled" definition in `docs/upstream/CHALLENGE.md`: a touch without control is not a change of
  possession. Other thresholds (0-5 s) were tested. 2 s had the smallest worst-case loss when the
  convention is unknown (`docs/label-convention.md`).
- **Restarts.** The team taking a restart owns the ball from the restart. A spell after dead ball is
  never relabelled.
- **Loose balls and challenges.** We have no explicit rule. The decoder keeps the last owner until the
  evidence for a change outweighs the prior, which in effect gives a contested ball to the side that had
  it.
- **Frames outside the halves** are X.

## 5. Limitations

- **Short possessions.** Spells under 8 s are 75-85% of the true possessions we miss. Even on clean
  tracking the model finds about 0.7-0.9 predicted spells per true one.
- **Domain gap.** Real tracking is far smoother than our target-like noise (frame-to-frame jitter about
  0.05 m against about 1 m), so external scores may understate or misstate the real one.
- **Label convention.** How the organisers' corrected tracking defines "owning" moves scores by up to
  0.1 for the same model, and it cannot be measured externally.
- **Pre-kickoff frames.** The graded second half starts before its kickoff. Those frames are D in the
  ground truth, and the model must find that from the tracking alone. We use no frame constants from
  this match.
- **TODO:** what we would fix with more time (gap filling for the 41% of graded frames without a ball,
  a kickoff detector, Smart Tagging).

## 6. Reproduction

```bash
make submit TEAM=<team_name>
```

Full details: `docs/REPRODUCE.md`. Every dependency is pinned in `uv.lock`, seeds are fixed, LightGBM and
torch run deterministically, and the CSV is written as bytes with LF endings. TODO: the submission sha256
from two machines.
