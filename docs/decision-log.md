# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-07: D1, native-frame-rate touch features (David)

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
