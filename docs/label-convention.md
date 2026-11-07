# Label convention (D4)

What "owning the ball" means for our training and evaluation labels, checked against the written
definition, with the evidence and a recommendation. External development matches only: the locked
benches (`DFL-MAT-J03WQQ`, `Sample_Game_2`) are excluded from every number here, and `data/` and
`evaluation/gt/` were not read. Reproduce with `scripts/label_evidence.py` and `scripts/label_tmix.py`
(commands at the end).

## Summary

- **The written definition is about controlled possession; our labels are not.** IDSSE's per-frame
  owner is touch-level, and the sandwich merge removes brief opponent flips that return to the owner,
  which matches the definition. It never removes a brief flip that runs into dead ball (a touch, then
  out). At T = 2 about 35 of those are left per IDSSE match (11% of spells), and raising T does not
  remove them. The new `tail` option does.
- **3.5 s and T = 2 answer different questions.** David's 3.5 s reproduces without the locked bench
  (pooled 3.5 s, leave-one-match-out 3.5-3.8 s, bootstrap 90% [3.5, 4.0] s). It is the flip length at
  which DFL's own events become more likely than not to record a deliberate action by the flipping
  team: a reading of "controlled" in DFL data. T = 2 is a training choice that loses least when the
  target's convention is unknown. Neither says what the target's corrected tracking does.
- **External data cannot identify the target's convention.** No external dataset is labelled by the
  corrected-tracking process behind the ground truth. The written definition is qualitative, and the
  public window may not be used to choose. Perfect labels in one plausible convention score only
  0.58-0.94 spellF1 against another, so the convention alone moves the absolute score by about 0.1.
- **Metrica's dead ball runs long** (D5 audit A3, confirmed): main's rule keeps D through live play
  after a trigger and misses goal kicks tagged BALL LOST. David's backward-from-restart rule makes the
  ball missing on 99.1% of Game 3's dead frames (main: 88.0%). Tested as candidate `MR`, section 5.
- **Training across T:** see section 5. TBD.
- **Recommendation:** TBD. Keep `protocol.T_MAIN = 2` until then.

## 1. What the upstream documents say

| Case | CHALLENGE.md / OVERVIEW.md |
|---|---|
| Change of possession | Only when a player of the other team gains **controlled** possession |
| What "controlled" means | Ambiguous by their own account. Suggested proxies: "a player keeps the ball for a few seconds", "a team completes more than two consecutive passes" |
| Not a change | "A simple touch of the ball without control", "a deflection of a pass that does not intercept it", "an aerial duel won without subsequent control of the ball by that team" |
| Ball loose or contested, pass in flight | "Possession stays with the last team that had it" |
| Restart after a stoppage | "There is a clear possessor" |
| Dead ball | `D`. The tracking still names a team ("whoever will restart, or touched it last"); the ground truth throws it away |
| How the ground truth is made | Mechanically from corrected tracking: `D` where it marks the ball dead, otherwise the team it names as owning. Nothing smoothed or hand-adjusted |
| Which reading wins | The 5 public minutes, "if your reading of the definition disagrees with it". RULES.md forbids training or tuning on them |

So the question is what the corrected tracking's owner field does, and the documents describe the
intended meaning (control), not the field's logic. "Touched it last" for dead balls hints that the
field may follow touches at least there.

## 2. How each external dataset's labels are made

| | IDSSE (6 development matches) | Metrica (2) | SkillCorner (20) |
|---|---|---|---|
| Owner | DFL/TRACAB per-frame `BallPossession` (always 1 or 2) | Derived from events: team of the last PASS, CARRY, RECOVERY or SET PIECE | Per-frame `possession.group` |
| Dead ball | Per-frame `BallStatus` = 0 | Derived: dead trigger until the next restart event | None |
| Unknown | Never | Outside the first and last event of a half | `group` null -> X (46% of in-play frames) |
| Same mechanics as the target? | Yes: status plus owner per frame | No: event-derived; D runs long (section 8) | No: no D, X splits spells |
| Raw spells, mean / under 2 s | 7.2 s / 30% | 10.7 s / 17% | 9.6 s / 22% |

**IDSSE is touch-level.** Around duels that DFL's own event operators scored as *no* possession change
(`TacklingGame PossessionChange="false"`, n = 742), the per-frame owner briefly flips within 1 s in 20%
of cases, against 7% at random alive frames. After duels scored as a change (n = 307), the winner owns
the ball 2 s later in 71%.

**SkillCorner's X breaks spells.** 54% of its raw spells touch an X frame. The scorer splits spells at X
in the ground truth, and the sandwich cannot merge across X. If X means a loose ball, that contradicts
"possession stays with the last team that had it". This does not touch the target, but 20 of the 28
development folds are SkillCorner.

## 3. The harmonisation against the definition

`katena.labels.sandwich(y, runs, fps, T)` relabels an H or A segment shorter than T that has the
other team on both sides (no D between) to that team, left to right, to a fixed point.

| Situation | Definition | Sandwich at T | Verdict |
|---|---|---|---|
| Brief opponent touch, deflection, block or duel, ball back to the owner | Not a change | Merged if shorter than T | Consistent; T sets "a few seconds" |
| Brief opponent touch, then the ball goes out | Not a change (touch without control) | **Never merged** | Gap: 35 per IDSSE match left at T = 2, 30 at T = 5 |
| Pass in flight, loose ball | Stays with the last team | The source decides: IDSSE and Metrica keep the last team; SkillCorner gives X | Consistent except SkillCorner |
| Duel won without subsequent control | Not a change | As the first row: merged if it returns within T | Consistent |
| Restart possessor, however short | A possession | Kept (needs the other team on both sides) | Consistent |
| Chain H, A 1 s, H 0.3 s, A 10 s | A gains control at the 10 s spell | Scan order: change at the 10 s spell. David's shortest-first: change at the 1 s touch | Scan order is the closer reading |
| Dead ball | `D` | Never changed | Consistent (IDSSE mechanics match the target) |

The touch-then-out gap is measurable: of DFL clearances that are followed by dead ball within 3 s
(n = 72), 26% show a brief flip to the clearing team just before the ball goes dead in the raw labels,
24% at T = 2 and 25% at T = 5. Long T also has a cost: after duels DFL scored as a possession change, the
winner owns the ball 2 s later in 73% at T = 2 but 64% at T = 5, so T = 5 erases real changes.

**New options (defaults unchanged, tested against the original code):**
`sandwich(..., tail=True)` also merges a short opponent flip that runs into dead ball or the end of the
run. Restart spells are still kept. `order="shortest"` merges the shortest flip first. With
`tail=True, order="shortest"` it reproduces David's `harmonise`.

## 4. Evidence: 3.5 s and T = 2

### David's control evidence, rerun without the locked bench

`katena.external.control_evidence` ports David's tool to `katena.types.Match` and Justin's loaders. It
reproduces his per-match flip counts, clock shifts, Play-team agreement and per-match best t exactly;
two held-out accuracies differ in the third decimal (the two loaders map event times to frames
differently by up to a frame, and the leave-one-out pool no longer holds the bench). A flip is a
spell that starts straight from the other team's spell. It has control evidence when its team makes a
DFL `Play`, `ShotAtGoal` or `BallClaiming` during it (plus or minus 0.4 s, after a per-match clock
shift). Six IDSSE development matches:

| Flip length | Flips | Share with a deliberate action by that team |
|---|---|---|
| 0 - 0.5 s | 135 | 0.05 |
| 0.5 - 1 s | 199 | 0.08 |
| 1 - 1.5 s | 210 | 0.16 |
| 1.5 - 2 s | 169 | 0.22 |
| 2 - 3 s | 275 | 0.36 |
| 3 - 4 s | 200 | 0.47 |
| 4 - 6 s | 264 | 0.72 |
| 6 - 10 s | 248 | 0.89 |
| 10 s and over | 349 | 0.99 |

| t (s) | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|
| Accuracy of "controlled if at least t" | 0.765 | 0.793 | 0.804 | **0.813** | 0.809 | 0.794 | 0.781 |

| Estimate | Without the bench (6 IDSSE) | David's (7 IDSSE incl. bench + Metrica Game 3) |
|---|---|---|
| Pooled best t | 3.5 s (accuracy 0.813) | 3.7 s (0.819) |
| Leave one match out | 3.5-3.8 s | 3.5-3.8 s |
| Match bootstrap, 90% | [3.5, 4.0] s | [2.7, 4.0] s |

Variants: only flips that return (what sandwich merges) 3.5 s [2.7, 3.8]; only flips into dead ball
4.7 s [3.5, 5.4]; completed passes, shots and claims only 4.6 s [3.8, 4.8]. Metrica's labels are made
from the same events, so its evidence is circular (pooled 3.1 s, Game 1 4.2, Game 3 2.7). It is kept out.

### What the two numbers measure

- **3.5 s estimates a convention:** where DFL flips become more likely than not to carry a deliberate
  action, if "controlled" means "made a pass, shot or claim". The stricter reading (completed actions)
  moves it to about 4.7 s, and "more than two consecutive passes" would move it further. In David's rule
  (merging flips into dead ball too, shortest first) 3.5 s gives labels about as smooth as sandwich
  T = 5 by spell count (231 against 230 per match), with far fewer short spells (3.7% under 2 s against
  16%). The two scales are not comparable.
- **T = 2 is a training choice:** the bake-off trained at each T and evaluated at each T, and T = 2 had
  the smallest worst-case regret (0.017). It assumes nothing about where the target sits in 0-5 s.
- **They do not conflict.** A 3.5 s convention is a candidate truth; T = 2 is what to train on when the
  truth is uncertain. Section 5 checks training against David's convention (`H3.5`) and the
  event-defined one (`E`) directly.

Spell statistics per convention, IDSSE development matches (`E`: every flip without an action removed,
no duration threshold, capped at 10 s):

| Convention | Spells per match | Mean s | Median s | Under 2 s | Short flips into dead ball per match |
|---|---|---|---|---|---|
| S0 (raw) | 443 | 7.2 | 3.6 | 30.3% | 40.5 |
| S1 | 388 | 8.1 | 4.4 | 23.0% | 37.8 |
| S2 (training default) | 324 | 9.7 | 5.7 | 15.1% | 35.2 |
| S3 | 282 | 11.1 | 6.8 | 15.8% | 33.3 |
| S5 | 230 | 13.6 | 8.5 | 16.0% | 30.3 |
| S2tail | 289 | 10.9 | 6.7 | 4.3% | 0 |
| H3.5 (David) | 231 | 13.6 | 9.1 | 3.7% | 0 |
| E (event-defined) | 222 | 14.2 | 9.6 | 5.1% | 0.2 |

### How far apart the conventions are

Perfect labels of one convention scored against another (spellF1, IDSSE development mean; the matrix is
symmetric):

| | S0 | S1 | S2 | S3 | S5 | S2tail | H3.5 | E |
|---|---|---|---|---|---|---|---|---|
| S0 | 1 | 0.923 | 0.811 | 0.725 | 0.603 | 0.749 | 0.610 | 0.578 |
| S2 | 0.811 | 0.882 | 1 | 0.898 | 0.756 | 0.938 | 0.779 | 0.709 |
| S3 | 0.725 | 0.788 | 0.898 | 1 | 0.839 | 0.834 | 0.839 | 0.740 |
| S5 | 0.603 | 0.660 | 0.756 | 0.839 | 1 | 0.690 | 0.752 | 0.707 |
| S2tail | 0.749 | 0.820 | 0.938 | 0.834 | 0.690 | 1 | 0.838 | 0.764 |
| H3.5 | 0.610 | 0.680 | 0.779 | 0.839 | 0.752 | 0.838 | 1 | 0.821 |
| E | 0.578 | 0.639 | 0.709 | 0.740 | 0.707 | 0.764 | 0.821 | 1 |

The nearest pure sandwich to both `H3.5` and `E` is S3. Merging the flips into dead ball (S2tail) brings
S2 closer to `E` (0.709 to 0.764) than any change of T does.

### What external data can and cannot identify

| Can identify | Cannot identify |
|---|---|
| How a provider's owner field behaves against its own event annotations (DFL: touch-level, flips at duels) | What the target's corrected tracking means by owning: touch-level, a time threshold, event-based control, or a manual call |
| Where DFL flips start to look controlled under an event definition (3.5 s; 4.7 s strict) | Whether the target merges a touch that goes straight out of play |
| How much each training choice loses when the evaluation convention differs (regret) | Which T, if any, the target uses, so which column of the regret table applies |
| That the ranking of models is stable across conventions | The absolute score level, which the convention alone moves by about 0.1 |

## 5. Training across T

**Setup** (`katena.eval.tmix`, `scripts/label_tmix.py`). Conventions only swap H and A, never D or X, so
the alive head is the same at every T: its M2 out-of-fold probabilities are reused, and only the team
head is refitted, on M2's rows in M2's order with M2's parameters and seed. Checks: the refitted T = 2
team head equals M2's cached one (and so do both heads refitted together); the re-score of M2 equals
`results.pkl` on all 28 folds (largest difference 0); pooled labels rebuilt from the native ground
truth equal the cache.

| Scheme | Team-head training labels |
|---|---|
| S0, S1, S3, S5 | One sandwich convention (references for regret) |
| **S2 = M2** | The incumbent, T = 2 |
| **MIX** | Each pooled row takes the label of one T drawn uniformly from {0, 1, 2, 3, 5}, fixed seed |
| **AUG** | Every row carries all five T at once: target = share of the five at which the focal team owns the ball (weighted 0/1 copies, same binary objective) |
| H3.5 | David's convention (tail flips merged, shortest first, 3.5 s) |
| S2tail | T = 2 plus the tail option |
| MR | T = 2, Metrica dead ball from restarts (both heads refitted; section 8) |

Decoding is M2's Viterbi with priors from the training labels' own convention (pooled over T for MIX
and AUG) and **tau fixed at 2** (D5 audit A2). Gate: at target-like noise against S2 ground truth,
wins on at least 70% of folds and the paired bootstrap interval excludes zero. Regret: for each
evaluation convention, the best scheme's mean minus this scheme's mean; worst case over S0-S5. Every
table is shown for all folds and for the 8 IDSSE + Metrica folds (D5 audit A18), because SkillCorner
has no dead ball and splits spells at X.

**Decoder priors alone do nothing.** M2's heads decoded with the priors of S5 or H3.5 move every score
by less than 0.004 (28 folds), so a known target convention would have to be matched through the
training labels, not the decoder.

### Results: 8 IDSSE + Metrica folds (screen), target-like, tau = 2

Mean spellF1; rows are the training scheme, columns the evaluation convention (E: 6 IDSSE folds).

| Train \ eval | S0 | S1 | S2 | S3 | S5 | S2tail | H3.5 | E |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.468 | 0.508 | 0.573 | 0.618 | 0.629 | 0.607 | 0.673 | 0.686 |
| S1 | **0.468** | **0.509** | **0.574** | 0.619 | 0.628 | **0.608** | 0.675 | 0.683 |
| **S2 = M2** | 0.463 | 0.504 | 0.569 | 0.619 | 0.633 | 0.603 | 0.676 | 0.686 |
| S3 | 0.451 | 0.490 | 0.557 | 0.613 | 0.636 | 0.591 | 0.669 | 0.679 |
| S5 | 0.422 | 0.460 | 0.530 | 0.591 | **0.638** | 0.564 | 0.652 | 0.659 |
| MIX | 0.457 | 0.497 | 0.563 | 0.616 | **0.638** | 0.597 | 0.672 | 0.680 |
| AUG | 0.452 | 0.492 | 0.558 | 0.612 | 0.636 | 0.591 | 0.669 | 0.678 |
| H3.5 | 0.445 | 0.484 | 0.553 | 0.607 | **0.638** | 0.588 | 0.670 | 0.678 |
| S2tail | 0.461 | 0.502 | 0.570 | **0.622** | 0.637 | 0.604 | **0.680** | **0.692** |
| MR | 0.462 | 0.503 | 0.567 | 0.619 | 0.634 | 0.602 | 0.673 | 0.691 |

TBD: 28 folds, gate and regret tables.

## 6. Recommendation

1. **Keep `protocol.T_MAIN = 2` and single-convention training.** Mixing T (MIX) and soft-T
   augmentation (AUG) lose at the T = 2 gate and raise the worst-case regret. The model's output is
   already smoother than its training labels (M2 scores 0.569 against S2 but 0.676 against H3.5 on the 8
   folds), so blurring the team head's targets buys no robustness.
2. **Do not train on the 3.5 s convention.** Trained on David's labels, the model loses at T = 2 and
   also against its own ground truth. If the target were 3.5 s-like, T = 2 training would still be the
   better choice. That is the reconciliation: 3.5 s is a defensible *description* of controlled
   possession in DFL data; T = 2 is the better *training* convention under it and under the sweep.
3. **Training at T = 1 is not a reason to move T_MAIN.** It wins at the T = 2 gate on the 8 folds, but
   doubles the worst-case regret and loses where the written definition points (S5, H3.5, E).
4. **`tail=True`: not adopted, keep ready.** A tie at T = 2, and better against the smoother conventions
   on IDSSE. Switch if the organisers answer "no" to question 2 below.
5. **Metrica stoppages from restarts: not adopted under the rule, but the labels are cleaner.** Neutral
   on the gate. Adopting it would be a data-quality decision that moves the Metrica development score
   (M2: 0.643 against the default labels, 0.625 against the restart-rule labels); the team should make
   it explicitly, not as a score change.
6. **Ask the organisers (section 7)**, and treat the absolute score as uncertain by about 0.1 from the
   convention alone: the same M2 predictions score 0.50 (S0) to 0.65 (H3.5) on 28 folds.
7. **For the technical document:** say we train on per-frame owner labels with opponent touches shorter
   than 2 s that return to the owner merged, restart possessors kept, dead ball from ball status (IDSSE)
   or derived from events (Metrica), and that the convention was chosen for robustness, not fitted to
   the target.

## 7. Question for the organisers

This replaces the second half of question 3 in `docs/PLAN.md`. It asks how the owner field works, not
for any statistic of the ground truth, and each part has a short answer:

> In the corrected tracking from which the ground truth is derived, when does the team that owns the
> ball change during live play?
>
> 1. A defender touches the ball without controlling it (a block, a deflection, a header in a duel) and
>    the attacking team gets it straight back. Does the owning team change for the length of that touch,
>    or stay with the attacking team?
> 2. The same touch sends the ball straight out of play. Does the owning team switch to the defender's
>    team before the ball is marked dead?
> 3. If control is decided by a rule, which one: a minimum time on the ball (how many seconds), a number
>    of touches or passes, or an operator's judgement in each case?

How the answers map to our options: "changes" to 1 means touch-level, train near T = 0. "Stays" to 1
plus a number of seconds in 3 sets T. "No" to 2 means `tail=True`. "An operator's judgement" means an
event-based convention like `E`.

## 8. Metrica dead ball (D5 audit, finding A3)

Main's Metrica rule makes the ball dead from a trigger event until the next SET PIECE or a PASS tagged
GOAL KICK. The event vocabulary shows three ways that runs long: a goal kick tagged `BALL LOST | GOAL
KICK` (11 in the two development games) does not end the stoppage; `BALL OUT` starts D at the event's
start (the last touch) rather than when the ball is out; and a trigger with no tagged restart keeps
the ball dead through live play. It also leaves an untagged stoppage alive.

`katena.external.metrica_stoppages` ports David's rule, which anchors every stoppage on its restart
and looks backwards for the first stop marker after the last in-play action (or that action's end).
Ownership and windows are main's. Metrica's open tracking removes the ball while it is out of play,
so the share of D frames with the ball missing measures how clean a D label is:

| Game | Rule | Dead share | Ball missing on D | Ball present when alive | Dead span median / p90 | Spells (raw) |
|---|---|---|---|---|---|---|
| 1 | default | 51.3% | 0.731 | 0.992 | 28.0 / 66.7 s | 276 |
| 1 | restart | 42.0% | 0.850 | 0.963 | 20.4 / 48.1 s | 359 |
| 3 | default | 34.6% | 0.880 | 0.980 | 13.5 / 49.6 s | 330 |
| 3 | restart | 30.6% | **0.991** | 0.980 | 11.8 / 40.5 s | 383 |

On Game 3 this reproduces the audit (David's own loader: 0.992; the 7,215 frames that only the default
calls dead have the ball present 77% of the time). On Game 1 the dead side improves too, but some
frames the restart rule calls alive have no ball (0.992 to 0.963): a stoppage with no marker starts
when the last action ends, which can be late. The two rules agree on 90.5% (Game 1) and 93.9% (Game 3)
of frames and on the team on 99.9-100% of frames both call alive. Scored against each other at T = 2
they reach only 0.83 and 0.89 spellF1, so the choice moves every Metrica number. Metrica is 2 of 28
development folds but about a quarter of the alive head's training rows.

The gate for this candidate (`MR`: both heads refitted on the restart-rule labels) is in section 5.
IDSSE and SkillCorner folds keep their ground truth, so they are the clean test; Metrica folds are
scored against both rules.

## Reproduce

```bash
export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-d4
uv run python scripts/label_evidence.py --metrica      # sections 2-4, about 1 min
uv run python scripts/label_tmix.py --folds 8 --check  # section 5 screen (needs the bake-off cache)
uv run python scripts/label_tmix.py --folds 28
```
