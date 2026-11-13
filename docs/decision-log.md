# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-13

### KIN on the Mac: the interim read (information only; the registered gate is running on David's side)

**With F3's TCN the 35 columns add about +0.009 for both bases, on both kinds of fold. With A's own 4,800-step
TCN, on the 20 folds the Mac has of it, the gain at `target` is +0.0018 with an interval that includes 0. The
second reading is the nearer one to the registered test, so this does not say KIN will pass. Nothing is decided
or adopted.** Claude on Justin's Mac, read at 11:00 CET on Nov 13; Justin has not read it. Registered as
information in the entry below before it was read. `scripts/run_kin.py kin-gbdt kin-gate` at `8efd184`, the
Mac's own cache; 56 team heads, about 48 s each, 45 minutes. External development matches only.

| `target`, spellF1 | Folds | With the columns | Base | Diff [95% CI] | Wins |
|---|---|---|---|---|---|
| **AK against A, F3's TCN**, all 28, action truth on SkillCorner | 28 | 0.6454 | 0.6368 | +0.0086 [+0.0050, +0.0122] | 24/28 |
| The same, 8 IDSSE + Metrica folds | 8 | 0.6675 | 0.6606 | +0.0069 [+0.0002, +0.0131] | 7/8 |
| The same, 20 SkillCorner folds, action truth | 20 | 0.6366 | 0.6273 | +0.0093 [+0.0050, +0.0137] | 17/20 |
| **AWK against AW, F3's TCN**, all 28, action truth on SkillCorner | 28 | 0.6637 | 0.6542 | +0.0095 [+0.0060, +0.0131] | 23/28 |
| The same, 8 IDSSE + Metrica folds | 8 | 0.6944 | 0.6814 | +0.0131 [+0.0062, +0.0198] | 8/8 |
| The same, 20 SkillCorner folds, action truth | 20 | 0.6514 | 0.6434 | +0.0080 [+0.0038, +0.0124] | 15/20 |
| **AK against A, A's own 4,800-step TCN**, the 20 folds the Mac has | 20 | 0.6781 | 0.6763 | +0.0018 [-0.0021, +0.0059] | 14/20 |
| The same, 8 IDSSE + Metrica folds | 8 | 0.6785 | 0.6763 | +0.0022 [-0.0044, +0.0086] | 6/8 |
| The same, 12 SkillCorner folds, action truth | 12 | 0.6778 | 0.6762 | +0.0016 [-0.0032, +0.0066] | 8/12 |

| `severe`, all folds, action truth on SkillCorner | Folds | Diff [95% CI] | Wins |
|---|---|---|---|
| AK against A, F3's TCN | 28 | +0.0126 [+0.0088, +0.0165] | 23/28 |
| AWK against AW, F3's TCN | 28 | +0.0113 [+0.0072, +0.0153] | 23/28 |
| AK against A, A's own 4,800-step TCN | 20 | +0.0088 [+0.0048, +0.0128] | 14/20 |

- **Beside a TCN that never saw an action label, the columns help everywhere**: every row with F3's TCN has its
  interval above 0, at both severities, and 23 or 24 of 28 folds are won. Possessions of 2 to 4 s found at IoU
  0.5 go from 0.31 to 0.35 (A) and from 0.42 to 0.47 (AW) on SkillCorner against the action truth, and from 0.41
  to 0.46 (AW) on the 8 folds. Predicted possessions per true one rise by 0.02.
- **Beside A's own TCN most of the `target` gain is gone**, +0.0018 where it was +0.0086. A TCN trained on the
  action labels already turns at a change of owner, and the blend takes half of what the team head adds. The
  gain at `severe` stays (+0.0088). The Mac has this TCN on 20 of 28 folds (all 8 IDSSE + Metrica, 12 of 20
  SkillCorner); the rest are training.
- **The LightGBM member alone** (`target`, 28 folds, action truth on SkillCorner): AK against A +0.0080 [+0.0037,
  +0.0124], 17/28; AWK against AW +0.0018 [-0.0021, +0.0058], 13/28. With the weights the columns gain on the 8
  folds (+0.0134 [+0.0057, +0.0213]) and not on SkillCorner (-0.0029 [-0.0072, +0.0017]).
- **On the standing SkillCorner truth nothing moves** (all 28, `target`, F3's TCN: +0.0006 and +0.0005).
- Against F3 every reading above meets rule 2's three tests. That is the base's doing, not the columns'.
- Check: F3's scores in this gate equal the stored ones, 56 of 56.

**What it does not show.** The registered test: AK2 against A2 and AWK2 against AW2 with the 9,600-step TCNs,
which the Mac does not have. Anything about AWK beside AW's own TCN. Anything about the provided match.

**The registered run.** David's side dispatched it at `8efd184` (GitHub run 43399700582, 4 jobs) and will gate
on the PC. This entry will be amended with the 28-fold reading beside A's own TCN when the Mac's run finishes.

### KIN (pre-registration): 35 ball-free timing columns in the LightGBM team head of A and of AW

**Written at 10:15 CET on Nov 13, before any score of it.** Claude on Justin's Mac. Justin asked on Nov 12 for
this work to start at 01:00 his time and has not read this; David has not either. The 20th and 21st packages on
the same 28 folds. Code: `src/katena/featurize/kin.py`, `src/katena/eval/kin_gate.py`, `scripts/run_kin.py`,
`tests/test_kin.py`. External development matches only. It adopts nothing.

**Why.** The research read of Nov 12 (entry below) put F3's loss in the team head just after the owner changes:
team accuracy 0.988 more than 4 s from both ends of a possession, 0.49 in the first second after a direct change
and 0.68 in the next. The columns the team head reads describe team movement with windows centred on the sample
(displacement from 2 s before to 2 s after, and from 5 s before to 5 s after). There is no one-sided window, no
acceleration, and no rate of the lines or of the spread. A tree reads one row at a time and cannot difference
neighbouring rows, and a centred window says the same thing at a change as half a second either side of it. Both
teams turn towards the same goal when the ball changes owner, with or without a ball detection, and the same
read found the team probability slowest to turn where the ball was not in view (with S1's and S2's weights it
crossed 0.5 within 0.4 s of a true change on 0.48 of IDSSE's changes with the ball in view and on 0.30 without).
The TCN reads the sequence and can difference it by itself, so it is left alone.

**The change, and nothing else.** `katena.featurize.kin` computes 35 columns from player positions only, in the
focal team's frame, each half by itself:

- the team's mean slot displacement along x over the second before and the second after the sample, the same
  over 2 s, for each team; after minus before for each team and for both together (14 columns);
- the share of a team's slots moving forward minus the share moving back, 1 s before and 1 s after (4);
- the rate of each team's defensive line, 1 s before and after (4); the rate of each team's length and width,
  2 s before and after (8);
- the distance of the closest pair of opposing players, and its mean over the second before and after (3);
- each team's mean speed in the second after minus the second before (2).

A slot counts in a displacement only where it is seen at both ends and did not switch player in between. The
constants are fixed here: positions averaged over 1 sample either side before differencing; a slot moves
"forward" above 1.0 m/s; movement above 12.5 m/s is blanked (`build.py`'s value); a slot that lies more than 5 m
from its previous sighting at more than 12.5 m/s has switched player; a team value needs 3 slots.

The columns go into the LightGBM team head and nowhere else. Two team heads are fitted on every fold with the 213
+ 35 columns: **AK** on A's rows and labels, unweighted, and **AWK** with AW's possession-length weights. Rows,
labels, weights, parameters, 1,200 trees and seed are the base candidate's. The alive head, the TCN and the
decoder are the base candidate's, read from its cache. No constant, window or column is changed, dropped or added
after a score, and no subset of the columns is tried, whatever comes out.

**Rule, for each of the two against its own base** (the rule of A2 and AW2, entry below). With the base's
9,600-step TCN: AK2 against A2 and AWK2 against AW2. On the 28 folds with the action truth on SkillCorner: at
`target` at least 70% of folds won and the paired 95% interval above 0; not worse at `severe`. And not worse at
`target` on the 8 IDSSE + Metrica folds, nor on the 20 SkillCorner folds. A candidate that passes must also meet
rule 2 against F3, as its base does. AK2 can only replace A2 and AWK2 only AW2. Which of A and AW stands is not
settled here and is not reopened by this: a pass for one base and a fail for the other gives the columns to that
base only.

**A pass is not an adoption.** Unlike the longer TCN, this adds code to the inference path: the columns would be
computed on the provided match's player tracks. Before it could be adopted it needs the pipeline's `train` and
`predict` to carry the columns, the same model and CSV on the three OSes and the PC, and a human's look at
aggregates on the provided match (the share of finite values of each column, against the external worlds). The
cut-off is Nov 14, 18:00 CET. The two who decide may leave a passing KIN out for lack of time.

**Where it runs.** The registered run is the one A2 and AW2 had: the team heads fitted on GitHub's runners by
`gate.yml` (tag `gha`), the gate on the PC, where the bases' members, the 9,600-step TCNs and F3's cache are.
The Mac has no 9,600-step TCN. Tonight the Mac fits the same team heads on its own cache and reads them with
F3's TCN (`AK_f` against `A_f`, `AWK_f` against `AW_f`), and with A's own 4,800-step TCN if that run finishes:
**information only, read before the registered gate, and named here so that it counts as seen.**

```
# runners (4 jobs, 7 folds each; every job builds the columns of all 28 matches, then fits its folds)
gh workflow run gate.yml --ref <branch> -f script=run_kin.py -f stages="kin-gbdt" \
  -f variants="gha_ak_gbdt1200ak gha_awk_gbdt1200awk" -f shards=4 -f extra="--tag gha --workers 4"
# the gate, on the PC, after the two folders are in oof/
uv run python scripts/run_kin.py kin-gate --tag gha --inc f3_gbdt,f3_tcn
# the Mac, tonight
uv run python scripts/run_kin.py kin-build --workers 4 && uv run python scripts/run_kin.py kin-check kin-gbdt kin-gate
```

A file of `run_kin.py` holds a team head's probabilities only. The gate puts them in place of the base's team
probabilities and takes the alive probabilities and the TCN from the base's files, so a runner needs no cache of
the base.

**Checks, none of which is a score.**

- `tests/test_kin.py`, synthetic matches only: the one-sided velocities reverse at a scripted turn, and the
  reversal peaks within a sample of it; a slot that is lost, found again, or taken over by a player 30 to 40 m
  away is not movement; swapping the teams and rotating the pitch gives the other focal's columns; rows line up
  with `featurize`; one half does not leak into the other; no column changes when the ball is moved or removed.
- `kin-build`: every world's corrupted match is recovered with the corpus seeds and checked against the cached
  features of that world (rows, M0's predictions, the degrader's statistics, the attack directions) before its
  columns are computed. Done on the Mac at 10:08 CET: 300 worlds of the 28 matches, all
  recovered; pool 614,802 rows by 213 + 35 columns; no column is constant. On IDSSE and Metrica every column is
  finite on every pool row. On SkillCorner 21% of pool rows have no value in any of the 35 (fewer than 3 slots
  of a team in view), and the columns are finite on 0.70 to 0.77 of rows, as the base columns `vx_f_2s` (0.73)
  and `vx_f_5s` (0.66) are.
- `kin-check` on fold `idsse/DFL-MAT-J03WMX`: done on the Mac at 10:10 CET. On F3's 213 columns the script's
  team-head fit gives A's cached team probabilities byte for byte, and with the weights AW's, at both gate
  worlds. So the only difference between AK and A, and between AWK and AW, is the 35 columns. The pool's rows of
  that match equal the per-world rows they were drawn from (21,842 of 21,842).

**Already seen when this was written** (so nothing below is blind to them): the three research reads of Nov 12,
which are where the idea and the 1 s and 2 s windows come from; A's interim gate on the Mac; K at kind 0, S1 and
S2 against the action truth; the transfer probe; and the results of A, AW, AT, ATW, A2 and AW2 in the entries
above. No model with these columns has been fitted and no score of one exists.

**Information, decides nothing:** the LightGBM member alone; the reads with F3's TCN and with the 4,800-step
TCNs; each against F3; possessions found by true length; predicted possessions per true one.

### A2 and AW2 on four machines, and the pipeline's training against the gate's (result)

**Trained on all 30 matches on this PC and on macOS, Ubuntu and Windows runners, A2 and AW2 each give one
LightGBM model and one dressed CSV. And the pipeline trains, byte for byte, what the gate trained.** Nov 13,
08:01 to 09:31 CET. External matches only. Nothing is adopted by it.

**Four machines** (`repro.yml` at `761a599`, runs 43382929535 for A2 and 43382925800 for AW2; the PC's fits by
`D:/katena/tools/pc_ref.sh` in `D:/katena/work-submit`, beside F3's model).

| | A2 | AW2 |
|---|---|---|
| Machines | This PC (Windows 10, Intel i7-7700); macOS 15 on Apple M1; Ubuntu 24.04 on AMD EPYC 7763; Windows Server 2025 on AMD EPYC 9V74 | This PC; macOS 15 on Apple M1; Ubuntu 24.04 on AMD EPYC 9V74; Windows Server 2025 on AMD EPYC 7763 |
| LightGBM member | `eb1030c8ec94...` on all four: A's | `55de693c7ea2...` on all four: AW's |
| TCN weights | four sets (the PC's: `0f5b9b27ea54...`) | four sets (the PC's: `78a241f571ad...`) |
| Dressed CSV | `edb8c25495976693...` on all four | `4e0ab1d06ba721a0...` on all four |
| `bundle.pkl` on the PC | `dfda7e14c88242e6...` | `9b4c2cfaa553af43...` |
| `train` on the PC | 27 minutes | 26 minutes |

The LightGBM member does not change with the TCN's steps, so it is A's and AW's, which four machines had
already agreed on. The TCN's bytes differ by OS and CPU, as always, and no label changes with them.

**The pipeline's training against the gate's** (`D:/katena/tools/a_equiv.py`, on the PC, in the gate's cache).
`katena.pipeline.stage_train` with spec A on the 28 development matches, `idsse/DFL-MAT-J03WMX` held out, gives
the hashes the gate's fold stored: LightGBM `fad2db189fbe...` and TCN `20c48773980a...`, both equal. So
`katena.eval.actlabels` hands the final model the labels, the rows and the sequences that `run_a.py` handed the
gate. The weights' path (AW) has the synthetic test of `tests/test_pipeline.py` and no fold of this kind yet: no
AW fold has been fitted on the PC.

**Running on the PC until about 13:15:** for AW2 and then A2, the checks F3 had before its adoption (the same
machine twice, the held-out model down both paths, other torch kernels) and the stress check.

### A2 and AW2: result; and the PC's own all-30 fits of A and AW (Claude on David's PC; neither David nor Justin has read it)

**Both pass their registered rule: a TCN trained for 9,600 steps is ahead of the same candidate at 4,800 on
every kind of fold. So the candidate that stands is AW2, AW with the longer TCN. Nothing is adopted here.** Read
at 08:00 CET on Nov 13. `gate.yml` at `b5a8afd` (runs 43367189118 and 43367192771), 28 folds each on Ubuntu
runners, about 12 minutes a fold; the gate on the PC, `scripts/run_x2.py --tag gha --inc f3_gbdt,f3_tcn`, log
`D:/katena/logs/a-win/gha_x2_gate.log`. External development matches only; nothing in the registration changed.

| The longer TCN against its own base | A2 minus A | AW2 minus AW |
|---|---|---|
| **28 folds, action truth on SkillCorner, `target` (rule)** | **+0.0067 [+0.0040, +0.0093], 23/28** | **+0.0071 [+0.0046, +0.0096], 20/28** |
| The same at `severe` (rule: not worse) | +0.0082 [+0.0050, +0.0114], 24/28 | +0.0077 [+0.0047, +0.0106], 24/28 |
| 8 IDSSE + Metrica folds, `target` (rule: not worse) | +0.0075 [+0.0034, +0.0120], 7/8 | +0.0070 [+0.0018, +0.0118], 6/8 |
| 20 SkillCorner folds, action truth, `target` (rule: not worse) | +0.0064 [+0.0031, +0.0095], 16/20 | +0.0071 [+0.0044, +0.0099], 14/20 |
| Verdict | Passes: A2 replaces A | Passes, with exactly the 20 folds the rule asks for: AW2 replaces AW |

| spellF1, `target` | F3 | A | A2 | AW | AW2 |
|---|---|---|---|---|---|
| 8 IDSSE + Metrica folds | 0.6374 | 0.6763 | 0.6838 | 0.6923 | **0.6993** |
| 20 SkillCorner folds, action truth | 0.6094 | 0.6647 | **0.6712** | 0.6586 | 0.6657 |
| All 28, action truth on SkillCorner | 0.6174 | 0.6680 | 0.6748 | 0.6682 | 0.6753 |
| IDSSE (6) | 0.6024 | 0.6453 | 0.6571 | 0.6725 | 0.6814 |
| Metrica (2) | 0.7427 | 0.7692 | 0.7642 | 0.7517 | 0.7530 |

- **The gain is about 0.007 everywhere**, on the 8 folds and on SkillCorner alike, at both severities. It is the
  first change since `LT` to win folds of both kinds, and it is the same kind of change.
- **Against F3 the candidates now stand at** +0.0464 (A2) and +0.0618 (AW2) on the 8 folds, 8 of 8 each, and
  +0.0617 and +0.0563 on SkillCorner against the action truth (20 and 18 of 20).
- **The weights read the same at 9,600 steps** (information; the registered test was at 4,800): AW2 minus A2 is
  +0.0155 [+0.0032, +0.0275] on the 8 folds (6 of 8), -0.0055 [-0.0131, +0.0024] on SkillCorner against the
  action truth, and -0.0103 [-0.0183, -0.0025] there at `severe`. They gain on IDSSE and lose on the other two.
- The longer TCN leaves the count alone: 0.80 and 1.02 predicted possessions per true one for A2, 0.98 and 1.23
  for AW2, on the 8 folds and on SkillCorner.
- As registered, no other step count is tried.

**A2 and AW2 against F3, with intervals** (the same score file, read on the PC at 08:20 CET; for the documents).

| | A2 against F3 | AW2 against F3 |
|---|---|---|
| 8 IDSSE + Metrica folds, `target` | 0.6838 vs 0.6374, +0.0464 [+0.0355, +0.0575], 8/8 | 0.6993 vs 0.6374, +0.0618 [+0.0462, +0.0772], 8/8 |
| The same at `severe` | 0.6357 vs 0.5900, +0.0457 [+0.0351, +0.0564], 7/8 | 0.6581 vs 0.5900, +0.0681 [+0.0535, +0.0830], 7/8 |
| 20 SkillCorner folds, action truth, `target` | 0.6712 vs 0.6094, +0.0617 [+0.0528, +0.0706], 20/20 | 0.6657 vs 0.6094, +0.0563 [+0.0455, +0.0666], 18/20 |
| The same at `severe` | 0.6207 vs 0.5615, +0.0593 [+0.0505, +0.0682], 20/20 | 0.6105 vs 0.5615, +0.0490 [+0.0392, +0.0595], 19/20 |
| All 28, action truth on SkillCorner, `target` | 0.6748 vs 0.6174, +0.0573 [+0.0503, +0.0644], 28/28 | 0.6753 vs 0.6174, +0.0579 [+0.0494, +0.0657], 26/28 |
| Rule 1: 28 folds, standing truth, `target` | 0.6120 vs 0.6180, -0.0060 [-0.0123, +0.0002], 10/28 | 0.5996 vs 0.6180, -0.0184 [-0.0264, -0.0108], 8/28 |
| Rule 1: the 20 SkillCorner folds, standing truth | 0.5833 vs 0.6103, -0.0270 [-0.0349, -0.0190], 2/20 | 0.5597 vs 0.6103, -0.0505 [-0.0598, -0.0412], 0/20 |

Both still fail rule 1 and meet rule 2. The weights at 9,600 steps, by the process that made the truth (AW2 minus
A2, `target`): +0.024 on the six IDSSE folds, -0.011 on the two Metrica folds, -0.006 on SkillCorner against the
action truth. Fit times of the PC's all-30 AW: LightGBM 165 s, the TCN 716 s.

**The PC's own fits of A and AW on all 30 matches** (`D:/katena/tools/pc_ref.sh`, in `D:/katena/work-submit`
beside F3's model, which is untouched; 07:29 to 08:00 CET).

| | A | AW |
|---|---|---|
| LightGBM member | `eb1030c8ec94...`: the three runners' | `55de693c7ea2...`: the three runners' |
| TCN weights | `209beece8bf2...`: its own | `7f21c128504d...`: the Windows runner's (also Intel) |
| `bundle.pkl` | its own | `33bae152771c...`: the Windows runner's, byte for byte |
| Dressed CSV | `98b2919621653c99...`: the three runners' | `280060af276f1665...`: the three runners' |
| `train` | 15 minutes | 15 minutes |

So for each of the two, four machines (this PC and one runner each of macOS, Ubuntu and Windows) give one
LightGBM model and one CSV. `repro.yml` is running for A2 and AW2, and the PC is fitting them.

**Still open before anything can be adopted.** Justin's name on rule 2; both names on a candidate; the three
OSes and the PC for the 9,600-step models; the equality of the pipeline's training with the gate's on a fold;
the Mac's own A and AW; the documents and the expected hashes.

### A on the Windows PC: result (the registered second-machine run)

**The PC's own run of A gives the runners' verdicts and the runners' numbers: A fails rule 1 and meets rule 2.**
The registered run of the entry "A on the Windows PC (pre-registration)", at `abe8079`, 02:43 to 07:29 CET on
Nov 13: LightGBM 84 minutes (151 s a fold), the TCN 198 minutes in two processes (810 s a fold), the gate 5
minutes. `registered_run: true`. External development matches only; nothing in A's registration was changed.
Logs: `D:/katena/logs/a-win`.

| A against F3, `target` | This PC (Windows, Intel i7-7700) | GitHub runners (Ubuntu, AMD EPYC 7763) |
|---|---|---|
| Rule 1: 28 folds, standing truth | 0.6093, -0.0087 [-0.0149, -0.0024], 10/28 | 0.6092, -0.0088 [-0.0150, -0.0025], 10/28 |
| Rule 2: 8 IDSSE + Metrica folds | 0.6763, +0.0389 [+0.0276, +0.0498], 8/8 | the same |
| Rule 2: the same at `severe` | 0.6263, +0.0363 [+0.0254, +0.0470], 8/8 | the same |
| Rule 2: 20 SkillCorner folds, action truth | 0.6645, +0.0551 [+0.0461, +0.0639], 20/20 | 0.6647, +0.0553 [+0.0461, +0.0641], 20/20 |
| All 28, action truth on SkillCorner | 0.6679 | 0.6680 |

- **The agreement asked for in the registration holds:** the same verdict under both rules, and every mean
  within 0.0002, against a limit of 0.002.
- **27 of the 28 folds score the same on the two machines to the last digit**, at both severities and on both
  truths. The TCNs differ in their last bits, as always, and no decoded possession changes with them.
- **The one fold that differs is `skillcorner/2010085`,** the fold whose LightGBM team head grew other trees on
  Linux (entry below): by 0.0018 at `target` on the standing truth and by 0.0036 on the action truth. That is
  what a LightGBM difference between two machines costs on one match, and it is the size to keep in mind for a
  reproduction of the final file on another machine.
- The Mac's run of A is still to come; with it A will have three machines.

**What it is not.** New evidence: the folds are the same. It adopts nothing. AW, AT and ATW have one machine
kind each (the runners); this entry is the reason to read their numbers as stable to the fourth decimal unless a
LightGBM fold flips.

### A and AW trained on all 30 matches on three OSes: one LightGBM model and one CSV each (result)

**The all-30 model of A, and of AW, gives the same dressed-match CSV on macOS, Ubuntu and Windows runners, and
its LightGBM member is byte-identical on all three.** `repro.yml` at `575c958`, runs 43366847192 (A) and
43366843380 (AW), 05:37 to 06:35 CET on Nov 13. Each runner built the corpus and the cache from empty, trained
the spec through `make_submission.py train --spec`, exported IDSSE `DFL-MAT-J03WMX` in the provided format and
labelled it. External matches only. Nothing is adopted by it.

| | A | AW |
|---|---|---|
| Runners | macOS 15 on Apple M1, Ubuntu 24.04 on AMD EPYC 9V45, Windows Server 2025 on AMD EPYC 7763 | macOS 15 on Apple M1, Ubuntu 24.04 on Intel Xeon 6973P-C, Windows Server 2025 on Intel Xeon Platinum 8573C |
| LightGBM member | `eb1030c8ec94...` on all three | `55de693c7ea2...` on all three |
| Its alive head | `59bfcd31dc1e...` | the same: the weights touch the team head only |
| TCN weights | three sets | three sets |
| Dressed CSV | `98b2919621653c99...` on all three | `280060af276f1665...` on all three |
| Rows of the alive head / the team head | 549,880 / 363,838 of 657,133 | the same |

- **This is the check the differing fold asked for** (entry below: one of A's 28 fold models grew other trees on
  Linux than on the PC). For the model that would be submitted, three OSes and five CPU models agree. The PC's
  own all-30 fit is still to be compared with these hashes; it is the reference machine.
- **It is also the first end-to-end run of the new training path** (`katena.eval.actlabels`, the specs in
  `katena.pipeline`): the action labels are built from the pinned raw files on each runner, and the label counts
  in the three manifests agree.
- The TCN's bytes differ by OS and CPU, as F's and F3's do, and the labels do not.
- The dressed match is one of the 30 training matches here, so its score says nothing and is not quoted.

**What it does not show.** The same file on the provided match, where a near-tie could flip a label; and a fold
has now shown that LightGBM itself can differ between two machines. If A or AW is adopted, section 7 of the
technical document has to say both.

### A2 and AW2 (pre-registration): the same two candidates with a TCN trained twice as long

**Written at 05:40 CET on Nov 13, before any score of either.** Claude on David's PC; neither David nor Justin
has seen it. The 18th and 19th packages on the same 28 folds. Code: `scripts/run_x2.py`. External development
matches only. It adopts nothing.

**Why.** The one change since F that passed on both kinds of fold was more fitting: 1,200 trees for 150, and a
TCN at 4,800 steps for 1,200 (`LT`, +0.0120, 21 of 28). The step count was never screened beyond 4,800. The
candidates of the entry above now train on labels with other mechanics on 20 matches, and the TCN is the member
that carries them. Longer training changes no feature and no line of the inference path, so it adds no risk on
the provided match; it doubles the TCN's share of `train` (about 11 more minutes on the PC).

**The change, and nothing else.** A2 is A, and AW2 is AW, with 9,600 TCN steps in place of 4,800. The one-cycle
schedule spans the steps, as it does now. Sequences, labels, weights, seed and threads are the base candidate's,
and so is the LightGBM member, read from its cache. No other step count is tried, whatever comes out.

**Rule, for each of the two against its own base.** F's rule on the 28 folds with the action truth on SkillCorner
(the truth rule 2 reads): at `target` at least 70% of folds won and the paired 95% interval above 0; not worse at
`severe`. And it must not be worse at `target` on the 8 IDSSE + Metrica folds, nor on the 20 SkillCorner folds.
A2 can only replace A, and AW2 only AW; which of A and AW stands is settled by the entry above and by the two
who decide, not here. A pass makes the longer TCN part of that candidate if it is adopted; a fail leaves it at
4,800 steps.

**Why this rule and not rule 2's.** The change is of the kind rule 1 was written for: it should help every kind
of fold, so it has to win folds. The standing SkillCorner truth is not used because the candidates no longer
train on it.

**Information, decides nothing:** the TCN member alone; each against F3; predicted possessions per true one.

**Run.** `gate.yml` on GitHub's runners, tag `gha2x`: `run_a.py a-tcn --steps 9600` and `run_aw.py aw-tcn --steps
9600`, 4 jobs each. The gate runs on the PC, as the entry above.

### A, AW, AT and ATW on GitHub's runners: result (Claude on David's PC; neither David nor Justin has read it)

**All four meet rule 2 and fail rule 1. By the registered tests AW stands. Nothing is adopted here.** Read on
Nov 13 between 05:10 and 05:35 CET. External development matches only. Nothing in any registration was changed.

**What ran.** `gate.yml` at `9f5b9bd`, four runs of four jobs, Ubuntu 24.04, 7 folds a job (runs 43350818678,
43350823891, 43350828569, 43350833484). The runners were AMD EPYC 7763, 9V45 and 9V74 and Intel Xeon 6973P-C and
Platinum 8573C; a TCN fold took about 5 minutes and a job 55 to 110 minutes with its cache. The gates ran on the
PC against F3's cached members there: `run_a.py a-gate`, `run_aw.py aw-gate` and `run_at.py at-gate`, each with
`--tag gha --inc f3_gbdt,f3_tcn`. Logs: `D:/katena/logs/a-win/gha_*_gate.log`.

**Checks, made before a score was read.**

- **A's LightGBM member from the runners equals the PC's on 27 of 28 folds**, hash and every probability. On
  `skillcorner/2010085` the alive head is identical and the team head is not: the two machines grew different
  trees, and its probabilities differ by 0.008 on average and by up to 0.15. So LightGBM is not byte-identical
  between Windows and Linux for every training set, which F and F3 had not shown on any machine we tried. It bears on
  reproduction, not on these gates: the all-30 model of whatever is adopted has to be compared across machines
  as F3's was (`repro.yml` is running for A and for AW).
- AW's LightGBM alive probabilities equal A's on 28 of 28 folds; every variant has its 28 folds.
- A's LightGBM member with F3's TCN gives, on the 8 IDSSE + Metrica folds, +0.0232 [+0.0144, +0.0323], 8 of 8:
  the Mac's interim number to the digit.

**Against F3** (spellF1; `target` unless stated; SkillCorner on the action truth in rule 2).

| | F3 | A | AW | AT | ATW |
|---|---|---|---|---|---|
| Rule 1: 28 folds, standing truth | 0.6180 | -0.0088, 10/28 | -0.0224, 8/28 | -0.0084, 10/28 | -0.0191, 8/28 |
| Rule 2: 8 IDSSE + Metrica folds | 0.6374 | **0.6763, +0.0389 [+0.0276, +0.0498], 8/8** | **0.6923, +0.0548 [+0.0399, +0.0697], 8/8** | **0.6725, +0.0351 [+0.0253, +0.0448], 8/8** | **0.6964, +0.0590 [+0.0448, +0.0733], 8/8** |
| Rule 2: the same at `severe` | 0.5900 | +0.0363, 8/8 | +0.0613, 6/8 | +0.0287, 7/8 | +0.0625, 6/8 |
| Rule 2: 20 SkillCorner folds, action truth | 0.6094 | 0.6647, +0.0553 [+0.0461, +0.0641], 20/20 | 0.6586, +0.0491 [+0.0387, +0.0597], 19/20 | 0.6669, +0.0575 [+0.0490, +0.0660], 20/20 | 0.6577, +0.0483 [+0.0386, +0.0585], 19/20 |
| All 28, action truth on SkillCorner | 0.6174 | 0.6680 | 0.6682 | 0.6685 | 0.6688 |
| Meets rule 2 | | yes | yes | yes | yes |

Rule 1 fails for all four, as A's entry expected: on the standing SkillCorner truth they lose 0.026 to 0.053.

**The tests that say which one stands** (registered in AW's entry and in AT's; `target`, the 8 folds of rule 2).

| Step | Test | Diff [95% CI] | Wins | Result |
|---|---|---|---|---|
| 1. Weights, alive/dead as A | AW minus A | +0.0160 [+0.0041, +0.0282] | 5/8 | Interval above 0: AW |
| 1. Weights, alive/dead as F3 | ATW minus AT | +0.0239 [+0.0115, +0.0362] | 5/8 | Interval above 0: ATW |
| 2. Alive/dead training | ATW minus AW | +0.0041 [-0.0017, +0.0099] | 6/8 | Not above 0: AW stays |

**So the registered procedure names AW**, if rule 2 is in force. AT and ATW change nothing: leaving the
SkillCorner rows out of the alive heads neither helps nor hurts (AT minus A: -0.0038 [-0.0100, +0.0023] on the 8
folds, +0.0022 on SkillCorner against the action truth; alive/dead F1 on the 8 folds is 0.962 for F3 and for
all four). The worry behind AT was not borne out.

**By the process that made the truth** (`target`; predicted possessions per true one in brackets). This is not
part of any registered test.

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

- **The step from F3 to any of the four is large on every truth with these mechanics:** +0.035 to +0.059 on the
  8 folds (every fold, for every candidate), +0.048 to +0.058 on SkillCorner against the action truth, +0.009 to
  +0.027 on the two Metrica folds. K read at kind 0 (+0.0265 there, the Mac's entry below) and the transfer
  probe point the same way with models that never saw an action label.
- **Among the four there is nothing to choose on all 28 folds:** 0.6680 to 0.6688.
- **The weights move score from two kinds of truth to the third.** They gain on IDSSE (+0.027 and +0.034) and
  lose on Metrica (-0.018 and -0.006) and on SkillCorner against the action truth (-0.006 and -0.009; at `severe`
  AW is worse than A there, -0.0106 [-0.0185, -0.0028]). IDSSE's labels hold about twice the share of
  possessions under 2 s. With the weights the models predict 0.90 possessions per true one on IDSSE and 1.17 to
  1.24 on the other two.
- **So the test that names AW rests on the six IDSSE folds.** AW minus A by fold: +0.060, +0.035, +0.033,
  +0.027, +0.010 and -0.001 on IDSSE; -0.009 and -0.026 on Metrica.

**For the two who decide (my reading; it is not a registered test).**

- Leaving F3 for one of these is the decision that matters, and it rests on the documentation: the provided
  match's ground truth has dead ball, an owner on every live frame, and a change at the first controlled touch.
- AW or A is close to even. AW is what the registration names. A is never behind AW on the two kinds of truth
  that are not IDSSE, one of which is Metrica's, and Metrica made the provided match's ground truth. Two Metrica
  folds are thin evidence, and the action truth is our own construction.
- `DATA.md` prints the public window's possession count and mean length. They are not used here: `RULES.md`
  says the window is not a tuning set.

**Not shown.** That the action labels are right; that the provided match follows them; anything about a second
machine for AW, AT and ATW (the PC fits AW after its own A; the Mac's A and AW are still to come). The cut-off
is Nov 14, 18:00 CET, and adoption needs Justin's name on rule 2 and both names on the candidate.
### K at kind 0, S1 and S2 against the action truth: result (Claude; Justin has not read it yet)

**Read at 04:05 CET on Nov 13**, after the entry below was pushed (`c463d70`). `scripts/read_act.py`, 28 s,
`work/final/read_act.json`. 20 SkillCorner folds, action truth, 20 of 20 with every member; A's TCN on none.

| Comparison | `target` | `severe` |
|---|---|---|
| **K at kind 0 vs F3** | 0.6359 vs 0.6094, **+0.0265 [+0.0178, +0.0349], 18/20** | +0.0243 [+0.0165, +0.0320], 17/20 |
| K at its own kind vs F3 | -0.0141 [-0.0190, -0.0091], 2/20 | -0.0102 [-0.0153, -0.0053], 4/20 |
| K at kind 0 vs K at its own kind | +0.0405 [+0.0313, +0.0497], 18/20 | +0.0346 [+0.0259, +0.0428], 19/20 |
| S1 vs F3 | +0.0083 [+0.0022, +0.0150], 14/20 | +0.0094 [+0.0040, +0.0149], 15/20 |
| S2 vs F3 | -0.0057 [-0.0123, +0.0011], 5/20 | -0.0061 [-0.0125, +0.0004], 9/20 |
| A1 (interim) vs K at kind 0 | -0.0086 [-0.0155, -0.0012], 7/20 | -0.0077 [-0.0145, -0.0009], 6/20 |

True possessions found at IoU 0.5 by true length, `target`:

| Model | under 2 s | 2-4 s | 4-8 s | 8-16 s | over 16 s |
|---|---|---|---|---|---|
| F3 | 0.05 | 0.22 | 0.55 | 0.75 | 0.85 |
| K at kind 0 | 0.12 | 0.39 | 0.69 | 0.80 | 0.88 |
| K at its own kind | 0.05 | 0.18 | 0.51 | 0.73 | 0.84 |
| S1 | 0.07 | 0.31 | 0.63 | 0.80 | 0.88 |
| S2 | 0.08 | 0.30 | 0.64 | 0.81 | 0.87 |
| A1 | 0.10 | 0.31 | 0.61 | 0.81 | 0.89 |

**Read as the entry below said.**

- **K at kind 0 is ahead of F3 with the interval above 0, so the mechanics of IDSSE and Metrica carry over to
  real broadcast tracking** when the truth changes owner at the first touch: 18 of 20 folds, and 39% of the 2-4 s
  possessions found against 22%. Neither model trained on that truth. The same reading lost on every fold
  against the standing truth (-0.0326, 0 of 20; K's entry). On SkillCorner the truth decides the verdict, as the
  transfer probe said with one head.
- **Kind 0 is the value K's registration fixed for the provided match.** So K, read the way it would be used,
  is ahead of F3 on the 8 IDSSE + Metrica folds (+0.0345, 8 of 8) and on these 20 against the action truth.
  **That does not reopen K.** It failed the rule it was registered under, rule 2 was declared for A, and this
  reading was chosen after K's scores were known. It says what K's column is worth to a model that still
  trains on the standing SkillCorner labels, and it is the bar A has to clear: A1 is 0.0086 behind it here with
  the interval below 0, with half of A (K1, the comparable half of K, was +0.0175 on the 8 folds where A1 is
  +0.0232).
- **The length weights split by member.** S1 is ahead (14 of 20); S2 is not (5 of 20, interval touching 0). Both
  were trained with weights taken from the standing SkillCorner labels, whose short possessions are the ones
  the action labels disagree with, so this is not AW's or ATW's TCN; it is a reason to read their TCN members
  alone when their gates print.

**What it does not show.** That the action truth is right or that the provided match follows it (A's entry).
And the action truth was built by this team from SkillCorner's events with two constants; a model can gain on
it by sharing those constants only if it trained on them, which K, S1, S2 and F3 did not.

### K at kind 0, S1 and S2 against the action truth (information; written before the read)

**Written at 03:55 CET on Nov 13, before `scripts/read_act.py` was run without `--dry`.** Claude on Justin's Mac;
Justin said to go ahead with it (his message of 03:25 CET) and has not read this entry. It adopts nothing, no
candidate comes out of it, and it changes no registration. Asked for by David's side (PR #31, comment of 02:44
CET, point 3). External matches only; no model is fitted; a few minutes.

**The question.** The same as the transfer probe's below, with whole models: on real broadcast tracking, scored
against a truth whose owner changes at the first touch, are the models that learned IDSSE's and Metrica's
mechanics ahead of F3? None of K, S1, S2 or F3 trained on an action label, and the action labels were built
after all four were trained.

**What is read.** The 20 SkillCorner folds, `target` and `severe`, cached out-of-fold probabilities, decoded as
F3's are, scored against the action labels of `run_a.py` (`gt_act`) the way an IDSSE fold is scored.

| Name | Members |
|---|---|
| F3 | `g1_gbdt1200` + `g2_tcn4800t` |
| K0 | K's two members with the label-kind column read at 0, the value of the IDSSE and Metrica rows (`run_k.py` stored this reading on SkillCorner folds as its "Kx") |
| K | K's two members at the fold's own value, 1 |
| S1, S2 | `s1_gbdt1200w` + F3's TCN; F3's LightGBM + `s2_tcn4800tw` |
| A1 | A's LightGBM member + F3's TCN (trained on the action labels; see "already seen") |
| A | A's two members, only once A's TCN is cached on the Mac for all 20 folds |

Printed: K0 vs F3, K vs F3, K0 vs K, S1 vs F3, S2 vs F3, A1 vs K0, and A vs K0 when it exists; means, paired 95%
interval, folds won; possessions found by true length. The dry run found every file and key on 20 of 20 folds
and A's TCN on none.

**How it will be read.** K0 ahead of F3 with the interval above 0: the gain on the 8 folds carries over to real
automatic tracking, on 20 folds no package was chosen on by this truth. K0 behind F3: the gain on the 8 folds
may belong to the simulated noise, and that is to be said wherever A, AW, AT or ATW is discussed. S1 and S2 say
the same for the length weights that AW and ATW carry. Whatever comes out, the two rules stand as written.

**Already seen on this side, so this is not a blind read of everything in it.**

- **A's interim gate on the Mac (A1: A's LightGBM member with F3's TCN; not the registered run).** Claude read
  it at 23:57 CET on Nov 12 and told Justin. Rule 1 fails as A's entry expected: `target`, 28 folds, -0.0023
  [-0.0073, +0.0025], 12 of 28; the 20 SkillCorner folds on the standing truth -0.0125 [-0.0183, -0.0066], 4 of
  20. On the 8 IDSSE + Metrica folds +0.0232 [+0.0144, +0.0323], 8 of 8 (`severe` +0.0191, 7 of 8); on the 20
  SkillCorner folds against the action truth +0.0179 [+0.0111, +0.0245], 16 of 20. A1 vs K on the 8 folds:
  -0.0113. A's TCN has not started on the Mac (it waits for the charger), so nothing of A itself is known here.
  **Justin has therefore seen an interim score of A before giving or refusing his name to rule 2.**
- **K's gate** (entry of Nov 12), on the standing truth, including K read with the other kind (-0.0361 on all
  28).
- **The transfer probe's result** (entry below): +0.0115 on the action truth, 15 of 20.
- `run_a.py` computed K at its own kind against the action truth into `a_scores.pkl`; it was never printed.

### Three research reads on the evening of Nov 12 (description only; what was looked at)

**Written at 03:55 CET on Nov 13.** Justin asked for two or three research agents on how to improve further
(01:45 CET). Three ran on the Mac, each a Claude subagent: read-only on the repo, external matches only,
aggregates only, no model fitted, no spellF1 of any new variant computed. None read A's or AW's cached
probabilities. They are listed here because any later registration has to count them as seen.

**1. Decoder and metric** (F3's cached members, 28 folds, `target` and `severe`).

- Looked at: found share and credit by length; why each missed possession is missed; the team probability over
  true possessions; runs the Viterbi discards and dips with p(other team) of 0.4 or more against the truth; the
  credit of F3's predicted possessions by length and neighbour; transition lags; reliability tables of both
  heads; costs of single errors on synthetic sequences with the official scorer.
- Found: of the missed 2-4 s possessions, 4.4% had the right team in the probabilities and lost it in the
  decoder, 2.2% were called dead, and in 64% the team probability was wrong on most frames (mean p of the true
  team 0.27, against 0.77 on found possessions). In most of the rest the prediction ran on because the
  neighbouring opponent possession was missed. An oracle that restored every decoder-smoothed miss would add
  0.009 to 0.017; a rule without the oracle about 0. **The decoder has no sandwich merge:** the T = 2 merge
  defines the truth and the training labels only, and applying it to predictions would change 3 of 5,922
  possessions. Claude's earlier statement to Justin that the merge erases short predicted flips was wrong.
- Not recommended: any decoder candidate. The precision of the p of 0.4 or more dips (43 to 59%) has been seen.

**2. Data and labels** (web, licences, and the public external files).

- No further open dataset is usable before the cut-off: PFF's World Cup 2022 set publishes no licence text; the
  rest is in use already, gated, or carries no possession label.
- A's SkillCorner labels: the fixed 1.0 s restart differs from the start of SkillCorner's own following phase
  (`*_phases_of_play.csv`) by more than 0.5 s on 51% of received restarts; and on 11 stoppages a match the ball
  goes dead only at the broadcast cutting away, a median 8.1 s after the last touch. David's side found the
  second one independently (AT's entry).
- IDSSE's label changes owner a median 0.16 s after the new team's player is within 1.5 m (2 matches, 709
  changes); Metrica's at the new team's first pass, carry or recovery, with no lag.
- One probe printed event-type counts of `Sample_Game_2`, a locked bench (counts only, no score, no label).

**3. Model** (F3's and the closed S1/S2 members' cached probabilities, `target`; SkillCorner folds read against
the action truth).

- Team accuracy by position in the true possession, F3: 0.988 more than 4 s from both ends; 0.49 in the first
  second after a direct change of owner and 0.68 in the next (IDSSE + Metrica); 0.97 in the first second after
  dead ball.
- Timing of the 0.5 crossing at true direct changes, within 0.4 s: F3 0.21 (IDSSE), 0.43 (Metrica), 0.27
  (SkillCorner, action truth); with the length weights (S1 and S2 together) 0.38, 0.63, 0.40. Split by whether
  the ball is in view around the change, with the weights: 0.48 in view against 0.30 not (IDSSE), 0.72 against
  0.31 (Metrica).
- **This is a look at the length weights on the action truth before AW's or ATW's score is known**, in timing
  terms, not spellF1.
- Read in the code: the feature set has team velocity only over centred windows of 2 s and 5 s either side and
  no one-sided window, no acceleration and no rate of the lines or the spread (`featurize/build.py`).

### AT and ATW (pre-registration): A's team labels with F3's alive/dead training; and the candidates on GitHub's runners

**Written at 03:10 CET on Nov 13, before any score of A, AW, AT or ATW is known on this side.** The PC's A run is
in its LightGBM stage and its gate has not run; the Mac's branch still ends at `e5550c4`, with no result. Claude
on David's PC; Justin has not seen it. The 16th and 17th packages on the same 28 folds. Code:
`scripts/run_at.py`. External development matches only.

**A's entry says no other variant is tried.** AT is one, and this entry does not hide that. It comes from David's
side and from readings that saw no score of A: a code review of A's labels (a fresh reader; labels and events
only, no model output) and an error analysis of F3's cached folds. Justin can refuse it; then AT and ATW are
information and nothing more.

**Why.**

- **The review found no bug.** A second implementation written from the registration's text gives A's labels on
  all 20 matches, frame for frame, and the table and row counts of A's entry come out exactly.
- **It found that A's dead-ball boundaries are weaker than its owner changes.** Of 1,851 flagged stoppages, 228
  take their start from the broadcast cutting away, a median of 6.5 to 9.1 s after the last touch, depending on
  the kind of stoppage; 138 of them are free kicks, where the ball never leaves the pitch. Placing those starts
  1.0 s after the last touch instead would turn 18,267 frames (2.8% of the team-labelled ones) from a team into
  dead ball: the ball moves 3.2 m a second on them (2.6 on A's dead frames, 6.3 on its live ones), and the
  standing label is unknown on 81.5% of them. Restarts sit 1.0 s before the reception, by a constant.
- **F3's alive/dead is already good where it can be measured.** On the 8 IDSSE + Metrica folds F3 finds 93% of
  the stoppages and times restarts to a median of 0.00 s (quartiles -0.12 and +0.08). The true alive/dead
  pattern would add 0.024 spellF1 there.
- **What F3 lacks is short possessions after an opponent's.** They are 26% of the true possessions on those
  folds and F3 finds 11% of them; putting in the missed ones under 4 s would add 0.19. Boundary timing, the
  decoder and the blend weight are each worth under 0.02.
- **So A can gain through its owner labels and lose through its dead-ball labels.** In A the SkillCorner rows
  become 66% of the alive head's rows (336,097 of 507,549), with boundaries that a constant or a cutaway sets.

**The change, and nothing else.**

| | Team heads | Alive/dead |
|---|---|---|
| **AT** | A's. The LightGBM team head is A's cached one. The TCN's team loss reads the action labels on SkillCorner rows and the standing labels elsewhere | F3's. The LightGBM alive head is F3's cached one. The TCN's alive loss reads IDSSE and Metrica sequences only |
| **ATW** | AW's. The LightGBM team head is AW's cached one. The TCN's team loss carries AW's weights (`lenweight.inv_length` of the label each frame trains on) | As AT |

No LightGBM model is fitted for either: only the two TCNs, at F3's sizes and seed. Labels, constants, T = 2,
the folds and the decoder's priors are A's. The label rule itself is not changed: an amended rule would be one
more variant, and with F3's alive/dead training the frames it would move train nothing but a team head.

**Rules.** The two rules of A's entry, word for word, with AT or ATW in A's place, against F3. Rule 2 counts only
with David's and Justin's names, as there (David's side: the entry below).

**Which candidate stands, under whichever rule is in force.** Only candidates that pass that rule against F3
are eligible. Two paired tests choose among them, each at `target` on that rule's folds (28 on the standing
truth for rule 1, the 8 IDSSE + Metrica folds for rule 2), and each needs a 95% interval above 0 to leave the
simpler choice:

1. **Weights**, inside each alive/dead design: AW over A as AW's entry says, and ATW over AT the same way. If
   only one of a pair is eligible, it stands for its design.
2. **Alive/dead training:** the AT-side candidate replaces the A-side one only if it is ahead of it with the
   interval above 0. If only one side has an eligible candidate, that one stands.

If none is eligible, F3 stands. Adoption is David's and Justin's either way, before Nov 14, 18:00 CET.

**Machines.** The Mac's and the PC's runs of A go on as registered. In addition A, AW, AT and ATW are fitted on
GitHub's Ubuntu runners with the new `gate.yml` (PR #32): the last `repro.yml` run trained F3 on all 30 matches
in under 6 minutes there, against about 20 on the PC. Each candidate takes 4 jobs of 7 folds; files made on a
runner carry the tag `gha`. No score is computed on a runner: the gate stages run on the PC against F3's cached
members there.

- A LightGBM member from a runner must carry the hash of the same fold fitted on the PC. A's are being fitted
  on the PC now, so that is checked before a score of a runner-made candidate is read.
- A TCN from a runner differs from the PC's in its last bits. When the PC's own A is done, A is scored with each
  machine's members and the two are reported side by side. That difference is the noise floor for reading the
  other three, which have no second machine.

**Information, decides nothing:** each member alone; A's LightGBM team head alone in F3; AT's TCN alone in F3;
alive/dead F1, team accuracy and predicted possessions per true one; true possessions found by length; ATW
against A and AT against AW.

**What it cannot show.** The same as A's entry: that the action labels are right, or that the provided match
follows them. And four related candidates on eight seen folds is a choice among four, which the two tests above
are there to keep honest: each step away from A needs its own interval above 0.

### Where F3 loses, and an audit of the inference path (description; nothing changed in the model)

Two fresh readers, at Claude's request on David's PC, on external matches and synthetic fixtures only. Neither
opened anything of the provided match. Their scripts are outside the repository (the session's scratch folder),
so this entry is the record.

**1. Where F3 loses spellF1** (the 28 cached folds of the Windows gate, decoded as the gate decodes them; the
loader reproduces 0.6180, 0.6374 and 0.6103). Gain in spellF1 from an oracle edit of F3's output:

| Edit | IDSSE + Metrica (8) | SkillCorner (20) |
|---|---|---|
| Missed true possessions under 4 s put in | +0.194 | +0.181 |
| Only those that follow an opponent's possession | +0.183 | +0.124 |
| Only those where the blended probability leans to the true team | +0.025 | +0.010 |
| Predicted possessions that match nothing taken out | +0.033 | +0.049 |
| True alive/dead with the predicted team | +0.024 | 0 |
| Every boundary within 1.0 s of a true one moved onto it | +0.017 | +0.007 |
| The best fixed tau and switch cost | +0.008 | +0.002 |
| Every label 0.2 s earlier | +0.003 | 0.000 |

- **The points are in short possessions after an opponent's, and the probabilities do not show them.** They are
  26% of the true possessions on IDSSE + Metrica, and 11% are found (46% of the short ones that follow dead
  ball). In the missed ones the blend leans to the true team 41% of the time; where the decoder gave the spell to
  the other team, 9%. No decoder setting recovers what the members do not see.
- **Timing is not where the points are.** Team changes are a median of 0.2 s late on IDSSE and Metrica and on
  time on SkillCorner; restarts are on time (quartiles -0.12 and +0.08 s). Moving team changes 0.4 s earlier
  gives +0.004 on IDSSE + Metrica and -0.003 on SkillCorner.
- **Dead ball is mostly found:** 93% of stoppages (IDSSE 95%, Metrica 85%), and 96% of the predicted ones are
  real. Metrica's stoppages start a median of 0.9 s late, as section 7 of the technical document says of its labels.
- **The labels have a step at 2 s and the model has a ramp.** An opponent's spell between two spells of the
  same team is decoded as one 26% of the time at 1.5 to 2 s, 33% at 2 to 2.5 s, 55% at 3 to 4 s and 78% at 4 to
  6 s.
- **The TCN is the better member almost everywhere** (right in 66% of the samples where the two disagree on
  IDSSE + Metrica), but both are wrong on 36% of the samples inside possessions under 4 s. A TCN weight of 0.75
  is +0.0041 on 28 folds with 17 won: it fails the rule.
- **The same output scored against labels that also merge a short touch running straight out of play**
  (`tail`, section 6 of the technical document) is 0.6728 on IDSSE + Metrica, against 0.6374. That is the size
  of what the unknown convention can do to the level, in either direction, with no model change.

**2. Audit of the inference path** (the real loader, `label_match`, features, both members, decoder, expansion to
frames; experiments on the dressed external match and on synthetic hand-overs).

| Finding | Evidence | What was done |
|---|---|---|
| Training and inference are time-aligned | Dressed match: all transitions -0.65 +- 0.72 frames, median 0; synthetic exact hand-overs: median +3 frames; the score peaks at zero shift; the 5 Hz phase of a half moves boundaries by under 0.2 frames | Nothing to do |
| The decoder subtracts class priors that differ for H and A (0.295 and 0.258, from the external labels), although the team probability is symmetric by construction | It asks for p(home) above 0.534. With the two averaged: +0.0000 at `target` (12/28), +0.0027 at `severe`, +0.0036 in `target_smooth` (18/28) | Left: no gain at `target`, and it would change F3's decoder |
| The TCN is applied to a whole half in one pass, but trained on 256-sample windows, so its GroupNorm sees other statistics | Held-out model on the dressed match, five conditions: applying it in 256-sample windows lowers its log-loss by 1 to 9% (alive) and about 1% (team) every time; spellF1 moves by -0.0002 to +0.0056 | Left: one match, and a change to the inference path three days before the deadline |
| `spd_nearest_f` and `spd_nearest_o` read the speed of each team's first slot when no ball is available | 0.3% of split gain; forcing them empty changes 40 frames of 70,507 and no score | Left for the next retrain of features, which is not planned |
| A ball tracked outside the lines carries the alive head (three such features in its top eight). If the provided tracking never reports the ball off the pitch, the dressed match loses 0.03 at clean and at target-like input | Dressed match with the ball removed when outside: 0.722 to 0.688, 0.679 to 0.648 | Nothing chosen. The coupled worlds of `d2j` already train with the ball mostly missing in stoppages |
| `check_target.py` passed when a direction was found and flipped at half-time, which a direction wrong in both halves also does | The two teams' mean positions are 10.9 m or more apart on all 60 external halves | PR #33: the check also asks for 8 m in both halves, pass or fail only |

Checked and fine: tracked frames outside the halves and 14 extra empty slots change no label; no feature uses a
frame index, the clock or the pitch size; a half that opens 117 s before its kickoff is labelled dead to within 4
frames of the kickoff by the model alone; the 213 column names, the scaler and the clipping before the logit.

### When each provider's owner changes (web research, public pages only; nothing downloaded besides three PDFs read)

**Nothing was fitted, scored or chosen.** A research agent, at Claude's request on David's PC, read providers'
public documentation for one question: at a change of possession, when does the per-frame owner switch? It
bears on which training labels have the mechanics of the provided match's ground truth, which is the bet
behind A, AW and rule 2.

| Source | What its documentation says | How sure |
|---|---|---|
| **Metrica Sports** (organiser; the ground truth comes from its corrected tracking, `OVERVIEW.md`, `DATA.md`) | Event definitions of its open sample data: a player is in possession "as soon as they touch the ball"; possession is not lost when an opponent's touch has no significant impact; the ball is out of play when it leaves the pitch or the referee stops play, and a set-piece event brings it back. Its open tracking carries no per-frame owner, so how the corrected tracking fills that field is not public | High for the definitions, low for the field |
| **SkillCorner dynamic events** (`player_possession`, the events A labels from) | A player possession "starts from the first touch of the player up to the last touch" and needs control; a block does not count; team possession "shifts to Team B at the start of their player possession" | High |
| **SkillCorner tracking** (`possession.group`, the standing label) | One line of description; how it is produced is not documented. Nothing public explains the switch about 1.0 s before the first touch that the probe of Nov 12 measured | Low |
| **DFL / TRACAB** (`BallPossession`, `BallStatus`; IDSSE) | "Defined by the player in control of the ball"; the ball is inactive "when the match is interrupted by the referee"; status is logged by human observers during the match. No minimum duration or timing study found | Medium |

Sources: Metrica's `events-definitions.pdf` in `metrica-sports/sample-data`; SkillCorner's Dynamic Events CSV
specification (2025-02-16); Bassek et al. 2025 and Lang et al. 2022 for DFL; kloppy's readers for the fields.

**What it says for us, and it is an inference.**

- The organiser's own definitions, SkillCorner's event specification and `CHALLENGE.md` agree: the owner changes
  at the first touch of the player who takes control, and stays with the last team while the ball is loose.
  SkillCorner's frame label is the one that does not.
- Our Metrica labels are derived from the events those definitions describe, so the two Metrica development
  folds and the Metrica bench are the matches whose labelling process is nearest the provided match's.
- None of this says how the corrected tracking times the change to the frame. It is still a bet.

**Two public competitor repositories, self-reported and not comparable** (found by the same search): one trains
gradient-boosted frame classifiers with a Viterbi decoder on degraded Metrica sample games only, weighting every
possession equally, and reports 0.77 leave-one-game-out on those games; one tunes rule-based state machines on
degraded SkillCorner matches and reports 0.35 on the public window. Nothing was taken from either.

**Method ideas it listed, none tried here:** possession-length weights (that is AW), a boundary-offset head for
timing below the 5 Hz grid, distilling a model trained on clean tracking into the one trained on degraded
tracking, and averaging weights or seeds. Mirroring the pitch across its long axis gives nothing: every lateral
feature is already an absolute value.

### A team head that never saw a SkillCorner label, on the action truth (information; written before the read)

**Written at 02:50 CET on Nov 13, before the script was run.** Claude on David's PC. It adopts nothing, no
candidate comes out of it, and it changes nothing in A's or AW's registrations. `scripts/probe_transfer.py`,
external matches only, about five minutes.

**The question.** Rule 2 rests on the 8 IDSSE + Metrica folds, which every package has seen and whose noise is
simulated. Does a team head with their label mechanics also do better on real broadcast tracking, when the
truth has those mechanics?

**What is read.** One LightGBM team head fitted on the IDSSE and Metrica rows of the development pool only, at
F3's size and parameters (the `7 + 0` member of the probe of Nov 10). It has never seen a SkillCorner label of
either kind, so the one head is out of fold for all 20 SkillCorner matches. It takes the place of the team head
in F3's LightGBM member; F3's alive head and TCN stay. The 20 SkillCorner folds are scored against the action
truth, which neither that head nor F3 trained on, and against the standing truth. Check: the alive head fitted
beside it equals F3's cached one on every fold.

**How it will be read, fixed here.**

- Ahead of F3 on the action truth (interval above 0 at `target`): the mechanics carry over to real broadcast
  tracking even without one row of it in training. That supports rule 2 with 20 folds whose truth nobody has
  fitted to.
- Behind F3 there: not evidence against the mechanics, because this head also lost its only rows of real
  broadcast tracking. The test that separates the two is K read at kind 0 on the same folds (asked of the Mac
  in PR #31).
- On the standing truth it should lose, as the same member lost on the IDSSE + Metrica side when it was read
  with the wrong kind.

**Result** (run once at 02:53 CET, 7 minutes; `D:/katena/logs/a-win/probe_transfer.log`). N is F3 with that team
head in its LightGBM member. Check: the alive head equals F3's cached one on 40 of 40 fold-conditions.

| N against F3, 20 SkillCorner folds | N | F3 | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|
| Action truth, `target`, blended | 0.6209 | 0.6094 | +0.0115 | [+0.0037, +0.0193] | 15/20 |
| Action truth, `severe`, blended | 0.5759 | 0.5615 | +0.0144 | [+0.0079, +0.0212] | 16/20 |
| Action truth, `target`, LightGBM member alone | 0.5828 | 0.5878 | -0.0050 | [-0.0136, +0.0036] | 8/20 |
| Standing truth, `target`, blended | 0.5775 | 0.6103 | -0.0328 | [-0.0395, -0.0261] | 0/20 |
| Standing truth, `target`, LightGBM member alone | 0.5352 | 0.5985 | -0.0633 | [-0.0714, -0.0556] | 0/20 |

- **Read as fixed above: ahead of F3 on the action truth, at both severities.** A team head trained on 8 matches
  of simulated noise, with no row of real broadcast tracking, beats one trained on those 8 and on 19 real
  SkillCorner matches with the standing label, once the truth changes owner at the first touch and has dead ball.
- **It is the blend that gains.** Alone, the member is level with F3's member (it has lost 72% of its team
  rows). Blended with F3's TCN, which still trains on the standing SkillCorner label, it is ahead.
- **On the standing truth the same model loses 0.033, on every fold.** So on SkillCorner the choice of truth
  decides the verdict, and rule 1 scores 20 of its 28 folds on the truth the documentation does not describe.
- Predicted possessions per true one at `target`: 0.889 for F3 and 1.069 for N on the action truth (0.760 and
  0.904 on the standing one). F3 no longer under-segments by a quarter when the truth is the action one, which
  has 239 possessions a match against 272.

**What it does not show.** That the action truth is right, or that the provided match follows it. Nothing here
was trained on the action labels, so it says nothing about A itself.

### A on the Windows PC (pre-registration): the second-machine run, and David's side on rule 2

**Written at 02:40 CET on Nov 13, before any score of A or of AW is known on this side.** Claude on David's PC,
at David's request ("pull and check what justin has done ... try out stuff ... improve the score"). The branch
`claude/status-check-7809aa` at `e5550c4` holds A's and AW's registrations and code and no result; whether the
Mac's run has finished is not known here. External development matches only.

**What runs.** `scripts/run_a.py a-labels a-check a-gbdt a-tcn a-gate --inc f3_gbdt,f3_tcn`, at this commit, in
`D:/katena/work-f3` (the cache of F3's Windows gate of Nov 11; A's files are added beside it and nothing of F3's
is rewritten). Nothing of A's registration (Nov 12, below) changes: the labels, the two 1.0 s constants, T = 2,
the sizes, the seeds, the 28 folds, the two truths and the two rules. The incumbent is F3 as cached on this PC
(0.6180 at `target`, 0.5840 at `severe`).

**What it is for.**

- F3 was adopted only after its gate had been repeated on this PC (Nov 11). A gets the same before anyone
  decides on it.
- The PC is the reference machine. If A is adopted, its all-30 model and the expected hashes are made here.
- It puts A's out-of-fold probabilities on this PC, so anything built on A can be gated here.

**What it is not.** New evidence: the folds are the Mac's. It adopts nothing.

**How the two machines are read together.** As for F3: the run agrees with the Mac's if rule 1 and rule 2 give
the same verdicts and the means at `target` on the 28, the 20 and the 8 folds differ by at most 0.002. If they
disagree, A is not adopted until the reason is known.

**Seen before this entry** (no model output, no score): `a-labels` on this PC gives the Mac's table to the
digits printed (home 0.280, away 0.273, dead 0.206, unknown 0.242; 239 possessions a match; agreement with the
standing label 0.847), so the labels are the same on both machines at that resolution.

**AW on this PC** follows A under its own registration, unchanged, if the PC is free before the cut-off. No
other variant of A is run here.

**David's side on rule 2 (Claude, under David's hand-over of Nov 12; David has not read this and can withdraw
it).** A's entry says rule 2 counts only if David and Justin both put their names to it. David's side agrees to
rule 2 as declared there, for A and for AW, now, while no score of either is known here. The reasons are public
documents, not scores:

- `CHALLENGE.md`: the ground truth has a dead-ball label, names an owner on every live frame, keeps the last
  team on a loose ball, and changes owner when the other team gains control. That is the IDSSE and Metrica kind
  of label. The standing SkillCorner label has no dead ball, is unknown on 46% of in-play frames, and (the probe
  of Nov 12) changes a median of 1.0 s before the new team's first touch.
- `EVALUATION.md`: a prediction that is perfect but 1 s late scores 0.65. So which mechanics the team head
  learns is worth more than anything the 28-fold rule has passed since F.
- Rule 1 scores 20 of its 28 folds against the standing SkillCorner label. Five candidates in a row (S1, S2, K1,
  K, and the unregistered probe without SkillCorner) gained 0.017 to 0.038 on the 8 folds of the other kind,
  each on 8 of 8, and the four that were gated failed on folds won. A rule that cannot pass a change of that
  kind cannot judge one.

It is a bet on the documentation, and the entry says so: if the provided match's ground truth behaved as the
standing SkillCorner label does, a model that meets rule 2 would lose to F3 by about what K loses with the wrong
value (0.03 to 0.045). **Justin's name is still needed, and adoption stays a separate decision of both.**

**Cost.** LightGBM about 55 minutes, the TCN about 3.4 hours in two processes.

## 2026-11-12

### AW (pre-registration): A with possession-length weights; an amendment to A's entry

**Written at 14:00 PST on Nov 12, before any score of A or of AW had been read.** A's LightGBM stage was at fold
5 of 28 and its gate had not run; nobody has opened a table of it. Claude, at Justin's request ("maybe there is a
way of combining the two ... the actions one with more weight for shorter possessions ... so it doesn't smooth
out too much"); David has not seen it. The 15th package on the same 28 folds. Code: `scripts/run_aw.py`.

**Amendment to A's entry (below).** It says "No other time constant, T, weighting or variant is tried, whatever
the result". That sentence was mine, an hour old, and Justin asked for one weighting before any result existed.
AW is that one. The sentence stands for everything else, and for AW too: no other weight, clip or T follows.

**Why this one.** S1 and S2 (Nov 10 and 11) put the weight `1 / L` on each training frame of the team heads, `L`
the length of the frame's possession in seconds, clipped to [1, 30]. Each gained on the 8 IDSSE + Metrica folds
and did nothing on SkillCorner, and failed on folds won (17/28). The probe of Nov 12 offers a reason: the
standing SkillCorner label has no short possession of the kind the weight is meant to protect. A changes those
labels, so the same weights may now act on all 28 matches. The weights were never run in both members at once.

**The change, and nothing else.** AW is A, plus `lenweight.inv_length` of the label each frame trains on (the
action label on SkillCorner, the standing one elsewhere) as the frame's weight in the LightGBM team head and in
the TCN's team loss. Same constants as S1 and S2, never tuned. The alive head and the TCN's alive loss carry no
weight, so AW's LightGBM alive probabilities must equal A's byte for byte: checked on every fold.

**Rules.** The two rules of A's entry, word for word, with AW in A's place, against F3. Rule 2 counts only with
David's and Justin's names, as there.

**Which of A and AW stands, under whichever rule is in force.** If only one of them passes it, that one. If both
pass it, AW when AW minus A at `target` on that rule's folds (28 for rule 1, the 8 IDSSE + Metrica folds for
rule 2) has a paired 95% interval above 0, otherwise A, the simpler one. If neither passes, F3. Adoption is
David's and Justin's either way, before Nov 14, 18:00 CET.

**Information, decides nothing:** the weights in one member only; each member alone; true possessions found by
length for F3, A and AW on each truth; predicted possessions per true one.

**Run.** After A's stages: `aw-gbdt` (about an hour), `aw-tcn` (about 4 hours, on the charger), `aw-gate`.

### A (pre-registration): SkillCorner training labels from its own action events

**Written before any score of A.** Claude, on Justin's word ("yes js do that", after reading the probe below);
David has not seen it. It is outside the final-week plan, adopts nothing, and is the 14th package on the same 28
folds. Code: `katena.external.skillcorner_actions`, `scripts/run_a.py`, `tests/test_skillcorner_actions.py`.

**Hypothesis.** The SkillCorner frame label (`possession.group`) is a phase-level label with other mechanics
than the target's ground truth is described as having (the probe, next entry). Relabelling the 20 SkillCorner
matches from SkillCorner's own `player_possession` events gives 20 more matches whose labels change owner at the
other team's first touch and carry dead ball, on the only real broadcast tracking we have.

**The change, and nothing else.**

- **Owner.** From the first frame of a player possession to the first frame of the next one in the same period,
  the ball is that player's team's.
- **Dead ball.** A gap between two player possessions holds a stoppage when SkillCorner flags the earlier one
  `game_interruption_after` or the later one `game_interruption_before`. The ball goes dead at the first gap
  frame where it is outside the pitch or the frame has no tracking, or 1.0 s after the last touch when neither
  happens, and at once after a foul. Play restarts 1.0 s before the next possession when that one starts as the
  reception or interception of the restart, and at its first frame otherwise. From the restart on, the ball is
  the taking team's.
- **Unknown.** Frames before a period's first player possession, and dead-ball frames with no tracking at all.
- **The two times (1.0 s each) are fixed here** from what a throw-in or a pass takes, not fitted to anything.
- **Then as every other label:** the sandwich merge at T = 2.
- **Training.** Both members exactly as F3's (rows, sequences, features, 1,200 trees, 4,800 steps, seeds), with
  two differences on SkillCorner rows only: the label, and `dead_ok` (they now train the alive head and the TCN's
  alive output). No IDSSE or Metrica row changes. The decoder's priors are F3's.

**What was looked at before this was written (no model output, no score).**

| | Home | Away | Dead | Unknown | Possessions a match | Median | Under 2 s | Under 4 s |
|---|---|---|---|---|---|---|---|---|
| Standing SkillCorner label, T = 2 | 0.275 | 0.266 | 0 | 0.459 | 272 | 7.2 s | 13.9% | 31.6% |
| Action label, T = 2 | 0.280 | 0.273 | 0.206 | 0.242 | 239 | 8.7 s | 9.3% | 25.0% |

The two agree on the team on 84.7% of the frames where both name one. `a-check`: a pool row's cached label is
the standing truth at its sample for all 28 matches; `gbdt.fit_heads` on the standing labels gives F3's cached
LightGBM probabilities of one fold byte for byte; no row outside SkillCorner changes; 148,637 of 443,350
SkillCorner rows change label; the alive head's rows go from 171,452 to 507,549 and the team head's from 334,270
to 339,490. The TCN stage is `run_k.py`'s loop with these labels, not checked against a cache.

**Two truths for a SkillCorner fold.** *Standing*: the label every gate so far has scored on, with predicted
dead ball read as no call. *Action*: the label above, scored as an IDSSE fold is. An IDSSE or Metrica fold has
one truth, and A does not change it.

**Rule 1, the standing rule, unchanged.** A against F3 on the 28 folds, every fold on its standing truth: at
`target` at least 70% of folds won and the paired 95% interval above 0; not worse at `severe`; not worse at
`target` on the 20 SkillCorner folds. I expect A to fail it: the standing SkillCorner truth is the label A stops
training on, so on those 20 folds A is scored against a convention it no longer learns.

**Rule 2, declared here and not standing.** The same thresholds on the 8 IDSSE + Metrica folds, whose truth A
does not touch: at `target` at least 70% of folds won (6 of 8) and the interval above 0; not worse at `severe`
on those 8; and on the 20 SkillCorner folds against the action truth, not worse than F3 at `target`. The 8 folds
have been seen by every package since Nov 6, and four of them passed this kind of test and were not adopted (S1,
S2, K1, K), so rule 2 is weaker evidence than rule 1 and says so. **It counts only if David and Justin both put
their names to it before the cut-off (Nov 14, 18:00 CET), as a change of protocol the technical document
states.** This entry does not adopt it.

**What stands.** If A passes rule 1 it is a candidate that passed the standing rule, and adopting it is David's
and Justin's decision. If it fails rule 1, F3 stands unless they adopt rule 2 and A meets it. Either way the
submission pipeline does not change here: adopting A would also need the all-30 training on action labels, new
expected hashes and the documents. No other time constant, T, weighting or variant is tried, whatever the result.

**Information, decides nothing:** each member swapped alone; the action team head with F3's alive head (is it
the owner labels or the dead-ball labels that move the score?); A against K on the folds both have; every other
evaluation world; alive/dead F1, team accuracy and predicted possessions per true one; true possessions found by
length, on each truth.

**What it cannot show.** That the action labels are right: SkillCorner makes its events automatically from the
same tracking, and a stoppage's two ends are approximations. That the target's ground truth follows them. And a
SkillCorner score against the action truth favours A by construction; it is a guard and a description, not
evidence that A is better.

**Run.** `a-gbdt` (about 50 minutes), `a-tcn` (about 4 hours, on the charger), `a-gate`. Smoke runs carry a tag
and are not the candidate.

### SkillCorner's own action events against the label we train on (description only; Claude, Justin has not read it yet)

**Nothing was fitted, scored or chosen.** `scripts/probe_sc_actions.py`, on the 20 SkillCorner development matches
(external), aggregates only. Justin asked for another approach to short possessions, "something like an action
focus". The pinned SkillCorner commit ships `*_dynamic_events.csv` for every match: about 936 `player_possession`
events a match (first touch to last, with team, start type and end type) and flags for game interruptions. We
downloaded them and use them for nothing (`docs/external-data.md`). The probe sets them beside the frame label
(`possession.group`, null -> X).

| Question | Answer, pooled over 20 matches |
|---|---|
| What is X (46% of in-play frames)? | 96% of X frames lie between a player possession flagged as followed by an interruption (throw-in, goal kick, corner, free kick, goal) and the next player possession. Half of those have no tracking (the broadcast cut away). Under 3% of X is a gap between possessions in open play |
| Does the frame label follow the last touch? | No. On labelled frames the team of the most recent player possession agrees with it on 82% |
| What does it do with a short possession of the other team? | It gives it to the team around it: 153 player possessions a match (17%), median 0.7 s, are labelled as the other team's. 22% of them start as an interception, 21% as a recovery |
| When does it switch? | A median of 1.0 s before the new team's first touch (25th to 75th percentile: 0.3 to 2.3 s) |
| What would last-touch labels with the interruption gaps as dead ball look like? | 247 possessions a match, median 6.9 s, 16% under 2 s, 33% under 4 s. IDSSE at T = 2: 324 a match, median 5.7 s, 15% under 2 s |

**What this suggests, and it is a hypothesis.** The SkillCorner label is a phase-level one: it swallows the other
team's short possessions, changes before the other team touches the ball, and marks dead ball as unknown. Those
are 72% of the team head's training rows and 20 of the 28 gate folds, and on those folds the scorer cuts spells
at X, so they do not test where a possession starts or ends, or dead ball. It would explain why S1, S2, K1 and K
each helped the 8 IDSSE + Metrica folds and did nothing on SkillCorner. The events would give SkillCorner labels
with the mechanics the target's ground truth is described as having: the owner changes at the other team's first
touch, dead ball is a label, and short possessions exist.

**Caveats.** SkillCorner's events are produced automatically from the same broadcast tracking, not by a person.
An interruption gap as defined here starts at the last touch before the stoppage and ends at the first reception
after it, so it runs a second or two long at each end. Relabelling changes the ground truth of 20 gate folds, so
it is not a candidate under the standing rule: it needs a registration that says which folds and which truth
decide, and David's and Justin's names. **Nothing is built and nothing is proposed for adoption here.**

**Literature, the same day** (a research agent, web only, nothing downloaded): candidate-based touch detection
(ELASTIC: local minima of player-ball distance and maxima of ball acceleration, 98% of true touches covered on
clean 25 Hz data), possession-gain rules (Vidal-Codina 2022, `databallpy`), boundary heads with segment voting
(ASRF) and ball-free transit states (PathCRF). All need a mostly clean ball or identified players. Our touch
features already do the acceleration half of the first.

### K: result (Claude; Justin has not read it yet)

**K fails the registered rule on the win count. F3 stands, and the label-kind column is closed.** One run on the
Mac, 01:00 to 05:07 PST on Nov 12, on the charger with Low Power Mode off: the TCN on 28 folds (245.6 min, 494 to
625 s a fold), then the gate. HEAD was `5315154`, the merge of main (F3's adoption) into the branch that holds
the registration of Nov 11 (`63e439f`); the merge changes no file the run reads besides an error message in
`gbdt.ensure_openmp`. External development matches only. Nothing in the registration was changed. Tables:
`work/final/run_k2.log`; verdict: `k_gate.json`, `registered_run: true`.

**Checks the run printed.** F3 re-decoded equals its stored scores on 56 of 56 values. No finite value was lost
when the sequences were stored as float16.

**K gate: the label-kind column in both members, against F3, 28 folds.**

| Comparison | Folds | K | F3 | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6267 | 0.6180 | +0.0086 | [+0.0043, +0.0131] | 17/28 |
| `severe`, all (rule: not worse) | 28 | 0.5919 | 0.5840 | +0.0079 | [+0.0038, +0.0122] | 16/28 |
| `target`, SkillCorner (rule: not worse) | 20 | 0.6086 | 0.6103 | -0.0017 | [-0.0059, +0.0026] | 9/20 |
| `severe`, SkillCorner | 20 | 0.5802 | 0.5816 | -0.0014 | [-0.0061, +0.0033] | 10/20 |
| `target`, IDSSE + Metrica | 8 | 0.6720 | 0.6374 | +0.0345 | [+0.0242, +0.0444] | 8/8 |
| `severe`, IDSSE + Metrica | 8 | 0.6213 | 0.5900 | +0.0313 | [+0.0221, +0.0407] | 6/8 |

K fails on the win count: 17 of 28 at `target`, the rule needs 20. The interval is above 0 and both guards hold
(neither interval on SkillCorner is below 0). It is the split K1 showed, twice as large: on the 8 IDSSE + Metrica
folds the gain doubles (+0.0345 against K1's +0.0175), and the 20 SkillCorner folds neither gain nor lose.

Information only:

- **Against K1:** +0.0041 [+0.0009, +0.0074] on 28 folds, 13/28; +0.0170 on IDSSE + Metrica (6/8).
- **The column in the TCN only, with `T1200`:** +0.0049 on 28 (17/28), +0.0221 on IDSSE + Metrica (8/8).
- **Members alone:** the TCN +0.0075 on 28 (16/28) and +0.0356 on IDSSE + Metrica (8/8), with -0.0038 on
  SkillCorner (8/20); the LightGBM member as in K1's entry.
- **Other worlds:** `mild` +0.0107 (17/28), `target_ids` +0.0100 (19/28), `target_smooth` +0.0110 (20/28); the
  two coupled worlds +0.045 and +0.042 (8/8 each).
- **Predicted possessions per true one at `target`:** 0.729 to 0.815 on IDSSE + Metrica, 0.760 to 0.741 on
  SkillCorner.

**K read with the other value of the column** (what a wrong value would cost; information, decides nothing).

| Folds, `target` | Wrong value against F3 | Wrong value against K |
|---|---|---|
| All (28) | -0.0361 [-0.0417, -0.0300], 0/28 | -0.0447 [-0.0514, -0.0379], 1/28 |
| IDSSE + Metrica read as kind 1 (8) | -0.0448 [-0.0549, -0.0349], 0/8 | -0.0793 [-0.0925, -0.0655], 0/8 |
| SkillCorner read as kind 0 (20) | -0.0326 [-0.0396, -0.0255], 0/20 | -0.0309 [-0.0383, -0.0236], 1/20 |

With the column in the TCN the wrong value costs four to seven times what it cost K1 (-0.006 to -0.009 against
F3). So K is a larger bet than K1 in both directions: about +0.035 if the provided match's ground truth follows
the IDSSE and Metrica process, as `CHALLENGE.md` describes it, and about -0.045 against F3 if it does not. No
external match tells us which, and the provided match may not.

**What stands, as registered: F3.** K1 failed and K failed, each on folds won, and neither is re-gated on 8
folds. No other encoding of the column, and no other use of it, is tried. With B1 these were the 11th to 13th
packages gated on the same 28 folds; the technical documents now say thirteen.

**What this does not say.** The same open question as in the K1 entry, now with a larger number behind it: all
four candidates since F3 that helped at all (S1, S2, K1, K) helped the 8 IDSSE + Metrica folds and did not help the
20 SkillCorner folds, and the rule counts folds. Whether to judge such a change another way is
David's and Justin's to decide before Nov 14, 18:00 CET; this entry does not reopen it and adopts nothing.

### Linux on an Intel CPU: both models give this PC's CSV (result)

**F3 and F, trained from an empty cache under Linux on the PC's Intel CPU, each give the Windows reference's CSV
for the dressed match.** That was the one pair of OS and CPU that F3's cross-machine check had not covered
(Nov 11, "Not covered"). Claude on David's PC, overnight. External matches only; no fold was scored.

**What ran.** `D:/katena/tools/wsl_repro.sh`: the steps of `repro.yml`, by hand, at `8f3040d` (main before the
adoption; both specs are named on the command line), in the Ubuntu 24.04 already installed under WSL 1 on the PC
(Intel i7-7700). It installed uv 0.13.1 and the locked environment (torch 2.15.0+cu132, LightGBM 4.7.0), and
copied the raw external downloads from the Windows side, as `actions/cache` restores them on a runner. Everything
else it built from empty: the 30 canonical matches, the cache, both models and the dressed bundle. All of it
lives in one folder of that distribution, and Ubuntu's `libgomp1` package was unpacked there, not installed.

| | F3 | F |
|---|---|---|
| Training inputs: 861 cache files, compared by content | Same as the Windows reference | Same |
| LightGBM member | `056ffa8a...`, same | `b2146150...`, same |
| TCN weights | `61bc4699...`, differs | `a7623330...`, differs |
| Dressed CSV | `72d7c3fb...`, **same** | `909f4ffa...`, **same** |

- **F's TCN is the one GitHub's Ubuntu runner on an Intel Xeon gave on Nov 10** (`a7623330`). So this run
  behaves as Linux on Intel does elsewhere, although under WSL 1 the kernel is Windows' and only the libraries
  are Linux's.
- **F3 now has five sets of TCN weights and one CSV**: the seven OS and CPU combinations of GitHub's runners and
  this PC (the entry on the adoption, below), and this one.
- The HSMM duration prior differs by OS, as on Nov 10. Viterbi never reads it.
- **Which OpenMP runtime LightGBM gets did not matter.** F was trained once more with the `libgomp.so.1` that torch ships in its wheel, in place of Ubuntu's package: the same `bundle.pkl`, byte for byte.

**What this is not.** A second physical machine: the CPU is the reference's own. And a real Linux kernel: numpy
warns under WSL 1 that `long double` is not the type it expects. Nothing in the pipeline uses that type, and
every array of every stage equals the reference's.

**Times.** Corpus 23.5 min and worlds 37.2 min with 2 workers. F3's `train` took 85.5 min: 73 for the tree model
while other heavy jobs shared the PC's four cores (it takes 3 alone on Windows), and 11 for the TCN. F's
`train` took 4.3 min once the PC was quiet: 29 s for the trees and 2.7 min for the TCN. So the slow tree model
was the shared machine, not Linux. `predict` took about a minute.

**It also found a missing library.** Its first attempt stopped at once because this Ubuntu has no
`libgomp.so.1`; that became PR #28 (below). Reports: `D:/katena/logs/wsl`.

### F3 is the submission model (decision)

**F3 replaces F as the submission model, from the merge of PR #26 on 2026-11-12.** That is two days before the
plan's cut-off (Nov 14, 18:00 CET).

- **Who decided.** David, by handing the decision over. He was asked twice whether to merge PR #26, and on
  Nov 12 at about 03:17 CET he answered "make the best decisions you see fit". Claude then decided yes.
- **Why yes.** The registration of Nov 11 said in advance what a pass would mean, and everything in it passed:
  the gate on both machines (+0.0161 [+0.0118, +0.0207], 21 of 28 folds; not worse at `severe`, not worse on the
  SkillCorner folds) and the reproduction checks R0 to R5. F3 also gains in the smooth and the coupled worlds, so
  the gain does not rest on one kind of simulated noise.
- **What it does not show.** The 28 folds had been seen before, 8 of them chose the tree count, and on the 20
  SkillCorner folds, the only broadcast tracking we have, the gain is +0.0073. Nothing here says the hidden score
  rises by 0.016.
- **Justin has not confirmed the merge itself.** He put F3 forward and ran its first gates (PR #23, ask 1), and
  PR #26 tells him. If he objects, reverting PR #26's merge commit undoes it: one constant
  (`katena.pipeline.SUBMISSION_SPEC`), the expected hashes and the documents. F's code path stays (`--spec F`).

**Checked on this branch before the merge.**

| Check | Result |
|---|---|
| CI on the pull request | Passes on Ubuntu, Windows and macOS |
| `repro.yml` with the branch's default spec, run 43132799061 at `d794f13` | F3 trained from an empty cache on three runners (Ubuntu on AMD EPYC 9V45, Windows Server on AMD EPYC 7763, macOS on Apple M1). The dressed CSV is this PC's `72d7c3fb...` on each, and the LightGBM hash is `056ffa8a...` on each. With Nov 11's runs that is seven OS and CPU combinations |
| The whole test suite on this PC | Passes |

**What changes for the real run.** `scripts/real_run.py --team <team_name>` now trains and checks F3: `train`
takes about 20 minutes on the PC, and the hashes it expects are gbdt `056ffa8a...`, tcn `4ec1a79e...` and bundle
`0a6359f3...` (`docs/repro/expected-windows.json`). The command itself is the same.

**Still open, and not Claude's to do:** the team name and the real run, which only David can do.

### The submitted technical document is a short one (decision)

**`docs/technical-document.md` is now 5 pages, and the 9-page text is `docs/technical-document-full.md`.** The
organisers ask for "a few pages". David handed this decision over with the adoption, and Claude decided to cut.

- **What the short one keeps.** The six things `SUBMISSION.md` asks for: the approach, the assets used and
  dropped, every labelled dataset with its source and who labelled it, the possession rules, the limitations and
  the reproduction. One results table takes the model from the starter algorithm to F3. No number is new: each
  is in the long version or in this log.
- **What went to the long version only.** The feature-group table, the severity table and the world formulas,
  the bench table, the per-package tables of F and F3, the per-variant rows of the label conventions, the list of
  machines, the cost table and the map of the repository.
- **Where the real run's blanks are.** Only in the short document: its three `TODO(after the real run)` markers
  (the public-window result, the tagged commit, the file's SHA-256). `scripts/build_document.py` prints the
  short one and refuses while a marker is left.

**The three disclosure points of the plan are decided: all three are disclosed**, in section 9 of the short
document and section 10 of the long one.

| Point | What the document says |
|---|---|
| A hosted AI assistant | Most of the code and of the document was written with one, under the team's direction. The competition's data was kept away from it: the repository's rules forbid it the data folder, only team members ran anything on the provided match, and those runs print pass or fail and aggregates only |
| The Zenodo record | An automated agent searching for external datasets opened a public record that held statistics about the provided game. The file was deleted and nothing from it was used |
| StatsBomb | 35 matches were downloaded to a temporary folder before the agreement was read. They were not used |

**David has to confirm the first row.** It says what was and was not shown to an assistant, and only David and
Justin know that for their own sessions. If a file, a frame or a per-frame value of the provided match ever
reached one, the sentence has to change before the document is submitted.

### The night's other work (the four entries below)

Claude on David's PC, overnight; David had read none of it when it was written. None of the four changes a
model, a feature, a label or a hash. Nothing ran on the provided match and no fold was scored.

### Stress check of the submission path: `scripts/stress_predict.py` (result)

**Every case passes, with F and with F3: 30 of 30 each.** `scripts/stress_predict.py` at `1851600`, on the dressed
external match (IDSSE `DFL-MAT-J03WMX`, 173,528 frames), with the all-30 models in `D:/katena/work-submit`, which it
reads and never writes. Claude on David's PC, overnight. External data only. No score was computed and nothing was
chosen.

**Why.** Until the real run, the provided match has been read by the loader check only. No feature, model or
decoder code has run on it, and nobody may look at it to debug. The training worlds corrupt external matches at
5 Hz, in the ways the degrader was built for. This check damages the native-rate match that `predict` reads, in
ways the degrader does not produce, and asks one thing of each case: does `label_match` finish and write a valid
file?

**What a pass means.** `label_match` returned. Every frame in the graded halves is H, A or D, and every other
frame is X. The CSV passed the official checks (`pipeline.write_submission`). A second labelling of the same case
gave the same labels (`--twice`). It says nothing about accuracy: the labels of a damaged match are not scored.

Every case but the first starts from `target_like`. The last two columns give, for information, the share of
graded frames whose label differs from that case's.

| Case | What is damaged | F: frames relabelled | F3: frames relabelled |
|---|---|---|---|
| `as_exported` | The match as exported: 22 identified players, the provider's ball | 2.5% | 2.8% |
| `target_like` | 46 anonymous slots, tracks cut into fragments, 41% of the ball missing in bursts (the reference) |  |  |
| `target_like_through_the_file` | The same, written as an ATD bundle and read back |  |  |
| `away_team_absent_second_half` | No away slot tracked in the second half | 4.7% | 4.3% |
| `ball_absent_everywhere` | No ball on any frame | 46.8% | 35.5% |
| `ball_absent_first_half` | No ball in the first half | 25.3% | 18.4% |
| `ball_flickers` | The ball missing on a random half of the frames, frame by frame | 1.8% | 1.7% |
| `ball_frozen_everywhere` | The ball on the centre spot whenever it is seen | 32.9% | 39.8% |
| `ball_off_pitch` | 2% of ball samples up to 30 m off the pitch, 0.5% up to 10 km away | 1.5% | 2.4% |
| `ball_one_coordinate` | On a tenth of the frames, only x or only y of the ball | 1.6% | 2.2% |
| `ball_random_every_frame` | The ball at a random place on every frame | 14.7% | 33.3% |
| `ball_seen_10_percent` | The ball seen on 10% of frames | 12.2% | 9.4% |
| `everyone_on_one_spot` | Every player on one point for 5 minutes | 4.2% | 4.1% |
| `everything_absent_5_minutes` | Nothing tracked for 5 minutes of each half | 7.1% | 7.3% |
| `ghost_duplicates` | Eight slots repeat another slot for a third of the match | 0.6% | 0.9% |
| `half_edges_empty` | Nothing tracked in the first and last 2 minutes of each half | 4.0% | 4.3% |
| `heavy_jitter` | 1 m of noise per frame on every player and the ball | 16.3% | 13.4% |
| `home_team_absent_everywhere` | No home slot tracked at all | 20.9% | 24.6% |
| `infinite_values` | 400 ball samples and 0.01% of player samples infinite | 0.0% | 0.1% |
| `players_absent_5_minutes` | No player tracked for 5 minutes of each half; the ball stays | 2.9% | 3.0% |
| `players_frozen_10_minutes` | Every player frozen for 10 minutes | 6.5% | 6.2% |
| `players_off_pitch` | 1% of player samples up to 500 m away | 1.1% | 1.2% |
| `slots_trade_teams` | 40 stretches of 20 to 120 s in which a home and an away slot swap | 0.4% | 0.5% |
| `speed_absent` | No speed value | 0.0% | 0.0% |
| `speed_garbage` | Random speed values | 0.0% | 0.0% |
| `teams_do_not_change_ends` | The second half mirrored, so the teams keep their ends | 0.0% | 0.0% |
| `teams_overlap` | Every player within a metre of the halfway line, so no attack direction can be read | 41.0% | 41.8% |
| `three_slots_per_team` | Three slots per team | 24.1% | 16.6% |
| `tracked_outside_the_halves` | Tracking present outside the graded halves too | 0.0% | 0.0% |
| `tracks_flicker` | Every slot seen on a random 30% of frames | 6.3% | 5.4% |

- **Through the file format.** The reference case was also written as an ATD bundle with 46 slots and read back
  by `load_target`: every field is equal and no label differs.
- **Four damages change no label** in either model: no speed values, random speed values, teams that keep their
  ends, and tracking outside the halves. So the features do not read the speed column, and the attack direction
  comes from the positions alone.
- **Infinite values behave like missing ones** (0.0% and 0.1% of frames).
- **The largest effects are the expected ones.** With no ball at all, a third to a half of the graded frames
  change, most of them to D: its share rises from 0.41 to 0.76 (F3) and 0.88 (F). With no attack direction to
  read, 41% change.
- **Cost.** 26 to 75 s a case for the two labellings, about 20 minutes (F) and 29 (F3) for the 30 cases beside
  another heavy job.

**Not covered.** Damage to the file itself (malformed lines, missing frames, another number of slots): the loader
check on the real file is the test of that, and it passed 16 of 16 on Nov 7. Several damages at once. Accuracy.

**No failure was found, so no code changed.** Tables: `D:/katena/logs/stress/stress_F.json` and `stress_F3.json`.
`tests/test_stress_predict.py` checks on a synthetic match that each case does the damage its name says.

### A bare Linux and an outside read of the documents: what would have stopped a reproduction (PR #28)

**Seven problems, all fixed, none of which touches a label.** They came from two places:

- **A Linux that is not a GitHub runner.** The pipeline's first run on Ubuntu 24.04 under WSL on the Windows PC,
  as installed, with nothing added (the entry on Linux on Intel, above).
- **An outside read.** A fresh agent got the repository and the organisers' `SUBMISSION.md`, and was asked to
  regenerate the file as an organiser would and to report what stops or misleads. It read only; it ran nothing
  but `--help`.

| Problem | Found by | Fix |
|---|---|---|
| Without the system's `libgomp.so.1`, every script stopped with "install it (brew install libomp)". LightGBM's Linux wheel takes that library from the system; GitHub's runners have it, a bare WSL distribution and slim container images do not | The WSL run | `gbdt.ensure_openmp` names the platform's package (`apt-get install libgomp1`, `dnf install libgomp`) and quotes the loader's error |
| A misplaced data bundle was noticed only by `predict`, after the download and the training, with no path in the message | The read | `make_submission.py bundle` checks that the three files `predict` reads exist (names only). `make submit` and `make_submission.py all` run it first |
| With no Smart Tagging file, `predict` skipped the kickoff rule and went on; only a line of its report said so. The result is a valid CSV with other labels | The read | `predict` on the competition match stops when the file is missing. An exported external match (`--timeline`) is unchanged |
| `make validate` ended with `shasum -a 256`, which is not on every Linux | The read | Python computes the hash |
| `$(OUT)` was unquoted, so a team name with a space broke `make submit` | The read | Quoted in every recipe |
| `REPRODUCE.md` did not say where the bundle goes, and left out git, network access, the uv version, `libgomp1`, and that torch's Linux wheels add about 6 GB | Both | A "Where the data bundle goes" section and a fuller Requirements list |
| `real-run.md` said `check_kickoff_tags.py` runs "once that script is on main"; the by-hand list and `make check-target` left it out | The read | One list in all three places |

**The Makefile was run with a real GNU make** (4.3, under WSL; `make` is on neither the Windows PC nor in CI's
steps), on a folder of empty files named like the bundle and on a published baseline file:

- `make -n submit TEAM="My Team"` prints the eight commands with the name quoted.
- With no bundle, `make submit` stops at its first step, names the three missing paths, and creates no cache
  and starts no download.
- `make validate TEAM="My Team"` validates the file and prints the hash `sha256sum` gives.
- Without `libgomp.so.1`, the first step prints the package to install.

**The read's other findings** are fixed in the technical document's long version: the summary names the one
Smart Tagging input, the machine lists match the tables of Nov 10 and 11, "ten gated packages" says what it
leaves out, and section 8 lists six steps. It checked and found right: every command and flag in the
documents, the library versions against `uv.lock`, F's and F3's sizes against the code, and the hashes in
`expected-windows.json` and `real-run.md` against this log.

**Its list of cuts** became the short technical document (the entry above).

### The technical document as a PDF (PR #27)

`uv run scripts/build_document.py --team <team_name>` writes `submissions/<team_name>.pdf`, the name the form
asks for: A4, page numbers, the team name in place of `<team_name>`. It refuses while a `TODO(` marker is left
(`--draft` builds anyway). Markdown goes to HTML with `markdown-it-py`, which the script declares for itself, so
`uv.lock` is unchanged; Edge or Chrome prints the page headless. The long document printed to 9 pages; the short
one (the entry above) prints to 5.

### For the record: F3's dressed-match scores to six decimals

The technical document on `david/f3-adopt` quotes them, and this log had four decimals (R2, Nov 11). From
`D:/katena/logs/f3/compare_held.log`: spellF1 0.722130 and macroF1 0.941533 on the evaluation path and from the
official scorer on the CSV, 260 predicted possessions against 341 true. One fold at clean input, as R2 said.

## 2026-11-11

### K1 and B1: result (Claude; Justin has not read it yet)

**Both fail the registered rule. F3 is unchanged. K's TCN has not been trained yet.** One run on the Mac at HEAD
`63e439f`, the commit that holds the registration (below), 13:44 to 14:34 PST on Nov 11, on battery in Low Power
Mode (a LightGBM fold took its usual time). External development matches only. Nothing in the registration was
changed. Tables: `work/final/run_kb.log`; verdicts: `k1_gate.json`, `b_gate_f3.json` beside it, both with
`registered_run: true`.

**Checks the run printed.** F3 re-decoded equals its stored scores on 56 of 56 values, in both scripts. B1's
decoder at lam = 0 gives the incumbent's decoded rows on 56 of 56. K1's team-head code on 213 columns gave
`T1200`'s cached probabilities byte for byte (the check before the registration).

**K1 gate: the label-kind column in the LightGBM team head, blended with `LT`, against F3, 28 folds.**

| Comparison | Folds | K1 | F3 | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6225 | 0.6180 | +0.0045 | [+0.0013, +0.0079] | 18/28 |
| `severe`, all (rule: not worse) | 28 | 0.5885 | 0.5840 | +0.0044 | [+0.0013, +0.0078] | 17/28 |
| `target`, SkillCorner (rule: not worse) | 20 | 0.6096 | 0.6103 | -0.0007 | [-0.0044, +0.0028] | 10/20 |
| `severe`, SkillCorner | 20 | 0.5823 | 0.5816 | +0.0007 | [-0.0027, +0.0040] | 11/20 |
| `target`, IDSSE + Metrica | 8 | 0.6550 | 0.6374 | +0.0175 | [+0.0104, +0.0250] | 8/8 |
| `severe`, IDSSE + Metrica | 8 | 0.6038 | 0.5900 | +0.0138 | [+0.0075, +0.0207] | 6/8 |

K1 fails on the win count: 18 of 28 at `target`, the rule needs 20. The interval is above 0 and both guards hold.
The split is the one S1 and S2 showed, with one difference: the SkillCorner folds are not hurt at either
severity, alone or blended. Information only: the member alone is +0.0090 at `target` on 28 folds (20/28), +0.0312
on IDSSE + Metrica (8/8) and +0.0001 on SkillCorner (12/20). In the worlds where ball loss is tied to dead ball
the blend gains +0.026 and +0.024 (8/8 each). The column takes 0.7% to 1.0% of the team head's split gain.
Predicted possessions per true one: 0.729 to 0.766 on IDSSE + Metrica, 0.760 to 0.747 on SkillCorner.

**K1 read with the other value of the column** (what a wrong value would cost; information, decides nothing).

| Folds, `target` | Wrong value against F3 | Wrong value against K1 |
|---|---|---|
| IDSSE + Metrica read as kind 1 (8) | -0.0063 [-0.0121, -0.0000], 3/8 | -0.0238 [-0.0327, -0.0151], 0/8 |
| SkillCorner read as kind 0 (20) | -0.0089 [-0.0139, -0.0038], 3/20 | -0.0082 [-0.0134, -0.0030], 5/20 |

So the column is used, and it is not free: with the right value K1 gains 0.0175 on the folds labelled as the
provided match's ground truth is described, and with the wrong value it loses 0.006 to 0.009 against F3.

**B1 gate: the boundary head in the decoder, on F3's probabilities, against F3, 28 folds.**

| Comparison | Folds | B1 | F3 | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6183 | 0.6180 | +0.0003 | [-0.0020, +0.0028] | 13/28 |
| `severe`, all | 28 | 0.5846 | 0.5840 | +0.0006 | [-0.0015, +0.0030] | 11/28 |
| `target`, SkillCorner | 20 | 0.6088 | 0.6103 | -0.0015 | [-0.0040, +0.0010] | 7/20 |
| `target`, IDSSE + Metrica | 8 | 0.6423 | 0.6374 | +0.0049 | [-0.0010, +0.0110] | 6/8 |

B1 fails on every part of the rule: no gain. It predicts more possessions (0.751 to 0.777 per true one) and they
are not the right ones. The head is weak: held-out average precision 0.107 to 0.515, mean 0.221, on a target whose
base rate is 0.043. Information only: lam = 0.5 is -0.0005, lam = 2 is -0.0970. **B1 is closed**: no other
tolerance, lam, size or target is tried, and the pair test with K is void.

**What stands, as registered.** K1 has failed and is not re-gated on 8 folds. K (the column in both members) is
still to run: its condition was that K1 is not worse than F3 at `target` on 28 folds, and it is not. Its TCN
needs about 4 hours on the charger with Low Power Mode off, so it waits for Justin. If K fails, F3 stands.

**What this does not say.** Whether the 28-fold rule is the right judge of a change that helps one labelling
process only. That question was left open on Nov 10 (the probe's point 2) and needs David's and Justin's names;
this entry does not reopen it and adopts nothing.

### K1, K and B1 (pre-registration): a label-kind column, and a boundary head for the decoder

**Registered before any score of any of the three.** Claude, at Justin's request ("implement possible
improvements"), branch `claude/status-check-7809aa`, the Mac (macOS arm64). External development matches only: no
`data/`, no `evaluation/gt/`, no public window, no locked bench. Code: `scripts/run_k.py`, `scripts/run_b.py`,
`katena.decode.tv`, `katena.models.boundary`. Nothing below changes after a result.

**Standing.** Outside the final-week plan, which adopts nothing after Nov 14, 18:00 CET. David has not seen or
agreed to it. A pass here makes a candidate for the team to decide on; it adopts nothing and changes no file of
the pipeline. These are the 11th, 12th and 13th packages gated on the same 28 folds.

**Why.** Two results below. (1) The probe of Nov 10: dropping the SkillCorner matches from the LightGBM team
head gained +0.035 blended on the 8 IDSSE + Metrica folds (8/8), and S1 and S2 then split the same way: gains on
IDSSE + Metrica, nothing or a loss on SkillCorner. The two groups are labelled by different processes
(`label-convention.md`, section 2): SkillCorner's owner comes with automatic tracking, is unknown on 46% of
in-play frames and has no dead ball; IDSSE's and Metrica's come from corrected tracking or events, keep the last
team on a loose ball and have dead ball. One team head is fitted to both without being told which is which. K
tells it. (2) The model predicts 0.75 possessions per true one, and no constant change of the decoder gains more
than 0.002. B1 lets the switch cost vary in time instead.

**What was seen before this entry.** Everything in this log up to here, including the probe, S1 and S2 on all 28
folds, and Justin's look at the public labels beside the video (the entry below; no prediction or score). Of the
new code: `run_k.py k1-check` and `run_b.py b-check` on `idsse/DFL-MAT-J03WMX` (both print one boolean: the code
path reproduces the incumbent's cache; no score). A smoke of K2's code at 20 steps on two folds and of the
boundary head at 20 trees on one fold, under the tag `smoke`: neither is the candidate, no spellF1 was computed,
and the head's smoke printed one held-out average precision (0.297). No fold of K1, K or B1.

**Incumbent.** F3 as it passed on both machines: the 1,200-tree LightGBM member (`T1200`) blended at w = 0.5 with
the 4,800-step touch TCN (`LT`), Viterbi at tau 2. Read from the Mac's cache of Nov 9, not refitted.

**The column** (`label_kind`). 1 on every row of a SkillCorner match, 0 on every row of an IDSSE or Metrica
match. It is a property of the dataset's labelling process, known without reading a label. A held-out match is
predicted with its own dataset's value. **For the provided match the value would be 0**, fixed here from
`docs/upstream/CHALLENGE.md` alone: its ground truth is made mechanically from corrected tracking, keeps the last
team on a loose ball, and has dead ball. The public window did not choose it and will not be used to check it.

| | Changes | Everything else |
|---|---|---|
| **K1** | The LightGBM team head reads the column (214 columns). | As `T1200`: pool, parameters, 1,200 trees. The alive head has only kind-0 rows, so it is not refitted: `T1200`'s cached alive probabilities are used. |
| **K** | K1, and the TCN reads the column as a 214th input channel. Its alive output is always read with the column at 0 (no kind-1 row has an alive target), its team output with the match's own value. | As `LT`: 4,800 steps, seed 0, 4 threads, the `d2j` sequences, the scaler (now over 214 columns). |
| **B1** | Decoder only, on F3's probabilities. A boundary head (LightGBM, F's parameters, 1,200 trees, 213 columns, all 27 training matches) is trained on "the owner changes within 0.4 s of this sample" in the T = 2 labels, unknown samples filled with the last owner. In the Viterbi, the odds of a switch into a sample are multiplied by the head's odds there over its training base rate (`katena.decode.tv`, lam = 1). | Emissions, priors, tau 2, blend weight, both members. |

The tolerance (0.4 s), lam (1) and the head's size (1,200) are fixed here and not tuned. lam = 0.5 and 2 are
printed as information and are not candidates.

**Gates.** Each on all 28 folds against F3, under the rule F3 itself passed: at `target` at least 70% of folds
won and the paired 95% bootstrap interval above 0; not worse at `severe`; not worse at `target` on the 20
SkillCorner folds alone.

- **K1** against F3 (K1's LightGBM member blended with `LT`).
- **K** against F3. The TCN is trained unless K1 is worse than F3 at `target` on 28 folds (mean below 0 and the
  interval below 0); in that case K is dropped unrun.
- **B1** against F3.

**What stands, fixed in advance.** K if it passes; otherwise K1 if it passes; otherwise F3. K and K1 are two
tests of one idea on the same folds, and are reported as such. If B1 passes and a K passes, the pair is not
assumed: B1's decoder on that K's probabilities is gated against that K under the same rule
(`run_b.py b-gate --on k`). A candidate that wins on the 8 IDSSE + Metrica folds and fails on 28 has failed: it is
not re-gated on 8.

**Information the run prints, and how it is read.** For K1 and K, the same model read with the *other* value of
the column (`K1x`, `Kx`), against F3 and against itself. This is what the candidate would score if the value
chosen for a match were wrong. It decides nothing here; it is the number the team needs to weigh the 0 above.
Also: the members alone, the SkillCorner and IDSSE + Metrica folds apart, the other evaluation worlds, predicted
possessions per true one, the column's share of the team head's split gain, the boundary head's held-out average
precision.

**Checks the run prints.** K1's team-head code on 213 columns gives `T1200`'s cached probabilities byte for byte
(done: true). B1's decoder at lam = 0 gives the incumbent's decoded rows on every fold. F3 re-decoded here equals
the stored scores.

**Cost.** K1 about 20 minutes, B1 about 20, K's TCN about 4 hours (it needs the charger and Low Power Mode off).

### A look at the public window beside the tactical video, labels only (Justin; written up by Claude)

**Nothing was changed because of this look.** Justin watched the public 5 minutes of the tactical video next to
the public label file, to understand what counts as possession (allowed by the Nov 3 clarification: "inspecting
the provided data and the video to better understand what counts as possession"). No prediction of ours was
involved and no score was computed. Claude saw the public label file and Justin's description in words, not the
video or any provided file. The times Justin gave are estimates and the reading is one person's.

What he reported, as general statements:

- A loose ball, and a ball in the air, stay with the last team. The label changes at the other team's first
  deliberate touch, not when the first team loses the ball.
- A pass that strikes an opponent, and a tackle that knocks the ball loose, did not change the label until a
  player controlled the ball.
- Both possessions under 2 s in the window (0.8 s and 1.3 s) followed a deliberate play by the team: a regain with
  a pass, and an intercepting header. Each was lost again at once. The second sits on the edge of the written
  rule ("an aerial duel won without subsequent control" is not a change).
- The one short possession that ran into dead ball (3.5 s) was an interception whose first touch was a pass that
  rolled out of play. It is labelled as a possession of the intercepting team.
- Dead ball starts at the whistle (after an offside, play stays live until it) and ends when the restart is
  played. No stoppage was seen without a dead-ball label.

How this sits against `docs/label-convention.md`:

- The sandwich at T = 2 would erase both possessions under 2 s. This is three cases in 17 possessions, and a
  person describing play describes actions, so it does not show that the rule is action-based.
- The touch-then-out case agrees with the default `tail=False`. One case.
- The dead-ball mechanics agree with IDSSE's `BallStatus`.

**No decision.** `T_MAIN` stays 2 and `tail` stays off. The window may not choose between versions, and none was
chosen. The question for the organisers (label-convention.md, section 7) is still the way to settle T; as far as
this log shows it has not been sent.

### F3 on the Windows PC: the gate (result)

**F3 passes the registered rule on the second machine. With R0 to R5 passed (below), F3 is put forward for
adoption, as the registration reads this outcome. It is not adopted here:** that is David's and Justin's
decision, and the plan's cut-off is Nov 14, 18:00 CET. One run of `scripts/run_f3.py gbdt tcn score gate` at
`e9f8f95`, cache `D:/katena/work-f3`, external development matches only. Nothing in the registration was
changed. The process was restarted once, at 12:58, so that two processes could share the TCN stage; the four
LightGBM folds already cached were kept. `f3_gate.json` has `registered_run: true`.

**The gate: F3 against F (F's stored Windows scores), 28 folds.**

| Comparison | Folds | F3 | F | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6180 | 0.6019 | +0.0161 | [+0.0118, +0.0207] | 21/28 |
| `severe`, all (rule: not worse) | 28 | 0.5840 | 0.5696 | +0.0144 | [+0.0095, +0.0191] | 19/28 |
| `target`, SkillCorner (rule: not worse) | 20 | 0.6103 | 0.6030 | +0.0073 | [+0.0021, +0.0122] | 13/20 |
| `severe`, SkillCorner | 20 | 0.5816 | 0.5804 | +0.0012 | [-0.0044, +0.0064] | 11/20 |
| `target`, IDSSE + Metrica | 8 | 0.6374 | 0.5991 | +0.0383 | [+0.0288, +0.0469] | 8/8 |
| `severe`, IDSSE + Metrica | 8 | 0.5900 | 0.5426 | +0.0474 | [+0.0373, +0.0573] | 8/8 |

All three registered conditions hold: 21 of 28 folds won with the interval above 0 at `target`, not worse at
`severe`, not worse at `target` on the SkillCorner folds alone.

**Checks the run printed.** The 1,200-tree member's first 150 trees give F's cached member byte for byte on 28
of 28 folds. F, its LightGBM member and its TCN member, re-decoded from the copied probabilities, equal F's
stored scores on 164 of 164 fold and world pairs.

**The two machines agree to the printed digits.** The Mac's means were 0.6180 at `target` and 0.5840 at
`severe`; so are these. Each change alone also reproduces the Mac's table: `T1200` against F +0.0041
[+0.0015, +0.0070], 21/28, and F3 against `T1200` +0.0120 [+0.0082, +0.0160], 21/28. The TCN's weights differ
between the machines, and that moved no score at four decimals.

**Information only.**

| Comparison at `target`, 28 folds | Diff | 95% paired CI | Wins |
|---|---|---|---|
| `LT` with F's 150 trees, against F | +0.0114 | [+0.0074, +0.0154] | 21/28 |
| LightGBM member alone, 1,200 trees against 150 | +0.0067 | [+0.0024, +0.0111] | 17/28 |
| TCN member alone, `LT` against F's TCN | +0.0244 | [+0.0186, +0.0302] | 24/28 |

- **Every evaluation world.** F3 meets the adoption rule in `mild`, `target`, `target_smooth` and the three
  coupled worlds (8/8 each, +0.043 to +0.055). In `target_ids` it gains +0.0127 [+0.0079, +0.0177] with 18 of 28
  folds, and at `severe` 19 of 28: above 0, under the 70% bar that only `target` has to clear.
- **Where the gain is.** IDSSE + Metrica: +0.038 at `target`, 8/8. SkillCorner: +0.007 at `target`, and
  nothing at `severe` (+0.0012, interval through 0).
- **The TCN member alone is above the blend**: 0.6205 against 0.6180 at `target`, 0.5901 against 0.5840 at
  `severe`. That was not a registered comparison and has no interval. The blend weight stays 0.5, fixed before
  any result, and no other weight is tried. With F it was the other way round (TCN alone 0.5961, blend 0.6019).
- macroF1 on the 8 dead-label folds: 0.8781 to 0.8920 at `target`. Predicted possessions per true one, 28 folds:
  0.738 to 0.751.

**What this is not.** New evidence of the weight of unseen folds: the Mac had scored the same models on the same
folds, and 8 of the 28 chose the tree count. `T1200` and `LT` were the 7th and 8th packages gated on these folds.
The locked benches are spent, so F3 has no bench read.

**Cost.** LightGBM stage: 100 to 150 s a fold. TCN stage: 657 to 984 s a fold (median 747 s) with two heavy
jobs running; 15 folds in one process and 13 in the other. Start to verdict: 12:44 to 17:22.

**Next, if David and Justin say yes.** Merge the prepared change (`david/f3-adopt`: `SUBMISSION_SPEC`, the
expected hashes, `docs/real-run.md`, `docs/REPRODUCE.md`). Then David's real run trains and checks F3 with the
same one command. If the answer is no, or does not come before the cut-off, F stays and nothing else changes.

### F3's R5 on Justin's Mac: the 1,200-tree member's hash agrees with the Windows PC (Claude; Justin asleep)

For the check R5 that `david/f3-assembly` registered (PR #25). `scripts/make_submission.py train --spec F3`,
run on the Mac (macOS arm64, Apple M5) from an export of that branch at `3863b47`, with this branch's
locked environment (the lock file is the same on both) and the Mac's own cache of the 30 external matches. No
score was computed and nothing ran on the provided match. It wrote `work/final/model_F3/`; F's `model/` was not
touched. Log: `work/final/f3_train_mac.log`.

| | Mac | Windows PC (PR #25) | |
|---|---|---|---|
| Pool | 657,133 rows x 213 columns | 657,133 rows x 213 columns | same |
| gbdt | `056ffa8a41eaf246b741234351e30dcd8e5191446493e24e8c1ea3f2e6a5bc47` | `056ffa8a...6a5bc47` | **same** |
| tcn | `bf52132cc2d52eb903390598c99dc328aacfbdbea4a9679aa6ce0b6beeb1070b` | `4ec1a79e...` | differs |
| bundle.pkl | `36cc2eb289495e9b0c3cc07fdef395d70978ba59b5b56db485076a30b077d940` | `0a6359f3...` | differs (it holds the TCN) |

**R5 holds on this Mac.** The TCN's bytes differ, as they do between every pair of machines in PR #25's table;
they also differ from the `macos-15` runner's (`55cc7100...`), which is an M1. Whether this Mac's TCN gives the
same labels on the dressed match was not checked here: that is R4's question, and `dressed_check.py` was not
run. Times on the Mac: LightGBM 32 s, TCN 486 s.

### S1 and S2: result (Justin + Claude)

**Both fail. The unweighted members stay, and `T1200` + `LT` (F3 on `david/f3-assembly`) is unchanged.** One
unattended run of `scripts/night_s12.sh` on the Mac at HEAD `3f742ca`, the commit that holds the registration
(2026-11-10, below), from 01:00 to 06:03 PST on Nov 11, on the charger and out of Low Power Mode. External
development matches only. Nothing in the registration was changed. Tables: `work/final/night_s12.log`; verdicts:
`s1_gate.json`, `s2_report_8.json`, `s2_report_28.json` beside it, all with `registered_run: true`.

**Checks the run printed.** S1's alive head equals `T1200`'s cached probabilities byte for byte on 28 of 28
folds. S1's scores on the 8 seen folds equal the probe's on 32 of 32 values.

**S1 gate: weighted LightGBM team head against `T1200`, both blended with `LT`, 28 folds.**

| Comparison | Folds | S1 | `T1200` | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6233 | 0.6180 | +0.0053 | [+0.0004, +0.0101] | 17/28 |
| `severe`, all (rule: not worse) | 28 | 0.5914 | 0.5840 | +0.0074 | [+0.0027, +0.0120] | 18/28 |
| `target`, SkillCorner (rule: not worse; unseen) | 20 | 0.6075 | 0.6103 | -0.0028 | [-0.0088, +0.0029] | 9/20 |
| `severe`, SkillCorner (unseen) | 20 | 0.5826 | 0.5816 | +0.0010 | [-0.0045, +0.0064] | 12/20 |
| `target`, IDSSE + Metrica (seen in the probe) | 8 | 0.6628 | 0.6374 | +0.0254 | [+0.0162, +0.0353] | 8/8 |
| `severe`, IDSSE + Metrica (seen in the probe) | 8 | 0.6133 | 0.5900 | +0.0233 | [+0.0146, +0.0323] | 6/8 |

S1 fails on the win count: 17 of 28 at `target`, the rule needs 20. It is not worse at `severe` and not worse on
SkillCorner by the rule's meaning (the interval reaches 0), so the two guard conditions hold. The 20 unseen folds
give 9 wins where 12 were needed, and a mean below the incumbent's. The mean gain on 28 folds is the 8 seen
folds. Information only: the member alone is worse on 28 folds (-0.0116 at `target`, 7/28) and loses every
SkillCorner fold (-0.0317, 0/20), while it gains +0.0388 on IDSSE + Metrica (7/8).

**S2 screen: weighted team loss in the TCN against `LT`, both blended with `T1200` (S1 did not pass), 8 IDSSE +
Metrica folds.**

| Cond | S2 | `LT` | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|
| `target` | 0.6752 | 0.6374 | +0.0378 | [+0.0261, +0.0495] | 8/8 |
| `severe` | 0.6318 | 0.5900 | +0.0418 | [+0.0308, +0.0532] | 7/8 |

The screen passes, so S2 went to the gate, as registered.

**S2 gate: the same comparison on 28 folds.**

| Comparison | Folds | S2 | `LT` | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6288 | 0.6180 | +0.0108 | [+0.0051, +0.0165] | 17/28 |
| `severe`, all (rule: not worse) | 28 | 0.5907 | 0.5840 | +0.0067 | [+0.0012, +0.0121] | 11/28 |
| `target`, SkillCorner (information) | 20 | 0.6102 | 0.6103 | -0.0000 | [-0.0069, +0.0069] | 9/20 |
| `severe`, SkillCorner (information) | 20 | 0.5742 | 0.5816 | -0.0074 | [-0.0137, -0.0014] | 4/20 |

S2 fails on the win count too: 17 of 28 at `target`. On the SkillCorner folds it does nothing at `target` and is
worse at `severe` (the interval is below 0, 4 of 20 folds won). Information only: the TCN alone is worse on 28
folds (-0.0196 at `target`, 8/28). Against F the blend is +0.0269 at `target` (21/28), which is mostly `T1200`
and `LT`'s own gain and was not a registered comparison.

**The two together are worse than either.** S2 with the weighted LightGBM member against S1 on 28 folds: -0.0140
at `target` (6/28), -0.0202 at `severe` (7/28). On the 8 IDSSE + Metrica folds the pair is still ahead of S1
(+0.0199 at `target`, 6/8). The pair was never a candidate once S1 failed; this is the number, for the record.

**Predicted possessions per true one, at `target`.**

| Folds | F | `T1200` + `LT` | S1 + `LT` | `T1200` + S2 |
|---|---|---|---|---|
| All 28 | 0.738 | 0.751 | 0.850 | 0.897 |
| SkillCorner (20) | | 0.760 | 0.869 | |
| IDSSE + Metrica (8) | 0.685 | 0.729 | 0.807 | 0.872 |

**What this says.** The weight does what it was built to do: the model commits to more short possessions (0.75
predicted per true one rises to 0.85 and 0.90). On the 8 IDSSE + Metrica folds that pays in both members, at both
conditions. On the 20 SkillCorner folds it pays nothing at `target` and costs at `severe`. The split is by
dataset, and it is the same split the probe of Nov 10 showed for SkillCorner's place in the team head.

**What this does not say.** Which kind the provided match is. The two groups differ in two ways at once: the
labels (IDSSE and Metrica carry a dead label and come from corrected tracking, as the provided match's ground
truth is described in `docs/upstream/CHALLENGE.md`; SkillCorner has no dead label and its possession comes with
its automatic tracking) and the input (the provided match's tracking is automatic broadcast tracking, SkillCorner's kind). No
development match has the first kind of label on the second kind of input, so no fold separates the two
readings. The public 5 minutes cannot be used to choose between them.

**Closed, as registered.** "If S1 wins on the IDSSE + Metrica folds and fails on 28, that is the result: it is
not re-gated on 8 folds." The same holds for S2. No other weight form, clip or exponent is tried, and no rule is
re-read after the fact. Length weighting is not in F3 and is not a candidate. These were the 9th and 10th
packages gated on the same 28 folds.

**Run times on the Mac.** S1: 31 min for 28 LightGBM folds. S2 screen: 101 min (about 760 s a fold). S2 gate:
169 min for the 20 remaining folds (about 500 s a fold).

### F3 on the Windows PC: the reproduction checks (result)

**R0 to R5 pass.** R5 on Justin's Mac is the entry above. External matches only; nothing ran on the provided
match. Code at `c97d4eb` and later on `david/f3-assembly`; logs in
`D:/katena/logs/f3`. Written while the 28-fold gate was still running; its result is the entry above.

| # | Check | Result |
|---|---|---|
| R0 | F after the code change | Pass. `train` gives gbdt `b2146150...`, tcn `01488c84...`, bundle `cb0ea4fc...`, as logged on Nov 8 |
| R1 | Same machine, fresh processes | Pass. F3 trained twice on all 30: the same bundle both times, and a byte-identical dressed CSV |
| R2 | Dressed match, held-out model, both paths | Pass. Features and all six probability arrays bitwise equal, 0 of 29,123 work-grid labels differ, 0 CSV frames differ |
| R3 | Kernel swap on this PC | Pass. The held-out TCN's hash changes (`fdc7797b...` to `b3185586...`); the dressed CSV is byte-identical (`4db583ae...`), 0 of 145,608 graded frames |
| R4 | Other machines (`repro.yml`, `spec=F3`, runs 43015046674 and 43024430520) | Pass on six runners: the dressed CSV equals this PC's on each |
| R5 | LightGBM across machines | Pass. The 1,200-tree member's hash is the same on this PC, on every runner and on Justin's Mac (Apple M5, entry above) |

**F3's hashes on this PC** (all 30 matches, 657,133 rows x 213 columns):

| | sha256 |
|---|---|
| gbdt | `056ffa8a41eaf246b741234351e30dcd8e5191446493e24e8c1ea3f2e6a5bc47` |
| tcn | `4ec1a79e8f1303e1fa890879ffa34c983344c57fb3335139e49a35741f719bb3` |
| bundle.pkl | `0a6359f3411082e32a9b1e97385eb430abc3eaf96198df6306a934b36e982cd9` |
| dressed CSV (`idsse/DFL-MAT-J03WMX`, all-30 model) | `72d7c3fb09f854bf06872d0996201ca338b247d27335a1829363379bd2cb9a7f` |

**R4 by runner.** As with F, the TCN's bytes follow the OS and the CPU vendor, and no label does: four different
sets of TCN weights on six OS and CPU combinations, one CSV.

| Machine | CPU | gbdt | tcn | Dressed CSV | `train` |
|---|---|---|---|---|---|
| This PC, Windows 10 | Intel i7-7700 | reference | reference (`4ec1a79e...`) | reference | 23 min beside another job |
| `windows-2025`, pass 1 | Intel Xeon Platinum 8573C | same | same | same | 12.9 min |
| `windows-2025`, pass 2 | AMD EPYC 7763 | same | differs (`50b82c14...`) | same | 13.9 min |
| `ubuntu-24.04`, pass 1 | AMD EPYC 9V74 | same | differs (`12687343...`) | same | 8.4 min, 5.7 GB peak |
| `ubuntu-24.04`, pass 2 | AMD EPYC 7763 | same | as pass 1 | same | 9.8 min |
| `macos-15`, both passes | Apple M1 | same | differs (`55cc7100...`) | same | 36.5 and 37.8 min |

Not covered: Linux on an Intel CPU, which F's check did cover (two more Ubuntu passes both drew an AMD EPYC 7763 and gave the same CSV), and
the dressed match labelled with the Mac's own TCN, a fifth set of weights
(`dressed_check.py label --bundle <the Mac's model_F3/bundle.pkl>`; the CSV should hash to `72d7c3fb...`). `docs/repro/windows-reference-F3.json` is this
PC's reference for `repro.yml`. Pass 2 ran before that file was committed, so its own summary compares with F's
reference; the rows above compare its reports with F3's.

**Also checked.**

- The pipeline's TCN code (`katena.eval.g1` at 4,800 steps) against `run_g12.py`'s `LT` code, before any score:
  on 5 matches at 20 steps the sequences, the scaler, the trained weights and the probabilities are identical.
  `tests/test_pipeline.py` repeats this on the synthetic corpus.
- The gate's LightGBM stage is done: on all 28 folds the 1,200-tree member's first 150 trees give F's cached
  member byte for byte.

**One fold score was seen in R2.** `dressed_check.py compare` prints the held-out match's score at clean input:
F3 0.7221 on both paths (F: 0.6506 on Nov 8). It is one fold at a condition the gate does not use, it was part
of the registered check, and it decides nothing.

**Cost.** `train` for F3 on this PC: LightGBM 2.6 to 2.8 min, TCN 20 to 21 min with a second heavy job running
(F: 0.6 and 3.7 min alone). The organisers' one command grows by about that much.

**Scheduling.** From 14:27 a second process shares the gate's TCN stage (`run_f3.py tcn --reverse`): each
process claims a fold before training it. A fold's result does not depend on which process trains it; a test
checks that against `g1.stage_tcn`.

### F3 on the Windows PC (pre-registration): the second-machine gate and the reproduction checks

**Registered before any Windows score of `T1200`, `LT` or F3, and before F3's final model is trained.** David +
Claude, branch `david/f3-assembly`, the Windows PC (i7-7700, the machine that holds the provided match). External
matches only: no `data/`, no `evaluation/gt/`, no public window, no locked bench. Nothing below changes after a
result.

**What F3 is.** F with the two changes that passed on Justin's branch (entries of 2026-11-09 and 2026-11-10,
PR #23), and nothing else:

| Member | F | F3 |
|---|---|---|
| LightGBM | 150 trees | 1,200 trees (`T1200`) |
| TCN | 1,200 steps, M2's 160 columns | 4,800 steps, the 160 plus D1's 53 touch columns (`LT`) |

Unchanged: `gbdt.PARAMS`, the `d2j` pool and sequences, seed 0, 4 torch threads, the blend at w = 0.5, Viterbi at
tau 2, the priors, T = 2. S1 and S2 are not part of F3. If either passes on the Mac it gets its own entry.

**Standing.** The final-week plan fixed its own two candidates, and both failed. `T1200` and `LT` were registered
on Nov 9, before the plan was written, and passed F's rule on the Mac. Whether they count under the plan is
David's and Justin's decision and is not made here. This entry adopts nothing. It fixes what the Windows PC runs
and how each outcome is read, so that the decision can be acted on before the plan's cut-off (Nov 14, 18:00 CET).

**What was seen before this entry.** Everything in PR #23: the Mac's gates (`T1200` +0.0041, `LT` +0.0120, F3's
means 0.6180 against 0.6019 at `target`), the probes, the failure diagnosis. On Windows: F's gate, and the plan's
G1 (the touch columns in a 1,200-step TCN, +0.0017, failed). No Windows fold of a 1,200-tree member or of a
4,800-step TCN.

**1. The second-machine gate** (`scripts/run_f3.py`, cache `D:/katena/work-f3`, F's inputs copied from
`work-final` with SHA-256 checks). The 28 development folds, leave one match out, F3 against F.

- F3's LightGBM member: `gbdt.fit_heads(..., n_trees=1200)` on F's pool and 213 columns. Check printed: its first
  150 trees give F's cached member byte for byte on every fold.
- F3's TCN member: `katena.eval.g1`'s sequences, scaler and predict (the plan's G1 code) at 4,800 steps. Check
  run first, with no score: on one fold at 20 steps it gives the same model hash as `run_g12.py`'s `LT` code.
- F is not refitted: its stored Windows scores are the incumbent, re-decoded from the copied probabilities and
  compared with the store.
- **Rule** (F's, plus the condition Justin registered for `T1200`): at `target` at least 20 of 28 folds won and
  the paired 95% bootstrap interval above 0; not worse at `severe`; not worse at `target` on the 20 SkillCorner
  folds alone.
- These folds are not unseen. The Mac scored the same models on them, up to the TCN's last bits. This is a
  replication on a second machine, not a new package and not new evidence of the same weight as an unseen fold.
- Reported beside it, deciding nothing: each change alone (`T1200` with F's TCN, `LT` with F's trees), each
  member alone, per dataset, every evaluation world, macroF1, predicted possessions per true one, and the
  difference between the Windows and the Mac means.

**2. The reproduction checks** (pass or fail, they choose nothing). F3 in `katena.pipeline` behind a `--spec F3`
switch; F stays the default and its hashes must not move.

| # | Check | Passes when |
|---|---|---|
| R0 | F after the code change | `train` gives F's logged hashes (gbdt `b2146150...`, tcn `01488c84...`, bundle `cb0ea4fc...`) |
| R1 | Same machine, fresh processes | F3 trained twice on all 30 matches gives the same `bundle.pkl`; the dressed CSV is byte-identical |
| R2 | Dressed match (`idsse/DFL-MAT-J03WMX`, held out) | The submission path equals the evaluation path: features and probabilities bitwise, labels equal |
| R3 | Kernel swap on Windows | The held-out TCN retrained with `ATEN_CPU_CAPABILITY=default`, `ONEDNN_MAX_CPU_ISA=SSE41`, `MKL_CBWR=COMPATIBLE` gives a byte-identical dressed CSV |
| R4 | Other machines (`repro.yml`, dispatched with the F3 spec) | The dressed CSV on every runner equals the Windows one |
| R5 | LightGBM across machines | The 1,200-tree member's hash on Windows equals the Mac's (needs Justin's hash) |

**How the outcomes are read, fixed now.**

- F3 is put forward for adoption only if the Windows gate passes its rule **and** R0 to R4 pass. The model that
  is submitted is trained on this PC, so a pass on the Mac alone is not enough.
- If the gate passes and R3 or R4 shows a label that differs between machines, F3 is not put forward as it
  stands. F labelled the dressed match identically on 7 OS/CPU combinations, and byte-identical regeneration is a
  competition rule. That result is reported, and F stays unless the team finds a fix that needs no new score.
- If the gate fails on Windows, that is the result. F stays, and nothing is re-run to change it.
- R5 failing does not block by itself (the CSV is what must agree), but it is reported and looked into.
- No other tree count, step count, input set, blend weight or decoder setting is tried here.

**Cost.** Gate: LightGBM about 2 hours, TCN about 8 hours on one `heavy.py` slot. R0 to R3: about 2 hours on the
other slot. R4 runs on GitHub's runners.

### PR #23's asks: where each stands (David's side, Claude; David has not confirmed any of it)

Nothing here is a decision. Items 1, 3 and 5 are David's to answer.

1. **Is `T1200` + `LT` a candidate under the plan?** Not answered yet. The entry above prepares it: the Windows
   gate and the reproduction checks run today, so a yes can be acted on before Nov 14, 18:00 CET. Claude's
   recommendation is yes, on the condition written there (the Windows gate and R0 to R4 all pass): both changes
   were registered before their scores and before the plan existed, and passed the plan's own rule.
2. **The real run with F stays David's.** Unchanged. `scripts/real_run.py` still trains and checks F: the spec it
   runs is the one named in `docs/repro/expected-windows.json`.
3. **No SkillCorner match in the team head.** Recommendation: leave it out of the submission. Its only evidence is
   the 8 folds that were already seen, in a probe that looked at five variants, and no unseen fold of that kind
   is left to gate it on. The design note under "Final-week gates: outcome" was about registering the affected
   folds before looking, which is no longer possible here. It also removes the only training matches whose input
   is automatic broadcast tracking, which is the provided match's kind of input.
4. **PFF World Cup 2022.** Recommendation: do not start it. A web search (nothing downloaded, no form filled)
   found no licence text for the dataset. The release post sets no conditions and asks only for a mention. The
   only formal text is the provider's website terms, which forbid using "the Website or its contents for
   competitive or benchmarking purposes" (gradientsports.com/terms-conditions); whether they cover the dataset is
   unclear. Access is a sign-up form that reveals a shared folder, reported at about 1.4 GB of JSON per match,
   so roughly 90 GB for 64 matches. Without a written permission the licence check the organisers ask for
   cannot be passed, and three days are left.
5. **Team name.** Unknown on this side. David's registration email (Nov 3) and the challenge email (Nov 5) carry
   no team name. The Nov 3 clarification is not in David's inbox (he registered that afternoon), so its
   wording was taken from Justin's entry of Nov 10, not read again here.

## 2026-11-10

### Naming: two pairs of gates are called G1 and G2 (read this first)

Two branches named candidates independently. In the entries headed **"Justin's branch"** below, `G1` is a
1,200-tree LightGBM member and `G2` a 4,800-step TCN: registered on Nov 9 (commit `11c11fa`), run in the night
of Nov 9 to 10, both passed. In the final-week plan and the entries headed **"Gate G1"** and **"Gate G2"**, written
on Nov 10 without sight of that branch, `G1` is the TCN reading the touch columns at 1,200 steps and `G2` is
set-piece tag features: both failed. Neither side had seen the other's work: this branch was not pushed until the
evening of Nov 10. From here on Justin's two are called **`T1200`** (the trees) and **`LT`** (the TCN: 4,800
steps, touch columns). File and cache names (`run_g12.py`, `g1_gbdt1200`, `g2_tcn4800t`) keep the old letters.

### S1 and S2 (pre-registration): possession-length weights in both members

**Registered before any SkillCorner fold of either candidate and before any fold of S2.** Justin + Claude, branch
`claude/git-pull-analyze-strategy-e4504e`, the Mac (macOS arm64). External development matches only: no `data/`,
no `evaluation/gt/`, no public window, no locked bench. Code: `scripts/run_s12.py`, run unattended by
`scripts/night_s12.sh`, which refuses to start unless this entry and the code are committed. Nothing below
changes after a result.

**Standing.** This is outside the final-week plan, which fixed its own two candidates and adopts nothing after
Nov 14, 18:00 CET. David has not seen or agreed to it. A pass here makes a candidate for the team to decide on;
it adopts nothing. These are the 9th and 10th packages gated on the same 28 folds (the plan's two were the 5th
and 6th, `T1200` and `LT` the 7th and 8th).

**Why.** The diagnosis below: on the IDSSE + Metrica folds possessions under 4 s are 36% of the true ones and
65% of the misses, and no training weight depends on possession length, while spellF1 counts every possession
once.

**What was seen before this entry.** The exploratory probe below: `inv` in the LightGBM member on the 8 IDSSE +
Metrica folds (blended +0.0254 at `target`, 8/8; +0.0233 at `severe`, 6/8), and `sqrt`, which was smaller and is
dropped. One smoke fold of S1 (`idsse/DFL-MAT-J03WMX`, one of those 8; it reproduced the probe's scores). One
smoke of S2's code on that fold at 20 steps instead of 4,800, which is not the candidate. No SkillCorner fold of
a weighted model, and no fold of S2.

**Incumbent.** The model that passed the night run of Nov 9: the 1,200-tree LightGBM member (`T1200`) blended at
w = 0.5 with the 4,800-step TCN that reads the touch columns (`LT`). Read from that run's cache, not refitted.

**The weight** (`katena.models.lenweight`, fixed before the probe ran). A training frame weighs 1 / L, L the
length in seconds of the constant-label run it sits in, in the T = 2 training labels, clipped to [1, 30]. No
other form, clip, exponent or label convention is tried.

| | Changes | Everything else |
|---|---|---|
| **S1** | The LightGBM team head's rows carry the weight, scaled to mean 1 (`gbdt.fit_heads(team_weight=...)`). | As `T1200`: pool, 213 columns, parameters, 1,200 trees. The alive head is unweighted. |
| **S2** | The TCN's team loss carries the weight per frame (`tcn.train` reads `w`; the loss already divides by the weight sum). | As `LT`: 4,800 steps, 213 columns, seed 0, 4 threads, the `d2j` sequences, the scaler. The alive loss is unweighted. |

Decoder (Viterbi, tau 2), priors, blend weight 0.5 and T = 2 are F's and unchanged.

**S1, gate.** All 28 folds, S1 blended with `LT` against `T1200` blended with `LT`. F's rule: at `target` at
least 70% of folds won and the paired 95% bootstrap interval above 0; not worse at `severe`. And, as for `T1200`:
not worse at `target` on the 20 SkillCorner folds alone. Eight of the 28 folds were seen (8/8 at `target`), so
the rule needs 12 of the 20 unseen SkillCorner folds. The SkillCorner rows of the table are the only unseen
evidence and are reported as such.

**S2, screen.** The 8 IDSSE + Metrica folds, S2 against `LT`, both blended with the LightGBM member that stands
after S1's gate (weighted if S1 passed, `T1200` if not). Passes if at `target` it wins at least 6 of 8 folds with
the interval above 0, and it is not worse at `severe`. **S2, gate:** only if the screen passes, the same
comparison on all 28 folds under F's rule.

**Fixed in advance.** Each of S1 and S2 that passes its gate is a candidate for adoption, S1 + S2 together if
both pass (S2's gate is then already the test of the pair). If S1 wins on the IDSSE + Metrica folds and fails on
28, that is the result: it is not re-gated on 8 folds. The design note under "Final-week gates: outcome" is about
changes that cannot reach SkillCorner folds; this one can.

**Checks the run prints.** S1's alive head equals `T1200`'s cached probabilities byte for byte on every fold
(same pool, columns, parameters and features). S1's scores on the 8 seen folds equal the probe's.

**Cost.** S1 about 40 minutes, S2's screen about 65, S2's gate about 155.

### Justin's branch: G1 and G2: result (Justin + Claude)

**G1 passes. G2's screen passes for both variants, `LT` goes to the gate and passes it.** One unattended run of
`scripts/night_g12.sh` on the Mac at HEAD `11c11fa`, the commit that holds the registration below, from 01:30 to
07:54 on Nov 10, on the charger and out of Low Power Mode. External development matches only. Nothing in the
registration was changed. Tables: `work/final/night_g12.log`; verdicts: `g1_gate.json`, `g2_report_8.json`,
`g2_report_28.json` beside it, all with `registered_run: true`.

**The incumbent is F.** The 150-tree read and the TCN trained here are byte-identical to the F2 screen's cache on
all 8 folds that cache holds. F at `target` on 28 folds: 0.6019 here, 0.6019 on the Windows PC.

**G1 gate: F with 1,200 trees against F, 28 folds.**

| Comparison | Folds | G1 | F | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target`, all (rule) | 28 | 0.6060 | 0.6019 | +0.0041 | [+0.0015, +0.0070] | 21/28 |
| `severe`, all (rule: not worse) | 28 | 0.5738 | 0.5696 | +0.0042 | [+0.0010, +0.0074] | 17/28 |
| `target`, SkillCorner (rule: not worse) | 20 | 0.6046 | 0.6030 | +0.0016 | [-0.0018, +0.0050] | 14/20 |
| `severe`, SkillCorner | 20 | 0.5789 | 0.5804 | -0.0014 | [-0.0054, +0.0024] | 9/20 |
| `target`, IDSSE + Metrica | 8 | 0.6095 | 0.5991 | +0.0104 | [+0.0049, +0.0159] | 7/8 |
| `severe`, IDSSE + Metrica | 8 | 0.5610 | 0.5426 | +0.0185 | [+0.0126, +0.0247] | 8/8 |

All three registered conditions hold, so G1 passes. Read it with care: the gain sits in the 8 folds that chose
the size. On the 20 SkillCorner folds, the only unseen ones, the difference at `target` is +0.0016 with an interval
through 0, and at `severe` the mean is slightly below F's. G1 is not shown to help there; it is shown not to hurt.
The member alone (information): +0.0067 at `target` on 28 folds, 17/28 wins, which is not an adoption by itself.

**G2 screen: TCN at 4,800 steps against F's TCN, both blended with the 1,200-tree member, 8 IDSSE + Metrica folds.**

| Variant | Cond | Candidate | Incumbent | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `L` | `target` | 0.6297 | 0.6095 | +0.0202 | [+0.0129, +0.0272] | 8/8 |
| `L` | `severe` | 0.5895 | 0.5610 | +0.0285 | [+0.0212, +0.0363] | 8/8 |
| `LT` | `target` | 0.6374 | 0.6095 | +0.0280 | [+0.0194, +0.0360] | 8/8 |
| `LT` | `severe` | 0.5900 | 0.5610 | +0.0289 | [+0.0204, +0.0380] | 8/8 |

Both pass. `LT` has the higher mean at `target`, so only `LT` went to the gate, as registered. `L` has no
SkillCorner fold and no 28-fold score.

**G2 gate: `LT` against F's TCN, both blended with the 1,200-tree member, 28 folds.**

| Cond | Folds | `LT` + 1,200 | F's TCN + 1,200 | Diff | 95% paired CI | Wins |
|---|---|---|---|---|---|---|
| `target` (rule) | 28 | 0.6180 | 0.6060 | +0.0120 | [+0.0082, +0.0160] | 21/28 |
| `severe` (rule: not worse) | 28 | 0.5840 | 0.5738 | +0.0102 | [+0.0061, +0.0142] | 20/28 |

`LT` passes the gate. The run did not print the SkillCorner folds separately for G2; from the two tables, the mean
difference at `target` on the 20 SkillCorner folds is about +0.0056 (no interval computed), a fifth of the 8-fold
one. Information only: with the 150-tree member `LT` is +0.0114 at `target` (21/28); the TCN alone is +0.0244
(24/28); at `target_ids` the blend is +0.0090 with 15/28 wins; `mild`, `target_smooth` and both coupled worlds are
above 0 with at least 75% of folds. Predicted possessions per true one at `target`: 0.749 to 0.751, so the gain is
not from cutting more spells.

**Both together, means only (no paired interval was registered for this pair).** F to `LT` + 1,200 trees on the
28 folds: 0.6019 to 0.6180 at `target`, 0.5696 to 0.5840 at `severe`. By group at `target`: 0.5991 to 0.6374 on
IDSSE + Metrica, 0.6030 to about 0.6102 on SkillCorner.

**Run times on the Mac.** G1: 91 min (LightGBM 21, TCN 71). G2 screen: 136 min. G2 gate: 155 min. An `LT` fold
is about 465 s, so the final model's TCN costs about four times F's.

**What this changes.** No submission and no final model yet. Still open before either result is adopted, all from
the registration: `LT` repeats D3's kernel-swap check on the dressed match; F3's assembly gets its own registration
before the final model is trained; the final model's LightGBM hash must agree between the Mac and the Windows PC.
No other tree count, step count or input set is tried.

### Where the gated model loses: `scripts/inspect_fold.py` (description only, selects nothing)

Run after the gate, on the cached out-of-fold probabilities of the 28 external folds, decoded as the gate decodes
them. Its per-group spellF1 equals the gate's (0.6374 and 0.6103 for `LT` + 1,200 trees at `target`).

| `LT` + 1,200 trees, `target` | IDSSE + Metrica (8) | SkillCorner (20) |
|---|---|---|
| True possessions found at IoU 0.5 | 55% | 55% |
| Found, by true length: under 2 s / 2-4 s / 4-8 s / 8-16 s / over 16 s | 7% / 30% / 58% / 79% / 91% | 24% / 27% / 51% / 68% / 85% |
| Share of the misses that are under 4 s | 65% | 52% |
| Misses given to the other team / right team but too long / called dead | 47% / 41% / 10% | 61% / 33% / 4% |
| Predicted possessions per true one | 0.73 | 0.76 |
| F1 at IoU 0.3 / 0.5 / 0.7 | 0.749 / 0.645 / 0.506 | 0.714 / 0.622 / 0.484 |
| spellF1 with the true team on the frames predicted alive | 0.905 | 0.990 (no dead-ball label, so trivial) |
| spellF1 with the true alive/dead pattern | 0.661 | 0.610 |

The same picture as F in the strategy review: short hand-overs are missed, the neighbouring possession then runs
too long, and dead ball is not where the points are. No training weight depends on possession length anywhere in
`gbdt.py` or `tcn.py`, so a possession of 2 s weighs a thirtieth of one of 60 s in training and the same in the
metric. That is a candidate for a registered screen, not a result.

`inspect_fold.py fold <match>` writes a timeline page for one held-out external match under `runs/`.
`inspect_fold.py public <submission>` writes the same page for the public window under `artifacts/`, for a person
only, and prints no score. **Public window:** that mode was tested once, on a synthetic submission built from the
public labels themselves (shifted, two invented errors). No prediction of ours was compared with the window.

### The organisers' rule clarification of Nov 3, read on Nov 10 (Justin + Claude)

The email "Clarification of the rules for the Initial Challenge" (hackathon.talentarena@mobileworldcapital.com,
2026-11-03, sent to all participants) was in Justin's inbox, unread. It says its rules take precedence over the
documentation, earlier emails and individual answers. In our words:

- **Not allowed:** checking your output against the video; hand labels or any manually created data on the
  provided game; uploading the provided data to AI services; training, fitting or calibrating on the ground-truth
  possession data or on the provided video.
- **Allowed:** automatic checks that use only the provided data; inspecting the provided data and the video to
  understand what counts as possession; using the video, the tracking, the Smart Tagging or other generated data
  as input, as long as the algorithm was not trained on the ground truth or the provided video; files of
  intermediate steps; any dataset we have the rights to use, licence checked.
- **Training and tuning:** training must use only external data, and parameters must be tuned on external data,
  never on the ground truth or the provided video.

What it changes here:

- **Nobody watches the video beside our predictions.** `inspect_fold.py public` stays a check against the public
  labels only, and its page now says so.
- **Pseudo-labels on the provided match are out.** `CLAUDE.md` still says algorithmic pseudo-labels are allowed;
  "training must use only external data" overrides that. We had none planned.
- **G3 cannot be a person reading a statistic of the match and picking a training mix.** That is tuning on
  provided data. A rule inside the algorithm is different: both models trained on external data, a ball-jitter
  statistic of the input computed by the code, and a threshold fixed on external worlds. The tracking is an
  allowed input. This reading is ours and has not been put to the organisers.
- **Smart Tagging is an allowed input** (P5, the second-half kick-off, uses one tag).
- **External data is explicitly welcome**, licence permitting.

Also from the organisers' emails: every member must have registered under the same team name before Nov 10, and
the landing page showed "registration closed" when opened on Nov 10 at about 20:30 CET. Submission instructions
arrive by Nov 14; the deadline is Nov 17, 23:59 CET.

### Two exploratory probes of the LightGBM member: `scripts/probe_levers.py` (written before the run)

Exploratory, like the tree screen of Nov 9: not registered, 8 IDSSE + Metrica folds only, nothing adopted from
it. The 20 SkillCorner folds are not scored, so they stay unseen for a registered gate. The unchanged model is
read from the gate's cache. Fixed before the first run:

- **`curve`: would more training matches help?** The member refitted with every second SkillCorner match dropped
  (7 + 10), with none (7 + 0), and with every second match of the held-out match's own kind dropped (4 + 20).
  Read at 150 and at 1,200 trees. The bake-off's "four more matches are worth 0.003" was measured with a
  150-tree member, which G1 showed was too small to use what it had.
- **`weights`: possession-length weights on the team head.** Each training row weighs 1 / L (`inv`) or
  1 / sqrt(L) (`sqrt`), L the length in seconds of its possession in the T = 2 labels, clipped to [1, 30],
  scaled to mean 1. The alive head, the pool, the decoder and the priors are unchanged. No third form is tried.

**Result** (run once, 40 fits, 42 minutes; pool rows agree with the native labels on 1.0000 of rows). Variant
minus the unchanged model at 1,200 trees, spellF1, 8 folds, 95% paired CI, folds won. "Blended" is with the
4,800-step touch TCN at w = 0.5, and that TCN is unchanged: it still trains on all 27 matches, unweighted.

| Variant | Member alone, `target` | Blended, `target` | Blended, `severe` | Predicted per true, blended `target` |
|---|---|---|---|---|
| unchanged | 0.6042 | 0.6374 | 0.5900 | 0.729 |
| `7 + 10` | +0.0220 [+0.0134, +0.0303], 8/8 | +0.0128 [+0.0064, +0.0193], 8/8 | +0.0066 [-0.0004, +0.0137], 5/8 | 0.750 |
| `7 + 0` | +0.0553 [+0.0422, +0.0685], 8/8 | +0.0346 [+0.0238, +0.0459], 8/8 | +0.0316 [+0.0222, +0.0415], 8/8 | 0.824 |
| `4 + 20` | -0.0167 [-0.0250, -0.0083], 0/8 | -0.0082 [-0.0138, -0.0023], 1/8 | -0.0103 [-0.0159, -0.0051], 0/8 | 0.725 |
| `inv` | +0.0388 [+0.0259, +0.0519], 7/8 | +0.0254 [+0.0162, +0.0353], 8/8 | +0.0233 [+0.0146, +0.0323], 6/8 | 0.807 |
| `sqrt` | +0.0253 [+0.0161, +0.0344], 7/8 | +0.0145 [+0.0077, +0.0218], 8/8 | +0.0128 [+0.0059, +0.0198], 6/8 | 0.764 |

At 150 trees every sign is the same and the sizes are close (`7 + 0` blended +0.0337, `inv` blended +0.0293).

What it says, on these 8 folds only:

- **The SkillCorner matches cost the IDSSE and Metrica folds.** Fewer of them is better and none is best:
  0.6374 to 0.6721 blended with only the LightGBM member changed. SkillCorner has no dead-ball label, so in this
  member it reaches only the team head, where it is 20 of 27 matches. Its possession field comes from automatic
  broadcast tracking, the other eight from corrected tracking or events.
- **Matches of the held-out match's own kind are worth a lot.** Dropping 3 or 4 of the 7 costs 0.017 alone and
  0.008 blended, losing on every fold alone. This is the opposite of the bake-off's "four more matches are worth
  0.003", which mixed both kinds and used a 150-tree member.
- **Length weights help, and the full form more than the half form.** `inv` is +0.025 blended at both severities.
- **All three gains come with more predicted possessions**, 0.73 per true one rising to 0.75-0.82. They attack
  the same fault (short hand-overs merged away), so they will not add up. No combination was run.

What it does not say:

- Nothing about the 20 SkillCorner folds, which were not scored. Training without SkillCorner will presumably
  cost them, and they are 20 of the 28 folds in F's adoption rule.
- Nothing about the provided match. Which kind it resembles cannot be read from the public window. The public
  argument is `docs/upstream/CHALLENGE.md`: the ground truth "is derived from corrected tracking data" and has a
  dead-ball label, which is the IDSSE and Metrica kind of label; its input tracking is automatic, which is the
  SkillCorner kind of input.
- Five variants were looked at on eight folds, unregistered. Nothing is adopted from this entry.

Open, for Justin and David:

1. A registered 28-fold gate of `inv` on the LightGBM team head under F's rule. Cost: about 30 minutes of fits.
2. Whether a gate may judge "no SkillCorner in the team head" on the 8 dead-label folds as primary. Under the
   28-fold rule it cannot be judged fairly. This changes the protocol, so it needs both names and a registration
   written before any SkillCorner fold of it is scored.
3. Whether more matches of the corrected-tracking kind can be had in time (see the dataset search below).

### Dataset search, Nov 10 (web only, nothing downloaded)

- **PFF FC (Gradient Sports) World Cup 2022:** 64 matches, broadcast tracking with the ball, possession and
  ball-out carried as events embedded in the frames (kloppy has a loader; fouls and offside are not marked dead
  by it). Access through a form. No licence text found, so competition use is unverified: a person must read the
  terms before anything is downloaded.
- **SkillCorner's nine 2019/20 matches** (in the history of `SkillCorner/opendata`): same kind as the 20 we have,
  licence of that old release unclear. After the result above, not wanted.
- **Simulators** (Google Research Football, RoboCup 2D): exact labels, unrealistic motion. Not pursued.
- Checked and unusable: event-only releases (StatsBomb, Wyscout, Impect), clip datasets (SoccerNet-GAR),
  SoccerTrack v2 (events only, gated), Hudl's broadcast tracking (proposal-gated, closed), GPS sets without a
  ball. No Sportec release beyond the seven matches we have.

### J3: fresh-clone rebuild on Windows (W1)

External matches only. Nothing ran on the real match. This is the same PC as the J2/J3 entry of 2026-11-08: Windows
10, i7-7700, Python 3.11.17, torch 2.15.0+cpu, LightGBM 4.7.0, numpy 2.2.6.

**Setup.**
- `git clone` into `D:/katena/fresh/repo`, then a checkout of `origin/main` at `28450b4`. The global
  `core.autocrlf=true` is overridden by `.gitattributes`: every text file is LF in the working tree.
- The venv: `uv venv --python <uv's 3.11.17>`, then `uv sync --frozen`. The package list matches the main checkout's
  venv.
- Then the "Without make" steps of `docs/REPRODUCE.md`, with `KATENA_EXTERNAL` and `KATENA_WORK` pointing at a new
  empty corpus and an empty cache: `build_external.py` (a fresh download), then `corpus`, `worlds` and `train` with
  `--workers 2`, then `predict` on the dressed IDSSE bundle `D:/katena/dressed/J03WMX`, exactly like `A_full`.

**Result: identical to `D:/katena/work-submit` at every stage. No stage differs.**

| what | files | fresh clone vs `work-submit` / `D:/katena/external_data` |
|---|---|---|
| raw downloads (IDSSE, Metrica, SkillCorner; `.git` excluded) | 21 + 11 + 139 | all identical (SHA-256) |
| canonical matches (`canonical/*/*.npz`) | 30 | 30 identical |
| gt, pool | 30, 30 | all identical |
| feat, touch | 381, 330 | all identical |
| pool_d2j, touch_pool_d2j | 30, 30 | all identical |
| gbdt | | `b2146150e006ead3b543f2ee424c84d9a12446ff551dd505f9480eb6a582ec0e`, same |
| tcn | | `01488c84721327147b3975886e3b758a13f049531146bc54985b88976e8f6eb3`, same |
| bundle.pkl | | `cb0ea4fc546014665888a69ae86e4ba83f79cf7ff10dacc051f54b00f105a85c`, same |
| model parts (both LightGBM strings, TCN state, scaler, Viterbi and HSMM priors, names) | 7 | all identical |
| dressed J03WMX CSV, model trained on all 30 | 1 | `909f4ffa6aac5705d5269af5b7072a424afe6ca81c02e565bf904e1147aeb788`, byte-identical to `A_full.csv` and `B_full.csv` |

- `scripts/cache_digests.py diff` names no differing stage. The fresh dump is
  `D:/katena/dressed/runs/digests_windows_fresh.json`. `work-submit` also holds the two `model_without_*` check
  models, which a plain rebuild does not make.
- **The corpus sidecars** differ from the old corpus only in `code_sha256`, as the J2/J3 entry explains. The fresh
  ones are reproducible:
  - A second `build_external.py --skip-download` reports all 30 matches up to date and rebuilds none (1.7 min).
  - The clone's working tree stays clean, so `scripts/external_manifest.json` is not rewritten.
- **The dressed bundle re-exported** from the fresh corpus with the fresh code (`dressed_check.py export`) is
  byte-identical to the one the reference labelled: the raw ATD, the metadata XML, `timeline.json` and `gt_T2.csv`.
- **`origin/main` moved during the run,** to `c18d76e` (PRs #16-#18).
  - Since `28450b4`, none of these changed: `uv.lock`, the loader code that `build_external.py` hashes, or the
    corpus, worlds and train stages.
  - `predict` gained the kickoff rule.
  - `train` was re-run at `c18d76e` on the fresh cache: the same three hashes.
  - `predict` at `c18d76e` writes the same CSV, `909f4ffa...`. Its `kickoff_rule` reports `no_tagging_file` for both
    halves, so the rule changes nothing on a dressed bundle.

**Timings on this PC.** Each heavy step ran through `heavy.py`, beside the other agents' gate runs.

| step | fresh clone | 2026-11-08 (`work-submit`) |
|---|---|---|
| `uv sync --frozen` | 70 s | |
| `build_external.py` | 13.1 min: download 6.2 (IDSSE 50 s, Metrica 6 s, SkillCorner 316 s), building the 30 matches 4.9, the rest SHA-256 checks | |
| `corpus --workers 2` | 17.6 min | 18.0 min |
| `worlds --workers 2` | 38.8 min, after 14.4 min waiting for a slot | 33.7 min |
| `train` | 4.9 min (LightGBM 29 s, TCN 243 s); 4.4 min at `c18d76e` (31 s, 197 s) | 4.5 min alone (34 s, 222 s) |
| `predict`, dressed bundle | 57 s | 38 s |
| `cache_digests.py dump` | 3.1 min | |

From the start of the download to the dressed CSV took 1 h 39 min, including the 14.4 min wait. The disk needed was
5.1 GB for the corpus, 19 GB for the cache and 0.9 GB for the venv.

**Not covered here.**
- Other machines. W2 runs CI on GitHub's runners, and Justin's Mac digests are still to come.
- The real match.
- `corpus` and `worlds` were not re-run at `c18d76e`. Their code did not change.

### Gate G2: set-piece tag features (pre-registration)

**Registered before any G2 score was computed.** Candidate G2 is fixed by `docs/plan-final-week.md` (W4b, commit
`ff2e0b1`, merged in `c9be514`). Branch `david/w4-tags`, cut from `origin/main` at `28450b4`, with `c9be514` merged
in. External development matches only: no `data/`, no `evaluation/gt/`, no public window, no locked bench. Code:
`katena.featurize.tags`, `katena.external.setpieces`, `katena.eval.g2`, `scripts/run_g2.py`. Cache
`D:/katena/work-g2`. F's cache `D:/katena/work-final` is read only. Nothing below changes after a result.

**Candidate.** G2 is F with one change: the LightGBM member reads F's 213 columns plus 9 tag columns (222). Its rows,
labels, `d2j` pool, parameters and 150 trees are F's (`final.FinalPools` with the tag columns of the same rows
appended, through `Pools.fit`). Everything else is F's: the TCN member, `blend_heads` at w = 0.5, Viterbi at tau 2,
T = 2, and priors from `priors_for` without the held-out match. The TCN member is not retrained. Its stored
out-of-fold probabilities are read from `work-final/oof/final_tcn/T2`. In fold k, G2's LightGBM member trains on the
same 27 development matches as F's. The held-out match's tags are in no model.

**The 9 tag columns** (`katena.featurize.tags`), per 5 Hz sample and focal team, from the `SET PIECES` intervals.
Times are in seconds, computed at the sample's native frame (`idx * step`).

| column | definition |
|---|---|
| `tag_since_out`, `tag_until_out` | time since the start of the latest, and until the start of the next, `out` set piece: throw-in, corner kick or goal kick (a restart after the ball went out). "Latest" starts at or before the sample, "next" strictly after it. NaN when there is none in the match. Capped at 300 s. |
| `tag_since_fk`, `tag_until_fk` | the same for free kicks |
| `tag_since_ko`, `tag_until_ko` | the same for kick-offs, `Kick Off (Start)` and `Kick Off (After Goal)` |
| `tag_last_team`, `tag_next_team` | the tagged team of the latest and of the next set piece of any type: +1 the focal team, -1 the other team, 0 `N/A`. NaN when there is none. |
| `tag_inside` | 1 when the sample lies inside a tagged interval `[start, end)`, else 0 |

A match without tagging (SkillCorner, which has no events) gets NaN in all 9, `tag_inside` included. Real tags are
read with `katena.target.tagging.read_tags`. An instance counts when its code is `SET PIECES` and its Type is one of
the six above (both compared case-insensitively). Each one becomes a `SetPiece` on the frame grid, at
`round(seconds * 25)`.

**Imitation tags on external matches** (`katena.external.setpieces`). They come from each match's own real event
data, never from possession labels.
- **IDSSE / DFL** (`DFL_03_02_events_raw`): `ThrowIn`, `CornerKick`, `GoalKick`, `FreeKick` and `KickOff`. A kick-off
  with a `GameSection` is a `Kick Off (Start)`, one without is a `Kick Off (After Goal)`. The time is the event's
  `EventTime`, on the canonical grid (row = `round((EventTime - t0_utc) * 25)`). The team is the event's `Team`, or
  for a kick-off the team of its nested `Play`. Penalties and referee balls have no Smart Tagging Type and are left
  out. The DFL event clock is used as it is: no offset is estimated, because that estimate (`control_evidence.align_dfl`)
  uses the labels.
- **Metrica** (Sample_Game_1 CSV, Sample_Game_3 JSON): `SET PIECE` events with subtype `THROW IN`, `CORNER KICK`,
  `FREE KICK` or `KICK OFF`. The first kick-off of a period is the start, later ones follow a goal. Goal kicks are any
  other event whose subtype contains `GOAL KICK`. The time is the event's start frame minus 1.
- **SkillCorner:** no events, so the tag columns are NaN.
- The raw event files are the ones the canonical matches were built from (SHA-256 checked against the canonical
  record).

**Simulating the real tags' character.** These are stated assumptions, fixed now.
- **Recall per Type.** DATA.md publishes 26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 4 kick-offs after a
  goal and 2 at the start. Of an external match's n events of a Type, min(published count, n) are kept, drawn
  uniformly without replacement. So the external match's own count stands in for the real match's true count, which
  is unknown. Free kicks are kept at about 5 in 20 to 32, throw-ins at 26 in 28 to 54, goal kicks at 10 in 11 to 22.
- **Timing.** Where the restart sits inside a real interval is unknown. `starter/load_smart_tagging.py` says the
  intervals are windows of about 5 s around the restart. Assumed: each interval starts U(0, 5) s before its event and
  ends U(0, 5) s after it (at least 1 frame after), independently per event.
- **Team.** Every throw-in is `N/A` (DATA.md). Every other Type carries the team that takes it. The real file has 36
  `N/A` instances: the 26 throw-ins and 10 others, of any code. Which of those 10 are set pieces is unknown, so no
  other `N/A` is simulated.
- **No false positives** are added. The real file's false-positive rate is unknown.
- **Seeds.** Every draw comes from `default_rng(protocol.seed_for(match_id, "g2_tags"))`. So the tags depend only on
  the match id: one realisation per match, the same in every training world, every fold and evaluation.
- Tag columns do not depend on the world. The degrader corrupts tracking, not tags.

**Known optimism, recorded now.** External events and external labels come from the same provider, so restart timing
is cleaner in training than it will be on the real match.
- **IDSSE.** The labels are DFL's `BallStatus`/`BallPossession` and the events are DFL's own. The dead-to-alive edges
  of the labels and the set-piece `EventTime`s are two readings of the same restarts. The DFL event clock is offset
  from the tracking clock by up to about 2 s per match (`control_evidence`), and that is all that separates them.
- **Metrica.** The labels are derived from these very events (`metrica.derive_labels`: every `SET PIECE` is a
  restart for its team). So on Metrica the tags and the label transitions coincide, up to the simulated timing noise.
- **The real tags** are Metrica's automatic output from the same video and tracking. Their recall per Type is matched
  only in count, and their timing, false positives and team errors are not matched at all. Any gain measured here is
  therefore an upper bound on the gain on the real match.

**An expectation, recorded before the result.** 20 of the 28 folds are SkillCorner, where the tag columns are NaN.
There G2's member can differ from F's only through the trees: column sampling (`feature_fraction` 0.5) and splits
learned on the 7 tagged training matches. The 70% rule needs 20 wins out of 28, so G2 must also beat F on most
SkillCorner folds without any tag information. That is a hard bar for this candidate, and it stays the bar.

**Checks, before the gate is read** (they decide nothing, but if one fails the gate is not read until it is explained):
- **F's inputs reproduced.** On fold `idsse/DFL-MAT-J03WMX`, the pool copied here, without its tag columns, trains F's
  LightGBM member byte for byte: the model hash stored with F's out-of-fold file, and its probabilities at target
  and severe.
- **F re-decoded.** F re-decoded here from its two stored members (`F_check`) equals F's stored spellF1, macroF1,
  spell counts and bootstrap flags on all 56 (fold, condition) pairs.

**Rule** (`katena.eval.bootstrap`, unchanged; the plan's gate). Adopt G2 if and only if both hold, on the 28
development folds against F's stored per-fold scores (`work-final/final_results.pkl`):
- at `target`: it wins on at least 70% of folds (at least 20 of 28, a win being a strictly higher spellF1), and the
  paired 95% bootstrap interval of the mean difference lies above 0 (possessions resampled, 2000 draws, seed 0);
- at `severe`: it is not worse. Worse means a mean difference below 0 with the interval's upper bound also below 0.

**Reported, deciding nothing.** The 8 IDSSE + Metrica folds, per-dataset means and wins, G2's LightGBM member alone
(`G2G`) against F's (`FG`), macroF1 on the 8 folds, predicted possessions per true one, and the run time. Only `target`
and `severe` are predicted. D2's other worlds are not run.

**What follows.** The plan fixes the order: G12 if G1 and G2 both pass and G12 passes, otherwise G1 if it passed,
otherwise G2 if it passed, otherwise F. The plan's cut rule applies: if G2 is not gated by Nov 13, it is dropped.
Nothing is adopted after Nov 14 18:00 CET. G2 is not switched on in `katena.pipeline` whatever the result. A separate
step (W6) integrates it if it is adopted. `katena.pipeline` can already read the real file's tags into the same
columns (`stage_predict(tag_columns=True)`, off by default). F refuses that option, because its columns do not match.

### Gate G2: set-piece tag features: result

Everything below was computed after the pre-registration commit `95e97d1`, pushed at 12:20 CET on 2026-11-10. The
code is unchanged from that commit (`scripts/run_g2.py seed tags check`, then `gbdt score gate`, both through
`heavy.py`). Nothing was changed or tuned after a score. External development matches only.

**Checks (both hold).**
- **F's inputs reproduced.** On fold `idsse/DFL-MAT-J03WMX`, the copied pool without its tag columns trains F's
  LightGBM member byte for byte: the model hash, and the probabilities at target and severe.
- **F re-decoded.** `F_check` equals F's stored scores on all 56 (fold, condition) pairs.

**Inputs.** 196 files (3.1 GB) were copied from `work-final`. The 8 IDSSE + Metrica matches got 49 to 58 imitation
set pieces each, 428 in all, and about 4 to 5% of their samples lie inside a tagged interval. The kept counts per
Type follow the rule: throw-ins 26, free kicks 5, goal kicks 10, every corner (2 to 11), and every kick-off. The 20
SkillCorner matches are NaN in all 9 columns.

**Gate** (28 development folds, against F's stored per-fold scores, paired 95% bootstrap over possessions, 2000
draws, seed 0).

| comparison | folds | noise | G2 | F | diff | 95% paired CI | wins | rule |
|---|---|---|---|---|---|---|---|---|
| G2 vs F | 28 | target | 0.6065 | 0.6019 | +0.0046 | [+0.0014, +0.0079] | 16/28 (57%) | **fails** (wins below 70%) |
| G2 vs F | 28 | severe | 0.5728 | 0.5696 | +0.0032 | [-0.0000, +0.0063] | 16/28 (57%) | not worse |
| G2 vs F | 8 IDSSE + Metrica | target | 0.6220 | 0.5991 | +0.0229 | [+0.0148, +0.0309] | 8/8 | (information) |
| G2 vs F | 8 IDSSE + Metrica | severe | 0.5636 | 0.5426 | +0.0210 | [+0.0141, +0.0279] | 8/8 | (information) |

| noise | dataset | folds | G2 | F | diff | wins | fold diffs: min / median / max |
|---|---|---|---|---|---|---|---|
| target | IDSSE | 6 | 0.5894 | 0.5670 | +0.0224 | 6/6 | +0.0063 / +0.0183 / +0.0570 |
| target | Metrica | 2 | 0.7196 | 0.6955 | +0.0241 | 2/2 | +0.0238 / +0.0241 / +0.0244 |
| target | SkillCorner | 20 | 0.6003 | 0.6030 | -0.0027 | 8/20 | -0.0190 / -0.0010 / +0.0082 |
| severe | IDSSE | 6 | 0.5320 | 0.5107 | +0.0213 | 6/6 | +0.0111 / +0.0214 / +0.0357 |
| severe | Metrica | 2 | 0.6583 | 0.6382 | +0.0201 | 2/2 | +0.0130 / +0.0201 / +0.0273 |
| severe | SkillCorner | 20 | 0.5765 | 0.5804 | -0.0039 | 8/20 | -0.0233 / -0.0022 / +0.0086 |

The severe interval's lower end is -0.00004.

**Decision: G2 fails; F stays.** At target the interval lies above 0, but G2 wins only 16 of 28 folds, below the 20
the rule needs. It is not worse at severe. This is the outcome the pre-registration expected. Every tagged fold gains,
by about +0.02. The 20 untagged SkillCorner folds lose a little on average (-0.003 at target, 8 wins of 20). There
the tag columns are NaN and can only reshuffle the trees. Under the plan's adoption order, G2 is not adopted, and
G12 is not gated, since that needs both G1 and G2 to pass.

**Information only (none of it changes the decision).**
- **The LightGBM members alone.** `G2G` against `FG`: +0.0087 [+0.0043, +0.0130], 17/28 at target, and +0.0013
  [-0.0032, +0.0061], 13/28 at severe. On the 8 folds: +0.0308 (7/8) and +0.0224 (7/8).
- **macroF1, 8 IDSSE + Metrica folds.** At target: G2 0.8824, F 0.8781 (`G2G` 0.8789, `FG` 0.8721). At severe: G2
  0.8599, F 0.8549.
- **Spell counts.** Predicted possessions per true one, 28 folds: G2 0.737 against F 0.738 at target, and 0.721
  against 0.723 at severe.
- **The 8-fold gain is an upper bound**, as registered. The imitation tags come from the same provider's events as
  the labels, and on Metrica the labels are derived from those very events. The real tags' timing, false positives
  and team errors are not simulated. So the +0.02 says the columns can carry information when tags are good. It
  does not say what they carry on the real match.

**Cost** (i7-7700, sharing the machine with W1 and G1). Seeding, tags and the check took 1.8 min in a slot: 69 s of
copying, 4 s of tags, and 28 s for the check fit, at a peak of 2.0 GB. The gate stage took 12.1 min in a slot: the
28 LightGBM folds 11.1 min wall and 47 min CPU (fit 18 to 22 s per fold), scoring 17 s, the gate tables 34 s, at a
peak of 2.1 GB. The TCN member was not retrained. Cache `D:/katena/work-g2`: 3.3 GB, almost all of it copies.

**Not done, or not verified.**
- **The real tags.** No test can say how Metrica's automatic `SET PIECES` intervals sit around the real restarts,
  how many are false positives, or whether a non-throw-in Team tag names the taking side. The imitation fixes all
  of these by assumption.
- **The pipeline option.** `katena.pipeline` can read the real file into the same 9 columns
  (`stage_predict(tag_columns=True)`). It is tested on synthetic XML only, and it stays off, since G2 is not adopted.

### W4a: kickoff from Smart Tagging (adopted by the plan, no gate)

FAQ.md: the second half is graded from before its kickoff, and those frames are all D. External matches have no
graded pre-kickoff frames, so there is nothing to gate on, and the plan adopts the rule directly. It is
`katena.pipeline.apply_kickoff_rule`, applied in `stage_predict` after decoding.
- **The rule.** In each graded half block, take that half's one Smart Tagging instance whose Type is
  `Kick Off (Start)`. Type and Half are compared case-insensitively, and Half is a string. Map its start time to a
  video frame with `round(seconds * 25)` (DATA.md), and set the block's frames before it to D.
  `Kick Off (After Goal)` never matches.
- **When nothing changes.** There is no such tag, there are several, the tag lies outside the block, or it lies more
  than `KICKOFF_MAX_S` = 180 s into the block. That bound was fixed in the code before the rule ever ran on the match,
  from no frame of it: a half block can open while the teams line up, but much later a kickoff tag is more likely an
  error than a long wait. No frame constant of the match is used.
- **Report.** `predict`'s report gains `kickoff_rule`. For each half it gives whether the rule applied, how many
  frames it changed, and a status (`applied`, `no_tag`, `several_tags`, `before_block`, `after_block`,
  `beyond_bound`, `no_tagging_file`, `unreadable_<type>`). Counts and flags only.
- **Using the start time is the safe direction.** The tag is a window around the restart. When it starts before the
  real kickoff, a few pre-kickoff frames keep the model's label, and no frame of play becomes D.
- **The human check, `scripts/check_kickoff_tags.py`.** It prints PASS/FAIL only, two lines per half. The first says
  whether the tag-derived kickoff lies within 2 s of the published kickoff: for the first half that is the first
  graded frame, 11,185 (DATA.md), and for the second half 113,742 (FAQ.md). The second says whether it is not more
  than 2 s after the published kickoff, so the rule cannot turn play into D. The published numbers are check
  references only. Nothing in the pipeline reads them.
- **Tests** (`tests/test_kickoff.py`, `tests/test_pipeline_atd.py`). They run on synthetic SportsCode XML and a
  synthetic ATD bundle on the competition timeline. A bundle without a tagging file, such as a dressed external
  match, is unchanged, so the dressed-match check still compares like with like.

### Gate G1: TCN reads the touch columns (pre-registration)

**Registered before any G1 score was computed.** Workstream W3 of `docs/plan-final-week.md`, which fixes the
candidate and the gate: commit `ff2e0b1` (2026-11-10 11:50 +0100) on `origin/docs/final-week-plan`, PR #16. Branch
`david/g1-tcn-touch`, cut from `origin/main` at `28450b4`. External development matches only: no `data/`, no
`evaluation/gt/`, no public window, no locked bench. Code: `katena.eval.g1` and `scripts/run_g1.py`, written after
this entry. Cache `D:/katena/work-g1`. Nothing below changes after a result.

**Candidate.** `G1` is F with one change: its TCN member reads M2's 160 columns plus D1's 53 `touch` columns
(213 in all) instead of 160. Everything else stays as in F ("Final candidate F (pre-registration)", 2026-11-07).

- **TCN member (`G1T`).** `katena.models.tcn` unchanged apart from the input width: the first 1x1 convolution
  takes 213 channels. Trained through `ensemble.fit_tcn` with seed 0, 1200 steps, 4 threads and the deterministic
  settings, and `pin_cpu_env` runs before torch loads.
- **Sequences.** One whole-match sequence per (training match, world) of the match's `d2j` mix, as
  `final.tcn_seqs`: 15 per IDSSE or Metrica match, 9 per SkillCorner match. The order is the same (development
  match, severity, world), and the orientation draws are the same, from `default_rng(0)` over all 28 matches. So
  every sequence has F's orientation. Each row holds that world's 160 base columns, then the same world's 53 touch
  columns in the same orientation (`X0` with `A0`, `X1` with `A1`), with rows matched on `idx`. The sequences are
  held in float16 and scaled when a window is drawn, as in F.
- **Scaler.** `tcn.Scaler` on every 6th `d2j` pool row of the 27 training matches, as in F, but on all 213
  columns of `FinalPools(dev)`. The touch columns of each pool row come from the same world as the row.
- **Prediction.** Each evaluation world is predicted from its 213 columns (`touch_gate.features`), averaging both
  orientations as `tcn.predict` does.
- **LightGBM member: unchanged.** F's stored out-of-fold probabilities (`work-final/oof/final_gbdt`) are
  reused. They are provably identical: nothing in that member's training or prediction reads the TCN, and its
  inputs are the same files.
- **Decoding.** `ensemble.blend_heads(FG, G1T, w=0.5)`, the 3-state Viterbi at tau 2, priors from
  `bakeoff.priors_for` without the held-out match, T = 2. Scored with `final.decode_score`.

**Inputs.** These are copied read-only from `D:/katena/work-final` into `D:/katena/work-g1`, development matches
only: ground truth, the `d2j` pools and their touch pools, the base features and touch columns of every world of
the 28 matches (308 pairs), F's two out-of-fold probability sets, and `final_results.pkl`. Each copy is checked
byte-identical to its source by SHA-256. The touch columns carry `touch.py`'s source hash (unchanged since F), and
`touch_gate.load_tfeat` refuses stale ones. Nothing is rebuilt.

**Check before the gate.** F re-decoded from the copied probabilities with this code must reproduce F's stored
scores (spellF1, macroF1, spell counts and flags) on every fold and evaluation world. The gate reads F's stored
scores, as the plan says, and any mismatch is reported beside it.

**Rule** (fixed by the plan; `katena.eval.bootstrap` unchanged). G1 passes if and only if both hold on the 28
development folds, against F's stored per-fold scores (`work-final/final_results.pkl`, model `F`):

- at `target`, it wins on at least 70% of folds (at least 20 of 28, a win being a strictly higher spellF1), and
  the paired 95% bootstrap interval of the mean difference (possessions resampled within each match, 2000 draws,
  seed 0) lies above 0;
- at `severe`, it is not worse. Worse means a mean difference below 0 with the interval's upper bound also below
  0.

This entry decides only "G1 passes" or "G1 fails". The adoption order is the plan's: G12 if G1 and G2 both pass
and G12 passes, otherwise G1 if it passed, otherwise G2 if it passed, otherwise F. Nothing is adopted after
Nov 14 18:00 CET. Integration is W6, not this workstream.

**Information only, never a reason to change G1.**

- The 8 IDSSE + Metrica folds (`stages.FOLDS8`), at `target` and `severe`.
- The per-dataset means and wins.
- The TCN member alone: `G1T` against F's `FT`, on 28 and 8 folds.
- G1 against F, and `G1T` against `FT`, in every `worlds.EVAL_WORLDS` world that applies to the fold (the coupled
  worlds on the 8 folds only). These come from the same runs.
- MacroF1 on the 8 folds; the wall time, CPU time and peak memory of each stage.

**Recorded now.**

- These are the 5th and 6th packages gated on these 28 folds (G1 and G2), as the plan says.
- The wider input layer changes the TCN's initial weights, because torch draws every parameter in order from the
  same seed. So `G1T` against `FT` includes a fresh initialization, not only the touch columns. The F2 screen
  (2026-11-07) found that seed 0 did unusually well for F on the 8 folds. This changes nothing in the rule.

### Gate G1: result

Everything below was computed after the pre-registration commit `115f7e2` (pushed 2026-11-10 10:56 UTC), with the
code at `752d170` (`scripts/run_g1.py seed`, then `tcn score gate` in one `heavy.py` slot). Nothing was changed or
tuned after a score. External development matches only.

**Checks.**

- **Copied inputs.** All 757 files (15.0 GB) were copied from `work-final` and are byte-identical to their sources
  by SHA-256. Nothing was rebuilt.
- **Width.** The joined pool and all 300 sequences have 213 columns, and every fold's TCN has 213 input channels.
- **F reproduced.** F, `FG` and `FT`, re-decoded from the copied probabilities with this code, equal F's stored
  scores (spellF1, macroF1, spell counts and flags) on all 164 (fold, evaluation world) pairs. So G1 and F are
  decoded and scored by the same path.

**Gate** (28 development folds, fixed tau 2, paired 95% bootstrap over possessions, 2000 draws, seed 0).

| comparison | folds | noise | G1 | F | diff | 95% paired CI | wins | rule |
|---|---|---|---|---|---|---|---|---|
| G1 vs F | 28 | target | 0.6036 | 0.6019 | +0.0017 | [-0.0014, +0.0047] | 16/28 (57%) | **fails** (needs 20/28 and CI above 0) |
| G1 vs F | 28 | severe | 0.5668 | 0.5696 | -0.0028 | [-0.0062, +0.0006] | 8/28 (29%) | not worse (CI upper bound above 0) |
| G1 vs F | 8 IDSSE + Metrica | target | 0.6013 | 0.5991 | +0.0022 | [-0.0037, +0.0080] | 5/8 | fails |
| G1 vs F | 8 IDSSE + Metrica | severe | 0.5396 | 0.5426 | -0.0030 | [-0.0091, +0.0035] | 3/8 | not worse |

| noise | dataset | folds | G1 | F | diff | wins |
|---|---|---|---|---|---|---|
| target | IDSSE | 6 | 0.5679 | 0.5670 | +0.0009 | 3/6 |
| target | Metrica | 2 | 0.7016 | 0.6955 | +0.0061 | 2/2 |
| target | SkillCorner | 20 | 0.6044 | 0.6030 | +0.0015 | 11/20 |
| severe | IDSSE | 6 | 0.5111 | 0.5107 | +0.0004 | 3/6 |
| severe | Metrica | 2 | 0.6250 | 0.6382 | -0.0133 | 0/2 |
| severe | SkillCorner | 20 | 0.5777 | 0.5804 | -0.0027 | 5/20 |

At target-like, G1 loses on 11 folds (8 SkillCorner, 3 IDSSE) and ties on 1. The worst loss is -0.0128, and the
median fold gains +0.0033. At severe, it loses on 20 folds. The worst loss is -0.0231, and the median fold loses
-0.0036.

**Information only (none of it changes the decision).**

- **The TCN member alone, `G1T` against F's `FT`.**

  | folds | noise | G1T | FT | diff | 95% paired CI | wins |
  |---|---|---|---|---|---|---|
  | 28 | target | 0.6002 | 0.5961 | +0.0041 | [-0.0001, +0.0085] | 17/28 |
  | 28 | severe | 0.5730 | 0.5708 | +0.0022 | [-0.0027, +0.0072] | 18/28 |
  | 8 | target | 0.5993 | 0.5969 | +0.0023 | [-0.0060, +0.0107] | 5/8 |
  | 8 | severe | 0.5575 | 0.5477 | +0.0097 | [+0.0007, +0.0183] | 7/8 |

  The member gains more than the blend does. On 28 folds it would not pass the rule on its own at target-like
  either (17/28, and the interval touches 0). One reading, not tested here: once both members read the touch
  columns, their errors overlap more, so blending them helps less. The fresh initialization noted in the
  pre-registration is also part of every `G1T` difference.
- **D2's evaluation worlds.** G1 against F, and `G1T` against `FT`, in the same world:

| world | folds | F | G1 | diff [95% CI] | wins | G1T - FT [95% CI] | wins |
|---|---|---|---|---|---|---|---|
| `mild` | 28 | 0.6236 | 0.6273 | +0.0036 [+0.0005, +0.0068] | 19/28 | +0.0079 [+0.0035, +0.0124] | 18/28 |
| `target` | 28 | 0.6019 | 0.6036 | +0.0017 [-0.0014, +0.0047] | 16/28 | +0.0041 [-0.0001, +0.0085] | 17/28 |
| `severe` | 28 | 0.5696 | 0.5668 | -0.0028 [-0.0062, +0.0006] | 8/28 | +0.0022 [-0.0027, +0.0072] | 18/28 |
| `target_ids` | 28 | 0.6037 | 0.6003 | -0.0035 [-0.0068, -0.0003] | 8/28 | +0.0020 [-0.0022, +0.0062] | 14/28 |
| `target_smooth` | 28 | 0.6000 | 0.6007 | +0.0007 [-0.0024, +0.0038] | 15/28 | +0.0068 [+0.0024, +0.0111] | 19/28 |
| `target_cpl50` | 8 | 0.6240 | 0.6286 | +0.0046 [-0.0012, +0.0106] | 4/8 | +0.0147 [+0.0056, +0.0242] | 6/8 |
| `target_cpl100` | 8 | 0.6300 | 0.6372 | +0.0072 [+0.0014, +0.0130] | 8/8 | +0.0169 [+0.0085, +0.0257] | 8/8 |
| `target_cpl100_ids` (never trained on) | 8 | 0.6325 | 0.6367 | +0.0042 [-0.0024, +0.0113] | 6/8 | +0.0225 [+0.0137, +0.0319] | 7/8 |

- **What the worlds show.** G1 passes the full rule only in `target_cpl100`, on 8 folds. In `target_ids` it is
  worse than F by D2's test. The TCN member gains in every world, most in the coupled ones.
- **macroF1, 8 IDSSE + Metrica folds.** At target-like: G1 0.8793, `G1T` 0.8794, F 0.8781, `FG` 0.8721, `FT`
  0.8754. At severe: G1 0.8589, `G1T` 0.8592, F 0.8549, `FG` 0.8493, `FT` 0.8486.
- **Spell counts.** Predicted possessions per true one, pooled over 28 folds: G1 0.736 against F 0.738 at
  target-like, and G1 0.711 against F 0.723 at severe.

**CPU** (i7-7700, 4 cores / 8 threads; torch 4 threads).

| stage | wall | CPU | peak memory |
|---|---|---|---|
| copy F's inputs (757 files, 15.0 GB), SHA-256 of source and copy | 6.9 min | 1.8 min | 0.2 GB |
| TCN member, 28 folds: pool and 28 scalers (20 s), 300 sequences of 213 columns (3.47 GB in float16, 4.8 min to load), training 141-253 s per fold (mean 179 s), prediction 3.6 min in all | 92.2 min | 305.9 min | 4.1 GB |
| decode and score (G1 and `G1T`, plus F, `FG` and `FT` re-decoded) | 1.2 min | 0.9 min | (same process) |
| gate tables (bootstrap) | 1.6 min | 1.6 min | (same process) |
| **`heavy.py` slot, TCN through gate** | **95.0 min** | | |

W1 held the other slot when this run started. The scalers are fitted before training, so the pool is freed
before the sequences load: the peak is 4.1 GB, against 3.7 GB for F's 160-column member.

**Decision: G1 fails.** At target-like on 28 folds it wins 16 (57%, below 70%), and the paired interval includes 0
(+0.0017 [-0.0014, +0.0047]). At severe it is not worse by the rule (-0.0028 [-0.0062, +0.0006]), but that does not
rescue the target-like failure. F stays. Under the plan's fixed order, G12 is not gated. G2 is adopted if it
passes its own gate, and otherwise F stays.

**Not done, or not verified.**

- **Determinism.** No second fresh-process run of a G1 fold was made, so G1's same-machine byte identity is
  untested. F's determinism run covers the same code path at 160 columns.
- **Other machines.** Untested, as for F.
- **The cache.** Only G1's outputs are kept in `D:/katena/work-g1`: `oof/g1_tcn`, `g1_results.pkl`,
  `g1_gate.json`, `final_cost.jsonl` and `logs/`. The copied inputs were deleted, since they are byte-identical to
  `work-final`. `scripts/run_g1.py seed` restores them before `score` or `gate` can be re-run.

### Final-week gates: outcome (coordinator)

Both candidates the plan fixed before any score failed their gate against F on the 28 folds:
- G1 (TCN reads the touch columns): +0.0017 [-0.0014, +0.0047], 16/28.
- G2 (set-piece tag features): +0.0046 [+0.0014, +0.0079], 16/28.

So G12 is not gated, and under the plan's fixed order **F stays the submission model**. The kickoff rule (W4a),
adopted by the plan without a gate, is the only change to the submission. It changes `predict`, not the model, so
the expected model hashes in `docs/repro/expected-windows.json` stand.

**A design note for the record (2026-11-10).** G2's columns exist only where external events exist. That is the 8
IDSSE + Metrica folds, where G2 gained +0.0229 (8/8). The 20 SkillCorner folds cannot gain, so a 70%-of-28 rule
could hardly be met. The gate should have been registered on the 8 tagged folds, as D2 reports dead-ball changes.
That is a mistake in the plan, found after the result. Re-gating G2 on 8 folds now would be choosing after seeing
the scores, so G2 stays out.

### J3: cross-OS reproduction in CI (`ci/repro-workflow`, plan W2)

External matches only. CI never has the match. `.github/workflows/repro.yml` (on demand, `docs/ci.md`) makes every
runner rebuild the external corpus from its SHA-256-pinned sources (no new pin: the manifest is unchanged). It then
builds the cache from empty (workers = CPU count), trains F on all 30, and dumps the stage digests. Finally it exports
IDSSE `DFL-MAT-J03WMX` as an ATD bundle and labels it with the all-30 model, exactly as David's Nov 8 reference was
made (`docs/repro/windows-reference.json`, from "J2 checks and J3 on Windows"). Only hashes and aggregates leave a
runner. Three runs, all with uv 0.13.1, Python 3.11.17, numpy 2.2.6, LightGBM 4.7.0 and torch 2.15.0:
[run 1](https://github.com/jusstinn/katenaball/actions/runs/42805119378),
[run 2](https://github.com/jusstinn/katenaball/actions/runs/42811511848),
[run 3](https://github.com/jusstinn/katenaball/actions/runs/42814879839).

**Answer: the labels agree on every machine; the models do not.**
- **Dressed CSV.** Byte-identical everywhere: `909f4ffa6aac5705d5269af5b7072a424afe6ca81c02e565bf904e1147aeb788`
  from every training below: 11 trainings on 7 combinations of OS and CPU, giving 5 different TCNs. So all 173,528
  labels agree, including the 145,608 graded frames.
- **Training inputs.** The arrays of all 861 cache files are equal on every OS. That covers 30 canonical matches,
  30 ground truth files, 30 base pools, 381 feature files, 330 touch files, 30 `d2j` pools and 30 touch pools. The
  one exception is on macOS arm64: the base features and pools (`X`, `X0`, `X1`) differ only in NaN sign bits. An
  arithmetic NaN has its sign bit set on x86-64 and clear on arm64, and nothing in the pipeline reads it.
- **Models.** These parts are identical everywhere: the LightGBM member (`b2146150...`, both heads), the feature
  names, the TCN scaler and the Viterbi priors.
- **Where they part.** Two parts differ.
  - **The TCN fit.** This is the first stage whose values differ. Its weights depend on the OS and on the CPU
    vendor. The HSMM `logdur` prior also differs by OS. F decodes with Viterbi and never reads it.
  - **File bytes.** Between Windows and Unix, the files differ from the first stage on. Python's `zipfile`
    records the writing OS in every `.npz` entry (`create_system`).

| machine | OS | CPU | TCN | bundle.pkl | HSMM `logdur` | CSV |
|---|---|---|---|---|---|---|
| David's PC (reference, Nov 8, twice) | Windows 10 | Intel i7-7700 (AVX2) | `01488c84` | `cb0ea4fc` | `7c7d25a0` | `909f4ffa` |
| `windows-2025`, runs 1 and 3 | Windows Server 2025 | Intel Xeon 8370C (AVX-512) | `01488c84` | `cb0ea4fc` | `7c7d25a0` | `909f4ffa` |
| `windows-2025`, run 2 | Windows Server 2025 | AMD EPYC 9V45 | `6d5ce4da` | `fe340e71` | `7c7d25a0` | `909f4ffa` |
| `ubuntu-24.04`, runs 1 and 2 | Ubuntu 24.04 | AMD EPYC 9V74, then 9V45 | `2b273d3f` | `7d25a18a` | `80a0e444` | `909f4ffa` |
| `ubuntu-24.04`, run 3 | Ubuntu 24.04 | Intel Xeon 8573C | `a7623330` | `12621be3` | `80a0e444` | `909f4ffa` |
| `macos-15`, runs 1, 2 and 3 | macOS 15 | Apple M1 (virtual, 3 cores, 7 GB) | `a4e9a8e1` | `eb817ccf` | `0e80d4de` | `909f4ffa` |

- **Windows on Intel is bit-exact.** David's PC and a different Intel CPU, twice, reproduce every file, the bundle
  and the CSV byte for byte, so the `CPU_ENV` pinning works there. Repeated runs on the same OS and CPU vendor give
  the same TCN each time.
- **Elsewhere, the TCN depends on both the OS and the CPU vendor.** The torch builds differ: GCC with MKL (`+cu132`)
  on Linux, MSVC with MKL (`+cpu`) on Windows, and Accelerate without MKL or oneDNN on macOS. Windows and Linux
  differ even on the same CPU model (EPYC 9V45). Intel and AMD differ on both OSes, and two AMD models agree on
  Linux. Justin's Mac (Nov 8, TCN `456fc2c2...`) differs from the macOS runner, too.
- **What an organiser would see.** On Windows with an Intel CPU, the same submission bytes and model bytes. On
  anything else, the same labels on this external match, with a different TCN.

**Found and fixed on the way.**
- **Cross-OS digests.** `cache_digests.py` compared file bytes, which can never match between Windows and Unix
  (run 1: 0 of 861 files equal, while the dressed bundle and LightGBM matched). The Nov 8 claim "equal arrays give
  equal file bytes" holds within one OS family only. `dump` now also hashes each file's arrays, exactly and with
  NaNs made canonical, plus a short digest per member. `diff` judges a stage by content and reports a NaN-bits-only
  stage as `nan_payload`. The HSMM part hash is split into `logT`, `logdur` and `dur_params`. The reference was
  re-read from the unchanged `D:/katena/work-submit` (every file's bytes as on Nov 8). Tests cover the container,
  NaN and old-dump cases.
- **Report crash on Windows.** The report step crashed when it sized a cache-restored SkillCorner clone on Windows
  (run 2): one entry there refuses `stat`. The step now skips unreadable entries. Run 2's Windows hashes come from
  its uploaded digests and its log.

**Cost per runner** (minutes, wall; peak RSS of the largest process in brackets, Unix only):

| | uv sync | corpus download and build | corpus | worlds | train | digests | export and predict | job |
|---|---|---|---|---|---|---|---|---|
| `ubuntu-24.04` (4 vCPU, 16 GB), run 1 | 0.4 | 6.6 (0.7 GB) | 5.1 (0.9) | 11.2 (0.9) | 2.1 (4.7) | 0.8 | 0.4 (1.2) | 28.6 |
| `windows-2025` (4 vCPU, 16 GB), run 1 | 0.1 | 6.8 | 9.6 | 19.5 | 3.9 | 0.8 | 0.7 | 42.5 |
| `macos-15` (3 vCPU, 7 GB), run 1 | 0.1 | 7.2 (1.2) | 4.3 (0.8) | 11.7 (0.8) | 9.5 (2.7) | 0.5 | 0.5 (1.0) | 35.2 |

Whole jobs in runs 2 and 3: Ubuntu 21.8 and 26.0 min, macOS 31.4 and 36.6, Windows 20.3 (AMD) and 40.6 (Intel).
Each run took 32-43 min end to end. Run 1 downloaded the raw files (0.92 GB per OS in `actions/cache`); later runs
restore them, and the corpus download and build drops to 2-4 min. Disk was never short: after cleanup, 117 GB free
on Ubuntu, 69 GB on macOS, 146 GB on Windows `D:`. The cache is 18.1 GB and the corpus 5.1 GB on every OS. macOS
fits in 7 GB of RAM (train peaks at 2.7 GB).

**Not verified.**
- **One match.** Labels were compared on one external match. On the real match, a near-tie could flip a label
  where the TCN's last bits differ. No bundle leaves a runner, so the cross-OS probability differences were not
  measured. The Nov 8 kernel probe on Windows moved them by at most 5e-7 and changed no label.
- **Dispatch path.** The runs came from the temporary `push` trigger. The `workflow_dispatch` path (the `os`
  input) runs only once the file is on main.
- **Reference timings.** The reference's timings are the Nov 8 entry's (2 workers), not remeasured.

## 2026-11-09

### Justin's branch: G1 and G2 (pre-registration)

**Registered before any SkillCorner fold and any 28-fold score of either candidate.** Justin + Claude, branch
`claude/git-pull-analyze-strategy-e4504e`, the Mac (macOS arm64). External development matches only: no `data/`, no
`evaluation/gt/`, no public window, no locked bench. Code: `scripts/run_g12.py`, run unattended by
`scripts/night_g12.sh`, which refuses to start unless this entry and the code are committed. Cache: the Mac's
`work/final`, as `make model` built it. Nothing below changes after a result.

**What had been seen when this was written.** The strategy review below: the 8 IDSSE + Metrica folds with the
LightGBM member at 150, 300, 600 and 1,200 trees. The size 1,200 comes from there. It is the largest size that
finished; one fold of a 3,000-tree run was seen and is not used. No SkillCorner fold of a larger LightGBM member, and
no score of any TCN other than F's two seeds. The code was smoke-tested on two folds (one IDSSE, one SkillCorner)
with a 200-tree member and 30-step TCNs, under the tag `smoke`. Those numbers decide nothing and are not reported.

**G1, candidate.** F with 1,200 trees in the LightGBM member: `gbdt.fit_heads(..., n_trees=1200)`. Unchanged:
`gbdt.PARAMS` (learning rate 0.1, 31 leaves, bagging, seed 0), the `d2j` pool, the 213 columns, the TCN member
(seed 0, 1,200 steps, 160 columns), w = 0.5, Viterbi at tau 2, the priors, T = 2.

**G1, incumbent.** F, recomputed in the same run on the same machine. Its LightGBM member is the same 1,200-tree
model read at its first 150 trees. LightGBM's rounds are sequential, so that is F's member, and the run checks it
byte for byte against the F2 screen's cached member on the 8 folds that cache holds. F's TCN is trained here for
all 28 folds as `final.stage_tcn` trains it, checked the same way against the F2 screen's cache. Both sides of every
comparison share those TCN probabilities. Side check: F's 28-fold mean at `target` here against 0.6019 on the
Windows PC.

**G1, rule.** G1 passes if and only if all three hold on the 28 development folds:

- at `target`, it wins at least 20 of 28 folds and the paired 95% bootstrap interval lies above 0
  (`bootstrap.adopted`, F's rule);
- at `severe`, it is not worse (mean difference below 0 with the interval's upper bound below 0);
- at `target` on the 20 SkillCorner folds alone, it is not worse. This is added because the other 8 folds chose
  the size, so the SkillCorner folds are the only unseen ones.

Reported beside it and deciding nothing: the SkillCorner and the IDSSE + Metrica folds separately, the member alone,
every `worlds.EVAL_WORLDS` world this cache holds, macroF1, predicted possessions per true one. `make model` builds
the training worlds only, so the Mac's cache has no `target_cpl100_ids`, the one evaluation world outside the
training mix. It is not built for this run. If G1 fails, F keeps 150 trees and no other size is tried.

**G1, if it passes.** It becomes the incumbent side of G2 and goes into F3's assembly, which gets its own
registration before the final model is trained. It changes no submission by itself. The final model's LightGBM
hash must again agree between the Mac and the Windows PC.

**G2, candidates.** Two, fixed now. Both replace F's TCN member and change nothing else.

| Variant | Steps | Input columns | Everything else |
|---|---|---|---|
| `L` | 4,800 (`ensemble.fit_tcn(steps=4800)`; the one-cycle schedule spans them) | M2's 160, as F | As F: peak learning rate, batch 32, window 256, seed 0, 4 threads, the `d2j` sequences, the scaler |
| `LT` | 4,800 | The 160 plus D1's 53 `touch` (213) | As `L`. Each sequence's touch columns come from the same world and orientation as its base columns, from the files F's LightGBM pool was built from, held in float16 like the rest. The scaler is fit on all 213 columns of the same pool rows |

**G2, screen.** The 8 IDSSE + Metrica folds. Each variant is blended at w = 0.5 with the LightGBM member that
stands after G1's gate (1,200 trees if G1 passed, 150 if not) and compared with F's TCN blended with that same
member. A variant passes the screen if at `target` it wins at least 6 of 8 folds with the paired 95% interval
above 0, and it is not worse at `severe`. This is the F2 screen's rule.

**G2, gate.** A variant that passes the screen goes to the 28 folds under F's rule (at least 20 of 28 and the
interval above 0 at `target`, not worse at `severe`), against the same incumbent, in the same run if the night
allows. If both pass the screen, only the one with the higher 8-fold mean at `target` goes, and a tie goes to
`L`. If neither passes, G2 is dropped and no other step count or input set is tried. A variant that passes the
gate repeats D3's kernel-swap check on the dressed match before it is adopted, because a longer run can amplify
last-bit differences between CPUs.

**Information only.** The same comparisons with the other LightGBM size, each TCN alone, every evaluation world,
macroF1, predicted possessions per true one, and run times.

### Strategy review: one look at the public labels, two exploratory screens (Justin + Claude)

Branch `claude/git-pull-analyze-strategy-e4504e`, `main` at `28450b4`, the Mac. External matches and the public
label file only: no `data/`, nothing on the provided match, no locked bench. Nothing below was registered before it
ran, so nothing below decides anything. Detail, tables and the plan: [improvement-plan.md](improvement-plan.md).

**Checks.** `uv run pytest -q`: 225 passed, 14 skipped. F re-decoded from the cached probabilities of
`screen_tcn_seeds.py` gives 0.5991 at `target` and 0.5426 at `severe` on the 8 IDSSE + Metrica folds, and the
LightGBM member refitted from the pool gives 0.5839 and 0.5221. All four equal the logged values.

**A look at the public window, labels only** (`scripts/probe_label_structure.py`). No prediction and no score was
involved, and no file of ours was compared with it. The window's 17 possessions (12% under 2 s, 35% under 4 s,
median 7.6 s, 71% direct hand-overs) fall in the middle of what 5-minute windows of IDSSE labels at T = 2 give:
52% of those windows have 17 possessions or fewer, 48% have as few under 4 s. Mean length leans towards T = 3 to
5 (16% of T = 2 windows reach 14.2 s), and only T = 0 looks unlikely. Seventeen possessions cannot separate T = 2
from T = 3. **`T_MAIN` stays 2.** No model, constant or rule was changed because of this look.

**Screen 1: the LightGBM member's size** (`scripts/screen_gbdt_trees.py`, 8 folds, tau 2). One 1,200-tree model
per fold, read at 150, 300, 600 and 1,200 trees, everything else as F.

| | target | severe | target_smooth | target_cpl100 |
|---|---|---|---|---|
| Member alone, 150 trees (F's) | 0.5839 | 0.5221 | 0.6011 | 0.6281 |
| Member alone, 1,200 trees | 0.6042 | 0.5511 | 0.6289 | 0.6545 |
| Difference [95% paired CI], wins | +0.0203 [+0.0114, +0.0287], 8/8 | +0.0290 [+0.0196, +0.0377], 8/8 | +0.0278 [+0.0185, +0.0366], 8/8 | +0.0264 [+0.0178, +0.0357], 8/8 |
| F-style blend, 150 trees (F) | 0.5991 | 0.5426 | | |
| F-style blend, 1,200 trees | 0.6095 | 0.5610 | | |
| Difference [95% paired CI], wins | +0.0104 [+0.0049, +0.0159], 7/8 | +0.0185 [+0.0126, +0.0247], 8/8 | | |

The member is under-sized. Four sizes were looked at on 8 folds with no SkillCorner fold, so this is a reason to
gate, not a result.

**Screen 2: a smooth-only training mix** (same script). Every severity's rows from its `_smooth` world. Against
`d2j` at 600 trees: +0.0215 [+0.0132, +0.0299] in `target_smooth` (7/8), +0.0140 in `mild_smooth`, +0.0030 in
`severe_smooth`; -0.0852 in `target`, -0.1339 in `severe`, -0.0740 in `target_cpl100` (0/8 each). The `d2j`
hedge costs about 0.02 if the real tracking is smooth, and dropping it costs about 0.1 if it is rough. `d2j`
stays unless the team decides that the jitter statistic of `check_touch_target.py` may choose a mix.

**Decoder, information only.** On F's cached probabilities no change gains more than 0.002: tau 1.5 to 6, a bonus
on team or alive/dead switches, the blend weight.

**Proposed, not yet registered** (owners and rules in the plan): G1, F with a 1,200-tree LightGBM member, on the
28 folds under F's rule; G2, a 4,800-step TCN, screened on 8 folds first; G3, a smooth-major mix, conditional on
the organisers' wording.


## 2026-11-08

### Locked-bench sanity read of F (registered before the read)

Written before any bench score of F was computed. Branch `david/bench-read`, cut from `origin/david/j2-tests` at
`8bbd252` (PR #12 is still open). External matches only: no `data/`, no `evaluation/gt/`, no public window. Script
`scripts/run_bench_read.py`, cache `D:/katena/work-bench`.

**What is read.** F trained once on the 28 development matches. Both benches are left out of every model, scaler and
prior. The recipe is `katena.eval.final` unchanged, run as a gate fold with no development match held out:
- the LightGBM member on `FinalPools(dev)` (213 columns, the `d2j` pool);
- the TCN member from `tcn_seqs(dev)` and `fit_tcn` (seed 0, 4 threads, the deterministic settings, 1200 steps);
- `blend_heads` at w = 0.5, Viterbi at tau 2, T = 2, and priors from `priors_for` on the 28 matches' labels.

So the TCN's orientation draws run over the 28 matches, as in the gate, not over all 30 as in `pipeline.stage_train`.
Each bench is scored with `final.decode_score` (`lomo.score_rows`, the official `evaluate`), for spellF1 and macroF1.

**Conditions.** `target` and `severe` on both benches. The other `worlds.EVAL_WORLDS` are information only:
`mild`, `target_cpl50`, `target_cpl100`, `target_ids`, `target_cpl100_ids` and `target_smooth`. Their features and
touch columns come from `work-submit`. The exception is `target_cpl100_ids`, an evaluation-only world, which is built
here for the two benches with `d2.build_ref` and `final.build_touch`, recovery checks included.

**Compared with M2's stored bench scores.** These are in bakeoff.md Table 8, from `stages.stage_bench`: M2 trained on
the 28 development matches, Viterbi at tau 2 (the argmax of M2's tau table), priors from the 28, T = 2.
- No bench probabilities were stored. `work-base` never ran `stage_bench`, and no cache on this PC holds a bench
  score, so there is nothing to re-decode.
- Table 8 prints 3 decimals, and its scorer predates D5's block-edge fix.
- So M2 is also retrained here on the same 28 matches, the way `stage_bench` does it (`Pools(dev)`, all 160 columns).
  It is decoded and scored exactly as F, in every condition read.
- This M2 must reproduce Table 8's M2 at `native`, `mild`, `target` and `severe` on both benches, to the 3 decimals
  printed. If it does, the table compares F with this M2 at full precision. If it does not, the gap is reported and
  explained before any F - M2 difference is read.

**Inputs.**
- Development files come from `work-final`, the gate's own inputs. Each is compared byte for byte with its copy in
  `work-submit`.
- Bench files come from `work-submit`. Their ground truth and their `native`, `mild`, `target` and `severe` features
  are compared with `work-base`.
- Every bench world that F reads is re-corrupted and re-featurized, and its touch columns are recomputed. Both must
  be byte-identical to the copies.

**Report only.** The read changes no decision, whatever the result. F stays adopted, and no constant, seed, weight,
world or tau changes. Two matches give no usable interval. A gross failure means F below M2 by more than 0.05
spellF1 on either bench at `target` or `severe`, about three times the worst development-fold loss (-0.018). It is
treated as a possible bug to investigate, not as a model choice.

**Result.** Computed after the registration commit `4aa8bf1`, with `scripts/run_bench_read.py seed worlds train score`
(code `2cea318`), in one slot on the i7-7700. Nothing was changed after a score.

**Checks.**
- **Inputs.** 476 files (12.85 GB) were copied. All 422 that have a second copy are byte-identical to it:
  - from the 28 development matches (`work-final` against `work-submit`): ground truth, base and `d2j` pools, touch
    pools, and 300 training-world features;
  - from the benches (`work-submit` against `work-base`): ground truth and the `native`, `mild`, `target` and
    `severe` features.
- **Bench worlds.** `target_cpl100_ids` was built for both benches and passed the recovery check. All 16 bench worlds
  that F reads (8 per bench) re-corrupt to the cached rows, M0 and degrader statistics. They re-featurize byte for
  byte, and their touch columns recompute byte for byte.
- **M2 is comparable.** M2 was retrained here on the 28 development matches (614,802 rows x 160 columns). It
  reproduces all 8 M2 cells of Table 8 to the 3 decimals printed, in spellF1 and macroF1: `native`, `mild`, `target`
  and `severe` on both benches. So the stored numbers used T = 2, tau 2 and the same conditions. The tables below use
  this M2 at full precision.

**F against M2 on the benches** (official scorer, T = 2, Viterbi at tau 2; the registered conditions).

| bench | condition | F spellF1 | M2 spellF1 | diff | F macroF1 | M2 macroF1 | diff |
|---|---|---|---|---|---|---|---|
| IDSSE `DFL-MAT-J03WQQ` | target | 0.6092 | 0.5669 | +0.0422 | 0.8912 | 0.8848 | +0.0064 |
| IDSSE `DFL-MAT-J03WQQ` | severe | 0.5605 | 0.5478 | +0.0126 | 0.8662 | 0.8625 | +0.0037 |
| Metrica `Sample_Game_2` | target | 0.6820 | 0.6311 | +0.0509 | 0.8958 | 0.8837 | +0.0121 |
| Metrica `Sample_Game_2` | severe | 0.5945 | 0.5371 | +0.0574 | 0.8640 | 0.8447 | +0.0193 |

**Information: the other evaluation worlds** (M2's heads in the same world).

| bench | world | F spellF1 | M2 spellF1 | diff | F macroF1 | M2 macroF1 | diff |
|---|---|---|---|---|---|---|---|
| IDSSE | `mild` | 0.6299 | 0.6126 | +0.0173 | 0.9123 | 0.9036 | +0.0086 |
| IDSSE | `target_cpl50` | 0.6206 | 0.6045 | +0.0162 | 0.9031 | 0.8934 | +0.0098 |
| IDSSE | `target_cpl100` | 0.6197 | 0.5934 | +0.0263 | 0.9098 | 0.8935 | +0.0163 |
| IDSSE | `target_ids` | 0.6034 | 0.6064 | -0.0030 | 0.8953 | 0.8972 | -0.0019 |
| IDSSE | `target_cpl100_ids` | 0.6414 | 0.6193 | +0.0221 | 0.9118 | 0.8976 | +0.0142 |
| IDSSE | `target_smooth` | 0.6281 | 0.5915 | +0.0366 | 0.9050 | 0.8892 | +0.0158 |
| Metrica | `mild` | 0.7202 | 0.6622 | +0.0580 | 0.9009 | 0.8857 | +0.0152 |
| Metrica | `target_cpl50` | 0.6991 | 0.6585 | +0.0406 | 0.8937 | 0.8846 | +0.0091 |
| Metrica | `target_cpl100` | 0.6970 | 0.6685 | +0.0285 | 0.8971 | 0.8833 | +0.0137 |
| Metrica | `target_ids` | 0.6836 | 0.6341 | +0.0495 | 0.8932 | 0.8833 | +0.0099 |
| Metrica | `target_cpl100_ids` | 0.6890 | 0.6488 | +0.0402 | 0.8946 | 0.8840 | +0.0105 |
| Metrica | `target_smooth` | 0.7116 | 0.6667 | +0.0450 | 0.9006 | 0.8873 | +0.0133 |

- **The members alone, spellF1.** At `target`: IDSSE `FG` 0.5695 and `FT` 0.6064; Metrica `FG` 0.6691 and `FT`
  0.6787. At `severe`: IDSSE `FG` 0.5278 (M2 0.5478) and `FT` 0.5694; Metrica `FG` 0.5749 and `FT` 0.6037. The
  LightGBM member falls below M2 only on IDSSE at `severe`, where D2 already saw `d2j` cost IDSSE.
- **Spell counts.** Predicted possessions per true one at `target`: IDSSE F 0.663 against M2 0.689, Metrica F 0.847
  against M2 0.804.
- **Hashes, this PC.** The 28-match models: LightGBM member `c2ec4827...`, TCN `5cb5928e...`, M2 `7db0d235...`. This
  is not the submission model, which also trains on the benches.

**Reading.** F scores above M2 on both benches at `target` and `severe`: +0.013 to +0.057 spellF1 and +0.004 to
+0.019 macroF1. That is nowhere near the registered gross-failure line, so nothing needs to be investigated. The gaps
look like the development folds. There, F gained +0.023 (IDSSE) and +0.052 (Metrica) at `target`, and +0.019 and
+0.053 at `severe`. Here the gains are +0.042 and +0.051 at `target`, and +0.013 and +0.057 at `severe`. In 15 of
16 bench worlds F is ahead. The exception is IDSSE `target_ids` (-0.003), smaller than the worst development-fold loss at
`target` (-0.018).
The read also checks the path end to end. A model that never saw the benches labels them in all 8 worlds through the
gate's code, and the M2 reproduction ties the comparison to the bake-off's own bench numbers. Two matches give no
interval, so this read is consistent with the gate and is not more evidence for F. It changes no decision.

**Cost.** Copying took 8.2 min (disk bound). The worlds stage and checks took 3.5 min. Training took 13.0 min:
LightGBM 56 s, TCN 660 s, M2 66 s. Scoring took 5 s. The total in the slot was 24.9 min, with a 3.5 GB peak. The TCN
took 3x its time in J2's `train`, while the train stage's CPU time (749 s) was about its wall time (782 s), so torch
got well under its 4 threads. The cause was not investigated.

**Kept.** In `D:/katena/work-bench`: `bench_read.json` (every score and check), `bench_results.pkl`,
`final_cost.jsonl`, `logs/bench.log`, and `oof/` (the bench probabilities of `FG`, `FT` and M2, 11 MB). The copied and
built inputs (12.8 GB) were deleted. They are copies of `work-final` and `work-submit`, and `target_cpl100_ids` is
rebuilt by the `worlds` stage.

### J2 checks and J3 on Windows (David + Claude, `david/j2-tests`)

External matches and synthetic files only. Nothing ran on the real match. Windows 10, i7-7700 (AVX2), torch
2.15.0+cpu, LightGBM 4.7.0, cache `D:/katena/work-submit`, built from scratch with `j2-final-pipeline`'s code.

**The ATD reader is now in the tests.** J2's smoke test labelled a hand-built `Match`, so `load_target` was never on
the tested path. `tests/test_pipeline_atd.py` writes a synthetic bundle in the real format: EPTS metadata that
passes `layout.assumptions_hold`, and raw lines with frame ids 1..199,319. It uses `games.py`'s timeline and graded
halves and runs `stage_predict` on it through `load_target`. `evaluation/validate_submission.py` accepts the CSV
unchanged: one LF row per frame in order, no X in the halves, X outside. The test takes about 45 s.

**Dressed match: the submission path scores exactly what the evaluation path scores.**
- `katena.target.dress` writes a canonical match in the ATD format. `load_target` gets an optional `Timeline` for
  the match's own frame count, half blocks, pitch and frame rate. The competition match still uses `games.py`.
- `IDSSE DFL-MAT-J03WMX` (25 Hz, 173,528 frames, 32 slots) reads back identical in every field, bitwise: positions,
  speed, ball, teams, half blocks, pitch, fps, and the attack directions inferred from positions.
- F trained without it on the other 29 (`train --hold-out`, saved to `model_without_*/`, never `model/`).
  `predict --hold-out --timeline` labelled the bundle. The cache's `native` features, the `touch_gate` native touch
  columns and every probability of both members and the blend are bitwise equal on both paths. All 29,123 work-grid
  labels and every graded frame agree.

| match | official scorer on the CSV | evaluation path (`lomo.score_rows`, native) | difference |
|---|---|---|---|
| IDSSE J03WMX, model without it | spellF1 0.650611, macroF1 0.930641, 232 / 341 possessions | 0.650611, 0.930641, 232 / 341 | 0 |
| SkillCorner 1874553 (10 Hz), model without it | 0.640625 as written; 0.652756 with D -> X | 0.652756, macroF1 0.573902 | -0.012131 as written, 0 with D -> X |

- The SkillCorner gap is the scoring convention, not plumbing. SkillCorner has no dead-ball label, so the evaluation
  path turns a predicted D into X (`score(sc_mode=True)`), which the scorer fills with the previous team. On the
  CSV, the official scorer reads a D on an alive frame as a break. With the same mapping the two agree exactly.
- The only other difference is 5 frames (1 on SkillCorner) just outside the halves. `to_native_codes` labels them
  from the nearest sample and `predict` writes X. They are not graded.

**J3: Windows against Justin's Mac (only 8-character prefixes of the Mac hashes are recorded).**

| hash | Mac (arm64) | Windows (x86-64) | |
|---|---|---|---|
| gbdt | `b2146150...` | `b2146150e006ead3b543f2ee424c84d9a12446ff551dd505f9480eb6a582ec0e` | same |
| tcn | `456fc2c2...` | `01488c84721327147b3975886e3b758a13f049531146bc54985b88976e8f6eb3` | differs |
| bundle.pkl | `16eeaed0...` | `cb0ea4fc546014665888a69ae86e4ba83f79cf7ff10dacc051f54b00f105a85c` | differs (holds the TCN) |

- **Where they part.** The LightGBM member is the same on both machines (657,133 pool rows x 213 columns on both), so
  its inputs almost certainly are too: the corpus, the worlds, and the `d2j` pools with their touch columns. The
  TCN weights differ, as D3 predicted (torch kernels per CPU). The scaler and the priors are inside the bundle and
  are not separately confirmed.
- **Windows is self-consistent.** The from-scratch cache equals the gate's own Windows cache (`work-final`) on every
  common file: 30 canonical matches, 28 ground truth and base pools, 300 feature and 300 touch files, 28 `d2j` pools
  and 28 touch pools.
- **To finish the localisation,** Justin runs `scripts/cache_digests.py dump` on the Mac and `diff`s it against
  Windows. That compares each model part. For labels, `scripts/dressed_check.py label --bundle <Mac bundle.pkl>`
  labels the exported match with his model. The bundle is derived from external data only.
- **What the labels do across kernels, measured on Windows.** The held-out TCN was retrained with other torch kernels
  (`ATEN_CPU_CAPABILITY=default`, `ONEDNN_MAX_CPU_ISA=SSE41`, `MKL_CBWR=COMPATIBLE`). The TCN hash changes and its
  probabilities move by at most 4e-7 (blend 5e-7). The dressed CSV is byte-identical: no graded frame of 145,608
  changes. That is a proxy, not the Mac.

**Same machine, fresh processes.** `train` on all 30 and then `predict` on the dressed bundle, run twice: the same
`bundle.pkl` (`cb0ea4fc...`), and a byte-identical CSV (`909f4ffa...`). The manifests differ only in run times.

**Cost on this PC** (2 workers): corpus 18.0 min, worlds 33.7 min, cache 19 GB. `train` takes 4.5 min alone
(LightGBM 34 s, TCN 222 s), 6.8 min beside another run, and peaks at 4.0 GB. `predict` takes 40-50 s on the
173,528-frame bundle and peaks at 0.7 GB.

**Fixed.** `write_submission` left a file that failed the official checks at the output path. It now checks a
temporary file and moves it into place only when valid. Test added.

**Found, not code bugs.**
- `make` is not installed on the Windows PC, so `make submit` does not run there. `docs/REPRODUCE.md` lists the
  plain commands.
- `build_external.py` (the first `make submit` step) will rebuild all 30 matches once. No LF checkout reproduces the
  sidecars' loader digests: IDSSE was built from a CRLF checkout, and Metrica and SkillCorner by code edited after
  the build. The rebuild was run in memory on Windows, and all 30 canonical files come out byte-identical.

## 2026-11-07

### Screen: F2, two TCN seeds (pre-registration, written before any score)

**Candidate F2.** F with its TCN member replaced by the logit-space mean (`ensemble.blend`, w = 0.5) of two
TCNs trained identically except for the seed: 0 and 1. Then F2 blends exactly as F does, with the LightGBM
member at w = 0.5 and Viterbi at tau 2. Nothing else changes: worlds, pool, columns, priors, steps and
threads are all the same.

**Screen.** The 8 IDSSE + Metrica development folds at `target` and `severe`, LOMO on the 27 other
development matches (benches excluded), on this Mac. F (seed 0 only) is recomputed here the same way.

**Rule, fixed now.** F2 is promising, and goes to David's 28-fold gate, if at `target` it wins at least 6
of 8 folds with a paired 95% CI above 0, and it is not worse at `severe` (mean < 0 and CI upper bound < 0).
Otherwise it is dropped. The screen decides nothing for the submission by itself.

**Side check.** F recomputed here should match David's Windows means on the same 8 folds (target 0.5991,
severe 0.5426). Any difference is a cross-machine reproducibility finding for J3.

**Result (`scripts/screen_tcn_seeds.py`, about 35 min on the Mac): drop F2.**

| Condition | F | F2 | F2 - F [95% paired CI] | F2 wins |
|---|---|---|---|---|
| target | 0.5991 | 0.5939 | -0.0052 [-0.0089, -0.0018] | 1/8 |
| severe | 0.5426 | 0.5375 | -0.0051 [-0.0099, -0.0003] | 1/8 |

- **Cross-machine reproduction.** F recomputed on macOS arm64 gives the same 8-fold means as David's Windows
  x86 run, to the 4 decimals reported: 0.5991 and 0.5426. Torch's bytes can differ across CPUs, but the
  scores did not move.
- **What it says about F.** Averaging seed 1 into the TCN lowers the score on 7 of 8 folds. So seed 0 does
  unusually well on these folds, and seed 1 does not (D3 also saw a lower seed-1 ensemble gain, +0.0084
  against +0.0120). F was assembled with seed 0 after seeing these folds, so part of its 8-fold gain over M2
  (+0.030) may be seed luck. The 28-fold gate (+0.016, CI [+0.011, +0.021]) is the better guide. This
  changes no rule outcome. It is a caution for the team's decision on PR 8, and an argument for keeping
  seed 0 fixed rather than re-drawing it.
- **How large the seed effect is (information only; F's seed stays 0).** On the same cached probabilities,
  with M2 for reference:

  | Condition | M2 | LightGBM member alone | F, seed 0 | F, seed 1 |
  |---|---|---|---|---|
  | target | 0.5689 | 0.5839 | 0.5991 | 0.5914 |
  | severe | 0.5149 | 0.5221 | 0.5426 | 0.5326 |

  With the other seed, F is still about +0.02 above M2 on these folds. Roughly a quarter of the 8-fold gain
  depends on the seed, and the rest holds.

### J4 gap filling: deprioritised, not run

M2's base features already carry possession across ball gaps:
- `interp_short` fills short gaps.
- Control and margin windows of 1-6 s (past, future and centred) average over the visible samples.
- `since_tight` and `until_tight` (30 s cap) and `last_ctrl_sign` / `next_ctrl_sign` hold who last and next
  controlled the ball.
- The miss group has the time since and until the ball was seen.
- D1's touch features hold the last and next touching team.

A J4 gate would need F's TCN out-of-fold probabilities rebuilt on this machine (about 100 min) for a small
expected gain. We dropped it before computing anything, so no score informed this.

### Public-window check: `scripts/check_public_window.py` (pipeline check only)

It prints PASS/FAIL only, never a score:
- the official file checks;
- home/away orientation: swapping H and A must lower team accuracy;
- frame alignment: shifts of ±5 and ±10 s must lower spellF1;
- better than the published smooth50 baseline.

A FAIL points at a bug (slot order, home team, offsets), never at tuning. `tests/test_check_public_window.py`
checks, on public files only, that it catches a swap and a 10 s shift. Log every run here.

### J2: final pipeline (Justin + Claude, `j2-final-pipeline`)

**What.** `katena.pipeline` and `scripts/make_submission.py` (`make submit`) train candidate F exactly as
pre-registered. Every constant comes from `katena.eval.final`, and nothing was tuned. Training uses all 30
external matches with nothing held out, the two benches included. The model is saved as
`work/final/model/bundle.pkl`, with its hashes in `manifest.json`. On the real match, `predict` computes
features from the real 5 Hz grid and touch columns from the real 25 Hz ball, blends the members at
w = 0.5, decodes with Viterbi at tau 2 and writes the CSV with LF line ends. It runs the official checks on
the bytes written and prints aggregates only. A crash prints the exception type and code lines only. No
real-match run yet: the data bundle is not on our machines.

**Training run on this Mac** (M-series arm64, 10 cores, 16 GB):

| Step | Result |
|---|---|
| Worlds | 330 touch files in 7.5 min (3 workers); every world's re-run corruption matched its cached features |
| Pool | 657,133 rows x 213 columns |
| LightGBM member | 9 s |
| TCN member | 189 s |
| Cache size | 16 GB |
| Hashes | gbdt `b2146150...`, tcn `456fc2c2...`, bundle `16eeaed0...` |

**Prediction smoke test, synthetic only.** A 199,319-frame synthetic match with two 45-minute halves was
labelled in 17 s, and the official validator accepts the file.

**Reproducibility, same machine: byte-identical from scratch.** A second cache rebuilt everything without
copying anything (corpus, worlds, train, 3 workers) in 18 min.
- All 150 corpus files (ground truth, pools, mild/target/severe features) are byte-identical to the
  bake-off cache's.
- The model hashes are identical: gbdt, tcn and `bundle.pkl`.
- `tests/test_pipeline.py` checks the same property on a synthetic corpus.
- Not yet tested on a second machine.

**Open for the team.**
- **Kickoff.** The graded second half starts before its kickoff, and those frames are D. F has to find
  that from the tracking. `predict` reports the D share over the first 60 s and 120 s of each half, an
  aggregate. If the second half's share is high, F already handles it. If it is low, build a kickoff
  detector from the tracking, never from a frame constant.
- **Cross-machine.** The two-machine hash test (J3) compares submission bytes, because torch's last bits
  differ by CPU.

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

### Final candidate F: result

Everything below was computed after the pre-registration commit `3921643`, with the code at `ae509cd`
(`scripts/run_final.py seed touch pools`, `gbdt`, `tcn`, `score gate`, `det`). Nothing was changed or tuned after a
score. External development matches only.

**Checks.**

- **Copied inputs.** All 308 (match, world) feature files were recovered: identical in-play rows, M0 predictions and
  degrader statistics. Nothing was rebuilt.
- **Re-featurization.** All 41 worlds of the first IDSSE, Metrica and SkillCorner match were re-featurized, and all 41
  are byte-identical to the copies.
- **Pools.** The 28 `d2j` pools rebuilt here equal `work-d2/pool_d2j` byte for byte.
- **The incumbent.** Re-decoding M2 with this code at tau 2 gives the stored M2's spellF1, spell counts and bootstrap
  flags on all 84 (fold, condition) pairs, for both `work-base/oof/full` and D2's `cur`. Their probabilities are
  byte-identical on all 84.
- **Deviation from the registered wording.** The registered check said "equal on every fold and condition", and it
  does not hold for the frame-level metrics. MacroF1, alive/dead F1 and team accuracy differ, by at most 2.4e-5, on 6
  IDSSE + Metrica folds per condition. The stored M2 was scored by `8537084`, which predates D5's block-edge fix
  (`5f72876`), and that fix moves only frame-level metrics. The gate reads spellF1 and the flags only, so it is
  unaffected. The 8-fold macroF1 of M2 moves by 2e-6 (target 0.867747 stored, 0.867749 re-decoded).

**Gate** (28 development folds, fixed tau 2, paired 95% bootstrap over possessions, 2000 draws).

| comparison | folds | noise | F | M2 | diff | 95% paired CI | wins | rule |
|---|---|---|---|---|---|---|---|---|
| F vs M2 | 28 | target | 0.6019 | 0.5861 | +0.0158 | [+0.0111, +0.0205] | 22/28 (79%) | **pass** |
| F vs M2 | 28 | severe | 0.5696 | 0.5475 | +0.0221 | [+0.0173, +0.0271] | 26/28 (93%) | not worse (passes the full rule too) |
| F vs M2 | 8 IDSSE + Metrica | target | 0.5991 | 0.5689 | +0.0302 | [+0.0218, +0.0388] | 7/8 | pass |
| F vs M2 | 8 IDSSE + Metrica | severe | 0.5426 | 0.5149 | +0.0277 | [+0.0186, +0.0366] | 8/8 | pass |

| noise | dataset | folds | F | M2 | diff | wins |
|---|---|---|---|---|---|---|
| target | IDSSE | 6 | 0.5670 | 0.5441 | +0.0229 | 5/6 |
| target | Metrica | 2 | 0.6955 | 0.6434 | +0.0521 | 2/2 |
| target | SkillCorner | 20 | 0.6030 | 0.5930 | +0.0100 | 15/20 |
| severe | IDSSE | 6 | 0.5107 | 0.4916 | +0.0191 | 6/6 |
| severe | Metrica | 2 | 0.6382 | 0.5849 | +0.0533 | 2/2 |
| severe | SkillCorner | 20 | 0.5804 | 0.5605 | +0.0199 | 18/20 |

At target-like, 5 of the 6 losing folds are SkillCorner. The worst loss is -0.0181, the median fold gains +0.0174.
At severe, the two losses (both SkillCorner) are at most -0.0018.

**Information only (none of it changes F).**

- **The members alone, 28 folds.**
  - LightGBM `FG` (touch + `d2j`): +0.0073 [+0.0024, +0.0122], 17/28 at target-like; +0.0068 [+0.0021, +0.0116],
    16/28 at severe.
  - TCN `FT` (`d2j`): +0.0100 [+0.0046, +0.0158], 17/28 at target-like; +0.0233 [+0.0174, +0.0291], 25/28 at severe.
  - Neither member passes at target-like on its own. The blend does, as D3 found: the members err on different folds.
  - On the 8 folds: `FG` +0.0150 (6/8) at target-like and +0.0072 (5/8, CI includes 0) at severe; `FT` +0.0280 (7/8)
    and +0.0328 (7/8).
  - The gains add up roughly. D1 alone (+0.0069), D3's ensemble (+0.0065) and `d2j` (+0.0019) sum to +0.0153;
    F gains +0.0158.
  - The TCN gains most from `d2j`: `FT` +0.0100 against D3's TCN alone at +0.0036. Part of that may be the heavier
    weight on dead-labelled matches recorded in the pre-registration. That part was not isolated.
- **D2's evaluation worlds.** F against M2's heads in the same world:

| world | folds | M2 | F | diff [95% CI] | wins | FG diff | FT diff |
|---|---|---|---|---|---|---|---|
| `mild` | 28 | 0.6116 | 0.6236 | +0.0121 [+0.0078, +0.0164] | 24/28 | +0.0100 | +0.0073 |
| `target` | 28 | 0.5861 | 0.6019 | +0.0158 [+0.0111, +0.0205] | 22/28 | +0.0073 | +0.0100 |
| `severe` | 28 | 0.5475 | 0.5696 | +0.0221 [+0.0173, +0.0271] | 26/28 | +0.0068 | +0.0233 |
| `target_ids` | 28 | 0.5844 | 0.6037 | +0.0194 [+0.0147, +0.0241] | 26/28 | +0.0075 | +0.0143 |
| `target_smooth` | 28 | 0.5796 | 0.6000 | +0.0204 [+0.0155, +0.0255] | 24/28 | +0.0145 | +0.0119 |
| `target_cpl50` | 8 | 0.5873 | 0.6240 | +0.0367 [+0.0271, +0.0460] | 8/8 | +0.0290 | +0.0298 |
| `target_cpl100` | 8 | 0.5951 | 0.6300 | +0.0349 [+0.0263, +0.0432] | 8/8 | +0.0330 | +0.0341 |
| `target_cpl100_ids` (never trained on) | 8 | 0.5900 | 0.6325 | +0.0424 [+0.0335, +0.0511] | 8/8 | +0.0359 | +0.0394 |

- **What the table shows.** F passes the adoption rule in every world. The independent target-like world, the one
  that decides, is its smallest gain.
- **macroF1, 8 IDSSE + Metrica folds.** At target-like: F 0.8781, FG 0.8721, FT 0.8754, M2 0.8677. At severe: F
  0.8549, FG 0.8493, FT 0.8486, M2 0.8432.
- **Spell counts.** Predicted possessions per true one, target-like, 28 folds: F 0.738 against M2 0.737. At severe:
  F 0.723 against M2 0.747.

**CPU** (i7-7700, 4 cores / 8 threads; LightGBM 8 threads, torch 4).

| stage | wall | CPU | peak memory |
|---|---|---|---|
| copy reused inputs (364 files, 11.2 GB) | 3.5 min | 0.1 min | 0.2 GB |
| touch columns, 308 (match, world) pairs, 2 workers, with all checks | 7.5 min | 14.1 min | 0.5 GB per process |
| `d2j` pools and touch pools | 2.3 min | 0.5 min | 0.3 GB |
| LightGBM member, 28 folds (fit 18-25 s per fold) | 13.2 min | 52.5 min | 2.0 GB |
| TCN member, 28 folds (300 sequences, 2.6 GB in float16; training 150-398 s per fold, median 206 s) | 104.1 min | 316.6 min | 3.7 GB |
| decode and score (F, FG, FT, M2 checks) | 1.1 min | 1.0 min | 0.2 GB |
| gate tables (bootstrap) | 2.6 min | 2.6 min | 0.2 GB |
| **pipeline total** | **134 min** | **387 min** | **3.7 GB** |
| determinism, one fold, per fresh process (x2) | 9.3 min | 13-17 min | 3.5 GB |

The TCN stage shared the CPU with the LightGBM stage and the determinism runs for its first 32 minutes, so its wall
time is pessimistic. The cache `D:/katena/work-final` is 16 GB, 11 GB of it the copied features.

**Determinism.** Two fresh processes trained both members on the 27 training matches of fold `DFL-MAT-J03WMX` and
decoded the fold. They gave byte-identical results for the LightGBM models, the TCN weights, both members' and the
blended probabilities, and the decoded labels at `target` and `severe`. The target-like blend hashes to
`ab47d2e6...` and its labels to `81adaa9b...`. Both processes also reproduce the LOMO run's LightGBM model, TCN
weights and probabilities for that fold byte for byte. This covers one machine only: Windows 10, i7-7700 (AVX2),
torch 2.15.0+cpu, LightGBM 4.7.0. D3's cross-machine outlook still applies: expect identical labels, not identical
probability bytes.

**Decision: adopt F.** At target-like on 28 folds it wins on 22 (79%), and the paired interval lies above 0
(+0.0158 [+0.0111, +0.0205]). At severe it is not worse: +0.0221 [+0.0173, +0.0271], 26/28. So it passes the
pre-registered rule, and it also passes on the 8 IDSSE + Metrica folds and in every D2 world. M2 is replaced as the
baseline by F.

**Not done, or not verified.**

- **The final training run.** It must train both members on all development matches plus the two benches. That needs
  the D2 worlds and touch columns for the benches, which nobody has built yet (D2's note). It also needs
  `katena.eval.final` turned into a training and inference entry point. The locked benches were not built, trained on
  or scored here.
- **The real match.** Inference there uses its real 25 Hz ball for the touch group, which no development world
  reproduces exactly (D1's jitter caveat). It is for the human-run alarm (`scripts/check_touch_target.py`).
- **Other machines.** Byte identity across machines is untested.
- **The pooled calibration (D2 review note).** It still applies to the coupled and smooth training worlds.

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
