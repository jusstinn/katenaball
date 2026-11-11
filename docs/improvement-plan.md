# Improvement plan after the strategy review (2026-11-09)

Justin + Claude, branch `claude/git-pull-analyze-strategy-e4504e`, `main` at `28450b4`. External matches and the
public label file only. Nothing here ran on the provided match. The screens below are exploratory: they were not
registered before they ran, and they decide nothing. They say what is worth a registered 28-fold gate.

## 1. Where F stands

- **The code is healthy.** `uv run pytest -q`: 225 passed, 14 skipped.
- **F reproduces on this Mac.** Re-decoded from the cached out-of-fold probabilities, the 8 IDSSE + Metrica folds
  give 0.5991 at target-like and 0.5426 at severe, the logged values. The LightGBM member refitted from the pool
  gives 0.5839 and 0.5221, also the logged values.
- **Nothing has run on the provided match yet**, apart from the loader check (16/16 on Nov 7). That is still the
  critical path, ahead of any model work below.

## 2. What limits F

F at target-like noise, the 8 IDSSE + Metrica folds, 2,454 true possessions. Computed from the out-of-fold
probabilities that `scripts/screen_tcn_seeds.py` cached, with throwaway probes that are not committed.

| True possession length | under 2 s | 2-4 s | 4-8 s | 8-16 s | 16-32 s | over 32 s |
|---|---|---|---|---|---|---|
| True possessions | 335 | 554 | 578 | 508 | 338 | 141 |
| Found at IoU 0.5 | 4% | 24% | 51% | 76% | 87% | 91% |
| Share of all misses | 26% | 35% | 24% | 10% | 4% | 1% |
| Predicted possessions of that length that are right | 24% | 54% | 70% | 78% | 85% | 84% |

- **Possessions under 4 s are 61% of the misses.** F predicts 0.69 possessions per true one.
- **Boundaries cost as much as misses.** F1 falls from 0.722 at IoU 0.3 to 0.459 at IoU 0.7. Of the true
  possessions, 61% are matched at IoU 0.3 and 38% still at 0.7. A possession that ends in a hand-over is predicted
  to end a median 0.9 s late, and 47% of them more than 1 s late, mostly because F swallows the short opponent
  spell that follows.
- **The team label is still where the points are.** With true team labels F scores 0.911. With true alive/dead it
  scores 0.661.
- **The decoder has nothing left.** On F's probabilities, no decoder change gains more than 0.002 over the current
  0.5991: tau from 1.5 to 6 (0.595 to 0.600), a bonus or penalty on team switches (0.596 to 0.601), on alive/dead
  switches (0.598 to 0.599), and the blend weight from 0.5 to 1 (0.597 to 0.599). The probabilities are the
  limit, so the plan works on the two members.

## 3. Findings

### 3.1 The LightGBM member is too small

Its parameters were fixed before the bake-off and never revisited: 150 trees at learning rate 0.1. One model of
1,200 trees per fold, read at each smaller size, same pool, seed and everything else
(`scripts/screen_gbdt_trees.py`, 8 folds, Viterbi at tau 2):

| LightGBM member alone | target | severe | mild | target_cpl100 | target_smooth | severe_smooth | mild_smooth |
|---|---|---|---|---|---|---|---|
| 150 trees (F's member) | 0.5839 | 0.5221 | 0.6409 | 0.6281 | 0.6011 | 0.5486 | 0.6454 |
| 300 trees | 0.5910 | 0.5392 | 0.6470 | 0.6395 | 0.6128 | 0.5618 | 0.6536 |
| 600 trees | 0.5946 | 0.5466 | 0.6557 | 0.6468 | 0.6191 | 0.5733 | 0.6640 |
| 1,200 trees | 0.6042 | 0.5511 | 0.6634 | 0.6545 | 0.6289 | 0.5732 | 0.6685 |
| 1,200 - 150 | +0.0203 | +0.0290 | +0.0225 | +0.0264 | +0.0278 | +0.0246 | +0.0231 |
| 95% paired CI | [+0.011, +0.029] | [+0.020, +0.038] | [+0.014, +0.032] | [+0.018, +0.036] | [+0.019, +0.037] | [+0.016, +0.034] | [+0.015, +0.032] |
| Folds won | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

Blended with the cached TCN (seed 0) exactly as F does:

| F-style blend | target | severe |
|---|---|---|
| 150 trees (F) | 0.5991 | 0.5426 |
| 600 trees | 0.6057 | 0.5561 |
| 1,200 trees | 0.6095 | 0.5610 |
| 1,200 - 150 [95% CI] | +0.0104 [+0.0049, +0.0159], 7/8 | +0.0185 [+0.0126, +0.0247], 8/8 |

- The member alone at 1,200 trees (0.6042) is above the whole of F (0.5991).
- macroF1 of the blend moves with it: 0.8837 against 0.8781 at target-like, 0.8626 against 0.8549 at severe.
- The curve has not flattened at 1,200. Larger sizes are untested: the laptop throttled on battery before a
  3,000-tree run got past its first fold.
- **Caveats.** Eight folds, four sizes looked at, no SkillCorner fold, nothing registered. The 20 SkillCorner
  folds are the honest test. LightGBM is the member that is byte-identical across our two machines, so growing it
  costs nothing in reproducibility. Training goes from 34 s to about 5 minutes on the Windows PC.

### 3.2 Smooth or rough positions: a one-sided bet that one bit would settle

D5, D2 and D1 all measured that the degrader's jitter is about 20 times rougher than real automatic tracking.
F's mix gives the smooth worlds one share in five. Same rows, orientations and labels, with every row taken from
its severity's `_smooth` world instead (LightGBM member alone, 8 folds):

| smooth-only mix minus `d2j` | target_smooth | mild_smooth | severe_smooth | target | severe | mild | target_cpl100 |
|---|---|---|---|---|---|---|---|
| 150 trees | +0.0143 (8/8) | +0.0129 (6/8) | +0.0124 (5/8) | -0.0883 | -0.1188 | -0.0686 | -0.0605 |
| 600 trees | +0.0215 (7/8) | +0.0140 (7/8) | +0.0030 (6/8) | -0.0852 | -0.1339 | -0.0728 | -0.0740 |

- If the real tracking is smooth, the `d2j` hedge costs about 0.015 to 0.02. If it is rough, training smooth-only
  costs about 0.09 to 0.13. Without knowing which, `d2j` is the right choice, and F made it.
- A human already has the answer on screen: `scripts/check_touch_target.py` prints the real ball's jitter beside
  the smooth and target-like references. Today that number is "an alarm, never a choice".
- RULES.md allows unsupervised use of the provided match and inspecting it to design features. Our stricter
  "inference only" stance rests on a second-hand report of the Nov 3 organiser email that nobody has read. Reading
  that email decides whether this bit may be used.

### 3.3 The public window's structure fits T = 2 s

One look at the public labels, with no prediction and no score (`scripts/probe_label_structure.py`). The window
has 17 possessions: 12% under 2 s, 35% under 4 s, median 7.6 s, 3.4 per minute, 71% of hand-overs direct.

| Where the public window falls among 5-minute windows of IDSSE labels | T = 0 | T = 1 | T = 2 | T = 3 | T = 5 |
|---|---|---|---|---|---|
| Median possessions per window (public: 17) | 23 | 21 | 17 | 15 | 12 |
| Share of windows with 17 or fewer | 0.20 | 0.29 | 0.52 | 0.77 | 0.95 |
| Share with as few possessions under 4 s | 0.11 | 0.23 | 0.48 | 0.66 | 0.77 |
| Share with a mean of 14.2 s or longer | 0.05 | 0.07 | 0.16 | 0.27 | 0.43 |

- Possessions under 2 s exist in the ground truth (0.8 s and 1.3 s), so it is not a heavily smoothed convention.
- Counts and short-spell shares sit in the middle of the T = 2 distribution. Mean and median length lean towards
  T = 3 to 5, because the window has little dead ball (20% against 45% in IDSSE). Only T = 0 looks unlikely.
- Seventeen possessions cannot separate T = 2 from T = 3. **Nothing here argues for changing T**, and T = 1
  (which passed the D4 gate) gets no support. No change proposed.

## 4. The plan

Ordered by value for the time it takes. Each model change needs its registration in the decision log before its
28-fold run, with F's rule: at target-like it wins at least 70% of the 28 folds and the paired 95% interval lies
above 0, and it is not worse at severe.

| # | What | Evidence | Expected on F | Cost | Owner |
|---|---|---|---|---|---|
| P0 | **Run F on the provided match now** and keep the file as the insurance submission: `docs/real-run.md` steps 1 to 3, including the three checks never run (`check_target_layout.py`, `check_touch_target.py`, `check_domain_gap.py`) | Nothing has run end to end on the real file. Every later step assumes it does | Removes the largest risk | 1 hour | Whoever holds the data |
| P0 | **Register every member under the same team name by Nov 10**, at least 3 members | PLAN.md | A team under 3 is invalid | Minutes | Everyone |
| G1 | **LightGBM member at 1,200 trees**, nothing else changed | Section 3.1 | +0.010 at target-like, +0.019 at severe (8 folds) | About 2 hours on the Mac, unattended: `scripts/night_g12.sh` retrains F's TCN for the 20 SkillCorner folds as well | Justin, registered Nov 9 |
| G2 | **A longer-trained TCN**: 4,800 steps instead of 1,200; second variant also reads the 53 touch columns | The TCN sees about one pass of its data, differs by 0.008 between two seeds, and never saw the touch group. Not measured yet | Unknown; +0.005 to +0.02 if it behaves like G1 | Screen on 8 folds: about 70 minutes per variant on the Mac. Gate of the variant that passes: 3 more hours, same run | Justin, registered Nov 9 |
| G3 | **A smooth-major training mix**, used only if the real tracking is smooth | Section 3.2 | +0.015 to +0.02 if smooth; not used otherwise | Half a day: three new worlds per severity (smooth with each coupling level and with identity switches), then the gate in the smooth worlds | David, after the email question |
| G4 | **Assemble F3 = F + whatever passed**, registered before its own score, as F was | F's own procedure | Sum of the parts | 2 to 8 hours on Windows | David |
| P5 | **Second-half kickoff**: only if `predict`'s report shows a low dead share at the start of the second half, label the frames before the tagged `Kick Off (Start)` as D | The trap is known; the tag gives the time with no frame constant | One possession and some macroF1 | 2 hours | Justin |

**G1 and G2 are registered** in the decision log (2026-11-09, "G1 and G2 (pre-registration)"), with one addition
to what was proposed here: G1 must also not be worse on the 20 SkillCorner folds alone, the only folds that did
not choose its size. One command runs both, on external matches only, and resumes if interrupted:

```bash
scripts/night_g12.sh
```

- It runs the G1 gate on 28 folds (about 2 hours), the G2 screen on 8 folds (about 2.5 hours) and, if a G2
  variant passes its screen, that variant's 28-fold gate (about 3 hours).
- It refuses to start on battery, in Low Power Mode, or while the registration and code are uncommitted.
- Tables go to `work/final/night_g12.log`; verdicts to `g1_gate.json` and `g2_report_8.json` beside it.
- G2's incumbent is whichever LightGBM member stands after G1's gate, so G2 is tested on top of G1 if G1 passes.

**G3, proposed rule.** Build `d2s`: the `d2j` worlds with smooth jitter in four shares of five and the rough
world in one. Gate it against `d2j` in `target_smooth` and its coupled versions. Use it for the final model only
if (a) it passes there, (b) the team has confirmed that an unlabelled statistic of the provided match may choose
between two training mixes, and (c) `check_touch_target.py` puts the real ball nearer the smooth reference than
the target-like one. The script would print that as one bit, as `check_domain_gap.py` does. Otherwise `d2j`
stays.

## 5. Not worth the remaining days

| Idea | Why not |
|---|---|
| Decoder work: tau, HSMM, switch costs, minimum lengths | Flat to 0.002 on F's probabilities (section 2), and HSMM and rescoring already failed |
| Changing T | Section 3.3 |
| Filling the ball from the broadcast tracking | Going from target-like to mild noise, which halves ball loss, is worth 0.02 to 0.04 on the benches. That caps the gain, for an unsolved alignment problem |
| More external matches | Four more training matches were worth 0.003 (bake-off, LOMO against grouped) |
| A second TCN seed | Dropped at the F2 screen. It lowers variance, not the mean |
| Smart Tagging as team features | No external match has tags, so no gate can test them. P5 uses the one tag that needs no test |

## 6. Calendar

| Day | Justin + Claude | David + Claude |
|---|---|---|
| Mon Nov 9 | P0 real run with whoever holds the data. Find the Nov 3 email. Overnight: `scripts/night_g12.sh` (G1 gate, G2 screen, G2 gate) | Review the G1 and G2 registration |
| Tue Nov 10 | **Registration closes.** G1 and G2 results into the decision log | Build the G3 worlds if the email allows |
| Wed Nov 11 | P5 if the report asks for it. Kernel-swap check if a G2 variant passed | G3 gate |
| Thu Nov 12 | **Insurance submission**: the best model that has passed so far | LightGBM hash of that model on Windows |
| Fri Nov 13 | Register F3 | F3 gate |
| Sat Nov 14 | **Freeze.** Final training on 30 matches, real run | Same on Windows |
| Sun Nov 15 | Cold-clone regeneration on both machines, hashes | Bench read of F3, report only |
| Mon Nov 16 | Technical document, validate, **submit** | Review, confirm the hash |
| Tue Nov 17 | Buffer (form closes 23:59 CET) | Buffer |

If G1 alone passes and everything else slips, F with 1,200 trees is the submission.

## 7. Questions only the team can answer

1. Has anyone read the Nov 3 organiser email? Its wording decides G3.
2. Who holds the data bundle today, and can they do the P0 run?
3. Are all members registered under the same team name?
