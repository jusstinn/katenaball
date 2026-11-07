# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-07

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
