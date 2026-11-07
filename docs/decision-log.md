# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

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
