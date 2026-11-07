# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-07

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
`scripts/check_domain_gap.py`; aggregates only, fits nothing) and a brute-force Viterbi test. Not ported: `fit_noise_cfg`
and `noise_fit_target.py`, which fit the degrader to the target. Everything else is superseded by main or is a candidate
for its owner (D1 features, D2 identity-switch noise, D4 harmonisation and control evidence).

**Domain-gap alarm on external data:** real SkillCorner and Metrica tracking are separated from every degraded world
with AUC 1.00, mainly by positional jitter (median second difference 0.04-0.05 m against 0.47-1.38 m). A D2 candidate:
calibrate jitter on SkillCorner, through the gate. Not run on the target.

**Final training (D5 second half): blocked.** `origin/main` has no J2 training script yet.

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
