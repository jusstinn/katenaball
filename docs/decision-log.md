# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-07

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
- **The Metrica-like world matches the whole fingerprint.** `target_cpl100` comes within 0.03 to 0.04 on
  all eight published numbers. It also gets the published order of the windows: macroF1 peaks at
  smooth25 and spellF1 at smooth50. The independent world peaks spellF1 at smooth125.
- **Caveats.** This is consistent with the real ATD dropping the ball through stoppages, but it is not a
  measurement. It also depends on the target's label convention, its dead share, and 5 Hz against
  25 Hz. On Metrica's two development games every world already looks like this (smooth50 about
  0.72 / 0.30), because their own ball loss is coupled.
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
