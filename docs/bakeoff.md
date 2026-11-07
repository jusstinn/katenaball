# Architecture bake-off: results

Protocol (pre-registered, with dated amendments): [bakeoff-protocol.md](bakeoff-protocol.md). Reproduce with
`uv run python scripts/run_bakeoff.py` (cached, resumable; `--fresh` starts over), then
`uv run python scripts/make_bakeoff_tables.py` to regenerate the tables below. All numbers are
`possession_metrics.evaluate` outputs (imported unchanged), rank metric `spell_f1_mean`, external matches
only (no `data/`, no `evaluation/gt/`), no commits. Seeds are fixed; LightGBM is deterministic on CPU.

## Summary

- **Ranking (28 development matches, LOMO, target-like noise, T = 2 s labels)**: M2 0.586 = M3 0.585 = M6 0.584
  (grouped) > M1 0.432 > M0 0.280. Ball-free features are the one component that clearly helps (+0.036, 14/14
  folds). HSMM, spell rescoring, dead-ball cues and ball-missingness features are not shown to help.
- **M5 (TCN)** beats M2 by +0.013 on 8 folds with seed 0 (adopted by the rule) and +0.010 with seed 1 (not adopted,
  CI touches 0). Not established; treat as a tie with a possible 0.01.
- **What limits the score is the team label, not dead ball or boundaries.** With perfect team labels M3 goes from
  0.57 to 0.90; with perfect alive/dead only to 0.64.
- **Label convention (T) moves the score more than any architecture choice beyond M0 -> M2**: the same model scores
  0.43 to 0.63 on IDSSE as the evaluation T goes from 0 to 5 s. The model ranking is stable across T. T cannot be
  selected from external data; training at T = 2 s has the smallest worst-case regret.
- **Locked benches (scored once)**: IDSSE `DFL-MAT-J03WQQ` M2/M3/M5 = 0.567/0.571/0.599; Metrica `Sample_Game_2`
  0.631/0.634/0.631 (target-like noise).

## Setup actually run

- 30 matches: 28 development (6 IDSSE, 2 Metrica, 20 SkillCorner) and 2 locked benches chosen by hash
  (`DFL-MAT-J03WQQ`) and by the brief (`Sample_Game_2`). Nothing trained or selected on the benches.
- 5 Hz working grid, predictions expanded to the native frames and scored natively. 160 features in five groups
  (ball 62, ball-free 61, dead-ball ball cues 13, dead-ball player cues 16, ball-missingness 8), computed for each
  team as the focal team (attack direction inferred from player positions; it equals the stored direction in all 141 match-by-condition pairs,
  including every corrupted condition) and averaged, so predictions are side-symmetric.
- Degrader (`katena.sim.degrade`): anonymous 46-slot fragments, player gaps, ghosts and duplicates, burst ball loss
  tied to congestion and ball speed, jitter. Constants from `scripts/calibrate_degrader.py` on SkillCorner and
  Metrica (see Amendment A1). Realised on the 7 IDSSE matches: mild 20% ball missing / 21.3 mean tracked objects,
  target-like 41% / 20.2, severe 55% / 18.2, against 41% and 20.5 in the public target description.
  SkillCorner and Metrica are topped up to the target missing fraction, they are never cleaned.
- Training rows: every training match under mild, target-like and severe (stride 4 at 5 Hz, random focal
  orientation per row). LightGBM 150 trees, 31 leaves, seed 0. Priors and temperature as in the protocol.
- Compute: about 2 h wall clock on the M5 (corpus build 25 min, LOMO heads about 30 min, label sweep 16 min, TCN
  20 min per seed, M6 12 min), well inside the 3 h budget; peak process RSS about 2 GB. The machine was under
  memory pressure from other applications throughout (swap in use), which slowed jobs but did not change results.

## Tables

<!-- AUTO:BEGIN -->
### Table 1. Ranked models, development matches, LOMO, T = 2 s

spellF1 is the mean over matches of the official `spell_f1_mean`; CI = bootstrap over ground-truth possessions. Dataset and severity columns are mean spellF1. `n` = matches scored (M4 variants: 14 folds; M5: 8 folds; M3g/M6: grouped folds, see Table 3).

| model | n | spellF1 (target-like) | 95% CI | macroF1 (IDSSE+Metrica) | IDSSE | Metrica | SkillCorner | native | mild | target | severe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2 | 28 | 0.586 | [0.575, 0.597] | 0.868 | 0.544 | 0.643 | 0.593 | 0.621 | 0.612 | 0.586 | 0.547 |
| M3 | 28 | 0.585 | [0.574, 0.596] | 0.868 | 0.544 | 0.641 | 0.592 | 0.620 | 0.611 | 0.585 | 0.546 |
| M6 | 28 | 0.584 | [0.572, 0.594] | 0.855 | 0.535 | 0.589 | 0.598 | 0.617 | 0.605 | 0.584 | 0.546 |
| M4_nomiss | 14 | 0.583 | [0.566, 0.599] | 0.867 | 0.537 | 0.654 | 0.607 | - | 0.619 | 0.583 | 0.540 |
| M3g | 28 | 0.582 | [0.571, 0.593] | 0.856 | 0.537 | 0.590 | 0.595 | 0.617 | 0.607 | 0.582 | 0.545 |
| M5 | 8 | 0.582 | [0.561, 0.606] | 0.874 | 0.560 | 0.649 | - | 0.639 | 0.611 | 0.582 | 0.535 |
| M5s1 | 8 | 0.579 | [0.558, 0.602] | 0.873 | 0.558 | 0.642 | - | 0.636 | 0.612 | 0.579 | 0.527 |
| M4_nodead | 14 | 0.579 | [0.562, 0.594] | 0.859 | 0.532 | 0.608 | 0.615 | - | 0.611 | 0.579 | 0.532 |
| M4_nofree | 14 | 0.548 | [0.533, 0.563] | 0.847 | 0.494 | 0.621 | 0.579 | - | 0.603 | 0.548 | 0.484 |
| M4_freeonly | 14 | 0.507 | [0.491, 0.522] | 0.817 | 0.456 | 0.507 | 0.559 | - | 0.518 | 0.507 | 0.473 |
| M1 | 28 | 0.432 | [0.423, 0.441] | 0.509 | 0.242 | 0.334 | 0.499 | 0.528 | 0.493 | 0.432 | 0.357 |
| M0 | 28 | 0.280 | [0.274, 0.286] | 0.473 | 0.173 | 0.306 | 0.310 | 0.412 | 0.367 | 0.280 | 0.213 |

### Table 2. Adoption ladder (target-like, paired bootstrap over possessions, 2000 draws)

| comparison | folds | n | mean diff | 95% paired CI | wins | adopted |
|---|---|---|---|---|---|---|
| M1 vs M0 | 28 LOMO | 28 | +0.1518 | [+0.1431, +0.1601] | 96% | yes |
| M2 vs M1 | 28 LOMO | 28 | +0.1539 | [+0.1442, +0.1637] | 100% | yes |
| M3 vs M2 | 28 LOMO | 28 | -0.0007 | [-0.0015, +0.0002] | 25% | no |
| M3 vs M4_nofree | 14 (ablation folds) | 14 | +0.0357 | [+0.0259, +0.0449] | 100% | yes |
| M3 vs M4_freeonly | 14 (ablation folds) | 14 | +0.0767 | [+0.0659, +0.0871] | 100% | yes |
| M3 vs M4_nodead | 14 (ablation folds) | 14 | +0.0054 | [-0.0011, +0.0114] | 64% | no |
| M3 vs M4_nomiss | 14 (ablation folds) | 14 | +0.0007 | [-0.0041, +0.0055] | 50% | no |
| M5 vs M3 | 8 (IDSSE+Metrica) | 8 | +0.0137 | [+0.0029, +0.0242] | 88% | yes |
| M5 vs M2 | 8 (IDSSE+Metrica) | 8 | +0.0132 | [+0.0024, +0.0234] | 75% | yes |
| M5s1 vs M2 | 8, second TCN seed | 8 | +0.0101 | [-0.0003, +0.0200] | 62% | no |
| M5s1 vs M5 | 8, seed 1 vs seed 0 | 8 | -0.0031 | [-0.0093, +0.0033] | 12% | no |
| M6 vs M3g | 28 grouped | 28 | +0.0012 | [-0.0005, +0.0028] | 57% | no |
| M3 vs M3g | 28, LOMO vs grouped training | 28 | +0.0030 | [-0.0005, +0.0064] | 50% | no |

### Table 2b. Decoder temperature tau: mean target-like spellF1 over the folds (selected per fold from the other folds)

| model | tau=0.25 | tau=0.5 | tau=1.0 | tau=2.0 | tau=4.0 |
|---|---|---|---|---|---|
| M2 | 0.535 | 0.567 | 0.583 | 0.586 | 0.582 |
| M3 | 0.533 | 0.568 | 0.582 | 0.585 | 0.581 |
| M5 | 0.476 | 0.538 | 0.563 | 0.579 | 0.582 |
| M3g | 0.530 | 0.564 | 0.579 | 0.582 | 0.577 |

### Table 3. Same models restricted to the folds they were run on (spellF1 / macroF1)

| model (spellF1 / macroF1) | native | mild | target | severe |
|---|---|---|---|---|
| M0 | 0.362 / 0.488 | 0.282 / 0.488 | 0.206 / 0.473 | 0.133 / 0.428 |
| M2 | 0.637 / 0.901 | 0.612 / 0.889 | 0.569 / 0.868 | 0.515 / 0.843 |
| M3 | 0.637 / 0.901 | 0.611 / 0.889 | 0.568 / 0.868 | 0.513 / 0.843 |
| M5 | 0.639 / 0.901 | 0.611 / 0.892 | 0.582 / 0.874 | 0.535 / 0.850 |
| M5s1 | 0.636 / 0.901 | 0.612 / 0.891 | 0.579 / 0.873 | 0.527 / 0.846 |

(8 IDSSE + Metrica folds above.)

| model (spellF1 / macroF1) | mild | target | severe |
|---|---|---|---|
| M3 | 0.621 / 0.752 | 0.584 / 0.735 | 0.536 / 0.717 |
| M4_nofree | 0.603 / 0.745 | 0.548 / 0.720 | 0.484 / 0.690 |
| M4_freeonly | 0.518 / 0.699 | 0.507 / 0.695 | 0.473 / 0.679 |
| M4_nodead | 0.611 / 0.748 | 0.579 / 0.731 | 0.532 / 0.710 |
| M4_nomiss | 0.619 / 0.752 | 0.583 / 0.735 | 0.540 / 0.717 |

(14 ablation folds above.)

| model (spellF1 / macroF1) | native | mild | target | severe |
|---|---|---|---|---|
| M3g | 0.617 / 0.655 | 0.607 / 0.651 | 0.582 / 0.638 | 0.545 / 0.626 |
| M6 | 0.617 / 0.655 | 0.605 / 0.651 | 0.584 / 0.638 | 0.546 / 0.626 |

(28 grouped folds above.)


### Table 3b. Label-semantics sweep (spellF1 on 6 IDSSE development matches, target-like; rows = training T, columns = evaluation T)


Design A: LOMO, all other development matches train (pre-registered)

| train T \ eval T | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| 0 s | 0.435 | 0.482 | 0.546 | 0.584 | 0.596 |
| 1 s | 0.439 | 0.486 | 0.547 | 0.585 | 0.592 |
| 2 s | 0.427 | 0.475 | 0.542 | 0.590 | 0.612 |
| 3 s | 0.416 | 0.462 | 0.532 | 0.586 | 0.618 |
| 5 s | 0.379 | 0.425 | 0.502 | 0.566 | 0.629 |

Same models on the 2 development Metrica matches (event-derived labels; eval T = sandwich applied to those labels, 0 = raw)

| train T \ eval T | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| 0 s | 0.583 | 0.599 | 0.649 | 0.684 | 0.661 |
| 1 s | 0.585 | 0.602 | 0.656 | 0.690 | 0.666 |
| 2 s | 0.568 | 0.586 | 0.641 | 0.698 | 0.694 |
| 3 s | 0.559 | 0.576 | 0.635 | 0.691 | 0.688 |
| 5 s | 0.508 | 0.525 | 0.588 | 0.648 | 0.706 |

Design B: IDSSE and SkillCorner rows only, no Metrica rows (Amendment A3)

| train T \ eval T | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| 0 s | 0.431 | 0.478 | 0.543 | 0.580 | 0.595 |
| 1 s | 0.433 | 0.481 | 0.544 | 0.582 | 0.593 |
| 2 s | 0.424 | 0.472 | 0.540 | 0.585 | 0.608 |
| 3 s | 0.413 | 0.460 | 0.534 | 0.584 | 0.619 |
| 5 s | 0.377 | 0.423 | 0.502 | 0.565 | 0.629 |

Same models on the 2 development Metrica matches (event-derived labels; eval T = sandwich applied to those labels, 0 = raw)

| train T \ eval T | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| 0 s | 0.539 | 0.556 | 0.604 | 0.636 | 0.613 |
| 1 s | 0.536 | 0.551 | 0.601 | 0.637 | 0.613 |
| 2 s | 0.524 | 0.542 | 0.595 | 0.641 | 0.623 |
| 3 s | 0.510 | 0.526 | 0.582 | 0.639 | 0.640 |
| 5 s | 0.465 | 0.481 | 0.539 | 0.602 | 0.659 |

No-training references per evaluation T (IDSSE / Metrica)

| model | eval T=0 | eval T=1 | eval T=2 | eval T=3 | eval T=5 |
|---|---|---|---|---|---|
| M0 | 0.170 / 0.302 | 0.177 / 0.305 | 0.173 / 0.306 | 0.161 / 0.290 | 0.133 / 0.255 |
| M1 | 0.197 / 0.295 | 0.216 / 0.301 | 0.242 / 0.334 | 0.252 / 0.347 | 0.255 / 0.315 |

### Table 4. Severity and ball-loss/dead-ball coupling, IDSSE development matches (6)

| model (spellF1 / macroF1) | native | mild | target | severe | target_dead3 | target_dead15 |
|---|---|---|---|---|---|---|
| M0 | 0.337 / 0.391 | 0.246 / 0.396 | 0.173 / 0.389 | 0.115 / 0.353 | 0.182 / 0.425 | 0.199 / 0.463 |
| M1 | 0.452 / 0.696 | 0.348 / 0.622 | 0.242 / 0.541 | 0.162 / 0.475 | 0.253 / 0.527 | 0.258 / 0.499 |
| M2 | 0.622 / 0.909 | 0.591 / 0.894 | 0.544 / 0.870 | 0.492 / 0.848 | 0.556 / 0.879 | 0.556 / 0.880 |
| M3 | 0.623 / 0.910 | 0.591 / 0.894 | 0.544 / 0.870 | 0.491 / 0.848 | 0.556 / 0.879 | 0.555 / 0.880 |
| M4_nomissw | - | - | 0.537 / 0.869 | - | 0.557 / 0.878 | 0.558 / 0.881 |

### Table 5. Oracles (IDSSE and Metrica development matches, M3, target-like)

| oracle (spellF1 / macroF1, target-like) | IDSSE | Metrica | both |
|---|---|---|---|
| M3 as is | 0.544 / 0.870 (n=6) | 0.641 / 0.860 (n=2) | 0.568 / 0.868 (n=8) |
| perfect ball positions (IDSSE only) | 0.615 / 0.905 (n=6) | - | 0.615 / 0.905 (n=6) |
| perfect team labels, predicted alive/dead | 0.902 / 0.958 (n=6) | 0.886 / 0.907 (n=2) | 0.898 / 0.946 (n=8) |
| perfect alive/dead, predicted team | 0.604 / 0.905 (n=6) | 0.744 / 0.946 (n=2) | 0.639 / 0.915 (n=8) |
| perfect boundaries, predicted team per spell | 0.655 / 0.932 (n=6) | 0.808 / 0.971 (n=2) | 0.694 / 0.942 (n=8) |
| perfect alive/dead and team | 0.997 / 0.996 (n=6) | 0.995 / 0.997 (n=2) | 0.996 / 0.996 (n=8) |

### Table 6. Error types of unmatched ground-truth possessions (IoU 0.5), M3, target-like

| dataset | GT spells | ok | wrong_team | dead_miss | split | merged | boundary | mixed | pred/GT spells | spurious in dead | spurious elsewhere | GT-dead frames called alive | GT-alive frames called dead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idsse | 1944 | 45.3% | 24.2% | 7.7% | 0.7% | 13.3% | 7.9% | 0.8% | 0.65 | 34 | 139 | 0.041 | 0.045 |
| metrica | 510 | 60.6% | 18.0% | 1.8% | 0.6% | 11.6% | 7.3% | 0.2% | 0.83 | 47 | 22 | 0.190 | 0.018 |
| skillcorner | 5434 | 53.3% | 28.5% | 0.0% | 1.0% | 8.9% | 6.0% | 2.3% | 0.76 | 0 | 791 | 0.000 | 0.000 |

### Table 7. Is a missing ball a dead-ball cue? (labels of each dataset)

| dataset | condition | P(ball missing | dead label) | P(ball missing | alive label) |
|---|---|---|---|
| idsse | native | 0.000 | 0.000 |
| idsse | target | 0.345 | 0.465 |
| idsse | target_dead3 | 0.471 | 0.360 |
| idsse | target_dead15 | 0.636 | 0.222 |
| metrica | native | 0.791 | 0.015 |
| metrica | target | 0.808 | 0.111 |

### Table 8. Locked benches (scored once, trained on all 28 development matches)

| model (spellF1 / macroF1) | DFL-MAT-J03WQQ target | DFL-MAT-J03WQQ mild | DFL-MAT-J03WQQ severe | DFL-MAT-J03WQQ native | Sample_Game_2 target | Sample_Game_2 mild | Sample_Game_2 severe | Sample_Game_2 native |
|---|---|---|---|---|---|---|---|---|
| M0 | 0.184 / 0.415 | 0.265 / 0.419 | 0.112 / 0.377 | 0.368 / 0.416 | 0.330 / 0.769 | 0.379 / 0.793 | 0.203 / 0.703 | 0.432 / 0.813 |
| M1 | 0.245 / 0.460 | 0.316 / 0.494 | 0.169 / 0.425 | 0.418 / 0.566 | 0.334 / 0.416 | 0.384 / 0.424 | 0.247 / 0.370 | 0.393 / 0.435 |
| M2 | 0.567 / 0.885 | 0.613 / 0.904 | 0.548 / 0.863 | 0.655 / 0.920 | 0.631 / 0.884 | 0.662 / 0.886 | 0.537 / 0.845 | 0.678 / 0.894 |
| M3 | 0.571 / 0.886 | 0.610 / 0.904 | 0.549 / 0.863 | 0.650 / 0.919 | 0.634 / 0.883 | 0.656 / 0.885 | 0.534 / 0.845 | 0.678 / 0.894 |
| M4_nofree | 0.531 / 0.862 | 0.575 / 0.881 | 0.454 / 0.815 | 0.643 / 0.916 | 0.600 / 0.876 | 0.662 / 0.886 | 0.524 / 0.828 | 0.683 / 0.897 |
| M4_freeonly | 0.468 / 0.837 | 0.519 / 0.845 | 0.444 / 0.809 | 0.504 / 0.850 | 0.487 / 0.820 | 0.497 / 0.819 | 0.442 / 0.799 | 0.482 / 0.817 |
| M4_nodead | 0.573 / 0.886 | 0.612 / 0.906 | 0.526 / 0.851 | 0.651 / 0.925 | 0.627 / 0.878 | 0.641 / 0.884 | 0.515 / 0.838 | 0.669 / 0.889 |
| M4_nomiss | 0.577 / 0.883 | 0.612 / 0.905 | 0.547 / 0.861 | 0.652 / 0.922 | 0.657 / 0.886 | 0.652 / 0.884 | 0.527 / 0.846 | 0.679 / 0.895 |
| M5 | 0.599 / 0.894 | 0.614 / 0.906 | 0.557 / 0.874 | 0.637 / 0.921 | 0.631 / 0.883 | 0.667 / 0.886 | 0.564 / 0.854 | 0.687 / 0.894 |
<!-- AUTO:END -->

## Findings

### Model comparison (protocol section 6)

1. **M0 -> M1 -> M2 are real, large steps** (+0.15 each, every interval far from 0). M1 (rules with 1 s hand-over
   dwell, hold through gaps, simple dead rule) is already 1.5x M0; the learned model is another 1.35x.
2. **HSMM (M3) does not beat Viterbi (M2)**: -0.0007, 25% wins. With posteriors from a model that already sees
   +-8 s context, the extra duration prior adds nothing. PLAN.md's +0.07 to +0.11 HSMM gain came from a
   simulator whose prior matched its generator; it does not transfer to measured labels. Viterbi is the cheaper
   and equal choice. (tau between 1 and 4 is a plateau, Table 2b; the plateau is flat to 0.004.)
3. **Ball-free features: keep.** Removing them costs 0.036 spellF1 at target-like, 0.052 at severe, 0.018 at mild
   (14/14 folds each). Using them alone (no ball at all, including no ball-based dead cues) reaches 0.507, 87% of
   the full model, with 41% of the ball missing; this is the strongest evidence in the bake-off for the
   ball-free direction.
4. **Dead-ball cues: not shown** (+0.005, CI [-0.001, +0.011], 9/14 wins). **Missingness features: not shown**
   (+0.0007). Both are harmless to keep. The dead-ball cue result is a statement about spellF1; dead-ball cues
   still matter for the tie-break (alive/dead F1), which we did not isolate.
5. **TCN (M5)**: +0.013 over M2 on the 8 IDSSE+Metrica folds with seed 0 (7 of 8 wins vs M3, 6 of 8 vs M2, rule
   passed) but +0.010 with seed 1 (5 of 8 wins, CI [-0.0003, +0.020], rule failed); the two seeds differ by 0.003.
   It was not run on SkillCorner. Gain concentrates in two matches (`J03WN1`, `J03WR9`). Not adopted as the
   default: a second library with a determinism risk for byte-identical regeneration, for a gain inside the
   scorer's own noise band (0.04). It is a reasonable second ensemble member, which we did not test.
6. **Spell rescoring (M6): negative.** +0.0012 over its own grouped baseline, CI includes 0, 57% wins. The base
   model's errors are mostly confident wrong-team spells, which spell-level features of the same posteriors
   cannot see.
7. **LOMO vs grouped training (M3 vs M3g)**: -0.003 from training on 23 instead of 27 matches; a rough size of
   the data-volume effect.

### Label semantics (T)

Tables 3b (design A pre-registered; design B cross-provider, Amendment A3).

- **The convention dominates the absolute score.** One model, 6 IDSSE matches: 0.43 (T = 0) -> 0.55 (2 s) -> 0.61
  (5 s) as the *evaluation* labels get smoother. Models trained at the wrong T lose up to 0.06 (train 5 s,
  evaluate 0 s) and at most 0.037 in the other direction (train 0-1 s, evaluate 5 s).
- **Minimax regret** over the evaluation Ts 0-5 s: train T = 2 s: 0.017; T = 3 s: 0.024; T = 0: 0.033; T = 1: 0.037;
  T = 5: 0.061. T = 2 s is the choice that is least wrong under every convention, which is what we used.
- **Cross-provider (Metrica, event-derived)**: with Metrica rows removed from training (design B), the model
  trained at T = 0 is best on raw Metrica labels (0.539) and T = 5 is worst (0.465): over-smoothed training
  labels hurt on a different provider's convention by 0.07. Metrica's own labels are smooth (mean spell 11 s)
  yet still prefer light merging, so "smoother is better" is not what the cross-provider data say.
- **Ranking is stable across T**: M3 > M1 > M0 at every evaluation T on both IDSSE and Metrica, in both designs.
- **Can T be chosen without the competition GT? No.** Nothing in the external corpus identifies the competition
  convention. The only non-GT information is the public description (a mean spell of 14.2 s in the public
  window, which on IDSSE corresponds to T near 5 s) and the written "touch without control does not change
  possession", which is not a duration. We did not use the public window. A published-baseline check does not
  settle it either: M0 on degraded IDSSE gets spellF1 0.17 (T <= 3) versus the published 0.2433 on the real
  target, and macroF1 0.39 versus 0.6724, so the simulator is not a faithful stand-in for M0 and cannot arbitrate T.
  Recommendation: train at T = 2 s (robust), keep the duration/temperature knobs not tuned to one T, and treat
  the absolute spellF1 as uncertain by about +-0.1 from the convention alone.

### Oracles (Table 5) and what limits the score

| oracle (M3, target-like, IDSSE+Metrica, 8 folds) | spellF1 | gain |
|---|---|---|
| as is | 0.568 | |
| perfect ball positions (IDSSE only, 6 folds: 0.544 -> 0.615) | 0.615 | +0.07 |
| perfect alive/dead, predicted team | 0.639 | +0.07 |
| perfect boundaries, predicted team per spell | 0.694 | +0.13 |
| perfect team labels, predicted alive/dead | 0.898 | +0.33 |
| perfect team and alive/dead (sanity) | 0.996 | |

Error census (Table 6; unmatched GT possessions at IoU 0.5): wrong team 24% (IDSSE), 18% (Metrica), 29%
(SkillCorner); merged GT possessions 13% / 12% / 9%; boundary offset 8% / 7% / 6%; dead-ball confusion 7.7% /
1.8% (SkillCorner has no D); splits about 1% or less. The model under-segments: 0.65 / 0.83 / 0.76 predicted
possessions per true one. In dead-ball frames: 4.1% (IDSSE) and 19% (Metrica) of GT dead frames are called alive.
**The loss is mostly the team assigned to a spell**, then merged short possessions, then ball positions, with
alive/dead the smallest. PLAN.md also listed wrong team as the largest source; the measurement puts its share at about a quarter of true possessions.

### Does a missing ball predict a dead ball? (Table 7)

- **Real data, Metrica**: ball NaN on 79% of dead-label frames and 1.5% of alive-label frames (73-88% and 0.8-4.3%
  across the three games). In the Metrica sample the missing ball *is* a dead-ball signal. Caveat: those D
  labels are event-derived and dead spans are long, and a different pipeline may drop the ball when it is out of
  play.
- **Degrader (independent world)**: 34.5% vs 46.5%, i.e. the wrong sign, because the hazard follows congestion and
  speed and a stationary ball in a dead period scores low on both. This is an artefact of the degrader's
  construction, as the brief suspected.
- **SkillCorner cannot answer it**: no D label; null possession is not a dead label.
- **What the target looks like is not identifiable from this corpus.** The published baseline treats missing as D and
  gets macroF1 0.67; our independent world gives M0 macroF1 0.39, the dead-coupled worlds 0.43 / 0.46, Metrica
  development matches 0.72 (target-like) and 0.78 (native). The target sits between, which suggests some coupling, but this mixes with the label-convention
  uncertainty and is not a measurement.
- **Consequence for the models**: trained in the independent world, M3 gains little from coupling (IDSSE 0.544 ->
  0.556 / 0.555 under odds 3 / 15) and loses nothing by dropping missingness features (`M4_nomissw` 0.537 / 0.557 /
  0.558 vs M3 0.544 / 0.556 / 0.555). A model trained across coupling levels might gain if the target is
  Metrica-like; we did not train one, and cannot check it without the target.

### Severity and near-clean warning (Tables 1 and 4)

M3 on IDSSE: native 0.623, mild 0.591, target-like 0.544, severe 0.491; pooled 0.620 / 0.611 / 0.585 / 0.546.
**Near-clean inputs would mislead**: they flatter M3 by +0.08 on IDSSE, flatter M0 by +0.16 (0.337 vs 0.173), hide the
value of ball-free features (the ablation gap is 0.018 at mild and 0.052 at severe, and presumably near 0 on clean
input), and shrink the gap between M1 and M2. A bench of near-clean data would choose a ball-reliant design.
Expected score on a noisier target: about 0.05 lower per step from target-like to severe; severe is 0.49
(IDSSE) to 0.55 (pooled).

### Locked benches (Table 8; scored once, after all selections)

Trained on all 28 development matches. Target-like: IDSSE `DFL-MAT-J03WQQ` M0 0.184, M1 0.245, M2 0.567, M3 0.571,
M5 0.599; Metrica `Sample_Game_2` M0 0.330, M1 0.334, M2 0.631, M3 0.634, M5 0.631. The development-set means for
the same datasets (0.544 IDSSE, 0.641 Metrica) are consistent with these, so the selection did not overfit
the development folds. The bench ordering matches the development ordering, including the ball-free ablation
(`nofree` -0.04 on IDSSE, -0.03 on Metrica; `freeonly` -0.10 / -0.15).

### Smart Tagging imitation (not built)

Events could simulate tags: SET PIECE events give restart intervals, event teams give the phase team, throw-ins
get N/A as in DATA.md; noise would be timing jitter (1-2 s), dropped or spurious tags, and flipped teams. We did
not build it because (a) the only use is the end of a dead stretch, and Table 5 caps perfect alive/dead at +0.07
spellF1, so realistic tags are worth at most about +0.03; (b) with no external instance of the target's tags, the noise
rates would be our own assumptions, so a simulated test measures nothing about the real tags. The first useful
step is to run the structure on the target tags (human-run, aggregates only) and compare restart times with
the model's predicted dead-ball ends.

### Not done, and failures

- **M7 PathCRF: not run.** Desk assessment only (README and abstract). It expects a fixed set of 22 identified player
  trajectories; our anonymous input is up to 46 fragmented slots, and re-linking fragments into identities is itself a
  research task. The released weights were trained on all 7 IDSSE matches, so any IDSSE score, including our locked
  bench, would be contaminated; only Metrica and SkillCorner could test it fairly. Not forced.
- M5 only on 8 folds, M6 on 6 grouped folds (not LOMO), M4 on 14 folds (Amendments A6, A7). TCN not run on SkillCorner.
- The degrader does not reproduce the published M0 numbers on the real target (spellF1 0.17-0.20 vs 0.2433, macro
  0.39 vs 0.67), and the calibration sources disagree (Metrica: ball loss rises with congestion and speed;
  SkillCorner: it falls). Absolute levels carry that uncertainty; rankings were stable across every world we tried.
- `src/katena/features/` could not be created (`.claude/settings.json` denies that path); the package is
  `src/katena/featurize/`. LightGBM needs a libomp re-exec on this machine (`ensure_openmp`).
- Tau for match k was chosen from the other matches' out-of-fold predictions, which were produced by models that saw
  match k's training rows; this leaks only through the decoder temperature (a three-way plateau), not through labels.
- SkillCorner scoring replaces predicted D with X (the scorer then repeats the last team). Its macroF1 is not
  meaningful and is excluded from Table 1's macro column.
- Positional detail: the `T` and tau knobs were not tuned to the target; no tuning used the public window.

## Addendum: focus probes after the bake-off (2026-11-06)

Run after the results above, on development matches only. The locked benches were not touched. Reproduce with
`uv run python scripts/probe_focus.py` and `uv run python scripts/probe_clean_ceiling.py`. A re-score of M2 from
the cached probabilities matches Table 1 exactly (0.5861, largest per-match difference 0).

**"Wrong team" is mostly short possessions.** Share of true possessions matched at IoU 0.5 (M2, target-like, T = 2 s):

| true spell length | IDSSE | Metrica | SkillCorner |
|---|---|---|---|
| under 2 s | 2% | 23% | 22% |
| 2-4 s | 20% | 23% | 24% |
| 4-8 s | 47% | 55% | 50% |
| 8-16 s | 71% | 76% | 66% |
| 16-32 s | 84% | 86% | 81% |
| over 32 s | 88% | 96% | 92% |

Spells under 8 s are 75-85% of all unmatched true possessions but only 14-24% of possession time. Table 6's
"wrong team" category is largely these: a short spell the model never segmented takes the surrounding team.

**Noise is not the only ceiling.** The same features, decoder and folds, trained and tested on clean input:

| | clean-trained on clean | noise-trained on clean | noise-trained on target-like |
|---|---|---|---|
| IDSSE (6) | 0.657 | 0.622 | 0.544 |
| Metrica (2) | 0.756 | 0.681 | 0.641 |

With identities and a perfect ball, IDSSE still predicts only 0.71 possessions per true one. Noise costs about 0.11;
the rest is lost even on clean input. The features are computed after downsampling to 5 Hz, so a touch or a
sharp change in the ball's direction is mostly gone before the model sees it. Part of the clean gap may be label noise,
so this does not say how much is recoverable. Next test: features computed at the native frame rate.

**Correction to "Smart Tagging imitation" above.** Tags are not only about the end of a dead stretch: 602 of 638 carry
a team (`docs/upstream/DATA.md`). The `DEFENDING IN ...` and `DEFENSIVE TRANSITION` tags name the team *without* the
ball (142 tags), so a direct mapping is backwards on those. A crude test simulated team tags from external labels at the published
density, with random placement and a median length of 6 s (our assumption), and added them to the team log-odds:

| tag accuracy | gain, errors independent of the model | gain, errors concentrated where the model errs |
|---|---|---|
| 60% | +0.003 at best | -0.001 at best |
| 70% | +0.004 at best | -0.003 at best |
| 80% | +0.012 at best | +0.003 at best |
| 90% | +0.023 at best | +0.015 at best |

Best of four fusion weights, chosen after the fact, over 28 folds, so these flatter the tags. It uses no information from where
real tags start and end, and a learned tag feature could do better. An automatic tagger reading the same
tracking is likely to err where we err, so the right column is the more realistic. Tags stay a cheap, timeboxed
experiment, not the main bet. One rival reports +0.086 from imitation tags, which is self-reported and not comparable.

## Recommended architecture

**M2: LightGBM alive/dead and team heads on ball-aware + ball-free + dead-ball + missingness features at 5 Hz,
both focal orientations averaged, 3-state Viterbi with transition priors fit on external labels, trained at T = 2 s
on all 30 matches under mild/target-like/severe corruption.** No HSMM, no spell rescoring. Reasons: it is the
simplest design that no alternative beat by the pre-registered rule (M5 passed only on one seed), it is
deterministic, and it keeps ball-free features, which are the only component whose value grows with noise.

Expected spellF1 on the target (my judgement, not a measurement): **central 0.55, plausible 0.45-0.62**. Anchors:
0.585 pooled / 0.544 IDSSE target-like, 0.49 IDSSE severe, benches 0.57 and 0.63, convention +-0.1.
What could make it wrong: (1) the competition GT convention, which can move the score by 0.1-0.2 either way
and is not identifiable here; (2) the degrader not matching real ATD (it does not match M0's published score);
(3) dead-ball coupling of the target, where a missing-ball-aware model could do better than we measured;
(4) player fragmentation in the real ATD being worse than 46 slots implies (identity switches); (5) team
mapping and pitch-scale errors in the real file; (6) the corrected-tracking ground truth defining "owning"
differently from DFL's possession field.

Next experiments (all external-only): train with ball loss coupled to dead ball at mixed levels; a TCN +
GBDT ensemble; run TCN on SkillCorner; a team-label-focused head (the oracle says this is where the points are),
for example spell-level features that use both teams' paths; test the sensitivity to T-aware duration priors.

## Tests

`DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.11/site-packages/sklearn/.dylibs uv run python -m pytest tests -q`:

```
........................................................................ [ 97%]
..                                                                       [100%]
74 passed in 4.01s
```

Without that variable (LightGBM cannot load libomp), `uv run python -m pytest tests -q`:

```
........................s............................................... [ 97%]
..                                                                       [100%]
73 passed, 1 skipped in 3.05s
```
