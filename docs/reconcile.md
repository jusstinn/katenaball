# Reconciling `origin/david/foundation` with `main`

Written 2026-11-07 for D5. Before Justin's main landed, David's side built a parallel pipeline on
`origin/david/foundation` (D1 loaders and labels, D2 noise simulator, D3 features, D4 model). It was
read with `git show` and run from an extracted copy outside the repository. It was not merged. This
page maps every module to its equivalent on main and records a decision for each:

- **superseded**: main has an equivalent that the bake-off validated; nothing to port.
- **port**: standalone value and no other owner; ported on `david/d5-audit`.
- **candidate**: an idea worth testing through the gate (`docs/PLAN.md`, "Validating blind"). The
  owner is named. Several are being ported right now by other agents.
- **do not port**: conflicts with the rules.

**Scores are not comparable.** The foundation's headline (two-head LightGBM, spellF1 0.752) is 6 LOMO
folds (5 IDSSE + Metrica Game 3) under David's own simulator, scored against labels harmonised at
3.5 s, with a different locked pair (J03WQQ and J03WR9). M2's 0.586 is 28 folds under
`katena.sim.degrade` at target-like noise against T = 2 s sandwich labels. Different noise, labels,
folds and decoder: neither number says anything about the other.

## Module map

### Shared format and loaders (D1)

| foundation | main | decision | reason |
|---|---|---|---|
| `katena/match.py` (`Match`, `from_normalised`) | `katena.types.Match` | superseded | Main's Match is the shared format (corner-origin metres, float fps, `extras`); David's uses centred metres and int fps. |
| `katena/io.py` (`load_target_match` via kloppy EPTS) | `katena.target.atd.load_target` | superseded | J1 owns the target loader. Its layout-awareness is kept as a check: `katena.target.layout` (ported idea, below). |
| `katena/synth.py` (plausible clean match) | `tests/synth.py` (tiny fixture) | candidate (D1, D2) | Only needed by the foundation's feature and noise tests; port it together with whichever of those tests the D1/D2 agents bring over. |
| `external/common.py` (half assembly, `detect_attack_dir`, speed, npz cache) | `katena.external.common`, `katena.types.save_match` | superseded | Both attack-direction methods agree on every half checked (cross-check below). |
| `external/idsse.py` | `katena.external.idsse` | superseded | Cross-checked on J03WMX and J03WN1: labels, positions, ball, speed and attack direction identical. Only the block convention differs (tracking extent vs kick-off to final-whistle events). |
| `external/idsse.py`: `owner_raw` (DFL names the restarting team on dead frames, 99.7% per David) | none | candidate (J5, D4) | Free supervision for "who restarts", which the tags and kick-off handling need. |
| `external/epts.py` (generic EPTS through kloppy) | `katena.external.metrica.build_game3`, `katena.target.atd` | superseded; idea ported | On Game 3 it reads exactly what main reads. On a synthetic EPTS pair it follows the DataFormatSpecification where main and the starter follow `<Player>` order; ported as `katena.target.layout` + `scripts/check_target_layout.py`. kloppy is not a main dependency, and its regex reader raises on a malformed line where main counts and skips it. |
| `external/metrica_open.py` (Game 3 labels from events) | `katena.external.metrica.derive_labels` | candidate (D4) | Its stoppages agree far better with Metrica's own ball removal: P(ball missing given D) 0.992 vs 0.880 for main on Game 3 (audit finding A3). |
| `external/harmonise.py` (`min_control_s` 3.5 s, restart protection, shortest first) | `katena.labels.sandwich` (T = 2 s) | candidate (D4, being ported) | A different label convention; training and evaluating at it is D4's sweep. |
| `external/control_evidence.py` (event evidence for the threshold) | none | candidate (D4, being ported) | The only external evidence for "a few seconds" of control. |
| `external/corpus.py` (build, cache, report) | `scripts/build_external.py`, `katena.eval.corpus` | superseded | Main's corpus has 30 matches against 8. |

### Noise (D2)

| foundation | main | decision | reason |
|---|---|---|---|
| `noise/sim.py` (`corrupt`, slot policies, identity switches) | `katena.sim.degrade.corrupt` | superseded as the training degrader; ideas are candidates (D2, being ported) | Identity-switch and slot-fragmentation noise is the D2 agent's port. |
| `noise/fit.py`: `fit_noise_cfg` (moment matching to the target) | none | **do not port** | Fits the degrader to the target. Banned until the organisers confirm (decision log, 2026-11-06). |
| `noise/fit.py`: `severity_grid` | `corpus.POOL_CONDS` (mild, target, severe) | candidate (D2) | Main already trains across three worlds; a wider or coupled grid is D2's work. |
| `noise/stats.py` (`target_stats`, label-free noise statistics) | none | candidate (D2), not ported | On the target its only use beyond the alarm below is moment matching, which is banned. Fine for comparing external worlds. |
| `noise/gap.py` (`domain_gap`, real-vs-sim window classifier) | `katena.diag.domain_gap` | **port (done)** | An aggregate-only alarm a human can run on the target. Adapted to main's coordinates and 5 Hz grid, speed statistics dropped, and a same-world reference AUC added. Returns no configuration. |
| `noise/_util.py` | inside `katena.diag.domain_gap` | port (the parts the alarm needs) | |

### Features (D3)

| foundation | main | decision | reason |
|---|---|---|---|
| `feats/core.py`, `ball.py`, `team.py`, `temporal.py`, `spec.py` (191 features) | `katena.featurize.build` (160 features) | superseded; new feature ideas are candidates (D1, being ported) | Main's set won the bake-off. Ideas without an equivalent on main (team de-duplication, mark distances, gap-aware hold positions) are the D1 agent's. |
| `feats/core.py`: `swap_teams` (home/away augmentation) | two focal orientations, averaged (`gbdt.Heads.predict`) | superseded | Main is side-symmetric by construction. |
| `feats/stress.py` (cheap fixed degradations) | `katena.sim.degrade` | superseded | Test helper only. |

### Model and decoder (D4)

| foundation | main | decision | reason |
|---|---|---|---|
| `model/lgbm.py` (two heads or 3-class, per-head temperatures) | `katena.models.gbdt.fit_heads` | superseded; 3-class and temperatures are candidates (model owner) | David's 3-class was +0.008 on 6 folds and did not pass his gate. Main tunes one decoder temperature instead. |
| `model/decode.py` (segment Viterbi, switch penalty, minimum dwell) | `katena.decode.hmm` | superseded; minimum dwell is a candidate (decoder owner) | Main's Viterbi uses fitted transitions; its exactness is now tested by brute force (ported test). |
| `model/data.py` (locked J03WQQ + J03WR9, `fill_missing_ball`, copies) | `katena.eval.protocol`, `katena.eval.corpus` | superseded; `fill_missing_ball` is a candidate (D2) | Metrica removes the ball on dead frames, which teaches "missing ball = dead". Main keeps the real NaNs and tops them up instead; which is right for the target is D2's coupling question. |
| `model/lomo.py` (nested calibration and decoder choice inside the training folds) | `katena.eval.lomo`, `bakeoff.decode_and_score` | superseded; the nesting is a recommendation | It is the clean version of main's tau selection, which leaks at second order (audit finding A6). |
| `model/baseline.py` (starter baseline on a Match) | `katena.models.baseline.m0_predict` | superseded | |

### Scripts, docs, tests

| foundation | main | decision | reason |
|---|---|---|---|
| `scripts/external_corpus.py` | `scripts/build_external.py` | superseded | |
| `scripts/feature_bench.py`, `feature_smoke.py` | none | superseded | Synthetic benchmarks of the old feature store. |
| `scripts/model_cache.py`, `model_lomo.py` | `scripts/run_bakeoff.py`, `scripts/run_m2.py` | superseded | |
| `scripts/noise_fit_target.py` | none | **do not port** | Fits the degrader to the target. Its domain-gap step lives on in `scripts/check_domain_gap.py --target`, which fits no degrader parameter and prints one alarm bit. |
| `scripts/train_model.py`, `predict_target.py` | J2 (not on main yet) | superseded | Justin owns the final pipeline. `predict_target.py` printed only aggregates and a file hash, a good pattern for J2. |
| `docs/external_data.md`, `docs/model.md` | `docs/external-data.md`, `docs/bakeoff.md` | superseded | Findings worth keeping are listed below. |
| `tests/test_external_*`, `test_io`, `test_match` | `tests/test_external.py`, `tests/test_target.py` | superseded | Loader behaviour is cross-checked on real external files instead (below). |
| `tests/test_model_decode.py` (brute-force decoder check) | none | **port (done, as an idea)** | `tests/test_d5_audit.py::test_viterbi_matches_brute_force` checks main's Viterbi. |
| `tests/test_noise_*`, `test_features_*` | none | candidate (D2, D1) | With the code they test. |
| `tests/test_noise_gap.py` | `tests/test_d5_audit.py` (domain gap) | port (done) | Same properties: same world indistinguishable, different worlds separated, aggregates only, deterministic, label-blind. |

## Cross-checks run (external data only)

| check | result |
|---|---|
| IDSSE J03WMX and J03WN1, both loaders | Labels inside main's blocks agree on 100% of 145,608 and 141,448 frames (no H/A/D disagreement). Player and ball positions agree to 8e-6 m after the coordinate map (main x = x + L/2, main y = W/2 - y), with identical NaN patterns; speed to 1e-6 m/s; attack direction identical in all 4 halves and equal to DFL's `TeamLeft`. Blocks: main starts 0-56 frames later and ends 0-201 frames earlier (events vs tracking extent). |
| Metrica Game 3, both loaders | Tracking identical (positions within 2e-5 m, ball NaN patterns identical, attack direction equal). Labels: 93.4% frame agreement, 99.3% on the team where both say alive, 93.9% on alive/dead. Main has 330 possessions (mean 11.3 s) and 34.6% dead, David 406 (9.8 s) and 31.1%. Scoring one as the other's prediction gives spellF1 0.878 (T = 0) and 0.877 (T = 2). On the 7,215 frames main calls D and David calls alive, the ball is present 77% of the time; on the 1,531 frames main calls alive and David D, it is missing on all of them. |
| Synthetic EPTS pair, `katena.target.atd` vs David's kloppy loader | Raw groups in `<Player>` order: positions, speed and ball identical, slot teams identical. Groups shuffled: main's slot teams are right for 33% of slots, David's for 100%. This is the assumption `scripts/check_target_layout.py` now checks on the real file. |

## Findings from the foundation docs worth keeping

From `docs/external_data.md` and `docs/model.md` on the branch; not re-measured here unless stated.

1. DFL names a team on dead frames, and on the last dead frame it is the restarting team 99.7% of the
   time (699 stoppages). Main maps those frames to D and drops the information.
2. DFL dead ball starts when the ball crosses the line: 549 of 811 onsets had the ball outside in the
   preceding 3 s, with a median lag of 0.16 s.
3. Metrica's open tracking removes the ball while it is dead. Confirmed here on Game 3 with David's
   labels: the ball is missing on 99.2% of dead frames and present on 98.0% of alive frames.
4. DFL halves start with the ball already alive at kick-off; Metrica Game 3 has 14 s and 17 s of
   pre-kick-off dead ball, like the target's second half.
5. DFL event clocks are offset from tracking by -1.2 to +1.7 s per match; any event-to-frame join
   needs a per-match shift.
6. Event evidence for "controlled" possession: the share of opponent flips with a deliberate action by
   the flipping team rises from about 10% under 1 s to 48% at 3-4 s and 90% at 6-10 s; the pooled
   accuracy-best threshold is 3.7 s (match bootstrap 90% interval 2.7-4.0 s). That evidence is the
   D4 agent's to port; note it points to a longer convention than main's T = 2 s.
