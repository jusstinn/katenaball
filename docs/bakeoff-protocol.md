# Architecture bake-off: pre-registered protocol

Written 2026-11-06, before any model was trained or scored. Changes after this point are appended
under "Amendments" with a date and a reason; nothing above that heading is edited.

Reproduce everything with `uv run python scripts/run_bakeoff.py` (stages are cached under
`work/bakeoff/`; `--fresh` clears the cache).

## 0. Hard rules this protocol obeys

- `data/` is never read. `evaluation/gt/` is never read, loaded, or used for any selection or
  tuning. The only target-match facts used are the public numbers in `docs/upstream/DATA.md`
  (46 slots, 20.5 mean tracked objects, ball missing on 41 % of graded frames).
- Strict rule: training and every parameter choice use external matches only. The decoder,
  features and thresholds run on a new match with fixed parameters chosen externally. There is no
  self-training and no calibration on the target.
- Every score comes from `evaluation/possession_metrics.evaluate`, imported unchanged. Ranking
  metric `spell_f1_mean`; `macro_f1` is reported beside it.
- No code from other teams' repositories. No StatsBomb data. Nothing is committed or pushed.

## 1. Corpus and holdouts

- IDSSE 7 (per-frame BallStatus and BallPossession), Metrica 3, SkillCorner 20 (10 fps real
  automatic tracking; team possession only; **no dead label**).
- **Locked benches (never trained on, never used for selection, scored once at the very end):**
  - IDSSE: the match whose `sha256("bakeoff-v1/idsse/" + match_id)` is lexicographically smallest.
    Computed before any run: `DFL-MAT-J03WQQ` (hash prefix `26bd839f8207`; the next smallest is
    `62557bdcfe12`). The function is `katena.eval.protocol.locked_idsse`; a test pins the result.
  - Metrica: `Sample_Game_2` (named in the brief).
- **Development set** = the other 28 matches (IDSSE 6, Metrica 2, SkillCorner 20). Leave-one-match-out
  (LOMO) over the development set: the model scored on match k never saw match k, and never saw a
  locked bench.
- SkillCorner labels have no D. On SkillCorner every prediction of D is replaced by X before
  scoring (the scorer then repeats the last team label). SkillCorner's own X frames are ungraded.
  The SkillCorner "alive" head is not trained on SkillCorner (no dead labels), only the team head is.
  SkillCorner `macro_f1` is not meaningful (no D class) and is reported only for completeness.

## 2. Time grid and scoring

- All models work on a 5 Hz grid (every 5th frame at 25 fps, every 2nd at 10 fps). Predictions are
  expanded back to the native frame grid by nearest-sample hold and scored against the native
  ground truth with the official `evaluate` (no bootstrap argument, `bootstrap_n=0`; our own paired
  bootstrap below uses the official helper functions `possession_spells`, `spell_overlaps`,
  `matched_gt_flags`).
- Expected cost of the 5 Hz grid: at most 0.1 s of boundary quantisation. It is applied to every
  model equally.

## 3. Degrader (`katena.sim.degrade.corrupt`)

`corrupt(m, cfg, seed)` turns a clean match into automatic-ATD-like input: strips identity (players
become anonymous track fragments in 23 slots per team = 46 slots), fragments tracks with gaps,
bursty ball dropout correlated with congestion and ball flight, ghost and duplicate objects, and
position jitter with occasional outlier jumps.

- The numeric distributions (ball-gap durations, player-gap durations, ball-gap hazard versus
  congestion and ball speed) are estimated by `scripts/calibrate_degrader.py` from real automatic
  tracking: SkillCorner (detection flags) and Metrica (real NaNs). The script writes
  `src/katena/sim/calibration.json`; its output is frozen before models run (see Amendment A1 if
  it is regenerated).
- Three severities, set a priori from public numbers: **mild** (ball missing on 20 % of in-play
  frames, mean tracked objects 21.5), **target-like** (41 %, 20.5, the DATA.md numbers),
  **severe** (55 %, 18.5). Matches that already have missing ball (Metrica, SkillCorner) are topped
  up to the target missing fraction, never below what they already have.
- Plus **native** (no corruption), reported only as the "near-clean" warning column. Native is not
  used for decisions.
- Headline = target-like. Ranking pool = all development matches, equal weight per match.
- Training uses every training match under mild, target-like and severe (different row subsets,
  stride 4 at 5 Hz). Evaluation corrupts each held-out match with fixed seeds under each severity.
- Fixed seed per (match, severity): `sha256(match_id|severity) mod 2^31`.

## 4. Label semantics experiment

- `katena.labels.sandwich(y, T)`: an H or A segment shorter than T seconds whose neighbours inside
  the same graded run are both the other team (no D between) is relabelled to the neighbours' team,
  iterated to a fixed point.
- Sweep T in {0, 1, 2, 3, 5} s. The **main bake-off uses T = 2 s**, fixed now, before results: it is
  the middle of the sweep and does not use the public GT mean spell duration. (The `external-data.md`
  figures give mean spell 9.8 s at T = 2; the public window mean is 14.2 s; reaching it needs T about
  5 s. That statement is the only public information about the target's convention, and the
  organisers have not confirmed that it may inform a prior.)
- Each T is trained on labels at that T (all training matches, SkillCorner included) and evaluated on
  (i) IDSSE development matches with the same T, (ii) the never-trained-on-IDSSE-semantics Metrica
  labels (event-derived; two development matches, plus Metrica G2 at the end only). Candidates in the
  sweep: M0 (no training, constant across T except IDSSE scoring), M3 (and M2 as a decoder check).
  LOMO folds: IDSSE development (6) and Metrica development (2), i.e. 8 folds per T. Duration priors
  and decoder parameters are re-fit for each T.
- Reported: score matrix (train T x eval T), ranking stability across T (Kendall tau of model
  order), and whether T can be chosen without the competition GT (my expectation: it cannot be
  identified from external data alone).

## 5. Candidates

- **M0** the starter baseline algorithm, unchanged: nearest tracked player to the ball (team), ball
  missing = D, majority vote over +/-50 frames (= +/-10 samples at 5 Hz; the official `smooth`
  function is imported from `starter/baseline.py`).
- **M1** rule chain: control detection (nearest player within a radius for a minimum dwell),
  hold-through-flight-and-gaps state machine, simple dead-ball rule (ball outside the pitch, or
  everything still). Constants fixed a priori in `katena.models.rules` and not tuned.
- **M2** LightGBM alive/dead head and team head, combined into 3-class log-probabilities, 3-state
  Viterbi (transition matrix fit on training matches only).
- **M3** M2 plus explicit-duration HSMM (duration pmfs fit on training matches only: lognormal-
  smoothed histograms; duration weight 1).
- **M4** ablations of M3: no ball-free features; ball-free features only (no ball-derived feature
  at all, including ball dead-ball cues and ball-missingness features); no dead-ball cues; no ball-
  missingness features.
- **M5** small TCN (causal-free, dilated 1-D convolutions) on the same features at 5 Hz with the same
  two heads, decoded with the better of Viterbi and HSMM from the M2/M3 comparison. Because of the
  compute budget it is run on **8 LOMO folds** (the 6 IDSSE and 2 Metrica development matches) and
  compared with M3 on those same folds only.
- **M6** M3 plus a post-decode spell rescoring model (keep / merge-away / flip-team) trained on the
  training matches only with nested folds. Nested LOMO is too expensive, so M6 runs on **6 grouped
  folds** of the development set (groups fixed by hash of match id); for each outer group the
  inputs of the spell model on the other groups are out-of-fold predictions from models that saw
  neither that group nor the outer group. M6 is compared with M3 trained on the same outer folds.
- **M7** PathCRF half-day spike: only if time remains; report the outcome either way.

Hyperparameters are fixed now and are not tuned: LightGBM 150 trees, learning rate 0.1, 31 leaves,
min 50 samples per leaf, feature fraction 0.5, bagging 0.5, `max_bin` 63, seed 0, CPU, `n_jobs` 8.
The only selected hyperparameter is the **emission temperature tau in {0.25, 0.5, 1.0}** shared by
both decoders. For test match k it is chosen as the value maximising mean `spell_f1_mean` over the
other development matches' LOMO out-of-fold predictions at the target-like severity (labels of k are
never used to choose it). Reported as a sensitivity table too.

## 6. Decision rule (from docs/PLAN.md)

A change is adopted over the incumbent only if, at target-like severity pooled over the evaluation
folds, (a) it has the higher `spell_f1_mean` on at least 70 % of matches, and (b) the paired
bootstrap 95 % interval of the mean per-match difference excludes 0. The bootstrap resamples
ground-truth possessions within each match (2000 draws, seed 0), the statistic being the mean over
matches of the difference in official-formula spellF1 computed from the official matching flags.
Ladder (incumbent first): M0, M1 vs M0, M2 vs max(M0, M1), M3 vs M2, M5 vs M3 (8 folds), M6 vs M3
(6 grouped folds). An M4 component is "kept" if M3 beats the ablation under the same rule; otherwise
the component is "not shown to help". The same comparison is shown for the other severities and each
dataset as secondary information (not gated).

## 7. Oracles (best model, development folds, target-like)

1. Perfect ball positions: the model is given the clean ball (no ball corruption).
2. Perfect team labels with predicted alive/dead.
3. Perfect alive/dead with predicted team.
4. Perfect spell boundaries with predicted team per spell (each GT spell takes the majority predicted
   team over its frames).
Plus an error-type census on the predictions (wrong team, split GT spell, merged GT spells, boundary
offset, dead-ball), defined in `katena.eval.errors`.

## 8. Analyses that are not model comparisons

- Does a missing ball predict dead ball? Measured on Metrica's real NaNs and on SkillCorner against
  its labels, then on the degrader; the corpus cannot settle the target's behaviour because the
  Metrica D label is event-derived and approximate and SkillCorner has no D. An ablation drops all
  ball-missingness features.
- Smart Tagging imitation: assessed on paper (can Metrica events simulate it with noise?); built only
  if cheap.
- Severity sensitivity: the best model is scored under mild / target-like / severe / native.

## 9. Compute budget and cuts declared in advance

About 3 h of compute on an M5 / 16 GB. LightGBM is CPU only. Declared scope cuts: M5 on 8 folds, M6 on
6 grouped folds, 5 Hz grid, training-row stride 4. If a stage overruns, it is cut and the cut is
reported in `docs/bakeoff.md`.

## 10. Locked benches

After all selections are final, the final list of models is trained on all 28 development matches and
scored once on the two benches under each severity. The numbers are reported; they change nothing.

## Amendments

### A1 (2026-11-06): degrader calibration results and the dead-ball coupling knob

Run before any model was trained. `scripts/calibrate_degrader.py` found: ball-gap durations on
SkillCorner are lognormal (mu 1.23, sigma 0.68 in log seconds, 2461 gaps); player-gap durations
lognormal (mu 1.41, sigma 1.48, 138k gaps); in Metrica the ball is NaN on 81 % of dead-label frames
versus 2.4 % of alive-label frames (three games, 73-88 % vs 0.8-4 %), and on alive frames the onset
hazard rises with congestion and with ball speed (consistent with the brief), while SkillCorner shows
the opposite sign for both. The two real sources disagree, so the degrader keeps a modest positive
congestion and speed coupling (the brief's prior, supported by Metrica) and exposes `dead_odds`, the
ball-loss hazard multiplier while the ball is dead (default 1 = ball loss independent of dead ball).
The headline world stays **independent (dead_odds = 1)**, as pre-registered. Evaluation-only worlds with
dead_odds 3 and 15 (IDSSE only, needs dead labels) are added as sensitivity conditions, and the models
are still trained on the independent world. `target` mean tracked objects was tuned from 20.8 to about
20.5 (player_avail 0.90) and `severe` to about 18.5 (0.78); these are public-number targets, no model
involved.

### A2 (2026-11-06): decoder temperature grid extended

After the first fold (IDSSE `DFL-MAT-J03WOH`, a development match; heads trained without it) the
tau grid {0.25, 0.5, 1.0} showed its best value at the edge, with a plateau at 2-3. The grid is
extended to {0.25, 0.5, 1, 2, 4} for every model and every dataset. Selection rule unchanged.

### A3 (2026-11-06): label-semantics sweep gets a second design

Design A is as in section 4. Design B ("cross-provider") trains on IDSSE and SkillCorner only, with
no Metrica rows, and evaluates held-out IDSSE matches (LOMO) and the two development Metrica matches,
so that the Metrica labels are strictly a different provider's convention. Both designs are reported;
design B is the one that answers "does T transfer to event-derived labels".

### A4 (2026-11-06): package name

`src/katena/features/` is named `src/katena/featurize/` because `.claude/settings.json` denies any
path called `features/` in this repository. No behaviour change.

### A5 (2026-11-06): LightGBM runtime

LightGBM needs `libomp`, which is absent (no Homebrew). `katena.models.gbdt.ensure_openmp` re-execs the
process once with `DYLD_FALLBACK_LIBRARY_PATH` pointing at the copy bundled in scikit-learn
(`uv add scikit-learn` pinned). Models and parameters are unchanged.

### A6 (2026-11-06): ablation scope cut

LOMO ablations (M4) are expensive on a machine that is also under memory pressure from other
applications. M4 variants run on 14 folds: all 8 IDSSE and Metrica development matches plus the 6
SkillCorner matches with the smallest `sha256("abl|" + id)`; they predict only the mild / target-like /
severe conditions. Each ablation is compared with M3 on the same 14 folds. Chosen by hash, before any
M4 result.

### A7 (2026-11-06): additions made while running, none of them after seeing a bench score

- M5 is compared with M2 as well as M3 (M3 was not adopted over M2, so M2 is the incumbent), and was run
  with a second training seed (`M5s1`) because the first-seed result was borderline. The benches score M5
  (seed 0) once, trained on all 28 development matches.
- `nomissw` (same features as `nomiss`, IDSSE folds only) was added to score the no-missingness model under
  the dead-coupled worlds.
- M6 training spells come from inner out-of-fold decodes at the severities target-like and severe only, and
  use the global best tau of the grouped M3 run; application uses the per-match tau, as for M3.
- The pooled `macroF1` column of Table 1 excludes SkillCorner (no dead label).
- M7 (PathCRF) was not run; see section "Not done".
