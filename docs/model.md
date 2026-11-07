# D4 model: per-frame log P(H, A, D), validated leave-one-match-out

Everything here is measured on **external open matches only** (IDSSE/DFL and Metrica Game 3),
corrupted by D2's simulator. Nothing reads `data/` or the competition match. Code:
`src/katena/model/`. Tests: `tests/test_model_*.py`.

## Reproduce

From the repository root, with the team venv (`C:/Users/User/Desktop/barca/.venv`):

```
python scripts/external_corpus.py build                 # D1 corpus, once (about 5 min)
python scripts/model_cache.py --workers 2               # 60 corrupted, featurized copies (5 min, 0.7 GB)
python scripts/model_lomo.py run                        # the LOMO study below (resumable, about 48 min)
python scripts/model_lomo.py robust                     # severity-mismatch check (about 9 min)
python scripts/model_lomo.py report --md runs/d4_lomo/report.md
python scripts/train_model.py                           # final model on the 6 dev matches -> artifacts/
python scripts/predict_target.py --synthetic             # end-to-end smoke test, temp output
python -m pytest -q tests/test_model_*.py
```

The human, on the real match (prints aggregates only):

```
python scripts/train_model.py                           # once; writes artifacts/katena_model.json
python scripts/predict_target.py                        # writes evaluation/results/katenaball.csv
python evaluation/validate_submission.py evaluation/results/katenaball.csv
```

`$KATENA_MODEL_CACHE` overrides the cache location (default
`C:/Users/User/Desktop/barca/external_data/derived/d4_feats/`). `runs/` and `artifacts/` are
gitignored.

## Protocol

**Locked holdout.** Rule, fixed before any model was trained: the last two DFL match ids in sorted
order, `idsse-J03WQQ` and `idsse-J03WR9`. `katena.model.data.check_names` refuses to load them
unless called with `allow_locked=True`, which nothing in the repository does. They are to be scored
once, at the end (PLAN.md, "Validating blind" step 3). Development uses the other six:
`idsse-J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY` and `metrica-game3`.

**Labels.** D1's harmonisation at `min_control_s = 3.5 s`, for training and for scoring.

**Metrica Game 3 ball.** Game 3's open tracking removes the ball while it is dead (D1 finding 3), so
"ball missing" is an almost perfect D cue there and nowhere else. Before corrupting it, every NaN
ball frame inside a half is filled, label-free: linear in time between the last and the next seen
position, held at the ends of a half (`data.fill_missing_ball`). That fills 46,325 frames (32.2% of
graded frames, 95.7% of them dead ball). After the fill every clean match has a ball on every
graded frame, so ball loss comes only from `corrupt`. On Game 3's corrupted evaluation copy the
ball is then present on 55.7% of D rows and 60.9% of alive rows, the mild dead-ball dependence D2
builds in (`ball_beta_dead`), the same as for DFL. Linear interpolation was chosen over "hold the
last seen position" so the filled ball neither sits perfectly still through every stoppage nor
jumps at every restart, both of which would be Game-3-only artefacts.

**Training copies.** Each training match contributes 5 corrupted copies, one per config of
`severity_grid(DEFAULT_CFG, 5)` (severity 0.5, 0.71, 1.0, 1.41, 2.0: graded ball presence 0.80 to
0.19), each with its own seed in [1e6, 2e6), featurized every 10th frame with offsets 0, 2, 4, 6, 8
(so the copies tile every 2nd frame), plus a team-swapped duplicate of every row (`swap_teams`, H and
A swapped). About 142k rows per match. The no-augmentation ablation uses 5 copies with the default
config and other seeds, with the same offsets and the middle copy shared.

**Evaluation copy.** Default config, seed 9,000,000 + a per-match code (disjoint from every training
seed), featurized every 5th frame. The model predicts those rows; the decoder upsamples them to
every frame (linear in log space, renormalised). The starter baseline runs on the same corrupted
match.

**Model.** LightGBM 4.7, CPU, `deterministic=True`, `force_row_wise=True`, `num_threads=6`,
`seed=20261107`, 200 rounds, learning rate 0.1, 63 leaves, `min_data_in_leaf=400`, feature and
bagging fraction 0.6, `max_bin=63`, L2 2.0. Fixed a priori, not tuned. Two heads (alive/dead on all
rows, home/away on alive rows) or one 3-class booster.

**Nested calibration and decoder choice.** Inside each fold's 5 training matches only: 2 groups by
sorted name (round robin), train on one group and predict the other group's evaluation copies. These
out-of-fold scores fit the temperatures (NLL, per head for the two-head model) and choose the
placeholder decoder's `(min_dwell_s, switch_penalty)` on the grid {0.25, 0.5, 1, 2} s x
{4, 8, 16, 32, 64, 128} by mean spellF1 against harmonised labels. The held-out match is used for
nothing but the final score. `train_model.py` repeats exactly this over all 6 development matches.

**Placeholder decoder** (`katena.model.decode_simple`, J3 owns the real one). Exact Viterbi per half
over {H, A, D}: maximise the summed per-frame log-probs minus a penalty per label change, with every
run at least `min_dwell` long. Solved as a segment DP in blocks of `min_dwell` frames, vectorised
over penalties (checked against brute force in the tests). Log-probs are rounded to 4 decimals
before decoding.

**Scoring.** The official `possession_metrics.evaluate(gt, pred)` per held-out match. Paired
comparisons give per-match wins and two 90% bootstrap intervals of the mean spellF1 difference: over
matches only, and hierarchical (resample matches, then the ground-truth possessions inside each,
shared by both systems, as the scorer's own spell bootstrap does). The gate is PLAN.md's: at least
70% of matches (5 of 6) and the interval above 0.

## Results (LOMO over the 6 development matches, default-noise evaluation copies)

Generated by `python scripts/model_lomo.py report` (`runs/d4_lomo/report.md`).

### spellF1 / macroF1 per held-out match

| variant | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | Game3 | mean spellF1 | mean macroF1 |
|---|---|---|---|---|---|---|---|---|
| starter baseline | 0.210 / 0.414 | 0.217 / 0.367 | 0.195 / 0.400 | 0.181 / 0.385 | 0.212 / 0.413 | 0.197 / 0.462 | **0.2019** | 0.4067 |
| **twohead** (final v1) | 0.745 / 0.921 | 0.720 / 0.899 | 0.759 / 0.919 | 0.723 / 0.903 | 0.778 / 0.921 | 0.785 / 0.903 | **0.7517** | 0.9110 |
| 3class | 0.753 / 0.922 | 0.734 / 0.898 | 0.748 / 0.914 | 0.734 / 0.905 | 0.795 / 0.921 | 0.791 / 0.907 | **0.7594** | 0.9112 |
| twohead_ballfree | 0.620 / 0.866 | 0.527 / 0.804 | 0.700 / 0.877 | 0.647 / 0.860 | 0.702 / 0.888 | 0.630 / 0.845 | **0.6378** | 0.8568 |
| twohead_noaug | 0.753 / 0.922 | 0.713 / 0.896 | 0.760 / 0.918 | 0.733 / 0.904 | 0.798 / 0.926 | 0.793 / 0.902 | **0.7585** | 0.9112 |

### predicted / true possessions, and mean predicted duration in s (truth in brackets)

| variant | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | Game3 |
|---|---|---|---|---|---|---|
| starter baseline | 600/231, 9.1 (14.8) | 669/251, 7.9 (10.9) | 573/215, 8.7 (13.4) | 584/245, 8.9 (12.5) | 632/225, 8.5 (14.5) | 763/238, 6.9 (16.6) |
| twohead | 209/231, 16.3 (14.8) | 220/251, 12.2 (10.9) | 195/215, 14.8 (13.4) | 227/245, 13.6 (12.5) | 198/225, 16.3 (14.5) | 230/238, 17.4 (16.6) |
| 3class | 230/231, 14.8 (14.8) | 217/251, 12.4 (10.9) | 196/215, 14.9 (13.4) | 229/245, 13.5 (12.5) | 197/225, 16.5 (14.5) | 226/238, 17.7 (16.6) |
| twohead_ballfree | 197/231, 16.6 (14.8) | 218/251, 11.6 (10.9) | 187/215, 15.3 (13.4) | 213/245, 14.6 (12.5) | 191/225, 16.8 (14.5) | 208/238, 17.9 (16.6) |
| twohead_noaug | 215/231, 15.9 (14.8) | 219/251, 12.3 (10.9) | 192/215, 15.0 (13.4) | 229/245, 13.5 (12.5) | 197/225, 16.5 (14.5) | 229/238, 17.5 (16.6) |

### frame-level detail (means over matches)

| variant | spellF1 @ IoU 0.5 | accuracy | alive/dead F1 | team acc (both alive) | spell count error |
|---|---|---|---|---|---|
| starter baseline | 0.1919 | 0.4410 | 0.6951 | 0.7870 | +402.7 |
| twohead | 0.7650 | 0.9228 | 0.9779 | 0.9041 | -21.0 |
| 3class | 0.7751 | 0.9231 | 0.9780 | 0.9043 | -18.3 |
| twohead_ballfree | 0.6489 | 0.8722 | 0.9391 | 0.8804 | -31.8 |
| twohead_noaug | 0.7722 | 0.9232 | 0.9792 | 0.9026 | -20.7 |

Without the decoder (per-frame argmax of the same calibrated log-probs) the two-head model scores
spellF1 0.440 with 550 predicted spells per match: the decoder is worth +0.31.

### paired comparisons on spellF1 (A minus B)

Gate (PLAN.md): wins on at least 70% of matches (5 of 6) and the 90% interval above 0.

| A vs B | wins | mean diff | 90% CI (hierarchical) | 90% CI (matches) | gate |
|---|---|---|---|---|---|
| two-head vs starter baseline | 6/6 | +0.5498 | [+0.5235, +0.5749] | [+0.5307, +0.5673] | pass |
| 3-class vs starter baseline | 6/6 | +0.5575 | [+0.5323, +0.5819] | [+0.5404, +0.5737] | pass |
| two-head vs 3-class | 1/6 | -0.0077 | [-0.0167, +0.0021] | [-0.0131, -0.0013] | 3-class passes on matches only |
| all features vs ball-free only (two-head) | 6/6 | +0.1139 | [+0.0789, +0.1508] | [+0.0830, +0.1493] | pass |
| severity augmentation vs default config only (two-head) | 1/6 | -0.0067 | [-0.0169, +0.0029] | [-0.0121, -0.0010] | no-aug passes on matches only |

### severity mismatch (`python scripts/model_lomo.py robust`)

The same fold models (refit; deterministic, so byte-identical predictions on the default copy in
6/6 folds), with their nested temperatures and decoder parameters, on evaluation copies whose noise
is milder (severity 0.5, ball present 80% of graded frames) or harsher (2.0, ball present 19%) than
the default (59%):

| variant | severity 0.5 | severity 1.0 (default) | severity 2.0 |
|---|---|---|---|
| starter baseline | 0.2540 | 0.2019 | 0.1127 |
| twohead (severity-augmented) | 0.7697 | 0.7517 | **0.6752** |
| twohead_noaug | **0.7766** | **0.7585** | 0.6499 |

Augmented minus not: severity 0.5 -0.0068 (0/6 wins, CI [-0.0098, -0.0037]); severity 2.0
**+0.0253 (6/6, CI [+0.0136, +0.0368])**.

### chosen decoder parameters and temperatures (nested, inside the training matches)

| variant | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | Game3 |
|---|---|---|---|---|---|---|
| twohead | pen 32, dwell 1 s, T 1.25, 1.39 | pen 32, dwell 1 s, T 1.25, 1.40 | pen 32, dwell 0.25 s, T 1.27, 1.39 | pen 32, dwell 0.25 s, T 1.27, 1.37 | pen 32, dwell 2 s, T 1.23, 1.39 | pen 32, dwell 2 s, T 1.06, 1.34 |
| 3class | pen 16, dwell 2 s, T 1.29 | pen 32, dwell 2 s, T 1.29 | pen 32, dwell 2 s, T 1.29 | pen 32, dwell 2 s, T 1.29 | pen 32, dwell 0.25 s, T 1.26 | pen 32, dwell 0.25 s, T 1.19 |

(T = T_alive, T_team for the two-head model.) The out-of-fold spellF1 inside the training matches
(0.73-0.75) matches the held-out scores (0.72-0.79), so the nested choice is not optimistic.

## Decisions

- **Final v1 = two-head, all features, severity-augmented.** It beats the baseline on every match
  by about 0.55 spellF1. The 3-class model and the no-augmentation model are each about 0.007
  better on default noise (5/6 wins), but neither interval excludes 0 once possessions are
  resampled, so neither passes the gate as a change. Augmentation is kept because it buys +0.025
  (6/6) if the real noise is harsher than D2's default, against -0.007 if it is milder, and D2's
  config is a guess beyond its three anchored numbers.
- **Ball-free features alone are clearly worse** (-0.114, 6/6), but 0.64 is far above the baseline:
  usable as a D5 student or fallback.
- **Decoder**: a switch penalty of 32 log units (after temperature scaling) is the stable operating
  point; the minimum dwell hardly matters. Handed to J3 as the reference for the semi-Markov decoder.

## Caveats

- **The absolute numbers are simulator numbers.** The starter baseline gets 0.202 / 0.407 on our
  corrupted inputs against the published 0.2433 / 0.6724 on the real match. In D2's simulator ball
  loss is nearly independent of dead ball (P(missing | D) 0.38-0.44, P(missing | alive) 0.39-0.43),
  while the baseline's published macroF1 implies the real ATD loses the ball much more in dead time.
  Flagged to D2 (decision log).
- **No pre-kickoff dead ball in DFL.** The target's second half has about 2,900 graded pre-kickoff D
  frames. Only Game 3 shows the model a pre-kickoff layout (14 s and 17 s). J5's kickoff detection
  should override the model there.
- **Spells run slightly long**: about 20 too few per match and mean durations 0.8-1.8 s long.
  The real decoder's duration prior could correct that.
- Locked matches `idsse-J03WQQ`, `idsse-J03WR9` not scored yet, by design.

## Runtimes (this 8-thread, 16 GB machine, LightGBM on 6 threads)

| step | time | peak memory |
|---|---|---|
| feature cache, 60 copies (2 workers) | 4.8 min | about 1.2 GB per worker |
| LOMO study, 4 variants x 6 folds (nested) | 44 min (two-head 11.1, 3-class 15.9, ball-free 6.3, no-aug 10.7) | 1.6 GB |
| severity-mismatch check, 12 refits | 9 min | 1.6 GB |
| `train_model.py` (6 matches, 857k rows, plus 2 nested inner fits) | 2.1 min | about 1.8 GB |
| `predict_target.py --synthetic --external idsse-J03WMX` (146k frames, full length) | 14 s (7 s loading model and match, 5 s features) | under 1 GB |
| `predict_target.py --synthetic` (6-min synthetic match) | 7 s | |

The final model (`artifacts/katena_model.json`, gitignored): two-head, temperatures T_alive 1.163,
T_team 1.298, decoder penalty 32 with dwell 0.25 s, out-of-fold spellF1 0.744 / macroF1 0.907 over
the 6 development matches. Two runs of the predict script give byte-identical files (same SHA-256).

## Placeholders and hand-offs

- `decode_simple` is a placeholder. J3's decoder should take `Model.predict_logp` output (float32
  [T, 3], log P(H, A, D), rows at `model.info["predict_stride"]`) and `feats.t_frame`.
- `katena.model.baseline.starter_baseline` re-implements `starter/baseline.py` on a `Match`, for
  comparisons on external data.
