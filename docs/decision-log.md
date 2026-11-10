# Decision log

Every model, feature, or data decision: what changed, the external score, keep or drop. Newest first. Scores are
`spellF1` at target-like noise on leave-one-match-out development folds, unless stated. Log every look at the public
5 minutes here too.

## 2026-11-10

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
