# D5: independent audit of the bake-off

Written 2026-11-07 on `david/d5-audit`, for the bake-off in [../bakeoff.md](../bakeoff.md) and the code on
`main` at `fec7ada` plus `david/infra`. Method: read every module on the path from the corpus to the
score, rebuilt the base cache on this Windows machine and re-scored M2, and cross-checked the loaders
against David's independent implementation on `origin/david/foundation` ([../reconcile.md](../reconcile.md)).
External data and synthetic fixtures only. Nothing under `data/` was opened, listed or run, and no
`evaluation/gt` file was read.

## Verdict

**No leakage that moves a reported number was found, and M2 reproduces exactly.** The base cache built here
scores M2 at 0.5861 on the 28 development folds (bake-off: 0.586, and 0.5861 in the addendum), with every
per-dataset, per-severity and fixed-tau mean equal to the published tables at the published precision. The
one leak in the protocol (tau chosen from folds that saw the held-out match) has zero effect, because all 28
folds chose the same tau.

The real problems are elsewhere: Metrica's event-derived dead-ball labels run long (A3), the target loader
rests on a layout assumption that the baseline check cannot test (A7), and byte-identical regeneration across
the two machines is not assured (A11). Six small bugs were fixed on this branch, each with a test.

| id | severity | finding | status |
|---|---|---|---|
| A1 | ok | LOMO isolation of training rows, priors, TCN scaler, M6 nesting | verified |
| A2 | low | tau chosen from folds whose models and priors saw the held-out match | no effect here; recommendation |
| A3 | **medium** | Metrica dead-ball spans run long | open, D4 |
| A4 | low | X written on graded block-edge frames | fixed `5f72876` |
| A5 | low | locked IDSSE bench recomputed from whatever is on disk | fixed `ce55d11` |
| A6 | ok | scorer wrapper equals `possession_metrics.evaluate` | verified |
| A7 | **medium** | ATD reader assumes raw groups follow `<Player>` order; untestable by `check_target.py` | check added `be3f955`; human run pending |
| A8 | low | `probe_schema.py` could print a homography data row | fixed `020c6a6` |
| A9 | low | text files read with the OS default encoding (cp1252 on Windows) | fixed by the coordinator `47a3f4b`; residue listed |
| A10 | low | corpus code digest depends on line endings | fixed `5da7a5a` |
| A11 | **medium** | byte-identical submission across macOS and Windows not assured | recommendation, J2/J3 |
| A12 | ok | frame alignment, half blocks, attack direction | verified |
| A13 | ok | label relabelling (`katena.labels.sandwich`) | verified |
| A14 | low | degrader seeds fine; calibration used every SkillCorner and Metrica match, bench included | info |
| A15 | ok | no feature sees a label | verified |
| A16 | ok | X on graded frames: only A4 | verified |
| A17 | ok / low | scripts that read `data/` print aggregates only, with two caveats | verified |
| A18 | low | headline mean is 71% SkillCorner, which has no dead-ball label | recommendation |
| A19 | low | SkillCorner "target-like" has 15.3 tracked objects, not 20.5 | info, D2 |
| A20 | low | the degrader is far rougher than real automatic tracking on positional jitter | alarm result, D2 |
| A21 | low | base cache READY named the wrong commit | fixed in READY (approved) and `b4cc087` |

## Reproduction on this machine

`scripts/run_m2.py` (new; the minimum of `run_bakeoff.py`: `corpus.build_all` with 3 workers, LOMO heads for
the full feature set, Viterbi M2, M0/M1) through the slot runner, `KATENA_WORK=D:/katena/work-base`, code at
`8537084` (clean). Windows 10, i7-7700, Python 3.11.17, LightGBM 4.7.0 with `num_threads=8`, CRLF checkout.
31.4 min wall clock (corpus cache 11.7, LOMO heads 15.8, decode and score 3.1, baselines 0.9), with a second
heavy job running alongside.

| | this machine | bake-off (macOS) |
|---|---|---|
| M2, 28 folds, target-like | **0.5861** | 0.586 (0.5861 re-scored) |
| IDSSE / Metrica / SkillCorner | 0.5441 / 0.6434 / 0.5930 | 0.544 / 0.643 / 0.593 |
| native / mild / target / severe | 0.6210 / 0.6116 / 0.5861 / 0.5475 | 0.621 / 0.612 / 0.586 / 0.547 |
| macroF1, IDSSE + Metrica | 0.8677 | 0.868 |
| IDSSE target_dead3 / target_dead15 | 0.5560 / 0.5561 | 0.556 / 0.556 |
| fixed tau 0.25 / 0.5 / 1 / 2 / 4 | 0.5353 / 0.5672 / 0.5833 / 0.5861 / 0.5820 | 0.535 / 0.567 / 0.583 / 0.586 / 0.582 |
| M0 / M1 | 0.2804 / 0.4322 | 0.280 / 0.432 |
| realised IDSSE ball missing / objects, mild-target-severe | 20.0% / 21.30, 41.0% / 20.20, 55.0% / 18.19 | 20% / 21.3, 41% / 20.2, 55% / 18.2 |

Thread counts, platform floats, the corpus and CRLF were the suspects for a mismatch; none shows at this
precision. The corpus built here is identical in content to what the committed loaders produce (two matches
rebuilt in memory and compared array by array); CRLF changes only the corpus code digest (A10). Bit-identity of
the trained models across platforms was not tested, because no macOS cache is available here (see A11).

## Findings

### A1. LOMO isolation: verified

- Training rows: `bakeoff.stage_lomo` calls `lomo.run_fold(..., exclude={r.key})` (`bakeoff.py:67`) and
  `Pools.fit` masks every row of the held-out key. `Pools` is built from the development matches only, so the
  benches are never in a pool.
- Label-sequence priors (transitions, class prior, durations): `Pools.label_seqs(exclude, T)` skips the
  excluded keys and SkillCorner (`lomo.py:48-57`); `decode_and_score` excludes the held-out match by default
  (`bakeoff.py:88`), M3g excludes its whole group.
- TCN: the scaler and the training sequences both exclude the held-out match (`stages.py:88-89`).
- M6: outer models exclude the group, inner models exclude both groups, and spell-model priors exclude both.
- M1 constants and the feature definitions are fixed and use no labels. Duration and transition priors are
  fitted on IDSSE and Metrica labels only.

### A2. Tau selection leaks at second order (low; no effect on the reported numbers)

For held-out match k, `decode_and_score` picks tau maximising the mean spellF1 over the other folds
(`bakeoff.py:98`). Those folds' probabilities come from models trained on k's rows, and they are decoded with
priors that include k's labels. So k's labels can influence k's tau. The same pattern is in `_decode_sweep`
(`stages.py:65`) and in M6's `tau_global` (`stages.py:183`, Amendment A7). The bake-off notes the model part of
this; the prior part is not mentioned.

Effect here: none. All 28 folds chose tau = 2.0, and the fixed-tau-2 mean equals the selected mean (0.5861). The
grid was also extended after the first fold's result (Amendment A2), which is documented and harmless on a
plateau. **Recommendation:** for the D1-D4 gates, fix tau = 2 a priori (the plateau is known: 0.583 / 0.586 /
0.582 for 1 / 2 / 4) or nest the choice inside the training folds as the foundation's `model/lomo.py` did. This
also removes the five decodes per fold.

### A3. Metrica dead-ball spans run long (medium; D4)

`katena.external.metrica.derive_labels` makes the ball dead from a dead trigger's start frame until the next
`SET PIECE` or goal kick. A restart that is not tagged keeps the label at D through live play, and on-ball
events in between are ignored (`external-data.md`: 135 / 77 / 154 such events in the three games).

Evidence: Metrica's open tracking removes the ball while it is dead (David's finding, confirmed here). So the
share of D frames with the ball missing measures how clean the D label is:

| | P(ball missing given D) | P(ball present given alive) |
|---|---|---|
| main, Game 1 | 0.731 | 0.992 |
| main, Game 2 (locked bench) | 0.860 | 0.957 |
| main, Game 3 | 0.880 | 0.980 |
| David's derivation, Game 3 | **0.992** | 0.980 |

David's `metrica_open.labels_from_events` derives each stoppage backwards from its restart (from the first stop
marker after the last in-play action). On Game 3 the two agree on 93.4% of frames and on the team for 99.3% of
frames both call alive. On the 7,215 frames where main says D and David says alive, the ball is present 77% of the
time. Scoring one labelling as the other's prediction gives spellF1 0.878: the two conventions disagree on
about one possession in eight.

Impact: Metrica is 2 of 28 development folds (and the locked bench), but 24.9% of the alive head's training rows.
Its D labels teach "dead" on live frames, and the Metrica development score (0.643) is measured against them.
Table 7's "ball missing on 79% of Metrica dead frames" understates the coupling in that sample, at least in Game 3
(the only game with a second labelling).
**Recommendation (D4):** derive stoppages backwards from restarts, or end a D span when on-ball play resumes,
and take it through the gate. Do not change it ad hoc: it moves every Metrica number.

### A4. X on graded frames at block edges (low; fixed)

`score.to_native_codes` expands 5 Hz predictions by nearest sample. When a half block does not start or end on
the 5-frame grid, its first or last one or two frames map to a sample outside the block, which is X. In the base
cache that is 14 graded frames over the 30 external matches (J03WN1 3; J03WOH, J03WPY, J03WR9, Games 1 and 2 two
each; Game 3 one), 12 of them on development folds. The target's graded halves happen to sit on the grid (0
frames), but any change of blocks in J2 would reintroduce it, and the rule is never to write X on a graded frame.

Fix `5f72876`: `to_native_codes(..., blocks)` clips each frame to the nearest sample inside its own block;
`score_rows`, the error census and `probe_focus.py` pass the blocks. Re-scoring M2 from the cache: 12 frames
change in 6 matches, spellF1 0.586133 before and after (largest per-match change 0).

### A5. The locked IDSSE bench was recomputed from the files present (low; fixed)

`protocol.split` took the smallest hash among the IDSSE matches on disk. With `DFL-MAT-J03WQQ` missing (a failed
or partial build, as happened this morning), `J03WPY` would have become the bench and J03WQQ, once built, would
have joined training. Fix `ce55d11`: the bench is the constant `LOCKED_IDSSE`, checked against `locked_idsse()`
by a test. The hash rule itself was re-derived independently: J03WQQ `26bd839f8207`, then J03WPY `62557bdcfe12`.
`run_m2.py` also refuses to start unless the split is exactly 28 + {J03WQQ, Sample_Game_2}.

### A6. Scorer wrapper: verified

`katena.eval.score.score` calls `possession_metrics.evaluate` unchanged and reads `spell_f1_mean` and
`macro_f1` from it. The extra `flags` use the official `fill_abstentions`, `possession_spells`, `spell_overlaps`
and `matched_gt_flags`, so the paired bootstrap's statistic with identity resampling equals the official
spellF1 (existing test `test_score_matches_official_and_flags_reproduce_f1`). `sc_mode` (SkillCorner, no dead
label) turns predicted D into X only for scoring, so the scorer repeats the last team; it never reaches a
submission. The adoption rule (`bootstrap.adopted`) matches the protocol: wins on at least 70% of matches and a
95% interval above zero, with paired resampling of ground-truth possessions.

### A7. The target loader's layout assumption cannot be tested by `check_target.py` (medium; check added)

`katena.target.atd.read_slot_teams` takes the team of raw group k from the k-th `<Player>` element, as
`starter/baseline.py` does. FIFA/EPTS defines the group order through `DataFormatSpecification` ->
`PlayerChannelRef` -> `PlayerChannel` -> `Player`, which need not follow the `<Player>` order: in Metrica Game 3
it does not (35 players, 22 groups per spec, 11 specs). `check_target.py` proves we read the file the way the
organisers' baseline does, but that baseline makes the same assumption, so a violation would pass every check
while mislabelling slot teams. On a synthetic EPTS pair with shuffled groups, `read_slot_teams` gets 33% of slot
teams right, while David's layout-aware kloppy reader gets 100%; with groups in `<Player>` order the two agree on
every value. The published baseline's macroF1 of 0.67 suggests the real file is in order, but that is indirect.

Added `be3f955`: `katena.target.layout.layout_report` follows the format's own chain and reports booleans and
counts (specs, groups per spec, one player per group, order matches, channels per group, first team is the local
team); `scripts/check_target_layout.py` is the human-run wrapper. **Human action:** run it with `check_target.py`
(J1).

### A8. `probe_schema.py` could print a homography data row (low; fixed)

`probe_csv_header` printed the first line of the homography CSV verbatim. DATA.md documents a header, but if the
file had none, a frame-level row would be in output that humans are told is safe to paste. Fix `020c6a6`: a first
line with any numeric field is reported by its field count only.

### A9. Default text encoding (low; fixed)

On Windows `open()` and `read_text()` default to cp1252, so the SkillCorner match JSON failed to decode and
stopped the corpus build at 13 of 30. Fixed by the coordinator in `47a3f4b` (loaders and the ATD reader read
UTF-8). The organisers would hit the same failure regenerating on Windows. Still on the OS default, ASCII today
and harmless: `fetch.Manifest` (JSON), `degrade._cal` (calibration JSON), `check_target.read_published` (CSV),
`probe_schema.probe_raw`, and every `write_text` of a sidecar or manifest (CRLF on Windows).

### A10. The corpus code digest depends on line endings (low; fixed)

`build_external.code_digest` hashed source bytes, and Git for Windows checks the repository out with CRLF, so
identical code had a different digest per OS: a corpus moved between machines would rebuild, and sidecar
provenance differed. The Metrica and SkillCorner sidecars built here match neither the raw nor the normalised
digest of the committed loaders, so they were built from a working tree whose bytes differed from the commit; the
data are unaffected (two matches rebuilt in memory are identical array by array). Fix `5da7a5a`: the digest
hashes LF-normalised bytes, and sidecars carrying the old raw digest are still accepted, so nothing rebuilds.

### A11. Byte-identical regeneration across macOS and Windows is not assured (medium; J2, J3)

- **Line endings.** A CSV written in text mode on Windows has CRLF line ends (`starter/baseline.write_submission`
  does this). The scorer reads either, but the file hash differs from the macOS file. J2 should write the
  submission with `open(path, "w", newline="\n")` or in binary, and the two-machine hash test should compare the
  submission bytes, not just labels.
- **Floating point.** LightGBM 4.7 is deterministic for a fixed thread count on one platform, but the macOS ARM
  and Windows x86 builds may differ in the last bits of histograms and leaf values, and NumPy's `exp`/`log` used by
  the degrader may differ by an ulp. This run matches the macOS bake-off at every published decimal, which says
  any difference is small, not that there is none. **Recommendation:** in J3, train the final model on one
  reference machine, hash the model files and the submission on both machines, and if they differ, regenerate on
  the reference machine and document the platform in the technical document.

### A12. Frame alignment, half blocks, attack direction: verified

- IDSSE, main versus David's loader on J03WMX and J03WN1: labels inside main's blocks agree on all 287,056
  frames; positions agree to 8e-6 m after the coordinate map, ball and speed likewise, NaN patterns identical;
  attack direction identical in all four halves and equal to DFL's `TeamLeft`. Blocks differ by convention only:
  main uses the kick-off and final-whistle events (starts 0-56 frames later, ends 0-201 frames earlier than the
  tracking), David the tracking extent.
- Metrica Game 3: tracking identical between the two loaders (2e-5 m).
- Attack direction inferred from positions (the features' focal orientation) equals the canonical direction in
  **141 of 141** match-condition pairs of the base cache, corrupted conditions included.
- Work grid: `grid.downsample` keeps every 5th (IDSSE, Metrica) or 2nd (SkillCorner) frame and moves block edges
  inward; the expansion back is A4. At 10 fps, ties of `np.rint` alternate (banker's rounding); harmless.
- Target: `games.half_video_blocks` gives inclusive (11185, 90651) and (110808, 190197), 158,857 graded frames as
  published; `katena.target.atd` maps `frame_id - 1` to the submission row, as DATA.md states.

### A13. Label relabelling: verified

`katena.labels.sandwich` relabels an H or A segment shorter than T seconds (strictly) whose neighbours are both
the other team inside one graded run, to a fixed point (at most 50 passes; one pass clears alternating runs). It
never crosses D or X, works at the native frame rate, and is applied to the native ground truth before both the
training labels (subsampled) and the scoring labels are taken, so both sides share the convention. Merge order is
left to right, a convention choice.

### A14. Degrader seeds and calibration (low; info)

- Seeds: `sha256(match|condition)` per match and condition (`corpus.py:66`), and a separate seed for the training
  orientation (`corpus.py:64`). A match's training rows and its own evaluation input are the same realisation,
  which leaks nothing across folds. Labels enter the degrader only in the dead-coupled evaluation worlds
  (`degrade.py:181`).
- Calibration (`scripts/calibrate_degrader.py:81-84`) uses external data only, but **every** SkillCorner and
  Metrica match: the locked bench `Sample_Game_2` and each held-out fold shape the noise, and Metrica's labels
  select the alive-onset gaps. The effect on any score is negligible; it is worth one line in the technical
  document.

### A15. No feature sees a label: verified

`featurize(m, focal, dirs)` takes positions, ball, teams, frame rate and pitch; `infer_dirs` uses player
positions; M0 and M1 use features only. Labels enter only as training targets (`gbdt.fit_heads`), priors and
scoring.

### A16. X on graded frames: only A4

The decoders emit H, A or D inside every block; outside the blocks the expansion writes X, which costs nothing.
`sc_mode` is scoring-only. With A4 fixed, no X reaches a graded frame on the external matches or the target.

### A17. Scripts that read `data/`: aggregates only, with two caveats

Read as code; none was run on the match.

- `scripts/check_target.py`: PASS/FAIL lines and mismatch counts; exceptions print their type only. Safe.
- `scripts/probe_schema.py`: line shapes, XML tag and attribute names, the homography header (A8 fixed), video
  stream properties. Safe.
- `katena.target.atd`, `katena.target.tagging`: no printing; error messages carry counts. Caveat: `read_tags` and
  the float parse in `read_raw` raise ValueErrors that quote a value; `read_raw` catches its own, and
  `check_target.py` catches the rest, but J2 must call them inside the same guard.
- New here: `scripts/check_target_layout.py` (booleans, counts, channel ids) and `scripts/check_domain_gap.py
  --target` (AUCs and statistic names; medians only with `--detail`). Both print exception types only.
- Upstream, by design not aggregate-only: `starter/load_atd.py` (landmark frames), `load_homography.py`
  (projected pixels), `load_smart_tagging.py` (tagged kick-off frames). Humans only, never paste.
  `evaluation/validate_submission.py` prints label counts and the possession count and mean duration of a
  submission: aggregates, but derived from the match, so share only as needed.

### A18. The headline mean is 71% SkillCorner (low; recommendation)

20 of the 28 equally weighted folds are SkillCorner, which has no dead-ball label (its null possession is
ungraded) and 71.7% of the team head's training rows. A change to dead-ball handling cannot show on those folds.
**Recommendation:** report the 8 IDSSE + Metrica folds beside the 28-fold mean in every gate (M2: 0.569), and gate
dead-ball changes (D2 coupling, J5 tags, J6 cutaways) on those 8.

### A19. SkillCorner "target-like" is not target-like on players (low; D2)

The degrader applies player availability on top of SkillCorner's own losses (26% of in-play frames have no
tracking at all, from broadcast cut-aways), so SkillCorner's realised mean tracked objects are 16.1 / 15.3 / 13.8
at mild / target / severe, against 20.5 on the target. The ball is topped up to the target fraction; players are
not. Worth a line in D2's world design.

### A20. The degrader's positional jitter is far rougher than real automatic tracking (low; D2)

The ported domain-gap alarm (`scripts/check_domain_gap.py --real <external match> --detail`, external only)
separates a real SkillCorner match (1874553) and the Metrica sample (Game 1) from degraded IDSSE matches with AUC
1.00 at every severity, against 0.58-0.69 between degraded matches of the same world. What gives the degrader away:

| statistic (5 Hz windows, median) | SkillCorner 1874553 | Metrica Game 1 | degraded mild / target / severe |
|---|---|---|---|
| per-slot second difference (jitter), m | 0.045 | 0.039 | 0.47 / 0.92 / 1.38 |
| ball second difference, m | 0.21 | 0.04 | 0.93 / 1.82 / 3.0 |
| home minus away present slots | 0 | 0 | 0.82 / 1.24 / 1.64 |

Neither source is the target: SkillCorner is smoothed and extrapolated, the Metrica sample keeps identities. But
both real sources are an order of magnitude smoother than the degrader's i.i.d. jitter (0.15-0.5 m per sample),
which suggests the training worlds are rougher than real tracking on positions. **Candidate for D2:** calibrate
jitter on external automatic tracking (SkillCorner), as the gap and burst constants already are, and test through
the gate. On the target, a human run of `--target` says which world it resembles, as an alarm only, never to tune
the degrader.

### A21. The base cache's READY named the wrong commit (low; fixed)

`run_m2.py` read git HEAD when it wrote READY, after a 31-minute run during which this branch moved on. READY was
corrected by hand with the coordinator's approval (8537084, clean, with a note); nothing else in `work-base` was
touched. `b4cc087` captures the commit at start.

## Environment notes for this machine

- `uv sync` fails with "Missing expected target directory for Python minor version link": the Claude app's MSIX
  virtualisation of `%APPDATA%` hides the junction's target from uv. `uv sync --python
  C:\Users\User\AppData\Roaming\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe` (with
  `UV_PYTHON_DOWNLOADS=never`) works; then run `.venv/Scripts/python.exe` directly.
- GNU tar in Git Bash reads `D:` as a host name; use `--force-local` or `/d/...`.

## Commands for a human to run on the real match

All print booleans, counts, rates, AUCs or statistic names only; exceptions print their type only.

```bash
uv run python scripts/check_target.py           # loader and published-baseline reproduction (J1)
uv run python scripts/check_target_layout.py    # A7: does the ATD layout match the reader's assumptions?
KATENA_EXTERNAL=D:/katena/external_data uv run python scripts/check_domain_gap.py --target   # A20 alarm
```

`check_domain_gap.py --target` loads the target, downsamples it to 5 Hz and compares it with three degraded IDSSE
development matches; it writes nothing and fits nothing. Its output is an alarm, never a reason to change the
degrader or to choose between versions (PLAN.md, "Validating blind", step 5).
