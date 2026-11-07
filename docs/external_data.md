# External data (D1): loaders, label harmonisation, corpus

Everything here is **external open data from other matches**. Nothing reads `data/` or the
competition match. Code: `src/katena/external/`. Tests: `tests/test_external_*.py`.

## Reproduce

From the repository root, with the team venv:

```
python scripts/external_corpus.py build      # download, convert, cache (about 5 min, idempotent)
python scripts/external_corpus.py report     # per-match spell statistics, raw vs harmonised
python scripts/external_corpus.py evidence   # derivation of harmonise's min_control_s
python -m pytest -q                          # corpus tests run only if the cache exists
```

The corpus lives outside the repo, shared by all worktrees:
`C:/Users/User/Desktop/barca/external_data/` (override with `KATENA_EXTERNAL_DATA`).

| path | content | size |
|---|---|---|
| `cache/<name>.npz` | one compressed `Match` per match, **raw** provider labels | 8 files, 185 MB |
| `raw/idsse/<id>/<id>_{meta,events}.xml` | DFL match information and events, kept | 5 MB |
| `raw/metrica/game3/Sample_Game_3_*` | Metrica EPTS pair and event JSON, kept | 53 MB |

Total 243 MB. The DFL tracking XMLs (350-420 MB each, 2.6 GB in all) are deleted after
conversion. They can be downloaded again, and the conversion is deterministic.

```python
from katena.external.corpus import load_corpus
from katena.external.harmonise import DEFAULT_HARMONISE
ms = load_corpus(harmonise_cfg={})   # y harmonised with the default rule, raw labels in meta["y_raw"]
```

## Sources and conversion

All sources are converted to the shared `Match` (`src/katena/match.py`): centred metres, `team`
0 home / 1 away, `y` 0 H / 1 A / 2 D / 3 X, inclusive `half_blocks`, and `attack_dir[half]` = +1
when home attacks +x. Each half becomes one block. **250 X rows (10 s) are inserted between the
halves**, so the scorer's `valid_runs` sees one run per half. `meta["frame_id"]` maps every row
back to the source frame counter (-1 in the gap).

| | IDSSE / DFL (7 matches) | Metrica open sample Game 3 |
|---|---|---|
| licence | CC BY 4.0, Bassek et al. 2025, Sci Data 12:195 | Metrica Sports sample data |
| loader | `idsse.load_idsse` (streams the positions XML with lxml) | `epts.load_epts` (generic EPTS) + `metrica_open.load_metrica_game3` |
| download | figshare file ids from kloppy 3.18.0 `get_IDSSE_url`, fetched from `ndownloader.figshare.com`. kloppy's `springernature.figshare.com` host returns a bot challenge. | `raw.githubusercontent.com/metrica-sports/sample-data` |
| labels | `BallStatus` 0 gives D. Otherwise `BallPossession` 1 gives H and 2 gives A. | derived from events, see below |
| home / away | `Team Role="home"/"guest"`. The guest team is listed **first** in these files. | `Score@idLocalTeam`, else the first Team (kloppy and `starter/baseline.py` use the same convention) |
| coordinates | centred metres. **y flipped** (DFL y points up) to match EPTS after `from_normalised` (y points down). | normalised 0-1 converted to centred metres with FieldSize 105 x 68 |
| speed | provider `S` is km/h and is stored as m/s. Checked: the median ratio to position-derived speed is 1.000. | no speed channel, so it is derived from positions (central difference) |
| attack direction | position-based (below), cross-checked against the `KickOff@TeamLeft` event: **14/14 halves agree** | position-based, cross-checked against `attack_direction_first_half`: **2/2 agree** |

**Attack direction** (`common.detect_attack_dir`): the median over the half of (home centroid x
minus away centroid x). Each centroid contains that team's goalkeeper at its own goal, so the
team attacking +x sits behind the other on average. The first 10 s of each half (kickoff layout) is
a second signal. It agreed with the first in every half.

**The EPTS loader is generic.** It reads the layout from the metadata's
DataFormatSpecifications through kloppy's EPTS reader. Game 3 has 11 of them, because the
layout changes at substitutions. It writes straight into numpy arrays and skips kloppy Frame
objects, which takes 7 s for Game 3. It accepts `half_blocks` in source rows, `half_gap`, and a
`pitch` override, so the competition ATD loads with
`load_epts(meta, raw, half_blocks=games.half_video_blocks(...), pitch=(105.3, 68.0))`. **This
overlaps Justin's J1 ATD loader.** J1 owns the competition loader, and the two should be
reconciled when J1 lands.

### Metrica Game 3 labels from events

The tracking carries no possession flags, so `metrica_open.labels_from_events` derives them:

- **Restarts:** every `SET PIECE`, plus the `GOAL KICK` subtype, which Metrica tags on a PASS or
  BALL LOST.
- **Start of a stoppage:** one frame after the earliest stop marker that follows the last in-play
  action. Markers are BALL OUT, SHOT GOAL/OUT, OFFSIDE, END HALF, FAULT RECEIVED, a FAULT
  challenge without ADVANTAGE, and HAND BALL. Without a marker (20 of 101 stoppages), the
  stoppage starts at the end of the last action plus one frame.
- **Possession:** the team of the last on-ball event (PASS, CARRY, SHOT, RECOVERY, BALL LOST, SET
  PIECE). A loose ball stays with the last team until the other side's RECOVERY.

**Validation against the tracking:** Game 3's ball is NaN on 99.2% of the event-derived D frames
and present on 98.0% of alive frames. Each of the 103 stoppages starts on the frame the ball
disappears (median offset 0 after the +1 shift, p95 1.0 s). It ends on the frame the ball
reappears, with offset 0 at p5 through p95.

## Label-semantics findings

1. **DFL names a team on dead frames, and it is the restarting team.** On the last dead frame of
   699 stoppages, `BallPossession` equalled the team that restarted 99.7% of the time. This
   matches CHALLENGE.md ("whoever will restart"). It is kept as `meta["owner_raw"]` and never
   enters `y`. It is free supervision for "who restarts", which J5 could use.
2. **DFL starts dead ball when the ball crosses the line.** 549 of 811 dead onsets had the
   ball outside the lines in the preceding 3 s, with a median lag of 0.16 s (p90 0.48 s). The
   other onsets are fouls and offsides.
3. **In Metrica's open tracking, the ball is removed while it is dead**, almost exactly (see
   above). The competition ground truth is "derived from corrected tracking" by Metrica, so the
   corrected tracking may encode dead ball the same way. That is a guess worth one organiser
   question. **Caution for D2 and D4:** in Game 3 a ball-missing feature is an almost perfect D
   detector. DFL has the ball on 100% of frames, and the competition ATD lacks it on 41% of
   graded frames because of occlusion. Ball presence in external data must be resimulated by the
   noise simulator before training, or the model will learn the wrong cue.
4. **DFL halves start at the kickoff with the ball already alive** (frame N = 10000 or 100000), so
   DFL has no pre-kickoff D. Game 3 has 14 s (half 1) and 17 s (half 2) of pre-kickoff D, like the
   competition's second-half trap. Train kickoff handling on Game 3 or on the simulator, not on
   DFL.
5. **There is more dead ball in DFL than in Metrica:** 41-52% of the halves (39-49 min per match),
   with a median stoppage of about 20 s. Game 3 has 31% (30 min) and a median of 12 s.
   Stoppage counts are similar, about 100 per match.
6. **DFL raw spells average 7.2 s** (5.5-9.2 s per match). PLAN.md says 4-6 s. 30% of raw
   spells last under 2 s. Metrica's event-derived spells are also touch-level: a mean of
   9.8 s, with 28% under 2 s.
7. **DFL event clocks are offset from tracking by -1.2 to +1.7 s per match.** After a per-match
   shift, a Play's team matches tracking `BallPossession` on 92-96% of events. Any join between
   events and frames must estimate this shift (`control_evidence.dfl_action_frames`).

## Harmonisation

**Rule** (`harmonise.harmonise`): an opponent spell shorter than `min_control_s` that starts
straight from the current owner's spell is relabelled to the owner. A spell with no dead ball
before it can only end in one of two ways. Either the owner gets the ball back ("it came back"),
or the ball goes dead or the half ends ("it went out"). So the rule says exactly this: a brief
flip that returns or dies is not a change. Two more parts:

- **Restart protection.** A spell that directly follows dead ball, or starts a half, is the
  restart possessor and is never removed. CHALLENGE.md: "whenever play restarts after a
  stoppage, there is a clear possessor".
- **Shortest first.** For H | A 1.0 s | H 0.3 s | A 10 s, the 0.3 s H touch is discarded first,
  so A keeps the ball from its first touch. The result does not depend on scan direction.
  `order="causal"` is available for comparison.

D and X frames are never changed, so every stoppage boundary is preserved.

**Parameter, from the written definition and external events, not from the target match.**
CHALLENGE.md's proxy is "a player keeps the ball for a few seconds". To turn "a few" into a
number, I take every removable flip in all 8 external matches. For each one, I ask the
provider's own event data whether that team made a deliberate on-ball action during the flip:

- DFL: a `Play` (pass or cross), `ShotAtGoal` or `BallClaiming`.
- Metrica: a completed PASS, a SHOT, or a CARRY of 1 s or more.

Touches, clearances (`OtherBallAction`) and duels do not count. I then pick the duration that
best separates flips with an action from flips without one (`python scripts/external_corpus.py
evidence`):

| duration of removable flip | flips | share with a deliberate action by that team |
|---|---|---|
| 0 - 0.5 s | 173 | 0.10 |
| 0.5 - 1 s | 243 | 0.09 |
| 1 - 1.5 s | 260 | 0.17 |
| 1.5 - 2 s | 220 | 0.20 |
| 2 - 3 s | 349 | 0.34 |
| 3 - 4 s | 241 | 0.48 |
| 4 - 6 s | 331 | 0.72 |
| 6 - 10 s | 328 | 0.90 |
| 10 s and over | 511 | 0.99 |

| match | flips | clock shift s | Play-owner agreement | best t (own) | best t (others, LOMO) | held-out accuracy |
|---|---|---|---|---|---|---|
| idsse-J03WMX | 408 | +0.16 | 0.935 | 2.7 | 3.5 | 0.801 |
| idsse-J03WN1 | 394 | -1.16 | 0.922 | 3.7 | 3.5 | 0.805 |
| idsse-J03WOH | 301 | +1.32 | 0.932 | 3.8 | 3.7 | 0.814 |
| idsse-J03WOY | 344 | +1.72 | 0.929 | 3.5 | 3.8 | 0.770 |
| idsse-J03WPY | 324 | +0.36 | 0.943 | 5.0 | 3.7 | 0.802 |
| idsse-J03WQQ | 302 | +1.20 | 0.957 | 3.9 | 3.8 | 0.851 |
| idsse-J03WR9 | 278 | +0.80 | 0.935 | 3.5 | 3.8 | 0.863 |
| metrica-game3 | 305 | n/a (frame-synced) | n/a | 2.7 | 3.5 | 0.830 |

Pooled over 2656 flips, the accuracy-best threshold is 3.7 s (accuracy 0.819). The match
bootstrap gives a median of 3.7 s and a 90% interval of [2.7, 4.0] s. The optimum is flat:

| t (s) | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|
| pooled accuracy | 0.770 | 0.802 | 0.813 | 0.819 | 0.817 | 0.805 | 0.791 |

**Chosen: `min_control_s = 3.5 s`**, the pooled optimum rounded to 0.5 s. It is inside every
leave-one-match-out estimate (3.5-3.8 s). The evidence uses the weaker of the two written proxies,
one deliberate action. Requiring "more than two consecutive passes" would push the threshold up.

**Sanity reference only.** DATA.md publishes the public window as 17 possessions of mean 14.2 s.
Harmonised external means are 13.9 s (DFL) and 16.7 s (Metrica), against raw means of 7.2 s and
9.8 s. With 17 heavy-tailed spells, that number has a standard error of a few seconds, so it can
rule out the raw labels but cannot choose between thresholds of 2.5-5 s. It was not used to choose
anything. The ground-truth CSV was not opened for this work. The public `_meta.json` was read once
while reviewing the scorer: it holds the same published numbers, and nothing from it was used.

Sensitivity of pooled spell statistics to the threshold (0 = raw):

| min_control_s | DFL spells/match | DFL mean s | DFL median s | Game 3 spells | Game 3 mean s | Game 3 median s |
|---|---|---|---|---|---|---|
| 0 (raw) | 437 | 7.23 | 3.84 | 406 | 9.76 | 4.66 |
| 1.0 | 366 | 8.64 | 5.00 | 341 | 11.62 | 6.56 |
| 2.0 | 292 | 10.83 | 6.90 | 290 | 13.66 | 8.20 |
| 3.0 | 245 | 12.89 | 8.68 | 256 | 15.48 | 10.32 |
| **3.5** | **228** | **13.85** | **9.72** | **238** | **16.65** | **11.48** |
| 4.0 | 217 | 14.54 | 10.44 | 230 | 17.23 | 12.14 |
| 5.0 | 197 | 16.02 | 11.42 | 209 | 18.96 | 13.92 |

DFL medians are the median of the per-match medians.

## Spell statistics per match (raw vs harmonised at 3.5 s)

`python scripts/external_corpus.py report`. `frames` includes the 250-row gap. `graded` is frames
inside the halves. The ball-missing and dead shares are over graded frames. Durations come from
`possession_metrics.possession_spells`, and the mean from `possession_stats`.

| match | frames | graded | ball missing | dead share | labels | spells | per alive min | mean s | median s | p10 s | p90 s | < 2 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idsse-J03WMX | 146217 | 145967 | 0.0% | 41.5% | raw | 511 | 8.98 | 6.68 | 3.00 | 0.68 | 15.28 | 37.4% |
|  |  |  |  |  | harmonised | 231 | 4.06 | 14.78 | 9.84 | 3.48 | 29.88 | 5.2% |
| idsse-J03WN1 | 141811 | 141561 | 0.0% | 51.7% | raw | 500 | 10.97 | 5.47 | 3.12 | 0.80 | 12.08 | 34.2% |
|  |  |  |  |  | harmonised | 251 | 5.51 | 10.90 | 7.36 | 2.76 | 22.48 | 4.8% |
| idsse-J03WPY | 146461 | 146211 | 0.0% | 44.3% | raw | 421 | 7.75 | 7.74 | 3.84 | 0.72 | 20.60 | 29.5% |
|  |  |  |  |  | harmonised | 225 | 4.14 | 14.48 | 10.28 | 3.88 | 32.05 | 2.7% |
| idsse-J03WOH | 137464 | 137214 | 0.0% | 47.6% | raw | 402 | 8.38 | 7.16 | 3.94 | 0.88 | 18.54 | 27.4% |
|  |  |  |  |  | harmonised | 215 | 4.48 | 13.39 | 9.72 | 3.66 | 28.74 | 3.7% |
| idsse-J03WQQ | 142595 | 142345 | 0.0% | 40.7% | raw | 401 | 7.13 | 8.41 | 4.68 | 1.08 | 21.28 | 25.2% |
|  |  |  |  |  | harmonised | 213 | 3.79 | 15.84 | 10.80 | 4.13 | 32.82 | 0.9% |
| idsse-J03WOY | 142786 | 142536 | 0.0% | 46.5% | raw | 456 | 8.97 | 6.69 | 3.60 | 1.04 | 17.58 | 28.1% |
|  |  |  |  |  | harmonised | 245 | 4.82 | 12.45 | 8.04 | 3.60 | 26.22 | 2.4% |
| idsse-J03WR9 | 147060 | 146810 | 0.0% | 41.9% | raw | 370 | 6.51 | 9.22 | 4.24 | 0.96 | 23.60 | 25.4% |
|  |  |  |  |  | harmonised | 217 | 3.82 | 15.73 | 9.32 | 3.58 | 35.69 | 3.7% |
| metrica-game3 | 144011 | 143761 | 32.2% | 31.1% | raw | 406 | 6.15 | 9.76 | 4.66 | 0.80 | 26.70 | 27.6% |
|  |  |  |  |  | harmonised | 238 | 3.60 | 16.65 | 11.48 | 3.42 | 38.36 | 4.2% |

| source | labels | matches | spells | mean s | median s | spells per alive min |
|---|---|---|---|---|---|---|
| IDSSE/DFL | raw | 7 | 3061 | 7.23 | 3.84 (median of matches) | 8.38 |
| IDSSE/DFL | harmonised | 7 | 1597 | 13.85 | 9.72 (median of matches) | 4.37 |
| Metrica Game 3 | raw | 1 | 406 | 9.76 | 4.66 | 6.15 |
| Metrica Game 3 | harmonised | 1 | 238 | 16.65 | 11.48 | 3.60 |

The spells under 2 s that remain after harmonising are protected restart spells, for example a
throw-in lost straight away.

## Caveats and next steps

- Harmonisation fixes the *training labels*. Whether 3.5 s is right for scoring should be
  confirmed by the leave-one-match-out harness (J2) once a model exists: train with harmonised
  labels at 2.5 / 3.5 / 4.5 s and score each held-out match against its harmonised labels at
  3.5 s. A loop that changes both sides at once would only be measuring itself.
- The evidence definition is coarse. A Play includes failed first-time passes, and DFL Play
  times carry about 0.2-0.5 s of clock noise after the shift. A stricter variant (completed passes
  only, or two or more actions) is an easy follow-up.
- `meta["attack_dir_diag"]` and `meta["diag"]` have integer half keys before caching and string
  keys after it, because the npz cache stores meta as JSON.
