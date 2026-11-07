# External training corpus

Public data only; nothing here touches `data/`. Built by `scripts/build_external.py` into
`external_data/canonical/<dataset>/<match>.npz` (gitignored) as `katena.types.Match`. Labels are the
**raw source labels** (H=0, A=1, D=2, X=3); no harmonisation has been applied. Profile with
`scripts/profile_external.py`.

## Conventions for every dataset

- Metres, origin at a pitch corner, x in [0, length], y in [0, width] (the challenge's 0-1 ATD
  convention times the pitch). Centre-origin sources are shifted by (L/2, W/2). Orientation of y is
  not mirrored; `attack_dir[half]` (+1 = home attacks towards larger x) is stored per half and
  checked against the data (see each section).
- `half_blocks` are the in-play windows, like the challenge's graded halves. Everything outside is
  `y = 3 (X)`. Frame indices are kept native (IDSSE: real-time grid including the half-time gap).
- `fps` is the source rate (IDSSE/Metrica 25, SkillCorner 10). Nothing is resampled.
- Hard checks on every match (`validate_canonical`, also run by the tests): shapes/dtypes, halves
  ordered, attack direction flips at half-time, X outside blocks, no speed above 14 m/s, and the
  **nearest-team check**: on labelled alive frames with a ball, the nearest player belongs to the
  labelled team more than 60% of the time (an inverted H/A mapping would give about 25%).
- Raw files are SHA256-pinned in `scripts/external_manifest.json` (first download pinned, later
  runs verify). Each canonical file has a `.json` sidecar with the input hashes, code hash, and all
  data-quality figures quoted below.
- `speed` is source speed in m/s where the source has it (IDSSE), otherwise derived from positions by
  central difference and blanked above 12.5 m/s (position jumps). Derived speed is noisy.

## 1. IDSSE / DFL open data

- Source: https://doi.org/10.6084/m9.figshare.28196177 (API `https://api.figshare.com/v2/articles/28196177`;
  the web page returns 403, the API works). Paper: Bassek, Rein, Weber, Memmert, *An integrated
  dataset of spatiotemporal and event data in elite soccer*, Sci Data 2025, doi 10.1038/s41597-025-04505-y.
- Licence: CC BY 4.0. Credit the authors, DFL and Sportec Solutions.
- 7 matches, 2022/23 (Bundesliga matchday 34: J03WMX, J03WN1; 2. Bundesliga matchdays 6-17: the other five): J03WMX, J03WN1, J03WOH, J03WOY,
  J03WPY, J03WQQ, J03WR9. 21 XML files, 2.5 GB (figshare MD5 verified, SHA256 pinned).
- Who labelled, how: tracking is TRACAB Gen5 (16-20 cameras, 25 Hz, optical, positions checked by
  live operators and a final quality check). `BallPossession` (1 home, 2 away) is "defined by the
  player in control of the ball" and `BallStatus` (0 inactive, 1 active) is inactive "when the match is
  interrupted by the referee" (paper, Box 3). Events come from trained human operators (5 per
  match). The paper does not say how the per-frame possession field is derived from them.
- Used: matchinformation (teams, roles, pitch, playing times), events (KickOff with
  `TeamLeft`/`GameSection`, `FinalWhistle`, cards), positions (players, ball).
- Changes: centre to corner origin; speed km/h to m/s; referees dropped; one slot per person across
  both halves (29-32 slots; subs and red cards show as NaN, see below); real-time grid from the
  `T` stamps (includes the half-time gap, 24-28k NaN frames, X); in-play block = kick-off event to
  final-whistle event of each half; `y`: `BallStatus==0` -> D, else `BallPossession` 1 -> H, 2 -> A.
- Verification (all 7 matches):
  - Clock: `T` advances 40 ms per `N` on every ball frame (0 bad); no duplicate or non-monotonic
    frames; pitch 105 x 68 in both files.
  - Dead = BallStatus 0 confirmed: our alive seconds equal the match file's `PlayingTime` per half
    within 0.05 s in all 14 halves.
  - Team mapping: nearest-team agreement 0.75-0.81; mean distance to nearest labelled-team player
    2.7-3.0 m against 5.9-7.4 m for the other team. 1 = home confirmed.
  - Attack direction: `TeamLeft` from the KickOff event equals the team-centroid direction in all
    14 halves; the goalkeeper heuristic agrees in 13 and is ambiguous in J03WR9 half 2.
  - Substitutions and red cards: active slots 22 for most frames; two red cards (J03WN1, J03WQQ) give
    21 after the card. Starting slots with fewer frames are substituted players.
  - Ball: never NaN. Outside the pitch on 17-46k frames per match, but 99% of those are dead-ball
    frames (up to 3 m beyond the lines); under 1% of alive frames. Ball-outside-pitch is
    therefore a strong dead-ball cue. 0 frames blanked as glitches (> 15 m outside).
- Known defects:
  - Dead share inside the blocks is 41-52% of frames, but the paper calls inactive only
    referee interruptions; throw-ins, goal kicks etc. are evidently included.
  - Alive frames outside the in-play window (marked X): before the KickOff event 56 and 47 frames in
    J03WN1 (otherwise 0-1); after the FinalWhistle event up to 118 frames (J03WMX), so the whistle event
    leads the status flip by up to 4.7 s.
  - J03WN1 second half frame 100000: ball at (-20, -21) m for one frame, before kick-off.
  - J03WR9 home goalkeeper: median x is 21-24 m from his own goal line in both halves, which defeats the
    goalkeeper heuristic. Probably a very high-standing keeper or identity swaps. Not investigated.
  - The paper warns of identity swaps (not detectable here; we do not use identities).
  - Raw `S` of the ball is stale on the first frame and meaningless as a ball speed; we keep player speed only.
  - Referee framesets exist (3 per half), dropped.

## 2. Metrica sample data

- Source: https://github.com/metrica-sports/sample-data, pinned commit `e706dd5`.
- Licence: **none stated**. The README asks users to "be responsible" and to "acknowledge the source"
  for anything public. Credit: Metrica Sports sample data. Treat as non-redistributable (we do not commit it).
- 3 games, anonymised. Games 1 and 2: CSV tracking (home and away files) plus CSV events. Game 3: FIFA
  EPTS tracking and XML metadata plus JSON events, the same layout as the competition ATD, except that
  it carries no speed channels (the competition ATD has `_s`), and 11 `DataFormatSpecification`
  roster segments that change the column set (parsed per segment).
- Who labelled, how: Metrica's own event tagging (the repo documents the event types in
  `documentation/events-definitions.pdf`, which we could not render here; not read). **There are no
  possession or dead-ball labels.** We derive them: `label_source = "derived_events"`.
  - Ball is D from a dead trigger (BALL OUT, FAULT RECEIVED, CHALLENGE with FAULT, BALL LOST with
    OFFSIDE/HAND BALL, PASS OFFSIDE, SHOT with GOAL or OUT) until the next restart (any SET PIECE, or a PASS
    GOAL KICK).
  - Otherwise the team of the last PASS, CARRY, RECOVERY or SET PIECE.
  - X outside [first event, last event end] of each period.
  - Two deviations from the brief, both needed: a goal-kick PASS ends D (Metrica tags goal kicks as PASS,
    so D would otherwise run to the next set piece), and goals/shots out start D.
- Changes: 0-1 to metres with 105 x 68; slots with no data dropped; derived speed.
- Verification: frames contiguous (0 gaps) and `Frame` = 25 Hz; Game 3 metadata attack direction
  matches the positions in half 1; for Games 1-2 the two heuristics (team centroid, goalkeeper) agree
  (no source statement exists); nearest-team agreement 0.80-0.83 (labelled-team distance 1.9-2.4 m vs
  5.7-6.1 m); active slots 22, with a few frames of 23 in Games 1-2.
- Known defects:
  - Ball is NaN on 38%, 41% and 32% of in-play frames (genuinely noisy, like the target).
  - Labels are approximate: event timestamps are not verified against the ball. Dead spans are long
    (median 28, 22, 13 s; 99th percentile 136, 85, 91 s), because a restart event can be missing. 135, 77,
    154 PASS/CARRY/RECOVERY events fall inside a D span and are ignored.
  - D share 51%, 44%, 35% (Game 3 differs from the other two, a tagging-style difference).
  - Spells follow event density, not ball control; do not read them as DFL-comparable.
  - In-play blocks are event-defined, so they start 0-360 frames after the first tracked frame.
  - Position jumps: 399 / 976 / 573 slot-frames had derived speed above 12.5 m/s and were blanked.
  - Game 2 has a track named "Player 26" (space) in the CSV header; handled.

## 3. SkillCorner open data

- Source: https://github.com/SkillCorner/opendata, pinned commit `4340d27`.
- Licence: MIT (c) 2020 SkillCorner; the README asks to credit SkillCorner.
- **20 matches** in the repo (A-League 2024/25), although the README text says 10. The tracking files
  are Git LFS objects (100 MB each), fetched through the public LFS batch API without an account and
  checked against the LFS SHA256 oid.
- Who labelled, how: broadcast video, computer vision and machine learning (README). The tracking
  frame carries `possession.group` ("home team", "away team", or null) and `possession.player_id`.
  How the group is produced is not documented in the repo.
- Used: `match.json` (teams, sides per period, pitch, periods) and the tracking jsonl. Dynamic events
  and phases-of-play CSVs were only used to investigate null possession, not for labels.
- Changes: centre to corner origin (pitch is 104 x 68 here, per match); 10 fps native; extrapolated
  positions kept as supplied, flags kept in `extras` (`ball_detected`, `player_detected`:
  1 detected, 0 extrapolated, -1 no data); `y`: group home -> H, away -> A, null -> **X**.
  **No D is ever emitted.**
- What null means (our reading, not documented): pooled over 20 matches, 46% of in-play frames are X.
  Of those, 47% have tracking and 53% have none (broadcast cut-aways). Among null frames with a ball:
  it is outside the pitch on 40% (0.6% for labelled frames) and the median ball speed is 1.3 m/s
  (5.5 m/s labelled). Null runs with data have median 6.8 s (10th-90th: 1.4-17 s). This is consistent
  with "ball out of play, stopped or uncontrolled", but 60% of null frames have the ball inside the pitch,
  so it is not a dead-ball label and is not used as one.
- Verification: `home_team_side` from `match.json` equals the team-centroid direction in all 40
  periods; match `id` links to `player_data.player_id`; home/away mapping: nearest-team agreement
  0.70-0.81 (lower than DFL: extrapolated positions) and our labelled minutes match the file's
  `minutes_tip` per team (mean absolute difference 0.9 min, worst 3.1 min).
- Known defects:
  - 11-20k in-play frames per match (about 26%) have no tracking at all; the ball is NaN on 19-34% of
    in-play frames, players on the same frames; about 43% of tracked player positions are extrapolated
    (detected share 49-66% per match); the ball is detected on 80% of tracked frames (75-85%).
  - `match_periods.end_frame` is one frame past the last tracked frame in 6 matches (block clipped).
  - Group can be non-null on frames that have no tracking (578 in 1886347: carried forward); labels kept.
  - Players: 22 per tracked frame always, so active-slot counts cannot show substitutions or red cards.
  - Speed is derived at 10 fps from partly extrapolated positions: noisy.
  - 5 matches (2011166 onwards) include pre-match video: period 1 starts at frame 1170-4040.
  - Spell statistics depend on how X is treated (below).

## Profile (pooled; `scripts/profile_external.py`)

Spells are the scorer's (`possession_spells`, maximal H or A runs inside X-free runs).

| | matches | spells | mean s | median s | < 1 s | < 2 s | < 5 s | dead share | ball NaN |
|---|---|---|---|---|---|---|---|---|---|
| IDSSE (raw) | 7 | 3061 | 7.2 | 3.6 | 12.6% | 30.0% | 60.6% | 44.8% | 0% |
| Metrica (derived) | 3 | 873 | 11.0 | 6.5 | 4.2% | 17.2% | 42.3% | 43.4% | 36.7% |
| SkillCorner, X breaks spells | 20 | 6684 | 9.6 | 5.6 | 10.8% | 22.4% | 46.2% | none | 25.6% |
| SkillCorner, X forward-filled | 20 | 5180 | 22.8 | 11.4 | 6.5% | 14.3% | 30.4% | none | 25.6% |

Mean spell (s) after the sandwich merge (opponent flips shorter than T relabelled to the surrounding team):

| | raw | 0.5 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|---|
| IDSSE | 7.2 | 7.6 | 8.2 | 9.8 | 11.4 | 13.8 |
| Metrica | 11.0 | 11.1 | 11.5 | 12.9 | 14.6 | 17.2 |
| SkillCorner (X breaks) | 9.6 | 9.8 | 10.5 | 11.8 | 12.7 | 14.3 |

IDSSE team changes: 2351 direct H<->A, 489 via D with a change of team, 207 via D to the same team.
Dead stretches in IDSSE: median 19.9 s, 90th percentile 49.5 s.

## Differences from docs/PLAN.md

- "DFL means 4-6 s per spell": measured 7.2 s pooled (5.5-9.2 per match), median 3.6 s. A sandwich of
  about 5 s is needed to reach the 14.2 s of the public window (13.8 s).
- BallStatus 0 is about 45% of in-play DFL frames (41-52%); it includes ball out of play, not only
  referee interruptions.
- Metrica Game 3 has no speed, unlike the competition ATD; its labels are derived, not given.
- SkillCorner has no dead-ball label at all, and 20 matches rather than 10.
