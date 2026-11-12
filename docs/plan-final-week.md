# Final week plan (written 2026-11-10, before any of the work below)

Deadline: 2026-11-17 23:59 CET. Treat Nov 16 as the real deadline. The submission form arrives by email by Nov 14.

## Where we are

- **Adopted model F** (D1 touch features + D3 LightGBM/TCN blend at w = 0.5 + D2 `d2j` mix; Viterbi tau 2, T = 2).
  On 28 external leave-one-match-out folds it gains +0.0158 [+0.0111, +0.0205] over M2, winning 22/28. On the one
  locked-bench read it beats M2 on both benches.
- **The pipeline is on main:** `make submit`, plus the Windows steps in `docs/REPRODUCE.md`. A dressed external match
  scores the same through the submission path and the evaluation path. On Windows the CSV is byte-identical run to
  run. Between Windows and macOS the LightGBM model is byte-identical and the TCN differs in its last bits.
- **Not done:** the real run (`docs/real-run.md`, David only), the team name, the `submission-v1` tag, and three
  disclosure choices in `docs/technical-document.md`.

## Rules for this week (unchanged, restated so nothing drifts)

- **No new look at the target.** Nothing is trained, tuned or chosen on the provided match or the public window.
  The public window is a pass/fail pipeline check only.
- **Candidates are fixed here, before any score: G1 and G2 below.** Each must pass the same gate against F on the
  28 folds: at least 70% wins and a paired 95% bootstrap interval above 0 at `target`, and not worse at `severe`.
  - If both pass, the combination G12 is gated the same way.
  - The outcome is fixed in advance, not chosen by score: adopt G12 if it passes, otherwise G1 if it passed,
    otherwise G2 if it passed, otherwise keep F.
  - Nothing new is adopted after **Nov 14 18:00 CET**. A later pass is reported, not used.
- **The locked benches are spent.** No further bench reads.
- These are the 5th and 6th packages gated on the same folds. The technical document will say so.

## Workstreams (run in parallel by subagents; heavy jobs share the two `heavy.py` slots)

| # | Workstream | Done means | CPU |
|---|---|---|---|
| W1 | **Fresh-clone rebuild on Windows (J3).** Clone the repo into a new folder, rebuild the external corpus from a fresh download, build the cache from empty, train, and predict a dressed external match | Every model hash and the dressed CSV hash equal those from `work-submit`, or the first differing stage is named | 1 slot, about 2 h |
| W2 | **Cross-OS reproduction in CI (J3).** A `workflow_dispatch` workflow that builds the external corpus and the cache on GitHub runners, trains F on all 30, predicts the same dressed external match and uploads the hashes | Linux (and macOS if it fits) hashes compared with Windows, model and CSV. Do the labels agree across machines? | GitHub runners only |
| W3 | **Gate G1: the TCN member also reads the 53 touch columns** (213 columns instead of 160). Everything else as in F | Pre-registration entry pushed before any score, then the gate table and the decision | 1 slot, about 2 h |
| W4a | **Kickoff from Smart Tagging.** The FAQ says the second half is graded before its kickoff, and those frames are all D. In each graded half, frames before that half's `Kick Off (Start)` tag (Metrica's automatic tagging) become D. No frame constant | A tested rule in `katena.pipeline` that the human run applies. A human check prints pass/fail: the tag-derived kickoff is within 2 s of the kickoff frame published in FAQ.md | none |
| W4b | **Gate G2: set-piece tag features (J5-lite).** Features from Smart Tagging `SET PIECES` intervals: time since and until, type, and tagged team relative to the focal team. On external matches the tags are imitated from real external event data (IDSSE and Metrica set-piece events), with the real tags' recall and timing noise simulated, never from possession labels | Pre-registration pushed before any score, then the gate. Cut if not gated by Nov 13 | 1 slot after W1/W3 |
| W5 | **One command for the real run.** `scripts/real_run.py` runs, for Windows without `make`: the human checks, `make_submission.py all`, the validator, the public-window check, the SHA-256, and an aggregates-only run report for the decision log. It also builds the technical document's PDF from the Markdown | Tested on a synthetic ATD bundle; `docs/real-run.md` points to it | none |
| W6 | **Integrate whatever is adopted** (only if G1, G2 or G12 is adopted): the pipeline trains it, the tests and expected hashes are updated, the technical document is updated | Contingent, Nov 13-14 | 1 slot |

Every workstream works on its own branch and opens a PR. The coordinator verifies and merges once CI is green.

## For the humans

- **David:**
  - Register the team name today (every member, same name). Then run `docs/real-run.md` with F now, as an
    insurance file.
  - Re-run it only if W4a or an adopted candidate changes the model or the pipeline.
  - Decide the three disclosure points.
- **Justin:** on the Mac, run `python scripts/cache_digests.py dump`, and
  `python scripts/dressed_check.py label --bundle <your bundle.pkl>`. Send the hashes (not the data) so the
  cross-machine label check covers macOS. Send or confirm the organiser questions in `docs/PLAN.md`.

## Status, 2026-11-10 evening (every workstream finished the same day)

| # | Result | PR |
|---|---|---|
| W1 | A fresh clone with an empty cache rebuilt every stage byte for byte, including both models and the dressed CSV. 1 h 39 min with the download | #20 |
| W2 | The dressed CSV is byte-identical in 11 trainings on 7 OS/CPU combinations. TCN bytes differ by OS and CPU vendor; LightGBM, scaler and priors are identical. `repro.yml` can be dispatched from main | #21 |
| W3 | G1 fails: +0.0017 [-0.0014, +0.0047], 16/28 | #19 |
| W4a | Kickoff from Smart Tagging, in `katena.pipeline`. The human check fails only on a late tag | #18 |
| W4b | G2 fails: +0.0046 [+0.0014, +0.0079], 16/28. On the 8 tagged folds +0.0229, 8/8, but the gate was registered on 28; see the decision log's design note | #18 |
| W5 | `scripts/real_run.py --team <name>`: the whole real run in one command | #17 |
| W6 | Not needed: nothing was adopted, so **F stays the submission model**. The kickoff rule changes `predict`, not the model hashes | |

**Left for the humans:** the team name (registration closes today), then the real run on David's PC with
`.venv\Scripts\python.exe scripts\real_run.py --team <team_name>`, the three disclosure choices in the technical
document, the `submission-v1` tag, and the form.

## Status, 2026-11-11: a candidate from outside this plan

Justin's branch was pushed on the evening of Nov 10 (PR #23). It held two changes registered on Nov 9, before
this plan was written, and both had passed F's rule on his Mac: a 1,200-tree LightGBM member (`T1200`) and a
4,800-step TCN that also reads the touch columns (`LT`). F with both is called F3.

| What | Result | Where |
|---|---|---|
| F3 against F on the Windows PC, registered before any score there | Passes: +0.0161 [+0.0118, +0.0207], 21/28; not worse at `severe` or on SkillCorner | PR #25 |
| F3's reproduction checks (same machine twice, both paths, kernel swap, six OS and CPU combinations, LightGBM hash on three machines) | All pass | PR #25 |
| Possession-length weights in either member (Justin's S1 and S2) | Both fail, 17/28 | Decision log, 2026-11-11 |

**This plan fixed its own two candidates, so F3 is not adopted by it.** Whether F3 replaces F is David's and
Justin's decision. The cut-off above still holds: nothing is adopted after Nov 14, 18:00 CET. The change is
prepared on `david/f3-adopt`. Until it is merged, F is the submission model and `scripts/real_run.py` trains F.

## Status, 2026-11-12: F3 is the submission model

David gave the decision to Claude ("make the best decisions you see fit"), and F3 was adopted that night, two
days before the cut-off: PR #26 and the decision log of 2026-11-12. Justin put F3 forward in PR #23; he has not
confirmed the merge itself, and undoing it is one revert. From that merge on, `scripts/real_run.py` trains and
checks F3. F stays in the code (`--spec F`).

**Left for the humans:** the team name, the real run on David's PC, the `submission-v1` tag and the form.
