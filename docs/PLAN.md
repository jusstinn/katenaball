# katenaball: plan to Nov 17

Revised Nov 6, 2026 after the architecture bake-off. Nobody has seen the match data yet, so every score below comes from
external matches or a simulation. Evidence: [bakeoff.md](bakeoff.md), [external-data.md](external-data.md),
[decision-log.md](decision-log.md).

David: finish `docs/handoff/DAVID_BRAINSTORM.md` before reading this file.

## The short version

- **Phase 1 has no judges.** One script ranks us on `spellF1`. UI, dashboards, polish, and originality earn nothing.
- **The score decides whether we reach the jury.** The top 20 teams advance to an open-ended, jury-judged round, and the
  top 5 of those pitch in Barcelona. Build nothing for that yet, but keep the pipeline reusable.
- **We have a working baseline at about 0.55 expected, plausible range 0.45-0.62.** LightGBM on ball-aware and ball-free
  features, then Viterbi, trained on 30 external matches. It runs on external data today and has not yet run on the real match.
- **That baseline is the entry ticket.** Our survey of 64 rival teams puts the top-20 cut near 0.50 (0.45-0.56) and the top 5
  near 0.56-0.65, both low confidence. Most serious teams have this stack, and the scorer's noise band is about 0.04.
- **The main bet is short possessions.** Spells under 8 s are 75-85% of the true possessions we miss, and even clean input misses
  them. It is untested, so it could still fail.
- **Biggest risks:** what "owning" means in the corrected tracking, which alone moves scores by about 0.1 and cannot be measured
  from external data, and having no feedback before the deadline.

## Do these first (humans)

| By | Who | Action |
|---|---|---|
| **Nov 10** | Everyone | Each member registers individually under the **exact same team name**. Registration closes Nov 10. Teams need 3-5 members, at least 3 with technical profiles. A team under 3 registered members is invalid regardless of score. |
| Now | Justin | Finish the data download and unpack it into `data/`. Then run `uv run python scripts/probe_schema.py` and `uv run python scripts/check_target.py`. Both print only structure and pass/fail, so their output is safe to paste. |
| Now | David | Accept the repo invite and download the bundle separately into your own `data/`. Nothing passes through a chat or a hosted tool. |
| Now | Justin | Decide whether `katenaball` goes private until submission day. It is public, and rival teams are active: 236 forks. |
| Nov 7 | Justin | Find the Nov 3 organiser email and share its exact wording. Send the questions in "Open questions" below. Repo issues are disabled. |
| Nov 12 | Team | Upload a valid insurance submission once the form arrives by email. The form can be edited and the last response counts. |
| **Nov 16** | Team | Submit the final file. The deadline hour is unconfirmed: one competitor's report says 23:59 CET, which would be 14:59 Pacific on Nov 17. Treat Nov 17 as buffer only. |

## Data firewall

The terms say participants "shall not upload, disclose or otherwise make available to any third-party hosted or cloud-based
LLM any videos, datasets ... or any extracts thereof" without written authorisation. Claude Code is a hosted LLM. Data sitting on
the disk is fine. Data reaching Claude's context is not, and that includes file reads, stdout, tracebacks, and frames.

How we work:

- Claude writes and tests code against **synthetic fixtures and external matches**, which it can read.
- A **human runs** the job on the real match and views results with local tools.
- Real-match jobs log **aggregates only**: counts, rates, durations, shapes, and runtimes. Whether aggregates count as "extracts"
  is undefined, so we are asking in writing. Until we hear back, keep them minimal.
- `CLAUDE.md` and `.claude/settings.json` enforce this in every clone. The deny rules are a backstop and cannot catch everything,
  so the working habit still matters.
- **No match data on Supabase, Vercel, or any hosted service.** Neither helps the score, so we use neither. A rented GPU is a no until
  the organisers answer in writing.

Other rules that shape the work:

- The public 5 minutes are a pipeline check only. No training, tuning, or choosing between versions.
- No hand labels or hand corrections on this match, and no constants tied to its frames.
- Watching the video to understand it is allowed. Turning what you saw into labels or rules is not.
- **The target is inference-only for now.** A second-hand report says a Nov 3 organiser email requires training and tuning on external
  data, while the public rules allow algorithmic pseudo-labels. Until we read the wording, we do no self-training and do not fit
  the degrader to the target.
- Metrica gets a perpetual, irrevocable commercial licence to submitted ideas and methods (T&Cs section 16). Everyone should be
  comfortable with that.
- On leaving the competition, delete all data, including weights or embeddings that could reconstruct it.

Source: [Terms and Conditions v2](https://25499644.fs1.hubspotusercontent-eu1.net/hubfs/25499644/More%20Than%20A%20Hack%202027%20-%20Terms%20and%20Conditions%20v2.pdf).
Check that it matches the copy you signed.

## What limits the score

All measured on external matches with the official scorer, with the baseline model at target-like noise.

**If one thing were perfect (8 IDSSE and Metrica folds, base 0.568):**

| Perfect input | spellF1 | Gain |
|---|---|---|
| Team on every frame (this also fixes boundaries between teams) | 0.898 | +0.33 |
| Spell boundaries, with the team still predicted | 0.694 | +0.13 |
| Alive or dead | 0.639 | +0.07 |
| Ball position (IDSSE only, model not retrained) | 0.615 vs 0.544 | +0.07 |

**Which true possessions we find, by length (matched at IoU 0.5):**

| Length | IDSSE | Metrica | SkillCorner |
|---|---|---|---|
| Under 2 s | 2% | 23% | 22% |
| 2-4 s | 20% | 23% | 24% |
| 4-8 s | 47% | 55% | 50% |
| 8-16 s | 71% | 76% | 66% |
| 16-32 s | 84% | 86% | 81% |
| Over 32 s | 88% | 96% | 92% |

**Clean input does not fix it.** Trained and tested on clean data, IDSSE reaches only 0.657, against 0.544 at target-like noise, and
it still predicts 0.71 possessions per true one. Noise costs about 0.11. The rest survives a perfect ball. Our features are computed
at 5 Hz, which discards most of the signal from a touch. Some of the gap may be label noise, so the recoverable amount is unknown.

What we conclude:

1. **Find the short possessions.** "Wrong team" errors are mostly short spells the model never segmented.
2. **Do not lose the long ones.** They carry most of the time and are mostly right. Every change must pass the gate below, because false
   flips cost real score.
3. **Dead ball is second tier for rank, first tier for the tie-break.** Perfect alive/dead is worth +0.07.
4. **Never write `X` on a graded frame.** It equals "repeat my last label", never beats emitting that label, and costs `macroF1`.
5. **The second half has a trap:** graded frames start before kickoff and are all `D`. Detect kickoff from tagging and player layout.
6. **Do not trust near-clean benchmarks.** They flatter the official baseline by +0.16 and hide the value of ball-free features.

## Architecture

### Locked baseline

| Component | Decision | Evidence |
|---|---|---|
| Training data | 30 external matches: IDSSE 7, Metrica 3, SkillCorner 20. Metrica's labels are derived from its events, see `external-data.md`. | Real automatic tracking in SkillCorner, per-frame dead-ball labels in IDSSE |
| Labels | Merge brief opponent flips under 2 s (T = 2) | Smallest worst-case regret (0.017) across conventions T = 0-5 s. T cannot be identified from external data |
| Noise | Train across mild, target-like, severe from `katena.sim.degrade` | Ablation gaps grow with noise, and near-clean benchmarks mislead |
| Features | 160 per 5 Hz frame: ball-aware, ball-free, dead-ball, missing-ball, for each focal team, averaged | Ball-free +0.036, won 14/14 folds. Ball-free alone reaches 0.507 |
| Model | LightGBM, separate alive/dead and team heads | M2 0.586, against rules 0.432 and the official baseline 0.280 |
| Decoder | 3-state Viterbi, priors from external labels, temperature 2 | Temperature plateau from 1 to 4. HSMM added nothing |

Locked-bench reads, scored once and trained on nothing from them: IDSSE 0.567, Metrica 0.631.

### What we are testing next, in priority order

Each must pass the gate below. Estimates are guesses, not measurements.

| # | Bet | Why | Estimate | Owner | Cut rule |
|---|---|---|---|---|---|
| 1 | **Native-frame-rate touch and turnover features**: ball speed and direction changes near players, computed at 25 Hz before downsampling, plus shorter context windows | Short spells are the largest measured loss, and clean input shows the 5 Hz features miss them | +0.02-0.08 | David | Never cut before a real test |
| 2 | **Train across ball-loss coupling levels** | Metrica's ball is missing on 79% of dead frames and 1.5% of alive ones. The degrader currently has no such link, and it misses the published baseline's macroF1 (0.39 against 0.67) | +0-0.03, mostly insurance | David | Keep if no world gets worse |
| 3 | **TCN + LightGBM ensemble**, CPU only and deterministic | TCN alone gained about +0.01. Nobody has tested it in an ensemble yet | +0.01-0.03 | David | Drop if regeneration is not byte-identical |
| 4 | **Ball gap filling and flight hold** | Perfect ball is worth +0.07. Cheap | +0.01-0.03 | Justin | Keep only if it passes the gate |
| 5 | **Smart Tagging as learned features**, with the `DEFENDING` and `DEFENSIVE TRANSITION` teams flipped | Allowed and free, but a crude simulation gives at most +0.02 and can go negative | +0-0.02 | Justin | One day |
| 6 | **Broadcast cutaways as a dead-ball prior** | Replays and close-ups may signal a stoppage, which is unmeasured. Rivals' attempts at filling ball gaps from broadcast went badly, so we do not try that | +0-0.02 | Justin | Drop if offsets are inconsistent by Nov 13 |

### Do not build

- **Any UI, dashboard, or demo for submission.** The only permitted tool is an optional local debug viewer for humans, and it never leaves
  the machine.
- **HSMM, spell rescoring, or expected-F1 search.** Each was tested and did not beat Viterbi.
- **PathCRF.** It needs 22 identified players. Its weights were trained on our IDSSE matches, so we cannot score it fairly on them.
- **Self-training or fitting the degrader to the target,** unless the organisers' wording clearly allows it and it passes the gate.
- **Our own ball detector, or YOLO.** Rivals got 11% ball recovery from YOLO. Licence and reproducibility problems too.
- **Whistle audio.** A rival measured it at 1.4 times chance.
- **Supabase or Vercel.** Nothing to host in Phase 1.

## Validating blind

There is no leaderboard, so this protocol is what makes our decisions real.

1. **Leave-one-match-out on the 28 development matches**, scored with the official `possession_metrics.evaluate` at target-like noise.
2. **Adopt a change only if** it wins on at least 70% of matches and the paired bootstrap interval excludes zero. Use `katena.eval.bootstrap`.
3. **The locked benches are spent for selection.** One final sanity read before the freeze, with no choices made from it.
4. **Check the loader without looking at data:** `scripts/check_target.py` must reproduce the published `evaluation/results/baseline_*.csv` files
   exactly.
5. **Check the target without labels:** a human runs it and reads aggregates only, such as spells per minute and agreement between
   ball-based and ball-free models. These are alarms. They never select between versions.
6. **Public 5 minutes:** pass/fail alignment check only. Log every look in `docs/decision-log.md`.

Warning signs of overfitting: gains that vanish on another dataset, large output changes with no external gain, a drifting spell count, or
results that depend on a narrow parameter band.

## Work split

The bake-off already built most of the old plan: external loaders, degrader, features, LightGBM, decoder, harness, and target loaders. This is the
**remaining** work, 100 points. Justin plus Claude take 45, David plus his Claude take 55.

### Justin + Claude: 45

| ID | Work package | Points |
|---|---|---|
| J1 | Real-match bring-up: run the schema probe and loader check, fix format surprises, run the baseline end to end with aggregates only | 8 |
| J2 | Final pipeline: train on all 30 matches, save models, deterministic inference, kickoff handling, `make submit` | 10 |
| J3 | Reproducibility: pinned environment, two-machine hash test, technical document, form, compliance | 9 |
| J4 | Ball gap filling and flight hold, through the gate | 6 |
| J5 | Smart Tagging features, imitation tags from external events, through the gate | 8 |
| J6 | Broadcast cutaway probe and dead-ball prior, timeboxed | 4 |

### David + Claude: 55

| ID | Work package | Points |
|---|---|---|
| D0 | Blind brainstorm, written before reading this plan | 2 |
| D1 | **Native-frame-rate touch and turnover features**, the main bet | 18 |
| D2 | Degrader realism: mixed ball-loss coupling, identity switches, train across worlds | 10 |
| D3 | TCN + LightGBM ensemble, deterministic on CPU | 13 |
| D4 | Label convention: review harmonisation against the written definition of "controlled", test training across T | 7 |
| D5 | Independent audit of the bake-off code and final model training on your machine | 5 |

David's work is the score driver. Justin's work turns it into a correct, validated, reproducible file. Justin hands over J1's aggregate findings as soon as
they exist, because real-format surprises can change D1 and D2.

### Code map

Package `src/katena/`. Add features as a new group in `featurize/build.py`, so ablations keep working.

```python
katena.types.Match, load_match, save_match
katena.external.*                                   # IDSSE, Metrica, SkillCorner loaders
katena.target.atd.load_target(data_dir, source)     # real match; humans run it
katena.target.tagging.read_tags(path)
katena.labels.sandwich(y, T)                        # label convention
katena.sim.degrade.corrupt(m, cfg, seed)
katena.featurize.build.featurize(m, focal, dirs)    # X, idx, half, names, groups
katena.models.gbdt.fit_heads(...).predict(X0, X1)   # p_alive, p_home
katena.decode.hmm.decode(p_alive, p_home, half, priors, tau, "viterbi")
katena.eval.score.score(gt, pred)                   # wraps the official scorer
katena.eval.bootstrap.paired_bootstrap(a, b)        # the adoption gate
```

Rebuild from a fresh clone, about 5 GB of downloads and about 2 hours of compute:

```bash
uv sync
uv run python scripts/build_external.py
uv run python scripts/run_bakeoff.py
uv run pytest -q
```

- Git: short-lived branches and pull requests. Merge to `main` only with green tests.
- Ownership: each person owns their modules. Propose interface changes before making them.
- Tag each submission candidate.

## Schedule

| Date | Justin + Claude | David + Claude | Gate |
|---|---|---|---|
| Fri Nov 6 | Bake-off review, revised plan | D0 brainstorm | Handoff done |
| Sat Nov 7 | J1 probes and loader check | Environment rebuilt, D5 audit, D1 starts | **Baseline files reproduced** |
| Sun Nov 8 | J1 baseline on real match, J2 `make submit` | D1 touch features v1 | |
| Mon Nov 9 | J2 complete, J4 gap filling | D1 first gate result | **Valid baseline submission file exists locally** |
| Tue Nov 10 | J5 tags | D1 iterate, D2 coupling worlds | **Registration closes** |
| Wed Nov 11 | J5 continued, J6 probe | D2 complete, D3 ensemble | |
| Thu Nov 12 | J6, J3 first hash test | D3, D4 label convention | **Checkpoint, insurance submission** |
| Fri Nov 13 | J6 go/no-go, integrate winners | D4, integrate winners | **Broadcast go/no-go** |
| Sat Nov 14 | Regenerate | Final training | **Feature freeze** |
| Sun Nov 15 | Cold-clone regeneration on both machines | Bench sanity read, one pass | **Byte-identical hash** |
| Mon Nov 16 | Technical document, validate, submit | Review document, confirm hash | **Final submission** |
| Tue Nov 17 | Buffer only | Buffer only | Repo public, no data in history |

If behind on Nov 12, cut in this order: broadcast, tags, ensemble. The baseline, the loader check, and the hash test are never cut.

## Top risks

| Risk | Mitigation |
|---|---|
| The corrected tracking's "owning" means something different from our external labels. Worth about 0.1, not measurable externally | Train at T = 2, the lowest-regret choice. Ask the organisers. Do not tune on the public window |
| The degrader does not match real tracking. It misses the published baseline's scores | Train across worlds, including coupled ones. Prefer changes that help in every world |
| Short-spell features add false flips on noisy data | The gate, at target-like and severe noise |
| Tuning to nothing, since there is no feedback | The gate and the decision log |
| Real format differs from the documented one | `probe_schema.py` and `check_target.py`, first thing |
| Regeneration differs between machines | CPU inference, fixed threads, lockfile, hash test on both machines by Nov 15. LightGBM needs libomp on Mac, see `ensure_openmp` |
| Accidental data leak into Claude or git | Firewall rules, aggregate-only logging, history check before going public |
| Deadline or registration miss | Register now, insurance upload Nov 12, submit Nov 16 |

## Open questions for the organisers

Send to hackathon.talentarena@mobileworldcapital.com. Draft for Justin to review and send:

> Hello, we are a registered team with a few questions about permitted workflow:
>
> 1. We use a hosted AI coding assistant to write code. It never receives the videos, datasets, or sample rows, and the code runs locally. May it see
>    aggregate run statistics, such as counts, rates, or scores on the public window? Would you authorise this in writing, or should we avoid it?
> 2. May we process the data on a rented cloud GPU under our own control, with no LLM involved?
> 3. May published statistics of the public window, such as its possession count and mean duration, inform a decoder's duration prior? Could you also
>    clarify what "owning" means in the corrected tracking?
> 4. Do algorithmic pseudo-labels on the provided match remain allowed, as RULES.md says, or must all training and tuning use external data only?
> 5. If our pipeline uses a large CPU run, what regeneration time is acceptable? May cached features be passed to you privately, rather than in the
>    public repository?
> 6. May pretrained weights with a non-commercial research licence be used in the pipeline?
> 7. What is the exact deadline time and timezone on November 17?
>
> Thank you.

## Later rounds, only if we make the top 20

Confirmed from the T&Cs: the Innovation Challenge runs Nov 26 to Jan 14, online, with 10 more Barcelona matches. It is open-ended and may open with a
Nov 26 session that a team member must attend. The top 5 pitch on Mar 2, 2027 at Fira Montjuic, with travel covered. There is no cash prize. The jury
scores innovation and tactical value, technical robustness, answering football questions, and presentation.

Last year's winner [reportedly](https://www.linkedin.com/posts/guillemolivart_ipurdiar-we-have-created-a-chatbot-that-activity-7457345043708665856-PeSQ)
built a way to chat with a match using tracking, commentary, and LLMs. Chatbots may be common now, so validated tactical insight looks like the better way to
stand out. That is inference, not a stated criterion.

Prep now, under a day in total:

- Keep the pipeline match-agnostic, so it runs on 10 more matches.
- Hold Nov 26, Jan 14, Feb 22, and Mar 2. Check travel documents.
- Keep a running list of football questions that possession segments could answer. Build nothing yet.
- Consider adding a football analyst before Nov 10. The jury rewards tactical value, and we could not confirm that rosters can change later.
- Any LLM touching match data in Phase 2 must be local or authorised in writing.

## Open: the PetricaNancu repository

[PetricaNancu/ball-possession-barca-hackathon](https://github.com/PetricaNancu/ball-possession-barca-hackathon) is public, last pushed Oct 31, and has nearly
this architecture working. It reports about 0.615 on external matches, which is self-reported and not this match. **Justin needs to confirm whether this is our
Petrica.**

- **If it is ours:** compare it with our baseline through the gate, and make it private, because it publishes a full submission file. Petrica also needs work
  packages, so re-split the 45/55.
- **If it is not ours:** it is a competitor. Do not copy its code or use its submission file.

Also note the team needs at least 3 registered members, and Petrica has no work package yet.
