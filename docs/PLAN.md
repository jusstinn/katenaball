# katenaball: plan to Nov 17

Written Nov 6, 2026 from five expert reviews of the public rules, scorer, and terms. Nobody has seen
the match data yet, so every score below is a prior or a synthetic measurement.

David: finish `docs/handoff/DAVID_BRAINSTORM.md` before reading this file.

## The short version

- **Phase 1 has no judges.** One script ranks us on `spellF1`. UI, dashboards, polish, and originality
  earn nothing.
- **The score decides whether we reach the jury.** The top 20 teams advance to an open-ended,
  jury-judged round, and the top 5 of those pitch in Barcelona. Build nothing for that yet, but keep
  the pipeline reusable.
- **Build:** a model trained on other matches, fed by geometric features that keep working when the
  ball is missing, and decoded by a semi-Markov model with duration priors.
- **Our edge:** the base stack is already public, so we also fuse the broadcast camera's ball data,
  add team-shape features, self-train on the target match, and fit the simulator to it.
- **Biggest risks:** a label-meaning gap between external data and our ground truth, and no feedback
  before the deadline.

## Do these first (humans)

| By | Who | Action |
|---|---|---|
| **Nov 10** | Everyone | Each member registers individually under the **exact same team name**. Registration closes Nov 10. Teams need 3-5 members, at least 3 with technical profiles. A team under 3 registered members is invalid regardless of score. |
| Now | Justin | Finish the data download. The bundle is not on this machine yet. Unpack it into `data/`. |
| Now | David | Download the bundle separately to his own `data/`, so nothing is transferred through a chat or a hosted tool. |
| Now | Justin | Make `katenaball` private until submission day. It is public and rival teams are active: 236 forks, about 80 pushed recently. |
| Nov 7 | Justin | Email the organisers the questions in "Open questions" below. Repo issues are disabled. |
| Nov 12 | Team | Upload a valid insurance submission once the form arrives by email. The form can be edited and the last response counts. |
| **Nov 16** | Team | Submit the final file. The deadline hour is unconfirmed: one competitor's report says 23:59 CET, which would be 14:59 Pacific on Nov 17. Treat Nov 17 as buffer only. |

## Data firewall

The terms say participants "shall not upload, disclose or otherwise make available to any third-party
hosted or cloud-based LLM any videos, datasets ... or any extracts thereof" without written
authorisation. Claude Code is a hosted LLM. Data sitting on the disk is fine. Data reaching Claude's
context is not, and that includes file reads, stdout, tracebacks, and frames.

How we work:

- Claude writes and tests code against **synthetic fixtures and external matches**, which it can read.
- A **human runs** the job on the real match and views results with local tools.
- Real-match jobs log **aggregates only**: counts, rates, durations, shapes, and runtimes. Whether
  aggregates count as "extracts" is undefined, so we are asking in writing. Until we hear back, keep
  them minimal.
- `CLAUDE.md` and `.claude/settings.json` enforce this in every clone. The deny rules are a backstop
  and cannot catch everything, so the working habit still matters.
- **No match data on Supabase, Vercel, or any hosted service.** Neither helps the score, so we use
  neither. A rented GPU is a no until the organisers answer in writing.

Other rules that shape the work:

- The public 5 minutes are a pipeline check only. No training, tuning, or choosing between versions.
- No hand labels or hand corrections on this match, and no constants tied to its frames.
- Watching the video to understand it is allowed. Turning what you saw into labels or rules is not.
- Metrica gets a perpetual, irrevocable commercial licence to submitted ideas and methods (T&Cs section
  16). Everyone should be comfortable with that.
- On leaving the competition, delete all data, including weights or embeddings that could reconstruct it.

Source: [Terms and Conditions v2](https://25499644.fs1.hubspotusercontent-eu1.net/hubfs/25499644/More%20Than%20A%20Hack%202027%20-%20Terms%20and%20Conditions%20v2.pdf).
Check that it matches the copy you signed.

## What the metric rewards

Possessions are matched one-to-one by temporal IoU, and F1 is averaged over IoU 0.3-0.7. These are
measured with the official scorer on a **synthetic match** (about 380 possessions, loosely shaped like
the public window), so trust the order more than the exact values.

| Error | spellF1 |
|---|---|
| Wrong team on 20% / 10% of spells | 0.74 / 0.87 |
| Boundary jitter of 2 s / 1 s / 0.5 s | 0.69 / 0.83 / 0.92 |
| Systematic lag of 1 s / 0.5 s / 0.2 s | 0.76 / 0.90 / 0.986 |
| 100 / 30 spurious flips | 0.59 / about 0.83 |
| Dropping every spell under 5 s / 3 s / 2 s / 1 s | 0.69 / 0.80 / 0.87 / 0.97 |
| 30 false dead-ball insertions | 0.86-0.90 |
| Missing 100% / 50% / 25% of dead-ball stretches | 0.81 / 0.90 / 0.95 |

What we conclude:

1. **Get each spell's team right.** It is the largest source of loss.
2. **Keep boundaries within 0.5 s, with no systematic lag.** Accuracy finer than about 0.25 s earns
   almost nothing.
3. **Avoid false events.** From the table above, each false event costs roughly 2-3 times what each missed
   dead-ball stretch costs. Use a minimum dwell of 1-2 s, and emit a sub-1 s flip only at about 0.7-0.8
   confidence.
4. **Treat dead ball as second tier for rank, first tier for the tie-break.** Missing all of it costs
   0.19 of `spellF1` but drops `macroF1` to 0.60.
5. **Never write `X` on a graded frame.** It equals "repeat my last label", never beats emitting that
   label, and costs `macroF1`. Use it only on ungraded frames.
6. **The second half has a trap:** graded frames start before kickoff and are all `D`. Detect kickoff
   from tagging and player layout.

Rough tiers, assuming the synthetic match holds:

| Tier | Boundaries | Wrong team | Short flips kept | spellF1 |
|---|---|---|---|---|
| A | 0.25 s | 3% | >= 1 s | 0.88 |
| B | 0.5 s | 8% | >= 2 s | 0.66 |
| C | 1 s | 15% | >= 3 s | 0.46 |
| D | 2 s | 25% | >= 5 s | 0.30 |

**Target: tier B.** The published floor is 0.2433. A low-confidence guess puts the top-20 cut at
0.40-0.50 and the best teams at 0.55-0.65. The noise band is about 0.04, so the cut will be crowded.

## Method decisions

### Build

| Component | Decision | Why |
|---|---|---|
| Training data | **IDSSE/DFL open data** (7 matches, 25 fps, per-frame `BallStatus` + `BallPossession`, CC BY 4.0) as primary. **Metrica sample Game 3** as a cross-provider holdout. PFF World Cup only if we need volume. | Closest structure to our ground truth, no sign-up, about 2.6 GB. |
| Label harmonisation | Relabel brief opponent flips in DFL to approximate "controlled" possession. Derive the rule from the challenge's written definition and external distributions. | **Top risk.** DFL means 4-6 s per spell against 14.2 s in our public window, so raw labels would teach the model to flicker. |
| Noise simulator | Corrupt clean tracking to look like ATD: strip identities, fragment tracks, bursty ball loss correlated with congestion, ghost tracks. Fit it to label-free statistics of the target, and train across a range of severities. | Lets us train and validate on realistic inputs. |
| Features | Ball-aware (gap-aware kinematics, nearest player per team, relative speed, time since seen) and **ball-free** (team shape, line height, velocity along the attack axis, pressing counts), both side-symmetric. | The ball is missing on 41% of graded frames, disproportionately at key moments. |
| Ball physics | Interpolate short gaps, flag flight, and hold the previous team during flight. Build first. | Cheap and low risk. Estimated +0.03-0.10. |
| Model | **LightGBM first**, a small TCN only if it wins on held-out matches. Separate alive/dead and team heads. | Fast, CPU-deterministic, works on 16 GB. |
| Decoder | **Hidden semi-Markov model with duration priors.** Calibrate with a label-free plateau: pick the middle of the range where the spell count is stable. | Beat tuned smoothing by +0.07 to +0.11 in simulation, the largest measured gain. Its prior came from the generator, so that gain is somewhat flattered. |
| Dead ball | Stillness, player-layout cues, ball out of bounds, and Smart Tagging restarts as evidence, never copied as labels. | Important for the tie-break, and also limits false events. |

### Our edge beyond the public stack

| Bet | Plan | Cut rule |
|---|---|---|
| **Broadcast fusion** | Compute one offset per half by matching projected player positions, reject replays, and fill tactical ball gaps. Broadcast stretches with no homography (replays, close-ups) also become a dead-ball prior. Estimated +0.02-0.05. | Drop if offsets are inconsistent by end of Nov 13. |
| **Self-training on the target** | Confident ball-based frames pseudo-label a ball-free model, which handles the ball-missing frames. Explicitly allowed if no person decides labels. Estimated +0-0.04. Confident frames are not representative, so weight for that. | Keep only if it wins on the external protocol. |
| **PathCRF spike** | [Published](https://arxiv.org/abs/2602.12080) 92.4% team accuracy on DFL with no ball input. [MPL-2.0 code and weights](https://github.com/hyunsungkim-ds/pathcrf). Fragmented anonymous tracks may break it, and its metric is frame accuracy over in-play segments. Half a day, local only. | Keep as a feature only if it beats our ball-free model. |
| **Touch timing** | Change-point detection near players, used to place boundaries only. Estimated +0.01-0.03. | First to go if behind. |
| **Whistle audio** | Band-energy detection, if the videos have audio. Check first, 0.5-1 day. | Skip if there is no audio. |

### Do not build

- **Any UI, dashboard, or demo for submission.** The only permitted tool is an optional local debug
  viewer for humans, and it never leaves the machine.
- **Expected-F1 search decoder.** In simulation it did not beat Viterbi and it degrades when posteriors
  are over-confident, which is likely under dataset shift.
- **Our own ball detector before Nov 12.** About 3-4 days of work for +0.01-0.03, with licence and
  reproducibility problems. Allowed as a one-day spike only if we are ahead.
- **Pose, video classifiers, or action spotting.** Near-zero value, no legal labels on this game, and
  about 8 points of accuracy in the literature only with far more data than we have.
- **Set-transformer as the main model.** Too much variance for 11 days.
- **Supabase or Vercel.** Nothing to host in Phase 1.

## Validating blind

There is no leaderboard, so this protocol is what makes our decisions real.

1. **Leave-one-match-out on external matches**, scored with the official `possession_metrics.evaluate`
   on simulator-corrupted inputs.
2. **Adopt a change only if** it wins on at least 70% of matches and the paired bootstrap interval
   excludes zero.
3. **Lock 2 matches** and score them once, at the end.
4. **Check domain gap without labels:** train a classifier to separate real from simulated windows
   (target AUC near 0.5), and sweep corruption severity.
5. **Check the target without labels:** spells per minute inside the external range, agreement between
   ball-based and ball-free models, consistency between halves, and stability of output across decoder
   settings.
6. **Check the loader without looking at data:** our loader plus the baseline must reproduce the
   published `evaluation/results/baseline_*.csv` files exactly.
7. **Public 5 minutes:** pass/fail alignment check only. Log every look in `docs/decision-log.md`.

Warning signs of overfitting: gains that vanish on locked matches, large output changes with no external
gain, a drifting spell count, or results that depend on a narrow parameter band.

## Work split

Effort is 100 points. Justin plus Claude take 45, David plus his Claude take 55.

### Justin + Claude: 45

| ID | Work package | Points |
|---|---|---|
| J1 | Package skeleton, `Match` format, ATD/tagging loaders, baseline reproduction test, synthetic fixture | 5 |
| J2 | Scorer wrapper, leave-one-match-out harness, bootstrap gate, decision log | 7 |
| J3 | Semi-Markov decoder with duration priors and label-free calibration | 10 |
| J4 | Ball physics: gap filling, flight flags, hold logic | 6 |
| J5 | Dead-ball signals, Smart Tagging evidence, kickoff detection | 7 |
| J6 | Broadcast sync and fusion, with data probes on day 1 (timeboxed) | 5 |
| J7 | Reproducibility, `make submit`, two-machine hash test, technical document, form, compliance | 5 |

### David + Claude: 55

| ID | Work package | Points |
|---|---|---|
| D0 | Blind brainstorm, written before reading this plan | 2 |
| D1 | External loaders (IDSSE, Metrica Game 3), label harmonisation, corpus build | 11 |
| D2 | Noise simulator, fit to target statistics, domain-gap check | 10 |
| D3 | Feature store: ball-aware and ball-free team-shape features | 12 |
| D4 | Model training, calibration, TCN ablation, PathCRF spike | 15 |
| D5 | Transductive self-training | 5 |

David's work is the core score driver. Justin's work is the frame around it: everything that turns model
outputs into a correct, validated, reproducible file.

### Shared interfaces (freeze on Nov 7)

Package `src/katena/`. Everyone tests against a shared synthetic fixture.

```python
@dataclass
class Match:
    P: "float32[n, S, 2]"       # metres, NaN = missing
    team: "int8[S]"             # 0 home, 1 away
    speed: "float32[n, S]"      # m/s, NaN if unavailable
    ball: "float32[n, 2]"       # metres, NaN = missing
    half_blocks: list           # [(half, first_frame, last_frame)]
    attack_dir: dict            # {half: +1 or -1} for home
    y: "int8[n] | None"         # 0 H, 1 A, 2 D, 3 X; None on the target match

load_match(path) -> Match
corrupt(m_clean, noise_cfg, seed) -> Match                     # D2
fit_noise_cfg(m_target) -> dict                                # D2, label-free stats only
fill_ball(m) -> BallTrack                                      # J4: position, source, in_flight, age
extra_signals(m) -> dict[str, "float32[n]"]                    # J5, J6: named per-frame columns
featurize(m, cfg, extras) -> Feats                             # D3: X[T, F], t_frame[T], mask
Model.predict_logp(feats) -> "float32[T, 3]"                   # D4: log P(H, A, D)
decode(logp, t_frame, half_blocks, cfg, seed) -> "int8[n]"     # J3
score(pred, gt) -> dict                                        # J2, wraps the official scorer
```

- Git: short-lived branches and pull requests. Merge to `main` only with green tests.
- Ownership: each person owns their modules. Propose interface changes before making them.
- Tag each submission candidate.

## Schedule

| Date | Justin + Claude | David + Claude | Gate |
|---|---|---|---|
| Fri Nov 6 | Admin, J1 skeleton, interfaces | D0 brainstorm | Handoff done |
| Sat Nov 7 | Loaders, J6 probes (audio, fps, gap histogram), J2 scorer | D1 IDSSE loader, harmonisation v1 | Interfaces frozen, baseline reproduced |
| Sun Nov 8 | J2 harness, J3 decoder v1, J4 physics | D1 corpus, D3 features v1 | |
| Mon Nov 9 | `make submit` end to end | D4 LightGBM v1 | **MVP locked, beats floor on external protocol** |
| Tue Nov 10 | J5 dead ball and tags | D2 simulator | **Registration closes** |
| Wed Nov 11 | J5 continued, J3 calibration | D2 domain check, D3 ball-free features, PathCRF spike | |
| Thu Nov 12 | J6 sync starts | D4 model v2 | **Checkpoint, insurance submission** |
| Fri Nov 13 | J6 fusion | D5 self-training, TCN ablation | **Sync go/no-go** |
| Sat Nov 14 | Integrate, regenerate | Integrate, final training | **Feature freeze** |
| Sun Nov 15 | Cold-clone regeneration on both machines | Locked-match score, one pass | **Byte-identical hash** |
| Mon Nov 16 | Technical document, validate, submit | Review document, confirm hash | **Final submission** |
| Tue Nov 17 | Buffer only | Buffer only | Repo public, no data in history |

If behind on Nov 12: cut whistle, touch timing, and PathCRF first. Then self-training. Then broadcast
fusion. Physics, external training, harmonisation, the decoder, and the reproducibility test are never cut.

## Top risks

| Risk | Mitigation |
|---|---|
| External labels mean something different from ours | Harmonisation, Metrica holdout, organiser question, decoder priors refined from target posteriors with strong shrinkage |
| Simulator does not match the real noise | Fit to the target, domain-gap check, train across severities |
| Tuning to nothing, since there is no feedback | The validation gate above, plus the decision log |
| Regeneration differs between machines | CPU inference, fixed threads, rounded logits before decoding, lockfile, hash test both machines by Nov 15 |
| Accidental data leak into Claude or git | Firewall rules, aggregate-only logging, history check before going public |
| Deadline or registration miss | Register now, insurance upload Nov 12, submit Nov 16 |

## Open questions for the organisers

Send to hackathon.talentarena@mobileworldcapital.com. Draft for Justin to review and send:

> Hello, we are a registered team with a few questions about permitted workflow:
>
> 1. We use a hosted AI coding assistant to write code. It never receives the videos, datasets, or
>    sample rows, and the code runs locally. May it see aggregate run statistics, such as counts, rates,
>    or scores on the public window? Would you authorise this in writing, or should we avoid it?
> 2. May we process the data on a rented cloud GPU under our own control, with no LLM involved?
> 3. May published statistics of the public window, such as its possession count and mean duration, inform
>    a decoder's duration prior? Could you also clarify what "owning" means in the corrected tracking?
> 4. If our pipeline uses a large CPU run, what regeneration time is acceptable? May cached features be
>    passed to you privately, rather than in the public repository?
> 5. May pretrained weights with a non-commercial research licence be used in the pipeline?
> 6. What is the exact deadline time and timezone on November 17?
>
> Thank you.

## Later rounds, only if we make the top 20

Confirmed from the T&Cs: the Innovation Challenge runs Nov 26 to Jan 14, online, with 10 more Barcelona
matches. It is open-ended and may open with a Nov 26 session that a team member must attend. The top 5 pitch
on Mar 2, 2027 at Fira Montjuic, with travel covered. There is no cash prize. The jury scores innovation and
tactical value, technical robustness, answering football questions, and presentation.

Last year's winner [reportedly](https://www.linkedin.com/posts/guillemolivart_ipurdiar-we-have-created-a-chatbot-that-activity-7457345043708665856-PeSQ)
built a way to chat with a match using tracking, commentary, and LLMs. Chatbots may be common now, so validated
tactical insight looks like the better way to stand out. That is inference, not a stated criterion.

Prep now, under a day in total:

- Keep the pipeline match-agnostic, so it runs on 10 more matches.
- Hold Nov 26, Jan 14, Feb 22, and Mar 2. Check travel documents.
- Keep a running list of football questions that possession segments could answer. Build nothing yet.
- Consider adding a football analyst before Nov 10. The jury rewards tactical value, and we could not confirm
  that rosters can change later.
- Any LLM touching match data in Phase 2 must be local or authorised in writing.

## Open: the PetricaNancu repository

[PetricaNancu/ball-possession-barca-hackathon](https://github.com/PetricaNancu/ball-possession-barca-hackathon)
is public, last pushed Oct 31, and has nearly this architecture working: external corpus, DFL harmonisation,
noise simulator, LightGBM, Viterbi. It reports about 0.615 on external matches, which is self-reported and not
this match. **Justin needs to confirm whether this is our Petrica.**

- **If it is ours:** fork it as our starting point. That gives us a head start of several days, so J1-J4 and
  D1-D4 become review, hardening, and upgrades, and the edge bets start Nov 7. Also make it private now, because
  it publishes our approach, trained model, and a full submission file. Re-split the 45/55 against what remains.
- **If it is not ours:** it is a competitor and a public benchmark. Do not copy its code or use its submission
  file. Treat this stack as the entry ticket and prioritise the edge bets.
