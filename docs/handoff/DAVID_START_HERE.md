# David: start here (after the blind brainstorm)

You own 55% of the remaining work, including the main bet: finding short possessions. Justin and his Claude own the other
45%: running on the real match, the final pipeline, reproducibility, and the smaller experiments. The full plan is in `docs/PLAN.md`.

The first version of the plan gave you the external loaders, simulator, features, and first model. Those now exist, so
your work starts from a working baseline. Read `docs/bakeoff.md` for what it scores and where it fails.

## Before any code

1. **Register for the competition by Nov 10** under our exact team name, if you haven't. Each member registers individually.
2. **Accept the repo invite:** https://github.com/jusstinn/katenaball/invitations
3. **Download the data bundle yourself** into `data/` in your clone. It must not pass through any chat or hosted tool.
4. **Read the Data firewall section of `docs/PLAN.md`.** The terms forbid exposing match data to hosted LLMs, and your Claude Code counts.
   `CLAUDE.md` loads automatically. The deny rules in `.claude/settings.json` are a backstop, not a guarantee. Your Claude must never see rows,
   frames, or per-frame outputs from the match.
5. **Tell Justin what compute you have**: CPU, RAM, GPU, and free disk. The rebuild needs about 11 GB of disk.

## Paste this into your Claude Code

> Read `CLAUDE.md`, `docs/PLAN.md`, `docs/bakeoff.md`, and `docs/handoff/DAVID_START_HERE.md`. I own work packages D1-D5. Follow the data firewall strictly:
> develop only against external matches and synthetic fixtures, and when something must run on the real match, give me the command and have it log aggregates only.
> Do not ask me to paste real rows. Rebuild the corpus and bake-off cache with `scripts/build_external.py` and `scripts/run_bakeoff.py`, and check that M2 scores
> 0.586 on the development folds. Audit the bake-off for leakage and mistakes (D5), without assuming it is right. Then propose your plan for D1: touch and turnover
> features computed at the native frame rate. Flag anything in `docs/PLAN.md` you disagree with.

## Your work packages

| ID | Work | Points | Done when |
|---|---|---|---|
| D0 | Blind brainstorm | 2 | `docs/brainstorm/david.md` is committed |
| D1 | **Native-frame-rate touch and turnover features** | 18 | A new feature group is tested through the gate at target-like and severe noise, and the result is in the decision log, win or lose |
| D2 | Degrader realism: mixed ball-loss coupling, identity switches, train across worlds | 10 | The training mix is tested through the gate, and no world gets worse |
| D3 | TCN + LightGBM ensemble | 13 | The ensemble is tested through the gate, and it regenerates byte-identically on CPU, or it is dropped |
| D4 | Label convention | 7 | Harmonisation is reviewed against the "controlled" definition in `docs/upstream/CHALLENGE.md`, and training across T is tested |
| D5 | Audit, and final model training | 5 | Your independent audit is written down, and the final model trains on your machine |

## What matters most

- **Short possessions are where the points are.** Spells under 8 s are 75-85% of the true possessions we miss. Spells over 16 s are mostly right already.
- **Clean input doesn't fix it.** Trained and tested on clean data, IDSSE reaches 0.657 and still predicts only 0.71 possessions per true one. Features are computed
  after downsampling to 5 Hz, so touches are largely gone. That's our hypothesis. Part of the gap may be label noise, so test it before building on it.
- **Don't buy short spells with false flips.** Test at target-like and severe noise. The ball is missing on 41% of graded frames.
- **Don't trust clean scores.** They flatter everything and hide the value of ball-free features.
- **The target match is inference-only for now,** so no self-training and no fitting the degrader to it. See the plan for why.

## Your schedule

| Date | Target |
|---|---|
| Nov 6 | Brainstorm |
| Nov 7 | Environment rebuilt, audit, D1 starts |
| Nov 8-9 | D1 touch features, first gate result by Nov 9 |
| Nov 10-11 | D1 iterate, D2 coupling worlds, D3 starts |
| Nov 12 | D3, D4, **checkpoint call** |
| Nov 13 | D4, integrate winners |
| Nov 14 | Final training, **feature freeze** |
| Nov 15 | Bench sanity read (one pass), regenerate on your machine, match Justin's hash |
| Nov 16 | Review the technical document |

## Working with Justin

- Branch per work package, pull request to `main`, tests green before merging.
- Add features as a new group in `src/katena/featurize/build.py`, so ablations keep working. Propose other interface changes first.
- Record every model or feature decision in `docs/decision-log.md`: what changed, external score, keep or drop.
- Never commit match data, frame-level exports, or submission CSVs.
- The locked benches have been scored once. Don't use them to choose between versions.
