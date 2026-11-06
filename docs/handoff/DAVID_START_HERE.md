# David: start here (after the blind brainstorm)

You own 55% of the work, and it's the core score driver: external data, the noise simulator, features, and the model.
Justin and his Claude own the other 45%: loaders, validation harness, decoder, ball physics, dead ball, broadcast
fusion, and reproducibility. The full plan is in `docs/PLAN.md`.

## Before any code

1. **Register for the competition by Nov 10** under our exact team name, if you haven't. Each member registers
   individually.
2. **Download the data bundle yourself** into `data/` in your clone. It must not pass through any chat or hosted tool.
3. **Read the Data firewall section of `docs/PLAN.md`.** The terms forbid exposing match data to hosted LLMs, and your
   Claude Code counts. `CLAUDE.md` loads automatically. The deny rules in `.claude/settings.json` are a backstop, not a
   guarantee. Your Claude must never see rows, frames, or per-frame outputs from the match.
4. **Tell Justin what compute you have**: CPU, RAM, GPU, and free disk.

## Paste this into your Claude Code

> Read `CLAUDE.md`, `docs/PLAN.md`, and `docs/handoff/DAVID_START_HERE.md`. I own work packages D1-D5. Follow the data
> firewall strictly: develop only against external matches and synthetic fixtures, and when something must run on the
> real match, give me the command and have it log aggregates only. Do not ask me to paste real rows. Start with D1 on a
> branch: download IDSSE/DFL open data through `kloppy` into `external_data/` (gitignored), map it to the `Match` format in
> `docs/PLAN.md`, verify the per-frame `BallStatus` and `BallPossession` fields, and report spell-duration statistics
> across all 7 matches. Propose your full D1-D5 plan first, flagging anything in `docs/PLAN.md` you disagree with.

## Your work packages

| ID | Work | Points | Done when |
|---|---|---|---|
| D0 | Blind brainstorm | 2 | `docs/brainstorm/david.md` is committed |
| D1 | IDSSE + Metrica Game 3 loaders, label harmonisation, corpus build | 11 | All matches load to `Match`, and external spell durations can be compared with and without harmonisation |
| D2 | Noise simulator and domain-gap check | 10 | `corrupt()` and `fit_noise_cfg()` run, a classifier struggles to separate real from simulated windows, and the check runs locally with aggregate output |
| D3 | Ball-aware and ball-free features, side-symmetric | 12 | `featurize()` runs on the fixture and on 16 GB of RAM |
| D4 | LightGBM, calibration, TCN ablation, PathCRF spike | 15 | Best model passes the validation gate, and `predict_logp()` is deterministic on CPU |
| D5 | Transductive self-training on the target | 5 | Kept only if it passes the gate on external matches |

## What matters most

- **Team-per-spell accuracy is the largest source of loss.** The ball is missing on 41% of graded frames, so your ball-free
  features carry a lot of weight.
- **Harmonisation is the top risk.** DFL possession flips on single touches, with means of 4-6 s per spell, while our public
  window averages 14.2 s. Derive the rule from the written definition and external distributions, and do not fit it to the
  public 5 minutes.
- **Publish log-probabilities that are well calibrated.** Justin's decoder depends on them. Agree on the format early.

## Your schedule

| Date | Target |
|---|---|
| Nov 6 | Brainstorm |
| Nov 7 | IDSSE loader, harmonisation v1, interfaces frozen with Justin |
| Nov 8 | Corpus, features v1 |
| Nov 9 | LightGBM v1, **MVP locked** |
| Nov 10-11 | Simulator, domain check, ball-free features, PathCRF spike |
| Nov 12 | Model v2, checkpoint call |
| Nov 13 | Self-training, TCN ablation |
| Nov 14 | Final training, **feature freeze** |
| Nov 15 | Locked-match score (one pass), regenerate on your machine, match Justin's hash |
| Nov 16 | Review the technical document |

## Working with Justin

- Branch per work package, pull request to `main`, tests green before merging.
- Propose interface changes before making them. They're in `docs/PLAN.md`.
- Record every model or feature decision in `docs/decision-log.md`: what changed, external score, keep or drop.
- Never commit match data, frame-level exports, or submission CSVs.
