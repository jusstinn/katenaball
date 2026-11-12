# katenaball

Frame-level ball possession for **More than a Hack 2027, Phase 1**: label every frame of one football match
`H`, `A`, or `D`, ranked on `spellF1`.

```bash
make submit TEAM=<team_name>     # external data -> model -> submissions/<team_name>.csv, validated
```

- Reproduction: [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Approach and results: [`docs/technical-document.md`](docs/technical-document.md), [`docs/decision-log.md`](docs/decision-log.md)
- Rules, data, and metric: [`docs/upstream/`](docs/upstream/)
- Setup for contributors: [`CLAUDE.md`](CLAUDE.md) has the data rules everyone must follow

## Where things are

| Path | What |
|---|---|
| `src/katena/pipeline.py`, `scripts/make_submission.py` | The submission pipeline: corpus, training worlds, model, labels |
| `src/katena/` | Features (`featurize/`), noise simulation (`sim/`), models (`models/`), decoder (`decode/`), the gates (`eval/`), the provided match's loader (`target/`) |
| `scripts/` | One script per experiment or check. Each one's docstring says what it ran and where it is logged |
| `evaluation/` | The organisers' scorer and validator, unchanged |
| `docs/technical-document.md`, `scripts/build_document.py` | The method, the training data and the results, for the submission; and the script that prints it as `<team_name>.pdf` |
| `docs/decision-log.md` | Every decision with its pre-registration and result, newest first |
| `docs/REPRODUCE.md`, `docs/real-run.md` | How to regenerate the file, and the checklist for the run on the provided match |
| `docs/external-data.md`, `docs/label-convention.md` | The external corpus and how its labels map to H, A and D |
| `docs/ci.md`, `.github/workflows/` | Tests, the data firewall check, and the cross-OS reproduction |
| `docs/PLAN.md`, `docs/plan-final-week.md`, `docs/improvement-plan.md` | Plans, kept as written |
| `docs/bakeoff*.md`, `docs/reconcile.md`, `docs/audit/`, `docs/handoff/` | Earlier stages, kept as the record |

## Setup

```bash
uv sync --frozen                              # the locked environment (uv.lock, Python 3.11)
uv run pytest -q
uv run python evaluation/run_evaluation.py
```

Put the match data in `data/` (see [`data/README.md`](data/README.md)). It is gated and must never be committed.

## Attribution

`evaluation/`, `starter/`, `games.py`, and `docs/upstream/` come from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027), under the MIT Licence
in [`LICENSE-upstream`](LICENSE-upstream). The match data is not covered by that licence.
