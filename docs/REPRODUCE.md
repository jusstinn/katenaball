# Reproducing the submission

One command, from a clone of this repository and the data bundle:

```bash
make submit TEAM=<team_name>
```

It writes `submissions/<team_name>.csv`, runs the official `evaluation/validate_submission.py` on that exact
file, and prints its sha256.

## What it runs

| Step | Command | Reads | Writes |
|---|---|---|---|
| External corpus | `scripts/build_external.py` | Public IDSSE, Metrica and SkillCorner data, downloaded and checked by SHA256 (`scripts/external_manifest.json`) | `external_data/` |
| Corpus features | `make_submission.py corpus` | `external_data/canonical/` | `work/final/{gt,pool,feat}` |
| Training worlds | `make_submission.py worlds` | The same | The `d2j` world features, touch columns, and pools |
| Model | `make_submission.py train` | The same | `work/final/model/bundle.pkl`, `manifest.json` (hashes) |
| Labels | `make_submission.py predict` | `data/` (ATD metadata and raw tracking, tactical source) | `submissions/<team>.csv` |

The model is candidate F (`docs/decision-log.md`). It trains on all 30 external matches, and nothing from the
provided match. The provided match is read only by `predict`, which prints aggregates only (counts, shares,
shapes, hashes, run times).

## Requirements

- [uv](https://docs.astral.sh/uv/). `uv.lock` pins every dependency, and `.python-version` pins Python 3.11.
- About 25 GB of free disk: 5 GB of external data and an 18 GB cache.
- 16 GB of RAM is enough.
- On an Apple-silicon laptop with 10 cores and `WORKERS=3`, corpus to trained model took 18 minutes:
  corpus 4.5, worlds 9.5, LightGBM 0.25, TCN 3.5. Labelling the match takes well under a minute. The
  external download is extra.
- On macOS without Homebrew's libomp, the scripts load LightGBM through scikit-learn's bundled copy and
  re-execute themselves once (`gbdt.ensure_openmp`).

## Determinism

- Seeds are fixed: the degrader seeds come from the match id and world name, the model seed is 0, and
  `PYTHONHASHSEED` does not matter.
- LightGBM runs with `deterministic=True` and the TCN with deterministic torch algorithms and 4 threads.
- The CSV is built as bytes with LF line endings on every OS.
- `tests/test_pipeline.py` checks that two caches give byte-identical models and labels.
- A full rebuild from scratch on one Mac reproduced every corpus file and the model hashes byte for byte
  (`docs/decision-log.md`, J2).
- CI checks a synthetic run for byte identity on Linux, macOS and Windows.
- Expect identical labels across machines but not identical model bytes: torch's last bits depend on the
  CPU. The D3 probe found that this changes no label. To confirm, compare the submission sha256 from two
  machines (J3).

## Faster rebuild from a bake-off cache

If `work/bakeoff` exists (from `scripts/run_bakeoff.py`), the corpus step can copy its files instead of
rebuilding them:

```bash
uv run python scripts/make_submission.py corpus --seed-from work/bakeoff
```

The copied files are byte-identical to what the corpus step writes, which was checked on all 150 files.
