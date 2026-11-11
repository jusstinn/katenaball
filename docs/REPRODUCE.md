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
| Model | `make_submission.py train` | The same | `work/final/model_F3/bundle.pkl`, `manifest.json` (hashes) |
| Labels | `make_submission.py predict` | `data/` (ATD metadata and raw tracking, tactical source) | `submissions/<team>.csv` |

The model is F3 (`docs/decision-log.md`, 2026-11-11): candidate F with a 1,200-tree LightGBM member and a
4,800-step TCN that also reads the touch columns. It trains on all 30 external matches, and nothing from the
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

## Without `make` (Windows)

`make` is not on every machine (it is not on the team's Windows PC). For the real run on that PC,
`python scripts/real_run.py --team <team_name>` runs, in one command, the checks on the provided files, every step
below except `build_external.py` (it expects a built corpus), the public-window check, and a run report
(`docs/real-run.md`). The same steps, from the repository root:

```bash
python scripts/build_external.py                       # skips matches that are up to date (see below)
python scripts/make_submission.py corpus --workers 2
python scripts/make_submission.py worlds --workers 2
python scripts/make_submission.py train
python scripts/make_submission.py predict --out submissions/<team_name>.csv
python evaluation/validate_submission.py submissions/<team_name>.csv
sha256sum submissions/<team_name>.csv                  # shasum -a 256 on macOS
```

`build_external.py` decides whether a match is up to date from the loader-code digest in its sidecar. The
sidecars of the current corpus carry digests that no LF checkout reproduces (CRLF line endings, or code edited
after the build), so the first run rebuilds all 30 matches. The rebuild was checked on Windows: all 30 canonical
files come out byte-identical to the existing ones, so the training data does not change.

## Checking the submission path without the match

- `tests/test_pipeline_atd.py` writes a synthetic ATD bundle in the real file format on the competition's own
  timeline (199,319 frames, the graded halves of `games.py`) and runs `predict` on it through
  `katena.target.atd.load_target`; the CSV must pass `evaluation/validate_submission.py` unchanged.
- `scripts/dressed_check.py` exports an external match in the ATD format, trains the model without it
  (`make_submission.py train --hold-out <key>`, saved to `model_F3_without_<key>/`, never `model_F3/`), labels it with
  `predict --hold-out <key> --data-dir <bundle> --timeline <timeline.json>`, and compares the official score of
  that CSV with the evaluation path's score for the same model. Both options refuse the competition data.
- `scripts/cache_digests.py dump` hashes every training input and the model stage by stage; `diff` on two
  machines' dumps names the first stage where they part (J3). It compares array content, because the `.npz`
  bytes always differ between Windows and Unix (the zip entries record the OS). `.github/workflows/repro.yml`
  runs the whole chain on GitHub's Linux, Windows and macOS runners and compares them with this PC
  (`docs/ci.md`).
