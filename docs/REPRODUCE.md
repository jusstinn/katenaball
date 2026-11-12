# Reproducing the submission

One command, from a clone of this repository and the data bundle:

```bash
make submit TEAM=<team_name>
```

It writes `submissions/<team_name>.csv`, runs the official `evaluation/validate_submission.py` on that exact
file, and prints its sha256. A team name with a space needs quotes: `make submit TEAM="My Team"`.

## Where the data bundle goes

Unpack the organisers' bundle into `data/`, or leave it anywhere and point `MTAH_DATA_DIR` at that folder
(`games.py`). The run reads three files of it, all from the tactical source:

```
data/                                       or $MTAH_DATA_DIR
`-- tactical/
    |-- BAR-ATH_tactical_atd.xml            ATD metadata
    |-- BAR-ATH_tactical_atd.txt            ATD raw tracking
    `-- BAR-ATH_tactical_smart_tagging.xml  Smart Tagging: one kickoff tag per half
```

`make submit` checks first that the three files exist, and stops with the missing paths if one does not. That
check opens nothing.

## What it runs

| Step | Command | Reads | Writes |
|---|---|---|---|
| Bundle check | `make_submission.py bundle` | The names of the three files above, not their contents | Nothing |
| External corpus | `scripts/build_external.py` | Public IDSSE, Metrica and SkillCorner data, downloaded and checked by SHA256 (`scripts/external_manifest.json`) | `external_data/` |
| Corpus features | `make_submission.py corpus` | `external_data/canonical/` | `work/final/{gt,pool,feat}` |
| Training worlds | `make_submission.py worlds` | The same | The `d2j` world features, touch columns, and pools |
| Model | `make_submission.py train` | The same | `work/final/model/bundle.pkl`, `manifest.json` (hashes) |
| Labels | `make_submission.py predict` | The bundle: ATD metadata and raw tracking, and the Smart Tagging file for the kickoff rule | `submissions/<team>.csv` |

The model is candidate F (`docs/decision-log.md`). It trains on all 30 external matches, and nothing from the
provided match. The provided match is read only by `predict`, which prints aggregates only (counts, shares,
shapes, hashes, run times).

## Requirements

- [uv](https://docs.astral.sh/uv/), checked with 0.13.1, the version CI pins. `uv.lock` pins every dependency
  and `.python-version` pins Python 3.11; `uv sync --frozen` installs both.
- `git` and `make`. Without `make`, run the steps by hand (below).
- Network access on the first run: the external data comes from figshare and GitHub, about 5 GB.
- **On Linux, the system's OpenMP runtime, `libgomp.so.1`.** LightGBM's Linux wheel does not bring its own.
  Desktop distributions and GitHub's runners have it; slim container images and a bare WSL distribution do not.
  Install it with `apt-get install libgomp1` (Debian, Ubuntu) or `dnf install libgomp` (Fedora, RHEL). Without
  it, `make submit` stops at its first step and says so.
- About 25 GB of free disk: 5 GB of external data and a 19 GB cache. On Linux the Python environment adds
  about 6 GB, because torch's Linux wheels bring the CUDA libraries. Nothing runs on a GPU: both models train
  and predict on the CPU, on every OS.
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
- Expect identical labels across machines but not identical model bytes. The LightGBM model, the scaler and
  the decoder priors are byte-identical everywhere. The TCN's weights differ in their last bits when the OS or
  the CPU vendor differs, because torch picks other kernels.
- `.github/workflows/repro.yml` trained the model from an empty cache on GitHub's Linux, Windows and macOS
  runners and on the team's own machines. An external match exported in the provided format got a
  byte-identical CSV every time (`docs/technical-document.md`, section 7; `docs/decision-log.md`, J3). That
  was shown on an external match, not on the provided one.

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
uv run python scripts/make_submission.py bundle               # the bundle's three files exist
uv run python scripts/build_external.py                       # skips matches that are up to date
uv run python scripts/make_submission.py corpus --workers 2
uv run python scripts/make_submission.py worlds --workers 2
uv run python scripts/make_submission.py train
uv run python scripts/make_submission.py predict --out submissions/<team_name>.csv
uv run python evaluation/validate_submission.py submissions/<team_name>.csv
sha256sum submissions/<team_name>.csv                         # shasum -a 256 on macOS, Get-FileHash on Windows
```

Every step skips what is already built, so a run that stopped can be started again with the same command.

## Checking the submission path without the match

- `tests/test_pipeline_atd.py` writes a synthetic ATD bundle in the real file format on the competition's own
  timeline (199,319 frames, the graded halves of `games.py`) and runs `predict` on it through
  `katena.target.atd.load_target`; the CSV must pass `evaluation/validate_submission.py` unchanged.
- `scripts/dressed_check.py` exports an external match in the ATD format, trains F without it
  (`make_submission.py train --hold-out <key>`, saved to `model_without_<key>/`, never `model/`), labels it with
  `predict --hold-out <key> --data-dir <bundle> --timeline <timeline.json>`, and compares the official score of
  that CSV with the evaluation path's score for the same model. Both options refuse the competition data.
- `scripts/cache_digests.py dump` hashes every training input and the model stage by stage; `diff` on two
  machines' dumps names the first stage where they part (J3). It compares array content, because the `.npz`
  bytes always differ between Windows and Unix (the zip entries record the OS). `.github/workflows/repro.yml`
  runs the whole chain on GitHub's Linux, Windows and macOS runners and compares them with this PC
  (`docs/ci.md`).
