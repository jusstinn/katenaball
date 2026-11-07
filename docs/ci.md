# CI and CD

Hosted CI never sees the competition data. The firewall job fails if any of it is in git, and every
other job uses published files (the four baselines, the public 5-minute ground truth) and synthetic
matches only. The real submission is only ever built on a machine with the data bundle.

Workflows: [.github/workflows/ci.yml](../.github/workflows/ci.yml) and
[.github/workflows/release.yml](../.github/workflows/release.yml).

## What runs

| Job | What it checks | OS | Blocks merge |
|---|---|---|---|
| `plan` | Chooses the OS matrix (below) | Ubuntu | yes |
| `firewall` | `scripts/firewall_check.py` on tracked files, plus the history of every ref (main, pull requests, tags) or of the branch (other pushes) | Ubuntu | yes |
| `lint` | `ruff check --select F,E9` (real errors only), `uv lock --check`, and a report of text files opened without `encoding=` (ruff PLW1514, warnings only) | Ubuntu | yes, except the report |
| `test` | `uv sync --frozen`, LightGBM loads, `pytest -q`, in Python's default text encoding (no `PYTHONUTF8`, so Windows cp1252 bugs surface) | matrix | yes |
| `scorer` | `scripts/check_scorer.py`: the scorer files match the published ones, the official scorer reproduces EVALUATION.md's public-window spellF1 for the baselines, and `validate_submission.py` accepts and rejects synthetic files per SUBMISSION.md | matrix | yes |
| `determinism` | `scripts/synthetic_e2e.py` twice in separate processes (different `PYTHONHASHSEED`); all six sha256 must match; the synthetic CSV passes the validator | matrix | yes |
| `determinism across OSes` | Compares the hashes across OSes, writes a table to the run summary, dry-runs the release note | Ubuntu | no, it only warns |
| `ci-ok` | Fails unless every blocking job above succeeded | Ubuntu | this is the required check |

**The matrix** is Ubuntu, Windows and macOS (`ubuntu-24.04`, `windows-2025`, `macos-15`, pinned so a
`-latest` migration cannot change results silently) on: pushes to main, pull requests to main, tags,
pushes to a branch with an open pull request to main, pushes that touch `.github/`, `uv.lock`,
`pyproject.toml`, `.python-version`, `.gitattributes` or `scripts/synthetic_e2e.py`, and commits whose
message contains `[ci full]`. Every other branch push runs Ubuntu only.

**No duplicate runs.** Pull requests trigger CI when opened, reopened or marked ready, not on every
push: pushes to their branch already run, with the full matrix. One concurrency group per branch
cancels a run when a newer push arrives. Tag runs are never cancelled.

**Pinned tools.** Actions by commit SHA, uv 0.13.1, ruff 0.17.3, Python 3.11.17 (`UV_PYTHON`, the
version on the team machines; `.python-version` only says 3.11).

## Run it locally

```bash
python scripts/firewall_check.py                  # the index: what is tracked or about to be committed
python scripts/firewall_check.py --history HEAD   # plus every blob in this branch's history
uvx ruff@0.17.3 check --select F,E9 .
uv lock --check
uv run pytest -q
uv run python scripts/check_scorer.py
PYTHONHASHSEED=1 uv run python scripts/synthetic_e2e.py run --out /tmp/e2e-1
PYTHONHASHSEED=2 uv run python scripts/synthetic_e2e.py run --out /tmp/e2e-2
python scripts/synthetic_e2e.py compare /tmp/e2e-1/hashes.json /tmp/e2e-2/hashes.json
```

On macOS without Homebrew's libomp, LightGBM tests skip unless you point at the copy scikit-learn
bundles: `DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.11/site-packages/sklearn/.dylibs uv run pytest -q`
(scripts do this themselves through `gbdt.ensure_openmp`).

### Git hooks

`.githooks/pre-commit` runs the firewall on the index; `.githooks/pre-push` runs it on the pushed tips
and on every commit not yet on a remote. This repository is public, so a push is publication. Enable
them once per clone. It is a per-repository setting; do not set it globally:

```bash
git config core.hooksPath .githooks
```

Git worktrees share their repository's config, so this enables the hooks in every worktree of that
clone. To try them once without changing any config: `git -c core.hooksPath=.githooks commit ...`.

## Reading failures

**firewall.** Each line is `FAIL [rule] path (where): why`. It never prints file contents, only paths,
sizes and row counts, so the output is safe to paste. `tracked` means the file is in the commit:
`git rm --cached -- <path>` keeps your local copy, then make sure `.gitignore` covers it. `history
<commit>` means an earlier commit holds it. If that commit was never pushed, rewrite it before pushing.
If it was pushed, it is public now: stop and tell the team. Removing it takes a history rewrite and a
force push, which only the repository owner should do. The rules:

| Rule | Meaning |
|---|---|
| `match-data` | anything under `data/` except `data/README.md` |
| `submission` | a CSV in `evaluation/results/` other than the four published baselines |
| `published-modified` | a baseline or the public ground truth differs from the published file (pinned by blob id) |
| `private-gt` | `*_possession_gt.csv`, `*_possession_gt_meta.json`, `*_possession_gt_reference.csv`, `possession_by_frame.pkl` |
| `binary` | npz, npy, parquet, feather, h5, pkl, joblib, weights (pt, pth, ckpt, onnx, safetensors), archives, video, pl3container, pdf |
| `generated-dir` | a path inside `artifacts/`, `cache/`, `features/`, `runs/`, `external_data/` or `work/` |
| `root-csv` | a CSV in the repository root |
| `bundle-file` | a file named like the data bundle (`BAR-ATH_tactical*`, `BAR-ATH_broadcast*`) |
| `large` | a blob over 5 MiB |
| `submission-format` | a `frame,label` CSV with 1,000 rows or more |
| `frame-export` | a CSV with a frame column and one row per frame of this match (199,319, 164,488 or 158,857 rows, or the 5 Hz equivalents), or holding this match's tracking frame numbers |
| `notebook-output` | a notebook with saved outputs: clear them before committing |
| `settings`, `claude-md` | `.claude/settings.json` no longer denies `Read(./data/**)`, or CLAUDE.md lost its Data firewall section |

A false positive means the rule needs changing in `scripts/firewall_check.py` (and its tests), not a
bypass.

**lint.** Ruff errors appear as annotations on the offending lines. `uv lock --check` failing means
`pyproject.toml` changed without `uv lock`. The PLW1514 report never fails the job.

**test.** Read `gh run view <id> --log-failed`. A failure on one OS only is usually a path, line-ending
or encoding difference. The step before pytest prints whether LightGBM loads, so a macOS skip cannot
hide behind a green run.

**scorer.** `... unchanged` failing means someone edited the official scorer. Revert it, unless the
organisers published a new version: then update the pins in `scripts/check_scorer.py`. The
public-window spellF1 check failing with the scorer unchanged means a baseline or the public ground
truth changed, which the firewall also reports.

**determinism.** The two runs on one OS differ: something depends on hash order, threads, time or
uninitialised memory. The table names which hash moved first: `training_pool` or `target_features`
point at featurisation, `models` at LightGBM, `probabilities` at prediction, `decoded_rows` at the
decoder, `submission_csv` alone at the writer.

**determinism across OSes.** Never fails. A difference is an early warning for the two-machine hash
test (docs/PLAN.md, J3). If only the inner hashes differ, the CSV still matched, but a near-tie on the
real match could flip.

**ci-ok** lists each job's result. It fails when any blocking job failed, was cancelled or was skipped.

## CD

The submission cannot be built in hosted CI. CD here is two things:

1. **Release notes for submission candidates.** Tag the commit and push the tag:

   ```bash
   git tag -a submission-2026-11-12 -m "insurance submission"
   git push origin submission-2026-11-12
   ```

   `release.yml` re-runs the whole CI on that commit, then publishes a GitHub release with the
   commit, the sha256 of the committed `uv.lock`, the Python and uv versions, and the synthetic
   determinism hashes per OS. No data, no CSV, no assets. Record the real submission's sha256 in the
   technical document; the release note records what makes that file reproducible.
2. **The git hooks** above, which keep data out of every commit and push.

## Line endings and encodings

`.gitattributes` makes every checkout LF on every OS (`* text=auto eol=lf`), because Git for Windows
defaults to `core.autocrlf=true`. In an existing Windows clone, files already checked out stay CRLF
until rewritten; on a clean tree run `git rm -r -q --cached . && git reset --hard`.

Anything that writes a file we hash must write LF itself, since Python's text mode writes CRLF on
Windows and `pandas.to_csv` uses `os.linesep`: use `open(path, "w", newline="\n")`,
`to_csv(path, lineterminator="\n")`, or bytes, as `scripts/synthetic_e2e.py` does. Open text files
with an explicit `encoding=`: Windows defaults to cp1252.

## Branch protection for main (Justin, as admin)

Settings, then Branches, then a rule for `main`:

- Require a pull request before merging.
- Require status checks to pass, with **`ci-ok`** as the required check. It covers the whole matrix,
  so the setting survives matrix changes. Also tick "Require branches to be up to date before
  merging": push runs test the branch head, not the merge result.
- Block force pushes and deletions.
- Optionally, include administrators, so the firewall also gates admin merges.

Settings, then Actions, then General: keep workflow permissions read-only by default. `release.yml`
asks for `contents: write` in its own job. If publishing a release fails with a 403, the repository
or organisation limits workflow permissions and an admin has to allow it.
