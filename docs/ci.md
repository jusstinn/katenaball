# CI and CD

Hosted CI never sees the competition data. The firewall job fails if any of it is in git, and every
other job uses published files (the four baselines, the public 5-minute ground truth) and synthetic
matches only. The real submission is only ever built on a machine with the data bundle.

Workflows: [.github/workflows/ci.yml](../.github/workflows/ci.yml),
[.github/workflows/release.yml](../.github/workflows/release.yml), and the on-demand
[.github/workflows/repro.yml](../.github/workflows/repro.yml) ([below](#cross-os-reproduction-repro)).

## What runs

| Job | What it checks | OS | Blocks merge |
|---|---|---|---|
| `plan` | Chooses the OS matrix (below) | Ubuntu | yes |
| `firewall` | `scripts/firewall_check.py` on tracked files, plus the history of every ref (main, pull requests, tags) or of the branch (other pushes). Everywhere but main it runs twice: the branch's own copy, then main's copy (`git show origin/main:scripts/firewall_check.py \| python3 -`), so a branch cannot loosen the rules it is checked by | Ubuntu | yes |
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

**Least privilege.** Every job gets `contents: read`. Only `plan` also gets `pull-requests: read`, to
look for an open pull request from the branch.

## Run it locally

```bash
python scripts/firewall_check.py                  # the index: what is tracked or about to be committed
python scripts/firewall_check.py --history HEAD   # plus every blob in this branch's history
git show origin/main:scripts/firewall_check.py | python - --history HEAD   # main's rules, as CI runs them
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
force push, which only the repository owner should do.

Path rules come first; a file whose path fails one is never opened. Content rules read every other
blob up to 5 MiB, whatever its extension, and look for structure, never for values: code is full of
small integers, so a file is never flagged for how many numbers it holds. The rules:

| Rule | Meaning |
|---|---|
| `match-data` | anything under `data/` except `data/README.md` |
| `submission` | a CSV in `evaluation/results/` other than the four published baselines |
| `published-modified` | a baseline or the public ground truth differs from the published file (pinned by blob id) |
| `private-gt` | `*_possession_gt.csv`, `*_possession_gt_meta.json`, `*_possession_gt_reference.csv`, `possession_by_frame.pkl` |
| `binary` | by extension: array and table caches (npz, npy, parquet, feather, arrow, h5, hdf5, mat), pickles (pkl, pickle, joblib), weights (pt, pth, ckpt, onnx, safetensors), archives, video, pl3container, pdf, images (png, jpg, jpeg, gif, webp, bmp, tif, tiff), spreadsheets (xlsx, xlsm, xls, ods) and databases (sqlite, sqlite3, db, duckdb). Frame images and crops are match data. An image may be committed only by exact path in `ALLOWED_IMAGES`; none is today |
| `binary-content` | a blob holding NUL bytes under any other name: a renamed image, array or archive |
| `lfs` | a Git LFS pointer, or a `filter=lfs` line in any `.gitattributes`: LFS content is published but stored outside git, where this scan cannot read it |
| `generated-dir` | a path inside `artifacts/`, `cache/`, `features/`, `runs/`, `external_data/` or `work/` |
| `root-csv` | a CSV in the repository root |
| `bundle-file` | a file named like the data bundle (`BAR-ATH_tactical*`, `BAR-ATH_broadcast*`) |
| `large` | a blob over 5 MiB |
| `submission-format` | a `frame,label` CSV with 1,000 rows or more |
| `frame-export` | any text file, with or without a header, whatever its columns are called, with one line per frame of this match; or a line, a JSON array or object, or a run of label or digit characters with one value per frame. The lengths are 199,319 (tactical), 164,488 (broadcast), 158,857 (graded), 79,467 and 79,390 (the halves), and each of them on the 5 Hz grid, all within 2%. Also: a CSV frame column holding this match's tracking frame numbers (IdFrame 1,713,242 to 1,892,255), or any file listing 10 or more of them in increasing order |
| `frame-labels` | 25 or more consecutive frame,label-style rows: one `H`, `A`, `D` or `X` among numbers, as in a per-frame export or a spell list |
| `frame-run` | 100 or more consecutive lines whose leading integers step evenly by 1 to 25: a run of frame ids |
| `atd-raw` | three or more records in the ATD raw format (`frame_id:x,y,s;...;x,y,s:ball`), or one record with this match's track-slot count (46 tactical, 51 broadcast) |
| `atd-meta` | an ATD metadata XML with 22 or more `<Player ... teamId=...>` entries |
| `homography` | three or more homography rows (frame, `h00`..`h22` with `h22` = 1, `reprojection_error`, `valid`), comma- or space-separated, with or without the frame |
| `tagging` | ten or more Smart Tagging records: `<instance>` intervals with a start and an end, or delimited rows holding one of the match's tagging codes and a time |
| `notebook-output` | a notebook with saved outputs: clear them before committing |
| `settings`, `claude-md` | `.claude/settings.json` no longer denies `Read(./data/**)`, or CLAUDE.md lost its Data firewall section |

The thresholds leave room for what the repository legitimately holds: DATA.md shows one homography
row, `tests/test_target.py` holds one ATD record and three tagging intervals, and the longest numbered
run in any tracked file is six lines. Test fixtures in the bundle's formats are generated at test time
into a temporary directory, never committed as literal lines.

**A false positive is not fixed on the branch that hits it.** The branch is also checked by main's
copy of the scanner, so a rule loosened there changes nothing until it is on main. Report the
false positive (rule and path only), and change the rule in a pull request of its own, with a test
for the case it now allows, reviewed as described under [Rule changes need review](#rule-changes-need-review).
Never rename, split or encode a file to get it past a rule.

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

## Cross-OS reproduction (repro)

`repro.yml` answers J3 on GitHub's runners: does F train and label identically on another OS? It runs only
when dispatched (Actions, then Repro, then Run workflow, or `gh workflow run repro.yml -f os=all`), never on
pushes or pull requests, because it takes hours. It never sees the competition match.

Each runner (`ubuntu-24.04`; `windows-2025` and `macos-15` are optional, so their failure does not fail the
run) frees disk, then: `uv sync --frozen`; restores the raw external downloads from `actions/cache` (keyed on
`scripts/external_manifest.json`); `build_external.py`, which checks every raw file against its pinned SHA-256
(a new pin fails the step); `make_submission.py corpus`, `worlds` (workers = CPU count) and `train`;
`cache_digests.py dump`; `dressed_check.py export` of IDSSE `DFL-MAT-J03WMX`; and `predict` of that bundle
with the all-30 model, as the Windows reference was made. `scripts/repro_ci.py report` then writes one JSON:
model and part hashes, stage digests, the dressed bundle's and CSV's sha256, versions, CPU facts, stage
timings and label aggregates (counts, one score, one hash per 1,000 frames). The CSV is never uploaded.

The `compare` job (non-blocking) puts every runner beside `docs/repro/windows-reference.json` (David's PC,
from the J2/J3 entry of the decision log) in the run summary: per item same or differs, per stage how many
files agree, and the earliest item that differs. If two dressed CSVs differ, the block hashes say how many
1,000-frame blocks hold the difference. To compare two machines by hand, `cache_digests.py diff` works on the
uploaded `digests-<os>.json`.

## CD

The submission cannot be built in hosted CI. CD here is two things:

1. **Release notes for submission candidates.** Tag the commit and push the tag:

   ```bash
   git tag -a submission-2026-11-12 -m "insurance submission"
   git push origin submission-2026-11-12
   ```

   Tag a commit that is already on main: merge through a pull request first, then tag the merge
   commit. `release.yml` first checks that the tagged commit is an ancestor of `origin/main` and
   stops otherwise, then re-runs the whole CI on that commit, then publishes a GitHub release with the
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

## Rule changes need review

The firewall's rules are what keep the match data out of a public repository, so changing one is a
decision for a reviewer, not for the branch that runs into it. CI already makes a loosened rule inert
until it reaches main (main's copy of the scanner checks every other ref). What it cannot do alone is
make sure someone other than the author reads the change before it lands on main. That takes code
owners plus branch protection, and only the repository owner can switch those on.

Proposed `.github/CODEOWNERS`. It is a proposal only and is not committed: without the branch
protection setting below it would request reviews but enforce nothing.

```text
# The data firewall and everything that decides what reaches this public repository.
/scripts/firewall_check.py     @jusstinn
/tests/test_firewall_check.py  @jusstinn
/.githooks/                    @jusstinn
/.github/                      @jusstinn
/.gitignore                    @jusstinn
/.gitattributes                @jusstinn
/.claude/settings.json         @jusstinn
/CLAUDE.md                     @jusstinn
/evaluation/                   @jusstinn
/docs/ci.md                    @jusstinn
```

To enforce it (repository owner, once):

1. Add the file above as `.github/CODEOWNERS` through a pull request. Add a second owner on each
   line, so that the owner's own rule changes also get a review: GitHub never lets authors approve
   their own pull request.
2. In the branch protection rule for `main`, also tick "Require review from Code Owners",
   "Dismiss stale pull request approvals when new commits are pushed" and "Require approval of the
   most recent reviewable push".
3. Tick "Do not allow bypassing the above settings", or accept that an admin merge skips the review.
