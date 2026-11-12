# The real run: checklist for the person with the data

Only a team member runs anything on the provided match, never an AI assistant (`CLAUDE.md`). Every script
below prints aggregates or pass/fail only. **Share only the PASS/FAIL lines, "Valid", and the hashes**
outside the team. Keep label shares, D shares, touch rates and jitter numbers to yourself. If something
looks wrong, describe the symptom in words.

The commands are for the team's Windows PC (PowerShell, from the repository root, venv in `.venv`). On
macOS or Linux, use `uv run python` and `make` instead (`docs/REPRODUCE.md`).

## 0. Set up

```powershell
git checkout main
git pull
$env:MTAH_DATA_DIR   = "D:\mtah2027"               # the data bundle (already set at user level on this PC)
$env:KATENA_EXTERNAL = "D:\katena\external_data"   # the external corpus
$env:KATENA_WORK     = "D:\katena\work-submit"     # the cache: corpus and worlds are already built there
```

## Steps 1 to 3 in one command

Replace `<team_name>` with the team name registered on the form (the file must be named after it).

```powershell
.venv\Scripts\python.exe scripts\real_run.py --team <team_name>
```

It runs the six steps listed here, which cover steps 1 to 3 below with the same commands and the same checks,
and it stops at the first failure:

1. **Environment.** `MTAH_DATA_DIR` is set and the bundle's files exist (it checks existence only, through
   `games.asset_paths`, and opens nothing). `KATENA_WORK` and `KATENA_EXTERNAL` exist, with 30 external
   matches. It prints the git commit, whether the tree is clean, and Python and the library versions against
   `pyproject.toml`.
2. **The checks on the provided files** (step 1 below). It prints only their PASS/FAIL lines, and for the touch
   check the feature shape. A FAIL or a `raised <SomeError>` line stops the run.
3. **`make_submission.py all --workers 2`** (step 2 below), with its output streamed. Then it compares the model
   hashes with [`docs/repro/expected-windows.json`](repro/expected-windows.json). A different gbdt hash stops
   the run, before `predict` when it can. A different tcn or bundle hash alone is printed as a `WARN`.
4. **`validate_submission.py`** on the exact file, **`check_public_window.py`** (PASS/FAIL lines only), and the
   file's SHA-256 (step 3 below).
5. **`submissions\<team_name>.run.json`**: the commit, the times, the PASS/FAIL results, the model hashes and
   the CSV's SHA-256. It holds no label shares and nothing per frame. Like the CSV, it is git-ignored.
6. **A decision-log snippet**, printed at the end between `== 6. Decision-log snippet` and `== end of snippet`:
   the date, the commit, the PASS/FAIL lines and the hashes. Paste it into `docs/decision-log.md`. That logs
   this look at the public window, as the plan asks.

Options:
- `--skip-checks`: skips the checks on the provided files, for a re-run after they have passed on the same
  loader code.
- `--work DIR` and `--external DIR`: used in place of `KATENA_WORK` and `KATENA_EXTERNAL`.

Exit status 0 means every step passed. Warnings are allowed: read each `WARN` line. 1 means a step failed:
the `FAIL  stopped at ...` line names the step.

**What to share.** Outside the team, share only the snippet: PASS/FAIL lines, "Valid" and hashes. The streamed
`make_submission.py` output in step 3 holds label shares, D shares and touch counts. Those stay in the team, like
the numbers the checks print when you run them by hand. If the run stops, the `FAIL` line is safe to paste.
Describe anything else in words.

Then go to step 4, Freeze and submit. Steps 1 to 3 below are the same run done by hand. Use them if the command
fails in a way you need to look at step by step, or to re-run one step.

## 1. Checks on the provided files (once; by hand)

```powershell
.venv\Scripts\python.exe scripts\check_target.py          # 16/16 PASS on 2026-11-07; re-run after any loader change
.venv\Scripts\python.exe scripts\check_target_layout.py   # never run yet: expect "PASS katena.target.atd reads this layout as intended"
.venv\Scripts\python.exe scripts\check_touch_target.py    # an alarm only: it must finish and print a feature shape
.venv\Scripts\python.exe scripts\check_kickoff_tags.py    # PASS or WARN; FAIL means a kickoff tag lies more than 2 s after the published kickoff
```

If any of them prints FAIL or `raised <SomeError>`, stop and report the line.

## 2. Train and label (by hand)

```powershell
.venv\Scripts\python.exe scripts\make_submission.py all --workers 2 --out submissions\<team_name>.csv
```

- `corpus` and `worlds` should report everything up to date.
- `train` takes about 20 minutes (LightGBM 3, TCN 15 to 20). On this PC it should print these model hashes
  (checked on 2026-11-11):
  - gbdt `056ffa8a41ea…`
  - tcn `4ec1a79e…`
  - bundle `0a6359f3…`

  The full hashes are in `docs/repro/expected-windows.json`. After the run, the trained model's hashes are in `model_F3\manifest.json`
  under `KATENA_WORK`. A different gbdt hash means the training data changed: stop and report it. A different tcn or bundle hash
  alone means torch's CPU kernels changed (a new torch or CPU): report it, but it is not by itself a bug.
- `predict` takes about a minute and ends with the official checks on the written file.

## 3. Validate and check the pipeline (by hand)

```powershell
.venv\Scripts\python.exe evaluation\validate_submission.py submissions\<team_name>.csv
.venv\Scripts\python.exe scripts\check_public_window.py submissions\<team_name>.csv
Get-FileHash -Algorithm SHA256 submissions\<team_name>.csv
```

- `validate_submission.py` must say the file is valid.
- `check_public_window.py` prints PASS/FAIL only: the official checks, home/away orientation, frame
  alignment, and better than the published smooth50 baseline. **Any FAIL is a bug to find, never something
  to tune.** Don't change the model because of it.
- Record the run in `docs/decision-log.md`: the date, the commit (`git rev-parse HEAD`), the PASS/FAIL
  lines, and the CSV's SHA-256. The plan asks for every look at the public window to be logged. The one
  command prints this entry for you as its snippet.

## 4. Freeze and submit

1. Tag the exact commit that produced the file: `git tag submission-v1` then `git push origin submission-v1`.
   The release workflow refuses a tag whose commit is not on `main`.
2. Fill the hash, tag and public-window result into `docs/technical-document.md` (the `TODO(after the real
   run)` markers) and commit it. Then build the PDF the form asks for:

   ```powershell
   uv run scripts\build_document.py --team <team_name>
   ```

   It writes `submissions\<team_name>.pdf` (A4, 9 pages on 2026-11-12) with the team name in the title, and
   prints its page count and SHA-256. It refuses to build while a marker is left; `--draft` builds anyway. It
   needs Edge or Chrome, which it runs headless, and no network. The form also accepts the `.md`.
3. Submit through the organisers' Google Form (link by email by 2026-11-14; deadline 2026-11-17 23:59 CET):
   - the team name;
   - `submissions\<team_name>.csv`, after validating that exact file;
   - the repository URL, https://github.com/jusstinn/katenaball;
   - the technical document.
4. Never commit the CSV. `submissions/` is git-ignored, and the firewall check blocks it anyway.

If the model or the code changes after this, run the one command again. Add `--skip-checks` if the loader code
has not changed. By hand, start again at step 2. The form keeps the last response before the deadline.
