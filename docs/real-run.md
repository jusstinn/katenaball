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

## 1. Checks on the provided files (once)

```powershell
.venv\Scripts\python.exe scripts\check_target.py          # 16/16 PASS on 2026-11-07; re-run after any loader change
.venv\Scripts\python.exe scripts\check_target_layout.py   # never run yet: expect "PASS katena.target.atd reads this layout as intended"
.venv\Scripts\python.exe scripts\check_touch_target.py    # an alarm only: it must finish and print a feature shape
```

If any of them prints FAIL or `raised <SomeError>`, stop and report the line.

## 2. Train and label

Replace `<team_name>` with the team name registered on the form (the file must be named after it).

```powershell
.venv\Scripts\python.exe scripts\make_submission.py all --workers 2 --out submissions\<team_name>.csv
```

- `corpus` and `worlds` should report everything up to date.
- `train` takes about 5 minutes. On this PC it should print these model hashes (checked on 2026-11-08):
  - gbdt `b2146150e006…`
  - tcn `01488c84…`
  - bundle `cb0ea4fc…`

  A different gbdt hash means the training data changed: stop and report it. A different tcn or bundle hash
  alone means torch's CPU kernels changed (a new torch or CPU): report it, but it is not by itself a bug.
- `predict` takes about a minute and ends with the official checks on the written file.

## 3. Validate and check the pipeline

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
  lines, and the CSV's SHA-256. The plan asks for every look at the public window to be logged.

## 4. Freeze and submit

1. Tag the exact commit that produced the file: `git tag submission-v1` then `git push origin submission-v1`.
   The release workflow refuses a tag whose commit is not on `main`.
2. Fill the hash, tag and public-window result into `docs/technical-document.md` (the `TODO(after the real
   run)` markers), and export it as `<team_name>.pdf` or keep it as `.md`.
3. Submit through the organisers' Google Form (link by email by 2026-11-14; deadline 2026-11-17 23:59 CET):
   - the team name;
   - `submissions\<team_name>.csv`, after validating that exact file;
   - the repository URL, https://github.com/jusstinn/katenaball;
   - the technical document.
4. Never commit the CSV. `submissions/` is git-ignored, and the firewall check blocks it anyway.

If the model or the code changes after this, start again at step 2. The form keeps the last response before
the deadline.
