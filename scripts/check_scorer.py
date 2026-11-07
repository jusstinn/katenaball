#!/usr/bin/env python3
"""CI's scorer job: the official scorer on published files only, and the validator's contract.

    uv run python scripts/check_scorer.py

1. The scorer is the specification, so evaluation/*.py and games.py must match the published copies
   (pinned by git blob id; update the pins only when the organisers publish a new version).
2. evaluation/run_evaluation.py scores the four published baselines against the public 5-minute
   ground truth, skips none, and reproduces the public-window spellF1 that EVALUATION.md prints.
3. evaluation/validate_submission.py accepts the published baselines and synthetic files that
   SUBMISSION.md calls fine (CRLF, lower case, padded labels), and rejects synthetic files that break
   the contract (row count, frame numbering, blank or unknown labels, header).

Reads only published files: the four baselines are copied by name into a temporary directory, so
whatever else sits in a local evaluation/results/ is never scored. Nothing under data/ is touched.
"""

from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALUATION = REPO / "evaluation"

UPSTREAM = {
    "evaluation/possession_metrics.py": "767494a053628cbdc81a8e9deaf43ef21cc7e58a",
    "evaluation/run_evaluation.py": "16adc82b66ec15c7d4bb1ca86a72eceb7327d45f",
    "evaluation/validate_submission.py": "9b25e3fc9f4973f5b58f1d58fd19ef80d34b206b",
    "games.py": "d330dc11b3c54d9c704f0c9df7de111395f9f1d4",
}
BASELINES = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
# docs/upstream/EVALUATION.md, "The public window is not a leaderboard": public-window spellF1.
PUBLISHED_PUBLIC_SPELL_F1 = {"baseline_smooth50": 0.3256, "baseline_smooth125": 0.4242}
N_FRAMES = 199319

FAILURES = []


def expect(name: str, ok: bool, detail: str = "") -> None:
    print("{}  {}{}".format("PASS" if ok else "FAIL", name, "  ({})".format(detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def blob_id(path: Path) -> str:
    """git's id for the file as it would be committed (line endings normalised), or '' without git."""
    try:
        done = subprocess.run(["git", "-C", str(REPO), "hash-object", "--", str(path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return ""
    return done.stdout.decode().strip() if done.returncode == 0 else ""


def validate(path: Path, rows: int = 0) -> int:
    command = [sys.executable, str(EVALUATION / "validate_submission.py"), str(path)]
    if rows:
        command += ["--rows", str(rows)]
    return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def synthetic_labels(n: int, seed: int = 0) -> list:
    """Runs of H/A/D/X with random lengths. Synthetic: no relation to any match."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out += [rng.choice("HADX")] * rng.randint(1, 400)
    return out[:n]


def write(path: Path, header: str, rows, eol: str = "\n") -> Path:
    with open(path, "w", encoding="ascii", newline="") as handle:
        handle.write(header + eol)
        handle.writelines("{},{}{}".format(frame, label, eol) for frame, label in rows)
    return path


def main() -> int:
    print("-- the scorer is the published version")
    for name, pinned in UPSTREAM.items():
        found = blob_id(REPO / name)
        expect("{} unchanged".format(name), found == pinned, "" if found == pinned else "blob " + (found or "unknown"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results, board_dir = tmp / "results", tmp / "board"
        results.mkdir()
        for name in BASELINES:
            shutil.copyfile(EVALUATION / "results" / "{}.csv".format(name), results / "{}.csv".format(name))

        print("\n-- official scorer: published baselines against the public ground truth")
        done = subprocess.run([sys.executable, str(EVALUATION / "run_evaluation.py"), "--gt", "public",
                               "--results", str(results), "--out", str(board_dir), "--no-detail"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(done.stdout.decode("utf-8", "replace").rstrip())
        expect("run_evaluation.py exits 0 with nothing skipped", done.returncode == 0)
        board = {}
        if (board_dir / "scoreboard.csv").is_file():
            with open(board_dir / "scoreboard.csv", encoding="utf-8", newline="") as handle:
                board = {row["submission"]: row for row in csv.DictReader(handle)}
        expect("all four baselines scored", sorted(board) == sorted(BASELINES), ", ".join(sorted(board)))
        for name, value in PUBLISHED_PUBLIC_SPELL_F1.items():
            got = float(board[name]["spell_f1_mean"]) if name in board else float("nan")
            expect("{} public spellF1 matches EVALUATION.md".format(name), abs(got - value) <= 5e-5,
                   "{:.4f} vs {:.4f}".format(got, value))

        print("\n-- validate_submission.py: published baselines and the contract, on synthetic files")
        for name in BASELINES:
            expect("accepts {}".format(name), validate(results / "{}.csv".format(name)) == 0)
        labels = synthetic_labels(N_FRAMES)
        good = list(enumerate(labels))
        cases = [
            ("accepts a valid synthetic file", 0, write(tmp / "ok.csv", "frame,label", good), 0),
            ("accepts CRLF line endings", 0, write(tmp / "crlf.csv", "frame,label", good, "\r\n"), 0),
            ("accepts lower case, padded labels and header", 0,
             write(tmp / "loose.csv", " Frame , LABEL ", [(f, " {} ".format(v.lower())) for f, v in good]), 0),
            ("accepts --rows for a short synthetic file", 0, write(tmp / "short.csv", "frame,label", good[:1000]), 1000),
            ("rejects a missing last row", 1, write(tmp / "short_by_one.csv", "frame,label", good[:-1]), 0),
            ("rejects frames numbered from 1", 1,
             write(tmp / "from_one.csv", "frame,label", [(f + 1, v) for f, v in good]), 0),
            ("rejects frames out of order", 1,
             write(tmp / "swapped.csv", "frame,label", [good[1], good[0]] + good[2:]), 0),
            ("rejects a blank label", 1, write(tmp / "blank.csv", "frame,label", good[:5] + [(5, "")] + good[6:]), 0),
            ("rejects an unknown label", 1, write(tmp / "unknown.csv", "frame,label", good[:5] + [(5, "Z")] + good[6:]), 0),
            ("rejects a wrong header", 1, write(tmp / "header.csv", "frame,labels", good), 0),
        ]
        for name, expected, path, rows in cases:
            code = validate(path, rows)
            expect(name, code == expected, "exit {}".format(code))

    print("\n{} check(s) failed".format(len(FAILURES)) if FAILURES else "\nall scorer checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
