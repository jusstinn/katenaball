#!/usr/bin/env python3
"""Release note for a submission tag: commit, uv.lock sha256 and the synthetic determinism hashes.

    python scripts/release_notes.py --tag submission-1 --commit SHA --run-url URL HASHES.json ...

The note carries no data and no CSV. The real submission can only be built on a machine with the
data bundle, so the note records what makes that build reproducible instead. Standard library only.
CI also runs it as a dry run into the job summary, so a tag never meets it for the first time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def committed_sha256(commit: str, path: str) -> str:
    """sha256 of a file as committed (LF), independent of how this checkout wrote it to disk."""
    blob = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", "{}:{}".format(commit, path)],
                          stdout=subprocess.PIPE, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def note(tag: str, commit: str, run_url: str, hash_files) -> str:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in hash_files]
    runs.sort(key=lambda r: r["env"]["label"])
    keys = sorted(set().union(*(r["sha256"] for r in runs))) if runs else []
    differ = [k for k in keys if len({r["sha256"].get(k) for r in runs}) > 1]
    lines = [
        "## {}".format(tag), "",
        "| | |", "|---|---|",
        "| Commit | `{}` |".format(commit),
        "| uv.lock sha256 | `{}` |".format(committed_sha256(commit, "uv.lock")),
        "| Python | {} (uv {}) |".format(os.environ.get("UV_PYTHON", "see .python-version"),
                                       os.environ.get("UV_VERSION", "?")),
    ]
    if run_url:
        lines.append("| CI run | {} |".format(run_url))
    lines += ["", "### Synthetic determinism hashes", ""]
    if not runs:
        lines.append("No determinism hashes were available.")
    else:
        names = [r["env"]["label"] for r in runs]
        verdict = ("Across OSes they are identical." if not differ else
                   "Across OSes {} of {} differ ({}), an early warning for the two-machine hash test rather "
                   "than a failure.".format(len(differ), len(keys), ", ".join(differ)))
        lines += ["CI ran the synthetic pipeline (scripts/synthetic_e2e.py) twice per OS and the two runs "
                  "were byte-identical. {}".format(verdict), "",
                  "| sha256 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        lines += ["| {} | {} |".format(k, " | ".join("`{}`".format((r["sha256"].get(k) or "-")[:16]) for r in runs))
                  for k in keys]
        lines += ["", "<details><summary>Full hashes and versions</summary>", "", "```"]
        for r in runs:
            env = r["env"]
            lines.append("{label}: {os} {machine}, Python {python}, numpy {numpy}, lightgbm {lightgbm}, "
                         "numba {numba}".format(**env))
            lines += ["  {:<16} {}".format(k, v) for k, v in sorted(r["sha256"].items())]
        lines += ["```", "</details>"]
    lines += ["", "This release has no assets. The match data and every file derived from it stay off "
              "GitHub; the submission is regenerated locally from the data bundle with the command in the "
              "technical document, and its sha256 is compared with the one recorded there."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("hashes", nargs="*", help="hashes.json files from the determinism jobs")
    args = parser.parse_args(argv)
    sys.stdout.write(note(args.tag, args.commit, args.run_url, args.hashes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
