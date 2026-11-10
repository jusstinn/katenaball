#!/usr/bin/env python
"""Human check of the kickoff rule (W4a): does each half's Smart Tagging kickoff land on the published one?

    uv run python scripts/check_kickoff_tags.py [--data-dir <bundle>]

A human runs this on the real match. It prints PASS or FAIL lines only, never a frame, a time or a count, so the
output is safe to paste. Exceptions print their type only, because some exception messages can quote the file.

For each graded half it derives the kickoff exactly as ``katena.pipeline.apply_kickoff_rule`` does: the half's one
Smart Tagging instance whose Type is ``Kick Off (Start)`` (Type and Half matched case-insensitively), its start
time, ``round(seconds * 25)``. It compares that frame with the kickoff the organisers publish:

* 1st half: the first graded frame, 11,185 (DATA.md, "Which frames are graded"). FAQ.md says the first half is
  graded from the kickoff itself.
* 2nd half: 113,742 (FAQ.md, "Which frames are actually graded?").

These two numbers are check references only. Nothing in the pipeline reads them.

Per half, two lines:

* within 2 s: the tag-derived kickoff lies within 2 s of the published one;
* not late: it is not more than 2 s after the published one. When it is earlier, the rule sets fewer frames to D
  than it could, which is safe. When it is later, the rule would turn frames of play into D.

Exit code 0 when every line passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "evaluation"):
    sys.path.insert(0, str(path))

import games  # noqa: E402

PUBLISHED_KICKOFF = {1: 11185, 2: 113742}      # video frames; DATA.md and FAQ.md (check references only)
TOLERANCE_S = 2.0
HALF_NAME = {1: "1st half", 2: "2nd half"}
RESULTS = {"pass": 0, "fail": 0}


def report(name: str, ok: bool, why: str = "") -> None:
    RESULTS["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + why + ')' if why and not ok else ''}")


def check(tags, fps: float) -> None:
    from katena.pipeline import tagged_kickoffs
    kick = tagged_kickoffs(tags)
    tol = int(round(TOLERANCE_S * fps))
    for half in sorted(PUBLISHED_KICKOFF):
        frames = kick.get(half, [])
        near = f"{HALF_NAME[half]}: the tag-derived kickoff lies within 2 s of the published kickoff"
        late = f"{HALF_NAME[half]}: the tag-derived kickoff is not more than 2 s after the published kickoff"
        if len(frames) != 1:
            why = "no Kick Off (Start) tag for this half" if not frames else "several Kick Off (Start) tags for this half"
            report(near, False, why)
            report(late, False, why)
            continue
        delta = frames[0] - PUBLISHED_KICKOFF[half]
        report(near, abs(delta) <= tol, "more than 2 s away")
        report(late, delta <= tol, "later: the rule would set frames of play to D")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=None, help="the data bundle (default: MTAH_DATA_DIR or data/)")
    a = ap.parse_args(argv)
    print("Output holds no match values. Safe to paste.\n")
    fps = float(games.GAMES[games.GAME]["fps"])
    try:
        from katena.target.tagging import read_tags
        path = games.require(["smart_tagging"], data_dir=str(a.data_dir) if a.data_dir else None)[0]
        tags = read_tags(path, fps=fps)
    except Exception as e:  # noqa: BLE001  the type alone, as messages can quote the file
        report("reading the Smart Tagging file", False, f"raised {type(e).__name__}")
        print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed")
        return 1
    try:
        check(tags, fps)
    except Exception as e:  # noqa: BLE001
        report("checking the tagged kickoffs", False, f"raised {type(e).__name__}")
    print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
