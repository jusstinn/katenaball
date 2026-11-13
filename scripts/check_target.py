"""Check that our loaders read the real match correctly, without showing any of it.

Humans run this once the data bundle is in data/:

    uv run python scripts/check_target.py

The output is only pass/fail against numbers the organisers already publish, plus mismatch counts.
It holds no match values, so it is safe to paste into a chat. Exceptions print their type only,
because some exception messages can quote the file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "starter"):
    sys.path.insert(0, str(path))

import games  # noqa: E402
from baseline import smooth  # noqa: E402

from katena.external.common import centroid_gap_m  # noqa: E402
from katena.target.atd import load_target, nearest_team, read_raw, read_slot_teams  # noqa: E402
from katena.target.tagging import read_tags  # noqa: E402

# Published in docs/upstream/DATA.md.
PUBLISHED = {
    "n_frames": 199319,
    "n_slots": 46,
    "ball_present_all": 113488,
    "ball_present_graded": 94214,
    "n_graded": 158857,
    "n_tags": 638,
}
BASELINES = (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50),
             ("baseline_smooth125", 125))
# The attack direction is read from the two teams' mean positions. On every half of the 30 external matches
# they are 10.9 m or more apart, so less than this means the direction cannot be trusted. Fixed from external
# matches only; the check prints pass or fail, never the distance.
MIN_CENTROID_GAP_M = 8.0
RESULTS = {"pass": 0, "fail": 0}


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def step(name: str, function) -> None:
    try:
        function()
    except Exception as error:  # noqa: BLE001  the type alone, as messages can quote the file
        report(name, False, f"raised {type(error).__name__}")


def read_published(path: Path) -> np.ndarray:
    with open(path) as handle:
        return np.array([row["label"].strip() for row in csv.DictReader(handle)], dtype="<U1")


def check_loader() -> None:
    match = load_target()
    read = match.meta["read_report"]
    report("one record per video frame, numbered from 1", read["contiguous_from_one"])
    report("no malformed or duplicate records",
           read["n_malformed"] == 0 and read["n_duplicate"] == 0 and read["n_out_of_range"] == 0)
    report("frame count matches the published count", match.n == PUBLISHED["n_frames"])
    report("slot count matches the published count", match.n_slots == PUBLISHED["n_slots"])

    present = np.isfinite(match.ball).all(axis=1)
    graded = np.zeros(match.n, dtype=bool)
    for _, first, last in match.half_blocks:
        graded[first:last + 1] = True
    report("graded frame count matches the published count",
           int(graded.sum()) == PUBLISHED["n_graded"])
    report("ball-present count matches the published count (all frames)",
           int(present.sum()) == PUBLISHED["ball_present_all"])
    report("ball-present count matches the published count (graded frames)",
           int((present & graded).sum()) == PUBLISHED["ball_present_graded"])

    report("both teams have slots", set(np.unique(match.team).tolist()) == {0, 1})
    directions = [match.attack_dir[half] for half, _, _ in match.half_blocks]
    report("attack direction found for both halves, and it flips at half-time",
           0 not in directions and directions[0] == -directions[1])
    gaps = [centroid_gap_m(match.P, match.team, first, last) for _, first, last in match.half_blocks]
    report(f"the two teams' mean positions are at least {MIN_CENTROID_GAP_M:g} m apart along the pitch in both halves",
           all(gap >= MIN_CENTROID_GAP_M for gap in gaps))


def check_baselines() -> None:
    paths = games.asset_paths()
    pitch = tuple(float(v) for v in games.GAMES[games.GAME]["pitch"])
    team = read_slot_teams(paths["atd_meta"])
    P, _, ball, _ = read_raw(paths["atd_raw"], PUBLISHED["n_frames"], len(team), pitch,
                             dtype=np.float64)
    codes = nearest_team(P, ball, team)
    for name, window in BASELINES:
        published = read_published(ROOT / "evaluation" / "results" / f"{name}.csv")
        mismatches = int((smooth(codes, window) != published).sum())
        report(f"reproduces published {name}", mismatches == 0, f"{mismatches} frames differ")


def check_tags() -> None:
    tags = read_tags(games.asset_paths()["smart_tagging"])
    report("Smart Tagging instance count matches the published count",
           len(tags) == PUBLISHED["n_tags"])
    report("every tag sits inside the video",
           all(0 <= t.start_frame <= t.end_frame <= PUBLISHED["n_frames"] for t in tags))
    report("team names all map to a side or to no team",
           any(t.team == 0 for t in tags) and any(t.team == 1 for t in tags))


def main() -> int:
    print("Output holds no match values. Safe to paste.\n")
    step("loader", check_loader)
    step("baseline reproduction", check_baselines)
    step("smart tagging", check_tags)
    print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
