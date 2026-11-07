"""Check the FIFA/EPTS layout of the competition ATD metadata, without showing any of it.

Humans run this once the data bundle is in data/:

    uv run python scripts/check_target_layout.py

katena.target.atd and starter/baseline.py assume that the raw player groups follow the order of the
<Player> elements, with x, y, speed in each group, and that the first <Team> is home. The published
baseline files cannot test this, because the organisers' baseline makes the same assumptions. This
script follows the metadata's own DataFormatSpecification instead. It prints booleans, counts and
channel ids only, never a position, name, id or frame, so the output is safe to paste. Exceptions
print their type only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

import games  # noqa: E402

from katena.target.layout import assumptions_hold, layout_report  # noqa: E402


def main() -> int:
    print("Output holds no match values. Safe to paste.\n")
    status = 0
    for source in ("tactical", "broadcast"):
        print(source)
        try:
            rep = layout_report(games.asset_paths(source=source)["atd_meta"],
                                n_frames=games.source_frames(source))
        except Exception as error:  # noqa: BLE001  the type alone, as messages can quote the file
            print(f"  raised {type(error).__name__}\n")
            status = 1
            continue
        for key, value in rep.items():
            print(f"  {key}: {value}")
        ok = assumptions_hold(rep)
        print(f"  {'PASS' if ok else 'FAIL'}  katena.target.atd reads this layout as intended\n")
        status |= 0 if ok else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
