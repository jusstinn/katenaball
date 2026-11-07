"""Read the Smart Tagging XML as intervals on the tactical video's frame grid.

Format from docs/upstream/DATA.md and starter/load_smart_tagging.py. Team names map to sides, with
None where the tag has no team (every throw-in, for example). Tags are noisy evidence, never labels.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

HOME_NAMES = {"barcelona"}
AWAY_NAMES = {"atletic club", "athletic club"}


@dataclass(frozen=True)
class Tag:
    code: str                # lower case, e.g. "set pieces"
    start_frame: int
    end_frame: int           # first frame after the interval
    team: int | None         # 0 home, 1 away, None when untagged
    half: int | None         # 1, 2, or None
    kind: str | None         # lower-case Type tag, e.g. "throw-in", for set pieces
    tags: dict = field(default_factory=dict, hash=False, compare=False)


def _team(value: str | None) -> int | None:
    name = (value or "").strip().lower()
    if name in HOME_NAMES:
        return 0
    if name in AWAY_NAMES:
        return 1
    return None


def _half(value: str | None) -> int | None:
    name = (value or "").strip().lower()
    return {"1st half": 1, "2nd half": 2}.get(name)


def read_tags(path: str | Path, fps: float = 25.0) -> list[Tag]:
    out = []
    for instance in ET.parse(path).iter("instance"):
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        kind = labels.get("Type")
        out.append(Tag(
            code=(instance.findtext("code") or "").strip().lower(),
            start_frame=int(round(float(instance.findtext("start")) * fps)),
            end_frame=int(round(float(instance.findtext("end")) * fps)),
            team=_team(labels.get("Team")),
            half=_half(labels.get("Half")),
            kind=kind.strip().lower() if kind else None,
            tags=labels,
        ))
    return out
