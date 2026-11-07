"""Print the structure of the data bundle without printing any of its contents.

Humans run this once the bundle is in data/:

    uv run python scripts/probe_schema.py

It answers format questions the public docs leave open: extra channels in the tracking, the metadata
layout, video frame rate, and whether the videos have audio. It prints field counts, XML tag and
attribute names, CSV headers, and video stream properties. It never prints a tracking value, a name,
a label, or a frame, so the output is safe to paste into a chat.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import games  # noqa: E402

SAMPLE_LINES = 2000
VIDEO_FIELDS = ("codec_type", "codec_name", "width", "height", "r_frame_rate", "avg_frame_rate",
                "nb_frames", "duration", "channels", "sample_rate")


def probe_raw(path: str) -> None:
    shapes = Counter()
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index >= SAMPLE_LINES:
                break
            parts = line.rstrip("\n").split(":")
            groups = [len(p.split(";")) for p in parts]
            fields = [sorted({len(g.split(",")) for g in p.split(";")}) for p in parts]
            shapes[(len(parts), tuple(groups), str(fields))] += 1
    print(f"  shape of the first {SAMPLE_LINES} lines (colon parts, groups per part, fields per group):")
    for shape, count in shapes.most_common(5):
        print(f"    {count:>5} lines: {shape}")


def probe_xml(path: str) -> None:
    counts = Counter()
    attributes: dict[str, set] = {}
    for _, element in ET.iterparse(path):
        counts[element.tag] += 1
        attributes.setdefault(element.tag, set()).update(element.attrib)
        element.clear()
    print("  tag: count, attribute names")
    for tag, count in counts.most_common():
        print(f"    {tag}: {count}, {sorted(attributes[tag])}")


def probe_csv_header(path: str) -> None:
    with open(path) as handle:
        print("  columns:", handle.readline().strip())


def probe_video(path: str) -> None:
    if not shutil.which("ffprobe"):
        print("  ffprobe not found. Run: brew install ffmpeg")
        return
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  ffprobe failed with exit code {result.returncode}")
        return
    for stream in json.loads(result.stdout).get("streams", []):
        print("  ", {k: stream[k] for k in VIDEO_FIELDS if k in stream})


def main() -> None:
    print("Structure only. Safe to paste.\n")
    probes = {"atd_raw": probe_raw, "atd_meta": probe_xml, "smart_tagging": probe_xml,
              "homography": probe_csv_header, "video": probe_video}
    for source in ("tactical", "broadcast"):
        for asset, path in games.asset_paths(source=source).items():
            print(f"{source} / {asset}")
            if not Path(path).exists():
                print("  file not found")
                continue
            try:
                probes[asset](path)
            except Exception as error:  # noqa: BLE001  the type alone, as messages can quote the file
                print(f"  raised {type(error).__name__}")
        print()


if __name__ == "__main__":
    main()
