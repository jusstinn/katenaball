#!/usr/bin/env python3
"""Data firewall: fail if git holds anything this public repository must never publish.

The competition terms forbid publishing the match data or any extract of it, and the full-match
ground truth must never leak (CLAUDE.md, "Data firewall"). This checks what git would publish, never
the working tree, so untracked files under data/ are never opened. It prints paths, sizes, row
counts and the rule that matched, and never any file content.

Standard library only (Python 3.8+), so CI, the git hooks in .githooks/ and a bare interpreter can
all run it:

    python scripts/firewall_check.py                  # the index: tracked files, as about to be committed
    python scripts/firewall_check.py --history HEAD   # ...plus every blob in the history of HEAD
    python scripts/firewall_check.py --history all    # ...plus every blob reachable from any ref
    python scripts/firewall_check.py --pre-push       # the git pre-push protocol on stdin

CI also runs main's copy of this file on every other branch, so a change that loosens a rule takes
effect only once it is reviewed and merged (docs/ci.md).

Exit status: 0 clean, 1 problems found, 2 could not run (no git, not a repository).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# -- what may never be published -------------------------------------------------------------------

MAX_BLOB_BYTES = 5 * 1024 * 1024

# Committed on purpose and pinned to their git blob ids. The baselines are computed from the ATD alone
# (evaluation/results/README.md) and the public ground truth is the published 5-minute slice. Any other
# content at these paths, such as a submission saved over a baseline, is reported.
PUBLISHED = {
    "evaluation/results/baseline_raw.csv": "5551ddcb06b969e14f0cbe556e8ff921e8fe77c2",
    "evaluation/results/baseline_smooth25.csv": "a2ef72adeab130a2de1329e8a2758da81b175334",
    "evaluation/results/baseline_smooth50.csv": "1e1e46757128d5434162b2e3e05195ea00635371",
    "evaluation/results/baseline_smooth125.csv": "fb75527cf9344d4c5579e5e65dd84d834556bca9",
    "evaluation/gt/BAR-ATH_possession_gt_public.csv": "99564b6d6c81805e404e4b16d890f4a62020ae49",
    "evaluation/gt/BAR-ATH_possession_gt_public_meta.json": "c87adb00e89ee22b13770b2d88296b838c4f394b",
}
DATA_README = "data/README.md"
PRIVATE_GT = ("*_possession_gt.csv", "*_possession_gt_meta.json", "*_possession_gt_reference.csv",
              "possession_by_frame.pkl")
# Local outputs: features, caches, runs, the external corpus and the bake-off cache.
GENERATED_DIRS = ("artifacts", "cache", "features", "runs", "external_data", "work")
# File types that are data, never code or prose. Frame images and crops are match data (CLAUDE.md),
# and none of these can be read in a diff, so a reviewer could not tell a chart from a frame.
FORBIDDEN_SUFFIXES = {
    "array and table caches": (".npz", ".npy", ".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".mat"),
    "pickles": (".pkl", ".pickle", ".joblib"),
    "model weights": (".pt", ".pth", ".ckpt", ".onnx", ".safetensors"),
    "archives": (".zip", ".gz", ".gzip", ".bz2", ".xz", ".7z", ".tar", ".tgz", ".rar"),
    "video": (".mp4", ".mkv", ".avi", ".mov", ".webm", ".pl3container"),
    "documents": (".pdf",),
    "images (frame images and crops are match data)": (
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"),
    "spreadsheets": (".xlsx", ".xlsm", ".xls", ".ods"),
    "databases": (".sqlite", ".sqlite3", ".db", ".duckdb"),
}
# Images committed on purpose, by exact path. None is tracked yet. Adding one is a rule change: it
# needs review (docs/ci.md), and the image must not show the match.
ALLOWED_IMAGES: Tuple[str, ...] = ()
# File-name stems of the data bundle (games.SOURCES); a copy outside data/ is still the bundle.
BUNDLE_STEMS = ("BAR-ATH_tactical", "BAR-ATH_broadcast")

# Frame-level exports. Lengths of the match per frame: tactical video, broadcast video, graded frames
# (games.py, DATA.md), and the two graded halves, each also on the 5 Hz work grid. A text file with
# about that many lines, a line with about that many values, or a JSON array of about that length is
# one value per frame of this match, whatever its columns are called.
PER_FRAME_LENGTHS = (199319, 164488, 158857)
HALF_LENGTHS = (79467, 79390)
MATCH_LENGTHS = PER_FRAME_LENGTHS + HALF_LENGTHS
LENGTH_NAMES = {199319: "tactical frames", 164488: "broadcast frames", 158857: "graded frames",
                79467: "first-half frames", 79390: "second-half frames"}
WORK_GRID_STEP = 5
LENGTH_TOLERANCE = 0.02
SUBMISSION_MIN_ROWS = 1000
# Tracking IdFrame span of the match (games.GAMES["BAR-ATH"]["halves"]). No external match uses these
# frame numbers, so a frame column holding them, or a file listing many of them in increasing order (a
# spell list), came from this match. In order, because uv.lock lists wheel sizes in that range at
# random; numbers inside paths do not count, because SkillCorner match ids are in that range too.
ID_FRAME_SPAN = (1713242, 1892255)
ID_FRAME_MIN_RUN = 10

# Structure of the bundle's files (DATA.md, katena.target.atd). Each signature is a whole record, never
# a bare number: code is full of small integers, so a count of them proves nothing.
ATD_MIN_RECORDS = 3             # <frame_id>:<x,y,s>;<x,y,s>;...:<ball>; one docs example is allowed
SLOT_COUNTS = (46, 51)          # track slots in this match's tactical and broadcast ATD (DATA.md)
EPTS_MIN_PLAYERS = 22           # <Player ... teamId=...> entries of an ATD metadata XML
HOMOGRAPHY_MIN_ROWS = 3         # frame, h00..h22, reprojection_error, valid; DATA.md shows one row
TAGGING_MIN_RECORDS = 10        # Smart Tagging intervals; tests/test_target.py holds three
TAGGING_CODES = frozenset((     # the codes of this match's Smart Tagging (DATA.md)
    "PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PROGRESSION",
    "DEFENDING IN MIDDLE THIRD", "SET PIECES", "CREATING CHANCES", "DEFENDING IN DEFENSIVE THIRD",
    "LONG BALL", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "BUILD UP",
    "DEFENDING IN ATTACKING THIRD", "GOALS"))
# Runs of rows: frame,label-style rows (one H/A/D/X among numbers), and lines whose leading integers
# step evenly (frame ids at 25 fps or on the work grid).
LABELS = frozenset(("H", "A", "D", "X"))
LABEL_RUN_MIN = 25              # one second at 25 fps, or a spell list of about 8 minutes of play
FRAME_RUN_MIN = 100             # four seconds at 25 fps; the longest numbered run in the repo is 6
FRAME_RUN_MAX_STEP = 25         # one second

LFS_POINTER = (b"version https://git-lfs.github.com/spec/", b"version https://hawser.github.com/spec/")
LFS_FILTER = re.compile(r"(?:^|\s)filter\s*=\s*lfs(?:\s|$)")

SETTINGS = ".claude/settings.json"
DENY_DATA_READ = re.compile(r"^Read\((\./|/)?data/\*\*\)$")
CLAUDE_MD = "CLAUDE.md"
FIREWALL_HEADING = re.compile(r"^#{1,6}[ \t]+Data firewall\b", re.IGNORECASE | re.MULTILINE)

ZERO_SHA = "0" * 40


class Problem(NamedTuple):
    rule: str
    path: str
    detail: str
    where: str          # "tracked", or "history <commit>"


# -- path rules ------------------------------------------------------------------------------------

def path_problems(path: str) -> List[Tuple[str, str]]:
    """(rule, detail) for a repository-relative POSIX path, judged by its name alone."""
    out = []
    parts = path.split("/")
    name = parts[-1]
    lower = name.lower()
    folded = path.lower()
    if parts[0].lower() == "data" and len(parts) > 1 and folded != DATA_README.lower():
        out.append(("match-data", "only data/README.md may be committed under data/"))
    if folded.startswith("evaluation/results/") and lower.endswith(".csv") and path not in PUBLISHED:
        out.append(("submission", "a CSV in evaluation/results/ is a full set of labels for the match"))
    if any(fnmatch.fnmatchcase(lower, pattern.lower()) for pattern in PRIVATE_GT):
        out.append(("private-gt", "file name of the full-match ground truth"))
    for kind, suffixes in FORBIDDEN_SUFFIXES.items():
        suffix = next((s for s in suffixes if lower.endswith(s)), None)
        if suffix and not (kind.startswith("images") and path in ALLOWED_IMAGES):
            out.append(("binary", "{} files are {}".format(suffix, kind)))
            break
    generated = [p for p in parts[:-1] if p.lower() in GENERATED_DIRS]
    if generated:
        out.append(("generated-dir", "{}/ holds local outputs".format(generated[0])))
    if len(parts) == 1 and lower.endswith(".csv"):
        out.append(("root-csv", "a CSV in the repository root is a stray submission or export"))
    if any(lower.startswith(stem.lower()) for stem in BUNDLE_STEMS):
        out.append(("bundle-file", "named like a file of the match data bundle"))
    return out


# -- content rules ---------------------------------------------------------------------------------

# Each number matches one way only (no "\d+\.?\d*"), so a near-miss cannot backtrack exponentially.
_NUM = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|NaN|NAN|inf|Inf))"
_NUMBER = re.compile(_NUM + r"\Z")
_INTEGER = re.compile(r"[-+]?\d+\Z")
_GROUP = "{0},{0},{0}".format(_NUM)
# One ATD raw record: frame_id, then x,y,speed per track slot separated by ';', then the ball.
_ATD_RECORD = re.compile(r"(?<![\w.:;,])\d{1,7}:(" + _GROUP + r"(?:;" + _GROUP + r")*):"
                         + _NUM + r"(?:," + _NUM + r"){1,3}(?![\w.])")
_EPTS_PLAYER = re.compile(r"<Player\b[^>]*\bteamId\s*=")
_TAG_START = re.compile(r"<start>\s*" + _NUM + r"\s*</start>")
_TAG_END = re.compile(r"<end>\s*" + _NUM + r"\s*</end>")
_TIME = re.compile(r"(?:\d+ days? )?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\Z|\d+(?:\.\d+)?\Z")
_ID_FRAME = re.compile(r"(?<![\w./\\-])1[78]\d{5}(?![\w./\\])")
_KEY = re.compile(r"""(?:"[A-Za-z_][\w ]*"|'[A-Za-z_][\w ]*')\s*:""")   # JSON and Python dict keys
_SPLIT = re.compile(r"""[\s,;|()\[\]{}"']+""")
_VALUES = re.compile(r"[\s,;]+")


def _targets(lengths: Sequence[int]) -> List[Tuple[int, str]]:
    """(length, name): each length per frame and on the 5 Hz work grid."""
    out = []
    for n in lengths:
        name = LENGTH_NAMES.get(n, "{:,} frames".format(n))
        out.append((n, name))
        out.append((-(-n // WORK_GRID_STEP), name + " at 5 Hz"))
    return out


def _length_match(n: int, targets: Sequence[Tuple[int, str]]) -> Optional[str]:
    for target, name in targets:
        if abs(n - target) <= LENGTH_TOLERANCE * target:
            return name
    return None


def _is_number(token: str) -> bool:
    return bool(_NUMBER.match(token))


def _homography_row(line: str) -> bool:
    """frame,h00..h22,reprojection_error,valid (12 values) or the same without the frame (11). The
    matrix is normalised, so h22 is 1, or NaN where valid is 0."""
    if len(line) < 21:
        return False
    cells = _VALUES.split(line.strip())
    if len(cells) not in (11, 12) or not all(_is_number(c) for c in cells):
        return False
    if len(cells) == 12 and not _INTEGER.match(cells[0]):
        return False
    h22, valid = float(cells[-3]), float(cells[-1])
    return (h22 == 1.0 or math.isnan(h22)) and valid in (0.0, 1.0)


def _tagging_row(line: str) -> bool:
    """A delimited Smart Tagging row: one of the match's codes and a time (seconds or a timedelta)."""
    upper = line.upper()
    if not any(code in upper for code in TAGGING_CODES):
        return False
    cells = [c.strip().strip("\"'").strip() for c in re.split(r"[,\t]", line)]
    return (len(cells) >= 3 and any(c.upper() in TAGGING_CODES for c in cells)
            and any(_TIME.match(c) for c in cells))


def _json_sizes(value) -> Iterable[int]:
    """Lengths of every array and object in a parsed JSON document. Strings are left to the
    character-run rule: a long string is often an encoded blob, a long run of labels never is."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, dict)):
            yield len(node)
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())


def text_problems(data: bytes, lengths: Sequence[int] = MATCH_LENGTHS) -> List[Tuple[str, str]]:
    """(rule, detail) for the bytes of any text file, by structure only. Reports counts, never values."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    text = data.decode("utf-8", "replace")
    targets = _targets(lengths)
    shortest = min(int(n * (1 - LENGTH_TOLERANCE)) for n, _ in targets)
    out = []

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n_lines = len(lines)
    for n, how in ((n_lines, ""), (n_lines - 1, " after a header line")):
        name = _length_match(n, targets)
        if name and n > 0:
            out.append(("frame-export", "{:,} lines{}: one per {}".format(n, how, name)))
            break

    # One line holding a value or a record per frame: a JSON array, a CSV row, a Python repr.
    for line in (line for line in lines if len(line) >= 2 * shortest - 1):
        counts = (len(_VALUES.split(line.strip())), line.count("("), line.count("["), line.count("{"))
        hits = [(n, _length_match(n, targets)) for n in counts]
        hit = next(((n, name) for n, name in hits if name), None)
        if hit:
            out.append(("frame-export", "a line holding {:,} values or records: one per {}".format(*hit)))
            break
    # One character per frame: "HHHAAD..." or "0001112...".
    for run in re.finditer(r"[HADX]{%d,}|[0-9]{%d,}" % (shortest, shortest), text):
        name = _length_match(len(run.group()), targets)
        if name:
            out.append(("frame-export", "a run of {:,} label or digit characters: one per {}".format(
                len(run.group()), name)))
            break
    if text.lstrip()[:1] in ("[", "{"):
        try:
            document = json.loads(text)
        except (ValueError, RecursionError):
            document = None
        for n in _json_sizes(document) if document is not None else ():
            name = _length_match(n, targets)
            if name:
                out.append(("frame-export", "a JSON array or object of {:,} items: one per {}".format(n, name)))
                break

    ids = [int(m.group()) for m in _ID_FRAME.finditer(text)]
    ids = [i for i in ids if ID_FRAME_SPAN[0] <= i <= ID_FRAME_SPAN[1]]
    ids = [value for i, value in enumerate(ids) if not i or value != ids[i - 1]]   # end == next start
    run = best = 0
    for i, value in enumerate(ids):
        run = run + 1 if i and value > ids[i - 1] else 1
        best = max(best, run)
    if best >= ID_FRAME_MIN_RUN:
        out.append(("frame-export", "{} tracking frame numbers (IdFrame) of this match in increasing "
                                    "order".format(best)))

    atd, slots = 0, set()
    for record in _ATD_RECORD.finditer(text):
        atd += 1
        slots.add(record.group(1).count(";") + 1)
    if atd >= ATD_MIN_RECORDS or slots & set(SLOT_COUNTS):
        out.append(("atd-raw", "{:,} record(s) in the ATD raw format (frame_id:x,y,s;...:ball){}".format(
            atd, ", with this match's track-slot count" if slots & set(SLOT_COUNTS) else "")))
    players = len(_EPTS_PLAYER.findall(text))
    if players >= EPTS_MIN_PLAYERS:
        out.append(("atd-meta", "an ATD metadata XML with {} Player entries".format(players)))

    instances = sum(1 for chunk in text.split("<instance")[1:]
                    if _TAG_START.search(chunk.split("</instance>")[0])
                    and _TAG_END.search(chunk.split("</instance>")[0]))
    homography = tagging_rows = 0
    label_run = best_label_run = 0
    frame_run = best_frame_run = 0
    best_step = step = 0
    previous: Optional[int] = None
    upper = text.upper()
    any_codes = any(code in upper for code in TAGGING_CODES)
    for line in lines:
        if _homography_row(line):
            homography += 1
        if any_codes and _tagging_row(line):
            tagging_rows += 1
        tokens = [t for t in _SPLIT.split(_KEY.sub(" ", line)) if t]
        labels = [t for t in tokens if t in LABELS]
        others = [t for t in tokens if t not in LABELS]
        labelled = len(labels) == 1 and all(_is_number(t) for t in others)
        label_run = label_run + 1 if labelled else 0
        best_label_run = max(best_label_run, label_run)
        first = int(tokens[0]) if tokens and _INTEGER.match(tokens[0]) else None
        gap = first - previous if first is not None and previous is not None else 0
        if frame_run >= 2 and gap == step:
            frame_run += 1                          # the run goes on at the same step
        elif 1 <= gap <= FRAME_RUN_MAX_STEP:
            frame_run, step = 2, gap                # the previous line and this one start a run
        else:
            frame_run, step = (1 if first is not None else 0), 0
        if frame_run > best_frame_run:
            best_frame_run, best_step = frame_run, step
        previous = first
    if homography >= HOMOGRAPHY_MIN_ROWS:
        out.append(("homography", "{:,} rows shaped like the homography (frame, h00..h22, "
                                  "reprojection_error, valid)".format(homography)))
    if instances + tagging_rows >= TAGGING_MIN_RECORDS:
        out.append(("tagging", "{:,} Smart Tagging records (<instance> intervals, or code rows with a "
                               "time)".format(instances + tagging_rows)))
    if best_label_run >= LABEL_RUN_MIN:
        out.append(("frame-labels", "{:,} consecutive frame,label-style rows (one H/A/D/X among "
                                    "numbers)".format(best_label_run)))
    if best_frame_run >= FRAME_RUN_MIN:
        out.append(("frame-run", "{:,} consecutive lines whose leading integers step by {}: a run of "
                                 "frame ids".format(best_frame_run, best_step)))
    return out


def csv_problems(data: bytes, lengths: Sequence[int] = MATCH_LENGTHS) -> List[Tuple[str, str]]:
    """(rule, detail) for the bytes of a CSV or TSV: every text rule, whatever the columns are called,
    plus the header rules (submission format, frame columns holding tracking frame numbers)."""
    out = text_problems(data, lengths)
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    end = data.find(b"\n")
    header = (data if end < 0 else data[:end]).rstrip(b"\r")
    sep = b"\t" if header.count(b"\t") > header.count(b",") else b","
    columns = [c.strip().strip(b'"').strip().decode("utf-8", "replace").lower() for c in header.split(sep)]
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    rows = max(0, lines - 1)
    if columns[:2] == ["frame", "label"] and rows >= SUBMISSION_MIN_ROWS:
        out.append(("submission-format", "frame,label CSV with {:,} rows".format(rows)))
    frame_cols = [i for i, c in enumerate(columns) if "frame" in c]
    if frame_cols and _holds_match_id_frames(data[end + 1:] if end >= 0 else b"", sep, frame_cols):
        out.append(("frame-export", "a frame column holds this match's tracking frame numbers"))
    return out


def _holds_match_id_frames(body: bytes, sep: bytes, cols: List[int]) -> bool:
    lo, hi = ID_FRAME_SPAN
    for line in body.split(b"\n"):
        cells = line.split(sep)
        for i in cols:
            if i < len(cells):
                cell = cells[i].strip().strip(b'"')
                if cell.isdigit() and lo <= int(cell) <= hi:
                    return True
    return False


def notebook_problems(data: bytes) -> List[Tuple[str, str]]:
    """Saved outputs in a notebook can hold match data: rows, plots, frames."""
    try:
        notebook = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return [("notebook", "unreadable notebook, so it cannot be shown to hold no outputs")]
    if not isinstance(notebook, dict):
        return [("notebook", "unreadable notebook, so it cannot be shown to hold no outputs")]
    cells = list(notebook.get("cells") or [])
    for sheet in notebook.get("worksheets") or []:          # nbformat 3
        cells += sheet.get("cells") or []
    n = sum(1 for c in cells if isinstance(c, dict) and c.get("outputs"))
    if n:
        return [("notebook-output", "{} cell(s) with saved outputs; clear them before committing".format(n))]
    return []


def lfs_problems(path: str, data: bytes) -> List[Tuple[str, str]]:
    """Git LFS keeps the content outside git, where this scan cannot see it, and still publishes it."""
    if data.startswith(LFS_POINTER):
        return [("lfs", "a Git LFS pointer: its content is stored outside git, unscanned")]
    if path.split("/")[-1] == ".gitattributes":
        text = data.decode("utf-8", "replace")
        n = sum(1 for line in text.splitlines()
                if not line.lstrip().startswith("#") and LFS_FILTER.search(line))
        if n:
            return [("lfs", "{} filter=lfs line(s): files they match would bypass this scan".format(n))]
    return []


def blob_problems(path: str, sha: str, size: int, read: Callable[[], bytes]) -> List[Tuple[str, str]]:
    """(rule, detail) from a blob's id, size and structure."""
    pinned = PUBLISHED.get(path)
    if pinned is not None:
        if sha != pinned:
            return [("published-modified", "differs from the published file pinned in scripts/firewall_check.py")]
        return []
    if size > MAX_BLOB_BYTES:
        return [("large", "{:,} bytes, over the {} MiB limit".format(size, MAX_BLOB_BYTES // 2 ** 20))]
    if path in ALLOWED_IMAGES:
        return []
    data = read()
    found = lfs_problems(path, data)
    if found:
        return found
    if b"\x00" in data:
        return [("binary-content", "binary content (NUL bytes) under a text or unknown file name")]
    lower = path.lower()
    if lower.endswith((".csv", ".tsv")):
        return csv_problems(data)
    found = text_problems(data)
    if lower.endswith(".ipynb"):
        found = notebook_problems(data) + found
    return found


def config_problems(read_file: Callable[[str], Optional[bytes]]) -> List[Tuple[str, str, str]]:
    """(path, rule, detail): the guard rails that keep match data out of Claude's context."""
    out = []
    raw = read_file(SETTINGS)
    deny = None
    if raw is not None:
        try:
            deny = json.loads(raw.decode("utf-8")).get("permissions", {}).get("deny", [])
        except (UnicodeDecodeError, ValueError, AttributeError):
            deny = None
    if raw is None:
        out.append((SETTINGS, "settings", "missing; it must deny Read(./data/**)"))
    elif not isinstance(deny, list):
        out.append((SETTINGS, "settings", "unreadable; it must deny Read(./data/**)"))
    elif not any(isinstance(rule, str) and DENY_DATA_READ.match(rule.strip()) for rule in deny):
        out.append((SETTINGS, "settings", "permissions.deny no longer contains Read(./data/**)"))
    raw = read_file(CLAUDE_MD)
    if raw is None or not FIREWALL_HEADING.search(raw.decode("utf-8", "replace")):
        out.append((CLAUDE_MD, "claude-md", "the 'Data firewall' section is missing"))
    return out


# -- git -------------------------------------------------------------------------------------------

class GitError(Exception):
    pass


class Git:
    def __init__(self, repo: str):
        self.repo = repo
        self._blobs: Dict[str, bytes] = {}

    def run(self, *args: str, stdin: Optional[bytes] = None, check: bool = True) -> bytes:
        try:
            done = subprocess.run(["git", "-C", self.repo] + list(args), input=stdin,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as error:
            raise GitError("cannot run git: {}".format(error))
        if check and done.returncode:
            message = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise GitError("git {} failed: {}".format(args[0], message[-1] if message else done.returncode))
        return done.stdout

    def exists(self, rev: str) -> bool:
        try:
            self.run("cat-file", "-e", rev)
            return True
        except GitError:
            return False

    def index(self) -> List[Tuple[str, str]]:
        """(path, blob) for every regular file and symlink in the index."""
        out = []
        for entry in _split_z(self.run("ls-files", "-s", "-z")):
            meta, path = entry.split("\t", 1)
            mode, sha, _stage = meta.split()
            if mode != "160000":
                out.append((path, sha))
        return out

    def tree(self, rev: str) -> List[Tuple[str, str]]:
        out = []
        for entry in _split_z(self.run("ls-tree", "-r", "-z", "--full-tree", rev)):
            meta, path = entry.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob":
                out.append((path, sha))
        return out

    def history(self, revs: Sequence[str]) -> List[Tuple[str, str, str]]:
        """(path, blob, commit) for every blob a commit in ``revs`` introduced at every path, plus any
        other blob ``git rev-list --objects`` reaches (it lists each object once, at its first path)."""
        found: Dict[Tuple[str, str], str] = {}
        raw = self.run("log", "--raw", "-z", "--no-renames", "--no-abbrev", "--diff-merges=first-parent",
                       "--format=%x01%H", *revs)
        commit = ""
        tokens = raw.split(b"\x00")
        i = 0
        while i < len(tokens):
            token = tokens[i].decode("utf-8", "surrogateescape").strip("\n")
            i += 1
            if token.startswith("\x01"):
                commit = token[1:].strip()
            elif token.startswith(":") and i < len(tokens):
                fields = token[1:].split()
                path = tokens[i].decode("utf-8", "surrogateescape")
                i += 1
                if len(fields) >= 5 and fields[1] != "160000" and fields[3] != ZERO_SHA:
                    found.setdefault((path, fields[3]), commit)
        listing = self.run("rev-list", "--objects", *revs).decode("utf-8", "surrogateescape")
        objects = [line.split(" ", 1) for line in listing.splitlines() if " " in line]
        kinds = self.kinds([sha for sha, _ in objects])
        for sha, path in objects:
            if kinds.get(sha, ("", 0))[0] == "blob":
                found.setdefault((path, sha), "")
        return [(path, sha, commit) for (path, sha), commit in found.items()]

    def kinds(self, shas: Iterable[str]) -> Dict[str, Tuple[str, int]]:
        """{sha: (type, size)} in one git call."""
        shas = sorted(set(shas))
        if not shas:
            return {}
        raw = self.run("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                       stdin="".join(s + "\n" for s in shas).encode())
        out = {}
        for line in raw.decode().splitlines():
            parts = line.split()
            if len(parts) == 3:
                out[parts[0]] = (parts[1], int(parts[2]))
        return out

    def blob(self, sha: str) -> bytes:
        if sha not in self._blobs:
            self._blobs[sha] = self.run("cat-file", "blob", sha)
        return self._blobs[sha]

    def prefetch(self, shas: Iterable[str]) -> None:
        """Read many blobs in one git call: one process per blob takes seconds on Windows."""
        wanted = sorted(set(shas) - set(self._blobs))
        if not wanted:
            return
        raw = self.run("cat-file", "--batch", stdin="".join(s + "\n" for s in wanted).encode())
        pos = 0
        while pos < len(raw):
            end = raw.index(b"\n", pos)
            header = raw[pos:end].decode("utf-8", "replace").split()
            pos = end + 1
            if len(header) != 3:                        # "<sha> missing": blob() will report it
                continue
            size = int(header[2])
            self._blobs[header[0]] = raw[pos:pos + size]
            pos += size + 1

    def file_at(self, rev: str, path: str) -> Optional[bytes]:
        """A file's content in the index (rev "") or in a commit, or None if absent."""
        spec = ":{}".format(path) if rev == "" else "{}:{}".format(rev, path)
        return self.run("cat-file", "blob", spec) if self.exists(spec) else None


def _split_z(raw: bytes) -> List[str]:
    return [t.decode("utf-8", "surrogateescape") for t in raw.split(b"\x00") if t]


# -- checks ----------------------------------------------------------------------------------------

def check_entries(git: Git, entries: Iterable[Tuple[str, str, str]]) -> List[Problem]:
    """entries: (path, blob, where). Path rules first; content is read only for paths that passed."""
    entries = sorted({(p, s, w) for p, s, w in entries})
    kinds = git.kinds(s for _, s, _ in entries)
    by_path = {path: path_problems(path) for path, _, _ in entries}
    git.prefetch(s for p, s, _ in entries
                 if not by_path[p] and p not in PUBLISHED and p not in ALLOWED_IMAGES
                 and kinds.get(s, ("", 0))[0] == "blob" and kinds[s][1] <= MAX_BLOB_BYTES)
    problems = []
    verdicts: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for path, sha, where in entries:
        found = by_path[path]
        if not found:
            kind, size = kinds.get(sha, ("blob", 0))
            if kind != "blob":
                continue
            if (path, sha) not in verdicts:
                verdicts[path, sha] = blob_problems(path, sha, size, lambda sha=sha: git.blob(sha))
            found = verdicts[path, sha]
        problems += [Problem(rule, path, detail, where) for rule, detail in found]
    return problems


def check_config(git: Git, rev: str, where: str) -> List[Problem]:
    return [Problem(rule, path, detail, where)
            for path, rule, detail in config_problems(lambda p: git.file_at(rev, p))]


def check_index(git: Git) -> Tuple[List[Problem], int]:
    entries = git.index()
    problems = check_entries(git, [(p, s, "tracked") for p, s in entries]) + check_config(git, "", "tracked")
    return problems, len(entries)


def check_history(git: Git, revs: Sequence[str]) -> List[Problem]:
    return check_entries(git, [(p, s, "history " + c[:10] if c else "history")
                               for p, s, c in git.history(revs)])


def check_pre_push(git: Git, lines: Iterable[str]) -> Tuple[List[Problem], int]:
    """git's pre-push protocol: '<local ref> <local sha> <remote ref> <remote sha>' per line."""
    problems, refs = [], 0
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or fields[1] == ZERO_SHA:
            continue                                     # malformed, or deleting a remote ref
        local, remote = fields[1], fields[3]
        refs += 1
        tip = "{}^{{commit}}".format(local)
        problems += check_entries(git, [(p, s, "pushed tip") for p, s in git.tree(tip)])
        problems += check_config(git, tip, "pushed tip")
        if remote != ZERO_SHA and git.exists(remote + "^{commit}"):
            revs = ["{}..{}".format(remote, local)]
        else:
            revs = [local, "--not", "--remotes"]
        problems += check_history(git, revs)
    return problems, refs


# -- report ----------------------------------------------------------------------------------------

HINTS = {
    "tracked": "Untrack it with `git rm --cached -- <path>` (your local copy stays) and make sure .gitignore "
               "covers it. For a notebook, clear its outputs and add it again.",
    "history": "It is in a commit. If that commit was never pushed, rewrite it before pushing. If it was "
               "pushed, it is public now: stop and tell the team. Removing it takes a history rewrite and a "
               "force push, which only the repository owner should do.",
}


def report(problems: List[Problem], summary: str, out=sys.stdout) -> int:
    unique = sorted(set(problems), key=lambda p: (p.path, p.rule, p.where))
    if not unique:
        print("firewall: OK  {}".format(summary), file=out)
        return 0
    for p in unique:
        print("FAIL  [{}] {}  ({}): {}".format(p.rule, p.path, p.where, p.detail), file=out)
    print("\nfirewall: {} problem(s)  {}".format(len(unique), summary), file=out)
    kinds = {"history" if p.where.startswith("history") else "tracked" for p in unique}
    for kind in sorted(kinds, reverse=True):
        print("  {}".format(HINTS[kind]), file=out)
    print("  Nothing above prints file contents. Rules: docs/ci.md, scripts/firewall_check.py. A rule that "
          "is wrong is changed in its own reviewed pull request, never loosened to let a file through.",
          file=out)
    return 1


def main(argv: Optional[Sequence[str]] = None, stdin=None, out=None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(description=(__doc__ or "Data firewall").splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository to check (default: the current directory)")
    parser.add_argument("--history", metavar="REV", help="also check every blob reachable from REV "
                        "('all' = every ref, as CI does on main and pull requests)")
    parser.add_argument("--pre-push", action="store_true", help="read git's pre-push lines from stdin and "
                        "check the pushed tips and every commit not yet on a remote")
    args = parser.parse_args(argv)
    try:
        git = Git(args.repo)
        git.run("rev-parse", "--git-dir")
        if args.pre_push:
            problems, refs = check_pre_push(git, (stdin or sys.stdin).read().splitlines())
            return report(problems, "({} ref(s) pushed)".format(refs), out)
        problems, n_tracked = check_index(git)
        summary = "({} tracked files".format(n_tracked)
        if args.history:
            revs = ["--all"] if args.history == "all" else [args.history]
            problems += check_history(git, revs)
            summary += ", history of {}".format(args.history)
        return report(problems, summary + ")", out)
    except GitError as error:
        print("firewall: could not run: {}".format(error), file=out)
        return 2


if __name__ == "__main__":
    sys.exit(main())
