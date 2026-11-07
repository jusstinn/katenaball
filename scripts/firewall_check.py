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

Exit status: 0 clean, 1 problems found, 2 could not run (no git, not a repository).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
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
FORBIDDEN_SUFFIXES = (
    ".npz", ".npy", ".parquet", ".feather", ".arrow", ".h5", ".hdf5",       # array caches
    ".pkl", ".pickle", ".joblib",                                           # pickles
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors",                        # model weights
    ".zip", ".gz", ".gzip", ".bz2", ".xz", ".7z", ".tar", ".tgz", ".rar",   # archives
    ".mp4", ".mkv", ".avi", ".mov", ".pl3container",                        # video and the bundle
    ".pdf",                                                                 # documents
)
# File-name stems of the data bundle (games.SOURCES); a copy outside data/ is still the bundle.
BUNDLE_STEMS = ("BAR-ATH_tactical", "BAR-ATH_broadcast")

# Frame-level exports. Lengths of the match per frame: tactical video, broadcast video, graded frames
# (games.py, DATA.md), also on the 5 Hz work grid. A CSV with a frame column and about that many rows
# is one row per frame of this match.
PER_FRAME_LENGTHS = (199319, 164488, 158857)
WORK_GRID_STEP = 5
LENGTH_TOLERANCE = 0.02
SUBMISSION_MIN_ROWS = 1000
# Tracking IdFrame span of the match (games.GAMES["BAR-ATH"]["halves"]). No external match uses these
# frame numbers, so a frame column holding them was computed from this match.
ID_FRAME_SPAN = (1713242, 1892255)

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


# -- rules -----------------------------------------------------------------------------------------

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
    suffix = next((s for s in FORBIDDEN_SUFFIXES if lower.endswith(s)), None)
    if suffix:
        out.append(("binary", "{} files are caches, archives, video, documents or weights".format(suffix)))
    generated = [p for p in parts[:-1] if p.lower() in GENERATED_DIRS]
    if generated:
        out.append(("generated-dir", "{}/ holds local outputs".format(generated[0])))
    if len(parts) == 1 and lower.endswith(".csv"):
        out.append(("root-csv", "a CSV in the repository root is a stray submission or export"))
    if any(lower.startswith(stem.lower()) for stem in BUNDLE_STEMS):
        out.append(("bundle-file", "named like a file of the match data bundle"))
    return out


def csv_problems(data: bytes, lengths: Sequence[int] = PER_FRAME_LENGTHS) -> List[Tuple[str, str]]:
    """(rule, detail) for the bytes of a CSV or TSV. Reads structure only and reports counts."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    end = data.find(b"\n")
    header = (data if end < 0 else data[:end]).rstrip(b"\r")
    sep = b"\t" if header.count(b"\t") > header.count(b",") else b","
    columns = [c.strip().strip(b'"').strip().decode("utf-8", "replace").lower() for c in header.split(sep)]
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    rows = max(0, lines - 1)
    out = []
    if columns[:2] == ["frame", "label"] and rows >= SUBMISSION_MIN_ROWS:
        out.append(("submission-format", "frame,label CSV with {:,} rows".format(rows)))
    frame_cols = [i for i, c in enumerate(columns) if "frame" in c]
    if not frame_cols:
        return out
    targets = list(lengths) + [-(-n // WORK_GRID_STEP) for n in lengths]
    if any(abs(rows - n) <= LENGTH_TOLERANCE * n for n in targets):
        out.append(("frame-export", "{:,} rows with a frame column: one row per frame of the match".format(rows)))
    if _holds_match_id_frames(data[end + 1:] if end >= 0 else b"", sep, frame_cols):
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


def blob_problems(path: str, sha: str, size: int, read: Callable[[], bytes]) -> List[Tuple[str, str]]:
    """(rule, detail) from a blob's id, size and, for CSVs and notebooks, its structure."""
    pinned = PUBLISHED.get(path)
    if pinned is not None:
        if sha != pinned:
            return [("published-modified", "differs from the published file pinned in scripts/firewall_check.py")]
        return []
    if size > MAX_BLOB_BYTES:
        return [("large", "{:,} bytes, over the {} MiB limit".format(size, MAX_BLOB_BYTES // 2 ** 20))]
    lower = path.lower()
    if lower.endswith((".csv", ".tsv")):
        return csv_problems(read())
    if lower.endswith(".ipynb"):
        return notebook_problems(read())
    return []


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

    def file_at(self, rev: str, path: str) -> Optional[bytes]:
        """A file's content in the index (rev "") or in a commit, or None if absent."""
        spec = ":{}".format(path) if rev == "" else "{}:{}".format(rev, path)
        return self.run("cat-file", "blob", spec) if self.exists(spec) else None


def _split_z(raw: bytes) -> List[str]:
    return [t.decode("utf-8", "surrogateescape") for t in raw.split(b"\x00") if t]


# -- checks ----------------------------------------------------------------------------------------

def check_entries(git: Git, entries: Iterable[Tuple[str, str, str]]) -> List[Problem]:
    """entries: (path, blob, where). Path rules first; content is read only for paths that passed."""
    entries = list({(p, s, w) for p, s, w in entries})
    kinds = git.kinds(s for _, s, _ in entries)
    problems = []
    for path, sha, where in sorted(entries):
        found = path_problems(path)
        if not found:
            kind, size = kinds.get(sha, ("blob", 0))
            if kind != "blob":
                continue
            found = blob_problems(path, sha, size, lambda sha=sha: git.blob(sha))
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
    print("  Nothing above prints file contents. Rules: docs/ci.md, scripts/firewall_check.py.", file=out)
    return 1


def main(argv: Optional[Sequence[str]] = None, stdin=None, out=None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
