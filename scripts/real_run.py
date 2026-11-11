#!/usr/bin/env python3
r"""The real run in one command: checks, training, labels, validation, public-window check, run report.

A team member runs this on the provided match, never an AI assistant (CLAUDE.md). On the team's Windows PC,
from the repository root, without make (docs/real-run.md):

    .venv\Scripts\python.exe scripts\real_run.py --team <team_name>
    .venv\Scripts\python.exe scripts\real_run.py --team <team_name> --work D:\katena\work-submit --external D:\katena\external_data

It runs these steps in order and stops at the first failure:

1. Environment: MTAH_DATA_DIR and the bundle's files (existence only, through games.asset_paths; nothing is
   opened), KATENA_WORK and KATENA_EXTERNAL (or --work and --external), the git commit and whether the tree
   is clean, Python and the library versions against pyproject.toml.
2. The human checks: check_target.py, check_target_layout.py, check_touch_target.py, then
   check_kickoff_tags.py when this checkout has it. --skip-checks skips this step.
3. make_submission.py all --workers 2 --out submissions/<team>.csv, its output streamed, then the trained
   model's hashes against docs/repro/expected-windows.json. A different gbdt hash stops the run (the training
   data changed), before predict when it can. A different tcn or bundle hash alone is a warning (torch's CPU
   kernels changed).
4. evaluation/validate_submission.py on the exact file, check_public_window.py, and the file's SHA-256.
5. submissions/<team>.run.json: commit, times, durations, PASS/FAIL results, model hashes, the CSV's SHA-256.
6. A decision-log snippet to paste: the date, the commit, the PASS/FAIL lines and the hashes.

Output policy (CLAUDE.md). Of the checks, the validator and the public-window check this prints only the
PASS/FAIL lines, "Valid", the touch features' shape and exception type names, never their other lines.
make_submission.py's own output (aggregates only, by its contract) is streamed as it comes, except that a
traceback is reduced to its exception type, because exception messages can quote a file. The run report and
the snippet hold no label share and nothing per frame. Outside the team, share the snippet only.

Exit status: 0 every step passed (warnings allowed), 1 a step failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"
EXPECTED = REPO / "docs" / "repro" / "expected-windows.json"
TOOLS = {
    "check_target": REPO / "scripts" / "check_target.py",
    "check_target_layout": REPO / "scripts" / "check_target_layout.py",
    "check_touch_target": REPO / "scripts" / "check_touch_target.py",
    "check_kickoff_tags": REPO / "scripts" / "check_kickoff_tags.py",
    "make_submission": REPO / "scripts" / "make_submission.py",
    "validate_submission": REPO / "evaluation" / "validate_submission.py",
    "check_public_window": REPO / "scripts" / "check_public_window.py",
}
CHECKS = ("check_target", "check_target_layout", "check_touch_target", "check_kickoff_tags")
OPTIONAL = {"check_kickoff_tags"}          # W4a adds it; a checkout without it skips it
WORKERS = 2
# katena.pipeline's names (tests/test_real_run.py checks they agree); imported, it would pull in torch.
MODEL_DIR, MANIFEST, BUNDLE = "model", "manifest.json", "bundle.pkl"
HASH_KEYS = ("gbdt", "tcn", BUNDLE)
N_EXTERNAL = 30                            # external matches F trains on
# The bundle files the run reads: the tactical tracking and tags (predict and check_target), and the
# broadcast metadata (check_target_layout only). Existence only: nothing here opens them.
NEEDED = {"tactical": ("atd_meta", "atd_raw", "smart_tagging")}
NEEDED_FOR_CHECKS = {"broadcast": ("atd_meta",)}
SHARE_NOTE = 'Shareable outside the team: the PASS/FAIL lines, "Valid" and the hashes (the snippet at the end).'


def say(line: str = "") -> None:
    print(line, flush=True)


class Stop(Exception):
    """A step failed and the run stops. The message is this script's own text, never a child's output."""


# ---------------------------------------------------------------------------------------------
# reading a child's output without passing on anything it should not

class TracebackFilter:
    """Drops Python tracebacks from a child's output and keeps only the exception type, as check_target.py
    does: an exception message can quote the file it was reading. Once a traceback starts, every line is
    held back until the child exits or logs a normal ``[HH:MM:SS]`` line again (it carried on)."""

    MARK = "Traceback (most recent call last):"
    RESUME = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")
    NAME = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)(?::(?:\s.*)?)?$")
    SUFFIXES = ("Error", "Exception", "Exit", "Interrupt", "Warning", "Traceback", "Failure", "Iteration")

    def __init__(self):
        self.active = False
        self.raised: list[str] = []
        self._last = None

    @classmethod
    def exception_name(cls, line: str) -> str | None:
        """The class name if ``line`` is a traceback's final 'Type: message' line, else None."""
        m = cls.NAME.match(line)
        if not m:
            return None
        name = m.group(1)
        known = isinstance(getattr(builtins, name, None), type) and issubclass(getattr(builtins, name), BaseException)
        return name if known or name.endswith(cls.SUFFIXES) else None

    def feed(self, line: str) -> list[str]:
        """The lines to pass on for one line of output."""
        out = []
        if self.active and self.RESUME.match(line):
            out.append(self._summary())
        if self.MARK in line:
            if not self.active:
                self.active, self._last = True, None
        if not self.active:
            return out + [line]
        if line[:1].strip():                          # no indent: a type line, a separator or message text
            self._last = self.exception_name(line) or self._last
        return out

    def close(self) -> list[str]:
        return [self._summary()] if self.active else []

    def _summary(self) -> str:
        self.active = False
        self.raised.append(self._last or "an exception")
        return f"raised {self._last or 'an exception'} (traceback held back: its message can quote a file)"


def run_captured(cmd: list[str], env: dict) -> tuple[int, list[str], list[str]]:
    """Exit status, the output lines with tracebacks reduced to their type, and the types raised."""
    p = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    f = TracebackFilter()
    shown = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        shown += f.feed(line)
    shown += f.close()
    return p.returncode, shown, f.raised


VERDICT = re.compile(r"^\s*(PASS|FAIL|WARN)\b\s*(.*?)\s*$")                  # "PASS  what" (check_target.py)
VERDICT_AFTER = re.compile(r"^\s*(\w[\w .()/-]*?):\s+(PASS|FAIL)\s*$")    # "what: PASS" (check_public_window.py)
EXCEPTION_LINE = re.compile(r"\braised [A-Za-z_][\w.]*|\b(?:failed|skipped) \([A-Za-z_][\w.]*[;)]")
TOUCH_SHAPE = re.compile(r"^target \(real 25 Hz ball\)\s+rows\s+(\d+)\s+cols\s+(\d+)\b")
SECTIONS = ("tactical", "broadcast")      # check_target_layout.py's headers (games.SOURCES)


def verdict_lines(key: str, rc: int, shown: list[str]) -> list[str]:
    """A check's PASS/FAIL lines and exception lines, nothing else. A check passes when it exits 0, prints at
    least one PASS line and no FAIL or exception line. check_touch_target.py prints no PASS line: it passes
    when it finishes and prints the real match's feature shape."""
    out, section = [], ""
    for line in shown:
        s = line.strip()
        if s in SECTIONS and line == s:
            section = f"[{s}] "
            continue
        m, m_after = VERDICT.match(line), VERDICT_AFTER.match(line)
        if m:
            out.append(f"{m.group(1)}  {section}{m.group(2)}".rstrip())
        elif m_after:
            out.append(f"{m_after.group(2)}  {section}{m_after.group(1)}")
        elif key == "check_touch_target" and s.startswith("reference") and EXCEPTION_LINE.search(line):
            # The reference worlds come from an external match, not the provided one: an alarm, never a stop.
            out.append(f"WARN  {section}{s}")
        elif s.startswith("raised ") or EXCEPTION_LINE.search(line):
            out.append(f"FAIL  {section}{s}")
        elif key == "check_touch_target" and TOUCH_SHAPE.match(line):
            r, c = TOUCH_SHAPE.match(line).groups()
            out.append(f"PASS  touch features computed on the real match, shape ({int(r)}, {int(c)})")
    fails = [x for x in out if x.startswith("FAIL")]
    if rc != 0 and not fails:
        out.append(f"FAIL  exited with status {rc}")
    elif not out:
        out.append("FAIL  printed no PASS line")
    return out


# ---------------------------------------------------------------------------------------------
# the run

class Run:
    def __init__(self, team: str, skip_checks: bool):
        self.t0 = time.time()
        self.csv = SUBMISSIONS / f"{team}.csv"
        self.csv_written = False
        self.env: dict = {}
        self.work: Path | None = None
        self.current = "arguments"
        self.report = {
            "schema": 1, "script": "scripts/real_run.py", "team": team, "result": None, "failed_step": None,
            "stop_reason": None, "started": now(), "finished": None, "seconds": {},
            "commit": None, "branch": None, "tree_clean": None, "skip_checks": skip_checks, "workers": WORKERS,
            "environment": {}, "checks": [], "make_submission": None, "model_sha256": {}, "expected_sha256": {},
            "hash_check": {}, "validate_submission": None, "public_window": None,
            "csv": {"file": self.csv.name, "sha256": None, "bytes": None}, "warnings": [],
        }

    @contextmanager
    def step(self, name: str, title: str | None = None):
        """Times one step under ``seconds``; a failure inside it is reported as this step."""
        self.current = name
        if title:
            say(f"\n== {title}")
        t = time.time()
        try:
            yield
        finally:
            self.report["seconds"][name] = round(time.time() - t, 1)

    def warn(self, text: str) -> None:
        say(f"WARN  {text}")
        self.report["warnings"].append(text)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)
    except OSError:
        return -1, ""
    return p.returncode, p.stdout.decode("utf-8", "replace").strip()


def git_info() -> dict:
    """The commit, branch, whether the tree is clean (and how many paths are not), and whether HEAD is on
    origin/main as last fetched. Paths are counted, not printed."""
    rc, commit = git("rev-parse", "HEAD")
    if rc != 0:
        return {"commit": None}
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    rc_s, status = git("status", "--porcelain")
    on_main, _ = git("merge-base", "--is-ancestor", "HEAD", "origin/main")
    return {"commit": commit, "branch": branch, "dirty": len(status.splitlines()) if rc_s == 0 else None,
            "on_main": {0: True, 1: False}.get(on_main)}


def pinned_versions() -> dict:
    """{distribution: pinned version} from pyproject.toml's dependencies."""
    import tomllib
    with open(REPO / "pyproject.toml", "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    return dict(d.replace(" ", "").split("==", 1) for d in deps if "==" in d)


def environment(run: Run, a) -> None:
    """Step 1. Collects every problem before stopping, so one run names all of them."""
    problems = []
    env = run.report["environment"]

    data = os.environ.get("MTAH_DATA_DIR")
    if not data:
        problems.append("MTAH_DATA_DIR is not set")
        say('FAIL  MTAH_DATA_DIR is not set ($env:MTAH_DATA_DIR = "D:\\mtah2027")')
    elif not Path(data).is_dir():
        problems.append("MTAH_DATA_DIR is not a folder")
        say(f"FAIL  MTAH_DATA_DIR is not a folder: {data}")
    else:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import games
        needed = dict(NEEDED, **({} if a.skip_checks else NEEDED_FOR_CHECKS))
        found, missing, extra = [], [], []
        for source in games.SOURCES:
            for asset, path in games.asset_paths(data_dir=data, source=source).items():
                name = f"{source} {asset}"
                if asset in needed.get(source, ()):
                    (found if os.path.isfile(path) else missing).append(name)
                elif os.path.isfile(path):
                    extra.append(name)
        env["data_dir"] = data
        env["bundle"] = {"needed_found": found, "needed_missing": missing}
        say(f"PASS  MTAH_DATA_DIR is set: {data}")
        if missing:
            problems.append("bundle files missing: " + ", ".join(missing))
            say("FAIL  bundle files missing: " + ", ".join(missing))
        if found:
            say("PASS  bundle files present: " + ", ".join(found))
        if extra:
            say("INFO  also present, not read by the run: " + ", ".join(extra))

    for var, flag, label in (("KATENA_WORK", a.work, "--work"), ("KATENA_EXTERNAL", a.external, "--external")):
        value = str(flag) if flag else os.environ.get(var)
        if not value:
            problems.append(f"{var} is not set")
            say(f"FAIL  {var} is not set (set it, or pass {label} DIR)")
            continue
        path = Path(value).resolve()
        env[var] = str(path)
        if not path.is_dir():
            problems.append(f"{var} does not exist")
            say(f"FAIL  {var} does not exist: {path}")
            continue
        say(f"PASS  {var} exists: {path}{' (from ' + label + ')' if flag else ''}")
        run.env[var] = str(path)
        if var == "KATENA_EXTERNAL":
            n = len(list((path / "canonical").glob("*/*.npz")))
            env["external_matches"] = n
            if n == 0:
                problems.append("no external matches under KATENA_EXTERNAL")
                say("FAIL  no external matches under canonical/ (run scripts/build_external.py)")
            elif n != N_EXTERNAL:
                run.warn(f"{n} external matches under canonical/, F trains on {N_EXTERNAL}: expect another gbdt hash")
            else:
                say(f"PASS  {n} external matches under canonical/")
    if run.env.get("KATENA_WORK"):
        run.work = Path(run.env["KATENA_WORK"])

    g = git_info()
    run.report.update(commit=g["commit"], branch=g.get("branch"),
                      tree_clean=None if g.get("dirty") is None else g["dirty"] == 0)
    if g["commit"] is None:
        problems.append("no git commit")
        say("FAIL  no git commit: run from a clone of the repository, with git on PATH")
    else:
        say(f"PASS  commit {g['commit']} on {g['branch']}")
        if g["dirty"]:
            run.warn(f"the tree has {g['dirty']} uncommitted or untracked paths (git status): this commit alone "
                     "does not reproduce the run")
        elif g["dirty"] == 0:
            say("PASS  tree clean")
        if g["on_main"] is False:
            run.warn("HEAD is not on origin/main (as last fetched): the release workflow refuses a tag off main")

    py = platform.python_version()
    env.update(python=py, os=platform.system(), machine=platform.machine(), libraries={})
    say(f"INFO  python {py} on {platform.system()} {platform.machine()}: {sys.executable}")
    if not py.startswith("3.11."):
        run.warn(f"python {py}, pyproject.toml pins 3.11")
    try:
        pins = pinned_versions()
    except (OSError, KeyError, ValueError) as e:
        pins = {}
        run.warn(f"could not read the pins in pyproject.toml ({type(e).__name__})")
    off = []
    for dist, want in sorted(pins.items()):
        try:
            got = metadata.version(dist)
        except metadata.PackageNotFoundError:
            problems.append(f"{dist} is not installed")
            say(f"FAIL  {dist} is not installed (pyproject.toml pins {want})")
            continue
        env["libraries"][dist] = got
        if got != want:
            off.append(f"{dist} {got} (pinned {want})")
    if pins and not off:
        say("PASS  libraries as pinned: " + ", ".join(f"{d} {v}" for d, v in env["libraries"].items()))
    elif off:
        run.warn("libraries other than the pins: " + ", ".join(off) + "; expect other model hashes")

    run.env = {**os.environ, **run.env, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if problems:
        raise Stop("; ".join(problems))


def checks(run: Run) -> None:
    """Step 2: each human check in order, its PASS/FAIL lines only."""
    for key in CHECKS:
        path = TOOLS[key]
        if not path.is_file():
            if key in OPTIONAL:
                say(f"SKIP  {path.name}: not in this checkout")
                run.report["checks"].append({"script": path.name, "result": "SKIP", "lines": []})
                continue
            raise Stop(f"{path.name} is missing")
        t = time.time()
        rc, shown, _ = run_captured([sys.executable, str(path)], run.env)
        lines = verdict_lines(key, rc, shown)
        n_pass = sum(x.startswith("PASS") for x in lines)
        n_fail = sum(x.startswith("FAIL") for x in lines)
        ok = n_fail == 0 and n_pass > 0 and rc == 0
        secs = round(time.time() - t, 1)
        say(path.name)
        for x in lines:
            say(f"  {x}")
        say(f"  {n_pass} PASS, {n_fail} FAIL ({secs:.0f} s)")
        run.report["checks"].append({"script": path.name, "result": "PASS" if ok else "FAIL", "pass": n_pass,
                                     "fail": n_fail, "exit_status": rc, "seconds": secs, "lines": lines})
        run.report["warnings"] += [f"{path.name}: {x[6:]}" for x in lines if x.startswith("WARN")]
        if not ok:
            first = next((x for x in lines if x.startswith("FAIL")), "FAIL")
            raise Stop(f"{path.name}: {first}")


class MakeWatch:
    """Reads make_submission.py's stream for two things: the gbdt hash on the ``train`` line, to stop before
    predict when it differs, and the hashes in predict's report. Nothing else of the report is kept."""

    TRAIN = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] train (\{.*\})\s*$")

    def __init__(self, expected_gbdt: str | None):
        self.expected_gbdt = expected_gbdt
        self.train_gbdt = None
        self.predict: dict | None = None
        self._buf: list[str] | None = None

    def feed(self, line: str) -> bool:
        """False: stop the child now."""
        m = self.TRAIN.match(line)
        if m:
            try:
                self.train_gbdt = json.loads(m.group(1)).get("sha256", {}).get("gbdt")
            except (ValueError, AttributeError):
                self.train_gbdt = None
            return not (self.train_gbdt and self.expected_gbdt and self.train_gbdt != self.expected_gbdt)
        if line == "{":
            self._buf = [line]
        elif self._buf is not None:
            self._buf.append(line)
            if line == "}":
                text, self._buf = "\n".join(self._buf), None
                try:
                    rep = json.loads(text)
                except ValueError:
                    return True
                sub = rep.get("submission") if isinstance(rep, dict) else None
                if isinstance(sub, dict):
                    self.predict = {"csv_sha256": sub.get("sha256"), "model_sha256": rep.get("model_sha256")}
        return True


def stream(cmd: list[str], env: dict, watch: MakeWatch) -> tuple[int, bool]:
    """Run ``cmd``, passing its output on as it comes (tracebacks reduced to their type). Returns the exit
    status and whether ``watch`` stopped it."""
    filt = TracebackFilter()
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stopped = False
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            for out in filt.feed(line):
                say(f"  {out}")
            if not watch.feed(line):
                stopped = True
                proc.terminate()
                break
        for out in filt.close():
            say(f"  {out}")
    except BaseException:
        proc.kill()
        raise
    finally:
        rc = proc.wait()
        proc.stdout.close()
    return rc, stopped


def read_expected() -> dict:
    with open(EXPECTED, encoding="utf-8") as fh:
        return json.load(fh)["sha256"]


def expected_spec() -> str:
    """The model spec the expected hashes belong to (``katena.pipeline.SUBMISSION_SPEC``; a test checks they agree)."""
    with open(EXPECTED, encoding="utf-8") as fh:
        return json.load(fh).get("spec", "F")


def model_folder(spec: str) -> str:
    """``katena.pipeline.model_dir``'s folder name for a spec: ``model`` for F, ``model_<spec>`` otherwise."""
    return MODEL_DIR if spec == "F" else f"{MODEL_DIR}_{spec}"


def submission(run: Run) -> None:
    """Step 3: train and label, then the model hashes."""
    expected = read_expected()
    spec = expected_spec()
    run.report["expected_sha256"] = expected
    run.report["model_spec"] = spec
    run.csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TOOLS["make_submission"]), "all", "--workers", str(WORKERS), "--out", str(run.csv),
           "--spec", spec]
    say(f"{TOOLS['make_submission'].name} all --workers {WORKERS} --out {run.csv} --spec {spec}  (streamed)")
    t = time.time()
    watch = MakeWatch(expected.get("gbdt"))
    rc, stopped = stream(cmd, run.env, watch)
    run.report["make_submission"] = {"exit_status": rc, "stopped_after_train": stopped,
                                     "result": "PASS" if rc == 0 and not stopped else "FAIL"}
    if stopped:
        run.report["model_sha256"] = {"gbdt": watch.train_gbdt}
        run.report["hash_check"] = {"gbdt": "differs"}
        say(f"FAIL  gbdt {watch.train_gbdt}, expected {expected.get('gbdt')}: the training data changed. "
            "Stopped make_submission.py before predict. Report it")
        raise Stop("gbdt hash differs from docs/repro/expected-windows.json: the training data changed")
    if rc != 0:
        raise Stop(f"make_submission.py exited with status {rc}")

    manifest = run.work / model_folder(spec) / MANIFEST
    if not manifest.is_file() or manifest.stat().st_mtime < t - 2:
        raise Stop(f"{model_folder(spec)}/{MANIFEST} was not written by this run")
    with open(manifest, encoding="utf-8") as fh:
        got = json.load(fh).get("sha256", {})
    run.report["model_sha256"] = {k: got.get(k) for k in HASH_KEYS}
    say("model hashes against docs/repro/expected-windows.json")
    kernels = []
    for key in HASH_KEYS:
        same = got.get(key) is not None and got.get(key) == expected.get(key)
        run.report["hash_check"][key] = "same" if same else "differs"
        if same:
            say(f"  PASS  {key:10s} {got.get(key)}")
        else:
            say(f"  {'FAIL' if key == 'gbdt' else 'WARN'}  {key:10s} {got.get(key)}, expected {expected.get(key)}")
            kernels += [] if key == "gbdt" else [key]
    if run.report["hash_check"]["gbdt"] != "same":
        raise Stop("gbdt hash differs from docs/repro/expected-windows.json: the training data changed")
    if kernels:
        run.warn(f"{' and '.join(kernels)} hash differs: torch's CPU kernels changed (a new torch or CPU). "
                 "Report it; it is not by itself a bug")

    if not run.csv.is_file() or run.csv.stat().st_mtime < t - 2:
        raise Stop(f"{run.csv.name} was not written by this run")
    run.csv_written = True
    if watch.predict is None:
        run.warn("could not read predict's report: the bundle and CSV cross-checks are skipped")
    elif watch.predict["model_sha256"] != got.get(BUNDLE):
        raise Stop("predict used another bundle.pkl than the one train wrote")


def validate(run: Run) -> None:
    """Step 4a: the official validator on the exact file. Its "OK" and "Valid" lines, or its FAIL line and
    reason; never its label counts."""
    rc, shown, raised = run_captured([sys.executable, str(TOOLS["validate_submission"]), str(run.csv)], run.env)
    lines = []
    for i, line in enumerate(shown):
        if line.startswith(("OK ", "Valid.")):
            lines.append(line.rstrip())
        elif line.startswith("FAIL"):
            lines.append(line.rstrip())
            reason = shown[i + 1].strip() if i + 1 < len(shown) else ""
            if reason and not reason.startswith("could not read"):
                lines.append(f"      {reason}")
    lines += [f"FAIL  raised {r}" for r in raised]
    ok = rc == 0 and any(x.startswith("Valid.") for x in lines)
    for x in lines or ["FAIL  printed nothing the run recognises"]:
        say(f"  {x}")
    run.report["validate_submission"] = {"result": "PASS" if ok else "FAIL", "exit_status": rc, "lines": lines}
    if not ok:
        raise Stop("validate_submission.py does not say Valid")


PUBLIC = re.compile(r"^(\w+): (PASS|FAIL)\s*$")


def public_window(run: Run) -> None:
    """Step 4b: the pipeline check on the public window, PASS/FAIL lines only."""
    rc, shown, raised = run_captured([sys.executable, str(TOOLS["check_public_window"]), str(run.csv)], run.env)
    lines = [f"{m.group(1)}: {m.group(2)}" for m in map(PUBLIC.match, shown) if m]
    lines += [f"FAIL  raised {r}" for r in raised]
    ok = rc == 0 and bool(lines) and not any("FAIL" in x for x in lines)
    if rc != 0 and not any("FAIL" in x for x in lines):
        lines.append(f"FAIL  exited with status {rc}")
    for x in lines or ["FAIL  printed no PASS line"]:
        say(f"  {x}")
    run.report["public_window"] = {"result": "PASS" if ok else "FAIL", "exit_status": rc, "lines": lines}
    if not ok:
        raise Stop("check_public_window.py: a FAIL is a bug to find, never something to tune")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(run: Run) -> str:
    sha = sha256_file(run.csv)
    run.report["csv"].update(sha256=sha, bytes=run.csv.stat().st_size)
    return sha


# ---------------------------------------------------------------------------------------------
# report and snippet

def snippet(r: dict) -> str:
    """Markdown for docs/decision-log.md, built from the report alone: the date, the commit, PASS/FAIL lines
    and hashes."""
    date = (r["started"] or "")[:10]
    tree = {True: "tree clean", False: "tree NOT clean", None: "tree state unknown"}[r["tree_clean"]]
    out = [f"### Real run: `scripts/real_run.py --team {r['team']}` ({date})", "",
           f"- Commit `{r['commit']}` ({tree}). Result: **{r['result']}**"
           + (f", stopped at {r['failed_step']}: {r['stop_reason']}." if r["result"] != "PASS" else ".")]
    if r["skip_checks"]:
        out.append("- Checks on the provided files: skipped (`--skip-checks`).")
    elif r["checks"]:
        parts = []
        for c in r["checks"]:
            if c["result"] == "SKIP":
                parts.append(f"`{c['script']}` not in this checkout")
            else:
                parts.append(f"`{c['script']}` {c['pass']} PASS, {c['fail']} FAIL")
        out.append("- Checks on the provided files: " + "; ".join(parts) + ".")
        out += [f"  - `{c['script']}`: {x}" for c in r["checks"] for x in c.get("lines", []) if x.startswith("FAIL")]
    if r["model_sha256"]:
        out.append("- Model, against `docs/repro/expected-windows.json`: "
                   + "; ".join(f"{k} `{v}` ({r['hash_check'].get(k, '?')})" for k, v in r["model_sha256"].items())
                   + ".")
    if r["validate_submission"]:
        v = r["validate_submission"]
        out.append(f"- `validate_submission.py`: {'Valid' if v['result'] == 'PASS' else 'FAIL'}.")
    if r["public_window"]:
        out.append("- `check_public_window.py`: " + ", ".join(r["public_window"]["lines"]) + ".")
    if r["csv"]["sha256"]:
        out.append(f"- `submissions/{r['csv']['file']}` SHA-256 `{r['csv']['sha256']}`.")
    return "\n".join(out)


def write_report(run: Run) -> Path:
    path = SUBMISSIONS / f"{run.report['team']}.run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(run.report, indent=1, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------------------------

BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def team_name(value: str) -> str:
    if not value or BAD_NAME.search(value) or value != value.strip() or value.endswith("."):
        raise argparse.ArgumentTypeError("the team name must be a plain file name (it names the CSV)")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True, type=team_name, help="the registered team name; names the CSV")
    ap.add_argument("--skip-checks", action="store_true", help="skip the human checks (step 2)")
    ap.add_argument("--work", type=Path, default=None, help="the cache (default: $KATENA_WORK)")
    ap.add_argument("--external", type=Path, default=None, help="the external corpus (default: $KATENA_EXTERNAL)")
    a = ap.parse_args(argv)

    run = Run(a.team, a.skip_checks)
    say(f"real_run.py: team {a.team}, {run.report['started']}")
    say("Only a team member runs this on the provided match. " + SHARE_NOTE)
    try:
        with run.step("environment", "1. Environment"):
            environment(run, a)
        if a.skip_checks:
            say("\n== 2. Checks on the provided files: skipped (--skip-checks)")
        else:
            with run.step("checks", "2. Checks on the provided files"):
                checks(run)
        with run.step("make_submission", "3. Train and label: make_submission.py all (its aggregates are for "
                                         "the team only)"):
            submission(run)
        say("\n== 4. Validate the exact file, check the public window, hash it")
        with run.step("validate_submission"):
            say(TOOLS["validate_submission"].name)
            validate(run)
        with run.step("public_window"):
            say(f"{TOOLS['check_public_window'].name} (a pipeline check: a FAIL is a bug, never something to tune)")
            public_window(run)
        with run.step("sha256"):
            say(f"SHA-256  {digest(run)}  {run.csv}")
        run.report["result"] = "PASS"
    except Stop as s:
        run.report.update(result="FAIL", failed_step=run.current, stop_reason=str(s))
    except KeyboardInterrupt:
        run.report.update(result="FAIL", failed_step=run.current, stop_reason="interrupted")
    except Exception as e:  # noqa: BLE001  this script's own failure: the type alone
        run.report.update(result="FAIL", failed_step=run.current, stop_reason=f"real_run.py raised {type(e).__name__}")

    if run.csv_written and run.report["csv"]["sha256"] is None:      # a look at the public window is logged
        say(f"SHA-256  {digest(run)}  {run.csv}")                     # with the hash of the file it looked at
    run.report["finished"] = now()
    run.report["seconds"]["total"] = round(time.time() - run.t0, 1)
    say("\n== 5. Run report")
    for w in run.report["warnings"]:
        say(f"WARN  {w}")
    if run.report["result"] == "PASS":
        say(f"PASS  every step passed in {run.report['seconds']['total'] / 60:.1f} min")
    else:
        say(f"FAIL  stopped at {run.report['failed_step']}: {run.report['stop_reason']}")
        say("      Report the FAIL line (it is safe to paste). Describe anything else in words; paste no output.")
    try:
        say(f"Written (commit, times, PASS/FAIL, hashes; no label shares): {write_report(run)}")
    except OSError as e:
        say(f"FAIL  could not write the run report ({type(e).__name__})")
        run.report["result"] = "FAIL"
    say("\n== 6. Decision-log snippet: paste into docs/decision-log.md under today's date. Safe to share.")
    say(snippet(run.report))
    say("== end of snippet")
    return 0 if run.report["result"] == "PASS" else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
