#!/usr/bin/env python
"""Cross-OS reproduction of the submission model (J3): time the stages, collect hashes, compare machines.

    python scripts/repro_ci.py time --out T.json --name corpus -- python scripts/make_submission.py corpus
    python scripts/repro_ci.py time --out T.json --name uv_sync --seconds 95
    python scripts/repro_ci.py env --out env.json
    python scripts/repro_ci.py report --label ubuntu-24.04 --dressed DIR --csv DIR/run/dressed.csv \
        --digests digests.json --manifest WORK/model/manifest.json --predict-log predict.log \
        --timings T.json --env env.json --out repro-ubuntu-24.04.json
    python scripts/repro_ci.py compare --reference docs/repro/windows-reference.json \
        --summary summary.md --out comparison.json repro-*.json

Used by ``.github/workflows/repro.yml`` (docs/ci.md). External matches only. A report holds hashes, versions,
CPU facts, timings and aggregates of the dressed external match: label counts, its score against the match's
own labels, and one hash per block of 1,000 frames. It never holds a CSV or a frame-level value. ``compare``
needs the standard library only; it never fails, it reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cache_digests import STAGES  # noqa: E402  (stdlib only at import time)

BLOCK = 1000                      # frames per block hash of the dressed labels
LABELS = "HADX"
# Model parts and dressed-match files, in pipeline order after the cache stages.
HASH_ROWS = (("gbdt", "LightGBM member"), ("part_gbdt_alive", "  alive head"), ("part_gbdt_team", "  team head"),
             ("part_names", "feature names"), ("part_scaler", "TCN scaler"),
             ("part_priors_viterbi", "Viterbi priors"), ("part_priors_hsmm", "HSMM priors"),
             ("tcn", "TCN weights"), ("bundle.pkl", "model bundle"),
             ("dressed_raw", "dressed ATD raw"), ("dressed_meta", "dressed ATD metadata"),
             ("dressed_timeline", "dressed timeline"), ("dressed_gt_T2", "dressed match's own labels"),
             ("dressed_csv", "dressed CSV, all-30 model"))
ORDER = tuple(STAGES) + tuple(k for k, _ in HASH_ROWS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------------
# time

def cmd_time(a) -> int:
    rec = _load(a.out) if Path(a.out).exists() else {}
    if a.seconds is not None:
        rec[a.name] = {"wall_s": round(a.seconds, 1)}
        _dump(rec, a.out)
        return 0
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not cmd:
        sys.exit("time: no command after --")
    t0 = time.time()
    rc = subprocess.call(cmd)
    wall = time.time() - t0
    peak = None
    if os.name != "nt":          # largest single descendant; ru_maxrss is KiB on Linux, bytes on macOS
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak = round(r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20, 2)
    rec[a.name] = {"wall_s": round(wall, 1), "exit": rc, "peak_rss_gb": peak}
    _dump(rec, a.out)
    print(f"[repro] {a.name}: {wall / 60:.1f} min, exit {rc}"
          + (f", largest process {peak} GB" if peak is not None else ""), flush=True)
    return rc


# ---------------------------------------------------------------------------------------------
# env

def _run(cmd) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        return None
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            return None
    return None


def _ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 2 ** 30, 1)
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemoryStatus()
            st.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 2 ** 30, 1)
    except Exception:
        return None
    return None


def env_facts() -> dict:
    """Versions and CPU facts, with the CPU environment the pipeline pins (set before torch is imported)."""
    from importlib import metadata
    sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]
    from katena.models import ensemble as E
    E.pin_cpu_env()
    from katena.models import gbdt
    gbdt.ensure_openmp()
    out = {"os": platform.system(), "release": platform.release(), "platform": platform.platform(),
           "machine": platform.machine(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
           "cpu_model": _cpu_model(), "ram_gb": _ram_gb(),
           "pinned_env": {k: os.environ.get(k) for k in sorted(E.CPU_ENV)},
           "runner": {k: os.environ.get(k) for k in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")},
           "uv": _run(["uv", "--version"]), "uv_lock_sha256": sha256_file(REPO / "uv.lock"),
           "commit": os.environ.get("GITHUB_SHA") or _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])}
    out["packages"] = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "torch"):
        try:
            out["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out["packages"][name] = None
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:
        feats = {}
    out["numpy_cpu_features"] = sorted(k for k, v in feats.items() if v)
    import torch
    t = {"version": torch.__version__, "num_threads": torch.get_num_threads(),
         "mkl": torch.backends.mkl.is_available(), "mkldnn": torch.backends.mkldnn.is_available()}
    try:
        t["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:
        t["cpu_capability"] = None
    t["config"] = [ln.strip() for ln in torch.__config__.show().splitlines() if ln.strip()]
    out["torch"] = t
    import lightgbm
    out["lightgbm_loads"] = lightgbm.__version__
    return out


def cmd_env(a) -> int:
    facts = env_facts()
    _dump(facts, a.out)
    print(f"[repro] {facts['os']} {facts['machine']}, {facts['cpu_model']}, {facts['cpu_count']} CPUs, "
          f"{facts['ram_gb']} GB; torch {facts['torch']['version']} ({facts['torch']['cpu_capability']}), "
          f"lightgbm {facts['packages']['lightgbm']}, numpy {facts['packages']['numpy']}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# report

def _predict_log(path: Path) -> tuple[dict | None, dict | None]:
    """The pipeline's env line and the aggregate report ``make_submission.py predict`` prints (indent=1)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    env = None
    for ln in lines:
        if "; env {" in ln:
            env = json.loads(ln.split("; env ", 1)[1])
    try:
        s = lines.index("{")
        rep = json.loads("\n".join(lines[s:lines.index("}", s) + 1]))
    except ValueError:
        rep = None
    if rep is not None:
        rep.get("submission", {}).pop("path", None)
    return env, rep


def label_summary(csv: Path, timeline: dict, gt_csv: Path | None) -> dict:
    """Aggregates of a dressed CSV: label counts in the graded halves, number of runs, one score, block hashes."""
    import numpy as np
    sys.path.insert(0, str(REPO / "evaluation"))
    import possession_metrics as pm
    codes = pm.label_codes(pm.read_label_csv(str(csv)))
    graded = np.zeros(len(codes), bool)
    for _, lo, hi in timeline["half_blocks"]:
        graded[lo:hi + 1] = True
    runs = sum(1 + int(np.count_nonzero(np.diff(codes[lo:hi + 1].astype(np.int16))))
               for _, lo, hi in timeline["half_blocks"])
    out = {"rows": int(len(codes)), "graded": int(graded.sum()),
           "counts_graded": {k: int((codes[graded] == i).sum()) for i, k in enumerate(LABELS)},
           "runs_graded": int(runs), "block_frames": BLOCK,
           "block_sha256": [hashlib.sha256(codes[i:i + BLOCK].tobytes()).hexdigest()[:16]
                            for i in range(0, len(codes), BLOCK)]}
    if gt_csv is not None and gt_csv.exists():
        gt = pm.label_codes(pm.read_label_csv(str(gt_csv)))
        r = pm.evaluate(gt, codes, bootstrap_n=0)
        out["score_vs_own_labels"] = {"spell_f1": round(float(r["spell_f1_mean"]), 6),
                                      "macro_f1": round(float(r["macro_f1"]), 6),
                                      "n_pred_spells": int(r["n_pred_spells"]), "n_gt_spells": int(r["n_gt_spells"])}
    return out


def _tree_gb(root: Path) -> float | None:
    if not root.exists():
        return None
    return round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 2 ** 30, 2)


def cmd_report(a) -> int:
    rep = {"schema": 1, "label": a.label, "kind": a.kind, "note": a.note, "missing": []}

    def opt(path):
        if path and Path(path).exists():
            return Path(path)
        rep["missing"].append(str(path))
        return None

    rep["env"] = _load(p) if (p := opt(a.env)) else None
    if a.commit:
        rep["env"] = {**(rep["env"] or {}), "commit": a.commit}
    rep["timings"] = _load(p) if (p := opt(a.timings)) else {}
    if os.environ.get("GITHUB_RUN_ID"):
        rep["run_url"] = (f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
                          f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    pipeline_env, predict = _predict_log(p) if (p := opt(a.predict_log)) else (None, None)
    rep["pipeline_env"], rep["predict"] = pipeline_env, predict
    man = _load(p) if (p := opt(a.manifest)) else None
    dig = _load(p) if (p := opt(a.digests)) else None
    rep["train"] = None if man is None else {k: (len(v) if k == "matches" else v)
                                             for k, v in man.items() if k != "sha256"}
    hashes = dict(man["sha256"]) if man else {}
    if dig:
        hashes.update(dig.get("models", {}).get("model", {}))
        rep["stages"] = dig["stages"]
    else:
        rep["stages"] = {}
    d = Path(a.dressed) if a.dressed else None
    for key, rel in (("dressed_raw", "bundle/tactical/BAR-ATH_tactical_atd.txt"),
                     ("dressed_meta", "bundle/tactical/BAR-ATH_tactical_atd.xml"),
                     ("dressed_timeline", "timeline.json"), ("dressed_gt_T2", "gt_T2.csv")):
        if d is not None and (p := opt(d / rel)):
            hashes[key] = sha256_file(p)
    rep["labels"] = None
    if (csv := opt(a.csv)):
        hashes["dressed_csv"] = sha256_file(csv)
        tl = d / "timeline.json" if d is not None else None
        if tl is not None and tl.exists():
            rep["labels"] = label_summary(csv, _load(tl), d / "gt_T2.csv")
    rep["hashes"] = hashes
    rep["sizes_gb"] = {"work": _tree_gb(Path(a.work)) if a.work else None,
                       "external": _tree_gb(Path(a.external)) if a.external else None}
    _dump(rep, a.out)
    print(f"[repro] {a.label}: wrote {a.out}")
    for k, name in HASH_ROWS:
        print(f"  {name:28s} {hashes.get(k, 'missing')}")
    print("  stages: " + ", ".join(f"{s} {len(rep['stages'].get(s, {}))}" for s in STAGES))
    if rep["missing"]:
        print(f"  missing inputs: {rep['missing']}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"### {a.label}\n\n| item | sha256 |\n|---|---|\n")
            fh.writelines(f"| {name.strip()} | `{hashes.get(k, 'missing')}` |\n" for k, name in HASH_ROWS)
            fh.write("\n| stage | wall (min) | largest process (GB) |\n|---|---|---|\n")
            fh.writelines(f"| {s} | {t['wall_s'] / 60:.1f} | {t.get('peak_rss_gb')} |\n"
                          for s, t in rep["timings"].items())
            fh.write("\n")
    return 0


# ---------------------------------------------------------------------------------------------
# compare

def compare_pair(x: dict, y: dict) -> dict:
    """Stage by stage and part by part: what agrees between two reports, and the earliest item that differs."""
    stages = {}
    for st in STAGES:
        sx, sy = x.get("stages", {}).get(st, {}), y.get("stages", {}).get(st, {})
        common = sorted(set(sx) & set(sy))
        bad = [k for k in common if sx[k] != sy[k]]
        stages[st] = {"common": len(common), "same": len(common) - len(bad), "only_one": len(set(sx) ^ set(sy)),
                      "first_differing": bad[:3]}
    hx, hy = x.get("hashes", {}), y.get("hashes", {})
    parts = {k: ("missing" if k not in hx or k not in hy else "same" if hx[k] == hy[k] else "differs")
             for k, _ in HASH_ROWS}
    status = {**{st: ("missing" if v["common"] == 0 else "differs" if v["same"] < v["common"] else "same")
                 for st, v in stages.items()}, **parts}
    earliest = next((k for k in ORDER if status[k] == "differs"), None)
    out = {"stages": stages, "parts": parts, "earliest_difference": earliest,
           "not_compared": [k for k in ORDER if status[k] == "missing"]}
    lx, ly = x.get("labels") or {}, y.get("labels") or {}
    if lx and ly:
        bx, by = lx.get("block_sha256", []), ly.get("block_sha256", [])
        out["labels"] = {
            "csv_identical": parts["dressed_csv"] == "same",
            "blocks_differing": (sum(a != b for a, b in zip(bx, by)) if len(bx) == len(by)
                                 and lx.get("block_frames") == ly.get("block_frames") else None),
            "blocks": len(bx), "block_frames": lx.get("block_frames"),
            "count_delta_graded": {k: ly.get("counts_graded", {}).get(k, 0) - lx.get("counts_graded", {}).get(k, 0)
                                   for k in LABELS},
            "runs_graded": [lx.get("runs_graded"), ly.get("runs_graded")],
            "score": [lx.get("score_vs_own_labels"), ly.get("score_vs_own_labels")]}
    return out


def _short(h):
    return f"`{h[:12]}`" if isinstance(h, str) else "missing"


def cmd_compare(a) -> int:
    ref = _load(a.reference)
    reports = []
    for p in a.reports:
        try:
            reports.append(_load(p))
        except (OSError, ValueError) as e:
            print(f"::warning::cannot read {p}: {e}")
    md = ["## Cross-OS reproduction (J3)\n", f"Reference: {ref.get('label')}: {ref.get('note') or ''}\n"]
    out = {"reference": ref.get("label"), "runs": [r.get("label") for r in reports], "vs_reference": {},
           "pairwise": {}}
    if not reports:
        md.append("No runner produced a report.\n")
    cols = [ref] + reports
    md.append("| item | " + " | ".join(str(c.get("label")) for c in cols) + " |")
    md.append("|---|" + "---|" * len(cols))
    for k, name in HASH_ROWS:
        rh = ref.get("hashes", {}).get(k)
        cells = [_short(rh)] + [_short(r.get("hashes", {}).get(k))
                                + ("" if r.get("hashes", {}).get(k) is None or rh is None
                                   else " same" if r["hashes"][k] == rh else " **differs**") for r in reports]
        md.append(f"| {name.strip()} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("| stage vs reference | " + " | ".join(str(r.get("label")) for r in reports) + " |")
    md.append("|---|" + "---|" * len(reports))
    cmp_ref = {r.get("label"): compare_pair(ref, r) for r in reports}
    out["vs_reference"] = cmp_ref
    for st in STAGES:
        cells = []
        for r in reports:
            s = cmp_ref[r.get("label")]["stages"][st]
            cells.append(f"{s['same']} of {s['common']} same" + (f", {s['only_one']} in one only" if s["only_one"] else "")
                         + (f"; first differing {', '.join(s['first_differing'])}" if s["first_differing"] else ""))
        md.append(f"| {st} | " + " | ".join(cells) + " |")
    md.append("")
    for i, x in enumerate(reports):
        for y in reports[i + 1:]:
            out["pairwise"][f"{x.get('label')} vs {y.get('label')}"] = compare_pair(x, y)
    md.append("**Verdicts**\n")
    for name, c in [(f"{lab} vs {ref.get('label')}", v) for lab, v in cmp_ref.items()] + list(out["pairwise"].items()):
        e = c["earliest_difference"]
        line = f"- {name}: " + ("identical in every compared item" if e is None else f"first difference at **{e}**")
        lab = c.get("labels")
        if lab:
            if lab["csv_identical"]:
                line += "; dressed CSV byte-identical, so every label agrees"
            else:
                line += (f"; dressed CSV differs: {lab['blocks_differing']} of {lab['blocks']} blocks of "
                         f"{lab['block_frames']} frames, graded label-count change {lab['count_delta_graded']}, "
                         f"runs {lab['runs_graded']}, score {lab['score']}")
        if c["not_compared"]:
            line += f" (not compared: {', '.join(c['not_compared'])})"
        md.append(line)
        if e is not None:
            print(f"::warning::{name}: first difference at {e}")
    md.append("\n| stage wall time (min) | " + " | ".join(str(c.get("label")) for c in cols) + " |")
    md.append("|---|" + "---|" * len(cols))
    names = []
    for c in cols:
        names += [k for k in (c.get("timings") or {}) if k not in names]
    for k in names:
        md.append(f"| {k} | " + " | ".join(
            f"{c['timings'][k]['wall_s'] / 60:.1f}" if k in (c.get("timings") or {}) else "" for c in cols) + " |")
    text = "\n".join(md) + "\n"
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    if a.out:
        _dump(out, a.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time", help="run a command and record its wall time (or record --seconds)")
    t.add_argument("--out", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--seconds", type=float, default=None)
    t.add_argument("command", nargs=argparse.REMAINDER)
    e = sub.add_parser("env", help="versions and CPU facts")
    e.add_argument("--out", required=True)
    r = sub.add_parser("report", help="one machine's hashes, digests, versions and timings")
    r.add_argument("--label", required=True)
    r.add_argument("--kind", default="ci", choices=("ci", "reference"))
    r.add_argument("--note", default="")
    r.add_argument("--commit", default="", help="override the commit recorded in the env facts")
    r.add_argument("--out", required=True)
    for k in ("dressed", "csv", "digests", "manifest", "predict-log", "timings", "env", "work", "external"):
        r.add_argument(f"--{k}", default=None)
    c = sub.add_parser("compare", help="compare reports with each other and with the reference")
    c.add_argument("--reference", required=True)
    c.add_argument("--summary", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("reports", nargs="*")
    a = ap.parse_args()
    return {"time": cmd_time, "env": cmd_env, "report": cmd_report, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
