"""What an AI assistant may read of a real run: verdicts and hashes from ``submissions/<team>.run.json``.

``scripts/real_run.py`` writes the run report. Its console output, which an assistant must not read, holds label
shares, D shares and touch counts. This prints less than the report: the result, the commit, each check's
verdict and its FAIL lines (``docs/real-run.md``: "the FAIL line is safe to paste"), the model and CSV hashes,
the validator's OK/Valid lines, the public-window check's PASS/FAIL lines, and warnings that do not come from a
check on the provided files (those name only their script). Nothing per frame, no share and no count of labels.

    python scripts/run_summary.py submissions/<team>.run.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def verdict_only(line: str) -> str:
    """A FAIL line as the run recorded it, cut at 200 characters."""
    return line.strip()[:200]


def summary(r: dict) -> list[str]:
    out = [f"team         {r.get('team')}",
           f"result       {r.get('result')}"
           + (f"  (stopped at {r.get('failed_step')}: {r.get('stop_reason')})" if r.get("result") != "PASS" else ""),
           f"commit       {r.get('commit')} on {r.get('branch')}, tree clean: {r.get('tree_clean')}",
           f"spec         {r.get('model_spec')}", f"started      {r.get('started')}",
           f"finished     {r.get('finished')}"]
    for c in r.get("checks") or []:
        out.append(f"check        {c.get('script')}: {c.get('result')}"
                   + (f" ({c.get('pass')} PASS, {c.get('fail')} FAIL)" if c.get("result") != "SKIP" else ""))
        out += [f"               {verdict_only(x)}" for x in c.get("lines", []) if x.startswith("FAIL")]
    ms = r.get("make_submission") or {}
    out.append(f"make_submission  {ms.get('result')} (exit {ms.get('exit_status')})")
    exp, got, chk = r.get("expected_sha256") or {}, r.get("model_sha256") or {}, r.get("hash_check") or {}
    for k in sorted(set(exp) | set(got)):
        g = got.get(k)
        out.append(f"hash {k:<8} {chk.get(k, '?'):<8} got {g if g and HEX64.match(str(g)) else g}")
    v = r.get("validate_submission") or {}
    out.append(f"validate     {v.get('result')}")
    out += [f"               {x.strip()[:120]}" for x in v.get("lines", []) if x.startswith(("OK ", "Valid.", "FAIL"))]
    p = r.get("public_window") or {}
    out.append(f"public window {p.get('result')}")
    out += [f"               {x}" for x in p.get("lines", []) if re.fullmatch(r"(\w+: (PASS|FAIL))|(FAIL .{0,150})", x)]
    csv = r.get("csv") or {}
    out.append(f"csv          {csv.get('file')}  sha256 {csv.get('sha256')}  bytes {csv.get('bytes')}")
    for w in r.get("warnings") or []:
        head = w.split(":", 1)[0]
        out.append(f"warning      {head}: (text withheld, a check on the provided files)"
                   if head.startswith("check_") else f"warning      {w[:200]}")
    out.append(f"seconds      {json.dumps(r.get('seconds'))}")
    return out


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    report = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    print("\n".join(summary(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
