#!/usr/bin/env python
"""SHA-256 of every training input and of the model, stage by stage, to compare two machines (J3).

    python scripts/cache_digests.py dump --out digests_<machine>.json     # KATENA_EXTERNAL, KATENA_WORK
    python scripts/cache_digests.py diff digests_mac.json digests_windows.json

``dump`` hashes the canonical external matches, then every file of the cache in pipeline order (ground truth,
base pools, features per world, touch columns, ``d2j`` pools, their touch columns) and the model manifest. ``diff``
reports, per stage, how many files agree, and names the first that differ: the earliest stage that differs is
where two machines part. The ``.npz`` writers fix the zip timestamps, so equal arrays give equal file bytes.
Reads external matches and the cache only; prints hashes, names of external matches, and counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

STAGES = ("canonical", "gt", "pool", "feat", "touch", "pool_d2j", "touch_pool_d2j")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def components(bundle: Path) -> dict:
    """Hashes of each part of a ``katena.pipeline.Model`` bundle: LightGBM strings, TCN weights, scaler, priors."""
    import pickle

    import numpy as np
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "src"), str(repo / "evaluation"), str(repo)]
    with open(bundle, "rb") as fh:
        m = pickle.load(fh)

    def h(*arrays) -> str:
        g = hashlib.sha256()
        for x in arrays:
            x = np.ascontiguousarray(x)
            g.update(f"{x.dtype.str}{x.shape}".encode())
            g.update(x.tobytes())
        return g.hexdigest()
    p = m.priors
    return {"part_gbdt_alive": hashlib.sha256(m.gbdt_alive.encode()).hexdigest(),
            "part_gbdt_team": hashlib.sha256(m.gbdt_team.encode()).hexdigest(),
            "part_tcn_state": h(*[m.tcn_state[k] for k in sorted(m.tcn_state)]),
            "part_scaler": h(m.scaler_mu, m.scaler_sd),
            "part_priors_viterbi": h(p.logA, p.log_init, p.log_class),
            "part_priors_hsmm": h(p.logT, p.logdur, p.dur_params),
            "part_names": hashlib.sha256("\n".join(map(str, m.names)).encode()).hexdigest()}


def dump(a) -> None:
    ext = Path(os.environ.get("KATENA_EXTERNAL") or a.external)
    work = Path(os.environ.get("KATENA_WORK") or a.work)
    out = {"machine": {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
           "stages": {}}
    out["stages"]["canonical"] = {p.relative_to(ext / "canonical").as_posix(): sha256_file(p)
                                  for p in sorted((ext / "canonical").glob("*/*.npz"))}
    for st in STAGES[1:]:
        out["stages"][st] = {p.name: sha256_file(p) for p in sorted((work / st).glob("*.npz"))}
        print(f"{st}: {len(out['stages'][st])} files", flush=True)
    models = {}
    for d in sorted(work.glob("model*")):
        man = d / "manifest.json"
        if man.exists():
            models[d.name] = {**json.loads(man.read_text(encoding="utf-8"))["sha256"], **components(d / "bundle.pkl")}
    out["models"] = models
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {sum(len(v) for v in out['stages'].values())} files, models {sorted(models)}")


def diff(a) -> int:
    x, y = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b))
    print(f"{a.a}: {x['machine']}\n{a.b}: {y['machine']}")
    first = None
    for st in STAGES:
        sx, sy = x["stages"].get(st, {}), y["stages"].get(st, {})
        common = sorted(set(sx) & set(sy))
        bad = [k for k in common if sx[k] != sy[k]]
        only = len(set(sx) ^ set(sy))
        print(f"{st:15s} {len(common) - len(bad)} of {len(common)} identical, {only} in one only"
              + (f"; first differing: {', '.join(bad[:5])}" if bad else ""))
        if bad and first is None:
            first = st
    mx, my = x.get("models", {}), y.get("models", {})
    for name in sorted(set(mx) | set(my)):
        if name not in mx or name not in my:
            print(f"model {name}: in one only")
            continue
        hx, hy = mx[name], my[name]
        print(f"model {name}: " + ", ".join(f"{k} {'identical' if hx.get(k) == hy.get(k) else 'DIFFERENT'}"
                                            for k in sorted(set(hx) | set(hy))))
    print(f"earliest stage that differs: {first or 'none'}")
    return 0 if first is None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--external", default="external_data")
    d.add_argument("--work", default="work/final")
    f = sub.add_parser("diff")
    f.add_argument("a")
    f.add_argument("b")
    a = ap.parse_args()
    return dump(a) or 0 if a.cmd == "dump" else diff(a)


if __name__ == "__main__":
    sys.exit(main())
