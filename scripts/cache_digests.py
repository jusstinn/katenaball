#!/usr/bin/env python
"""SHA-256 of every training input and of the model, stage by stage, to compare two machines (J3).

    python scripts/cache_digests.py dump --out digests_<machine>.json     # KATENA_EXTERNAL, KATENA_WORK
    python scripts/cache_digests.py diff digests_mac.json digests_windows.json

``dump`` hashes the canonical external matches, then every file of the cache in pipeline order (ground truth,
base pools, features per world, touch columns, ``d2j`` pools, their touch columns) and the model manifest. ``diff``
reports, per stage, how many files agree, and names the first that differ: the earliest stage that differs is
where two machines part.

Each ``.npz`` gets two digests: of the file's bytes, and of its arrays (``content``: member names, dtypes, shapes
and data, with a short digest per member). numpy fixes the zip timestamps, so on one OS equal arrays give equal
bytes. Across OS families they never do: Python's ``zipfile`` records the writing OS in every entry
(``create_system``, 0 on Windows, 3 on Linux and macOS). ``diff`` therefore judges a stage by its content digests
when both dumps have them, and by bytes otherwise.
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


def _array_hash(*arrays, canonical_nan: bool = False):
    """SHA-256 over the dtype, shape and data of each array: what a file holds, whatever container holds it.
    With ``canonical_nan`` every NaN is hashed as the same bit pattern. A NaN computed by arithmetic has the sign
    bit set on x86-64 and clear on arm64, and no step of the pipeline reads a NaN's sign or payload."""
    import numpy as np
    g = hashlib.sha256()
    for x in arrays:
        x = np.ascontiguousarray(x)
        if canonical_nan and x.dtype.kind in "fc":
            nan = np.isnan(x)
            if nan.any():
                x = x.copy()
                x[nan] = np.nan
        g.update(f"{x.dtype.str}{x.shape}".encode())
        g.update(x.tobytes())
    return g


def content_digest(path: Path) -> tuple:
    """Digests of an ``.npz`` file's arrays (member names in order, then each member's dtype, shape and data):
    exact, and with NaN bit patterns made canonical; and a 16-hex exact digest per member. Equal on every OS when
    the arrays are equal, unlike the file bytes."""
    import numpy as np
    g, gn, members = hashlib.sha256(), hashlib.sha256(), {}
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            x = z[k]
            d = _array_hash(x).digest()
            members[k] = d.hex()[:16]
            g.update(k.encode() + b":" + d)
            gn.update(k.encode() + b":" + _array_hash(x, canonical_nan=True).digest())
    return g.hexdigest(), gn.hexdigest(), members


def components(bundle: Path) -> dict:
    """Hashes of each part of a ``katena.pipeline.Model`` bundle: LightGBM strings, TCN weights, scaler, priors."""
    import pickle
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "src"), str(repo / "evaluation"), str(repo)]
    with open(bundle, "rb") as fh:
        m = pickle.load(fh)

    def h(*arrays) -> str:
        return _array_hash(*arrays).hexdigest()
    p = m.priors
    return {"part_gbdt_alive": hashlib.sha256(m.gbdt_alive.encode()).hexdigest(),
            "part_gbdt_team": hashlib.sha256(m.gbdt_team.encode()).hexdigest(),
            "part_tcn_state": h(*[m.tcn_state[k] for k in sorted(m.tcn_state)]),
            "part_scaler": h(m.scaler_mu, m.scaler_sd),
            "part_priors_viterbi": h(p.logA, p.log_init, p.log_class),
            "part_priors_hsmm": h(p.logT, p.logdur, p.dur_params),
            "part_priors_hsmm_logT": h(p.logT), "part_priors_hsmm_logdur": h(p.logdur),
            "part_priors_hsmm_dur_params": h(p.dur_params),
            "part_names": hashlib.sha256("\n".join(map(str, m.names)).encode()).hexdigest()}


def dump(a) -> None:
    ext = Path(os.environ.get("KATENA_EXTERNAL") or a.external)
    work = Path(os.environ.get("KATENA_WORK") or a.work)
    out = {"machine": {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
           "stages": {}, "content": {}, "content_nan": {}, "members": {}}
    files = {"canonical": [(p.relative_to(ext / "canonical").as_posix(), p)
                           for p in sorted((ext / "canonical").glob("*/*.npz"))]}
    for st in STAGES[1:]:
        files[st] = [(p.name, p) for p in sorted((work / st).glob("*.npz"))]
    for st in STAGES:
        for k in ("stages", "content", "content_nan", "members"):
            out[k][st] = {}
        for name, p in files[st]:
            out["stages"][st][name] = sha256_file(p)
            out["content"][st][name], out["content_nan"][st][name], out["members"][st][name] = content_digest(p)
        print(f"{st}: {len(files[st])} files", flush=True)
    models = {}
    for d in sorted(work.glob("model*")):
        man = d / "manifest.json"
        if man.exists():
            models[d.name] = {**json.loads(man.read_text(encoding="utf-8"))["sha256"], **components(d / "bundle.pkl")}
    out["models"] = models
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {sum(len(v) for v in out['stages'].values())} files, models {sorted(models)}")


def compare_stage(x: dict, y: dict, st: str) -> dict:
    """One stage of two dumps: files in common; how many agree by file bytes, by array content, and by array
    content with NaN bit patterns made canonical; the first that differ, and that file's differing members.

    ``verdict``: ``same`` (content, or bytes when a dump has no content digests: file bytes differ across OS
    families), ``nan_payload`` (the arrays differ only in NaN bit patterns), ``differs``, or ``missing``."""
    def get(d, k):
        return d.get(k, {}).get(st)
    bx, by = get(x, "stages") or {}, get(y, "stages") or {}
    common = sorted(set(bx) & set(by))
    bad_bytes = [k for k in common if bx[k] != by[k]]
    out = {"common": len(common), "only_one": len(set(bx) ^ set(by)), "bytes_same": len(common) - len(bad_bytes),
           "content_same": None, "nan_same": None, "by": "bytes", "first_differing": bad_bytes[:3],
           "members_differing": []}
    bad, bad_values = bad_bytes, bad_bytes
    if get(x, "content") is not None and get(y, "content") is not None:
        cx, cy = get(x, "content"), get(y, "content")
        bad = bad_values = [k for k in common if cx.get(k) != cy.get(k)]
        out.update(content_same=len(common) - len(bad), by="content", first_differing=bad[:3])
        if bad:
            mx = (get(x, "members") or {}).get(bad[0], {})
            my = (get(y, "members") or {}).get(bad[0], {})
            out["members_differing"] = sorted(k for k in set(mx) | set(my) if mx.get(k) != my.get(k))
        if get(x, "content_nan") is not None and get(y, "content_nan") is not None:
            nx, ny = get(x, "content_nan"), get(y, "content_nan")
            bad_values = [k for k in common if nx.get(k) != ny.get(k)]
            out["nan_same"] = len(common) - len(bad_values)
    out["first_value_differing"] = bad_values[:3]
    out["verdict"] = ("missing" if not common else "differs" if bad_values else "nan_payload" if bad else "same")
    return out


def diff(a) -> int:
    x, y = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b))
    print(f"{a.a}: {x['machine']}\n{a.b}: {y['machine']}")
    first = None
    for st in STAGES:
        c = compare_stage(x, y, st)
        content = "not in both dumps" if c["content_same"] is None else c["content_same"]
        print(f"{st:15s} {c['common']} in common, {c['only_one']} in one only; identical: bytes {c['bytes_same']}, "
              f"content {content}, up to NaN payloads {c['nan_same']}; {c['verdict']}"
              + (f"; first differing by {c['by']}: {', '.join(c['first_differing'])}"
                 if c["verdict"] in ("differs", "nan_payload") else "")
              + (f" (members {', '.join(c['members_differing'])})" if c["members_differing"] else ""))
        if c["verdict"] == "differs" and first is None:
            first = st
    mx, my = x.get("models", {}), y.get("models", {})
    for name in sorted(set(mx) | set(my)):
        if name not in mx or name not in my:
            print(f"model {name}: in one only")
            continue
        hx, hy = mx[name], my[name]
        print(f"model {name}: " + ", ".join(
            f"{k} {'in one dump only' if k not in hx or k not in hy else 'identical' if hx[k] == hy[k] else 'DIFFERENT'}"
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
