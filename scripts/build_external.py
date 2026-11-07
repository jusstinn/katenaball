#!/usr/bin/env python
"""Build the external training corpus: download -> verify SHA256 -> canonical Match npz.

    uv run python scripts/build_external.py                 # everything
    uv run python scripts/build_external.py --datasets metrica --skip-download

Idempotent: a match is rebuilt only if its raw inputs (SHA256), the loader code, or --force
changed. Output: external_data/canonical/<dataset>/<match>.npz plus a <match>.json sidecar.
Raw downloads are untrusted data; nothing in them is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from katena.external import fetch, idsse, metrica, skillcorner  # noqa: E402
from katena.external.common import validate_canonical  # noqa: E402
from katena.types import save_match  # noqa: E402

CODE_FILES = ["src/katena/types.py", "src/katena/external/common.py", "src/katena/external/idsse.py",
              "src/katena/external/metrica.py", "src/katena/external/skillcorner.py"]


def code_digest(dataset: str) -> str:
    h = hashlib.sha256()
    for f in CODE_FILES[:2] + [f for f in CODE_FILES[2:] if Path(f).stem == dataset]:
        h.update((REPO / f).read_bytes())
    return h.hexdigest()


def jobs(root: Path, dataset: str):
    """Yield (match_name, [input files], builder callable)."""
    if dataset == "idsse":
        for mid, f in idsse.match_files(root / "idsse" / "raw").items():
            yield mid, [f["info"], f["events"], f["positions"]], \
                (lambda f=f: idsse.build_match(f["info"], f["events"], f["positions"]))
    elif dataset == "metrica":
        d = root / "metrica" / "repo" / "data"
        for k in (1, 2, 3):
            gd = d / f"Sample_Game_{k}"
            yield f"Sample_Game_{k}", sorted(p for p in gd.iterdir() if p.is_file()), \
                (lambda k=k, gd=gd: metrica.build_game3(gd) if k == 3 else metrica.build_csv_game(gd, k))
    elif dataset == "skillcorner":
        md = root / "skillcorner" / "repo" / "data" / "matches"
        for mid in skillcorner.match_ids(md):
            yield mid, sorted(p for p in (md / mid).iterdir() if p.is_file()), \
                (lambda mid=mid: skillcorner.build_match(md / mid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__import__("os").environ.get("KATENA_EXTERNAL") or REPO / "external_data"))
    ap.add_argument("--datasets", nargs="+", default=["idsse", "metrica", "skillcorner"],
                    choices=["idsse", "metrica", "skillcorner"])
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root: Path = args.root
    manifest = fetch.Manifest(REPO / "scripts" / "external_manifest.json")

    if not args.skip_download:
        for ds in args.datasets:
            t = time.time()
            {"idsse": fetch.fetch_idsse, "metrica": fetch.fetch_metrica,
             "skillcorner": fetch.fetch_skillcorner}[ds](root, manifest)
            print(f"[{ds}] raw files present and SHA256-verified ({time.time() - t:.0f}s)", flush=True)

    for ds in args.datasets:
        out_dir = root / "canonical" / ds
        for name, inputs, build in jobs(root, ds):
            digests = {p.relative_to(root).as_posix(): manifest.check(root, p) for p in inputs}
            manifest.save()
            side = out_dir / f"{name}.json"
            npz = out_dir / f"{name}.npz"
            stamp = {"inputs": digests, "code_sha256": code_digest(ds)}
            if npz.exists() and side.exists() and not args.force:
                if {k: json.loads(side.read_text()).get(k) for k in stamp} == stamp:
                    print(f"[{ds}] {name}: up to date", flush=True)
                    continue
            t = time.time()
            m = build()
            validate_canonical(m)
            save_match(m, npz)
            side.write_text(json.dumps({**stamp, "name": m.name, "source": m.source, "fps": m.fps,
                                        "pitch": m.pitch, "n_frames": m.n, "n_slots": m.n_slots,
                                        "half_blocks": m.half_blocks, "attack_dir": m.attack_dir,
                                        "meta": m.meta}, indent=1, default=_j))
            q = m.meta["quality"]
            print(f"[{ds}] {name}: n={m.n} S={m.n_slots} inplay={q['n_inplay']} "
                  f"ballNaN={q['ball_nan_share_inplay']:.3f} "
                  f"nearest-team={q.get('nearest_team', {}).get('agree', float('nan')):.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
            del m
    return 0


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main())
