"""Build, cache and summarise the external training corpus.

    python -m katena.external.corpus build            # download + convert + cache all matches
    python -m katena.external.corpus report           # markdown spell-statistics table

Run from the repository root with `src` and `evaluation` importable (e.g. PYTHONPATH=src). The
corpus lives OUTSIDE the repository, shared by every worktree:

    C:/Users/User/Desktop/barca/external_data/        (override: KATENA_EXTERNAL_DATA)
        raw/idsse/<id>/<id>_{meta,events}.xml         kept (small); tracking XML deleted after
        raw/metrica/game3/Sample_Game_3_*             kept (55 MB)
        cache/<name>.npz                              one compressed Match per match (raw labels)

Caches hold the RAW provider labels. Apply `harmonise()` at load time
(`load_corpus(harmonise_cfg=...)`), so the rule can change without re-downloading anything.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ..match import A, D, H, X, Match
from .common import load_match_npz, possession_metrics, save_match_npz
from .harmonise import DEFAULT_HARMONISE, harmonise
from .idsse import IDSSE_MATCHES, download_idsse, idsse_paths, load_idsse
from .metrica_open import load_metrica_game3

EXTERNAL_DATA = Path(os.environ.get("KATENA_EXTERNAL_DATA",
                                    "C:/Users/User/Desktop/barca/external_data"))
IDSSE_NAMES = [f"idsse-{mid}" for mid in IDSSE_MATCHES]
CORPUS_NAMES = IDSSE_NAMES + ["metrica-game3"]


def cache_path(name: str, root=EXTERNAL_DATA) -> Path:
    return Path(root) / "cache" / f"{name}.npz"


def build_one(name: str, root=EXTERNAL_DATA, keep_raw_tracking: bool = False,
              force: bool = False, log=print) -> Path:
    dst = cache_path(name, root)
    if dst.exists() and not force:
        log(f"{name}: cached")
        return dst
    t = time.time()
    if name.startswith("idsse-"):
        mid = name.split("-", 1)[1]
        paths = download_idsse(mid, root, log=log)
        m = load_idsse(paths["meta"], paths["tracking"], paths["event"], name=name)
        save_match_npz(m, dst)
        if not keep_raw_tracking:
            # re-downloadable and fully converted; the cache is what everything reads
            Path(paths["tracking"]).unlink()
    elif name == "metrica-game3":
        m = load_metrica_game3(root)
        save_match_npz(m, dst)
    else:
        raise KeyError(f"unknown corpus match {name!r}; have {CORPUS_NAMES}")
    log(f"{name}: {m.n} frames, {m.n_slots} slots -> {dst.name} "
        f"({dst.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f} s)")
    return dst


def build_corpus(root=EXTERNAL_DATA, names=None, keep_raw_tracking: bool = False,
                 force: bool = False, log=print) -> list[Path]:
    """Download, convert and cache every external match. Idempotent: cached matches are kept."""
    root = Path(root)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    out = []
    for name in names or CORPUS_NAMES:
        free = shutil.disk_usage(root).free / 1e9
        if free < 2.0:
            raise RuntimeError(f"only {free:.1f} GB free under {root}; stopping before {name}")
        out.append(build_one(name, root, keep_raw_tracking, force, log))
    return out


def load_corpus(root=EXTERNAL_DATA, names=None, harmonise_cfg: dict | None = None,
                ) -> list[Match]:
    """Cached matches. With `harmonise_cfg`, `y` is harmonised and the raw labels are kept in
    `meta["y_raw"]`. Matches that are not cached are skipped."""
    out = []
    for name in names or CORPUS_NAMES:
        p = cache_path(name, root)
        if not p.exists():
            continue
        m = load_match_npz(p)
        if harmonise_cfg is not None and m.y is not None:
            m.meta["y_raw"] = m.y
            m.y = harmonise(m.y, harmonise_cfg)
            m.meta["harmonise_cfg"] = {**DEFAULT_HARMONISE, **harmonise_cfg}
        out.append(m)
    return out


# -- statistics --------------------------------------------------------------------------------

def spell_stats(y: np.ndarray, fps: int = 25) -> dict:
    """Possession-spell statistics over the non-X runs of `y`, via the official scorer."""
    pm = possession_metrics()
    runs = pm.valid_runs(y)
    spells = pm.possession_spells(y, runs)
    lengths = np.array([b - a for a, b, _ in spells], dtype=np.float64) / fps
    graded = y != X
    n_graded = int(graded.sum())
    alive_s = float(np.isin(y, (H, A)).sum()) / fps
    base = pm.possession_stats(y, runs)
    q = (lambda p: float(np.percentile(lengths, p))) if len(lengths) else (lambda p: float("nan"))
    return {"n_spells": len(spells), "mean_s": base["mean_duration_s"],
            "median_s": q(50), "p10_s": q(10), "p90_s": q(90),
            "share_lt_1s": float((lengths < 1).mean()) if len(lengths) else float("nan"),
            "share_lt_2s": float((lengths < 2).mean()) if len(lengths) else float("nan"),
            "spells_per_alive_min": len(spells) / (alive_s / 60) if alive_s else float("nan"),
            "dead_share": float((y[graded] == D).mean()) if n_graded else float("nan"),
            "home_share": base["home_share"], "n_graded": n_graded,
            "n_dead_stretches": sum(1 for a, b, k in pm.segments(y, runs) if k == D)}


def match_report(m: Match, cfg: dict | None = None) -> dict:
    g = m.graded_mask()
    h = harmonise(m.y, cfg)
    return {"name": m.name, "frames": m.n, "graded": int(g.sum()),
            "ball_missing_share": float(np.isnan(m.ball[g, 0]).mean()),
            "raw": spell_stats(m.y, m.fps), "harmonised": spell_stats(h, m.fps),
            "frames_relabelled": int((h != m.y).sum())}


def report_markdown(rows: list[dict]) -> str:
    head = ("| match | frames | graded | ball missing | dead share | labels | spells | "
            "per alive min | mean s | median s | p10 s | p90 s | < 2 s |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        for kind in ("raw", "harmonised"):
            s = r[kind]
            if kind == "raw":
                lead = [r["name"], str(r["frames"]), str(r["graded"]),
                        f"{r['ball_missing_share']:.1%}", f"{s['dead_share']:.1%}"]
            else:
                lead = [""] * 5
            cells = lead + [kind, str(s["n_spells"]), f"{s['spells_per_alive_min']:.2f}",
                            f"{s['mean_s']:.2f}", f"{s['median_s']:.2f}", f"{s['p10_s']:.2f}",
                            f"{s['p90_s']:.2f}", f"{s['share_lt_2s']:.1%}"]
            lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def pooled_markdown(rows: list[dict]) -> str:
    out = ["| source | labels | matches | spells | mean s | median s | spells per alive min |",
           "|---|---|---|---|---|---|---|"]
    groups = {"IDSSE/DFL": [r for r in rows if r["name"].startswith("idsse-")],
              "Metrica Game 3": [r for r in rows if r["name"] == "metrica-game3"]}
    for src, rs in groups.items():
        if not rs:
            continue
        for kind in ("raw", "harmonised"):
            n = sum(r[kind]["n_spells"] for r in rs)
            mean = sum(r[kind]["mean_s"] * r[kind]["n_spells"] for r in rs) / n
            med = float(np.median([r[kind]["median_s"] for r in rs]))
            spm = float(np.mean([r[kind]["spells_per_alive_min"] for r in rs]))
            out.append(f"| {src} | {kind} | {len(rs)} | {n} | {mean:.2f} | {med:.2f} "
                       f"(median of matches) | {spm:.2f} |")
    return "\n".join(out) + "\n"


def evidence_markdown(matches, root=EXTERNAL_DATA) -> str:
    """How `min_control_s` is derived: event evidence of control vs flip duration."""
    from .control_evidence import corpus_evidence, derive_threshold
    ev = corpus_evidence(matches, root)
    res = derive_threshold(ev["evidence"])
    out = ["| duration of removable flip | flips | share with a deliberate action by that team |",
           "|---|---|---|"]
    for lo, hi, n, p in res["by_bin"]:
        hi_s = f"{hi:g} s" if np.isfinite(hi) else "inf"
        out.append(f"| {lo:g} - {hi_s} | {n} | {p:.2f} |")
    out += ["", "| match | flips | clock shift s | Play-owner agreement | best t (own) | "
            "best t (others, LOMO) | held-out accuracy |", "|---|---|---|---|---|---|---|"]
    for name in sorted(ev["evidence"]):
        al = ev["alignment"].get(name, {})
        shift = f"{al['shift_s']:+.2f}" if al else "n/a (frame-synced)"
        agree = f"{al['play_owner_agreement']:.3f}" if al else "n/a"
        t_lomo, acc = res["lomo"][name]
        out.append(f"| {name} | {len(ev['evidence'][name][0])} | {shift} | {agree} | "
                   f"{res['per_match'][name]:.1f} | {t_lomo:.1f} | {acc:.3f} |")
    p = res["pooled"]
    lo, hi = res["bootstrap_p5_p95"]
    out += ["", f"Pooled over {res['n_flips']} flips: accuracy-best t = {p['accuracy_best_t']:.1f} s "
            f"(accuracy {p['accuracy']:.3f}), balanced-accuracy-best t = "
            f"{p['balanced_best_t']:.1f} s. Match bootstrap: median {res['bootstrap_median']:.1f} s, "
            f"90% interval [{lo:.1f}, {hi:.1f}] s.", "",
            "| t (s) | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |", "|---|---|---|---|---|---|---|---|"]
    grid = p["grid"]
    accs = [p["acc_curve"][int(np.argmin(np.abs(grid - t)))] for t in (2, 2.5, 3, 3.5, 4, 4.5, 5)]
    out.append("| pooled accuracy | " + " | ".join(f"{a:.3f}" for a in accs) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build", "report", "evidence"])
    ap.add_argument("--root", default=str(EXTERNAL_DATA))
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--keep-raw-tracking", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-control-s", type=float, default=DEFAULT_HARMONISE["min_control_s"])
    args = ap.parse_args(argv)
    if args.command == "build":
        build_corpus(args.root, args.names, args.keep_raw_tracking, args.force)
        return 0
    if args.command == "evidence":
        print(evidence_markdown(load_corpus(args.root, args.names), args.root))
        return 0
    cfg = {"min_control_s": args.min_control_s}
    rows = [match_report(m, cfg) for m in load_corpus(args.root, args.names)]
    print(f"harmonise cfg: {({**DEFAULT_HARMONISE, **cfg})}\n")
    print(report_markdown(rows))
    print(pooled_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
