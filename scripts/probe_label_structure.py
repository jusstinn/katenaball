#!/usr/bin/env python3
"""Label structure of the public 5-minute ground truth beside the external labels at each convention T.

    uv run python scripts/probe_label_structure.py        # needs a built corpus cache (KATENA_WORK)

Reads labels only: the public file in ``evaluation/gt/`` and the external ground-truth caches. No model, no
prediction and no score is involved, so nothing here can select between two versions of the pipeline. It is
a look at the public window all the same, so log each run in docs/decision-log.md.

The question: is the way the public window is cut into possessions (how many, how short, how often a
hand-over is direct) what our training labels at T = 2 s would produce? For each dataset and T it slides a
5-minute window over the external labels and reports where the public window's values fall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

import numpy as np  # noqa: E402
import possession_metrics as pm  # noqa: E402

from katena.eval import corpus as C  # noqa: E402
from katena.eval import protocol as P  # noqa: E402

PUBLIC = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
WINDOW_S, STRIDE_S = 300, 30


def structure(y: np.ndarray, fps: float, runs=None) -> dict:
    """Possession and dead-ball structure of a label array inside its graded runs."""
    runs = runs or pm.valid_runs(y)
    segs = pm.segments(y, runs)
    sp = np.array([(b - a) / fps for a, b, k in segs if k in (pm.H, pm.A)])
    dd = np.array([(b - a) / fps for a, b, k in segs if k == pm.D])
    direct = sum(1 for (_, b, k), (a2, _, k2) in zip(segs[:-1], segs[1:])
                 if b == a2 and k in (pm.H, pm.A) and k2 in (pm.H, pm.A))
    minutes = sum(b - a for a, b in runs) / fps / 60
    return {"sp": sp, "dd": dd, "n": len(sp), "mean": float(sp.mean()) if len(sp) else 0.0,
            "median": float(np.median(sp)) if len(sp) else 0.0, "per_alive_min": len(sp) / max(sp.sum() / 60, 1e-9),
            "lt2": float((sp < 2).mean()) if len(sp) else 0.0, "lt4": float((sp < 4).mean()) if len(sp) else 0.0,
            "per_min": len(sp) / minutes, "d_share": float(dd.sum() / (minutes * 60)),
            "direct": direct / max(len(sp), 1)}


def main() -> int:
    pub = structure(pm.label_codes(pm.read_label_csv(PUBLIC)), 25.0)
    print("public window (5 minutes of play):")
    print(f"  {pub['n']} possessions, mean {pub['mean']:.1f} s, median {pub['median']:.1f} s, under 2 s {pub['lt2']:.2f}, "
          f"under 4 s {pub['lt4']:.2f}, {pub['per_min']:.2f} per minute ({pub['per_alive_min']:.2f} per alive minute), "
          f"dead share {pub['d_share']:.2f}, direct hand-overs {pub['direct']:.2f}")
    print("  possession lengths, s:", np.round(np.sort(pub["sp"]), 1).tolist())
    print("  dead-ball lengths, s:", np.round(np.sort(pub["dd"]), 1).tolist())

    refs = P.all_refs()
    print("\nexternal labels, whole matches (mean over matches):")
    print(f"{'dataset':12s} T  mean s  median s  <2 s  <4 s  per min  per alive min  dead share  direct")
    for ds in P.DATASETS:
        for T in P.T_SWEEP:
            st = []
            for r in refs:
                if r.dataset == ds:
                    g = C.load_gt(r, T)
                    st.append(structure(g["y"], g["fps"]))
            sp = np.concatenate([s["sp"] for s in st])
            f = lambda k: float(np.mean([s[k] for s in st]))  # noqa: E731
            print(f"{ds:12s} {T}  {sp.mean():6.1f}  {np.median(sp):8.1f}  {(sp < 2).mean():.2f}  {(sp < 4).mean():.2f}  "
                  f"{f('per_min'):7.2f}  {f('per_alive_min'):13.2f}  {f('d_share'):10.2f}  {f('direct'):6.2f}")

    print(f"\nwhere the public window falls among {WINDOW_S // 60}-minute windows of external labels "
          f"(share of windows at or below the public value):")
    print(f"{'dataset':9s} T  windows  n median [p10, p90]  P(n <= {pub['n']})  P(mean >= {pub['mean']:.1f} s)  "
          f"P(share under 4 s <= {pub['lt4']:.2f})  P(direct <= {pub['direct']:.2f})")
    for ds in ("idsse", "metrica"):                                  # SkillCorner's X gaps break its windows
        for T in P.T_SWEEP:
            ns, lt4, direct, means = [], [], [], []
            for r in refs:
                if r.dataset != ds:
                    continue
                g = C.load_gt(r, T)
                y, fps = g["y"], g["fps"]
                w = int(WINDOW_S * fps)
                for a, b in pm.valid_runs(y):
                    for s0 in range(a, b - w, int(STRIDE_S * fps)):
                        s = structure(y[s0:s0 + w], fps, runs=[(0, w)])
                        if s["n"]:
                            ns.append(s["n"])
                            lt4.append(s["lt4"])
                            direct.append(s["direct"])
                            means.append(s["mean"])
            ns, lt4, direct, means = np.array(ns), np.array(lt4), np.array(direct), np.array(means)
            print(f"{ds:9s} {T}  {len(ns):7d}  {np.median(ns):4.0f} [{np.percentile(ns, 10):.0f}, {np.percentile(ns, 90):.0f}]"
                  f"        {np.mean(ns <= pub['n']):.2f}        {np.mean(means >= pub['mean']):.2f}"
                  f"               {np.mean(lt4 <= pub['lt4']):.2f}                          {np.mean(direct <= pub['direct']):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
