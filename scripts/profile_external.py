#!/usr/bin/env python
"""Profile possession-label structure of the canonical external matches.

    uv run python scripts/profile_external.py [--root external_data/canonical] [--datasets ...]

Spells are the official scorer's: ``possession_metrics.valid_runs/segments/possession_spells``
(maximal H or A runs, never spanning an X run or a half). Durations use each match's own fps.
Sandwich merge T: an H/A segment shorter than T seconds whose two neighbours (in the same run)
are both the *other* team, with no D between, is relabelled to the surrounding team; repeated
to a fixed point. Labels are raw source labels; nothing is harmonised here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evaluation"))

import possession_metrics as pm  # noqa: E402

from katena.types import load_match  # noqa: E402

MERGE_T = (0.5, 1, 2, 3, 5)
QS = (10, 25, 50, 75, 90, 99)


def sandwich(y: np.ndarray, runs, fps: float, T: float) -> np.ndarray:
    """Relabel brief opponent flips (< T s) between two same-team segments."""
    y = y.copy()
    lim = T * fps
    for run in runs:
        while True:
            segs = pm.segments(y, [run])
            changed = False
            for i in range(1, len(segs) - 1):
                a, b, c = segs[i - 1], segs[i], segs[i + 1]
                if (b[2] in pm.POSSESSION_LABELS and a[2] == c[2] and a[2] in pm.POSSESSION_LABELS
                        and a[2] != b[2] and (b[1] - b[0]) < lim):
                    y[b[0]:b[1]] = a[2]
                    changed = True
                    break          # segments changed; recompute (runs are short: <100 flips/min)
            if not changed:
                break
    return y


def match_stats(m, x_policy: str = "break") -> dict:
    """``x_policy``: "break" (default) leaves X as is, so spells never cross an X run (the scorer's
    handling of X in a ground truth); "ffill" forward-fills X inside each half block first (the
    scorer's handling of abstentions in a *prediction*), i.e. treats undetermined frames as
    continuing the previous label. Only differs for sources with X inside the halves."""
    y, fps = m.y, m.fps
    blocks = [(a, b + 1) for _, a, b in m.half_blocks]
    if x_policy == "ffill":
        y = pm.fill_abstentions(y, blocks)
    runs = pm.valid_runs(y)
    spells = pm.possession_spells(y, runs)
    dur = np.array([(b - a) / fps for a, b, _ in spells])
    segs_by_run = [pm.segments(y, [r]) for r in runs]
    dead = np.array([(b - a) / fps for s in segs_by_run for a, b, c in s if c == pm.D])
    direct = via_d_change = via_d_same = 0
    for s in segs_by_run:
        codes = [c for _, _, c in s]
        for i in range(1, len(codes)):
            if codes[i] in pm.POSSESSION_LABELS and codes[i - 1] in pm.POSSESSION_LABELS \
                    and codes[i] != codes[i - 1]:
                direct += 1
        for i in range(1, len(codes) - 1):
            if codes[i] == pm.D and codes[i - 1] in pm.POSSESSION_LABELS \
                    and codes[i + 1] in pm.POSSESSION_LABELS:
                if codes[i - 1] != codes[i + 1]:
                    via_d_change += 1
                else:
                    via_d_same += 1
    inrun = np.zeros(m.n, bool)
    for a, b in runs:
        inrun[a:b] = True
    inblk = np.zeros(m.n, bool)
    for a, b in blocks:
        inblk[a:b] = True
    n_in = max(int(inrun.sum()), 1)
    act = np.isfinite(m.P[..., 0])[inblk].sum(1)          # over the whole half blocks
    out = {
        "name": m.name, "fps": fps, "n_spells": len(spells), "dur": dur, "dead": dead,
        "dead_share": float((y[inrun] == pm.D).sum() / n_in),
        "x_share": float((m.y[inblk] == pm.X).mean()),
        "direct": direct, "via_d_change": via_d_change, "via_d_same": via_d_same,
        "ball_nan": float((~np.isfinite(m.ball[inblk, 0])).mean()),
        "act_min": int(act.min()), "act_mean": float(act.mean()), "act_max": int(act.max()),
        "merge": {},
    }
    for T in MERGE_T:
        y2 = sandwich(y, runs, fps, T)
        sp = pm.possession_spells(y2, runs)
        d2 = np.array([(b - a) / fps for a, b, _ in sp])
        out["merge"][T] = (len(sp), float(d2.mean()) if len(d2) else float("nan"))
    return out


def _pool(stats: list[dict]) -> dict:
    dur = np.concatenate([s["dur"] for s in stats]) if stats else np.zeros(0)
    dead = np.concatenate([s["dead"] for s in stats]) if stats else np.zeros(0)
    pooled = {"name": "POOLED", "n_spells": int(sum(s["n_spells"] for s in stats)), "dur": dur,
              "dead": dead}
    for k in ("dead_share", "x_share", "ball_nan", "act_mean"):
        pooled[k] = float(np.mean([s[k] for s in stats]))
    for k in ("direct", "via_d_change", "via_d_same"):
        pooled[k] = int(sum(s[k] for s in stats))
    pooled["act_min"] = min(s["act_min"] for s in stats)
    pooled["act_max"] = max(s["act_max"] for s in stats)
    pooled["merge"] = {T: (int(sum(s["merge"][T][0] for s in stats)),
                           float(np.average([s["merge"][T][1] for s in stats],
                                            weights=[s["merge"][T][0] for s in stats])))
                       for T in MERGE_T}
    return pooled


def _q(a: np.ndarray, qs=QS) -> str:
    return " ".join(f"{v:6.1f}" for v in np.percentile(a, qs)) if len(a) else "     -"


def report(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'match':<26}{'fps':>4}{'spells':>7}{'mean_s':>7}{'med_s':>6}{'<1s%':>6}{'<2s%':>6}{'<5s%':>6}"
          f"{'dead%':>6}{'X%':>6}{'ballNaN%':>9}{'slots min/mean/max':>20}   (ballNaN, slots: whole half blocks)")
    for r in rows:
        d = r["dur"]
        pct = lambda t: 100 * float((d < t).mean()) if len(d) else float("nan")  # noqa: E731
        slots = f"{r['act_min']}/{r['act_mean']:.1f}/{r['act_max']}"
        print(f"{r['name']:<26}{r.get('fps', 0):>4.0f}{r['n_spells']:>7}{d.mean() if len(d) else float('nan'):>7.1f}"
              f"{np.median(d) if len(d) else float('nan'):>6.1f}{pct(1):>6.1f}{pct(2):>6.1f}{pct(5):>6.1f}"
              f"{100 * r['dead_share']:>6.1f}{100 * r['x_share']:>6.1f}{100 * r['ball_nan']:>9.1f}{slots:>20}")
    print(f"\nspell duration quantiles (s) {QS}  |  dead-stretch quantiles (s)  |  H<->A direct / via D (team change) / via D (same team)")
    for r in rows:
        print(f"{r['name']:<26}{_q(r['dur'])}  |{_q(r['dead'])}  | {r['direct']:>5} {r['via_d_change']:>5} {r['via_d_same']:>5}")
    print(f"\nsandwich merge: spells / mean spell (s) after relabelling opponent flips < T s")
    print(f"{'match':<26}{'raw':>14}" + "".join(f"{'T=' + str(T):>14}" for T in MERGE_T))
    for r in rows:
        raw = (r["n_spells"], float(r["dur"].mean()) if len(r["dur"]) else float("nan"))
        cells = [raw] + [r["merge"][T] for T in MERGE_T]
        print(f"{r['name']:<26}" + "".join(f"{n:>6}/{m:>6.1f}  " for n, m in cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__import__("os").environ.get("KATENA_EXTERNAL") or REPO / "external_data") / "canonical")
    ap.add_argument("--datasets", nargs="+", default=["idsse", "metrica", "skillcorner"])
    ap.add_argument("--x-policy", choices=["break", "ffill"], default="break")
    ap.add_argument("--pooled-only", action="store_true", help="print only the per-dataset pooled rows")
    args = ap.parse_args()
    pooled = []
    for ds in args.datasets:
        files = sorted((args.root / ds).glob("*.npz"))
        if not files:
            print(f"[{ds}] no canonical matches under {args.root / ds}")
            continue
        stats = [match_stats(load_match(f), args.x_policy) for f in files]
        for s in stats:
            s.setdefault("fps", 0)
        fps = load_match(files[0]).fps
        p = _pool(stats)
        p["name"], p["fps"] = f"POOLED {ds} ({len(stats)})", fps
        pooled.append(p)
        report(ds, ([] if args.pooled_only else stats) + [p])
    if len(pooled) > 1:
        report("all datasets (pooled)", pooled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
