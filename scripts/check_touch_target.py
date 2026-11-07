#!/usr/bin/env python
"""Run the touch feature group on the real match and print aggregates only. A human runs this:

    uv run python scripts/check_touch_target.py

It prints counts, rates, a runtime, an array shape and two jitter statistics of the ball track, and the
same statistics for reference worlds built from one external development match, so the numbers can be
compared side by side. Nothing frame-level is printed, saved or returned. Exceptions print their type
only, because some exception messages can quote the file (see CLAUDE.md).

What the numbers are for (alarms, never a choice between versions):
* the features compute on the real format, with the expected shape, in reasonable time;
* touches per minute and the jitter statistics show whether the real 25 Hz ball is smooth frame to
  frame, like the external automatic tracking (Metrica sample games, SkillCorner), or rough, like the
  degrader's i.i.d. jitter that the touch group is trained on.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "evaluation"):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from katena.featurize import touch as TF  # noqa: E402
from katena.grid import block_mask, downsample  # noqa: E402

REFERENCE = ("idsse", "DFL-MAT-J03WMX")          # a development match (not a locked bench)


def stats(m_native, m5, nb) -> dict:
    """Aggregates of one native ball track and its touch features. ``m5``: the work-grid match whose
    players the features use; ``m_native``: for the in-play blocks only."""
    t0 = time.time()
    X, names, idx = TF.touch_features(m5, nb, focals=(0,))
    secs = time.time() - t0
    cnt = TF.count_touches(m5, nb)
    blk = block_mask(m_native)
    b = np.asarray(nb.ball, dtype=np.float64).copy()
    b[~blk] = np.nan
    vis = np.isfinite(b[:, 0])
    d2 = {}
    for k in (1, 5):
        ok = vis[k:-k] & vis[2 * k:] & vis[:-2 * k]
        dd = np.abs(b[2 * k:] - 2 * b[k:-k] + b[:-2 * k])
        d2[k] = float(np.median(dd[ok])) if ok.any() else float("nan")
    sig = X[0][:, names.index("t_sigma")]
    return {"rows": X[0].shape[0], "cols": X[0].shape[1], "secs": secs, "ball_vis": float(vis[blk].mean()),
            "touch_per_min": cnt["events"] / max(cnt["minutes"], 1e-9),
            "ident": cnt["identified"] / max(cnt["events"], 1), "sigma_med": float(np.nanmedian(sig)),
            "d2_1": d2[1], "d2_5": d2[5], "nan_share": float(np.isnan(X[0]).mean())}


def line(tag, s) -> str:
    return (f"{tag:40s} rows {s['rows']:6d} cols {s['cols']:3d} {s['secs']:5.1f}s | ball visible {s['ball_vis']:.2f} | "
            f"touches/min {s['touch_per_min']:5.1f} (attributed {s['ident']:.2f}) | jitter est {s['sigma_med']:.3f} m | "
            f"median |D2| lag1 {s['d2_1']:.4f} m, lag5 {s['d2_5']:.4f} m | NaN share {s['nan_share']:.2f}")


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        from katena.target.atd import load_target
        m = load_target()
        s = stats(m, downsample(m), TF.native_ball(m))
    except Exception as e:  # noqa: BLE001
        print(f"target: failed ({type(e).__name__})")
        return 1
    print(line("target (real 25 Hz ball)", s))
    try:
        from katena.eval import protocol as P
        from katena.eval import touch_gate as TG
        from katena.types import load_match
        ref = P.Ref(REFERENCE[0], REFERENCE[1], P.CANON / REFERENCE[0] / f"{REFERENCE[1]}.npz")
        mx = load_match(ref.path)
        m5 = downsample(mx)
        print(line(f"reference {REFERENCE[1]} clean", stats(mx, m5, TF.native_ball(mx))))
        for c in ("target", TG.SMOOTH, "severe"):
            mc, lifted = TG.corrupted(ref, mx, m5, c)
            print(line(f"reference {REFERENCE[1]} {c}", stats(mx, mc, TF.NativeBall(lifted, mx.fps, int(m5.meta["step"])))))
    except Exception as e:  # noqa: BLE001
        print(f"reference: skipped ({type(e).__name__}; needs the external corpus, KATENA_EXTERNAL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
