"""Runtime and peak memory of `featurize` on a full-length synthetic match (ATD-like: 46 slots).

    python scripts/feature_bench.py --cache <dir>     # builds the match once, then reuses it

The match is synthetic (about 199.5k frames, two 53-minute halves, real-match padding), spread
over 46 anonymous slots with dropouts, duplicates and ghosts, and with ~41% bursty ball loss.
Prints aggregates only.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from katena.feats import featurize  # noqa: E402
from katena.feats.stress import drop_ball_bursts, fragment_slots  # noqa: E402
from katena.match import Match  # noqa: E402
from katena.synth import synthetic_match  # noqa: E402


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB."""
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return float("nan")
        return c.PeakWorkingSetSize / 2 ** 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


def build(cache: Path) -> Match:
    f = cache / "bench_match_v1.npz"
    if f.exists():
        z = np.load(f)
        hb = [tuple(int(v) for v in row) for row in z["half_blocks"]]
        return Match(P=z["P"], team=z["team"], speed=z["speed"], ball=z["ball"], half_blocks=hb,
                     attack_dir={int(h): int(d) for h, d in z["attack_dir"]}, y=z["y"],
                     name="bench")
    t = time.time()
    m = synthetic_match(seed=11, half_minutes=53, pre=11185, gap=20156, post=9121)
    m = drop_ball_bursts(fragment_slots(m, n_slots=46, seed=1), seed=2)
    print(f"built synthetic match in {time.time() - t:.0f} s")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, P=m.P, team=m.team, speed=m.speed, ball=m.ball,
             half_blocks=np.array(m.half_blocks), y=m.y,
             attack_dir=np.array(sorted(m.attack_dir.items())))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    m = build(args.cache)
    g = m.graded_mask()
    print(f"match: {m.n} frames, {m.n_slots} slots, {g.sum()} graded, "
          f"objects/frame {np.isfinite(m.P[g, :, 0]).sum(1).mean():.1f}, "
          f"ball missing on graded {np.isnan(m.ball[g, 0]).mean():.1%}")
    base = peak_rss_mb()
    t = time.perf_counter()
    f = featurize(m, {"stride": args.stride})
    dt = time.perf_counter() - t
    peak = peak_rss_mb()
    print(f"featurize: X {f.X.shape} {f.X.dtype} ({f.X.nbytes / 2 ** 20:.0f} MB) in {dt:.1f} s")
    print(f"peak RSS: {peak:.0f} MB (before featurize {base:.0f} MB, input arrays "
          f"{(m.P.nbytes + m.speed.nbytes + m.ball.nbytes) / 2 ** 20:.0f} MB)")


if __name__ == "__main__":
    main()
