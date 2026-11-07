"""Domain-gap alarm: can a classifier tell the real input from our degrader's output?

    # a human, on the real match (prints aggregates only):
    KATENA_EXTERNAL=D:/katena/external_data uv run python scripts/check_domain_gap.py --target
    # anyone, on external data: an external match with real tracking noise as the "real" side
    KATENA_EXTERNAL=D:/katena/external_data uv run python scripts/check_domain_gap.py --real skillcorner/1874553
    # smoke test, synthetic only
    uv run python scripts/check_domain_gap.py --synthetic

The simulated side is exactly the bake-off's corruption: katena.sim.degrade under mild, target-like
and severe, with the protocol seeds, applied to a few clean IDSSE development matches on the 5 Hz
work grid. The real side is downsampled to the same grid. For each severity the script prints a
cross-validated AUC (0.5 = indistinguishable) and the names of the window statistics that give the
simulator away; --detail adds the medians of those statistics on each side.

On the target it prints one alarm bit and a runtime, nothing else: no AUC per severity (that would rank
the degrader settings by closeness to the target), no statistic names and no medians (the moments a
degrader fit would match), and --detail is refused. To compute that bit the script does fit
real-versus-simulated classifiers on the target's window statistics, cross-validated.

This is an alarm (docs/PLAN.md, "Validating blind", step 5). It never selects between versions and
is never used to tune the degrader: fitting the degrader to the target is out until the organisers
confirm. It reads no labels and writes nothing. On the target, an exception prints its type only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "evaluation"):
    sys.path.insert(0, str(path))

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()                      # may re-exec once (LightGBM needs libomp)

import numpy as np  # noqa: E402

from katena.diag.domain_gap import domain_gap, window_features  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval.corpus import CONFIG  # noqa: E402
from katena.grid import downsample  # noqa: E402
from katena.sim.degrade import corrupt  # noqa: E402
from katena.types import Match, load_match  # noqa: E402

CLEAN_DEFAULT = ("idsse/DFL-MAT-J03WMX", "idsse/DFL-MAT-J03WN1", "idsse/DFL-MAT-J03WOH")   # development matches
SEVERITIES = ("mild", "target", "severe")
ALARM_MARGIN = 0.10     # fixed a priori: AUC above the match-vs-match reference that counts as a gap


def synthetic_clean(seed: int, half_s: float = 360.0, fps: float = 5.0) -> Match:
    """A clean 5 Hz match: 11 + 11 random walkers around their own half, a ball at a carrier."""
    rng = np.random.default_rng(seed)
    n_half, gap, S = int(half_s * fps), 50, 22
    n = 2 * n_half + gap
    team = np.repeat(np.array([0, 1], np.int8), 11)
    P_ = np.full((n, S, 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    blocks = [(1, 0, n_half - 1), (2, n_half + gap, n - 1)]
    for h, a, b in blocks:
        T = b - a + 1
        left = (team == 0) if h == 1 else (team == 1)
        base = np.stack([np.where(left, 30.0, 75.0) + rng.normal(0, 8, S), rng.uniform(5, 63, S)], 1)
        walk = np.clip(np.cumsum(rng.normal(0, 0.4, (T, S, 2)), 0), -15, 15)
        P_[a:b + 1] = base[None] + walk
        carrier = np.repeat(rng.integers(0, S, T // 15 + 1), 15)[:T]
        ball[a:b + 1] = P_[np.arange(a, b + 1), carrier] + 0.5
    y = np.full(n, 3, np.int8)
    for _, a, b in blocks:
        y[a:b + 1] = rng.integers(0, 3, b - a + 1)
    return Match(P=P_, team=team, speed=np.full((n, S), np.nan, np.float32), ball=ball, half_blocks=blocks,
                 attack_dir={1: 1, 2: -1}, y=y, pitch=(105.0, 68.0), fps=fps, name=f"synthetic{seed}",
                 source="synthetic")


def simulated(cleans: list, severity: str) -> list:
    return [corrupt(m, CONFIG[severity], P.seed_for(m.name, severity)) for m in cleans]


def report(real: Match, cleans: list, detail: bool, top_k: int = 5, alarm_only: bool = False) -> bool:
    """Per severity: AUC of real versus the degraded clean matches, and the reference AUC of one
    degraded clean match versus the others (same world, different match). Window statistics also
    differ between matches for tactical reasons, so only an AUC well above the reference says the
    noise differs. Returns the alarm: in every severity the AUC exceeds its reference by more than
    ``ALARM_MARGIN``. With ``alarm_only`` (the target) it prints that bit and nothing per severity."""
    if alarm_only and detail:
        raise SystemExit("--detail is refused on the target")
    gaps = []
    for sev in SEVERITIES:
        t = time.time()
        sims = simulated(cleans, sev)
        r = domain_gap(real, sims, top_k=top_k)
        ref = domain_gap(sims[0], sims[1:], top_k=top_k)
        gaps.append(r["auc"] - ref["auc"])
        if alarm_only:
            continue
        print(f"{sev:7s} AUC {r['auc']:.3f} +- {r['auc_std']:.3f} (reference, match vs match in the same world: "
              f"{ref['auc']:.3f})  windows real {r['n_real_windows']}, sim {r['n_sim_windows']}  "
              f"({time.time() - t:.0f}s)")
        print("        gives it away: " + ", ".join(f"{a} {b:.2f}" for a, b in r["top_features"]))
        if detail:
            Xr, names, _ = window_features(real)
            Xs = np.vstack([window_features(s)[0] for s in sims])
            for a, _ in r["top_features"]:
                j = names.index(a)
                print(f"        {a:24s} median real {np.nanmedian(Xr[:, j]):.3g}  sim {np.nanmedian(Xs[:, j]):.3g}")
    alarm = min(gaps) > ALARM_MARGIN
    print(f"alarm: {'ON' if alarm else 'off'}  (ON = in every severity the real side is easier to tell from the "
          f"simulator than one simulated match from another, by more than {ALARM_MARGIN:g} AUC)")
    return alarm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", action="store_true", help="the competition match (humans only)")
    src.add_argument("--real", help="an external match as the real side, e.g. skillcorner/1874553")
    src.add_argument("--synthetic", action="store_true", help="smoke test, no data at all")
    ap.add_argument("--clean", nargs="+", default=list(CLEAN_DEFAULT), help="clean IDSSE matches to degrade")
    ap.add_argument("--detail", action="store_true", help="also print medians of the top statistics")
    a = ap.parse_args()
    if a.target and a.detail:
        raise SystemExit("--detail is refused on the target: it prints the statistics a degrader fit would match")
    t0 = time.time()
    if a.synthetic:
        print("synthetic smoke test")
        real = corrupt(synthetic_clean(99), CONFIG["target"], seed=7)
        report(real, [synthetic_clean(1), synthetic_clean(2), synthetic_clean(3)], a.detail)
        return 0
    locked = {("idsse", P.LOCKED_IDSSE), ("metrica", P.LOCKED_METRICA)}
    if a.real and tuple(a.real.split("/", 1)) in locked:
        raise SystemExit("the locked benches are spent: use development matches")
    cleans = []
    for key in a.clean:
        ds, mid = key.split("/", 1)
        if (ds, mid) in locked:
            raise SystemExit("the locked benches are spent: use development matches")
        cleans.append(downsample(load_match(P.CANON / ds / f"{mid}.npz")))
    if a.real:
        ds, mid = a.real.split("/", 1)
        print(f"real side: external {a.real} (5 Hz)")
        report(downsample(load_match(P.CANON / ds / f"{mid}.npz")), cleans, a.detail)
        return 0
    print("Output holds one alarm bit and a runtime. Safe to paste.\n")
    try:
        from katena.target.atd import load_target
        real = downsample(load_target())
        print(f"real side: target, {real.n} samples at {real.fps:g} Hz, {len(real.half_blocks)} halves")
        report(real, cleans, False, alarm_only=True)
    except Exception as error:  # noqa: BLE001  the type alone, as messages can quote the file
        print(f"raised {type(error).__name__}")
        return 1
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
