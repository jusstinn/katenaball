"""Fit the D2 noise simulator to the competition match. A HUMAN runs this on the real data.

It loads the target match (Metrica ATD, no labels), measures label-free noise statistics, fits the
corruption config by moment matching against a clean base match, optionally runs the real-vs-sim
domain-gap check, and writes the config to artifacts/noise_cfg.json (gitignored).

Data firewall: everything printed is an aggregate (counts, rates, quantiles, shapes, runtimes, an
AUC, feature names). Nothing per-frame, no rows, no tracks. On the real match an exception prints
only its type and the step it happened in, because a message or traceback can carry values; pass
--debug to see the full traceback locally, and do not paste it anywhere hosted.

    python scripts/noise_fit_target.py --clean <clean external match>      # real fit (human only)
    python scripts/noise_fit_target.py --clean <...> --domain-gap          # plus real-vs-sim AUC
    python scripts/noise_fit_target.py --synthetic --out <path>            # smoke test, no real data

Without --clean the base is a synthetic match, which is fine for a smoke test but biases the jitter
and speed parameters; use an external clean match (D1 corpus) for the real fit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from katena.match import Match  # noqa: E402
from katena.noise import (DEFAULT_CFG, corrupt, domain_gap, fit_noise_cfg,  # noqa: E402
                          target_stats)
from katena.synth import synthetic_match  # noqa: E402

PENDING = ("katena.io.load_target_match() is not available yet: the target-match loader is pending "
           "J1/D1. Once it lands, re-run this script. Use --synthetic for a smoke test meanwhile.")

# a config unlike the defaults, so the synthetic smoke test has something to recover
SMOKE_TRUTH = dict(DEFAULT_CFG, ball_present_rate=0.65, players_present_mean=19.8, ball_gap_median_s=1.2,
                   frag_median_s=5.0, pos_jitter_sd_m=0.12, ball_iso_drop_share=0.2, slot_policy="lru")


def _load_target(args) -> Match:
    if args.synthetic:
        clean = synthetic_match(seed=101, half_minutes=args.synthetic_half_minutes)
        return corrupt(clean, SMOKE_TRUTH, seed=7)
    try:
        from katena.io import load_target_match  # noqa: PLC0415
    except (ImportError, AttributeError):
        print(PENDING)
        sys.exit(2)
    return load_target_match()


def _load_clean(args) -> Match:
    if not args.clean:
        return synthetic_match(seed=0, half_minutes=args.synthetic_half_minutes)
    # the D1 corpus cache first, then a generic loader, then a plain reader of the same .npz layout
    for mod, fn in (("katena.external.common", "load_match_npz"), ("katena.io", "load_match")):
        try:
            loader = getattr(importlib.import_module(mod), fn)
        except (ImportError, AttributeError):
            continue
        return loader(args.clean)
    with np.load(args.clean, allow_pickle=False) as z:
        m = Match(P=z["P"].astype(np.float32), team=z["team"].astype(np.int8),
                  speed=z["speed"].astype(np.float32), ball=z["ball"].astype(np.float32),
                  half_blocks=[tuple(int(v) for v in row) for row in z["half_blocks"]],
                  attack_dir={int(h): int(d) for h, d in z["attack_dir"]},
                  y=z["y"].astype(np.int8) if "y" in z.files else None,
                  name=Path(args.clean).stem, fps=int(z["fps"]) if "fps" in z.files else 25,
                  pitch=tuple(float(v) for v in z["pitch"]) if "pitch" in z.files else (105.3, 68.0))
    m.validate()
    return m


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4g}"
    return str(v)


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.generic):
        return _json_safe(o.item())
    return o


def run(args) -> int:
    t0 = time.time()
    label = "SYNTHETIC smoke test" if args.synthetic else "TARGET match (aggregates only)"
    print(f"== noise_fit_target: {label}")

    step = "load target"
    try:
        target = _load_target(args)
        print(f"target: frames={target.n} slots={target.n_slots} "
              f"graded={int(target.graded_mask().sum())} halves={len(target.half_blocks)}")

        step = "target_stats"
        t = time.time()
        tgt = target_stats(target)
        print(f"\n-- target_stats ({time.time() - t:.1f} s)")
        for k in sorted(tgt):
            print(f"  {k:32s} {_fmt(tgt[k])}")

        step = "load clean base"
        clean = _load_clean(args)
        print(f"\nclean base: {'synthetic' if not args.clean else Path(args.clean).name} "
              f"frames={clean.n} slots={clean.n_slots}")
        if not args.clean:
            print("  WARNING: synthetic base; pass --clean <external match> for a real fit")

        step = "fit_noise_cfg"
        t = time.time()
        cfg = fit_noise_cfg(target, clean, max_evals=args.max_evals,
                            max_graded_minutes=args.max_minutes, seed=args.seed, verbose=True)
        fit = cfg["_fit"]
        print(f"\n-- fit ({time.time() - t:.1f} s, {fit['n_evals']} evals): "
              f"loss {fit['loss_init']:.4g} -> {fit['loss_final']:.4g}")
        print(f"  {'stat':32s} {'target':>10s} {'sim':>10s}")
        for k, (a, b) in fit["stats"].items():
            print(f"  {k:32s} {_fmt(a):>10s} {_fmt(b):>10s}")
        print("\n-- fitted config")
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            note = "" if v == DEFAULT_CFG.get(k) else f"   (default {_fmt(DEFAULT_CFG.get(k))})"
            if args.synthetic and k in SMOKE_TRUTH and SMOKE_TRUTH[k] != DEFAULT_CFG.get(k):
                note += f"   [smoke truth {_fmt(SMOKE_TRUTH[k])}]"
            print(f"  {k:24s} {_fmt(v)}{note}")

        if args.domain_gap:
            step = "domain_gap"
            t = time.time()
            res = {}
            for nm, c in (("fitted", cfg), ("default", DEFAULT_CFG)):
                sim = corrupt(clean, c, seed=args.seed + 1)
                res[nm] = domain_gap(target, sim, window_s=args.window_s, seed=args.seed)
            print(f"\n-- domain gap, real vs sim windows ({time.time() - t:.1f} s; target AUC near 0.5)")
            for nm, r in res.items():
                print(f"  {nm:8s} AUC {r['auc']:.3f} +- {r['auc_std']:.3f}  "
                      f"(windows real {r['n_real_windows']}, sim {r['n_sim_windows']})")
                print("           top: " + ", ".join(f"{a} {b:.2f}" for a, b in r["top_features"][:6]))
            cfg["_fit"]["domain_gap"] = {nm: {"auc": r["auc"], "auc_std": r["auc_std"],
                                              "top_features": r["top_features"]} for nm, r in res.items()}

        step = "write"
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(cfg), indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out} ({time.time() - t0:.1f} s total)")
        return 0
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        if args.debug or args.synthetic:
            traceback.print_exc()
        else:
            print(f"FAILED during '{step}': {type(e).__name__} (message suppressed by the data firewall; "
                  f"re-run locally with --debug)")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="smoke test on a synthetic target")
    ap.add_argument("--clean", default=None, help="clean base match (external, with identities)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "noise_cfg.json"))
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=20.0, help="graded minutes of the base to use")
    ap.add_argument("--domain-gap", action="store_true", help="also run the real-vs-sim AUC check")
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-half-minutes", type=float, default=6.0)
    ap.add_argument("--debug", action="store_true", help="full tracebacks (local eyes only)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
