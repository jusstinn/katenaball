"""Build the D4 feature cache: corrupted, featurized copies of the development matches.

    python scripts/model_cache.py [--workers 3]

External matches only (never the locked two, never data/). Idempotent: cached copies are kept.
Per development match: 5 severity-grid training copies, 4 extra default-config training copies
(for the no-severity-augmentation ablation) and 1 evaluation copy. Cache location:
`$KATENA_MODEL_CACHE`, default `<external_data>/derived/d4_feats/`.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")]

from katena.model import data as D  # noqa: E402


def full_plan(names=D.DEV_NAMES) -> dict:
    plan = {}
    for name in names:
        specs = D.train_plan(name, "severity") + D.train_plan(name, "default") + [D.eval_spec(name)]
        uniq = {s.key(): s for s in specs}
        plan[name] = list(uniq.values())
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args(argv)
    names = D.check_names(args.names or D.DEV_NAMES)
    t = time.time()
    plan = full_plan(names)
    print(f"{sum(len(v) for v in plan.values())} copies over {len(plan)} matches")
    D.ensure_all(plan, workers=args.workers)
    print(f"cache ready in {time.time() - t:.0f} s: {D.default_cache_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
