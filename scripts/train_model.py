"""Train the final D4 model on all 6 external development matches and save it to artifacts/.

    python scripts/train_model.py [--variant twohead] [--out artifacts/katena_model.json]

Never uses the 2 locked matches or the competition match. Calibration temperatures and the
placeholder decoder's parameters come from the same nested scheme as the LOMO study: out-of-fold
predictions over the development matches (k_inner groups), never from any target data. The saved
JSON carries the boosters, temperatures, decoder parameters, prediction stride and a record of the
training data (matches, noise configs, seeds, strides, code versions).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")]

from katena.feats import FEATURE_VERSION  # noqa: E402
from katena.model import data as D  # noqa: E402
from katena.model import lomo as L  # noqa: E402

FINAL_VARIANT = "twohead"
OUT = os.path.join(ROOT, "artifacts", "katena_model.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default=FINAL_VARIANT, choices=list(L.VARIANTS))
    ap.add_argument("--k-inner", type=int, default=L.K_INNER)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    t0 = time.time()
    v = L.VARIANTS[args.variant]
    names = list(D.DEV_NAMES)
    plan = {n: list({s.key(): s for s in D.train_plan(n, v.aug) + [D.eval_spec(n)]}.values())
            for n in names}
    D.ensure_all(plan, workers=args.workers)
    print(f"variant {v.name}: training on {names}")
    model = L.fit_variant(v, names)
    t_fit = time.time() - t0
    print(f"  fit: {model.info['n_train_rows']} rows, {len(model.columns)} features ({t_fit:.0f} s)")
    choice = L.calibrate_and_choose(v, model, names, args.k_inner)
    model.set_temperature(choice["temperature"])
    model.info.update({
        "decode": choice["decode"],
        "predict_stride": D.EVAL_STRIDE,
        "oof": {"spell_f1": choice["inner_spell_f1"], "macro_f1": choice["inner_macro_f1"],
                "nll_raw": choice["oof_nll_raw"], "nll_cal": choice["oof_nll_cal"],
                "k_inner": args.k_inner},
        "training_record": {
            "variant": v.name, "data_version": D.DATA_VERSION, "feature_version": FEATURE_VERSION,
            "harmonise": D.HARMONISE, "ball_fill": D.FILL_VERSION, "locked_never_used": list(D.LOCKED_NAMES),
            "copies": [{"match": s.match, "role": s.role, "index": s.index, "severity": s.severity,
                        "seed": s.seed, "stride": s.stride, "offset": s.offset, "cache_key": s.key()}
                       for n in names for s in D.train_plan(n, v.aug)],
            "team_swap_augmentation": True},
    })
    model.save(args.out)
    print(f"  temperature {choice['temperature']}, decoder {choice['decode']} "
          f"(OOF spellF1 {choice['inner_spell_f1']:.4f}, macroF1 {choice['inner_macro_f1']:.4f}; "
          f"NLL {choice['oof_nll_raw']:.4f} -> {choice['oof_nll_cal']:.4f})")
    print(f"saved {args.out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
