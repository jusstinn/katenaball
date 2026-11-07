"""End-to-end MVP submission: target match -> features -> D4 model -> placeholder decoder -> CSV.

HUMAN-RUN on the real match. It prints aggregates only (row count, label shares, possession count
and mean duration, runtime, output hash), never frame-level values.

    python scripts/predict_target.py                       # real match -> evaluation/results/katenaball.csv
    python scripts/predict_target.py --synthetic           # smoke test: corrupted synthetic match, temp path

Pipeline: `katena.io.load_target_match()` -> `featurize` (every `predict_stride`-th graded frame,
the stride the model was validated with) -> `Model.load(artifacts/katena_model.json)` ->
`predict_logp` -> `decode_simple` with the decoder parameters stored in the model (chosen by nested
CV on external matches) -> `frame,label` for every frame, X outside the graded halves. Then the
official `validate_submission` check on the written file. Deterministic: fixed threads and seeds,
log-probs rounded before decoding.

The decoder is a PLACEHOLDER until J3's semi-Markov decoder lands. There is no kickoff detection
yet (J5), so the pre-kickoff dead ball of the second half is whatever the model says.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")]

MODEL = os.path.join(ROOT, "artifacts", "katena_model.json")
OUT = os.path.join(ROOT, "evaluation", "results", "katenaball.csv")
N_TARGET = 199_319


def synthetic_target(half_minutes: float, seed: int = 7):
    """A corrupted synthetic match standing in for the target (labels dropped)."""
    from katena.noise import corrupt
    from katena.synth import synthetic_match
    m = corrupt(synthetic_match(seed=seed, half_minutes=half_minutes), {}, seed=seed + 1)
    m.y = None
    return m


def write_csv(path: str, codes: np.ndarray) -> None:
    letters = np.array(["H", "A", "D", "X"])[codes.astype(np.int64)]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n", encoding="ascii") as fh:
        fh.write("frame,label\n")
        fh.write("".join(f"{i},{c}\n" for i, c in enumerate(letters)))


def summarise(path: str, n_expected: int, fps: int = 25) -> dict:
    """Official validation plus the aggregates validate_submission.py prints."""
    import possession_metrics as pm
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_expected)
    if reason:
        return {"valid": False, "reason": reason}
    codes = pm.label_codes(table)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"valid": True, "rows": int(len(codes)),
            "shares": {name: float(np.mean(codes == i)) for i, name in enumerate(pm.LABELS)},
            "n_possessions": len(spells),
            "mean_possession_s": (n_alive / float(fps) / len(spells)) if spells else 0.0,
            "sha256": digest}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=None, help=f"default {OUT}; a temp file with --synthetic")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke test on a corrupted synthetic match instead of the target")
    ap.add_argument("--synthetic-half-minutes", type=float, default=3.0)
    args = ap.parse_args(argv)
    t0 = time.time()

    from katena.feats import featurize
    from katena.model import Model, decode_simple

    if not os.path.exists(args.model):
        print(f"FAIL  no trained model at {args.model}; run scripts/train_model.py first")
        return 2
    model = Model.load(args.model)
    stride = int(model.info.get("predict_stride", 5))
    decode_cfg = model.info.get("decode")
    if not decode_cfg:
        print("FAIL  the model carries no decoder parameters; retrain with scripts/train_model.py")
        return 2

    if args.synthetic:
        m = synthetic_target(args.synthetic_half_minutes)
        n_expected = m.n
        out = args.out or os.path.join(tempfile.mkdtemp(prefix="katena_smoke_"), "katenaball_synthetic.csv")
        print("SYNTHETIC smoke test (no competition data is read)")
    else:
        from katena.io import load_target_match  # FIREWALL: real match; aggregates only below
        m = load_target_match(args.data_dir)
        n_expected = N_TARGET
        out = args.out or OUT
    t_load = time.time() - t0

    feats = featurize(m, {"stride": stride})
    t_feat = time.time() - t0 - t_load
    logp = model.predict_logp(feats)
    codes = decode_simple(logp, feats.t_frame, m.half_blocks, m.n, decode_cfg)
    graded = m.graded_mask()
    if (codes[graded] == 3).any() or (codes[~graded] != 3).any():
        print("FAIL  decoder output has X on a graded frame or a label outside the halves")
        return 1
    write_csv(out, codes)
    s = summarise(out, n_expected)
    dt = time.time() - t0

    if not s["valid"]:
        print(f"FAIL  {s['reason']}")
        return 1
    print(f"OK    {out}")
    print(f"      model {os.path.basename(args.model)} ({model.kind}, {len(model.columns)} features, "
          f"stride {stride}), decoder {decode_cfg} [placeholder]")
    print(f"      rows {s['rows']}  (expected {n_expected})")
    print("      label shares  " + "  ".join(f"{k} {v:.3f}" for k, v in s["shares"].items()))
    print(f"      possessions {s['n_possessions']}, mean {s['mean_possession_s']:.1f} s")
    print(f"      runtime {dt:.1f} s (load {t_load:.1f} s, features {t_feat:.1f} s)")
    print(f"      sha256 {s['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
