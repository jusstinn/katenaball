#!/usr/bin/env python
"""Byte-identity check for D3 on CPU, on one external development fold.

Each ``run`` is a fresh process that trains both members on every other development match (LightGBM
heads and the TCN), predicts the held-out match, blends at w = 0.5, decodes with Viterbi and writes
every output with its SHA-256. Two runs on one machine must print identical hashes:

    python scripts/d3_determinism.py run --out D:/katena/work-d3/det/a.npz
    python scripts/d3_determinism.py run --out D:/katena/work-d3/det/b.npz
    python scripts/d3_determinism.py compare D:/katena/work-d3/det/a.npz D:/katena/work-d3/det/b.npz

Probes of what cross-machine identity needs: ``--threads`` (another torch thread count), ``--env K=V``
(set before torch loads, e.g. ATEN_CPU_CAPABILITY=default or ONEDNN_MAX_CPU_ISA=SSE41 to imitate another
CPU), ``--save-model`` / ``--load-model`` (the same weights, inference only). Each run also decodes
copies of the probabilities moved by one float32 ulp, with and without rounding, to test whether
rounding before decoding makes the labels robust. External matches only; never ``data/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

QUANT = {"none": None, "p16": ("p", 16), "p12": ("p", 12), "p8": ("p", 8), "z0.25": ("z", 0.25), "z1": ("z", 1.0)}
CONDS = ("target", "severe")
TAU = 2.0                     # fixed for the check (the M2 plateau value)


def _quant(E, p, q):
    if q is None:
        return p
    kind, v = q
    return E.quantize(p, int(v)) if kind == "p" else E.quantize_logit(p, v)


def run(a) -> None:
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    from katena.models import ensemble as E
    E.pin_cpu_env()
    from katena.models import gbdt
    gbdt.ensure_openmp()
    import numpy as np
    import torch

    from katena.decode import hmm
    from katena.eval import bakeoff as B
    from katena.eval import corpus as C
    from katena.eval import d3
    from katena.eval import lomo as Lm
    from katena.eval import protocol as P
    from katena.eval import stages as S
    from katena.eval.score import to_native_codes
    from katena.models import tcn

    T = P.T_MAIN
    dev, locked = B.dev_and_locked()
    if a.datasets:
        dev = [r for r in dev if r.dataset in a.datasets]
    ref = next(r for r in dev if r.key == a.fold) if a.fold else S.FOLDS8(dev)[0]
    assert ref.key not in {r.key for r in locked}
    env = {k: os.environ.get(k) for k in sorted(E.CPU_ENV)}
    print(f"fold {ref.key} | {len(dev)} development matches | threads {a.threads} | env {env}", flush=True)
    out = {"fold": ref.key, "threads": a.threads, "env": env, "torch": torch.__version__,
           "cpu_capability": torch.backends.cpu.get_cpu_capability(), "t": {}, "sha": {}, "metrics": {}}
    arrays = {}
    pools = Pools = None  # noqa: F841  (named for clarity below)
    t0 = time.time()
    pools = Lm.Pools(dev)
    out["t"]["pools_s"] = time.time() - t0

    t0 = time.time()
    heads = pools.fit({ref.key}, T, pools.cols())
    out["t"]["gbdt_fit_s"] = time.time() - t0
    feats = {c: C.load_feat(ref, c) for c in CONDS}
    for c in CONDS:
        arrays[f"g_pa_{c}"], arrays[f"g_ph_{c}"] = heads.predict(feats[c]["X0"], feats[c]["X1"])
    cached = Lm.oof_path("full", T, ref)
    if cached.exists():
        oof = Lm.load_oof("full", T, ref)
        out["gbdt_equals_bakeoff_cache"] = all(np.array_equal(arrays[f"g_{h}_{c}"], oof[f"{h}_{c}"])
                                               for c in CONDS for h in ("pa", "ph"))

    t0 = time.time()
    if a.load_model:
        blob = torch.load(a.load_model, weights_only=False)
        model = tcn.TCN(blob["n_in"])
        model.load_state_dict(blob["state"])
        model.eval()
        sc = blob["scaler"]
        out["t"]["tcn_train_s"] = 0.0
    else:
        seqs = d3.tcn_seqs(dev, T)
        out["t"]["tcn_seqs_s"] = time.time() - t0
        t0 = time.time()
        model, sc = d3.fit_tcn_excluding(pools, seqs, {ref.key}, threads=a.threads, steps=a.steps)
        out["t"]["tcn_train_s"] = time.time() - t0
        del seqs
        if a.save_model:
            Path(a.save_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"n_in": int(feats["target"]["X0"].shape[1]), "state": model.state_dict(), "scaler": sc},
                       a.save_model)
    out["sha"]["tcn_weights"] = E.model_hash(model)
    t0 = time.time()
    pt = d3.predict_tcn(model, sc, ref, CONDS, a.threads)
    out["t"]["tcn_predict_s"] = time.time() - t0
    for c in CONDS:
        arrays[f"t_pa_{c}"], arrays[f"t_ph_{c}"] = pt[f"pa_{c}"], pt[f"ph_{c}"]

    pri = B.priors_for(pools, {ref.key}, T)
    gt = C.load_gt(ref, T)
    rng = np.random.default_rng(12345)
    for c in CONDS:
        f = feats[c]
        pa, ph = E.blend_heads((arrays[f"g_pa_{c}"], arrays[f"g_ph_{c}"]), (arrays[f"t_pa_{c}"], arrays[f"t_ph_{c}"]),
                               E.W_MAIN)
        arrays[f"e_pa_{c}"], arrays[f"e_ph_{c}"] = pa, ph
        # one-ulp moves, the size of a last-bit difference between two machines' inference
        up = rng.random((2, len(pa))) < 0.5
        pa_u = np.where(up[0], np.nextafter(pa, np.float32(1)), np.nextafter(pa, np.float32(0))).astype(np.float32)
        ph_u = np.where(up[1], np.nextafter(ph, np.float32(1)), np.nextafter(ph, np.float32(0))).astype(np.float32)
        for qn, q in QUANT.items():
            rows = hmm.decode(_quant(E, pa, q), _quant(E, ph, q), f["half"], pri, TAU, "viterbi")
            rows_u = hmm.decode(_quant(E, pa_u, q), _quant(E, ph_u, q), f["half"], pri, TAU, "viterbi")
            arrays[f"rows_{qn}_{c}"] = rows
            out["metrics"][f"ulp_moved_samples_{qn}_{c}"] = int((rows != rows_u).sum())
            if qn in ("none", "p16", "z0.25"):
                nat = to_native_codes(rows, f["idx"], gt["step"], gt["n_native"])
                s = Lm.score_rows(ref, T, c, rows, f["idx"], gt)
                out["metrics"][f"spell_f1_{qn}_{c}"] = s["spell_f1"]
                out["sha"][f"labels_native_{qn}_{c}"] = E.sha256_arrays(nat)
        for m in ("g", "t"):
            out["sha"][f"{m}_probs_{c}"] = E.sha256_arrays(arrays[f"{m}_pa_{c}"], arrays[f"{m}_ph_{c}"])
        out["sha"][f"blend_probs_{c}"] = E.sha256_arrays(pa, ph)
        out["sha"][f"labels_{c}"] = E.sha256_arrays(arrays[f"rows_none_{c}"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, **arrays)
        Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def compare(a) -> None:
    import numpy as np
    za, zb = np.load(a.a), np.load(a.b)
    ja = json.loads(Path(a.a).with_suffix(".json").read_text())
    jb = json.loads(Path(a.b).with_suffix(".json").read_text())
    print("hashes identical:", {k: ja["sha"][k] == jb["sha"].get(k) for k in ja["sha"]})
    for k in sorted(za.files):
        x, y = za[k], zb[k]
        if k.startswith("rows_"):
            print(f"{k}: {int((x != y).sum())} of {len(x)} samples differ")
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            print(f"{k}: bytes equal {x.tobytes() == y.tobytes()}, max |diff| {d.max():.3g}, mean |diff| {d.mean():.3g}")
    for k in sorted(ja["metrics"]):
        if k.startswith("spell_f1"):
            print(f"{k}: {ja['metrics'][k]:.5f} vs {jb['metrics'][k]:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fold", default="", help="development match key, default the first IDSSE/Metrica fold")
    r.add_argument("--datasets", nargs="*", default=[], help="restrict the development set (prototype caches)")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--steps", type=int, default=None)
    r.add_argument("--env", nargs="*", default=[], help="KEY=VALUE set before torch is imported")
    r.add_argument("--save-model", default="")
    r.add_argument("--load-model", default="")
    r.add_argument("--out", default="")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
