#!/usr/bin/env python
"""D1 early read on the 6 IDSSE development matches only (no Metrica or SkillCorner, no locked bench, no
``data/``). Does a native-frame-rate touch group help, and is the native rate what helps?

    KATENA_EXTERNAL=... KATENA_WORK=... uv run python scripts/probe_touch.py

For each match and condition (native, mild, target, severe, and the evaluation-only smooth-ball world of
``katena.eval.touch_gate``) it rebuilds the base 5 Hz features exactly as ``katena.eval.corpus.build_one``
does (same degrader seeds) and two copies of the touch group: from the native 25 Hz ball lifted from the
5 Hz corruption (``touch25``), and from the 5 Hz ball itself (``touch5``: the same code with no native
frames). Then 6-fold IDSSE-only LOMO with M2's heads and Viterbi at tau = 2:

* ``noise``: trained on mild, target and severe rows of the other 5 matches (as M2), scored at target-like,
  severe and smooth-ball;
* ``clean``: trained on native rows, scored on native input (the hypothesis test without noise);
* ``mixed``: ``noise`` plus smooth-ball rows (training across ball-jitter worlds), same scoring.

Variants: base (M2's 160 features), base + touch25, base + touch5. Paired bootstrap as in the gate. This
is a screen, not the gate (5 training matches instead of 27); it adopts nothing on its own.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

import possession_metrics as pm  # noqa: E402
from katena.decode import hmm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.eval.bootstrap import adopted, paired_bootstrap  # noqa: E402
from katena.eval.score import score, to_native_codes  # noqa: E402
from katena.featurize import touch as TF  # noqa: E402
from katena.featurize.build import featurize, infer_dirs  # noqa: E402
from katena.grid import downsample  # noqa: E402
from katena.labels import relabel  # noqa: E402
from katena.types import load_match  # noqa: E402

T, TAU, STRIDE = 2, 2.0, 4
SM = TG.SMOOTH
CONDS = ("native", "mild", "target", "severe")
EXTRA = (SM,)
OUT = P.WORK / "touch_probe"
EDGES = [0, 2, 4, 8, 16, 32, 1e9]
SETTINGS = (("noise", ("mild", "target", "severe"), ("target", "severe", SM)),
            ("clean", ("native",), ("native",)),
            ("mixed", ("mild", "target", "severe", SM), ("target", "severe", SM)))
VARIANTS = ("base", "touch25", "touch5")


def dev_idsse() -> list:
    dev, locked = P.split(P.all_refs())
    assert all(r.id != "DFL-MAT-J03WQQ" for r in dev)
    return [r for r in dev if r.dataset == "idsse"]


TOUCH_SRC = hashlib.sha256(Path(TF.__file__).read_bytes()).hexdigest()[:16]   # rebuild touch columns on change


def _base_arrays(ref, m, m5, c) -> dict:
    mc, _ = TG.corrupted(ref, m, m5, c)
    dirs = infer_dirs(mc)
    f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
    return {f"{c}__X0": f0.X, f"{c}__X1": f1.X, "idx": f0.idx, "half": f0.half}


def _touch_arrays(ref, m, m5, c) -> dict:
    step = int(m5.meta["step"])
    mc, lifted = TG.corrupted(ref, m, m5, c)
    t25, names, idx = TF.touch_features(mc, TF.NativeBall(lifted, m.fps, step))
    t5, _, _ = TF.touch_features(mc, TF.NativeBall(mc.ball, mc.fps, 1))
    return {f"{c}__A0": t25[0], f"{c}__A1": t25[1], f"{c}__B0": t5[0], f"{c}__B1": t5[1], f"{c}__tidx": idx,
            "touch_names": np.array(names)}


def build(ref) -> None:
    """Base columns per condition (cached once) and touch columns (rebuilt whenever touch.py changes)."""
    OUT.mkdir(parents=True, exist_ok=True)
    main_p, extra_p, touch_p = OUT / f"{ref.id}.npz", OUT / f"{ref.id}__extra.npz", OUT / f"{ref.id}__touch.npz"
    fresh = touch_p.exists() and str(np.load(touch_p)["src"]) == TOUCH_SRC
    if main_p.exists() and extra_p.exists() and fresh:
        return
    t0 = time.time()
    m = load_match(ref.path)
    m5 = downsample(m)
    if not main_p.exists():
        arrays = {"y_native": relabel(m, T).y, "step": int(m5.meta["step"]), "n_native": m.n}
        for c in CONDS:
            arrays.update(_base_arrays(ref, m, m5, c))
        np.savez(main_p, **arrays)
    if not extra_p.exists():
        arrays = {}
        for c in EXTRA:
            arrays.update(_base_arrays(ref, m, m5, c))
        np.savez(extra_p, **arrays)
    if not fresh:
        arrays = {"src": np.array(TOUCH_SRC)}
        for c in CONDS + EXTRA:
            arrays.update(_touch_arrays(ref, m, m5, c))
        np.savez(touch_p, **arrays)
    print(f"built {ref.id} in {time.time() - t0:.0f}s", flush=True)


def load(ref) -> dict:
    out = {}
    for p in (OUT / f"{ref.id}.npz", OUT / f"{ref.id}__extra.npz", OUT / f"{ref.id}__touch.npz"):
        with np.load(p) as z:
            out.update({k: z[k] for k in z.files})      # touch columns override any older cached copy
    for c in CONDS + EXTRA:
        assert np.array_equal(out[f"{c}__tidx"], out["idx"])
    return out


def X_of(d: dict, c: str, focal: int, variant: str) -> np.ndarray:
    parts = [d[f"{c}__X{focal}"]]
    if variant == "touch25":
        parts.append(d[f"{c}__A{focal}"])
    if variant == "touch5":
        parts.append(d[f"{c}__B{focal}"])
    return np.concatenate(parts, 1)


def rows(d: dict, conds: tuple, variant: str, seed: int):
    y5 = d["y_native"][::int(d["step"])][d["idx"]]
    rng = np.random.default_rng(seed)
    Xs, os_, ys = [], [], []
    for ci, c in enumerate(conds):
        sel = np.flatnonzero((np.arange(len(d["idx"])) + ci) % STRIDE == 0)
        o = rng.integers(0, 2, len(sel)).astype(np.int8)
        X0, X1 = X_of(d, c, 0, variant)[sel], X_of(d, c, 1, variant)[sel]
        Xs.append(np.where(o[:, None] == 1, X1, X0))
        os_.append(o)
        ys.append(y5[sel])
    return np.concatenate(Xs), np.concatenate(os_), np.concatenate(ys)


def score_match(d: dict, rows_: np.ndarray) -> dict:
    pred = to_native_codes(rows_, d["idx"], int(d["step"]), int(d["n_native"]))
    return score(d["y_native"], pred)


def length_recall(d: dict, s: dict) -> np.ndarray:
    runs = pm.valid_runs(d["y_native"])
    gt_sp = pm.possession_spells(d["y_native"], runs)
    dur = np.array([(b - a) / 25.0 for a, b, _ in gt_sp])
    fl = s["flags"][list(pm.SPELL_IOUS).index(0.5)]
    out = np.zeros((len(EDGES) - 1, 2))
    for i in range(len(EDGES) - 1):
        m = (dur >= EDGES[i]) & (dur < EDGES[i + 1])
        out[i] = (m.sum(), fl[m].sum())
    return out


def main() -> None:
    refs = dev_idsse()
    print("IDSSE development folds:", [r.id for r in refs], flush=True)
    for r in refs:
        build(r)
    data = {r.id: load(r) for r in refs}
    y5s = {k: d["y_native"][::int(d["step"])] for k, d in data.items()}
    res = {}
    t0 = time.time()
    for setting, train_conds, eval_conds in SETTINGS:
        for v in VARIANTS:
            for r in refs:
                others = [k for k in data if k != r.id]
                parts = [rows(data[k], train_conds, v, P.seed_for(k, "pool")) for k in others]
                X = np.concatenate([p[0] for p in parts])
                o = np.concatenate([p[1] for p in parts])
                y = np.concatenate([p[2] for p in parts])
                heads = gbdt.fit_heads(X, o, y, np.ones(len(y), bool), np.arange(X.shape[1]))
                pri = hmm.fit_priors([y5s[k] for k in others])
                d = data[r.id]
                for c in eval_conds:
                    pa, ph = heads.predict(X_of(d, c, 0, v), X_of(d, c, 1, v))
                    res[(setting, v, c, r.id)] = score_match(d, hmm.decode(pa, ph, d["half"], pri, TAU, "viterbi"))
            print(f"[{time.time() - t0:.0f}s] {setting} {v} done", flush=True)
    pickle.dump(res, open(OUT / "results.pkl", "wb"))
    keys = [r.id for r in refs]
    print("\nspellF1 per match (mean), pred/GT possessions")
    for setting, _, eval_conds in SETTINGS:
        for c in eval_conds:
            for v in VARIANTS:
                s = [res[(setting, v, c, k)] for k in keys]
                print(f"{setting:5s} {c:17s} {v:8s} mean {np.mean([x['spell_f1'] for x in s]):.4f}  "
                      f"pred/GT {np.sum([x['n_pred'] for x in s]) / np.sum([x['n_gt'] for x in s]):.3f}  "
                      + " ".join(f"{x['spell_f1']:.3f}" for x in s))
            for a, b in (("touch25", "base"), ("touch5", "base"), ("touch25", "touch5")):
                bs = paired_bootstrap([res[(setting, a, c, k)] for k in keys], [res[(setting, b, c, k)] for k in keys])
                print(f"   {a} vs {b}: {bs['mean_diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}] wins {bs['win_rate']:.0%}"
                      f" adopted-rule {adopted(bs)}")
            print("   recall at IoU 0.5 by true spell length (s): "
                  + " ".join(f"{EDGES[i]:g}-{EDGES[i + 1]:g}" for i in range(len(EDGES) - 1)))
            for v in VARIANTS:
                tab = sum(length_recall(data[k], res[(setting, v, c, k)]) for k in keys)
                print(f"   {v:8s} " + " ".join(f"{h / max(n, 1):.3f}" for n, h in tab))
    for c in ("target", "severe", SM):                # does training across jitter worlds cost the M2 baseline?
        for v in VARIANTS:
            bs = paired_bootstrap([res[("mixed", v, c, k)] for k in keys], [res[("noise", v, c, k)] for k in keys])
            print(f"mixed vs noise training, {v:8s} on {c:17s}: {bs['mean_diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}]"
                  f" wins {bs['win_rate']:.0%}")


if __name__ == "__main__":
    main()
