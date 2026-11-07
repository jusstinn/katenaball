"""Build and cache everything derived from a canonical match: ground truth under each T, corrupted
inputs under each severity, features for both focal teams, M0 predictions, and the training pool."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np

import possession_metrics as pm
from katena.eval import protocol as P
from katena.featurize.build import featurize, infer_dirs
from katena.grid import downsample
from katena.labels import relabel
from katena.models.baseline import m0_predict
from katena.sim import degrade as dg
from katena.types import load_match

warnings.filterwarnings("ignore")

POOL_STRIDE = 4
POOL_CONDS = ("mild", "target", "severe")
BASE_CONDS = ("native", "mild", "target", "severe")
EXTRA_CONDS = {                         # IDSSE only: clean ball (oracle) and ball loss tied to dead ball
    "target_cb": dataclasses.replace(dg.TARGET, name="target_cb", ball_missing=0.0, sigma_ball=0.0, false_ball_rate=0.0),
    "target_dead3": dg.with_dead_odds(dg.TARGET, 3.0),
    "target_dead15": dg.with_dead_odds(dg.TARGET, 15.0),
}
CONFIG = {"mild": dg.MILD, "target": dg.TARGET, "severe": dg.SEVERE, **EXTRA_CONDS}


def conds_for(ref: P.Ref) -> list[str]:
    return list(BASE_CONDS) + (list(EXTRA_CONDS) if ref.dataset == "idsse" else [])


def paths(ref: P.Ref) -> dict:
    d = P.WORK
    return {"gt": d / "gt" / f"{ref.dataset}_{ref.id}.npz", "pool": d / "pool" / f"{ref.dataset}_{ref.id}.npz",
            "feat": lambda c: d / "feat" / f"{ref.dataset}_{ref.id}__{c}.npz"}


def build_one(ref: P.Ref, force: bool = False) -> str:
    pth = paths(ref)
    conds = conds_for(ref)
    if not force and pth["gt"].exists() and pth["pool"].exists() and all(pth["feat"](c).exists() for c in conds):
        return ref.key
    for p in (pth["gt"], pth["pool"], pth["feat"]("x")):
        p.parent.mkdir(parents=True, exist_ok=True)
    m = load_match(ref.path)
    step = int(round(m.fps / 5.0))
    # ground truth under every T (native grid), plus work-grid labels for training
    gt = {"fps": m.fps, "step": step, "n_native": m.n,
          "blocks": np.array(m.half_blocks, dtype=np.int64)}
    y5 = {}
    for T in P.T_SWEEP:
        yT = relabel(m, T).y
        gt[f"y_T{T}"] = yT
        y5[T] = yT[::step]
    np.savez_compressed(pth["gt"], **gt)
    m5 = downsample(m)
    pool_X, pool_y, pool_o, pool_i = [], {T: [] for T in P.T_SWEEP}, [], []
    rng = np.random.default_rng(P.seed_for(ref.id, "pool"))
    for ci, c in enumerate(conds):
        mc = m5 if c == "native" else dg.corrupt(m5, CONFIG[c], P.seed_for(ref.id, c))
        dirs = infer_dirs(mc)
        f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
        m0 = m0_predict(mc)
        np.savez(pth["feat"](c), X0=f0.X, X1=f1.X, idx=f0.idx, half=f0.half, m0=m0,
                 names=np.array(f0.names), groups=f0.groups,
                 dirs=np.array([dirs[h] for h, _, _ in m5.half_blocks]),
                 deg=np.array([mc.meta.get("degrade", {}).get("ball_missing", np.nan),
                               mc.meta.get("degrade", {}).get("mean_objects", np.nan)]))
        if c in POOL_CONDS:
            sel = np.flatnonzero((np.arange(len(f0.idx)) + POOL_CONDS.index(c)) % POOL_STRIDE == 0)
            o = rng.integers(0, 2, len(sel)).astype(np.int8)
            pool_X.append(np.where(o[:, None] == 1, f1.X[sel], f0.X[sel]))
            pool_o.append(o)
            pool_i.append(f0.idx[sel])
            for T in P.T_SWEEP:
                pool_y[T].append(y5[T][f0.idx[sel]])
    np.savez(pth["pool"], X=np.concatenate(pool_X), o=np.concatenate(pool_o), idx=np.concatenate(pool_i),
             **{f"y_T{T}": np.concatenate(pool_y[T]) for T in P.T_SWEEP}, names=np.array(f0.names),
             groups=f0.groups)
    return ref.key


def load_gt(ref: P.Ref, T: int) -> dict:
    with np.load(paths(ref)["gt"]) as z:
        return {"y": z[f"y_T{T}"], "step": int(z["step"]), "fps": float(z["fps"]),
                "n_native": int(z["n_native"]), "blocks": z["blocks"]}


def load_feat(ref: P.Ref, cond: str) -> dict:
    with np.load(paths(ref)["feat"](cond)) as z:
        return {k: z[k] for k in z.files}


def load_pool(ref: P.Ref) -> dict:
    with np.load(paths(ref)["pool"]) as z:
        return {k: z[k] for k in z.files}


def build_all(workers: int = 6, force: bool = False) -> None:
    import multiprocessing as mp
    refs = P.all_refs()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for k in pool.imap_unordered(_job, [(r, force) for r in refs]):
            print("built", k, flush=True)


def _job(args):
    return build_one(*args)
