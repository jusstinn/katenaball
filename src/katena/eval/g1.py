"""Gate G1: F with one change, its TCN member reads M2's 160 columns plus D1's 53 ``touch`` columns (213).

Pre-registered in docs/decision-log.md (2026-11-10, "Gate G1: TCN reads the touch columns (pre-registration)",
commit 115f7e2) before any G1 score was computed; the candidate and the gate are fixed by
docs/plan-final-week.md (commit ff2e0b1, workstream W3). Everything else is F (``katena.eval.final``):

* ``G1T`` (G1 at w = 1): the M5 TCN unchanged apart from its input width, through ``ensemble.fit_tcn`` (seed 0,
  4 threads, 1200 steps, deterministic settings). Sequences as ``final.tcn_seqs`` (same order, same orientation
  draws), each row the world's 160 base columns then the same world's 53 touch columns in the same orientation.
  Scaler as ``final.fit_tcn`` (every 6th ``d2j`` pool row of the training matches), on all 213 columns.
* ``FG``: F's LightGBM member. Its stored out-of-fold probabilities are reused (copied from the F cache).
* ``G1``: ``ensemble.blend_heads(FG, G1T, 0.5)``, Viterbi at tau 2, priors without the held-out match, T = 2.

The cache is ``KATENA_WORK`` (its own). Under it:

    gt/ feat/ touch/ pool_d2j/ touch_pool_d2j/   copied from the F cache (``seed_cache``), SHA-256 checked
    oof/final_gbdt/T2/ oof/final_tcn/T2/         F's out-of-fold probabilities, copied
    final_results.pkl                            F's stored scores, copied (read only here)
    oof/g1_tcn/T2/                               G1's TCN member, out of fold, every evaluation world
    g1_results.pkl                               G1 and G1T scores; final_cost.jsonl: cost per stage (final.Cost)

Never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench. Logs are aggregates of
external matches only.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np

from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import d3
from katena.eval import final as F
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.models import ensemble as E

T = F.T
VAR_T = "g1_tcn"
NAMES = {F.W_F: "G1", 1.0: "G1T"}
F_NAMES = {F.W_F: "F", 0.0: "FG", 1.0: "FT"}
RES_NAME = "g1_results.pkl"
F_STORE = F.RES_NAME                   # F's stored scores, copied from the F cache
N_IN = 213                             # 160 base + 53 touch columns


# ---------------------------------------------------------------------------------------------
# seeding the cache from the F cache, byte-checked

def seed_plan(dev: list, final_root: Path) -> list:
    """(source, destination) of every input reused from the F cache: ground truth, the ``d2j`` pools and
    their touch pools, the base features and touch columns of every world, F's two out-of-fold
    probability sets and F's stored scores. Same relative path on both sides."""
    dst = []
    for r in dev:
        pth = C.paths(r)
        dst += [pth["gt"], d2.pool_path(r, F.MIX), F.touch_pool_path(r)]
        for w in F.feature_worlds(r):
            dst += [pth["feat"](w), TG.tpaths(r)["feat"](w)]
        dst += [Lm.oof_path(v, T, r) for v in (F.VAR_G, F.VAR_T)]
    dst.append(P.WORK / F_STORE)
    return [(Path(final_root) / p.relative_to(P.WORK), p) for p in dst]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def seed_cache(dev: list, final_root: Path) -> dict:
    """Copy the reused inputs into ``KATENA_WORK`` (never the other way), then check every copy against its
    source by SHA-256. Development matches only. Raises if any copy differs."""
    if F._own(final_root):
        raise RuntimeError("the G1 cache must not be the F cache")
    locked = {r.key for r in P.split(P.all_refs())[1]}
    assert not locked & {r.key for r in dev}
    plan = seed_plan(dev, final_root)
    n_copied, n_bytes, n_same, bad = 0, 0, 0, []
    for src, dst in plan:
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
            shutil.copyfile(src, tmp)
            tmp.replace(dst)
            n_copied += 1
        n_bytes += dst.stat().st_size
        if sha256_file(src) == sha256_file(dst):
            n_same += 1
        else:
            bad.append(str(dst.relative_to(P.WORK)))
    if bad:
        raise RuntimeError(f"{len(bad)} copies differ from their source, first {bad[:3]}")
    return {"planned": len(plan), "copied": n_copied, "identical": n_same, "gb": round(n_bytes / 2 ** 30, 2)}


# ---------------------------------------------------------------------------------------------
# TCN member on 213 columns

def tcn_seqs(dev: list, T: int = P.T_MAIN, seed: int = 0, mix: str = F.MIX) -> dict:
    """``final.tcn_seqs`` with each row's touch columns appended: the same keys, order, orientation draws and
    labels; ``X`` is the world's base columns then its touch columns in the drawn orientation (float16)."""
    rng = np.random.default_rng(P.SEED + seed)
    seqs = {}
    for r in dev:
        g = C.load_gt(r, T)
        y5 = g["y"][::g["step"]]
        for w in F.train_worlds(r.dataset, mix):
            o = int(rng.integers(0, 2))
            with np.load(C.paths(r)["feat"](w)) as z:
                X, idx = z["X1" if o else "X0"].astype(np.float16), z["idx"]
            t = TG.load_tfeat(r, w)
            if not np.array_equal(t["idx"], idx):
                raise RuntimeError(f"{r.key} {w}: touch rows do not line up with the base features")
            A = t["A1" if o else "A0"].astype(np.float16)
            seqs[(r.key, w)] = {"X": np.concatenate([X, A], 1), "y": y5[idx], "o": o,
                                "dead_ok": r.dataset != "skillcorner", "key": r.key}
    return seqs


class _Rows:
    """What ``final.fit_tcn`` reads from a pool (``keys`` and ``X``), without ``n_base``: the scaler then
    takes every column."""

    def __init__(self, keys: np.ndarray, X: np.ndarray):
        self.keys, self.X = keys, X


def scaler(pools: Lm.Pools, exclude: set):
    """``final.fit_tcn``'s scaler on every column of ``pools``: every 6th pooled row of the matches not in
    ``exclude``."""
    from katena.models import tcn
    rows = np.flatnonzero(~np.isin(pools.keys, list(exclude)))[::6]
    return tcn.Scaler(np.ascontiguousarray(pools.X[rows]))


def fit_tcn(sc, seqs: dict, exclude: set, seed: int = 0, threads: int = E.TCN_THREADS, steps: int | None = None):
    """``final.fit_tcn`` with the scaler fitted beforehand (``scaler``), so the pool need not stay in memory
    while the TCNs train. Same sequences, same lazy scaling, same ``ensemble.fit_tcn`` call."""
    tr = [{**s, "X": F.ScaledSeq(s["X"], sc)} for s in seqs.values() if s["key"] not in exclude]
    model = E.fit_tcn(tr, seed=seed, threads=threads, steps=steps)
    del tr
    return model, sc


def predict_tcn(model, sc, ref: P.Ref, conds, threads: int = E.TCN_THREADS) -> dict:
    """``d3.predict_tcn`` on each world's 213 columns (``touch_gate.features``)."""
    from katena.models import tcn
    E.set_deterministic(threads)
    out = {}
    for c in conds:
        f, X0, X1 = TG.features(ref, c)
        out[f"pa_{c}"], out[f"ph_{c}"] = tcn.predict(model, sc(X0), sc(X1), f["half"])
    return out


def stage_tcn(dev: list, folds: list, seed: int = 0, threads: int = E.TCN_THREADS, steps: int | None = None,
              variant: str = VAR_T) -> dict:
    """LOMO TCN member of G1 for every fold in ``folds`` (trained on the other development matches)."""
    todo = [r for r in folds if not Lm.oof_path(variant, T, r).exists()]
    if not todo:
        return {"folds": 0}
    t0 = time.time()
    pools = F.FinalPools(dev)                                    # 160 base + 53 touch columns, d2j rows
    n_cols = pools.X.shape[1]
    scs = {r.key: scaler(pools, {r.key}) for r in todo}
    del pools
    Lm.timed(f"pool and {len(scs)} scalers ({n_cols} columns)", t0)
    t0 = time.time()
    seqs = tcn_seqs(dev, T, seed)
    widths = {s["X"].shape[1] for s in seqs.values()}
    if widths != {n_cols}:
        raise RuntimeError(f"sequence widths {widths} differ from the pool's {n_cols} columns")
    Lm.timed(f"TCN sequences ({len(seqs)}, {sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB)", t0)
    for r in todo:
        t1 = time.time()
        model, sc = fit_tcn(scs[r.key], seqs, {r.key}, seed, threads, steps)
        t2 = time.time()
        arrays = predict_tcn(model, sc, r, F.eval_worlds(r), threads)
        arrays["model_sha"] = np.array(E.model_hash(model))
        arrays["n_in"] = np.array(model.inp.in_channels)
        d3._save(Lm.oof_path(variant, T, r), arrays)
        print(f"[{time.strftime('%H:%M:%S')}] G1 TCN fold {r.key}: train {t2 - t1:.0f}s, "
              f"predict {time.time() - t2:.0f}s", flush=True)
    return {"folds": len(todo), "columns": n_cols, "sequences": len(seqs)}


# ---------------------------------------------------------------------------------------------
# decode and score

def results(name: str = RES_NAME) -> Lm.Results:
    return Lm.Results(P.WORK / name)


def f_store() -> dict:
    """F's stored scores (the copy of the F cache's ``final_results.pkl``), read only."""
    return F.load_store(P.WORK / F_STORE)


def stage_score(res: Lm.Results, dev: list, folds: list, store: dict) -> dict:
    """G1 and G1T in every evaluation world of each fold, priors without the held-out match. F, FG and FT
    are re-decoded from the copied probabilities and compared with ``store`` (F's stored scores)."""
    src = F.PriorSource(dev)
    chk = {"n": 0, "F_equal": 0, "FG_equal": 0, "FT_equal": 0}
    for r in folds:
        og, ot, o1 = (Lm.load_oof(v, T, r) for v in (F.VAR_G, F.VAR_T, VAR_T))
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in F.eval_worlds(r):
            g = (og[f"pa_{c}"], og[f"ph_{c}"])
            for w, name in NAMES.items():
                pa, ph = E.blend_heads(g, (o1[f"pa_{c}"], o1[f"ph_{c}"]), w)
                res.put((name, T, c, r.key), F.decode_score(r, c, pa, ph, pri, gt))
            chk["n"] += 1
            for w, name in F_NAMES.items():
                pa, ph = E.blend_heads(g, (ot[f"pa_{c}"], ot[f"ph_{c}"]), w)
                s = F.decode_score(r, c, pa, ph, pri, gt)
                chk[f"{name}_equal"] += int(F.same_score(s, store.get((name, T, c, r.key))))
    res.save()
    return chk
