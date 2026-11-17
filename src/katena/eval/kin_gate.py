"""KIN gate plumbing: the ball-free timing columns (``katena.featurize.kin``) on the final cache.

Nothing in the cache is changed. Under ``KATENA_WORK`` this adds

    kin/<dataset>_<id>__<world>.npz        the columns for both focal teams on every in-play sample (A0, A1, idx)
    kin_pool_d2j/<dataset>_<id>.npz        the same columns for exactly the rows of the ``d2j`` training pool

The columns are computed from the corrupted 5 Hz match of each world, recovered with the corpus seeds exactly as
``touch_gate`` recovers it and checked against the cached base features of that world (``final.recovery_ok``)
before it is used, so they see the same fragments, ghosts and player jitter as the cached groups. The ball is
never read, so the native ball is not lifted.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import d3
from katena.eval import final as F
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.featurize import kin as KF
from katena.featurize.build import infer_dirs
from katena.sim import degrade as dg

# the columns are tagged with the source hash of the feature code and rebuilt when it changes
SRC = hashlib.sha256(Path(KF.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def worlds(ref: P.Ref) -> list:
    """The worlds whose columns a gate needs for ``ref``: its training mix, then the gate worlds."""
    tw = F.train_worlds(ref.dataset)
    return tw + [c for c in F.GATE_CONDS if c not in tw]


def kpath(ref: P.Ref, world: str) -> Path:
    return P.WORK / "kin" / f"{ref.dataset}_{ref.id}__{world}.npz"


def pool_path(ref: P.Ref) -> Path:
    return P.WORK / f"kin_pool_{F.MIX}" / f"{ref.dataset}_{ref.id}.npz"


def _fresh(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return "src" in z.files and str(z["src"]) == SRC


def corrupted5(ref: P.Ref, m5, world: str):
    """The corrupted 5 Hz match of one world, with the corpus seeds (``touch_gate.corrupted`` without the ball lift)."""
    if world == "native":
        return m5
    cfg, seed_cond = TG._config(world)
    return dg.corrupt(m5, cfg, P.seed_for(ref.id, seed_cond))


def load_kin(ref: P.Ref, world: str) -> dict:
    with np.load(kpath(ref, world)) as z:
        out = {k: z[k] for k in z.files}
    if str(out.get("src", "")) != SRC:
        raise RuntimeError(f"{ref.key} {world}: kin columns are stale (built by other feature code)")
    return out


def _as_feat(ref: P.Ref, world: str) -> dict:
    k = load_kin(ref, world)
    return {"idx": k["idx"], "X0": k["A0"], "X1": k["A1"], "names": k["names"]}


def build_one(ref: P.Ref) -> dict:
    """The columns of every world of ``ref`` and of its pool rows. Returns aggregate counts only."""
    from katena.grid import downsample
    from katena.types import load_match
    t0 = time.time()
    st = {"key": ref.key, "worlds": len(worlds(ref)), "built": 0, "rows": 0}
    todo = [w for w in worlds(ref) if not _fresh(kpath(ref, w))]
    if todo:
        m5 = downsample(load_match(ref.path))
        for w in todo:
            mc = corrupted5(ref, m5, w)
            if not F.recovery_ok(ref, w, mc):
                raise RuntimeError(f"{ref.key} {w}: recovered corruption differs from the cached features")
            dirs = infer_dirs(mc)
            with np.load(C.paths(ref)["feat"](w)) as z:
                if "dirs" in z.files and not np.array_equal(z["dirs"], [dirs[h] for h, _, _ in mc.half_blocks]):
                    raise RuntimeError(f"{ref.key} {w}: attack directions differ from the cached features")
            X, names, idx = KF.kin_features(mc, dirs=dirs)
            d3._save(kpath(ref, w), {"A0": X[0], "A1": X[1], "idx": idx, "names": np.array(names), "src": np.array(SRC)})
            st["built"] += 1
            st["rows"] = len(idx)
    out = pool_path(ref)
    if not _fresh(out):
        base = C.load_pool(ref)
        pool = d2.assemble_pool(ref, d2.mix_for(ref.dataset, F.MIX), base, load=_as_feat)
        mine = d2.load_pool(ref, F.MIX)
        if not (np.array_equal(mine["o"], base["o"]) and np.array_equal(mine["idx"], base["idx"])):
            raise RuntimeError(f"{ref.key}: the {F.MIX} pool does not line up with the base pool")
        names = load_kin(ref, C.POOL_CONDS[0])["names"]
        d3._save(out, {"A": pool["X"], "o": base["o"], "idx": base["idx"], "names": names, "src": np.array(SRC)})
    st["wall_s"] = time.time() - t0
    return st


def build_all(refs: list, workers: int = 1) -> list:
    """``build_one`` for every match; a pool needs the columns of every development match."""
    out = []

    def note(st):
        out.append(st)
        print(f"kin {st['key']}: {st['built']} of {st['worlds']} worlds built ({st['wall_s']:.0f}s)", flush=True)

    todo = [r for r in refs if not (_fresh(pool_path(r)) and all(_fresh(kpath(r, w)) for w in worlds(r)))]
    if workers <= 1 or len(todo) <= 1:
        for r in todo:
            note(build_one(r))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for st in pool.imap_unordered(build_one, todo):
                note(st)
    return out


def features(ref: P.Ref, cond: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """(base feature dict, X0, X1) with the touch columns and then the kin columns appended."""
    f, X0, X1 = TG.features(ref, cond)
    k = load_kin(ref, cond)
    if not np.array_equal(k["idx"], f["idx"]):
        raise RuntimeError(f"{ref.key} {cond}: kin rows do not line up")
    return f, np.concatenate([X0, k["A0"]], 1), np.concatenate([X1, k["A1"]], 1)


class KinPools(F.FinalPools):
    """The ``d2j`` training pool with the touch columns, and the kin columns of the same rows appended."""

    def __init__(self, refs: list):
        super().__init__(refs)
        parts = []
        for r in refs:
            with np.load(pool_path(r)) as z:
                t = {k: z[k] for k in z.files}
            if str(t.get("src", "")) != SRC:
                raise RuntimeError(f"{r.key}: kin pool is stale (built by other feature code)")
            parts.append(t)
        A = np.concatenate([p["A"] for p in parts])
        if len(A) != len(self.X) or not np.array_equal(np.concatenate([p["o"] for p in parts]), self.o):
            raise RuntimeError("kin pool does not line up with the training pool")
        self.n_f = self.X.shape[1]                       # F3's columns come first
        self.X = np.concatenate([self.X, A.astype(self.X.dtype)], 1)
        self.names = self.names + [str(n) for n in parts[0]["names"]]
        self.groups = np.concatenate([self.groups, np.full(A.shape[1], KF.GROUP)])
