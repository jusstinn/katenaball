"""Final candidate F: D1 touch features, D3 TCN + LightGBM ensemble at w = 0.5 and D2's ``d2j`` training mix,
gated against M2 on the 28 external development folds.

Pre-registered in docs/decision-log.md (2026-11-07, "Final candidate F (pre-registration)", commit 3921643)
before any score of F was computed. In fold k both members train on the same 27 development matches:

* ``FG`` (F at w = 0): M2's LightGBM heads and parameters on M2's 160 columns plus D1's 53 ``touch`` columns,
  trained on the ``d2j`` pool (``d2.build_pool``), whose touch columns come from the same world as each row.
* ``FT`` (F at w = 1): the M5 TCN unchanged (``ensemble.fit_tcn``, seed 0, 4 threads) on M2's columns, one
  whole-match sequence per (training match, world) of the match's ``d2j`` mix.
* ``F``: ``ensemble.blend_heads(FG, FT, 0.5)``, Viterbi at tau = 2, priors without the held-out match.

The cache is ``KATENA_WORK`` (its own; nothing is written to another cache). Under it:

    gt/ pool/ feat/                copied from the read-only base caches (``seed_cache``), development matches
    pool_d2j/                      ``d2.build_pool(ref, "d2j")``
    touch/<key>__<world>.npz       touch columns per world (``touch_gate`` layout, keyed on touch.py's hash)
    touch_pool_d2j/<key>.npz       touch columns of exactly the ``d2j`` pool rows
    oof/final_gbdt/T2/, oof/final_tcn/T2/   out-of-fold probabilities per evaluation world
    final_results.pkl              scores; final_cost.jsonl: wall, CPU and peak memory per stage

Never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench. Logs are aggregates of
external matches only.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from katena.decode import hmm
from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import d3
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.eval.bootstrap import adopted, paired_bootstrap
from katena.featurize import touch as TF
from katena.models import ensemble as E
from katena.sim import worlds as W

T = P.T_MAIN
TAU = 2.0                       # fixed a priori (D5 audit A2)
MIX = "d2j"
W_F = E.W_MAIN                  # 0.5, fixed a priori
VAR_G = "final_gbdt"
VAR_T = "final_tcn"
NAMES = {W_F: "F", 0.0: "FG", 1.0: "FT"}
GATE_CONDS = ("target", "severe")
RES_NAME = "final_results.pkl"
COST_NAME = "final_cost.jsonl"
DET_FOLD = "idsse/DFL-MAT-J03WMX"


# ---------------------------------------------------------------------------------------------
# worlds

def train_worlds(dataset: str, mix: str = MIX) -> list:
    """Worlds of the training mix, in pool order: severity (mild, target, severe), then mix order."""
    m = d2.mix_for(dataset, mix)
    return [w for c in C.POOL_CONDS for w in m[c]]


def eval_worlds(ref: P.Ref) -> list:
    """``worlds.EVAL_WORLDS`` that apply to the fold's dataset (coupled worlds need dead labels)."""
    return [w for w in W.EVAL_WORLDS if W.applies(w, ref.dataset)]


def feature_worlds(ref: P.Ref) -> list:
    """Every world whose features this experiment reads for ``ref``: training mix, then evaluation-only."""
    tw = train_worlds(ref.dataset)
    return tw + [w for w in eval_worlds(ref) if w not in tw]


def dev_refs() -> list:
    dev, locked = P.split(P.all_refs())
    assert not ({r.key for r in dev} & {r.key for r in locked})
    return dev


# ---------------------------------------------------------------------------------------------
# cost accounting

def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GiB."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.PeakWorkingSetSize / 2 ** 30
    import resource
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


class Cost:
    """Wall time, process CPU time (all threads) and peak memory of a stage; ``extra_cpu`` adds worker
    processes."""

    def __init__(self, stage: str):
        self.stage, self.t0, self.c0 = stage, time.time(), time.process_time()
        self.extra_cpu, self.extra_peak = 0.0, 0.0

    def add_worker(self, cpu: float, peak: float) -> None:
        self.extra_cpu += cpu
        self.extra_peak = max(self.extra_peak, peak)

    def done(self, **info) -> dict:
        rec = {"stage": self.stage, "wall_s": round(time.time() - self.t0, 1),
               "cpu_s": round(time.process_time() - self.c0 + self.extra_cpu, 1),
               "peak_gb": round(max(peak_rss_gb(), self.extra_peak), 2), **info,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        p = P.WORK / COST_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[cost] {rec}", flush=True)
        return rec


def costs() -> list:
    p = P.WORK / COST_NAME
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------
# seeding the cache from the read-only base caches

def _own(path: Path) -> bool:
    return Path(path).resolve() == P.WORK.resolve()


def seed_plan(dev: list, base_root: Path, d2_root: Path) -> list:
    """(source, destination) of every input reused: ground truth, base pool and the mild/target/severe
    features from ``base_root``, every other world's features from ``d2_root``."""
    out = []
    for r in dev:
        name = f"{r.dataset}_{r.id}"
        pth = C.paths(r)
        out.append((Path(base_root) / "gt" / f"{name}.npz", pth["gt"]))
        out.append((Path(base_root) / "pool" / f"{name}.npz", pth["pool"]))
        for w in feature_worlds(r):
            root = base_root if w in C.POOL_CONDS else d2_root
            out.append((Path(root) / "feat" / f"{name}__{w}.npz", pth["feat"](w)))
    return out


def seed_cache(dev: list, base_root: Path, d2_root: Path) -> dict:
    """Copy the reused inputs into ``KATENA_WORK`` (never the other way). Development matches only."""
    if _own(base_root) or _own(d2_root):
        raise RuntimeError("the final cache must not be one of the source caches")
    locked = {r.key for r in P.split(P.all_refs())[1]}
    assert not locked & {r.key for r in dev}
    n_copied, n_bytes = 0, 0
    for src, dst in seed_plan(dev, base_root, d2_root):
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        n_copied += 1
        n_bytes += dst.stat().st_size
    return {"copied": n_copied, "gb": round(n_bytes / 2 ** 30, 2), "planned": len(seed_plan(dev, base_root, d2_root))}


# ---------------------------------------------------------------------------------------------
# touch columns in every world, with the recovery checks

def _in_play(m) -> np.ndarray:
    return np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])


def recovery_ok(ref: P.Ref, world: str, mc) -> bool:
    """The re-corrupted 5 Hz world matches the cached base features of that world: in-play rows, M0
    predictions (which depend on every player and ball position) and the degrader statistics."""
    from katena.models.baseline import m0_predict
    with np.load(C.paths(ref)["feat"](world)) as z:
        idx, m0, deg = z["idx"], z["m0"], z["deg"]
    d = mc.meta.get("degrade", {})
    mine = np.array([d.get("ball_missing", np.nan), d.get("mean_objects", np.nan)])
    return (np.array_equal(idx, _in_play(mc)) and np.array_equal(m0, m0_predict(mc))
            and np.allclose(mine, deg, rtol=0, atol=1e-9))


def base_features(mc) -> dict:
    """The corpus / D2 feature-file arrays of a corrupted 5 Hz match."""
    from katena.featurize.build import featurize, infer_dirs
    from katena.models.baseline import m0_predict
    dirs = infer_dirs(mc)
    f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
    d = mc.meta.get("degrade", {})
    return {"X0": f0.X, "X1": f1.X, "idx": f0.idx, "half": f0.half, "m0": m0_predict(mc),
            "names": np.array(f0.names), "groups": f0.groups,
            "dirs": np.array([dirs[h] for h, _, _ in mc.half_blocks]),
            "deg": np.array([d.get("ball_missing", np.nan), d.get("mean_objects", np.nan)]),
            "deg_json": np.array(json.dumps(d))}


def features_identical(ref: P.Ref, world: str, fresh: dict) -> bool:
    with np.load(C.paths(ref)["feat"](world)) as z:
        return all(z[k].tobytes() == fresh[k].tobytes() for k in ("X0", "X1", "idx", "half")) and \
            list(z["names"]) == list(fresh["names"])


def build_touch(ref: P.Ref, refeaturize: bool = False) -> dict:
    """Touch columns of every world of ``ref`` (``feature_worlds``), each checked against the cached base
    features first. A world that fails the check gets its base features rebuilt here (pre-registered:
    rebuilt, never dropped). With ``refeaturize`` every world's base features are also recomputed and
    compared byte for byte. Returns aggregate counts and this job's CPU time."""
    from katena.grid import downsample
    from katena.types import load_match
    c0, t0 = time.process_time(), time.time()
    st = {"key": ref.key, "worlds": 0, "built": 0, "recovered": 0, "rebuilt": [], "refeaturized": 0,
          "refeaturize_same": 0}
    todo = [w for w in feature_worlds(ref) if refeaturize or not TG._fresh(TG.tpaths(ref)["feat"](w))]
    st["worlds"] = len(feature_worlds(ref))
    if todo:
        m = load_match(ref.path)
        m5 = downsample(m)
        for w in todo:
            mc, lifted = TG.corrupted(ref, m, m5, w)
            if recovery_ok(ref, w, mc):
                st["recovered"] += 1
            else:
                d3._save(C.paths(ref)["feat"](w), base_features(mc))
                st["rebuilt"].append(w)
                if not recovery_ok(ref, w, mc):
                    raise RuntimeError(f"{ref.key} {w}: rebuilt features still differ")
            if refeaturize:
                st["refeaturized"] += 1
                st["refeaturize_same"] += int(features_identical(ref, w, base_features(mc)))
            out = TG.tpaths(ref)["feat"](w)
            if TG._fresh(out):
                continue
            X, names, idx = TF.touch_features(mc, TG._ball("touch", m, m5, mc, lifted))
            d3._save(out, {"A0": X[0], "A1": X[1], "idx": idx, "names": np.array(names), "src": np.array(TG.SRC)})
            st["built"] += 1
    st["cpu_s"], st["wall_s"], st["peak_gb"] = time.process_time() - c0, time.time() - t0, peak_rss_gb()
    return st


def _touch_job(args):
    return build_touch(*args)


def build_touch_all(dev: list, workers: int = 2, cost: Cost | None = None) -> list:
    """``build_touch`` for every development match; the first match of each dataset is re-featurized."""
    import multiprocessing as mp
    first = {}
    for r in dev:
        first.setdefault(r.dataset, r.key)
    jobs = [(r, r.key in first.values()) for r in dev]
    out = []

    def note(st):
        out.append(st)
        print(f"touch {st['key']}: {st['built']} built, {st['recovered']} recovered, rebuilt "
              f"{st['rebuilt'] or 'none'}, refeaturized {st['refeaturize_same']}/{st['refeaturized']} "
              f"identical ({st['wall_s']:.0f}s)", flush=True)
        if cost is not None and workers > 1:          # in-process jobs are already in this process's CPU
            cost.add_worker(st["cpu_s"], st["peak_gb"])

    if workers <= 1:
        for job in jobs:
            note(_touch_job(job))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for st in pool.imap_unordered(_touch_job, jobs):
                note(st)
    return out


# ---------------------------------------------------------------------------------------------
# pools

def touch_pool_path(ref: P.Ref) -> Path:
    return P.WORK / f"touch_pool_{MIX}" / f"{ref.dataset}_{ref.id}.npz"


def _touch_as_feat(ref: P.Ref, world: str) -> dict:
    t = TG.load_tfeat(ref, world)
    return {"idx": t["idx"], "X0": t["A0"], "X1": t["A1"], "names": t["names"]}


def build_touch_pool(ref: P.Ref) -> None:
    """Touch columns of the ``d2j`` pool rows, assembled exactly like the pool (``d2.assemble_pool``)."""
    out = touch_pool_path(ref)
    if TG._fresh(out):
        return
    base = C.load_pool(ref)
    pool = d2.assemble_pool(ref, d2.mix_for(ref.dataset, MIX), base, load=_touch_as_feat)
    names = TG.load_tfeat(ref, C.POOL_CONDS[0])["names"]
    d3._save(out, {"A": pool["X"], "o": base["o"], "idx": base["idx"], "names": names, "src": np.array(TG.SRC)})


def build_pools(dev: list, other_root: Path | None = None) -> dict:
    """``d2j`` base pools (``d2.build_pool``, which checks the layout against the bake-off pool) and their
    touch columns. With ``other_root``, each base pool is compared with the one built there."""
    same, compared = 0, 0
    for r in dev:
        d2.build_pool(r, MIX)
        build_touch_pool(r)
        if other_root is not None:
            theirs = Path(other_root) / f"pool_{MIX}" / f"{r.dataset}_{r.id}.npz"
            if theirs.exists():
                compared += 1
                a = d2.load_pool(r, MIX)
                with np.load(theirs) as z:
                    b = {k: z[k] for k in z.files}
                same += int(set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a))
    return {"pools": len(dev), "compared": compared, "identical": same}


class FinalPools(Lm.Pools):
    """The ``d2j`` training pool of ``refs``; with ``touch`` the touch columns of the same rows are appended
    (same rows, same order: orientations and sample indices are checked)."""

    def __init__(self, refs: list, touch: bool = True):
        self.refs = refs
        parts = [d2.load_pool(r, MIX) for r in refs]
        tparts = []
        if touch:
            for r, p in zip(refs, parts):
                with np.load(touch_pool_path(r)) as z:
                    t = {k: z[k] for k in z.files}
                if str(t.get("src", "")) != TG.SRC:
                    raise RuntimeError(f"{r.key}: touch pool is stale (built by other feature code)")
                if not (np.array_equal(t["o"], p["o"]) and np.array_equal(t["idx"], p["idx"])):
                    raise RuntimeError(f"{r.key}: touch pool does not line up with the d2j pool")
                tparts.append(t)
        n_base = parts[0]["X"].shape[1]
        n_cols = n_base + (tparts[0]["A"].shape[1] if touch else 0)
        n_rows = sum(len(p["o"]) for p in parts)
        self.X = np.empty((n_rows, n_cols), dtype=parts[0]["X"].dtype)
        off = 0
        for i, p in enumerate(parts):
            k = len(p["o"])
            self.X[off:off + k, :n_base] = p["X"]
            if touch:
                self.X[off:off + k, n_base:] = tparts[i]["A"]
            off += k
        self.n_base = n_base
        self.keys = np.concatenate([[r.key] * len(p["o"]) for r, p in zip(refs, parts)])
        self.o = np.concatenate([p["o"] for p in parts])
        self.y = {t: np.concatenate([p[f"y_T{t}"] for p in parts]) for t in P.T_SWEEP}
        self.dead_ok = np.concatenate([np.full(len(p["o"]), r.dataset != "skillcorner") for r, p in zip(refs, parts)])
        self.names = list(parts[0]["names"]) + (list(tparts[0]["names"]) if touch else [])
        self.groups = np.concatenate([parts[0]["groups"], np.full(n_cols - n_base, TF.GROUP)])


class PriorSource:
    """What ``bakeoff.priors_for`` needs: the label sequences of the development matches."""

    def __init__(self, refs: list):
        self.refs = refs

    label_seqs = Lm.Pools.label_seqs


# ---------------------------------------------------------------------------------------------
# LightGBM member

def gbdt_hash(heads) -> str:
    import hashlib
    h = hashlib.sha256()
    for b in (heads.alive, heads.team):
        h.update(b.model_to_string().encode())
    h.update(np.ascontiguousarray(heads.cols).tobytes())
    return h.hexdigest()


def predict_gbdt(heads, ref: P.Ref, conds) -> dict:
    out = {}
    for c in conds:
        _, X0, X1 = TG.features(ref, c)
        out[f"pa_{c}"], out[f"ph_{c}"] = heads.predict(X0, X1)
    return out


def stage_gbdt(pools: FinalPools, folds: list, variant: str = VAR_G) -> None:
    cols = pools.cols()
    for r in folds:
        out = Lm.oof_path(variant, T, r)
        if out.exists():
            continue
        t0 = time.time()
        heads = pools.fit({r.key}, T, cols)
        t1 = time.time()
        arrays = predict_gbdt(heads, r, eval_worlds(r))
        arrays["model_sha"] = np.array(gbdt_hash(heads))
        d3._save(out, arrays)
        print(f"[{time.strftime('%H:%M:%S')}] LightGBM fold {r.key}: fit {t1 - t0:.0f}s, "
              f"predict {time.time() - t1:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------
# TCN member

def tcn_seqs(dev: list, T: int = P.T_MAIN, seed: int = 0, mix: str = MIX) -> dict:
    """One whole-match sequence per (match, world) of the match's training mix, in the order match,
    severity, world, with the orientation drawn per sequence as ``stages._tcn_seqs`` does. With a mix of
    one world per severity this is ``_tcn_seqs`` itself."""
    rng = np.random.default_rng(P.SEED + seed)
    seqs = {}
    for r in dev:
        g = C.load_gt(r, T)
        y5 = g["y"][::g["step"]]
        for w in train_worlds(r.dataset, mix):
            o = int(rng.integers(0, 2))
            with np.load(C.paths(r)["feat"](w)) as z:
                X, idx = z["X1" if o else "X0"].astype(np.float16), z["idx"]
            seqs[(r.key, w)] = {"X": X, "y": y5[idx], "o": o, "dead_ok": r.dataset != "skillcorner", "key": r.key}
    return seqs


class ScaledSeq:
    """A stored sequence seen through a scaler: what ``tcn.train`` reads (``shape`` and row slices), scaled
    on access. ``tcn.Scaler`` is elementwise, so every slice equals the same slice of the scaled whole."""

    def __init__(self, raw: np.ndarray, sc):
        self.raw, self.sc = raw, sc

    @property
    def shape(self) -> tuple:
        return self.raw.shape

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, sl):
        return self.sc(self.raw[sl].astype(np.float32))


def fit_tcn(pools: Lm.Pools, seqs: dict, exclude: set, seed: int = 0, threads: int = E.TCN_THREADS,
            steps: int | None = None, lazy: bool = True):
    """Scaler and TCN trained on every sequence whose match is not in ``exclude``. The scaler is
    ``d3.fit_tcn_excluding``'s (every 6th pooled row of the training matches, M2 columns only)."""
    from katena.models import tcn
    n_base = getattr(pools, "n_base", pools.X.shape[1])
    rows = np.flatnonzero(~np.isin(pools.keys, list(exclude)))[::6]
    sc = tcn.Scaler(np.ascontiguousarray(pools.X[rows, :n_base]))
    wrap = (lambda X: ScaledSeq(X, sc)) if lazy else (lambda X: sc(X.astype(np.float32)))
    tr = [{**s, "X": wrap(s["X"])} for s in seqs.values() if s["key"] not in exclude]
    model = E.fit_tcn(tr, seed=seed, threads=threads, steps=steps)
    del tr
    return model, sc


def stage_tcn(pools: Lm.Pools, dev: list, folds: list, seed: int = 0, threads: int = E.TCN_THREADS,
              steps: int | None = None, variant: str = VAR_T) -> None:
    todo = [r for r in folds if not Lm.oof_path(variant, T, r).exists()]
    if not todo:
        return
    t0 = time.time()
    seqs = tcn_seqs(dev, T, seed)
    Lm.timed(f"TCN sequences ({len(seqs)}, {sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB)", t0)
    for r in todo:
        t1 = time.time()
        model, sc = fit_tcn(pools, seqs, {r.key}, seed, threads, steps)
        t2 = time.time()
        arrays = d3.predict_tcn(model, sc, r, eval_worlds(r), threads)
        arrays["model_sha"] = np.array(E.model_hash(model))
        d3._save(Lm.oof_path(variant, T, r), arrays)
        print(f"[{time.strftime('%H:%M:%S')}] TCN fold {r.key}: train {t2 - t1:.0f}s, "
              f"predict {time.time() - t2:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------
# decode and score

def results(name: str = RES_NAME) -> Lm.Results:
    return Lm.Results(P.WORK / name)


def load_store(path: Path) -> dict:
    """A results.pkl of another cache, read only (a plain dict: it is never saved back)."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _meta(ref: P.Ref, cond: str) -> dict:
    with np.load(C.paths(ref)["feat"](cond)) as z:
        return {"idx": z["idx"], "half": z["half"]}


def decode_score(ref: P.Ref, cond: str, pa, ph, pri, gt) -> dict:
    f = _meta(ref, cond)
    rows = hmm.decode(pa, ph, f["half"], pri, TAU, "viterbi")
    return Lm.score_rows(ref, T, cond, rows, f["idx"], gt)


def stage_score(res: Lm.Results, dev: list, folds: list, weights=(W_F, 0.0, 1.0)) -> None:
    """F and its members in every evaluation world of each fold, priors without the held-out match."""
    src = PriorSource(dev)
    for r in folds:
        og, ot = Lm.load_oof(VAR_G, T, r), Lm.load_oof(VAR_T, T, r)
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in eval_worlds(r):
            for w in weights:
                pa, ph = E.blend_heads((og[f"pa_{c}"], og[f"ph_{c}"]), (ot[f"pa_{c}"], ot[f"ph_{c}"]), w)
                res.put((NAMES[w], T, c, r.key), decode_score(r, c, pa, ph, pri, gt))
    res.save()


def same_score(a: dict, b: dict) -> bool:
    return (a is not None and b is not None and a["spell_f1"] == b["spell_f1"] and a["macro_f1"] == b["macro_f1"]
            and a["n_pred"] == b["n_pred"] and a["n_gt"] == b["n_gt"] and np.array_equal(a["flags"], b["flags"]))


def stage_incumbent(res: Lm.Results, dev: list, folds: list, base_root: Path, d2_root: Path) -> dict:
    """M2 re-decoded with this code at tau 2: ``M2_t2`` from the base cache's LOMO probabilities (target,
    severe, mild), and ``M2w`` in every evaluation world from D2's ``cur`` probabilities (M2's heads; byte
    identical to the base cache's on the base conditions). Both must equal the stored M2 scores."""
    base = load_store(Path(base_root) / "results.pkl")
    src = PriorSource(dev)
    chk = {"m2_t2_equal": 0, "m2w_equal": 0, "cur_probs_equal": 0, "n": 0}
    for r in folds:
        name = f"{r.dataset}_{r.id}.npz"
        with np.load(Path(base_root) / "oof" / "full" / f"T{T}" / name) as z:
            og = {k: z[k] for k in z.files}
        with np.load(Path(d2_root) / "d2" / "oof" / "cur" / f"T{T}" / name) as z:
            oc = {k: z[k] for k in z.files}
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in ("mild",) + GATE_CONDS:
            s = decode_score(r, c, og[f"pa_{c}"], og[f"ph_{c}"], pri, gt)
            res.put(("M2_t2", T, c, r.key), s)
            chk["n"] += 1
            chk["m2_t2_equal"] += int(same_score(s, base.get(("M2", T, c, r.key))))
            chk["cur_probs_equal"] += int(og[f"pa_{c}"].tobytes() == oc[f"pa_{c}"].tobytes()
                                          and og[f"ph_{c}"].tobytes() == oc[f"ph_{c}"].tobytes())
        for c in eval_worlds(r):
            s = decode_score(r, c, oc[f"pa_{c}"], oc[f"ph_{c}"], pri, gt)
            res.put(("M2w", T, c, r.key), s)
            if c in ("mild",) + GATE_CONDS:
                chk["m2w_equal"] += int(same_score(s, base.get(("M2", T, c, r.key))))
    res.save()
    return chk


# ---------------------------------------------------------------------------------------------
# gate

def compare(get, refs: list, cand: str, inc: str, cond: str) -> dict | None:
    """Paired bootstrap of ``cand`` - ``inc`` over the folds where both were scored, with the means,
    folds won, the adoption rule and D2's 'worse' test."""
    rr = [r for r in refs if get(cand, cond, r) is not None and get(inc, cond, r) is not None]
    if not rr:
        return None
    a, b = [get(cand, cond, r) for r in rr], [get(inc, cond, r) for r in rr]
    out = paired_bootstrap(a, b)
    out.update(cand_mean=float(np.mean([x["spell_f1"] for x in a])), inc_mean=float(np.mean([x["spell_f1"] for x in b])),
               cand_macro=float(np.mean([x["macro_f1"] for x in a])), inc_macro=float(np.mean([x["macro_f1"] for x in b])),
               wins=int(sum(x["spell_f1"] > y["spell_f1"] for x, y in zip(a, b))), adopted=adopted(out),
               worse=bool(out["mean_diff"] < 0 and out["hi"] < 0))
    return out


def decision(g_target: dict, g_severe: dict) -> bool:
    """Pre-registered: adopted at target (wins >= 70 % and CI above 0) and not worse at severe."""
    return bool(g_target["adopted"] and not g_severe["worse"])


def getter(res: Lm.Results, base: dict):
    """Scores of this experiment first, then the stored ones (where M2 lives)."""
    def get(name, cond, r):
        k = (name, T, cond, r.key)
        v = res.get(k)
        return v if v is not None else base.get(k)
    return get
