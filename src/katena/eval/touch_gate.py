"""D1 gate: M2 plus the native-frame-rate ``touch`` group (``katena.featurize.touch``), on the bake-off cache.

Nothing in the base cache is changed. Under the same work directory (``KATENA_WORK``) this adds

    touch/<dataset>_<id>__<cond>.npz   touch columns for both focal teams on every in-play sample (A0, A1, idx)
    touch_pool/<dataset>_<id>.npz      the same columns for exactly the rows of the base training pool (A, o, idx)
    oof/<variant>/T2/...               out-of-fold p_alive / p_home of M2's heads trained on base + touch columns

and scores them in ``results.pkl`` under the model name ``M2T`` (decoded exactly like M2: Viterbi, tau chosen
for each match from the other matches' out-of-fold scores).

Noise consistency: the corpus corrupts the 5 Hz match. The degrader is re-run here with the corpus seeds to
recover each corrupted 5 Hz match, and the recovered match is checked against the cache (its M0 predictions,
which depend on every player and ball position, and the degrader statistics) before it is used. The native
ball is then lifted from that corruption (``touch.lift_ball``), so the touch group and the cached base groups
see the same ball gaps, false detections and jitter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path

import numpy as np

import possession_metrics as pm
from katena.decode import hmm
from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.featurize import touch as TF
from katena.grid import downsample
from katena.models.baseline import m0_predict
from katena.sim import degrade as dg
from katena.types import load_match

VARIANT = "touch"
MODEL = "M2T"
# touch columns are tagged with the source hash of the feature code and rebuilt when it changes
SRC = hashlib.sha256(Path(TF.__file__).read_bytes()).hexdigest()[:16]
EDGES = (0, 2, 4, 8, 16, 32, 1e9)

# Evaluation-only world, declared before any gate result: the target-like corruption with the same seed
# (same ball gaps, false detections, fragments, ghosts and player jitter) but no ball jitter. Real native
# ball tracks (Metrica sample games 1 and 3, SkillCorner) are smooth frame to frame, while the degrader's
# ball jitter is i.i.d. at 5 Hz; this world checks that the touch group does not depend on the rough one.
# Models are never trained on it.
SMOOTH = "target_smoothball"
EXTRA_EVAL = {SMOOTH: (dataclasses.replace(dg.TARGET, name=SMOOTH, sigma_ball=0.0), "target")}


# ``kind`` "touch" is the D1 group (native-rate ball lifted from the 5 Hz corruption); "touch5" is the control:
# the same code on the corrupted 5 Hz ball, so "touch" vs "touch5" isolates the native frame rate.
KINDS = ("touch", "touch5")


def tpaths(ref: P.Ref, kind: str = "touch") -> dict:
    if kind not in KINDS:
        raise ValueError(kind)
    d = P.WORK
    return {"feat": lambda c: d / kind / f"{ref.dataset}_{ref.id}__{c}.npz",
            "pool": d / f"{kind}_pool" / f"{ref.dataset}_{ref.id}.npz"}


def _ball(kind: str, m, m5, mc, lifted) -> TF.NativeBall:
    if kind == "touch5":
        return TF.NativeBall(mc.ball, mc.fps, 1)
    return TF.NativeBall(lifted, m.fps, int(m5.meta["step"]))


def _config(cond: str):
    """(degrader config, seed condition) of a condition: the corpus ones, this module's evaluation world,
    or a D2 world (``katena.sim.worlds``, on ``origin/david/d2-degrader``; seeded as D2 seeds it)."""
    if cond in EXTRA_EVAL:
        return EXTRA_EVAL[cond]
    if cond in C.CONFIG:
        return C.CONFIG[cond], cond
    from katena.sim import worlds as W
    return W.config(cond), cond


def corrupted(ref: P.Ref, m, m5, cond: str):
    """(corrupted 5 Hz match, lifted native ball) for one condition, with the corpus seeds."""
    step = int(m5.meta["step"])
    if cond == "native":
        return m5, np.asarray(m.ball, dtype=np.float32)
    cfg, seed_cond = _config(cond)
    mc = dg.corrupt(m5, cfg, P.seed_for(ref.id, seed_cond))
    lifted = TF.lift_ball(m.ball, m5.ball, mc.ball, step, cfg.sigma_ball, P.seed_for(ref.id, seed_cond + "|native"))
    return mc, lifted


def build_extra(ref: P.Ref, cond: str = SMOOTH, force: bool = False, kind: str = "touch") -> str:
    """Base and touch features of an evaluation-only condition, written next to the cached ones in the
    corpus format (``feat/<key>__<cond>.npz``) so the shared decode and score code can read them."""
    from katena.featurize.build import featurize, infer_dirs
    out_b, out_t = C.paths(ref)["feat"](cond), tpaths(ref, kind)["feat"](cond)
    if not force and out_b.exists() and _fresh(out_t):
        return ref.key
    out_t.parent.mkdir(parents=True, exist_ok=True)
    m = load_match(ref.path)
    m5 = downsample(m)
    mc, lifted = corrupted(ref, m, m5, cond)
    if force or not out_b.exists():
        dirs = infer_dirs(mc)
        f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
        np.savez(out_b, X0=f0.X, X1=f1.X, idx=f0.idx, half=f0.half, m0=m0_predict(mc), names=np.array(f0.names),
                 groups=f0.groups, dirs=np.array([dirs[h] for h, _, _ in m5.half_blocks]),
                 deg=np.array([mc.meta["degrade"]["ball_missing"], mc.meta["degrade"]["mean_objects"]]))
    X, names, idx = TF.touch_features(mc, _ball(kind, m, m5, mc, lifted))
    np.savez(out_t, A0=X[0], A1=X[1], idx=idx, names=np.array(names), src=np.array(SRC))
    return ref.key


def build_one(ref: P.Ref, force: bool = False, check: bool = True, kind: str = "touch") -> str:
    """Touch columns for every condition of ``ref`` and for its training-pool rows."""
    tp = tpaths(ref, kind)
    conds = C.conds_for(ref)
    if not force and _fresh(tp["pool"]) and all(_fresh(tp["feat"](c)) for c in conds):
        return ref.key
    tp["pool"].parent.mkdir(parents=True, exist_ok=True)
    tp["feat"]("x").parent.mkdir(parents=True, exist_ok=True)
    m = load_match(ref.path)
    m5 = downsample(m)
    step = int(m5.meta["step"])
    rng = np.random.default_rng(P.seed_for(ref.id, "pool"))       # the corpus pool's orientation draws
    pool_A, pool_o, pool_i, names = [], [], [], None
    for c in conds:
        mc, lifted = corrupted(ref, m, m5, c)
        X, names, idx = TF.touch_features(mc, _ball(kind, m, m5, mc, lifted))
        if check:
            base = C.load_feat(ref, c)
            if not (np.array_equal(base["idx"], idx) and np.array_equal(base["m0"], m0_predict(mc))):
                raise RuntimeError(f"{ref.key} {c}: recovered corruption differs from the cached features")
            if c != "native":
                deg = np.array([mc.meta["degrade"]["ball_missing"], mc.meta["degrade"]["mean_objects"]])
                if not np.allclose(deg, base["deg"], rtol=0, atol=1e-9):
                    raise RuntimeError(f"{ref.key} {c}: degrader statistics differ from the cache")
        np.savez(tp["feat"](c), A0=X[0], A1=X[1], idx=idx, names=np.array(names), src=np.array(SRC))
        if c in C.POOL_CONDS:
            sel = np.flatnonzero((np.arange(len(idx)) + C.POOL_CONDS.index(c)) % C.POOL_STRIDE == 0)
            o = rng.integers(0, 2, len(sel)).astype(np.int8)
            pool_A.append(np.where(o[:, None] == 1, X[1][sel], X[0][sel]))
            pool_o.append(o)
            pool_i.append(idx[sel])
    o, ii = np.concatenate(pool_o), np.concatenate(pool_i)
    if check:
        bp = C.load_pool(ref)
        if not (np.array_equal(bp["o"], o) and np.array_equal(bp["idx"], ii)):
            raise RuntimeError(f"{ref.key}: touch pool rows do not line up with the base pool")
    np.savez(tp["pool"], A=np.concatenate(pool_A), o=o, idx=ii, names=np.array(names), src=np.array(SRC))
    return ref.key


def _fresh(path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return "src" in z.files and str(z["src"]) == SRC


def _job(args):
    return build_one(*args)


def build_all(refs: list, workers: int = 2, force: bool = False, kind: str = "touch") -> None:
    import multiprocessing as mp
    t0 = time.time()
    if workers <= 1:
        for r in refs:
            print(f"{kind} built", build_one(r, force, True, kind), f"({time.time() - t0:.0f}s)", flush=True)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for k in pool.imap_unordered(_job, [(r, force, True, kind) for r in refs]):
            print(f"{kind} built", k, f"({time.time() - t0:.0f}s)", flush=True)


def load_tfeat(ref: P.Ref, cond: str, kind: str = "touch") -> dict:
    with np.load(tpaths(ref, kind)["feat"](cond)) as z:
        out = {k: z[k] for k in z.files}
    if str(out.get("src", "")) != SRC:
        raise RuntimeError(f"{ref.key} {cond}: touch columns are stale (built by other feature code)")
    return out


class TouchPools(Lm.Pools):
    """The base training pool with the touch columns appended (same rows, same order)."""

    def __init__(self, refs: list, kind: str = "touch"):
        super().__init__(refs)
        self.kind = kind
        parts = []
        for r in refs:
            with np.load(tpaths(r, kind)["pool"]) as z:
                parts.append({k: z[k] for k in z.files})
        if any(str(p.get("src", "")) != SRC for p in parts):
            raise RuntimeError("touch pool is stale (built by other feature code)")
        A = np.concatenate([p["A"] for p in parts])
        if len(A) != len(self.X) or not np.array_equal(np.concatenate([p["o"] for p in parts]), self.o):
            raise RuntimeError("touch pool does not line up with the base pool")
        self.X = np.concatenate([self.X, A], 1)
        self.names = self.names + list(parts[0]["names"])
        self.groups = np.concatenate([self.groups, np.full(A.shape[1], TF.GROUP)])


def features(ref: P.Ref, cond: str, kind: str = "touch") -> tuple[dict, np.ndarray, np.ndarray]:
    """(base feature dict, X0, X1) with the touch columns appended."""
    f = C.load_feat(ref, cond)
    t = load_tfeat(ref, cond, kind)
    if not np.array_equal(t["idx"], f["idx"]):
        raise RuntimeError(f"{ref.key} {cond}: touch rows do not line up")
    return f, np.concatenate([f["X0"], t["A0"]], 1), np.concatenate([f["X1"], t["A1"]], 1)


def run_fold(pools: TouchPools, ref: P.Ref, cols: np.ndarray, variant: str = VARIANT, T: int = P.T_MAIN,
             conds: list | None = None) -> None:
    out = Lm.oof_path(variant, T, ref)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    heads = pools.fit({ref.key}, T, cols)
    arrays = {}
    for c in (conds or C.conds_for(ref)):
        _, X0, X1 = features(ref, c, getattr(pools, "kind", "touch"))
        arrays[f"pa_{c}"], arrays[f"ph_{c}"] = heads.predict(X0, X1)
    np.savez(out, **arrays)


def stage_lomo(pools: TouchPools, folds: list, variant: str = VARIANT, drop=(), T: int = P.T_MAIN,
               extra: tuple = (), conds: list | None = None) -> None:
    """LOMO heads on base + touch columns (minus ``drop`` groups); predictions for the corpus conditions of
    each fold plus ``extra`` ones, or exactly ``conds``."""
    t0 = time.time()
    cols = pools.cols(drop=drop)
    for r in folds:
        if Lm.oof_path(variant, T, r).exists():
            continue
        run_fold(pools, r, cols, variant, T, conds=list(conds) if conds else C.conds_for(r) + list(extra))
        Lm.timed(f"{variant} fold {r.key}", t0)


# ---------------------------------------------------------------------------------------------
# decoding at a fixed temperature

TAU = 2.0     # fixed, as the D5 audit recommends: per-fold tau selection uses folds whose models saw the
              # held-out match. The bake-off's selection picked 2.0 for all 28 M2 folds anyway.


def _rows_meta(ref: P.Ref, cond: str) -> dict:
    with np.load(C.paths(ref)["feat"](cond)) as z:          # lazy: only the two small arrays
        return {"idx": z["idx"], "half": z["half"]}


def decode_fixed(res: Lm.Results, pools, refs: list, variant: str, name: str, conds: tuple,
                 T: int = P.T_MAIN, tau: float = TAU, kind: str = "viterbi") -> None:
    """Decode the stored out-of-fold probabilities of ``variant`` at a fixed ``tau`` with M2's priors
    (fit on the label sequences of the other development matches) and score every listed condition the
    fold has. ``pools`` only needs ``label_seqs``."""
    t0 = time.time()
    for r in refs:
        oof = Lm.load_oof(variant, T, r)
        pri = B.priors_for(pools, {r.key}, T)
        g = C.load_gt(r, T)
        for c in conds:
            if f"pa_{c}" not in oof:
                continue
            f = _rows_meta(r, c)
            rows = hmm.decode(oof[f"pa_{c}"], oof[f"ph_{c}"], f["half"], pri, tau, kind)
            res.put((name, T, c, r.key), Lm.score_rows(r, T, c, rows, f["idx"], g))
    res.save()
    Lm.timed(f"decode {name} (tau {tau:g}, {len(refs)} folds)", t0)


# ---------------------------------------------------------------------------------------------
# spell counts and recall by true spell length

def spell_table(res: Lm.Results, model: str, refs: list, T: int = P.T_MAIN, cond: str = P.HEADLINE) -> dict:
    """Predicted possessions per true one (pooled, by dataset) and recall at IoU 0.5 by true length."""
    out = {"n_pred": 0, "n_gt": 0, "by_ds": {}, "bins": np.zeros((len(EDGES) - 1, 2))}
    i50 = list(pm.SPELL_IOUS).index(0.5)
    for r in refs:
        s = res.get((model, T, cond, r.key))
        if s is None:
            continue
        g = C.load_gt(r, T)
        gt_sp = pm.possession_spells(g["y"], pm.valid_runs(g["y"]))
        dur = np.array([(b - a) / g["fps"] for a, b, _ in gt_sp])
        fl = s["flags"][i50]
        for i in range(len(EDGES) - 1):
            m = (dur >= EDGES[i]) & (dur < EDGES[i + 1])
            out["bins"][i] += (m.sum(), fl[m].sum())
        out["n_pred"] += s["n_pred"]
        out["n_gt"] += s["n_gt"]
        d = out["by_ds"].setdefault(r.dataset, [0, 0])
        d[0] += s["n_pred"]
        d[1] += s["n_gt"]
    return out


def summary(res: Lm.Results, a: str, b: str, refs: list, T: int = P.T_MAIN, conds=("target", "severe", "mild", "native")) -> str:
    lines = []
    for c in conds:
        rr = [r for r in refs if res.get((a, T, c, r.key)) and res.get((b, T, c, r.key))]
        if not rr:
            continue
        cmp = B.compare(res, a, b, T, c, rr)
        ma = np.mean([res.get((a, T, c, r.key))["spell_f1"] for r in rr])
        mb = np.mean([res.get((b, T, c, r.key))["spell_f1"] for r in rr])
        lines.append(f"{c:7s} n={len(rr):2d}  {a} {ma:.4f}  {b} {mb:.4f}  diff {cmp['mean_diff']:+.4f} "
                     f"[{cmp['lo']:+.4f}, {cmp['hi']:+.4f}]  wins {cmp['win_rate']:.0%}  adopted-rule {cmp['adopted']}")
        for ds in P.DATASETS:
            rd = [r for r in rr if r.dataset == ds]
            if rd:
                da = np.mean([res.get((a, T, c, r.key))["spell_f1"] - res.get((b, T, c, r.key))["spell_f1"] for r in rd])
                wins = np.mean([res.get((a, T, c, r.key))["spell_f1"] > res.get((b, T, c, r.key))["spell_f1"] for r in rd])
                lines.append(f"          {ds:11s} n={len(rd):2d} diff {da:+.4f} wins {wins:.0%}")
        for name in (a, b):
            st = spell_table(res, name, rr, T, c)
            ds = " ".join(f"{k} {v[0] / max(v[1], 1):.3f}" for k, v in st["by_ds"].items())
            rec = " ".join(f"{h / max(n, 1):.3f}" for n, h in st["bins"])
            lines.append(f"          {name:6s} pred/GT {st['n_pred'] / max(st['n_gt'], 1):.3f} ({ds}) | recall@0.5 by length "
                         f"{'/'.join(f'{EDGES[i]:g}' for i in range(len(EDGES) - 1))}+ s: {rec}")
    return "\n".join(lines)
