"""D2 experiment: M2 trained on a mix across degrader worlds versus the current mix, scored in every world.

    uv run python scripts/d2_worlds.py build          # new-world features and M0, development matches
    uv run python scripts/d2_worlds.py pools          # d2 training pools (needs the base cache)
    uv run python scripts/d2_worlds.py lomo --folds 8 # cur and d2 heads per fold, all worlds predicted
    uv run python scripts/d2_worlds.py score --folds 8
    uv run python scripts/d2_worlds.py m0             # simulated official baseline in every world

The decision rule was registered before any gate result, in docs/decision-log.md (2026-11-07, D2), with
amendment A1 (tau fixed at 2; the 8 IDSSE + Metrica folds gated beside the 28; smooth-jitter worlds and the
``d2j`` candidate), also before any gate result.

Only development matches are used: the locked benches are never built, trained on or scored here. The
existing conditions (mild, target, severe), their pools and the ground truth come from the bake-off cache
under KATENA_WORK, made by ``katena.eval.corpus``. The new worlds (``katena.sim.worlds``) are written
beside them under new names, so no existing cache entry changes.

The candidates differ only in the corruption world of each training row. All use the same sample rows,
focal orientations and labels as the bake-off pool (stride 4, one residue class per severity). In ``d2``
and ``d2j``, each severity's rows are split evenly, interleaved, over the worlds that
``katena.sim.worlds.TRAIN_MIXES`` lists for the match's dataset. ``cur`` is the bake-off pool itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from katena.decode import hmm
from katena.eval import corpus as C
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval.bootstrap import adopted, paired_bootstrap
from katena.eval.score import score, to_native_codes
from katena.featurize.build import featurize, infer_dirs
from katena.grid import downsample
from katena.labels import relabel
from katena.models.baseline import _starter
from katena.sim import degrade as dg
from katena.sim import worlds as W
from katena.types import D, load_match

T = P.T_MAIN
TAU = 2.0                        # fixed a priori for every variant (amendment A1, D5 audit A2)
VARIANTS = ("cur", "d2", "d2j")
CANDIDATES = ("d2", "d2j")
# Official baseline windows (starter/baseline.py VARIANTS), as half-widths in 5 Hz samples, and the published
# full-match scores (macroF1, spellF1) from docs/upstream/EVALUATION.md.
M0_WINDOWS = {"raw": 0, "smooth25": 5, "smooth50": 10, "smooth125": 25}
PUBLISHED_M0 = {"raw": (0.6532, 0.0486), "smooth25": (0.6752, 0.1996), "smooth50": (0.6724, 0.2433),
                "smooth125": (0.6356, 0.2242)}
# Secondary worlds: built for the training mix anyway, so they are scored too, but not decision worlds.
SECONDARY = ("mild_cpl50", "mild_cpl100", "mild_ids", "mild_smooth", "severe_cpl50", "severe_cpl100",
             "severe_ids", "severe_smooth")
M0_WORLDS = ("mild", "target", "severe", "target_dead3", "target_dead15", "target_cpl50", "target_cpl100",
             "target_ids", "target_cpl100_ids", "target_smooth", "mild_cpl100", "severe_cpl100")


def root() -> Path:
    return P.WORK / "d2"


def dev_refs() -> list:
    dev, locked = P.split(P.all_refs())
    assert not ({r.key for r in dev} & {r.key for r in locked})
    return dev


def mix_for(dataset: str, variant: str = "d2") -> dict:
    return W.TRAIN_MIXES[variant]["dead_labels" if dataset in W.DEAD_LABEL_DATASETS else "no_dead_labels"]


# The bake-off's own dead-coupled worlds (IDSSE only, already in the base cache): scored as secondary too.
SECONDARY_IDSSE = ("target_dead3", "target_dead15")


def eval_worlds(ref: P.Ref) -> list:
    """Worlds a fold is scored in: the decision worlds, then the secondary ones."""
    extra = SECONDARY_IDSSE if ref.dataset == "idsse" else ()
    return [w for w in W.EVAL_WORLDS + SECONDARY if W.applies(w, ref.dataset)] + list(extra)


def new_worlds(ref: P.Ref) -> list:
    """D2 worlds to build for ``ref``: everything in its training mix or evaluation list that the base
    cache does not already hold."""
    names = {w for v in CANDIDATES for ws in mix_for(ref.dataset, v).values() for w in ws} | set(eval_worlds(ref))
    return sorted(n for n in names if n not in dg.CONFIGS and n not in C.CONFIG)


# ---------------------------------------------------------------------------------------------
# build: features for the new worlds (the same file layout as katena.eval.corpus)

def m0_windows(m) -> np.ndarray:
    """int8[len(M0_WINDOWS), n]: the official baseline with each published window, on the work grid
    (0 H, 1 A, 2 D inside the half blocks, 3 X outside). The smooth50 row equals ``m0_predict``."""
    sb = _starter()
    out = np.full((len(M0_WINDOWS), m.n), 3, dtype=np.int8)
    for _, a, b in m.half_blocks:
        Pb, ball = m.P[a:b + 1], m.ball[a:b + 1]
        codes = np.full(len(ball), sb.NOBODY, dtype=np.int8)
        vis = np.isfinite(ball[:, 0])
        d = np.linalg.norm(Pb - ball[:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        near = np.argmin(d, axis=1)
        ok = vis & np.isfinite(d.min(1))
        codes[ok] = m.team[near[ok]]
        for k, win in enumerate(M0_WINDOWS.values()):
            lab = sb.smooth(codes, win)
            out[k, a:b + 1] = np.select([lab == "H", lab == "A"], [0, 1], default=2)
    return out


def corrupt_world(m5, ref: P.Ref, world: str):
    """A development match in ``world``, with the corpus seed convention (match id and condition name)."""
    return dg.corrupt(m5, W.config(world), P.seed_for(ref.id, world))


def build_ref(ref: P.Ref, worlds: list | None = None) -> list:
    """Write the feature file of every missing world of ``ref``. Returns the worlds built."""
    done = []
    todo = [w for w in (worlds or new_worlds(ref)) if not C.paths(ref)["feat"](w).exists()]
    if not todo:
        return done
    m5 = downsample(load_match(ref.path))
    for w in todo:
        out = C.paths(ref)["feat"](w)
        out.parent.mkdir(parents=True, exist_ok=True)
        mc = corrupt_world(m5, ref, w)
        dirs = infer_dirs(mc)
        f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
        mw = m0_windows(mc)
        deg = mc.meta.get("degrade", {})
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, X0=f0.X, X1=f1.X, idx=f0.idx, half=f0.half, m0=mw[list(M0_WINDOWS).index("smooth50")],
                 m0w=mw, names=np.array(f0.names), groups=f0.groups,
                 dirs=np.array([dirs[h] for h, _, _ in m5.half_blocks]),
                 deg=np.array([deg.get("ball_missing", np.nan), deg.get("mean_objects", np.nan)]),
                 deg_json=np.array(json.dumps(deg)))
        tmp.replace(out)
        done.append(w)
    return done


def _build_job(key: str) -> tuple:
    ref = {r.key: r for r in dev_refs()}[key]
    t0 = time.time()
    return key, build_ref(ref), time.time() - t0


def build_all(workers: int = 2, keys: list | None = None) -> None:
    import multiprocessing as mp
    refs = [r for r in dev_refs() if keys is None or r.key in keys]
    todo = [r.key for r in refs if any(not C.paths(r)["feat"](w).exists() for w in new_worlds(r))]
    print(f"build: {len(todo)} of {len(refs)} development matches have worlds to build", flush=True)
    if workers <= 1:
        for k in todo:
            print("built", *_build_job(k), flush=True)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for key, built, secs in pool.imap_unordered(_build_job, todo):
            print(f"built {key}: {len(built)} worlds in {secs:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------
# pools

def pool_path(ref: P.Ref, variant: str = "d2") -> Path:
    return P.WORK / f"pool_{variant}" / f"{ref.dataset}_{ref.id}.npz"


def pool_rows(n_rows: int) -> dict:
    """Feature-row indices of each severity in the bake-off pool (katena.eval.corpus.build_one)."""
    return {c: np.flatnonzero((np.arange(n_rows) + ci) % C.POOL_STRIDE == 0) for ci, c in enumerate(C.POOL_CONDS)}


def assemble_pool(ref: P.Ref, mix: dict, base: dict, load=C.load_feat) -> dict:
    """Training rows of ``ref`` for ``mix`` ({severity: [worlds]}), with the base pool's rows, orientations
    and labels. A mix of one world per severity reproduces the base pool itself."""
    first = load(ref, mix[C.POOL_CONDS[0]][0])
    n_rows = len(first["idx"])
    X_parts, off = [], 0
    for c, sel in pool_rows(n_rows).items():
        o = base["o"][off:off + len(sel)]
        off += len(sel)
        worlds = mix[c]
        which = np.arange(len(sel)) % len(worlds)
        X = None
        for wi, w in enumerate(worlds):
            f = first if first is not None and w == mix[C.POOL_CONDS[0]][0] else load(ref, w)
            assert len(f["idx"]) == n_rows, (ref.key, w)
            k = which == wi
            part = np.where(o[k, None] == 1, f["X1"][sel[k]], f["X0"][sel[k]])
            if X is None:
                X = np.empty((len(sel), part.shape[1]), dtype=part.dtype)
            X[k] = part
            del f
        first = None
        X_parts.append(X)
    assert off == len(base["o"])
    out = {k: v for k, v in base.items() if k != "X"}
    out["X"] = np.concatenate(X_parts)
    return out


def build_pool(ref: P.Ref, variant: str = "d2", check: bool = True) -> None:
    out = pool_path(ref, variant)
    if out.exists():
        return
    base = C.load_pool(ref)
    if check:   # the layout reconstruction must give back the bake-off pool exactly
        again = assemble_pool(ref, {c: [c] for c in C.POOL_CONDS}, base)
        assert again["X"].tobytes() == base["X"].tobytes(), f"pool layout mismatch for {ref.key}"
    pool = assemble_pool(ref, mix_for(ref.dataset, variant), base)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **pool)
    tmp.replace(out)


def load_pool(ref: P.Ref, variant: str) -> dict:
    if variant == "cur":
        return C.load_pool(ref)
    with np.load(pool_path(ref, variant)) as z:
        return {k: z[k] for k in z.files}


class LoadedPools(Lm.Pools):
    """``katena.eval.lomo.Pools`` with a chosen pool loader."""

    def __init__(self, refs: list, loader):
        parts = [loader(r) for r in refs]
        self.refs = refs
        self.keys = np.concatenate([[r.key] * len(p["o"]) for r, p in zip(refs, parts)])
        self.X = np.concatenate([p["X"] for p in parts])
        self.o = np.concatenate([p["o"] for p in parts])
        self.y = {t: np.concatenate([p[f"y_T{t}"] for p in parts]) for t in P.T_SWEEP}
        self.dead_ok = np.concatenate([np.full(len(p["o"]), r.dataset != "skillcorner") for r, p in zip(refs, parts)])
        self.names = list(parts[0]["names"])
        self.groups = parts[0]["groups"]


# ---------------------------------------------------------------------------------------------
# LOMO heads

def folds(spec: str, dev: list) -> list:
    if spec == "8":
        return [r for r in dev if r.dataset in W.DEAD_LABEL_DATASETS]
    if spec in ("28", "all"):
        return list(dev)
    keys = set(spec.split(","))
    return [r for r in dev if r.key in keys]


def oof_path(variant: str, ref: P.Ref) -> Path:
    return root() / "oof" / variant / f"T{T}" / f"{ref.dataset}_{ref.id}.npz"


def run_lomo(fold_refs: list, dev: list, variants=VARIANTS) -> None:
    for v in variants:
        todo = [r for r in fold_refs if not oof_path(v, r).exists()]
        if not todo:
            continue
        t0 = time.time()
        pools = LoadedPools(dev, lambda r, v=v: load_pool(r, v))
        cols = pools.cols()
        for r in todo:
            heads = pools.fit({r.key}, T, cols)
            arrays = {}
            for w in eval_worlds(r):
                f = C.load_feat(r, w)
                arrays[f"pa_{w}"], arrays[f"ph_{w}"] = heads.predict(f["X0"], f["X1"])
            out = oof_path(v, r)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, **arrays)
            tmp.replace(out)                        # a half-written fold is never taken as done
            Lm.timed(f"{v} fold {r.key}", t0)
        del pools


# ---------------------------------------------------------------------------------------------
# decode, score, compare

def label_seqs(dev: list, exclude: set) -> list:
    out = []
    for r in dev:
        if r.key in exclude or r.is_sc:
            continue
        g = C.load_gt(r, T)
        out.append(g["y"][::g["step"]])
    return out


def score_folds(fold_refs: list, dev: list, variants=VARIANTS, tau: float = TAU) -> dict:
    """{(variant, world, match_key): score dict}. Viterbi at a fixed tau (2, amendment A1: per-fold tau
    selection would use folds whose models saw the held-out match), priors fit without the held-out match."""
    res: dict = {}
    gts = {r.key: C.load_gt(r, T) for r in fold_refs}
    pri = {r.key: hmm.fit_priors(label_seqs(dev, {r.key})) for r in fold_refs}
    for r in fold_refs:
        oofs = {}
        for v in variants:
            with np.load(oof_path(v, r)) as z:
                oofs[v] = {k: z[k] for k in z.files}
        for w in eval_worlds(r):
            with np.load(C.paths(r)["feat"](w)) as z:          # a world's rows are loaded once per fold
                half, idx = z["half"], z["idx"]
            for v in variants:
                rows = hmm.decode(oofs[v][f"pa_{w}"], oofs[v][f"ph_{w}"], half, pri[r.key], tau, "viterbi")
                res[(v, w, r.key)] = Lm.score_rows(r, T, w, rows, idx, gts[r.key])
        for v in variants:
            res[("tau", v, r.key)] = tau
    return res


def compare_worlds(res: dict, fold_refs: list, worlds=None, a: str = "d2", b: str = "cur") -> dict:
    """Per world: paired bootstrap of a - b over the folds where both were scored, and the verdict."""
    out = {}
    worlds = worlds or (W.EVAL_WORLDS + SECONDARY + SECONDARY_IDSSE)
    for w in worlds:
        rr = [r for r in fold_refs if (a, w, r.key) in res and (b, w, r.key) in res]
        if not rr:
            continue
        cmp = paired_bootstrap([res[(a, w, r.key)] for r in rr], [res[(b, w, r.key)] for r in rr])
        cmp["mean_a"] = float(np.mean([res[(a, w, r.key)]["spell_f1"] for r in rr]))
        cmp["mean_b"] = float(np.mean([res[(b, w, r.key)]["spell_f1"] for r in rr]))
        cmp["macro_a"] = float(np.mean([res[(a, w, r.key)]["macro_f1"] for r in rr]))
        cmp["macro_b"] = float(np.mean([res[(b, w, r.key)]["macro_f1"] for r in rr]))
        cmp["wins"] = int(sum(res[(a, w, r.key)]["spell_f1"] > res[(b, w, r.key)]["spell_f1"] for r in rr))
        cmp["worse"] = cmp["mean_diff"] < 0 and cmp["hi"] < 0
        cmp["better"] = adopted(cmp)
        out[w] = cmp
    return out


def keep_verdict(cmp: dict, cmp8: dict | None = None) -> dict:
    """The registered rule with amendment A1: keep a candidate mix iff no decision world is significantly
    worse, on all the folds given (``cmp``) or on their IDSSE + Metrica subset (``cmp8``)."""
    sets = [("all", cmp)] + ([("8", cmp8)] if cmp8 is not None else [])
    worse = [f"{w} ({tag})" for tag, c in sets for w in W.EVAL_WORLDS if w in c and c[w]["worse"]]
    better = [f"{w} ({tag})" for tag, c in sets for w in W.EVAL_WORLDS if w in c and c[w]["better"]]
    return {"keep": not worse, "worse": worse, "better": better,
            "missing": [w for w in W.EVAL_WORLDS if w not in cmp]}


def dead_label_subset(fold_refs: list) -> list:
    return [r for r in fold_refs if r.dataset in W.DEAD_LABEL_DATASETS]


# ---------------------------------------------------------------------------------------------
# simulated official baseline (M0) in every world

def m0_table(refs: list, worlds=M0_WORLDS, t_eval: int = T) -> dict:
    """{(world, window, match_key): score dict} for the official baseline algorithm, IDSSE and Metrica only.
    Corrupts afresh (deterministic; the existing worlds reproduce their cached condition exactly), so it
    needs the canonical matches only."""
    out = {}
    for r in refs:
        if r.dataset not in W.DEAD_LABEL_DATASETS:
            continue
        m = load_match(r.path)
        gt = relabel(m, t_eval).y
        m5 = downsample(m)
        step = int(m5.meta["step"])
        idx = np.concatenate([np.arange(a, b + 1) for _, a, b in m5.half_blocks])
        blk = np.zeros(m5.n, dtype=bool)
        blk[idx] = True
        for w in worlds:
            cfg = C.CONFIG.get(w) or W.config(w)
            mc = dg.corrupt(m5, cfg, P.seed_for(r.id, w))
            mw = m0_windows(mc)
            miss = ~np.isfinite(mc.ball[:, 0])
            dead = blk & (m5.y == D)
            for k, name in enumerate(M0_WINDOWS):
                pred = to_native_codes(mw[k][idx], idx, step, m.n)
                s = score(gt, pred)
                s.pop("flags")
                s["miss_dead"] = float(miss[dead].mean())
                s["miss_alive"] = float(miss[blk & (m5.y < D)].mean())
                s["miss_all"] = float(miss[blk].mean())
                out[(w, name, r.key)] = s
    return out


# ---------------------------------------------------------------------------------------------
# positional jitter, as the D5 audit measures it (A20; katena.diag.domain_gap on david/d5-audit)

def jitter_stats(m, window_s: float = 10.0) -> tuple[float, float]:
    """(player, ball) jitter of a work-grid match: the 2-D second difference of each present slot, capped at
    2 m and averaged over slots per sample, then averaged over 10 s windows of the in-play blocks (player);
    the ball's 2-D second difference, median per window (ball); each the median over windows."""
    n = m.n
    with np.errstate(invalid="ignore"):
        d2p = np.full((n, m.P.shape[1]), np.nan)
        d2p[1:-1] = np.linalg.norm(m.P[2:] - 2 * m.P[1:-1] + m.P[:-2], axis=-1)
        cnt = np.isfinite(d2p).sum(1)
        pl = np.where(cnt > 0, np.nansum(np.minimum(np.nan_to_num(d2p, nan=0.0), 2.0), 1) / np.maximum(cnt, 1), np.nan)
        bl = np.full(n, np.nan)
        bl[1:-1] = np.linalg.norm(m.ball[2:] - 2 * m.ball[1:-1] + m.ball[:-2], axis=-1)
    w = max(2, int(round(window_s * m.fps)))
    pw, bw = [], []
    for _, a, b in m.half_blocks:
        for s in range(a, b + 1 - w + 1, w):
            p, q = pl[s:s + w], bl[s:s + w]
            if np.isfinite(p).any():
                pw.append(np.nanmean(p))
            if np.isfinite(q).any():
                bw.append(np.nanmedian(q))
    return (float(np.median(pw)) if pw else float("nan"), float(np.median(bw)) if bw else float("nan"))


def save(obj, name: str) -> Path:
    import pickle
    p = root() / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(p)
    return p


def load(name: str):
    import pickle
    with open(root() / name, "rb") as fh:
        return pickle.load(fh)
