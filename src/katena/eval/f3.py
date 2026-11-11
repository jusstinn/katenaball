"""F3: F with a 1,200-tree LightGBM member (``T1200``) and a 4,800-step TCN member that also reads the 53 touch
columns (``LT``). Both changes passed F's rule on Justin's Mac (docs/decision-log.md, 2026-11-09 and 2026-11-10,
``scripts/run_g12.py``). This module is the second-machine gate of the pair, on the Windows PC.

Pre-registered in docs/decision-log.md (2026-11-11, "F3 on the Windows PC (pre-registration)", commit 005cd6c)
before any Windows score of either member. In fold k both members train on the same 27 development matches:

* ``F3G`` (F3 at w = 0): ``gbdt.fit_heads(..., n_trees=1200)`` on F's ``d2j`` pool and 213 columns. LightGBM's
  rounds are sequential, so its first 150 trees are F's member; ``stage_gbdt`` checks that byte for byte.
* ``F3T`` (F3 at w = 1): the plan's G1 code (``katena.eval.g1``: sequences with the touch columns of the same
  world and orientation, the scaler over all 213 columns) at 4,800 steps instead of 1,200.
* ``F3``: ``ensemble.blend_heads(F3G, F3T, 0.5)``, Viterbi at tau = 2, priors without the held-out match.
* ``T1200`` and ``LT`` (information): one change each, the other member F's.

The incumbent F is not refitted: its stored scores and probabilities are copied from the F cache
(``g1.seed_cache``, SHA-256 checked) and re-decoded as a check. Never reads ``data/`` or ``evaluation/gt/``, never
builds or scores a locked bench.
"""

from __future__ import annotations

import time

import numpy as np

from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import d3
from katena.eval import final as F
from katena.eval import g1 as G1
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.models import ensemble as E
from katena.models import gbdt

T = F.T
N_TREES = 1200                  # T1200, registered on Justin's branch (2026-11-09)
STEPS = 4800                    # LT, registered there too (F: tcn.STEPS = 1200)
VAR_G = "f3_gbdt"
VAR_T = "f3_tcn"
RES_NAME = "f3_results.pkl"
# name -> (LightGBM probabilities, TCN probabilities, TCN weight)
BLENDS = {"F3": (VAR_G, VAR_T, F.W_F), "F3G": (VAR_G, VAR_T, 0.0), "F3T": (VAR_G, VAR_T, 1.0),
          "T1200": (VAR_G, F.VAR_T, F.W_F), "LT": (F.VAR_G, VAR_T, F.W_F)}
F_NAMES = {F.W_F: "F", 0.0: "FG", 1.0: "FT"}


def predict_at(heads: gbdt.Heads, X0: np.ndarray, X1: np.ndarray, n: int) -> tuple:
    """``Heads.predict`` using the first ``n`` trees (``run_g12.predict_at``)."""
    c = heads.cols
    pa = 0.5 * (heads.alive.predict(X0[:, c], num_iteration=n) + heads.alive.predict(X1[:, c], num_iteration=n))
    ph = 0.5 * (heads.team.predict(X0[:, c], num_iteration=n) + 1.0 - heads.team.predict(X1[:, c], num_iteration=n))
    return pa.astype(np.float32), ph.astype(np.float32)


def stage_gbdt(pools: F.FinalPools, folds: list) -> dict:
    """The 1,200-tree member of every fold. ``first150_is_F`` in each file: its first 150 trees give F's cached
    member byte for byte in every evaluation world of the fold."""
    cols = pools.cols()
    done = 0
    for r in folds:
        out = Lm.oof_path(VAR_G, T, r)
        if out.exists():
            continue
        t0 = time.time()
        m = ~np.isin(pools.keys, [r.key])
        heads = gbdt.fit_heads(pools.X[m], pools.o[m], pools.y[T][m], pools.dead_ok[m], cols, n_trees=N_TREES)
        t1 = time.time()
        old = Lm.load_oof(F.VAR_G, T, r)
        arrays, same = {}, True
        for c in F.eval_worlds(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = predict_at(heads, X0, X1, N_TREES)
            pa, ph = predict_at(heads, X0, X1, gbdt.N_TREES)
            same &= pa.tobytes() == old[f"pa_{c}"].tobytes() and ph.tobytes() == old[f"ph_{c}"].tobytes()
        arrays["model_sha"] = np.array(F.gbdt_hash(heads))
        arrays["first150_is_F"] = np.array(bool(same))
        d3._save(out, arrays)
        done += 1
        print(f"[{time.strftime('%H:%M:%S')}] LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s, "
              f"first {gbdt.N_TREES} trees are F's member: {bool(same)}", flush=True)
    flags = [bool(Lm.load_oof(VAR_G, T, r)["first150_is_F"]) for r in folds if Lm.oof_path(VAR_G, T, r).exists()]
    return {"folds": done, "first150_is_F": int(sum(flags)), "of": len(flags)}


def stage_tcn(dev: list, folds: list, steps: int = STEPS) -> dict:
    """The 4,800-step TCN on 213 columns for every fold: ``g1.stage_tcn`` with more steps."""
    return G1.stage_tcn(dev, folds, steps=steps, variant=VAR_T)


def results(name: str = RES_NAME) -> Lm.Results:
    return Lm.Results(P.WORK / name)


def f_store() -> dict:
    """F's stored scores (the copy of the F cache's ``final_results.pkl``), read only."""
    return F.load_store(P.WORK / F.RES_NAME)


def stage_score(res: Lm.Results, dev: list, folds: list, store: dict) -> dict:
    """Every name in ``BLENDS`` in every evaluation world of each fold, priors without the held-out match. F, FG
    and FT are re-decoded from the copied probabilities and compared with ``store`` (F's stored scores)."""
    src = F.PriorSource(dev)
    chk = {"n": 0, "F_equal": 0, "FG_equal": 0, "FT_equal": 0}
    for r in folds:
        if not all(Lm.oof_path(v, T, r).exists() for v in (VAR_G, VAR_T)):
            continue                                            # a partial run scores the folds it has
        oof = {v: Lm.load_oof(v, T, r) for v in (F.VAR_G, F.VAR_T, VAR_G, VAR_T)}
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in F.eval_worlds(r):
            p = {v: (o[f"pa_{c}"], o[f"ph_{c}"]) for v, o in oof.items()}
            for name, (vg, vt, w) in BLENDS.items():
                pa, ph = E.blend_heads(p[vg], p[vt], w)
                res.put((name, T, c, r.key), F.decode_score(r, c, pa, ph, pri, gt))
            chk["n"] += 1
            for w, name in F_NAMES.items():
                pa, ph = E.blend_heads(p[F.VAR_G], p[F.VAR_T], w)
                chk[f"{name}_equal"] += int(F.same_score(F.decode_score(r, c, pa, ph, pri, gt),
                                                         store.get((name, T, c, r.key))))
    res.save()
    return chk


def passes(g_target: dict, g_severe: dict, g_sc_target: dict | None) -> bool:
    """Registered: F's rule on all folds (adopted at target, not worse at severe) and not worse at target on the
    SkillCorner folds alone."""
    return bool(F.decision(g_target, g_severe) and not (g_sc_target is not None and g_sc_target["worse"]))
