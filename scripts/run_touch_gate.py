#!/usr/bin/env python
"""D1 gate: M2 + the native-frame-rate touch group against M2, on the bake-off cache.

Needs the external corpus and a copy of the bake-off cache (``scripts/run_bakeoff.py`` output: gt/, pool/,
feat/, oof/full/, results.pkl) in ``KATENA_WORK``. External matches only; never touches ``data/``,
``evaluation/gt/`` or the locked benches (development matches only, and no bench stage).

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-d1
    uv run python scripts/run_touch_gate.py build                 # touch columns for all 28 development matches
    uv run python scripts/run_touch_gate.py check                 # M2 refit from the joined pool == cached M2 (one fold)
    uv run python scripts/run_touch_gate.py screen                # 8 IDSSE + Metrica folds
    uv run python scripts/run_touch_gate.py gate                  # all 28 development folds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402

T = P.T_MAIN


class _Priors:
    """What the decoders need from a pool: the label sequences for the decoder priors."""

    def __init__(self, refs):
        self.refs = refs

    label_seqs = Lm.Pools.label_seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=("build", "check", "screen", "gate", "rate", "worlds"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    for stage in a.stages:
        run(stage, a.workers)


def run(stage: str, workers: int) -> None:
    dev, locked = B.dev_and_locked()
    assert not ({r.key for r in locked} & {r.key for r in dev})
    t0 = time.time()
    if stage == "build":
        TG.build_all(dev, workers=workers)
        Lm.timed("touch build", t0)
        return
    res = Lm.Results()
    if stage == "rate":
        # Control for the native frame rate, on the 8 screen folds: the same touch code run on the corrupted
        # 5 Hz ball ("touch5"), trained and decoded exactly like the native-rate group.
        f8, sm = S.FOLDS8(dev), TG.SMOOTH
        TG.build_all(dev, workers=workers, kind="touch5")
        for r in f8:
            TG.build_extra(r, sm, kind="touch5")
        pools = TG.TouchPools(dev, kind="touch5")
        TG.stage_lomo(pools, f8, variant="touch5", extra=(sm,))
        del pools
        TG.decode_fixed(res, _Priors(dev), f8, "touch5", "M2T5_t2", B.ALL_CONDS + (sm,))
        print("== 8 IDSSE + Metrica folds: touch features from the 5 Hz ball vs M2 ==\n"
              + TG.summary(res, "M2T5_t2", "M2_t2", f8), flush=True)
        print("== native rate vs 5 Hz, same features ==\n" + TG.summary(res, "M2T_t2", "M2T5_t2", f8, conds=("target", "severe", "mild", "native", sm)),
              flush=True)
        print(f"== {sm}: 5 Hz touch vs M2 refit ==\n" + TG.summary(res, "M2T5_t2", "M2r_t2", f8, conds=(sm,)), flush=True)
        Lm.timed("rate done", t0)
        return
    if stage == "worlds":
        # Robustness read in D2's evaluation worlds (needs katena.sim.worlds from origin/david/d2-degrader):
        # ball loss coupled to dead ball at half and full Metrica strength, and SkillCorner-matched jitter.
        # Evaluation only, 8 IDSSE + Metrica folds; heads refit per fold exactly as in the gate.
        f8 = S.FOLDS8(dev)
        worlds = ["target_cpl50", "target_cpl100", "target_smooth"]
        for r in f8:
            for w in worlds:
                TG.build_extra(r, w)
        Lm.timed("D2-world features", t0)
        pools = TG.TouchPools(dev)
        TG.stage_lomo(pools, f8, variant="touch_w", conds=worlds)
        TG.stage_lomo(pools, f8, variant="m2refit_w", drop=(TG.TF.GROUP,), conds=worlds)
        del pools
        TG.decode_fixed(res, _Priors(dev), f8, "touch_w", "M2T_w", tuple(worlds))
        TG.decode_fixed(res, _Priors(dev), f8, "m2refit_w", "M2r_w", tuple(worlds))
        print("== D2 worlds, 8 IDSSE + Metrica folds, evaluation only ==")
        print(TG.summary(res, "M2T_w", "M2r_w", f8, conds=tuple(worlds)), flush=True)
        Lm.timed("worlds done", t0)
        return
    if stage == "check":
        r = S.FOLDS8(dev)[0]
        pools = TG.TouchPools(dev)
        TG.stage_lomo(pools, [r], variant="touch_m2check", drop=(TG.TF.GROUP,))
        new, old = Lm.load_oof("touch_m2check", T, r), Lm.load_oof("full", T, r)
        same = all(np.array_equal(new[k], old[k]) for k in old if k in new)
        print(f"M2 refit from the joined pool on {r.key}: identical to the cached M2 = {same}; "
              f"max |dp| = {max(float(np.abs(new[k] - old[k]).max()) for k in old if k in new):.2e}", flush=True)
        return
    folds = S.FOLDS8(dev) if stage == "screen" else dev
    sm = TG.SMOOTH
    if stage == "screen":                                     # evaluation-only smooth-ball world
        for r in folds:
            TG.build_extra(r, sm)
        Lm.timed(f"{sm} features", t0)
    pools = TG.TouchPools(dev)
    print(f"joined pool: {pools.X.shape[0]} rows x {pools.X.shape[1]} columns "
          f"({int((pools.groups == TG.TF.GROUP).sum())} touch)", flush=True)
    if stage == "screen":
        TG.stage_lomo(pools, folds, extra=(sm,))
        TG.stage_lomo(pools, folds, variant="m2refit", drop=(TG.TF.GROUP,), conds=["target", sm])
    else:
        TG.stage_lomo(pools, folds)
    del pools
    # Decoding at a fixed tau = 2 (D5 audit): per-fold tau selection would use folds whose models saw the
    # held-out match. With a fixed tau every fold decodes on its own, so the 8-fold numbers are a subset of
    # the 28-fold ones.
    pri = _Priors(dev)
    conds = B.ALL_CONDS + tuple(C.EXTRA_CONDS)
    TG.decode_fixed(res, pri, folds, "full", "M2_t2", conds)
    same = all(res.get(("M2_t2", T, c, r.key))["spell_f1"] == res.get(("M2", T, c, r.key))["spell_f1"]
               for r in folds for c in ("target", "severe") if res.get(("M2", T, c, r.key)))
    print(f"M2 at fixed tau 2 equals the cached M2 (tau selected per fold) on these folds: {same}", flush=True)
    TG.decode_fixed(res, pri, folds, TG.VARIANT, "M2T_t2", conds + (sm,))
    f8 = S.FOLDS8(dev)
    if stage == "screen":
        TG.decode_fixed(res, pri, folds, "m2refit", "M2r_t2", ("target", sm))
        same = all(res.get(("M2r_t2", T, "target", r.key))["spell_f1"] == res.get(("M2_t2", T, "target", r.key))["spell_f1"]
                   for r in folds)
        print(f"M2 refit from the joined pool reproduces the cached M2 on all {len(folds)} folds: {same}", flush=True)
        print("== 8 IDSSE + Metrica folds ==\n" + TG.summary(res, "M2T_t2", "M2_t2", f8), flush=True)
        print(f"== smooth-ball world ({sm}, evaluation only) ==\n" + TG.summary(res, "M2T_t2", "M2r_t2", f8, conds=(sm,)),
              flush=True)
    else:
        print(f"== all {len(folds)} development folds ==\n" + TG.summary(res, "M2T_t2", "M2_t2", folds), flush=True)
        print("== 8 IDSSE + Metrica folds ==\n" + TG.summary(res, "M2T_t2", "M2_t2", f8), flush=True)
        print("== IDSSE extra worlds (6 folds) ==\n" + TG.summary(res, "M2T_t2", "M2_t2", [r for r in f8 if r.dataset == "idsse"],
                                                                 conds=tuple(C.EXTRA_CONDS)), flush=True)
    Lm.timed(f"{stage} done", t0)


if __name__ == "__main__":
    main()
