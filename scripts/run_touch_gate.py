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
    """What decode_and_score needs from a pool: the label sequences for the decoder priors."""

    def __init__(self, refs):
        self.refs = refs

    label_seqs = Lm.Pools.label_seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=("build", "check", "screen", "gate"))
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
    pri = _Priors(dev)
    conds = B.ALL_CONDS
    if stage == "screen":
        B.decode_and_score(res, pri, folds, "full", T, "viterbi", "M2_f8")
        B.decode_and_score(res, pri, folds, TG.VARIANT, T, "viterbi", "M2T_f8", conds=conds + (sm,))
        B.decode_and_score(res, pri, folds, "m2refit", T, "viterbi", "M2r_f8", conds=("target", sm))
        same = all(res.get(("M2r_f8", T, "target", r.key))["spell_f1"] == res.get(("M2_f8", T, "target", r.key))["spell_f1"]
                   for r in folds)
        print(f"M2 refit from the joined pool reproduces the cached M2 on all {len(folds)} folds: {same}", flush=True)
        print(TG.summary(res, "M2T_f8", "M2_f8", folds), flush=True)
        print(TG.summary(res, "M2T_f8", "M2r_f8", folds, conds=(sm,)), flush=True)
    else:
        B.decode_and_score(res, pri, folds, TG.VARIANT, T, "viterbi", TG.MODEL, extra_conds=tuple(C.EXTRA_CONDS))
        print(TG.summary(res, TG.MODEL, "M2", folds), flush=True)
    Lm.timed(f"{stage} done", t0)


if __name__ == "__main__":
    main()
