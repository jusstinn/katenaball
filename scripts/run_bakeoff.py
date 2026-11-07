#!/usr/bin/env python
"""Run the architecture bake-off end to end:  uv run python scripts/run_bakeoff.py

Stages are cached under work/bakeoff/ and resumable (``--fresh`` clears the cache). Uses external
matches only; never touches data/ or evaluation/gt/. Protocol: docs/bakeoff-protocol.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()                      # may re-exec once (no libomp without Homebrew)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402
from katena.eval.lomo import Pools, Results, timed  # noqa: E402

import time  # noqa: E402


CACHE_MARKERS = {"results.pkl", "feat", "pool", "gt", "oof"}      # what a bake-off cache holds


def wipe_refusal(work: Path) -> str | None:
    """Why ``--fresh`` must not delete ``work``, or None. KATENA_WORK can point anywhere, so only an empty
    directory or one that looks like a bake-off cache is deleted, never one holding the corpus or the repo."""
    work = work.resolve()
    for name, other in (("the external corpus", P.EXTERNAL), ("the repository", REPO)):
        other = other.resolve()
        if work == other or work in other.parents:
            return f"{work} contains {name} ({other})"
    entries = {e.name for e in work.iterdir()}
    if entries and not entries & CACHE_MARKERS:
        return f"{work} does not look like a bake-off cache (none of {sorted(CACHE_MARKERS)})"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete the bake-off cache (KATENA_WORK) first")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip: tcn m6 sweep bench")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.fresh and P.WORK.exists():
        why = wipe_refusal(P.WORK)
        if why:
            sys.exit(f"--fresh refused: {why}")
        shutil.rmtree(P.WORK)
    P.WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    C.build_all(workers=a.workers)                       # skips matches already built
    dev, locked = B.dev_and_locked()
    print("locked benches:", [r.key for r in locked], "| development:", len(dev), flush=True)
    res = Results()
    pools = Pools(dev)
    T = P.T_MAIN
    abl = B.ablation_folds(dev)

    B.stage_baselines(res, dev)
    B.stage_lomo(pools, dev, "full", T)
    for v in ("nofree", "freeonly", "nodead", "nomiss"):
        B.stage_lomo(pools, dev, v, T, folds=abl, conds=["target", "mild", "severe"])
    extra = tuple(C.EXTRA_CONDS)
    B.decode_and_score(res, pools, dev, "full", T, "viterbi", "M2", extra_conds=extra)
    B.decode_and_score(res, pools, dev, "full", T, "hsmm", "M3", extra_conds=extra)
    for v in ("nofree", "freeonly", "nodead", "nomiss"):
        B.decode_and_score(res, pools, abl, v, T, "hsmm", f"M4_{v}", conds=("mild", "target", "severe"))

    ids = [r for r in dev if r.dataset == "idsse"]
    B.stage_lomo(pools, dev, "nomissw", T, folds=ids, conds=["target", "target_dead3", "target_dead15"])
    B.decode_and_score(res, pools, ids, "nomissw", T, "hsmm", "M4_nomissw", conds=("target",),
                       extra_conds=("target_dead3", "target_dead15"))

    if "sweep" not in a.skip:
        S.stage_sweep(pools, res, dev)
    if "tcn" not in a.skip:
        S.stage_tcn(pools, res, dev)
        f8 = S.FOLDS8(dev)
        m2 = np.mean([res.get(("M2", T, P.HEADLINE, r.key))["spell_f1"] for r in dev])
        m3 = np.mean([res.get(("M3", T, P.HEADLINE, r.key))["spell_f1"] for r in dev])
        kind = S._kind_m5(res, T)
        B.decode_and_score(res, pools, f8, "M5", T, kind, "M5")
        S.stage_tcn(pools, res, dev, seed=1, variant="M5s1")          # second seed: is M5 vs M3 stable?
        B.decode_and_score(res, pools, f8, "M5s1", T, kind, "M5s1")
    if "m6" not in a.skip:
        S.stage_m6_heads(pools, dev)
        S.stage_m6_score(pools, res, dev)
    S.stage_oracles(pools, res, dev)
    census = S.stage_census(pools, res, dev)
    if "bench" not in a.skip:
        S.stage_bench(pools, res, dev, locked)
        if "tcn" not in a.skip:
            S.stage_bench_tcn(pools, res, dev, locked)
    import pickle
    pickle.dump(census, open(P.WORK / "census.pkl", "wb"))
    timed("all stages done", t0)


if __name__ == "__main__":
    main()
