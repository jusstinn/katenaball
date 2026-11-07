#!/usr/bin/env python
"""Shared base cache: build the bake-off cache and score M2 on the 28 development folds.

    KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-base \
        .venv/Scripts/python.exe scripts/run_m2.py --workers 3 --require-build-done --ready

This is the part of scripts/run_bakeoff.py that the M2 headline needs, and nothing else:

1. ``corpus.build_all``: ground truth under every T, the corrupted conditions, features and the
   training pool for every external match (skips matches already built);
2. ``stage_lomo`` for the 'full' feature set: one pair of LightGBM heads per development fold;
3. ``decode_and_score`` M2: Viterbi, tau for match k chosen from the other folds (protocol 5);
4. ``stage_baselines``: the M0/M1 references (cheap, no training).

No ablations, label sweep, TCN, M6, oracles or locked benches. External matches only: nothing here
reads data/ or evaluation/gt/. With --ready, it writes $KATENA_WORK/READY (a small JSON: M2 mean
spellF1, folds, git commit, runtime) after M2 is scored. READY marks the cache as final: this script
refuses to touch a work directory that already has one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()                      # no-op where LightGBM loads; may re-exec once on macOS

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval.lomo import Pools, Results, timed  # noqa: E402

REFERENCE = {"spell_f1": 0.586, "source": "docs/bakeoff.md Table 1 (M2, 28 folds, target-like, T = 2)"}
EXPECTED_LOCKED = {"idsse/DFL-MAT-J03WQQ", "metrica/Sample_Game_2"}
EXPECTED_DEV = {"idsse": 6, "metrica": 2, "skillcorner": 20}


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable ({type(error).__name__})"


def check_split(dev: list, locked: list) -> None:
    """The protocol's split, or stop: a missing or half-written match would silently move the
    locked IDSSE bench (it is the smallest hash among the matches present) into training."""
    got_locked = {r.key for r in locked}
    got_dev = Counter(r.dataset for r in dev)
    if got_locked != EXPECTED_LOCKED or dict(got_dev) != EXPECTED_DEV:
        raise SystemExit(f"unexpected corpus split: locked {sorted(got_locked)}, development "
                         f"{dict(got_dev)}; expected {sorted(EXPECTED_LOCKED)} and {EXPECTED_DEV}")


def summarise(res: Results, dev: list, taus: dict, T: int) -> dict:
    def mean(model: str, cond: str, refs: list, key: str = "spell_f1") -> float | None:
        v = [res.get((model, T, cond, r.key)) for r in refs]
        v = [x[key] for x in v if x]
        return round(float(np.mean(v)), 4) if v else None

    by_ds = {ds: [r for r in dev if r.dataset == ds] for ds in EXPECTED_DEV}
    not_sc = [r for r in dev if not r.is_sc]
    out = {
        "m2_spell_f1_mean": mean("M2", P.HEADLINE, dev),
        "m2_macro_f1_mean_idsse_metrica": mean("M2", P.HEADLINE, not_sc, "macro_f1"),
        "m2_spell_f1_by_dataset": {ds: mean("M2", P.HEADLINE, refs) for ds, refs in by_ds.items()},
        "m2_spell_f1_by_condition": {c: mean("M2", c, dev) for c in ("native", "mild", "target", "severe")},
        "m2_idsse_extra_conditions": {c: mean("M2", c, by_ds["idsse"]) for c in C.EXTRA_CONDS},
        "m0_spell_f1_mean": mean("M0", P.HEADLINE, dev),
        "m1_spell_f1_mean": mean("M1", P.HEADLINE, dev),
        "tau_chosen_counts": {str(k): v for k, v in sorted(Counter(taus.values()).items())},
    }
    tab = res.get(("tau_table", T, "M2", "all"))
    if tab:
        out["m2_spell_f1_by_fixed_tau"] = {str(t): round(float(v), 4)
                                           for t, v in zip(P.TAUS, tab["table"].mean(1))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3, help="processes for the corpus build")
    ap.add_argument("--no-baselines", action="store_true", help="skip the M0/M1 references")
    ap.add_argument("--ready", action="store_true", help="write $KATENA_WORK/READY when M2 is scored")
    ap.add_argument("--require-build-done", action="store_true",
                    help="stop unless the corpus root has a BUILD_DONE marker (shared-machine convention)")
    a = ap.parse_args()
    if (P.WORK / "READY").exists():
        raise SystemExit(f"{P.WORK / 'READY'} exists: this cache is final, use another KATENA_WORK")
    if a.require_build_done and not (P.EXTERNAL / "BUILD_DONE").exists():
        raise SystemExit(f"{P.EXTERNAL} has no BUILD_DONE marker: the corpus is not finished")
    P.WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = P.T_MAIN
    print(f"corpus {P.CANON} | work {P.WORK} | commit {git('rev-parse', 'HEAD')}", flush=True)

    dev, locked = B.dev_and_locked()
    check_split(dev, locked)                              # before anything is written
    C.build_all(workers=a.workers)
    timed("corpus cache built", t0)
    stages = ["corpus.build_all"]

    res = Results()
    pools = Pools(dev)
    B.stage_lomo(pools, dev, "full", T)
    stages.append("stage_lomo full")
    taus = B.decode_and_score(res, pools, dev, "full", T, "viterbi", "M2", extra_conds=tuple(C.EXTRA_CONDS))
    stages.append("decode_and_score M2")
    if not a.no_baselines:
        B.stage_baselines(res, dev)
        stages.append("stage_baselines")

    summary = summarise(res, dev, taus, T)
    summary.update({
        "folds": len(dev), "T": T, "condition": P.HEADLINE, "decoder": "viterbi",
        "reference": REFERENCE,
        "diff_vs_reference": (round(summary["m2_spell_f1_mean"] - REFERENCE["spell_f1"], 4)
                              if summary["m2_spell_f1_mean"] is not None else None),
        "stages": stages, "runtime_s": round(time.time() - t0, 1),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "corpus": str(P.EXTERNAL), "work": str(P.WORK),
        "platform": platform.platform(), "python": platform.python_version(),
        "lightgbm_num_threads": gbdt.PARAMS["num_threads"],
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    text = json.dumps(summary, indent=1)
    (P.WORK / "m2_summary.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if a.ready:
        (P.WORK / "READY").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {P.WORK / 'READY'}", flush=True)
    timed("run_m2 done", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
