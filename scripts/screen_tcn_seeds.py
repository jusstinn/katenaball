#!/usr/bin/env python3
"""Screen F2 (two TCN seeds) against F on the 8 IDSSE + Metrica development folds. External data only.

    uv run python scripts/screen_tcn_seeds.py          # KATENA_WORK=work/final (needs `make model` first)

Pre-registered in docs/decision-log.md ("Screen: F2, two TCN seeds"). LOMO on the 27 other development
matches (the benches are never trained on or scored). Cached per fold under oof/screen_*; prints means,
the paired bootstrap and the rule's verdict.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("KATENA_WORK", str(REPO / "work" / "final"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()
from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval.bootstrap import paired_bootstrap  # noqa: E402
from katena.sim import worlds as W  # noqa: E402

T = P.T_MAIN
CONDS = ("target", "severe")
SEEDS = (0, 1)
VG, VT = "screen_gbdt", "screen_tcn_s{}"


def main() -> int:
    dev = F.dev_refs()
    folds = [r for r in dev if r.dataset in W.DEAD_LABEL_DATASETS]
    assert len(folds) == 8 and not {r.key for r in P.split(P.all_refs())[1]} & {r.key for r in dev}
    E.set_deterministic(E.TCN_THREADS)
    pools = F.FinalPools(dev)
    print(f"pool: {pools.X.shape} from {len(dev)} development matches", flush=True)
    cols = pools.cols()
    for r in folds:                                         # LightGBM member, as F.stage_gbdt
        out = Lm.oof_path(VG, T, r)
        if not out.exists():
            heads = pools.fit({r.key}, T, cols)
            d3._save(out, F.predict_gbdt(heads, r, CONDS))
    for s in SEEDS:                                         # TCN member, one seed at a time, as F.stage_tcn
        todo = [r for r in folds if not Lm.oof_path(VT.format(s), T, r).exists()]
        if not todo:
            continue
        seqs = F.tcn_seqs(dev, T, s)
        for r in todo:
            t0 = time.time()
            model, sc = F.fit_tcn(pools, seqs, {r.key}, s, E.TCN_THREADS)
            d3._save(Lm.oof_path(VT.format(s), T, r), d3.predict_tcn(model, sc, r, CONDS, E.TCN_THREADS))
            print(f"[{time.strftime('%H:%M:%S')}] TCN seed {s} fold {r.key}: {time.time() - t0:.0f}s", flush=True)
        del seqs
    src = F.PriorSource(dev)
    scores = {("F", c): [] for c in CONDS} | {("F2", c): [] for c in CONDS}
    for r in folds:
        og, t0, t1 = (Lm.load_oof(v, T, r) for v in (VG, VT.format(0), VT.format(1)))
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in CONDS:
            g = (og[f"pa_{c}"], og[f"ph_{c}"])
            tc0 = (t0[f"pa_{c}"], t0[f"ph_{c}"])
            tc2 = E.blend_heads(tc0, (t1[f"pa_{c}"], t1[f"ph_{c}"]), 0.5)
            for name, tcn in (("F", tc0), ("F2", tc2)):
                pa, ph = E.blend_heads(g, tcn, F.W_F)
                scores[(name, c)].append(F.decode_score(r, c, pa, ph, pri, gt))
    verdict = {}
    for c in CONDS:
        a, b = scores[("F2", c)], scores[("F", c)]
        bs = paired_bootstrap(a, b)
        wins = int(sum(x["spell_f1"] > y["spell_f1"] for x, y in zip(a, b)))
        verdict[c] = (bs, wins)
        print(f"{c}: F {np.mean([x['spell_f1'] for x in b]):.4f}  F2 {np.mean([x['spell_f1'] for x in a]):.4f}  "
              f"diff {bs['mean_diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}]  wins {wins}/8", flush=True)
    bt, wt = verdict["target"]
    bsv, _ = verdict["severe"]
    promising = wt >= 6 and bt["lo"] > 0 and not (bsv["mean_diff"] < 0 and bsv["hi"] < 0)
    print(f"verdict: {'PROMISING, run the 28-fold gate' if promising else 'DROP'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
