#!/usr/bin/env python3
"""Do the IDSSE and Metrica label mechanics carry over to real broadcast tracking? Information only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<the cache of an F3 gate>
    uv run python scripts/probe_transfer.py --inc f3_gbdt,f3_tcn

Registered in docs/decision-log.md (2026-11-13, "A team head that never saw a SkillCorner label, on the action
truth") before its numbers were read. External matches only; never reads ``data/`` or ``evaluation/gt/``. It
adopts nothing and no candidate comes out of it.

One LightGBM team head is fitted on the IDSSE and Metrica rows of the development pool only (the "7 + 0" member
of the probe of 2026-11-10, here with the 6 + 2 development matches): it has never seen a SkillCorner label of
either kind, nor a row of real broadcast tracking. No SkillCorner match is among its training rows, so the same
head serves all 20 SkillCorner folds out of fold. It replaces the team head of F3's LightGBM member; F3's alive
head and F3's TCN stay. The 20 SkillCorner folds are then scored against the action truth
(``katena.external.skillcorner_actions``), which neither that head nor F3 trained on, and against the standing
truth.

The alive head fitted here must equal F3's cached one on every SkillCorner fold (same rows): the script checks
it, and that is its test that the fit is F3's code path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_a as RA  # noqa: E402  (through run_g12: pins the CPU environment, loads libomp before torch and LightGBM)

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import gbdt  # noqa: E402

G = RA.G
T, N_TREES = RA.T, RA.N_TREES
log, HEAD, _row = G.log, G.HEAD, G._row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inc", default=f"{RA.INC_G},{RA.INC_T}", help="F3's cached members, as in run_a.py")
    a = ap.parse_args()
    inc_g, inc_t = a.inc.split(",")
    dev = F.dev_refs()
    sc_refs = [r for r in dev if r.is_sc]
    for r in sc_refs:
        RA.build_act(r)
    pools = F.FinalPools(dev)
    keep = ~np.char.startswith(pools.keys.astype(str), "skillcorner/")
    log(f"pool {pools.X.shape}; rows without SkillCorner: {int(keep.sum())}; team rows among them: "
        f"{int((pools.y[T][keep] < 2).sum())}")
    heads = gbdt.fit_heads(pools.X[keep], pools.o[keep], pools.y[T][keep], pools.dead_ok[keep], pools.cols(),
                           n_trees=N_TREES)
    del pools
    src = F.PriorSource(dev)
    sc, alive_same = {}, []
    for r in sc_refs:
        g0, t0 = Lm.load_oof(inc_g, T, r), Lm.load_oof(inc_t, T, r)
        pri, gt, gta = B.priors_for(src, {r.key}, T), C.load_gt(r, T), RA.load_act(r)
        for c in F.GATE_CONDS:
            _, X0, X1 = TG.features(r, c)
            pa, ph = G.predict_at(heads, X0, X1, N_TREES)
            alive_same.append(pa.tobytes() == g0[f"pa_{c}"].tobytes())
            n0 = {**g0, f"ph_{c}": ph}
            for name, g, w in (("F3", g0, F.W_F), ("N", n0, F.W_F), ("F3G", g0, 0.0), ("NG", n0, 0.0)):
                sc[(name, c, r.key)] = G.blend_score(r, c, g, t0, w, pri, gt)
                sc[(name + "@act", c, r.key)] = RA.score_on(r, c, g, t0, w, pri, gta)
        log(f"{r.key}: scored")
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    out = {"folds": len(sc_refs), "trees": N_TREES, "alive_head_equals_F3": [int(sum(alive_same)), len(alive_same)],
           "rows": {}}
    print(f"\n## A team head that never saw SkillCorner (N), in F3's place, {len(sc_refs)} SkillCorner folds\n" + HEAD)
    for label, cand, inc in (("blended with F3's TCN, action truth", "N@act", "F3@act"),
                             ("LightGBM member alone, action truth", "NG@act", "F3G@act"),
                             ("blended with F3's TCN, standing truth", "N", "F3"),
                             ("LightGBM member alone, standing truth", "NG", "F3G")):
        for c in F.GATE_CONDS:
            g = F.compare(get, sc_refs, cand, inc, c)
            out["rows"][f"{label}/{c}"] = g
            print(_row(f"N vs F3, {c}, {label}", g))
    print("\n| truth | model | team accuracy | predicted possessions per true one |\n|---|---|---|---|")
    for suffix, truth in (("@act", "action"), ("", "standing")):
        for name in ("F3", "N"):
            s = [get(name + suffix, "target", r) for r in sc_refs]
            v = {"team_acc": float(np.mean([x["team_acc"] for x in s])),
                 "per_true": sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)}
            out["rows"][f"parts/{truth}/{name}"] = v
            print(f"| {truth} | {name} | {v['team_acc']:.3f} | {v['per_true']:.3f} |")
    print(f"\nchecks: alive head equals F3's cached one on {sum(alive_same)} of {len(alive_same)} fold-conditions")
    G._dump(P.WORK / "probe_transfer.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
