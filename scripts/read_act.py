"""Cached models that never trained on an action label, read against SkillCorner's action truth. Information only.

Decision log, 2026-11-13 ("K at kind 0, S1 and S2 against the action truth"). No model is fitted. On the 20
SkillCorner development folds, at ``target`` and ``severe``, the out-of-fold probabilities already in the cache are
decoded as F3's are and scored against the action labels of ``scripts/run_a.py`` (``KATENA_WORK/gt_act``), the way an
IDSSE fold is scored (predicted dead ball stays dead ball):

* F3:  ``g1_gbdt1200`` + ``g2_tcn4800t``.
* K0:  K's two members (``k1_gbdt1200k``, ``k2_tcn4800tk``) with the label-kind column read at 0, the value of the
       IDSSE and Metrica rows. ``run_k.py`` stored that reading as ``phx`` (its "Kx") on SkillCorner folds.
* K:   the same members at the fold's own value (1).
* S1:  ``s1_gbdt1200w`` + F3's TCN.  S2: F3's LightGBM + ``s2_tcn4800tw``.
* A1:  A's LightGBM member + F3's TCN (trained on the action labels; already read in A's interim gate).
* A:   A's two members, only when A's TCN is cached for all 20 folds.

Usage: ``uv run python scripts/read_act.py [--dry]``. ``--dry`` checks that the files and keys are there and prints
no score. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_a as RA  # noqa: E402  (pins the CPU environment through run_g12)

from run_a import B, F, G, HEAD, Lm, T, _row, log  # noqa: E402

S1_G, S2_T = "s1_gbdt1200w", "s2_tcn4800tw"
A_G, A_T = "a_gbdt1200a", "a_tcn4800ta"
PAIRS = (("K0", "F3", "K read at kind 0 vs F3"), ("K", "F3", "K read at its own kind vs F3"),
         ("K0", "K", "K at kind 0 vs K at its own kind"), ("S1", "F3", "S1 (weighted LightGBM team head) vs F3"),
         ("S2", "F3", "S2 (weighted TCN team loss) vs F3"), ("A1", "K0", "A1 (interim) vs K at kind 0"),
         ("A", "K0", "A vs K at kind 0"))


def other_kind(o: dict) -> dict:
    """K's probabilities with the team head read at the other value of the column (``run_k._other``)."""
    return {**o, **{f"ph_{c}": o[f"phx_{c}"] for c in F.GATE_CONDS}}


def members(r) -> dict | None:
    """{model: (LightGBM probabilities, TCN probabilities)} of one fold, or None when a needed file is missing."""
    need = (RA.INC_G, RA.INC_T, RA.K_G, RA.K_T, S1_G, S2_T, A_G)
    if not all(Lm.oof_path(v, T, r).exists() for v in need):
        return None
    g0, t0, kg, kt, sg, st, ag = (Lm.load_oof(v, T, r) for v in need)
    out = {"F3": (g0, t0), "K0": (other_kind(kg), other_kind(kt)), "K": (kg, kt), "S1": (sg, t0), "S2": (g0, st),
           "A1": (ag, t0)}
    if Lm.oof_path(A_T, T, r).exists():
        out["A"] = (ag, Lm.load_oof(A_T, T, r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="check files and keys only; print no score")
    a = ap.parse_args()
    dev = F.dev_refs()
    refs = [r for r in dev if r.is_sc]
    src = F.PriorSource(dev)
    sc, with_a = {}, 0
    for r in refs:
        m = members(r)
        if m is None:
            log(f"fold {r.key}: a cached member is missing")
            continue
        with_a += "A" in m
        if a.dry:
            ok = all(f"{h}_{c}" in o for g, t in m.values() for o in (g, t) for c in F.GATE_CONDS for h in ("pa", "ph"))
            sc[("dry", "target", r.key)] = ok
            continue
        pri, gta = B.priors_for(src, {r.key}, T), RA.load_act(r)
        for c in F.GATE_CONDS:
            for name, (g, t) in m.items():
                sc[(name, c, r.key)] = RA.score_on(r, c, g, t, F.W_F, pri, gta)
    if a.dry:
        print(f"SkillCorner folds {len(refs)}; folds with every member {len(sc)}; keys complete on "
              f"{sum(sc.values())}; folds with A's TCN {with_a}")
        return 0
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    done = [r for r in refs if get("F3", "target", r) is not None]
    out = {"folds": len(done), "folds_with_A_tcn": with_a, "pairs": {}, "means": {}}
    print(f"\n## SkillCorner folds against the action truth: {len(done)} of {len(refs)} folds; A's TCN on {with_a}\n" + HEAD)
    for x, y, label in PAIRS:
        for c in F.GATE_CONDS:
            use = [r for r in done if get(x, c, r) is not None and get(y, c, r) is not None]
            if len(use) != len(done):
                continue
            g = F.compare(get, use, x, y, c)
            if g is not None:
                out["pairs"][f"{label}/{c}"] = g
                print(_row(f"{label}, {c}", g))
    print("\n## Possessions found at IoU 0.5 by true length, target, action truth\n| model | under 2 s | 2-4 s | 4-8 s | "
          "8-16 s | over 16 s |\n|---|---|---|---|---|---|")
    for name in ("F3", "K0", "K", "S1", "S2", "A1"):
        v = RA.recall_by_length({(n + "@act", c, k): s for (n, c, k), s in sc.items()}, name, done, "target", True)
        out["means"][f"recall/{name}"] = v
        print(f"| {name} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    G._dump(RA.P.WORK / "read_act.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
