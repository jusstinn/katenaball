#!/usr/bin/env python3
"""A2 and AW2: A and AW with a TCN trained for 9,600 steps instead of 4,800. External matches only.

    uv run python scripts/run_a.py a-tcn --tag <tag>2x --steps 9600        # A2's TCN (gate.yml runs this on runners)
    uv run python scripts/run_aw.py aw-tcn --tag <tag>2x --steps 9600      # AW2's TCN
    uv run python scripts/run_x2.py --tag <tag> --inc f3_gbdt,f3_tcn       # the gate, where F3's cache is

Pre-registered in docs/decision-log.md (2026-11-13, "A2 and AW2 (pre-registration)") before any score of either.
Never reads ``data/`` or ``evaluation/gt/``; no locked bench.

Nothing but the number of TCN training steps changes: the one-cycle schedule spans them, the sequences, labels,
weights, seed and threads are the base candidate's, and the LightGBM member is the base candidate's cached one.
``<tag>`` names the base candidates' files (``<tag>_a_gbdt1200a``, ``<tag>_a_tcn4800ta``, ...); the longer TCNs
carry ``<tag>2x`` (``<tag>2x_a_tcn9600ta``, ``<tag>2x_aw_tcn9600taw``).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_aw as RW  # noqa: E402  (through run_a and run_g12: pins the CPU environment, loads libomp first)

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402

RA = RW.RA
G = RA.G
T, N_TREES, STEPS, STEPS2 = RA.T, RA.N_TREES, RA.STEPS, 2 * RA.STEPS
log, HEAD, _row = G.log, G.HEAD, G._row


def variants(tag: str) -> dict:
    p, p2 = f"{tag}_", f"{tag}2x_"
    return {"A": (f"{p}a_gbdt{N_TREES}a", f"{p}a_tcn{STEPS}ta", f"{p2}a_tcn{STEPS2}ta"),
            "AW": (f"{p}aw_gbdt{N_TREES}aw", f"{p}aw_tcn{STEPS}taw", f"{p2}aw_tcn{STEPS2}taw")}


def score_all(tag: str, dev: list, inc: tuple) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    ex = lambda v, r: Lm.oof_path(v, T, r).exists()                  # noqa: E731
    for r in dev:
        if not all(ex(v, r) for v in inc):
            continue
        g0, t0 = (Lm.load_oof(v, T, r) for v in inc)
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        gta = RA.load_act(r) if r.is_sc else None
        rows = [("F3", g0, t0, F.W_F)]
        for base, (vg, vt, vt2) in variants(tag).items():
            if not (ex(vg, r) and ex(vt, r) and ex(vt2, r)):
                continue
            g, t, t2 = (Lm.load_oof(v, T, r) for v in (vg, vt, vt2))
            rows += [(base, g, t, F.W_F), (base + "2", g, t2, F.W_F), (base + "_T", g, t, 1.0), (base + "2_T", g, t2, 1.0)]
        for c in F.GATE_CONDS:
            for name, g, t, w in rows:
                s = G.blend_score(r, c, g, t, w, pri, gt)
                sc[(name, c, r.key)] = s
                sc[(name + "@act", c, r.key)] = RA.score_on(r, c, g, t, w, pri, gta) if r.is_sc else s
    with open(P.WORK / f"{tag}_x2_scores.pkl", "wb") as fh:
        pickle.dump(sc, fh)
    return sc


def passes(g_all_t: dict | None, g_all_s: dict | None, g_im_t: dict | None, g_sc_t: dict | None) -> bool:
    """Registered: F's rule on the 28 folds with the action truth on SkillCorner (adopted at target, not worse at
    severe), and not worse at target on the 8 IDSSE + Metrica folds or on the 20 SkillCorner folds."""
    return bool(g_all_t and g_all_s and g_im_t and g_sc_t and F.decision(g_all_t, g_all_s)
                and not g_im_t["worse"] and not g_sc_t["worse"])


def gate(tag: str, dev: list, inc: tuple) -> dict:
    sc = score_all(tag, dev, inc)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    out = {"tag": tag, "steps": [STEPS, STEPS2], "cands": {}}
    for base in variants(tag):
        cand = base + "2"
        done = [r for r in dev if get(cand, "target", r) is not None]
        if not done:
            continue
        sc_refs, im_refs = [r for r in done if r.is_sc], [r for r in done if not r.is_sc]
        res = {"folds": len(done), "registered_run": len(done) == len(dev) == 28, "rows": {}}
        print(f"\n## {cand} ({STEPS2} TCN steps) vs {base} ({STEPS}), {len(done)} of {len(dev)} folds\n" + HEAD)
        for tag_, refs, suffix in (("all, action truth on SkillCorner", done, "@act"),
                                   ("IDSSE + Metrica, one truth", im_refs, ""),
                                   ("SkillCorner, action truth", sc_refs, "@act"),
                                   ("all, standing truth", done, "")):
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, cand + suffix, base + suffix, c)
                if g is not None:
                    res["rows"][f"{tag_}/{c}"] = g
                    print(_row(f"{cand} vs {base}, {c}, {tag_}", g))
        rr = res["rows"]
        res["passes"] = passes(rr.get("all, action truth on SkillCorner/target"), rr.get("all, action truth on SkillCorner/severe"),
                               rr.get("IDSSE + Metrica, one truth/target"), rr.get("SkillCorner, action truth/target"))
        print(f"\nRULE ({'registered run' if res['registered_run'] else 'NOT the registered run: partial'}): {cand} "
              f"{'PASSES and replaces' if res['passes'] else 'fails and does not replace'} {base}")
        print("\n" + HEAD)
        for a, b, label in ((cand + "_T", base + "_T", "TCN member alone"), (cand, "F3", f"{cand} vs F3")):
            for tag_, refs, suffix in (("IDSSE + Metrica, one truth", im_refs, ""), ("SkillCorner, action truth", sc_refs, "@act")):
                for c in F.GATE_CONDS:
                    g = F.compare(get, refs, a + suffix, b + suffix, c)
                    if g is not None:
                        res["rows"][f"{label}/{tag_}/{c}"] = g
                        print(_row(f"{label}, {c}, {tag_}", g))
        for name in (base, cand):
            for tag_, refs, suffix in (("IDSSE + Metrica", im_refs, ""), ("SkillCorner, action truth", sc_refs, "@act")):
                s = [get(name + suffix, "target", r) for r in refs]
                per = sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)
                res["rows"][f"per_true/{name}/{tag_}"] = per
                print(f"predicted possessions per true one, target, {name}, {tag_}: {per:.3f}")
        out["cands"][cand] = res
    G._dump(P.WORK / f"{tag}_x2_gate.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="the tag of the base candidates' cached members (for example gha)")
    ap.add_argument("--inc", default=f"{RA.INC_G},{RA.INC_T}", help="F3's cached members, as in run_a.py")
    a = ap.parse_args()
    dev = F.dev_refs()
    for r in dev:
        if r.is_sc:
            RA.build_act(r)
    log(f"x2 gate; WORK={P.WORK}; tag {a.tag}; development matches {len(dev)}")
    gate(a.tag, dev, tuple(a.inc.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
