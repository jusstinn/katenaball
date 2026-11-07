"""Markdown tables for docs/bakeoff.md from the stored results."""

from __future__ import annotations

import numpy as np

from katena.eval import protocol as P
from katena.eval.bootstrap import single_ci
from katena.eval.lomo import Results

SEV = ("native", "mild", "target", "severe")


def _mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def scores(res: Results, model, T, cond, refs):
    return [res.get((model, T, cond, r.key)) for r in refs if res.get((model, T, cond, r.key))]


def ranked_table(res: Results, models: list, refs: list, T: int = P.T_MAIN, label: dict | None = None) -> str:
    """One row per model: pooled target-like mean spellF1 with bootstrap CI, macroF1, then by dataset
    (target-like) and by severity."""
    rows = []
    for m in models:
        s = scores(res, m, T, P.HEADLINE, refs)
        if not s:
            continue
        lo, hi = single_ci(s, draws=500)
        row = [label.get(m, m) if label else m, str(len(s)), fmt(_mean([x["spell_f1"] for x in s])),
               f"[{fmt(lo)}, {fmt(hi)}]", fmt(_mean([res.get((m, T, P.HEADLINE, r.key))["macro_f1"] for r in refs
                                                      if not r.is_sc and res.get((m, T, P.HEADLINE, r.key))]))]
        for ds in P.DATASETS:
            sd = scores(res, m, T, P.HEADLINE, [r for r in refs if r.dataset == ds])
            row.append(fmt(_mean([x["spell_f1"] for x in sd])) if sd else "-")
        for c in SEV:
            sc = scores(res, m, T, c, refs)
            row.append(fmt(_mean([x["spell_f1"] for x in sc])) if sc else "-")
        rows.append((_mean([x["spell_f1"] for x in s]), row))
    rows.sort(key=lambda t: -t[0])
    head = ["model", "n", "spellF1 (target-like)", "95% CI", "macroF1 (IDSSE+Metrica)", "IDSSE", "Metrica", "SkillCorner",
            "native", "mild", "target", "severe"]
    return _md(head, [r for _, r in rows])


def _md(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def matrix(res: Results, prefix: str, refs_by_eval: dict, trainTs=P.T_SWEEP, evalTs=P.T_SWEEP) -> str:
    head = ["train T \\ eval T"] + [f"{t} s" for t in evalTs]
    rows = []
    for tt in trainTs:
        row = [f"{tt} s"]
        for et in evalTs:
            s = scores(res, f"{prefix}_tr{tt}", et, "target", refs_by_eval["idsse"])
            row.append(fmt(_mean([x["spell_f1"] for x in s])))
        rows.append(row)
    return _md(head, rows)


# ---------------------------------------------------------------------------------------------
# comparison tables

def ladder_table(res: Results, pairs: list, T: int = P.T_MAIN) -> str:
    """pairs: [(challenger, incumbent, refs, note)]."""
    from katena.eval.bakeoff import compare
    rows = []
    for a, b, refs, note in pairs:
        c = compare(res, a, b, T, P.HEADLINE, refs)
        rows.append([f"{a} vs {b}", note, str(c["n"]), f"{c['mean_diff']:+.4f}", f"[{c['lo']:+.4f}, {c['hi']:+.4f}]",
                     f"{100 * c['win_rate']:.0f}%", "yes" if c["adopted"] else "no"])
    return _md(["comparison", "folds", "n", "mean diff", "95% paired CI", "wins", "adopted"], rows)


def by_condition(res: Results, models: list, refs: list, conds: list, T: int = P.T_MAIN) -> str:
    rows = []
    for m in models:
        row = [m]
        for c in conds:
            s = scores(res, m, T, c, refs)
            row.append(f"{fmt(_mean([x['spell_f1'] for x in s]))} / {fmt(_mean([x['macro_f1'] for x in s]))}" if s else "-")
        rows.append(row)
    return _md(["model (spellF1 / macroF1)"] + conds, rows)


def oracle_table(res: Results, T: int = P.T_MAIN) -> str:
    names = [("baseline", "M3 as is"), ("clean_ball", "perfect ball positions (IDSSE only)"),
             ("true_team", "perfect team labels, predicted alive/dead"),
             ("true_alive", "perfect alive/dead, predicted team"),
             ("true_boundaries", "perfect boundaries, predicted team per spell"),
             ("true_both", "perfect alive/dead and team")]
    rows = []
    for k, label in names:
        row = [label]
        for sub in (("idsse",), ("metrica",), ("idsse", "metrica")):
            v = [res.get((f"ORACLE_{k}", T, "target", key)) for key in _oracle_keys(res, k, T) if key.split("/")[0] in sub]
            v = [x for x in v if x]
            row.append(f"{fmt(_mean([x['spell_f1'] for x in v]))} / {fmt(_mean([x['macro_f1'] for x in v]))} (n={len(v)})" if v else "-")
        rows.append(row)
    return _md(["oracle (spellF1 / macroF1, target-like)", "IDSSE", "Metrica", "both"], rows)


def _oracle_keys(res, k, T):
    return [key[3] for key in res.d if key[0] == f"ORACLE_{k}" and key[1] == T and key[2] == "target"]


def census_table(census: dict) -> str:
    from katena.eval.errors import TYPES
    rows = []
    for ds, d in census.items():
        n = d["n_gt"]
        rows.append([ds, str(n)] + [f"{100 * d['types'][t] / n:.1f}%" for t in TYPES]
                    + [f"{d['n_pred'] / n:.2f}", f"{d['sp_dead']}", f"{d['sp_other']}",
                       fmt(float(np.mean(d['dead_miss_frac']))), fmt(float(np.mean(d['false_dead_frac'])))])
    return _md(["dataset", "GT spells"] + list(TYPES) + ["pred/GT spells", "spurious in dead", "spurious elsewhere",
                                                        "GT-dead frames called alive", "GT-alive frames called dead"], rows)


def miss_vs_dead(refs: list, conds: tuple, T: int = P.T_MAIN) -> str:
    """P(ball missing | dead) and P(ball missing | alive) per dataset and condition, from the stored
    corrupted inputs and the dataset's own dead labels."""
    from katena.eval import corpus as C
    rows = []
    for ds in ("idsse", "metrica"):
        for c in conds:
            md = nd = ma = na = 0
            for r in [x for x in refs if x.dataset == ds]:
                if c not in C.conds_for(r):
                    continue
                f = C.load_feat(r, c)
                g = C.load_gt(r, T)
                y = g["y"][::g["step"]][f["idx"]]
                vis = f["X0"][:, list(f["names"]).index("ball_vis")] > 0
                dead, alive = y == 2, y < 2
                md += int((~vis & dead).sum()); nd += int(dead.sum())
                ma += int((~vis & alive).sum()); na += int(alive.sum())
            if nd:
                rows.append([ds, c, fmt(md / nd), fmt(ma / na)])
    return _md(["dataset", "condition", "P(ball missing | dead label)", "P(ball missing | alive label)"], rows)


# ---------------------------------------------------------------------------------------------
# full table set

BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"


def build_tables(res: Results, dev: list, locked: list, census: dict, abl: list, f8: list) -> str:
    T = P.T_MAIN
    out = []
    add = out.append
    add("### Table 1. Ranked models, development matches, LOMO, T = 2 s\n")
    add("spellF1 is the mean over matches of the official `spell_f1_mean`; CI = bootstrap over ground-truth "
        "possessions. Dataset and severity columns are mean spellF1. `n` = matches scored (M4 variants: 14 "
        "folds; M5: 8 folds; M3g/M6: grouped folds, see Table 3).\n")
    add(ranked_table(res, ["M0", "M1", "M2", "M3", "M4_nofree", "M4_freeonly", "M4_nodead", "M4_nomiss", "M5", "M5s1", "M3g", "M6"],
                     dev))
    add("\n### Table 2. Adoption ladder (target-like, paired bootstrap over possessions, 2000 draws)\n")
    ladder = [("M1", "M0", dev, "28 LOMO"), ("M2", "M1", dev, "28 LOMO"), ("M3", "M2", dev, "28 LOMO")]
    for v in ("nofree", "freeonly", "nodead", "nomiss"):
        ladder.append(("M3", f"M4_{v}", abl, "14 (ablation folds)"))
    ladder.append(("M5", "M3", f8, "8 (IDSSE+Metrica)"))
    ladder.append(("M5", "M2", f8, "8 (IDSSE+Metrica)"))
    ladder.append(("M5s1", "M2", f8, "8, second TCN seed"))
    ladder.append(("M5s1", "M5", f8, "8, seed 1 vs seed 0"))
    ladder.append(("M6", "M3g", dev, "28 grouped"))
    ladder.append(("M3", "M3g", dev, "28, LOMO vs grouped training"))
    add(ladder_table(res, ladder))
    add("\n### Table 2b. Decoder temperature tau: mean target-like spellF1 over the folds (selected per fold from the other folds)\n")
    rows = []
    for m in ("M2", "M3", "M5", "M3g"):
        t = res.get(("tau_table", T, m, "all"))
        if t:
            rows.append([m] + [fmt(float(v)) for v in t["table"].mean(1)])
    add(_md(["model"] + [f"tau={x}" for x in P.TAUS], rows))
    add("\n### Table 3. Same models restricted to the folds they were run on (spellF1 / macroF1)\n")
    add(by_condition(res, ["M0", "M2", "M3", "M5", "M5s1"], f8, ["native", "mild", "target", "severe"]))
    add("\n(8 IDSSE + Metrica folds above.)\n")
    add(by_condition(res, ["M3", "M4_nofree", "M4_freeonly", "M4_nodead", "M4_nomiss"], abl, ["mild", "target", "severe"]))
    add("\n(14 ablation folds above.)\n")
    add(by_condition(res, ["M3g", "M6"], dev, ["native", "mild", "target", "severe"]))
    add("\n(28 grouped folds above.)\n")
    add("\n### Table 3b. Label-semantics sweep (spellF1 on 6 IDSSE development matches, target-like; rows = training T, columns = evaluation T)\n")
    ids8 = [r for r in f8 if r.dataset == "idsse"]
    me8 = [r for r in f8 if r.dataset == "metrica"]
    for design, text in (("A", "Design A: LOMO, all other development matches train (pre-registered)"),
                         ("B", "Design B: IDSSE and SkillCorner rows only, no Metrica rows (Amendment A3)")):
        add(f"\n{text}\n")
        add(matrix(res, f"M3{design}", {"idsse": ids8}))
        add("\nSame models on the 2 development Metrica matches (event-derived labels; eval T = sandwich applied to those labels, 0 = raw)\n")
        rows = []
        for tt in P.T_SWEEP:
            rows.append([f"{tt} s"] + [fmt(_mean([x["spell_f1"] for x in scores(res, f"M3{design}_tr{tt}", et, "target", me8)]))
                                       for et in P.T_SWEEP])
        add(_md(["train T \\ eval T"] + [f"{t} s" for t in P.T_SWEEP], rows))
    add("\nNo-training references per evaluation T (IDSSE / Metrica)\n")
    rows = [[m] + [f"{fmt(_mean([x['spell_f1'] for x in scores(res, m, et, 'target', ids8)]))} / "
                   f"{fmt(_mean([x['spell_f1'] for x in scores(res, m, et, 'target', me8)]))}" for et in P.T_SWEEP]
            for m in ("M0", "M1")]
    add(_md(["model"] + [f"eval T={t}" for t in P.T_SWEEP], rows))
    add("\n### Table 4. Severity and ball-loss/dead-ball coupling, IDSSE development matches (6)\n")
    ids = [r for r in dev if r.dataset == "idsse"]
    conds = ["native", "mild", "target", "severe", "target_dead3", "target_dead15"]
    add(by_condition(res, ["M0", "M1", "M2", "M3", "M4_nomissw"], ids, conds))
    add("\n### Table 5. Oracles (IDSSE and Metrica development matches, M3, target-like)\n")
    add(oracle_table(res))
    add("\n### Table 6. Error types of unmatched ground-truth possessions (IoU 0.5), M3, target-like\n")
    add(census_table(census))
    add("\n### Table 7. Is a missing ball a dead-ball cue? (labels of each dataset)\n")
    add(miss_vs_dead(dev, ("native", "target", "target_dead3", "target_dead15")))
    add("\n### Table 8. Locked benches (scored once, trained on all 28 development matches)\n")
    bm = ["BENCH_M0", "BENCH_M1", "BENCH_M2", "BENCH_M3", "BENCH_M4_nofree", "BENCH_M4_freeonly", "BENCH_M4_nodead", "BENCH_M4_nomiss", "BENCH_M5"]
    rows = []
    for m in bm:
        row = [m.replace("BENCH_", "")]
        for r in locked:
            for c in ("target", "mild", "severe", "native"):
                s = res.get((m, T, c, r.key))
                row.append(f"{fmt(s['spell_f1']) if s else '-'} / {fmt(s['macro_f1']) if s else '-'}")
        rows.append(row)
    head = ["model (spellF1 / macroF1)"] + [f"{r.id} {c}" for r in locked for c in ("target", "mild", "severe", "native")]
    add(_md(head, rows))
    return "\n".join(out)


def splice(doc_path, tables: str) -> None:
    import pathlib
    p = pathlib.Path(doc_path)
    text = p.read_text() if p.exists() else f"# Bake-off\n\n{BEGIN}\n{END}\n"
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    p.write_text(text[:a] + "\n" + tables + "\n" + text[b:])
