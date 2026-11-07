"""D4 leave-one-match-out study on the 6 external development matches (never the locked 2).

    python scripts/model_lomo.py run --variants twohead 3class twohead_ballfree twohead_noaug
    python scripts/model_lomo.py report                    # markdown tables + paired comparisons

`run` is resumable: one JSON (and one .npz of predicted codes) per (variant, held-out match) under
--out. It builds any missing cached copies first (scripts/model_cache.py does the same on its own).
External data only; nothing here reads data/ or the competition match.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")]

from katena.model import data as D  # noqa: E402
from katena.model import lomo as L  # noqa: E402

OUT = os.path.join(ROOT, "runs", "d4_lomo")

COMPARISONS = [
    # (A, B, question)
    ("twohead", "baseline", "model (two-head) vs starter baseline"),
    ("3class", "baseline", "model (3-class) vs starter baseline"),
    ("twohead", "3class", "two-head vs 3-class"),
    ("twohead", "twohead_ballfree", "all features vs ball-free only (two-head)"),
    ("twohead", "twohead_noaug", "severity augmentation vs default config only (two-head)"),
    ("3class", "3class_ballfree", "all features vs ball-free only (3-class)"),
    ("3class", "3class_noaug", "severity augmentation vs default config only (3-class)"),
]


def _paths(out, variant, held):
    base = Path(out) / variant / held
    return base.with_suffix(".json"), base.with_suffix(".npz")


def cmd_run(args):
    names = D.check_names(args.folds or D.DEV_NAMES)
    t0 = time.time()
    plan = {}
    for n in D.DEV_NAMES:
        specs = [D.eval_spec(n)]
        for v in args.variants:
            specs += D.train_plan(n, L.VARIANTS[v].aug)
        plan[n] = list({s.key(): s for s in specs}.values())
    D.ensure_all(plan, workers=args.workers)
    print(f"cache ready ({time.time() - t0:.0f} s)")
    for held in names:
        jp, npz = _paths(args.out, "baseline", held)
        if not jp.exists():
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, pred=L.eval_copy(held)["baseline"])
            L.save_json(L.baseline_fold(held), jp)
    for v in args.variants:             # variant-major, so each variant completes all folds in turn
        for held in names:
            jp, npz = _paths(args.out, v, held)
            if jp.exists():
                continue
            print(f"fold {held} / {v}", flush=True)
            res = L.run_fold(L.VARIANTS[v], held, D.DEV_NAMES, args.k_inner,
                             log=lambda s: print(s, flush=True))
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, pred=res.pop("pred"))
            L.save_json(res, jp)
    print(f"run done in {time.time() - t0:.0f} s")
    return 0


def cmd_robust(args):
    """Score finished folds on milder / harsher noise than the default (severity mismatch)."""
    names = D.check_names(args.folds or D.DEV_NAMES)
    t0 = time.time()
    plan = {n: [D.eval_spec(n, severity_index=i) for i in args.severity_index] for n in names}
    D.ensure_all(plan, workers=args.workers)
    for v in args.variants:
        for held in names:
            jp, npz = _paths(args.out, v, held)
            rp = Path(args.out) / "robust" / v / f"{held}.json"
            if rp.exists() or not jp.exists():
                continue
            fold = json.loads(jp.read_text(encoding="utf-8"))
            fold["pred"] = np.load(npz)["pred"]
            res = L.robustness_fold(L.VARIANTS[v], held, fold, tuple(args.severity_index),
                                    log=lambda s: print(s, flush=True))
            L.save_json(res, rp)
    print(f"robust done in {time.time() - t0:.0f} s")
    return 0


def robust_lines(out, names):
    root = Path(out) / "robust"
    if not root.exists():
        return []
    rob = {}
    for jp in sorted(root.glob("*/*.json")):
        r = json.loads(jp.read_text(encoding="utf-8"))
        rob.setdefault(r["variant"], {})[r["held_out"]] = r
    if not rob:
        return []
    lines = ["\n### severity mismatch: mean spellF1 when the evaluation noise is milder or harsher "
             "than the default (decoder and temperatures unchanged)\n"]
    sevs = sorted({s for v in rob.values() for r in v.values() for s in r["by_severity"]}, key=float)
    lines.append("| variant | " + " | ".join(f"severity {s}" for s in sevs) + " | refit reproduces fold |")
    lines.append("|---|" + "---|" * (len(sevs) + 1))
    first = next(iter(rob.values()))
    base = [f"{np.mean([first[n]['by_severity'][s]['baseline']['spell_f1_mean'] for n in names if n in first]):.4f}"
            for s in sevs]
    lines.append("| baseline | " + " | ".join(base) + " | n/a |")
    for v, rs in rob.items():
        ns = [n for n in names if n in rs]
        cells = [f"{np.mean([rs[n]['by_severity'][s]['model']['spell_f1_mean'] for n in ns]):.4f}" for s in sevs]
        rep = sum(bool(rs[n]["reproduced_default"]) for n in ns)
        lines.append(f"| {v} ({len(ns)} folds) | " + " | ".join(cells) + f" | {rep}/{len(ns)} |")
    vs = list(rob)
    if len(vs) >= 2:
        a, b = vs[0], vs[1]
        ns = sorted(set(rob[a]) & set(rob[b]))
        for s in sevs:
            pa = {n: rob[a][n]["by_severity"][s]["model"]["spell_f1_mean"] for n in ns}
            pb = {n: rob[b][n]["by_severity"][s]["model"]["spell_f1_mean"] for n in ns}
            c = L.paired_compare(pa, pb)
            lines.append(f"\n{a} minus {b} at severity {s}: wins {c['wins']}/{c['n']}, mean "
                         f"{c['mean_diff']:+.4f}, 90% CI over matches [{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}]")
    return lines


def load_results(out):
    res = {}
    for jp in sorted(Path(out).glob("*/*.json")):
        r = json.loads(jp.read_text(encoding="utf-8"))
        res.setdefault(jp.parent.name, {})[r["held_out"]] = r
    return res


def load_preds(out, variant, names):
    return {n: np.load(_paths(out, variant, n)[1])["pred"] for n in names}


def _short(n):
    return n.replace("idsse-", "").replace("metrica-game3", "Game3")


def cmd_report(args):
    res = load_results(args.out)
    order = [v for v in ["baseline"] + list(L.VARIANTS) if v in res]
    names = sorted(set.intersection(*[set(res[v]) for v in order])) if order else []
    lines = []
    lines.append("### spellF1 / macroF1 per held-out match\n")
    lines.append("| variant | " + " | ".join(_short(n) for n in names) + " | mean spellF1 | mean macroF1 |")
    lines.append("|---|" + "---|" * (len(names) + 2))
    for v in order:
        cells = [f"{res[v][n]['scores']['spell_f1_mean']:.3f} / {res[v][n]['scores']['macro_f1']:.3f}"
                 for n in names]
        ms = np.mean([res[v][n]["scores"]["spell_f1_mean"] for n in names])
        mm = np.mean([res[v][n]["scores"]["macro_f1"] for n in names])
        lines.append(f"| {v} | " + " | ".join(cells) + f" | **{ms:.4f}** | {mm:.4f} |")
    lines.append("\n### predicted / true possessions, and mean predicted duration (s; truth in brackets)\n")
    lines.append("| variant | " + " | ".join(_short(n) for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for v in order:
        cells = []
        for n in names:
            s = res[v][n]["scores"]
            cells.append(f"{s['n_pred_spells']}/{s['n_gt_spells']}, {s['mean_spell_duration_s']:.1f} "
                         f"({s['gt_mean_spell_duration_s']:.1f})")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("\n### frame-level detail (means over matches)\n")
    lines.append("| variant | spellF1 @ IoU 0.5 | accuracy | alive/dead F1 | team acc (both alive) | spell count error |")
    lines.append("|---|---|---|---|---|---|")
    for v in order:
        s = [res[v][n]["scores"] for n in names]
        lines.append(f"| {v} | {np.mean([x['spell_f1_at_headline'] for x in s]):.4f} | "
                     f"{np.mean([x['accuracy'] for x in s]):.4f} | {np.mean([x['alive_dead_f1'] for x in s]):.4f} | "
                     f"{np.mean([x['team_acc_both_alive'] for x in s]):.4f} | "
                     f"{np.mean([x['n_pred_spells'] - x['n_gt_spells'] for x in s]):+.1f} |")
    lines.append("\n### no decoder (per-frame argmax of the same log-probs)\n")
    lines.append("| variant | mean spellF1 | mean macroF1 | mean predicted spells |")
    lines.append("|---|---|---|---|")
    for v in order:
        if v == "baseline":
            continue
        a = [res[v][n]["scores_argmax"] for n in names]
        lines.append(f"| {v} | {np.mean([x['spell_f1_mean'] for x in a]):.4f} | "
                     f"{np.mean([x['macro_f1'] for x in a]):.4f} | "
                     f"{np.mean([x['n_pred_spells'] for x in a]):.0f} |")
    lines.append("\n### chosen decoder parameters and temperatures (nested, inside training matches)\n")
    lines.append("| variant | " + " | ".join(_short(n) for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for v in order:
        if v == "baseline":
            continue
        cells = []
        for n in names:
            r = res[v][n]
            t = ",".join(f"{x:.2f}" for x in r["temperature"].values())
            cells.append(f"pen {r['decode']['switch_penalty']:g}, dwell {r['decode']['min_dwell_s']:g} s, T {t}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("\n### paired comparisons on spellF1 (A minus B; gate: wins >= 70% of matches and 90% CI > 0)\n")
    lines.append("| A vs B | wins | mean diff | 90% CI (hierarchical) | 90% CI (matches) | gate |")
    lines.append("|---|---|---|---|---|---|")
    for a, b, q in COMPARISONS:
        if a not in res or b not in res:
            continue
        pa = {n: res[a][n]["scores"]["spell_f1_mean"] for n in names}
        pb = {n: res[b][n]["scores"]["spell_f1_mean"] for n in names}
        h = L.paired_compare(pa, pb, load_preds(args.out, a, names), load_preds(args.out, b, names),
                             n_boot=args.n_boot)
        m = L.paired_compare(pa, pb, n_boot=args.n_boot)
        lines.append(f"| {q} | {h['wins']}/{h['n']} | {h['mean_diff']:+.4f} | "
                     f"[{h['ci90'][0]:+.4f}, {h['ci90'][1]:+.4f}] | "
                     f"[{m['ci90'][0]:+.4f}, {m['ci90'][1]:+.4f}] | "
                     f"{'pass' if h['passes_gate'] else 'fail'} |")
    lines += robust_lines(args.out, names)
    tt = {v: sum(res[v][n].get("time_total_s", 0) for n in names) for v in order if v != "baseline"}
    lines.append("\nCompute per variant (6 folds, fit + nested inner fits + decoding): "
                 + ", ".join(f"{v} {t / 60:.1f} min" for v, t in tt.items()))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variants", nargs="+", default=["twohead", "3class", "twohead_ballfree",
                                                    "twohead_noaug"], choices=list(L.VARIANTS))
    r.add_argument("--folds", nargs="*", default=None)
    r.add_argument("--k-inner", type=int, default=L.K_INNER)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--out", default=OUT)
    rb = sub.add_parser("robust")
    rb.add_argument("--variants", nargs="+", default=["twohead", "twohead_noaug"],
                    choices=list(L.VARIANTS))
    rb.add_argument("--severity-index", type=int, nargs="+", default=[0, 4])
    rb.add_argument("--folds", nargs="*", default=None)
    rb.add_argument("--workers", type=int, default=2)
    rb.add_argument("--out", default=OUT)
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "robust": cmd_robust, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
