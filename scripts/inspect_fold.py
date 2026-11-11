#!/usr/bin/env python3
"""Where a model loses spellF1, and a timeline to look at it yourself.

    uv run python scripts/inspect_fold.py summary                          # all cached folds, aggregates
    uv run python scripts/inspect_fold.py fold idsse/DFL-MAT-J03WMX        # one held-out external match -> HTML
    uv run python scripts/inspect_fold.py fold skillcorner/1874553 --model F --cond severe
    uv run python scripts/inspect_fold.py public submissions/<team>.csv    # the public 5 minutes -> HTML

``summary`` and ``fold`` read external matches only: the out-of-fold probabilities that ``scripts/run_g12.py``
cached (KATENA_WORK), decoded exactly as the gate decodes them. They select nothing; they describe.

``public`` compares a submission with the public window of the provided match. It prints the output path and
nothing else, and writes under ``artifacts/``, which is ignored by git and closed to the coding agent: the page
is for a person. It is an automatic check on provided data, which the organisers allow. **Do not put it beside
the video:** their clarification of 2026-11-03 forbids checking your output against the video. The public window
is a pipeline check, never a way to tune or to choose between versions, so log each look in docs/decision-log.md.

Every true possession the scorer does not match at IoU 0.5 gets one of four causes, from the labels predicted
over its own frames:

* ``other team``   most of it was given to the opponent
* ``called dead``  most of it was predicted as dead ball
* ``too long``     the right team, but the predicted possession runs on past it (a neighbour was swallowed)
* ``too short``    the right team, but the predicted possession covers too little of it (cut or split)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

import numpy as np  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
MODELS = {"F": ("g1_gbdt150", "g1_tcn"), "G1": ("g1_gbdt1200", "g1_tcn"), "LT": ("g1_gbdt1200", "g2_tcn4800t")}
MODEL_TEXT = {"F": "F (150 trees, TCN 1,200 steps)", "G1": "F with 1,200 trees",
              "LT": "1,200 trees + TCN 4,800 steps with touch columns"}
EDGES = (0, 2, 4, 8, 16, 32, np.inf)
BUCKETS = ("under 2 s", "2-4 s", "4-8 s", "8-16 s", "16-32 s", "over 32 s")
MISS = ("other team", "called dead", "too long", "too short")
SPUR = ("wrong team", "ball was dead", "boundaries")
PUBLIC_GT = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"


# ---- analysis (no project imports: also used on a submission file) -----------------------------

def analyse(gt: np.ndarray, pred: np.ndarray, fps: float) -> dict:
    """Every true and predicted possession with what the scorer made of it at IoU 0.5."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    g_ok, p_ok = np.zeros(len(gs), bool), np.zeros(len(ps), bool)
    best, mate = np.zeros(len(gs)), {}
    for iou, i, j in ov:                                          # IoU-descending, as the scorer walks it
        best[i] = max(best[i], iou)
        if iou >= 0.5 and not g_ok[i] and not p_ok[j]:
            g_ok[i] = p_ok[j] = True
        inter = min(gs[i][1], ps[j][1]) - max(gs[i][0], ps[j][0])
        if inter > mate.get(i, (0, 0))[0]:
            mate[i] = (inter, ps[j][1] - ps[j][0])
    true = []
    for i, (a, b, k) in enumerate(gs):
        w = filled[a:b]
        same, other = float((w == k).mean()), float((w == 1 - k).mean())
        dead = 1.0 - same - other
        if g_ok[i]:
            cat = "found"
        elif other >= max(same, dead):
            cat = "other team"
        elif dead >= max(same, other):
            cat = "called dead"
        else:
            inter, plen = mate.get(i, (0, 0))
            cat = "too long" if plen - inter >= (b - a) - inter else "too short"
        true.append({"a": a, "b": b, "k": k, "cat": cat, "iou": float(best[i]), "len": (b - a) / fps})
    spur = []
    for j, (a, b, k) in enumerate(ps):
        w = gt[a:b]
        same, other = float((w == k).mean()), float((w == 1 - k).mean())
        cat = "found" if p_ok[j] else ("wrong team" if other >= max(same, 1 - same - other) else
                                       "ball was dead" if 1 - same - other >= same else "boundaries")
        spur.append({"a": a, "b": b, "k": k, "cat": cat, "len": (b - a) / fps})
    sc = pm.spell_scores(gt, filled, runs)
    return {"runs": runs, "filled": filled, "true": true, "pred": spur, "spell_f1": sc["mean_f1"],
            "f1": {t: sc["per_iou"][t]["f1"] for t in pm.SPELL_IOUS}}


def oracle_scores(gt: np.ndarray, an: dict) -> tuple[float, float]:
    """spellF1 with the true team on the predicted alive frames, and with the true alive/dead pattern."""
    runs, filled = an["runs"], an["filled"]
    alive_gt, alive_pr = (gt == H) | (gt == A), (filled == H) | (filled == A)
    team = np.where(alive_gt & alive_pr, gt, filled)
    carried = pm.fill_abstentions(np.where(filled == D, X, filled), runs)      # a team on every frame
    alive = np.where(gt == D, D, carried)
    return pm.spell_scores(gt, team, runs)["mean_f1"], pm.spell_scores(gt, alive, runs)["mean_f1"]


# ---- external folds --------------------------------------------------------------------------

def fold_codes(r, dev, model: str, cond: str):
    """(truth, prediction, fps) at native frames for one held-out external match, decoded as the gate does."""
    from katena.decode import hmm
    from katena.eval import bakeoff as B
    from katena.eval import corpus as C
    from katena.eval import final as F
    from katena.eval import lomo as Lm
    from katena.eval.score import to_native_codes
    from katena.models import ensemble as E

    gv, tv = MODELS[model]
    g, t = Lm.load_oof(gv, F.T, r), Lm.load_oof(tv, F.T, r)
    pri, gt = B.priors_for(F.PriorSource(dev), {r.key}, F.T), C.load_gt(r, F.T)
    pa, ph = E.blend_heads((g[f"pa_{cond}"], g[f"ph_{cond}"]), (t[f"pa_{cond}"], t[f"ph_{cond}"]), F.W_F)
    f = F._meta(r, cond)
    rows = hmm.decode(pa, ph, f["half"], pri, F.TAU, "viterbi")
    pred = to_native_codes(rows, f["idx"], gt["step"], gt["n_native"], gt["blocks"])
    if r.is_sc:                                                     # SkillCorner has no D label (score.score)
        pred = np.where(pred == D, X, pred).astype(np.int8)
    return gt["y"], pred, gt["fps"]


def summary(models: list, cond: str) -> None:
    from katena.eval import final as F
    from katena.eval import lomo as Lm

    dev = F.dev_refs()
    groups = (("IDSSE + Metrica", [r for r in dev if not r.is_sc]), ("SkillCorner", [r for r in dev if r.is_sc]))
    for model in models:
        if not all(Lm.oof_path(v, F.T, r).exists() for v in MODELS[model] for r in dev):
            print(f"\n{model}: the cache does not hold all 28 folds, skipped")
            continue
        print(f"\n## {MODEL_TEXT[model]}, {cond}, 28 held-out external matches")
        for gname, refs in groups:
            ans, orc = [], []
            for r in refs:
                gt, pred, fps = fold_codes(r, dev, model, cond)
                ans.append(analyse(gt, pred, fps))
                orc.append(oracle_scores(gt, ans[-1]))
            true = [s for an in ans for s in an["true"]]
            pred_ = [s for an in ans for s in an["pred"]]
            missed = [s for s in true if s["cat"] != "found"]
            spur = [s for s in pred_ if s["cat"] != "found"]
            mean = lambda xs: float(np.mean(xs))  # noqa: E731
            print(f"\n### {gname} ({len(refs)} matches, {len(true)} true possessions, "
                  f"{len(pred_) / len(true):.2f} predicted per true one)")
            print(f"spellF1 {mean([a['spell_f1'] for a in ans]):.4f};  F1 at IoU 0.3 / 0.5 / 0.7: "
                  + " / ".join(f"{mean([a['f1'][t] for a in ans]):.3f}" for t in (0.3, 0.5, 0.7))
                  + f";  with the true team {mean([o[0] for o in orc]):.3f};  with true alive/dead {mean([o[1] for o in orc]):.3f}")
            print("\n| True possession length | " + " | ".join(BUCKETS) + " | all |\n|---|" + "---|" * (len(BUCKETS) + 1))
            cut = lambda xs, i: [s for s in xs if EDGES[i] <= s["len"] < EDGES[i + 1]]  # noqa: E731
            cells = lambda fn: " | ".join(fn(i) for i in range(len(BUCKETS)))  # noqa: E731
            pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "-"  # noqa: E731
            print(f"| True possessions | {cells(lambda i: str(len(cut(true, i))))} | {len(true)} |")
            print("| Found at IoU 0.5 | " + cells(lambda i: pct(sum(s['cat'] == 'found' for s in cut(true, i)), len(cut(true, i))))
                  + f" | {pct(len(true) - len(missed), len(true))} |")
            print(f"| Share of all misses | {cells(lambda i: pct(len(cut(missed, i)), len(missed)))} | 100% |")
            for c in MISS:
                print(f"| Missed: {c} | " + cells(lambda i: pct(sum(s['cat'] == c for s in cut(missed, i)), len(cut(missed, i))))
                      + f" | {pct(sum(s['cat'] == c for s in missed), len(missed))} |")
            print("| Predicted of that length that are right | "
                  + cells(lambda i: pct(sum(s['cat'] == 'found' for s in cut(pred_, i)), len(cut(pred_, i))))
                  + f" | {pct(len(pred_) - len(spur), len(pred_))} |")
            print("Predicted possessions that match nothing: "
                  + ", ".join(f"{c} {pct(sum(s['cat'] == c for s in spur), len(spur))}" for c in SPUR))


# ---- the page --------------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root { --bg:#fff; --fg:#1f2328; --mute:#667085; --line:#e4e7ec; --h:#2f6fde; --a:#e8842c; --d:#c9ced6;
        --m1:#c62828; --m2:#6b7280; --m3:#7b1fa2; --m4:#00838f; --sp:#111827; }
@media (prefers-color-scheme: dark) { :root { --bg:#14161a; --fg:#e6e8eb; --mute:#98a2b3; --line:#2a2f37; --d:#4a505a; --sp:#e6e8eb; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, "Segoe UI", sans-serif; margin:24px auto; max-width:1180px; padding:0 16px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 8px; }
.note { color:var(--mute); margin:0 0 12px; } .warn { border:1px solid var(--m1); border-radius:6px; padding:8px 12px; margin:12px 0; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }
.card { border:1px solid var(--line); border-radius:6px; padding:8px 12px; min-width:110px; }
.card b { display:block; font-size:18px; } .card span { color:var(--mute); font-size:12px; }
.legend span { display:inline-block; margin-right:14px; white-space:nowrap; } .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:5px; }
svg { display:block; width:100%; height:auto; margin:2px 0; } svg text { fill:var(--mute); font-size:11px; }
table { border-collapse:collapse; width:100%; } th, td { text-align:left; padding:4px 10px; border-bottom:1px solid var(--line); }
th { color:var(--mute); font-weight:600; } tbody tr { cursor:pointer; } tbody tr:hover { background:var(--line); }
label { margin-right:14px; white-space:nowrap; } .flash { stroke:var(--fg); stroke-width:2; fill:none; }
</style></head><body>
<h1>__TITLE__</h1><p class="note">__NOTE__</p>__WARN__
<div class="cards" id="cards"></div>
<div class="legend">
<span><i class="sw" style="background:var(--h)"></i>home</span><span><i class="sw" style="background:var(--a)"></i>away</span>
<span><i class="sw" style="background:var(--d)"></i>dead ball</span>
<span><i class="sw" style="background:var(--m1)"></i>missed: other team</span><span><i class="sw" style="background:var(--m2)"></i>missed: called dead</span>
<span><i class="sw" style="background:var(--m3)"></i>missed: too long</span><span><i class="sw" style="background:var(--m4)"></i>missed: too short</span>
<span><i class="sw" style="background:var(--sp)"></i>predicted, matches nothing</span></div>
<p class="note">Each row is five minutes. Top strip: truth. Middle: prediction. Bottom: true possessions the scorer did not match at IoU 0.5
(upper marks) and predicted possessions that match nothing (lower marks). Hover for times.</p>
<div id="rows"></div>
<h2>True possessions that were missed</h2><div id="filters"></div>
<table><thead><tr><th>Starts</th><th>Length</th><th>Team</th><th>What was predicted</th><th>Best IoU</th></tr></thead><tbody id="tb"></tbody></table>
<script>
const Z = __DATA__, ROW = 300, W = 1100, L = 52, PX = (W - L) / ROW, NS = "http://www.w3.org/2000/svg";
const COL = {0: "var(--h)", 1: "var(--a)", 2: "var(--d)"}, TEAM = {0: "home", 1: "away", 2: "dead ball"};
const MC = {"other team": "var(--m1)", "called dead": "var(--m2)", "too long": "var(--m3)", "too short": "var(--m4)"};
const fmt = s => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
const span = (a, b) => fmt(a) + " to " + fmt(b) + " (" + (b - a).toFixed(1) + " s)";
const el = (p, tag, at, text) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); if (text) e.textContent = text; p.appendChild(e); return e; };
const svgs = {};
const rowIds = new Set(); Z.runs.forEach(([a, b]) => { for (let r = Math.floor(a / ROW); r * ROW < b; r++) rowIds.add(r); });
[...rowIds].sort((x, y) => x - y).forEach(r => {
  const s = el(document.getElementById("rows"), "svg", {viewBox: `0 0 ${W} 62`});
  el(s, "text", {x: 0, y: 26}, fmt(r * ROW));
  for (let m = 0; m <= 5; m++) el(s, "line", {x1: L + m * 60 * PX, x2: L + m * 60 * PX, y1: 2, y2: 60, stroke: "var(--line)"});
  svgs[r] = s;
});
function put(a, b, y, h, fill, tip) {
  for (let r = Math.floor(a / ROW); r * ROW < b; r++) {
    if (!svgs[r]) continue;
    const x0 = Math.max(a, r * ROW) - r * ROW, x1 = Math.min(b, (r + 1) * ROW) - r * ROW;
    el(el(svgs[r], "rect", {x: L + x0 * PX, y: y, width: Math.max((x1 - x0) * PX, 0.6), height: h, fill: fill}), "title", {}, tip);
  }
}
Z.truth.forEach(([a, b, k]) => { if (k < 3) put(a, b, 4, 16, COL[k], "truth: " + TEAM[k] + ", " + span(a, b)); });
Z.pred.forEach(([a, b, k]) => { if (k < 3) put(a, b, 22, 16, COL[k], "predicted: " + TEAM[k] + ", " + span(a, b)); });
Z.missed.forEach(m => put(m.a, m.b, 42, 8, MC[m.cat], "missed (" + m.cat + "): " + TEAM[m.k] + ", " + span(m.a, m.b)));
Z.spurious.forEach(m => put(m.a, m.b, 52, 6, "var(--sp)", "predicted, matches nothing (" + m.cat + "): " + TEAM[m.k] + ", " + span(m.a, m.b)));
Z.cards.forEach(([v, t]) => { const c = document.createElement("div"); c.className = "card"; c.innerHTML = "<b>" + v + "</b><span>" + t + "</span>"; document.getElementById("cards").appendChild(c); });
const on = new Set(Object.keys(MC)), tb = document.getElementById("tb");
function table() {
  tb.innerHTML = "";
  Z.missed.filter(m => on.has(m.cat)).forEach(m => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${fmt(m.a)}${Z.frames ? " (frame " + Math.round(m.a * Z.fps) + ")" : ""}</td><td>${(m.b - m.a).toFixed(1)} s</td><td>${TEAM[m.k]}</td><td>${m.cat}</td><td>${m.iou.toFixed(2)}</td>`;
    tr.onclick = () => { const r = Math.floor(m.a / ROW), s = svgs[r]; s.scrollIntoView({behavior: "smooth", block: "center"});
      const x0 = m.a - r * ROW, x1 = Math.min(m.b, (r + 1) * ROW) - r * ROW;
      const f = el(s, "rect", {x: L + x0 * PX - 2, y: 1, width: (x1 - x0) * PX + 4, height: 60, class: "flash"}); setTimeout(() => f.remove(), 2500); };
    tb.appendChild(tr);
  });
}
Object.keys(MC).forEach(c => { const n = Z.missed.filter(m => m.cat === c).length, lb = document.createElement("label");
  lb.innerHTML = `<input type="checkbox" checked> ${c} (${n})`; lb.firstChild.onchange = e => { e.target.checked ? on.add(c) : on.delete(c); table(); };
  document.getElementById("filters").appendChild(lb); });
table();
</script></body></html>
"""


def write_page(path: Path, title: str, note: str, gt: np.ndarray, an: dict, fps: float, warn: str = "",
               frames: bool = False) -> None:
    sec = lambda x: round(x / fps, 2)  # noqa: E731
    segs = lambda codes: [[sec(a), sec(b), int(k)] for a, b, k in pm.segments(codes, an["runs"])]  # noqa: E731
    pick = lambda xs: [{"a": sec(s["a"]), "b": sec(s["b"]), "k": int(s["k"]), "cat": s["cat"], "iou": round(s.get("iou", 0.0), 3)}  # noqa: E731
                       for s in xs if s["cat"] != "found"]
    true, pred = an["true"], an["pred"]
    found = sum(s["cat"] == "found" for s in true)
    cards = [[f"{an['spell_f1']:.3f}", "spellF1"], [str(len(true)), "true possessions"], [str(len(pred)), "predicted possessions"],
             [f"{100 * found / max(len(true), 1):.0f}%", "found at IoU 0.5"]]
    cards += [[str(sum(s["cat"] == c for s in true)), f"missed: {c}"] for c in MISS]
    data = {"fps": fps, "frames": frames, "runs": [[sec(a), sec(b)] for a, b in an["runs"]], "truth": segs(gt),
            "pred": segs(an["filled"]), "missed": pick(true), "spurious": pick(pred), "cards": cards}
    html = (PAGE.replace("__TITLE__", title).replace("__NOTE__", note)
            .replace("__WARN__", f'<div class="warn">{warn}</div>' if warn else "").replace("__DATA__", json.dumps(data)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def fold(key: str, model: str, cond: str, out: Path | None) -> None:
    from katena.eval import final as F

    dev = F.dev_refs()
    r = next((x for x in dev if x.key == key), None)
    if r is None:
        sys.exit(f"no development match {key}; choose from: " + ", ".join(x.key for x in dev))
    gt, pred, fps = fold_codes(r, dev, model, cond)
    an = analyse(gt, pred, fps)
    out = out or REPO / "runs" / "inspect" / f"{r.dataset}_{r.id}__{model}__{cond}.html"
    write_page(out, f"{key}, held out: {MODEL_TEXT[model]}, {cond} noise",
               "External match, not the provided one. Times are from the first frame of the tracking file."
               + (" SkillCorner has no dead-ball label, so predicted dead ball repeats the last team." if r.is_sc else ""),
               gt, an, fps)
    missed = [s for s in an["true"] if s["cat"] != "found"]
    print(f"{key} [{model}, {cond}]: spellF1 {an['spell_f1']:.4f}, {len(an['true'])} true possessions, {len(an['pred'])} predicted, "
          f"{len(missed)} missed (" + ", ".join(f"{c} {sum(s['cat'] == c for s in missed)}" for c in MISS) + ")")
    print(f"wrote {out}")


def public(csv: Path, out: Path | None) -> None:
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(csv))
    if len(pred) != len(gt):
        sys.exit("the submission and the public ground truth differ in length; run evaluation/validate_submission.py")
    an = analyse(gt, pred, pm.FPS)
    out = out or REPO / "artifacts" / "inspect" / "public_window.html"
    write_page(out, "Public window of the provided match", "Times are frame / 25 from row 0 of the submission.",
               gt, an, float(pm.FPS), frames=True,
               warn="Pipeline check only. Do not tune on this window or choose between two versions with it, and add this look "
                    "to docs/decision-log.md. Do not compare this page with the video: the organisers forbid checking output "
                    "against the video (clarification of 2026-11-03). This file holds frame-level labels of the match: keep it local.")
    print(f"wrote {out}")                                            # nothing else: no score leaves the page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--models", default="F,LT")
    s.add_argument("--cond", default="target")
    f = sub.add_parser("fold")
    f.add_argument("key")
    f.add_argument("--model", default="LT", choices=list(MODELS))
    f.add_argument("--cond", default="target")
    f.add_argument("--out", type=Path)
    p = sub.add_parser("public")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.models.split(","), a.cond)
    elif a.cmd == "fold":
        fold(a.key, a.model, a.cond, a.out)
    else:
        public(a.csv, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
