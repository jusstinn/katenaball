#!/usr/bin/env python
"""Event evidence for the label convention (D4), external development matches only.

    KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-d4 \
        uv run python scripts/label_evidence.py [--metrica]

Reruns David's control-evidence derivation (origin/david/foundation) on Justin's loaders with the
locked benches excluded (``protocol.split``): the IDSSE bench DFL-MAT-J03WQQ and Metrica
Sample_Game_2 are never loaded. Prints markdown tables; writes a JSON summary and the event-defined
labels (``control_evidence.control_labels``) of each IDSSE development match under
``$KATENA_WORK/labels/`` for ``scripts/label_tmix.py``. Never touches data/ or evaluation/gt/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

import numpy as np  # noqa: E402

import possession_metrics as pm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.external import control_evidence as ce  # noqa: E402
from katena.external import idsse, metrica  # noqa: E402
from katena.labels import sandwich  # noqa: E402
from katena.types import load_match  # noqa: E402

CONVENTIONS = {f"S{T}": dict(T=T) for T in P.T_SWEEP}
CONVENTIONS.update({"S2tail": dict(T=2, tail=True), "H3.5": dict(T=3.5, tail=True, order="shortest")})


def spell_stats(y: np.ndarray, fps: float) -> dict:
    runs = pm.valid_runs(y)
    d = np.array([b - a for a, b, _ in pm.possession_spells(y, runs)]) / fps
    return {"n": int(len(d)), "mean_s": float(d.mean()), "median_s": float(np.median(d)),
            "lt2": float((d < 2).mean())}


def convention(y: np.ndarray, fps: float, name: str) -> np.ndarray:
    kw = dict(CONVENTIONS[name])
    T = kw.pop("T")
    return sandwich(y, pm.valid_runs(y), fps, T, **kw)


def idsse_match(ref) -> dict:
    m = load_match(ref.path)
    files = idsse.match_files(P.EXTERNAL / "idsse" / "raw")[ref.id]
    home = idsse.parse_info(files["info"])["teams"]["home"]["id"]
    acts = ce.read_dfl_actions(files["events"], home)
    t0 = datetime.fromisoformat(m.meta["t0_utc"]).timestamp()
    rows, diag = ce.align_dfl(m.y, t0, m.fps, acts)
    ctrl = np.isin(acts["tag"], ce.CONTROL_TAGS)
    sit = ce.read_dfl_situations(files["events"], home)
    srows = np.rint((sit["t"] - t0) * m.fps).astype(np.int64) + int(round(diag["shift_s"] * m.fps))
    srows = np.where((srows >= 0) & (srows < m.n), srows, -1)
    out = {"name": ref.id, "fps": m.fps, "y": m.y, "rows": rows, "team": acts["team"], "diag": diag,
           "keep": ctrl, "keep_strict": ctrl & acts["done"], "sit": {**sit, "rows": srows}}
    return out


def metrica_match(ref) -> dict:
    m = load_match(ref.path)
    gd = P.EXTERNAL / "metrica" / "repo" / "data" / ref.id
    if ref.id == "Sample_Game_3":
        meta = metrica._parse_epts_meta(gd / "Sample_Game_3_metadata.xml")
        events = metrica._read_json_events(gd / "Sample_Game_3_events.json", meta["local"])
    else:
        events = metrica._read_csv_events(gd / f"{ref.id}_RawEventsData.csv")
    acts = ce.metrica_actions(events, m.fps)
    return {"name": ref.id, "fps": m.fps, "y": m.y, "rows": acts["rows"], "team": acts["team"],
            "diag": {"shift_s": 0.0, "play_team_agreement": float("nan"), "n_actions": int(len(acts["rows"]))},
            "keep": None, "keep_strict": None}


def evidence(ms: list, kinds, strict: bool = False) -> dict:
    per, kind = {}, {}
    for d in ms:
        ev = ce.spell_evidence(d["y"], d["rows"], d["team"], d["fps"], kinds=kinds,
                               keep=d["keep_strict"] if strict else d["keep"])
        per[d["name"]] = (ev["dur"], ev["has"])
        kind[d["name"]] = ev["kind"]
    return {"per": per, "kind": kind, "derived": ce.derive_threshold(per)}


def fmt_bins(rows) -> str:
    lines = ["| duration of flip | flips | share with a deliberate action by that team |", "|---|---|---|"]
    for lo, hi, n, share in rows:
        lab = f"{lo:g} s and over" if not np.isfinite(hi) else f"{lo:g} - {hi:g} s"
        lines.append(f"| {lab} | {n} | {share:.2f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica", action="store_true", help="also run Metrica development games (circular)")
    a = ap.parse_args()
    datasets = ("idsse", "metrica") if a.metrica else ("idsse",)
    refs = [r for r in P.all_refs() if r.dataset in datasets]
    dev, locked = P.split(refs)
    print("excluded (locked benches):", sorted(r.key for r in locked))
    ids = [idsse_match(r) for r in dev if r.dataset == "idsse"]
    print("IDSSE development matches:", [d["name"] for d in ids])
    summary = {"matches": [d["name"] for d in ids], "locked_excluded": sorted(r.key for r in locked)}

    # 1. the port: David's definition (return + to_dead flips, any Play/Shot/Claim) ---------------
    main_ev = evidence(ids, ("return", "to_dead"))
    der = main_ev["derived"]
    print("\n## David's definition rerun without the locked bench (IDSSE dev)\n")
    print(fmt_bins(der["by_bin"]))
    print("\n| match | flips | clock shift s | Play-team agreement | best t (own) | best t (LOMO) | held-out accuracy |")
    print("|---|---|---|---|---|---|---|")
    for d in ids:
        n = d["name"]
        print(f"| {n} | {len(main_ev['per'][n][0])} | {d['diag']['shift_s']:+.2f} | "
              f"{d['diag']['play_team_agreement']:.3f} | {der['per_match'][n]:.1f} | {der['lomo'][n][0]:.1f} | "
              f"{der['lomo'][n][1]:.3f} |")
    pooled = der["pooled"]
    grid = pooled["grid"]
    pick = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    print(f"\npooled over {der['n_flips']} flips: accuracy-best t = {pooled['accuracy_best_t']:.1f} s "
          f"(accuracy {pooled['accuracy']:.3f}); balanced-best t = {pooled['balanced_best_t']:.1f} s "
          f"({pooled['balanced_accuracy']:.3f}); bootstrap median {der['bootstrap_median']:.1f} s, "
          f"90% [{der['bootstrap_p5_p95'][0]:.1f}, {der['bootstrap_p5_p95'][1]:.1f}]")
    print("| t (s) | " + " | ".join(f"{t:.1f}" for t in pick) + " |")
    print("|---|" + "---|" * len(pick))
    print("| pooled accuracy | " + " | ".join(f"{pooled['acc_curve'][int(np.argmin(np.abs(grid - t)))]:.3f}"
                                            for t in pick) + " |")
    summary["david_definition"] = _jsonable(der)

    # 2. variants ----------------------------------------------------------------------------------
    variants = {"return only (what sandwich merges)": dict(kinds=("return",)),
                "to_dead only (what sandwich keeps)": dict(kinds=("to_dead",)),
                "strict: completed plays, shots, claims": dict(kinds=("return", "to_dead"), strict=True)}
    print("\n## Variants\n")
    print("| variant | flips | accuracy-best t | accuracy | balanced-best t | bootstrap median [90%] |")
    print("|---|---|---|---|---|---|")
    summary["variants"] = {}
    for label, kw in variants.items():
        v = evidence(ids, **kw)["derived"]
        p = v["pooled"]
        print(f"| {label} | {v['n_flips']} | {p['accuracy_best_t']:.1f} | {p['accuracy']:.3f} | "
              f"{p['balanced_best_t']:.1f} | {v['bootstrap_median']:.1f} [{v['bootstrap_p5_p95'][0]:.1f}, "
              f"{v['bootstrap_p5_p95'][1]:.1f}] |")
        summary["variants"][label] = _jsonable(v)
    k_all = np.concatenate([main_ev["kind"][d["name"]] for d in ids])
    dur_all = np.concatenate([main_ev["per"][d["name"]][0] for d in ids])
    print(f"\nflips under 2 s: return {int(((k_all == 'return') & (dur_all < 2)).sum())}, "
          f"to_dead {int(((k_all == 'to_dead') & (dur_all < 2)).sum())} (raw labels, {len(ids)} matches)")

    # 3. what each convention does to the spells, and the event-defined convention ----------------
    print("\n## Spell statistics per convention (IDSSE dev, pooled)\n")
    print("| convention | spells/match | mean s | median s | < 2 s | short to_dead flips left/match |")
    print("|---|---|---|---|---|---|")
    out_dir = P.WORK / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["conventions"] = {}
    names = list(CONVENTIONS) + ["E"]
    for name in names:
        st, left = [], 0
        for d in ids:
            if name == "E":
                yc = ce.control_labels(d["y"], d["rows"], d["team"], d["fps"], keep=d["keep"])
                np.savez_compressed(out_dir / f"idsse_{d['name']}__E.npz", y=yc)
            else:
                yc = convention(d["y"], d["fps"], name)
            st.append(spell_stats(yc, d["fps"]))
            left += sum(1 for a, b, _, kd in ce.spell_kinds(yc) if kd == "to_dead" and (b - a) < 2 * d["fps"])
        row = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
        row["to_dead_lt2_per_match"] = left / len(ids)
        summary["conventions"][name] = row
        print(f"| {name} | {row['n']:.0f} | {row['mean_s']:.1f} | {row['median_s']:.1f} | {row['lt2']:.1%} | "
              f"{row['to_dead_lt2_per_match']:.1f} |")

    # 4. what the per-frame owner does where DFL's event operators judged possession -------------
    print("\n## DFL per-frame owner around events DFL itself scored (IDSSE dev, pooled)\n")
    print("| labels | duels scored 'no possession change' with a brief owner flip within 1 s | same at random "
          "alive frames | duels scored 'possession change': owner = winner 2 s later | clearances going dead "
          "within 3 s: brief flip to the clearing team just before |")
    print("|---|---|---|---|---|")
    summary["owner_behaviour"] = {}
    for name in ("S0", "S2", "S5", "S2tail", "H3.5", "E"):
        acc = {}
        for d in ids:
            if name == "E":
                yc = np.load(out_dir / f"idsse_{d['name']}__E.npz")["y"]
            else:
                yc = convention(d["y"], d["fps"], name)
            ob = ce.owner_behaviour(yc, d["fps"], d["sit"]["rows"], d["sit"]["kind"], d["sit"]["team"])
            for k, v in ob.items():
                acc.setdefault(k, []).append(v)
        w = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
        row = {"duel_kept_flip": float(np.average(w["duel_kept_flip"], weights=w["n_duel_kept"])),
               "base_flip": float(w["base_flip"].mean()),
               "duel_changed_winner": float(np.average(w["duel_changed_winner"], weights=w["n_duel_changed"])),
               "clearance_out_flip": float(np.average(w["clearance_out_flip"], weights=w["n_clearance_out"])),
               "n_duel_kept": int(w["n_duel_kept"].sum()), "n_duel_changed": int(w["n_duel_changed"].sum()),
               "n_clearance_out": int(w["n_clearance_out"].sum())}
        summary["owner_behaviour"][name] = row
        print(f"| {name} | {row['duel_kept_flip']:.2f} (n={row['n_duel_kept']}) | {row['base_flip']:.2f} | "
              f"{row['duel_changed_winner']:.2f} (n={row['n_duel_changed']}) | {row['clearance_out_flip']:.2f} "
              f"(n={row['n_clearance_out']}) |")

    # 5. model-free distance between conventions: perfect labels of one scored against another ----
    print("\n## spellF1 of one convention's labels (row) scored against another's (column), IDSSE dev mean\n")
    print("| labels \\ truth | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    labs = []
    for d in ids:
        lab = {n: convention(d["y"], d["fps"], n) for n in CONVENTIONS}
        lab["E"] = np.load(out_dir / f"idsse_{d['name']}__E.npz")["y"]
        labs.append(lab)
    summary["convention_distance"] = {}
    for p_name in names:
        row = [float(np.mean([pm.evaluate(lab[t_name], lab[p_name])["spell_f1_mean"] for lab in labs]))
               for t_name in names]
        summary["convention_distance"][p_name] = dict(zip(names, row))
        print(f"| {p_name} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    # 6. the other datasets: spell statistics per convention, and SkillCorner's X boundaries -------
    others = [r for r in P.split(P.all_refs())[0] if r.dataset in ("metrica", "skillcorner")]
    summary["other_datasets"] = {}
    for ds in ("metrica", "skillcorner"):
        refs_ds = [r for r in others if r.dataset == ds]
        if not refs_ds:
            continue
        st = {n: [] for n in CONVENTIONS}
        n_sp = n_x = n_short = n_short_x = 0
        for r in refs_ds:
            m = load_match(r.path)
            for n in CONVENTIONS:
                st[n].append(spell_stats(convention(m.y, m.fps, n), m.fps))
            for s0, s1, _, _ in ce.spell_kinds(m.y):
                touches_x = (s0 == 0 or m.y[s0 - 1] == 3) or (s1 >= m.n or m.y[s1] == 3)
                n_sp += 1
                n_x += touches_x
                if (s1 - s0) < 2 * m.fps:
                    n_short += 1
                    n_short_x += touches_x
        print(f"\n## {ds} development matches ({len(refs_ds)}): spell statistics per convention\n")
        print("| convention | spells/match | mean s | median s | < 2 s |")
        print("|---|---|---|---|---|")
        rows_ds = {}
        for n, v in st.items():
            rows_ds[n] = {k: float(np.mean([s[k] for s in v])) for k in v[0]}
            print(f"| {n} | {rows_ds[n]['n']:.0f} | {rows_ds[n]['mean_s']:.1f} | {rows_ds[n]['median_s']:.1f} | "
                  f"{rows_ds[n]['lt2']:.1%} |")
        print(f"raw spells next to an X frame: {n_x / n_sp:.1%} (spells under 2 s: {n_short_x / max(n_short, 1):.1%})")
        summary["other_datasets"][ds] = {"conventions": rows_ds, "share_next_to_x": n_x / n_sp,
                                         "share_next_to_x_lt2": n_short_x / max(n_short, 1)}

    if a.metrica:
        mts = [metrica_match(r) for r in dev if r.dataset == "metrica"]
        mev = evidence(mts, ("return", "to_dead"))["derived"]
        print("\n## Metrica development games (labels derived from the same events: circular)\n")
        print(fmt_bins(mev["by_bin"]))
        print(f"\npooled accuracy-best t = {mev['pooled']['accuracy_best_t']:.1f} s over {mev['n_flips']} flips; "
              f"per game {mev['per_match']}")
        summary["metrica_circular"] = _jsonable(mev)

    (P.WORK / "label_evidence.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {P.WORK / 'label_evidence.json'} and {len(ids)} event-defined label files")
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.round(4).tolist()
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


if __name__ == "__main__":
    sys.exit(main())
