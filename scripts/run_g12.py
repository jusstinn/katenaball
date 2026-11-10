#!/usr/bin/env python3
"""G1 and G2: a larger LightGBM member and a longer-trained TCN member for F. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final     # the cache `make model` built
    uv run python scripts/run_g12.py g1-gbdt g1-tcn g1-score g1-gate    # G1 gate, 28 folds
    uv run python scripts/run_g12.py g2-train g2-report --folds 8       # G2 screen, 8 IDSSE + Metrica folds
    uv run python scripts/run_g12.py g2-train g2-report --variants L    # G2 gate of one variant, 28 folds
    scripts/night_g12.sh                                                # all of it, in that order

Pre-registered in docs/decision-log.md (2026-11-09, "G1 and G2 (pre-registration)") before any SkillCorner
fold or 28-fold score of either candidate. Never reads ``data/`` or ``evaluation/gt/``, never trains on or
scores a locked bench. LOMO on the 27 other development matches. Every stage caches per fold under
``KATENA_WORK/oof/`` and skips what exists, so an interrupted run resumes.

G1. One LightGBM model of 1,200 trees per fold (``gbdt.fit_heads``, F's pool, columns and parameters), read at
150 trees (F's member: LightGBM's rounds are sequential, so this equals fitting 150) and at 1,200 (the
candidate). The TCN member is F's, unchanged, trained here as ``final.stage_tcn`` trains it.

G2. F's TCN with 4,800 steps instead of 1,200 (``L``), and the same reading the 53 touch columns as well
(``LT``). Blended at w = 0.5 with the LightGBM member that stands after G1's gate, against F's TCN blended
with the same member.

``--tag`` keeps a smoke run's files apart; the registered sizes can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("KATENA_WORK", str(REPO / "work" / "final"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()                             # before torch is imported anywhere
from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import stages as S  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.sim import worlds as W  # noqa: E402

T = P.T_MAIN
N_INC = gbdt.N_TREES                        # 150, F's LightGBM member
N_CAND = 1200                               # G1, registered
G2_STEPS = 4800                             # G2, registered (F: tcn.STEPS = 1200)
G2 = {"L": False, "LT": True}               # variant -> reads the touch columns
SEED = 0
STAGES = ("g1-gbdt", "g1-tcn", "g1-score", "g1-gate", "g2-train", "g2-report", "g2-pick")
HEAD = ("| comparison | folds | candidate | incumbent | diff | 95% paired CI | wins | adoption rule | worse |\n"
        "|---|---|---|---|---|---|---|---|---|")


class Names:
    """Cache names of a run; a tag keeps a smoke run apart from the registered one."""

    def __init__(self, tag: str = ""):
        self.p = f"{tag}_" if tag else ""
        self.tcn = f"{self.p}g1_tcn"

    def gbdt(self, n: int) -> str:
        return f"{self.p}g1_gbdt{n}"

    def g2(self, k: str, steps: int) -> str:
        return f"{self.p}g2_tcn{steps}{'t' if G2[k] else ''}"

    def file(self, name: str) -> Path:
        return P.WORK / f"{self.p}{name}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _folds(dev: list, spec: str) -> list:
    if spec == "8":
        return S.FOLDS8(dev)
    if spec == "all":
        return list(dev)
    keys = set(spec.split(","))
    return [r for r in dev if r.key in keys]


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
                    encoding="utf-8", newline="\n")
    print("wrote", path, flush=True)


def _row(label: str, g: dict | None) -> str:
    if g is None:
        return f"| {label} | 0 | | | | | | | |"
    return (f"| {label} | {g['n']} | {g['cand_mean']:.4f} | {g['inc_mean']:.4f} | {g['mean_diff']:+.4f} | "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}] | {g['wins']}/{g['n']} ({100 * g['win_rate']:.0f}%) | "
            f"{'yes' if g['adopted'] else 'no'} | {'yes' if g['worse'] else 'no'} |")


def worlds_here(r: P.Ref) -> list:
    """The evaluation worlds of ``r`` whose features this cache holds. `make model` builds the training
    worlds only, so a cache made by it lacks ``target_cpl100_ids`` (information only, never a gate world)."""
    here = [w for w in F.eval_worlds(r) if C.paths(r)["feat"](w).exists() and TG.tpaths(r)["feat"](w).exists()]
    assert set(F.GATE_CONDS) <= set(here), f"{r.key}: a gate world is missing from the cache"
    return here


def blend_score(r: P.Ref, c: str, g: dict, t: dict, w: float, pri, gt) -> dict:
    """Blend of a LightGBM and a TCN probability file at TCN weight ``w``, decoded and scored as F."""
    pa, ph = E.blend_heads((g[f"pa_{c}"], g[f"ph_{c}"]), (t[f"pa_{c}"], t[f"ph_{c}"]), w)
    return F.decode_score(r, c, pa, ph, pri, gt)


def per_true(get, refs: list, name: str, cond: str) -> float:
    s = [get(name, cond, r) for r in refs]
    return sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)


def same_as(variant: str, old: str, folds: list) -> tuple:
    """(folds whose probabilities equal the older cache's byte for byte, folds compared)."""
    same = n = 0
    for r in folds:
        if Lm.oof_path(variant, T, r).exists() and Lm.oof_path(old, T, r).exists():
            a, b = Lm.load_oof(variant, T, r), Lm.load_oof(old, T, r)
            n += 1
            same += int(all(a[k].tobytes() == b[k].tobytes() for k in b if k[:3] in ("pa_", "ph_")))
    return same, n


# ---------------------------------------------------------------------------------------------
# G1

def predict_at(heads: gbdt.Heads, X0: np.ndarray, X1: np.ndarray, n: int) -> tuple:
    """``Heads.predict`` using the first ``n`` trees."""
    c = heads.cols
    pa = 0.5 * (heads.alive.predict(X0[:, c], num_iteration=n) + heads.alive.predict(X1[:, c], num_iteration=n))
    ph = 0.5 * (heads.team.predict(X0[:, c], num_iteration=n) + 1.0 - heads.team.predict(X1[:, c], num_iteration=n))
    return pa.astype(np.float32), ph.astype(np.float32)


def g1_gbdt(nm: Names, dev: list, folds: list, n_cand: int) -> None:
    sizes = (N_INC, n_cand)
    todo = [r for r in folds if not all(Lm.oof_path(nm.gbdt(n), T, r).exists() for n in sizes)]
    if not todo:
        return
    pools = F.FinalPools(dev)
    cols = pools.cols()
    log(f"pool {pools.X.shape}; LightGBM folds to do: {len(todo)}")
    for r in todo:
        t0 = time.time()
        m = ~np.isin(pools.keys, [r.key])
        heads = gbdt.fit_heads(pools.X[m], pools.o[m], pools.y[T][m], pools.dead_ok[m], cols, n_trees=n_cand)
        t1 = time.time()
        arrays = {n: {} for n in sizes}
        for c in worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            for n in sizes:
                arrays[n][f"pa_{c}"], arrays[n][f"ph_{c}"] = predict_at(heads, X0, X1, n)
        sha = F.gbdt_hash(heads)
        for n in sizes:
            d3._save(Lm.oof_path(nm.gbdt(n), T, r), {**arrays[n], "model_sha": np.array(sha), "trees": np.array(n)})
        log(f"LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s")


def g1_tcn(nm: Names, dev: list, folds: list, steps: int | None) -> None:
    """F's TCN member for every fold, as ``final.stage_tcn`` trains it, predicted in ``worlds_here``."""
    todo = [r for r in folds if not Lm.oof_path(nm.tcn, T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev, touch=False)
    t0 = time.time()
    seqs = F.tcn_seqs(dev, T, SEED)
    log(f"TCN sequences: {len(seqs)}, {sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB ({time.time() - t0:.0f}s)")
    for r in todo:
        t1 = time.time()
        model, sc = F.fit_tcn(pools, seqs, {r.key}, SEED, E.TCN_THREADS, steps)
        arrays = d3.predict_tcn(model, sc, r, worlds_here(r), E.TCN_THREADS)
        arrays["model_sha"] = np.array(E.model_hash(model))
        d3._save(Lm.oof_path(nm.tcn, T, r), arrays)
        log(f"TCN fold {r.key}: {time.time() - t1:.0f}s")


def g1_score(nm: Names, dev: list, folds: list, n_cand: int) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    for r in folds:
        vs = (nm.gbdt(N_INC), nm.gbdt(n_cand), nm.tcn)
        if not all(Lm.oof_path(v, T, r).exists() for v in vs):
            continue
        g0, g1, t = (Lm.load_oof(v, T, r) for v in vs)
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in worlds_here(r):
            for name, g, w in (("F", g0, F.W_F), ("G1", g1, F.W_F), ("FG", g0, 0.0), ("G1G", g1, 0.0), ("FT", g0, 1.0)):
                sc[(name, c, r.key)] = blend_score(r, c, g, t, w, pri, gt)
    with open(nm.file("g1_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    log(f"G1 scored: {len({k[2] for k in sc})} folds")
    return sc


def g1_decision(g_target: dict, g_severe: dict, g_sc_target: dict | None) -> bool:
    """Registered: F's rule on all folds, and not worse at target on the SkillCorner folds alone."""
    return bool(F.decision(g_target, g_severe) and not (g_sc_target is not None and g_sc_target["worse"]))


def g1_gate(nm: Names, dev: list, folds: list, n_cand: int) -> dict:
    with open(nm.file("g1_scores.pkl"), "rb") as fh:
        sc = pickle.load(fh)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    done = [r for r in folds if get("G1", "target", r) is not None]
    sets = (("all", done), ("SkillCorner", [r for r in done if r.is_sc]), ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    registered = not nm.p and n_cand == N_CAND and len(done) == len(dev) == 28
    out = {"registered_run": registered, "folds": len(done), "trees": [N_INC, n_cand], "gate": {}, "info": {}}
    print(f"\n## G1 gate: F with {n_cand} trees (G1) vs F with {N_INC} (F), {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "G1", "F", c)
            if g is not None:
                out["gate"][f"{tag}/{c}"] = g
                print(_row(f"G1 vs F, {c}, {tag}", g))
    gt_, gs_, gsc = (out["gate"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["adopt_G1"] = bool(gt_ and gs_ and g1_decision(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nDECISION ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): adopted at "
              f"target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on SkillCorner at target = "
              f"{gsc['worse'] if gsc else 'n/a'} -> {'G1 PASSES' if out['adopt_G1'] else 'F stays'}")
    print("\n## Information: the LightGBM member alone (G1G vs FG)\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, "G1G", "FG", c)
            if g is not None:
                out["info"][f"member/{tag}/{c}"] = g
                print(_row(f"G1G vs FG, {c}, {tag}", g))
    print("\n## Information: every evaluation world (coupled worlds: IDSSE + Metrica only)\n" + HEAD)
    for wld in W.EVAL_WORLDS:
        g = F.compare(get, done, "G1", "F", wld)
        if g is not None:
            out["info"][f"world/{wld}"] = g
            print(_row(f"G1 vs F, {wld}", g))
    print("\n## Information: means, macroF1 and predicted possessions per true one\n"
          "| cond | folds | F | G1 | FG | G1G | FT | macroF1 F | macroF1 G1 | per true F | per true G1 |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            if not refs:
                continue
            mean = lambda name, key="spell_f1": float(np.mean([get(name, c, r)[key] for r in refs]))  # noqa: E731
            v = {"F": mean("F"), "G1": mean("G1"), "FG": mean("FG"), "G1G": mean("G1G"), "FT": mean("FT"),
                 "macro_F": mean("F", "macro_f1"), "macro_G1": mean("G1", "macro_f1"),
                 "per_true_F": per_true(get, refs, "F", c), "per_true_G1": per_true(get, refs, "G1", c)}
            out["info"][f"means/{tag}/{c}"] = v
            print(f"| {c} | {tag} ({len(refs)}) | " + " | ".join(f"{v[k]:.4f}" for k in ("F", "G1", "FG", "G1G", "FT", "macro_F", "macro_G1"))
                  + f" | {v['per_true_F']:.3f} | {v['per_true_G1']:.3f} |")
    out["checks"] = {"gbdt150_equals_screen_gbdt": same_as(nm.gbdt(N_INC), "screen_gbdt", done),
                     "tcn_equals_screen_tcn_s0": same_as(nm.tcn, "screen_tcn_s0", done)}
    print(f"\nchecks (folds identical to the F2 screen's cache, of those it holds): {out['checks']}")
    print("side check: F at target on 28 folds was 0.6019 on the Windows PC (decision log, F's result)")
    _dump(nm.file("g1_gate.json"), out)
    return out


# ---------------------------------------------------------------------------------------------
# G2

def add_touch(dev: list, seqs: dict) -> int:
    """Append each sequence's touch columns (same world, same orientation), held in float16 like the rest.
    Returns how many finite values the cast made non-finite."""
    by_key = {r.key: r for r in dev}
    lost = 0
    for (key, w), s in seqs.items():
        r = by_key[key]
        t = TG.load_tfeat(r, w)
        with np.load(C.paths(r)["feat"](w)) as z:
            idx = z["idx"]
        if not np.array_equal(t["idx"], idx):
            raise RuntimeError(f"{key} {w}: touch rows do not line up")
        A = t["A1" if s["o"] else "A0"]
        with np.errstate(over="ignore"):
            A16 = A.astype(np.float16)
        lost += int((np.isfinite(A) & ~np.isfinite(A16)).sum())
        s["X"] = np.concatenate([s["X"], A16], 1)
    return lost


def fit_tcn_touch(pools: F.FinalPools, seqs: dict, key: str, steps: int):
    """``final.fit_tcn`` with the scaler over every column (base and touch) of the same pool rows."""
    from katena.models import tcn
    rows = np.flatnonzero(~np.isin(pools.keys, [key]))[::6]
    sc = tcn.Scaler(np.ascontiguousarray(pools.X[rows]))
    tr = [{**s, "X": F.ScaledSeq(s["X"], sc)} for s in seqs.values() if s["key"] != key]
    return E.fit_tcn(tr, seed=SEED, threads=E.TCN_THREADS, steps=steps), sc


def predict_tcn_touch(model, sc, r: P.Ref, conds) -> dict:
    from katena.models import tcn
    E.set_deterministic(E.TCN_THREADS)
    out = {}
    for c in conds:
        f, X0, X1 = TG.features(r, c)
        out[f"pa_{c}"], out[f"ph_{c}"] = tcn.predict(model, sc(X0), sc(X1), f["half"])
    return out


def g2_train(nm: Names, dev: list, folds: list, keys: list, steps: int) -> None:
    pools = None
    for k in keys:
        var = nm.g2(k, steps)
        todo = [r for r in folds if not Lm.oof_path(var, T, r).exists()]
        if not todo:
            continue
        pools = pools or F.FinalPools(dev)
        t0 = time.time()
        seqs = F.tcn_seqs(dev, T, SEED)
        if G2[k]:
            lost = add_touch(dev, seqs)
            log(f"{k}: touch columns appended; finite values lost to float16: {lost}")
        log(f"{k}: {len(seqs)} sequences x {next(iter(seqs.values()))['X'].shape[1]} columns, "
            f"{sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB, {steps} steps ({time.time() - t0:.0f}s)")
        for r in todo:
            t1 = time.time()
            if G2[k]:
                model, sc = fit_tcn_touch(pools, seqs, r.key, steps)
                arrays = predict_tcn_touch(model, sc, r, worlds_here(r))
            else:
                model, sc = F.fit_tcn(pools, seqs, {r.key}, SEED, E.TCN_THREADS, steps)
                arrays = d3.predict_tcn(model, sc, r, worlds_here(r), E.TCN_THREADS)
            arrays["model_sha"] = np.array(E.model_hash(model))
            d3._save(Lm.oof_path(var, T, r), arrays)
            log(f"{k} fold {r.key}: {time.time() - t1:.0f}s")
        del seqs


def g2_pick(verdict: dict, means: dict, keys: list) -> str | None:
    """The passing variant with the higher target mean; the earlier one in ``keys`` wins a tie."""
    ok = [k for k in keys if verdict.get(k)]
    return max(ok, key=lambda k: (means[k], -keys.index(k))) if ok else None


def g2_report(nm: Names, dev: list, folds: list, keys: list, steps: int) -> dict:
    gate = json.loads(nm.file("g1_gate.json").read_text(encoding="utf-8"))
    side = gate["trees"][1] if gate["adopt_G1"] else N_INC
    sizes = tuple(gate["trees"])
    need = lambda r: [nm.gbdt(n) for n in sizes] + [nm.tcn] + [nm.g2(k, steps) for k in keys]  # noqa: E731
    done = [r for r in folds if all(Lm.oof_path(v, T, r).exists() for v in need(r))]
    src = F.PriorSource(dev)
    sc = {}
    for r in done:
        gb = {n: Lm.load_oof(nm.gbdt(n), T, r) for n in sizes}
        tc = {"F": Lm.load_oof(nm.tcn, T, r), **{k: Lm.load_oof(nm.g2(k, steps), T, r) for k in keys}}
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in worlds_here(r):
            for k, t in tc.items():
                for n in sizes:
                    sc[(f"{k}+{n}", c, r.key)] = blend_score(r, c, gb[n], t, F.W_F, pri, gt)
                sc[(f"{k} alone", c, r.key)] = blend_score(r, c, gb[sizes[0]], t, 1.0, pri, gt)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    kind = "screen (8 IDSSE + Metrica folds)" if len(folds) == 8 else f"gate ({len(folds)} folds)"
    registered = (not nm.p and steps == G2_STEPS and gate["registered_run"] and len(done) == len(folds)
                  and len(folds) in (8, 28))
    out = {"registered_run": registered, "kind": kind, "folds": len(done), "side_trees": side, "steps": steps,
           "rule": {}, "info": {}, "verdict": {}}
    print(f"\n## G2 {kind}: TCN at {steps} steps vs F's TCN, both blended with the {side}-tree LightGBM member "
          f"(G1 {'passed' if gate['adopt_G1'] else 'did not pass'}), {len(done)} folds scored\n" + HEAD)
    means = {}
    for k in keys:
        g = {c: F.compare(get, done, f"{k}+{side}", f"F+{side}", c) for c in F.GATE_CONDS}
        if any(v is None for v in g.values()):
            continue
        for c in F.GATE_CONDS:
            out["rule"][f"{k}/{c}"] = g[c]
            print(_row(f"{k} vs F's TCN, {c}", g[c]))
        out["verdict"][k] = F.decision(g["target"], g["severe"])
        means[k] = g["target"]["cand_mean"]
    out["pick"] = g2_pick(out["verdict"], means, keys)
    print(f"\nVERDICT ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): "
          + ", ".join(f"{k} {'passes' if v else 'fails'}" for k, v in out["verdict"].items())
          + f" -> {out['pick'] or 'no variant goes on'}")
    other = [n for n in sizes if n != side]
    print("\n## Information: with the other LightGBM size, the TCN alone, and every evaluation world\n" + HEAD)
    for k in keys:
        for n in other:
            for c in F.GATE_CONDS:
                g = F.compare(get, done, f"{k}+{n}", f"F+{n}", c)
                out["info"][f"{k}+{n}/{c}"] = g
                print(_row(f"{k} vs F's TCN, {n} trees, {c}", g))
        for c in F.GATE_CONDS:
            g = F.compare(get, done, f"{k} alone", "F alone", c)
            out["info"][f"{k} alone/{c}"] = g
            print(_row(f"{k} alone vs F's TCN alone, {c}", g))
        for wld in W.EVAL_WORLDS:
            if wld not in F.GATE_CONDS:
                g = F.compare(get, done, f"{k}+{side}", f"F+{side}", wld)
                if g is not None:
                    out["info"][f"{k}/world/{wld}"] = g
                    print(_row(f"{k} vs F's TCN, {wld}", g))
    print("\n## Information: macroF1 and predicted possessions per true one, at target\n"
          "| blend | macroF1 | per true |\n|---|---|---|")
    for k in ["F"] + keys:
        if done and get(f"{k}+{side}", "target", done[0]) is not None:
            m = float(np.mean([get(f"{k}+{side}", "target", r)["macro_f1"] for r in done]))
            print(f"| {k}+{side} | {m:.4f} | {per_true(get, done, f'{k}+{side}', 'target'):.3f} |")
    _dump(nm.file(f"g2_report_{len(folds)}.json"), out)
    return out


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--variants", default="L,LT", help="G2 variants: L (longer), LT (longer, with touch columns)")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_CAND, help="G1 size (needs --tag)")
    ap.add_argument("--tcn-steps", type=int, default=None, help="steps of F's TCN member (needs --tag)")
    ap.add_argument("--g2-steps", type=int, default=G2_STEPS, help="G2 steps (needs --tag)")
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and (a.trees != N_CAND or a.tcn_steps is not None or a.g2_steps != G2_STEPS):
        ap.error("the registered sizes can only be changed together with --tag")
    keys = a.variants.split(",")
    if not set(keys) <= set(G2):
        ap.error(f"unknown variant in {keys}")
    nm = Names(a.tag)
    if a.stages == ["g2-pick"]:                 # for night_g12.sh: prints the screen's pick, or nothing
        p = nm.file("g2_report_8.json")
        rep = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        print(rep.get("pick") or "" if rep.get("registered_run") or a.tag else "")
        return 0
    dev = F.dev_refs()
    folds = _folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for stage in a.stages:
        t0 = time.time()
        if stage == "g1-gbdt":
            g1_gbdt(nm, dev, folds, a.trees)
        elif stage == "g1-tcn":
            g1_tcn(nm, dev, folds, a.tcn_steps)
        elif stage == "g1-score":
            g1_score(nm, dev, folds, a.trees)
        elif stage == "g1-gate":
            g1_gate(nm, dev, folds, a.trees)
        elif stage == "g2-train":
            g2_train(nm, dev, folds, keys, a.g2_steps)
        elif stage == "g2-report":
            g2_report(nm, dev, folds, keys, a.g2_steps)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
