#!/usr/bin/env python3
"""A: SkillCorner training labels from its own action events. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final     # the cache of scripts/night_g12.sh
    uv run python scripts/run_a.py a-labels               # the labels and their aggregates (no model, no score)
    uv run python scripts/run_a.py a-check                # the member code path equals the cache (no score)
    uv run python scripts/run_a.py a-gbdt                 # LightGBM member, 28 folds
    uv run python scripts/run_a.py a-tcn                  # TCN member, 28 folds (--reverse: a second process)
    uv run python scripts/run_a.py a-gate                 # the gate

Pre-registered in docs/decision-log.md (2026-11-12, "A (pre-registration)") before any score. Never reads
``data/`` or ``evaluation/gt/``, never trains on or scores a locked bench. LOMO on the 27 other development
matches. Every stage caches per fold under ``KATENA_WORK`` and skips what exists.

The incumbent is F3, read from its cache (``g1_gbdt1200`` and ``g2_tcn4800t``), not refitted.

The change. The 20 SkillCorner matches are relabelled by ``katena.external.skillcorner_actions``: the owner
changes at the other team's first touch (``player_possession`` events), and the gaps SkillCorner flags as game
interruptions are dead ball. The labels then take the same sandwich merge as every other label (T = 2). Both
members are trained as F3's are, on the same rows, sequences, features, sizes and seeds, with two differences on
the SkillCorner rows only: the label, and ``dead_ok`` (they now carry a dead-ball label, so they train the alive
head and the TCN's alive output too). Nothing changes for an IDSSE or a Metrica row. The decoder's priors are
F3's (IDSSE and Metrica label sequences).

Two truths for a SkillCorner fold. *Standing*: the label every gate so far has scored on (``possession.group``,
predicted dead ball read as no call). *Action*: the label above, scored as an IDSSE fold is. An IDSSE or Metrica
fold has one truth. Names ending in ``@act`` are scores on the action truth.

``--tag`` keeps a smoke run's files apart; the registered sizes can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g12 as G  # noqa: E402  (pins the CPU environment and loads libomp before torch and LightGBM)

import numpy as np  # noqa: E402
import possession_metrics as pm  # noqa: E402

from katena.decode import hmm  # noqa: E402
from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d2  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G1  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.eval.score import score, to_native_codes  # noqa: E402
from katena.external import skillcorner_actions as SA  # noqa: E402
from katena.labels import relabel  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import gbdt  # noqa: E402
from katena.types import load_match  # noqa: E402

T = P.T_MAIN
N_TREES = G.N_CAND                          # 1,200
STEPS = G.G2_STEPS                          # 4,800
INC_G, INC_T = "g1_gbdt1200", "g2_tcn4800t"  # F3's members (run_g12.py)
K_G, K_T = "k1_gbdt1200k", "k2_tcn4800tk"   # K's members (run_k.py), for information
STAGES = ("a-labels", "a-check", "a-gbdt", "a-tcn", "a-gate")
BINS = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))
IOU_HEAD = pm.SPELL_IOUS.index(pm.SPELL_IOU_HEADLINE)
H_, A_, D_, X_ = pm.H, pm.A, pm.D, pm.X
log, HEAD, _row = G.log, G.HEAD, G._row


class Names:
    def __init__(self, tag: str = ""):
        self.p = f"{tag}_" if tag else ""

    def g(self, n: int) -> str:
        return f"{self.p}a_gbdt{n}a"

    def t(self, steps: int) -> str:
        return f"{self.p}a_tcn{steps}ta"

    def file(self, name: str) -> Path:
        return P.WORK / f"{self.p}{name}"


# ---------------------------------------------------------------------------------------------
# labels

def raw_dir(r: P.Ref) -> Path:
    return P.EXTERNAL / "skillcorner" / "repo" / "data" / "matches" / r.id


def act_path(r: P.Ref) -> Path:
    return P.WORK / "gt_act" / f"{r.dataset}_{r.id}.npz"


def build_act(r: P.Ref) -> None:
    """The action labels of one SkillCorner match under every T, in the layout of ``corpus``'s ground-truth file."""
    out = act_path(r)
    if out.exists():
        return
    m = load_match(r.path)
    m2 = dataclasses.replace(m, y=SA.build_labels(raw_dir(r), m))
    gt = {"fps": m.fps, "step": int(round(m.fps / 5.0)), "n_native": m.n, "blocks": np.array(m.half_blocks, dtype=np.int64),
          "y_raw": m2.y}
    for t in P.T_SWEEP:
        gt[f"y_T{t}"] = relabel(m2, t).y
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **gt)
    tmp.replace(out)


def load_act(r: P.Ref, t: int = T) -> dict:
    """The action truth of a SkillCorner match; the one truth of any other match."""
    if not r.is_sc:
        return C.load_gt(r, t)
    with np.load(act_path(r)) as z:
        return {"y": z[f"y_T{t}"], "step": int(z["step"]), "fps": float(z["fps"]), "n_native": int(z["n_native"]),
                "blocks": z["blocks"]}


def _spell_lens(y: np.ndarray, fps: float) -> np.ndarray:
    return np.array([(b - a) / fps for a, b, _ in pm.possession_spells(y, pm.valid_runs(y))])


def a_labels(dev: list) -> dict:
    """Aggregates over the SkillCorner matches (no model, no score): what the labels are made of."""
    sc = [r for r in dev if r.is_sc]
    tot = np.zeros((2, 4))
    lens = {"standing": [], "action": []}
    agree = both = 0
    for r in sc:
        build_act(r)
        old, new = C.load_gt(r, T), load_act(r)
        inplay = np.zeros(old["n_native"], bool)
        for _, a, b in old["blocks"]:
            inplay[a:b + 1] = True
        for i, y in enumerate((old["y"], new["y"])):
            tot[i] += np.bincount(y[inplay], minlength=4)[:4]
        lens["standing"].append(_spell_lens(old["y"], old["fps"]))
        lens["action"].append(_spell_lens(new["y"], new["fps"]))
        m = inplay & (old["y"] < 2) & (new["y"] < 2)
        agree += int((old["y"][m] == new["y"][m]).sum())
        both += int(m.sum())
    out = {"matches": len(sc), "team_agreement_where_both_name_a_team": agree / max(both, 1)}
    print(f"\n## A labels: {len(sc)} SkillCorner matches, T = {T}\n| label | home | away | dead | unknown | possessions a match | "
          "median s | under 2 s | under 4 s |\n|---|---|---|---|---|---|---|---|---|")
    for i, name in enumerate(("standing", "action")):
        sh = tot[i] / tot[i].sum()
        ln = np.concatenate(lens[name])
        out[name] = {"share": sh.tolist(), "spells_per_match": len(ln) / len(sc), "median_s": float(np.median(ln)),
                     "under_2s": float((ln < 2).mean()), "under_4s": float((ln < 4).mean())}
        print(f"| {name} | {sh[0]:.3f} | {sh[1]:.3f} | {sh[2]:.3f} | {sh[3]:.3f} | {len(ln) / len(sc):.0f} | "
              f"{np.median(ln):.1f} | {(ln < 2).mean():.3f} | {(ln < 4).mean():.3f} |")
    print(f"team agreement on the frames where both labels name a team: {out['team_agreement_where_both_name_a_team']:.3f}")
    G._dump(P.WORK / "a_labels.json", out)
    return out


def act_pool_y(dev: list, pools: F.FinalPools) -> np.ndarray:
    """``pools.y[T]`` with the SkillCorner rows relabelled. A pool row's label is the truth at its work-grid
    sample; with the standing truth this gives ``pools.y[T]`` back (checked in ``a_check``)."""
    y, off = pools.y[T].copy(), 0
    for r in dev:
        idx = d2.load_pool(r, F.MIX)["idx"]
        if r.is_sc:
            g = load_act(r)
            y[off:off + len(idx)] = g["y"][::g["step"]][idx]
        off += len(idx)
    if off != len(y):
        raise RuntimeError("pool rows do not line up with the matches")
    return y


def act_seqs(dev: list) -> tuple:
    """F3's TCN sequences (213 columns) with the SkillCorner ones relabelled and ``dead_ok``. Returns
    (sequences, finite values lost to float16)."""
    by_key = {r.key: r for r in dev}
    seqs = F.tcn_seqs(dev, T, G.SEED)
    lost = G.add_touch(dev, seqs)
    for (key, w), s in seqs.items():
        r = by_key[key]
        if not r.is_sc:
            continue
        g = load_act(r)
        with np.load(C.paths(r)["feat"](w)) as z:
            idx = z["idx"]
        if len(idx) != len(s["y"]):
            raise RuntimeError(f"{key} {w}: sequence rows do not line up")
        s["y"], s["dead_ok"] = g["y"][::g["step"]][idx], True
    return seqs, lost


# ---------------------------------------------------------------------------------------------
# members

def a_check(dev: list, key: str, n: int) -> dict:
    """No score. (1) A pool row's cached label is the standing truth at its sample, for every match, so
    ``act_pool_y`` changes the SkillCorner rows' labels and nothing else. (2) ``gbdt.fit_heads`` on the standing
    labels gives the incumbent's cached probabilities of one fold byte for byte, so the only difference between
    A's LightGBM member and F3's is the labels and ``dead_ok`` of the SkillCorner rows."""
    pools = F.FinalPools(dev)
    off, same_rows = 0, True
    for r in dev:
        p = d2.load_pool(r, F.MIX)
        g = C.load_gt(r, T)
        same_rows &= bool(np.array_equal(g["y"][::g["step"]][p["idx"]], pools.y[T][off:off + len(p["idx"])]))
        off += len(p["idx"])
    r = next(x for x in dev if x.key == key)
    m = ~np.isin(pools.keys, [key])
    heads = gbdt.fit_heads(pools.X[m], pools.o[m], pools.y[T][m], pools.dead_ok[m], pools.cols(), n_trees=n)
    old, same = Lm.load_oof(INC_G, T, r), []
    for c in F.GATE_CONDS:
        _, X0, X1 = TG.features(r, c)
        pa, ph = G.predict_at(heads, X0, X1, n)
        same.append(pa.tobytes() == old[f"pa_{c}"].tobytes() and ph.tobytes() == old[f"ph_{c}"].tobytes())
    y = act_pool_y(dev, pools)
    sc_rows = np.char.startswith(pools.keys.astype(str), "skillcorner/")
    out = {"fold": key, "trees": n, "pool": list(pools.X.shape), "pool_labels_are_truth_at_idx": same_rows,
           "members_equal_cache": bool(all(same)), "rows_changed_outside_skillcorner": int((y != pools.y[T])[~sc_rows].sum()),
           "skillcorner_rows": int(sc_rows.sum()), "skillcorner_rows_relabelled": int((y != pools.y[T])[sc_rows].sum()),
           "alive_head_rows_before_after": [int((pools.dead_ok & (pools.y[T] < 3)).sum()), int((y < 3).sum())],
           "team_head_rows_before_after": [int((pools.y[T] < 2).sum()), int((y < 2).sum())]}
    log(f"a-check: {out}")
    return out


def a_gbdt(nm: Names, dev: list, folds: list, n: int) -> None:
    todo = [r for r in folds if not Lm.oof_path(nm.g(n), T, r).exists()]
    if not todo:
        return
    pools = F.FinalPools(dev)
    cols, y, dead_ok = pools.cols(), act_pool_y(dev, pools), np.ones(len(pools.o), bool)
    log(f"pool {pools.X.shape}; labels changed on {(y != pools.y[T]).mean():.3f} of rows; LightGBM folds to do: {len(todo)}")
    for r in todo:
        t0 = time.time()
        m = ~np.isin(pools.keys, [r.key])
        heads = gbdt.fit_heads(pools.X[m], pools.o[m], y[m], dead_ok[m], cols, n_trees=n)
        t1 = time.time()
        arrays = {}
        for c in G.worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = G.predict_at(heads, X0, X1, n)
        d3._save(Lm.oof_path(nm.g(n), T, r), {**arrays, "model_sha": np.array(F.gbdt_hash(heads)), "trees": np.array(n)})
        log(f"LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s")


def a_tcn(nm: Names, dev: list, folds: list, steps: int, reverse: bool = False, wait_s: float = 30.0) -> None:
    """Two processes may share the folds (claim files, as ``katena.eval.f3.stage_tcn``): a fold's result does not
    depend on which process trained it."""
    var = nm.t(steps)
    order = list(reversed(folds)) if reverse else list(folds)
    out_of = lambda r: Lm.oof_path(var, T, r)                        # noqa: E731
    claim_of = lambda r: out_of(r).with_suffix(".claim")             # noqa: E731
    if all(out_of(r).exists() for r in order):
        return
    t0 = time.time()
    pools = F.FinalPools(dev)
    n_cols = pools.X.shape[1]
    scs = {r.key: G1.scaler(pools, {r.key}) for r in order if not out_of(r).exists()}
    del pools
    seqs, lost = act_seqs(dev)
    if {s["X"].shape[1] for s in seqs.values()} != {n_cols}:
        raise RuntimeError(f"sequence widths differ from the pool's {n_cols} columns")
    log(f"A TCN: {len(scs)} scalers, {len(seqs)} sequences x {n_cols} columns, "
        f"{sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB, {steps} steps; sequences with a dead-ball "
        f"label: {sum(bool(s['dead_ok']) for s in seqs.values())}; finite values lost to float16: {lost} "
        f"({time.time() - t0:.0f}s)")
    for r in order:
        out, claim = out_of(r), claim_of(r)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                                                 # another process is training this fold
        try:
            t1 = time.time()
            model, sc = G1.fit_tcn(scs[r.key], seqs, {r.key}, G.SEED, E.TCN_THREADS, steps)
            t2 = time.time()
            arrays = G1.predict_tcn(model, sc, r, G.worlds_here(r))
            arrays["model_sha"] = np.array(E.model_hash(model))
            arrays["n_in"] = np.array(model.inp.in_channels)
            d3._save(out, arrays)
            log(f"A TCN fold {r.key}: train {t2 - t1:.0f}s, predict {time.time() - t2:.0f}s")
        finally:
            claim.unlink(missing_ok=True)
    while any(claim_of(r).exists() for r in order):                  # the other process is finishing its last fold
        time.sleep(wait_s)


# ---------------------------------------------------------------------------------------------
# scores and gate

def score_on(r: P.Ref, c: str, g: dict, t: dict, w: float, pri, gt: dict) -> dict:
    """``run_g12.blend_score`` against ``gt``, scored as an IDSSE fold is (predicted dead ball stays dead ball)."""
    pa, ph = E.blend_heads((g[f"pa_{c}"], g[f"ph_{c}"]), (t[f"pa_{c}"], t[f"ph_{c}"]), w)
    f = F._meta(r, c)
    rows = hmm.decode(pa, ph, f["half"], pri, F.TAU, "viterbi")
    return score(gt["y"], to_native_codes(rows, f["idx"], gt["step"], gt["n_native"], gt["blocks"]), sc_mode=False)


def score_all(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    for r in folds:
        if not all(Lm.oof_path(v, T, r).exists() for v in (INC_G, INC_T, nm.g(n))):
            continue
        g0, t0, ga = (Lm.load_oof(v, T, r) for v in (INC_G, INC_T, nm.g(n)))
        ta = Lm.load_oof(nm.t(steps), T, r) if Lm.oof_path(nm.t(steps), T, r).exists() else None
        kk = [Lm.load_oof(v, T, r) for v in (K_G, K_T)] if all(Lm.oof_path(v, T, r).exists() for v in (K_G, K_T)) else None
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        gta = load_act(r) if r.is_sc else None
        for c in G.worlds_here(r):
            gate_c = c in F.GATE_CONDS
            at = {**g0, f"ph_{c}": ga[f"ph_{c}"]}                    # the action team head with F3's alive head
            rows = [("F3", g0, t0, F.W_F), ("A1", ga, t0, F.W_F)]
            if ta is not None:
                rows.append(("A", ga, ta, F.W_F))
            if gate_c:
                rows += [("F3G", g0, t0, 0.0), ("AG", ga, t0, 0.0), ("A1t", at, t0, F.W_F)]
                if ta is not None:
                    rows += [("F3T", g0, t0, 1.0), ("AT", ga, ta, 1.0), ("A2", g0, ta, F.W_F)]
                if kk is not None:
                    rows.append(("K", kk[0], kk[1], F.W_F))
            for name, g, t, w in rows:
                s = G.blend_score(r, c, g, t, w, pri, gt)
                sc[(name, c, r.key)] = s
                if gate_c:
                    sc[(name + "@act", c, r.key)] = score_on(r, c, g, t, w, pri, gta) if r.is_sc else s
    with open(nm.file("a_scores.pkl"), "wb") as fh:
        pickle.dump(sc, fh)
    return sc


def stored_check(sc: dict, name: str, refs: list) -> tuple:
    """(scores of ``name`` equal to the incumbent's stored ones in ``s1_scores.pkl``, scores compared)."""
    p = P.WORK / "s1_scores.pkl"
    if not p.exists():
        return 0, 0
    with open(p, "rb") as fh:
        old = pickle.load(fh)
    pairs = [(old.get(("LT", c, r.key)), sc.get((name, c, r.key))) for r in refs for c in F.GATE_CONDS]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return int(sum(F.same_score(a, b) for a, b in pairs)), len(pairs)


def recall_by_length(sc: dict, name: str, refs: list, cond: str, act: bool) -> list:
    """Share of true possessions matched at IoU 0.5, by true length, pooled over ``refs``."""
    hit, n = np.zeros(len(BINS)), np.zeros(len(BINS))
    for r in refs:
        s = sc.get((name + ("@act" if act else ""), cond, r.key))
        if s is None:
            continue
        g = load_act(r) if act else C.load_gt(r, T)
        ln = _spell_lens(g["y"], g["fps"])
        ok = np.asarray(s["flags"][IOU_HEAD], bool)
        if len(ln) != len(ok):
            raise RuntimeError(f"{r.key}: spells do not line up with the stored flags")
        for i, (lo, hi) in enumerate(BINS):
            m = (ln >= lo) & (ln < hi)
            hit[i] += ok[m].sum()
            n[i] += m.sum()
    return (hit / np.maximum(n, 1)).tolist()


def rule2(g_im_target: dict | None, g_im_severe: dict | None, g_sc_act: dict | None) -> bool:
    """Declared, not standing: F's thresholds on the 8 IDSSE + Metrica folds, whose truth does not change; not
    worse there at severe; not worse than F3 on the SkillCorner folds against the action truth."""
    return bool(g_im_target and g_im_severe and g_sc_act and g_im_target["adopted"] and not g_im_severe["worse"]
                and not g_sc_act["worse"])


def gate(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    sc = score_all(nm, dev, folds, n, steps)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    cand = "A" if any(k[0] == "A" for k in sc) else "A1"
    done = [r for r in folds if get(cand, "target", r) is not None]
    sc_refs, im_refs = [r for r in done if r.is_sc], [r for r in done if not r.is_sc]
    sets = (("all", done), ("SkillCorner", sc_refs), ("IDSSE + Metrica", im_refs))
    registered = not nm.p and cand == "A" and n == N_TREES and steps == STEPS and len(done) == len(dev) == 28
    out = {"candidate": cand, "registered_run": registered, "folds": len(done), "trees": n, "steps": steps,
           "rule1": {}, "rule2": {}, "info": {}}

    print(f"\n## Rule 1 (the standing rule): {cand} vs F3 on the standing truth, {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand, "F3", c)
            if g is not None:
                out["rule1"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs F3, {c}, {tag}", g))
    gt_, gs_, gsc = (out["rule1"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["passes_rule1"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
    run = "registered run" if registered else "NOT the registered run: partial, tagged or without the TCN"
    if gt_ and gs_:
        print(f"\nRULE 1 ({run}): adopted at target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on "
              f"SkillCorner at target = {gsc['worse'] if gsc else 'n/a'} -> {cand} "
              f"{'PASSES' if out['passes_rule1'] else 'fails'} rule 1")

    print(f"\n## Rule 2 (declared; counts only with David's and Justin's names): {cand} vs F3\n" + HEAD)
    for tag, refs, suffix in (("IDSSE + Metrica, one truth", im_refs, ""), ("SkillCorner, action truth", sc_refs, "@act"),
                              ("all, action truth on SkillCorner", done, "@act")):
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand + suffix, "F3" + suffix, c)
            if g is not None:
                out["rule2"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs F3, {c}, {tag}", g))
    out["passes_rule2"] = rule2(out["rule2"].get("IDSSE + Metrica, one truth/target"),
                                out["rule2"].get("IDSSE + Metrica, one truth/severe"),
                                out["rule2"].get("SkillCorner, action truth/target"))
    print(f"\nRULE 2 ({run}): {cand} {'meets' if out['passes_rule2'] else 'does not meet'} rule 2")

    pairs = [("A1", "F3", "LightGBM member swapped, TCN F3's"), ("AG", "F3G", "LightGBM member alone"),
             ("A1t", "F3", "action team head, F3's alive head and TCN")]
    if cand == "A":
        pairs += [("A2", "F3", "TCN member swapped, LightGBM F3's"), ("AT", "F3T", "TCN member alone"), ("A", "A1", "A vs A1")]
    if any(k[0] == "K" for k in sc):
        pairs.append((cand, "K", f"{cand} vs K"))
    print("\n## Information, standing truth\n" + HEAD)
    for a, b, label in pairs:
        for tag, refs in sets:
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, a, b, c)
                if g is not None:
                    out["info"][f"{label}/{tag}/{c}"] = g
                    print(_row(f"{label}, {c}, {tag}", g))
    print("\n## Information: every other evaluation world, standing truth (coupled worlds: IDSSE + Metrica only)\n" + HEAD)
    for wld in G.W.EVAL_WORLDS:
        if wld not in F.GATE_CONDS:
            for tag, refs in (sets if wld in ("mild", "target_ids", "target_smooth") else sets[:1]):
                g = F.compare(get, refs, cand, "F3", wld)
                if g is not None:
                    out["info"][f"world/{wld}/{tag}"] = g
                    print(_row(f"{cand} vs F3, {wld}, {tag}", g))

    print(f"\n## Information: frame-level parts, target\n| folds | truth | model | alive/dead F1 | team accuracy | "
          f"predicted possessions per true one |\n|---|---|---|---|---|---|")
    for tag, refs, suffix, truth in (("IDSSE + Metrica", im_refs, "", "one"), ("SkillCorner", sc_refs, "", "standing"),
                                     ("SkillCorner", sc_refs, "@act", "action")):
        for name in ("F3", cand):
            s = [get(name + suffix, "target", r) for r in refs]
            if s and all(x is not None for x in s):
                v = {"alive_dead_f1": float(np.mean([x["alive_dead_f1"] for x in s])),
                     "team_acc": float(np.mean([x["team_acc"] for x in s])),
                     "per_true": sum(x["n_pred"] for x in s) / max(sum(x["n_gt"] for x in s), 1)}
                out["info"][f"parts/{tag}/{truth}/{name}"] = v
                print(f"| {tag} ({len(refs)}) | {truth} | {name} | {v['alive_dead_f1']:.3f} | {v['team_acc']:.3f} | {v['per_true']:.3f} |")

    print("\n## Information: true possessions found at IoU 0.5 by true length, target\n| folds | truth | model | under 2 s | "
          "2-4 s | 4-8 s | 8-16 s | over 16 s |\n|---|---|---|---|---|---|---|---|")
    for tag, refs, act, truth in (("IDSSE + Metrica", im_refs, False, "one"), ("SkillCorner", sc_refs, False, "standing"),
                                  ("SkillCorner", sc_refs, True, "action")):
        for name in ("F3", cand):
            v = recall_by_length(sc, name, refs, "target", act)
            out["info"][f"recall/{tag}/{truth}/{name}"] = v
            print(f"| {tag} ({len(refs)}) | {truth} | {name} | " + " | ".join(f"{x:.2f}" for x in v) + " |")

    out["checks"] = {"F3_equals_stored_scores": stored_check(sc, "F3", done)}
    print(f"\nchecks: {out['checks']}")
    G._dump(nm.file("a_gate.json"), out)
    return out


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every model file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="LightGBM size (needs --tag)")
    ap.add_argument("--steps", type=int, default=STEPS, help="TCN steps (needs --tag)")
    ap.add_argument("--reverse", action="store_true", help="a-tcn: walk the folds from the other end")
    ap.add_argument("--check-fold", default=F.DET_FOLD)
    a = ap.parse_args()
    if not set(a.stages) <= set(STAGES):
        ap.error(f"unknown stage in {a.stages}")
    if not a.tag and (a.trees != N_TREES or a.steps != STEPS):
        ap.error("the registered sizes can only be changed together with --tag")
    nm = Names(a.tag)
    dev = F.dev_refs()
    folds = G._folds(dev, a.folds)
    log(f"stages {a.stages}; WORK={P.WORK}; development matches {len(dev)}; folds {len(folds)}; tag '{a.tag}'")
    for r in dev:
        if r.is_sc:
            build_act(r)
    for stage in a.stages:
        t0 = time.time()
        if stage == "a-labels":
            a_labels(dev)
        elif stage == "a-check":
            a_check(dev, a.check_fold, a.trees)
        elif stage == "a-gbdt":
            a_gbdt(nm, dev, folds, a.trees)
        elif stage == "a-tcn":
            a_tcn(nm, dev, folds, a.steps, a.reverse)
        elif stage == "a-gate":
            gate(nm, dev, folds, a.trees, a.steps)
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
