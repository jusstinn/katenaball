#!/usr/bin/env python3
"""K1 and K: a label-kind column in the team heads of the two members. External matches only.

    export KATENA_EXTERNAL=<corpus root> KATENA_WORK=<work>/final     # the cache of scripts/night_g12.sh
    uv run python scripts/run_k.py k1-check                # the team-head code path equals the cache (no score)
    uv run python scripts/run_k.py k1-gbdt k1-gate         # K1 gate, 28 folds
    uv run python scripts/run_k.py k2-train                # the TCN with the column, 28 folds
    uv run python scripts/run_k.py k2-train --reverse      # optional: a second process takes folds from the other end
    uv run python scripts/run_k.py k-gate                  # K gate (both members), 28 folds

Pre-registered in docs/decision-log.md (2026-11-11, "K1, K and B1 (pre-registration)") before any score of either
candidate. Never reads ``data/`` or ``evaluation/gt/``, never trains on or scores a locked bench. LOMO on the 27
other development matches. Every stage caches per fold under ``KATENA_WORK/oof/`` and skips what exists.

The incumbent is F3: the 1,200-tree LightGBM member (cache ``g1_gbdt1200``) blended at w = 0.5 with the 4,800-step
TCN that reads the touch columns (``g2_tcn4800t``). It is read from that cache, not refitted.

The column. ``label_kind`` is 1 on rows of a match whose owner labels come with automatic broadcast tracking and
have no dead-ball label (SkillCorner), 0 on rows of a match whose labels come from corrected tracking or events
and have one (IDSSE, Metrica). It is a property of the dataset, known before any label is read. A held-out match
is predicted with its own dataset's value.

K1. The LightGBM team head reads the column (214 columns). The alive head trains only on kind-0 rows, where the
column is constant, so it is not refitted: its cached probabilities are used as they are.

K.  K1 plus the 4,800-step touch TCN with the column as a 214th input channel. Its alive output is always read
with the column at 0 (no kind-1 row carries an alive target), its team output with the match's own value.

``phx`` (information only): the team probabilities read with the other value of the column.

``--tag`` keeps a smoke run's files apart; the registered sizes can be changed only with a tag.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g12 as G  # noqa: E402  (pins the CPU environment and loads libomp before torch and LightGBM)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import d3  # noqa: E402
from katena.eval import final as F  # noqa: E402
from katena.eval import g1 as G1  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import touch_gate as TG  # noqa: E402
from katena.models import ensemble as E  # noqa: E402
from katena.models import gbdt  # noqa: E402

T = P.T_MAIN
N_TREES = G.N_CAND                          # 1,200
STEPS = G.G2_STEPS                          # 4,800
INC_G, INC_T = "g1_gbdt1200", "g2_tcn4800t"  # F3's members (run_g12.py)
F_G, F_T = "g1_gbdt150", "g1_tcn"           # F, for information
SC_PREFIX = "skillcorner/"
STAGES = ("k1-check", "k1-gbdt", "k1-gate", "k2-train", "k-gate")
log, HEAD, _row = G.log, G.HEAD, G._row


class Names:
    def __init__(self, tag: str = ""):
        self.p = f"{tag}_" if tag else ""

    def k1(self, n: int) -> str:
        return f"{self.p}k1_gbdt{n}k"

    def k2(self, steps: int) -> str:
        return f"{self.p}k2_tcn{steps}tk"

    def file(self, name: str) -> Path:
        return P.WORK / f"{self.p}{name}"


def kind_of(r: P.Ref) -> float:
    return 1.0 if r.is_sc else 0.0


def kind_rows(keys: np.ndarray) -> np.ndarray:
    return np.char.startswith(keys.astype(str), SC_PREFIX)


def with_kind(X: np.ndarray, k: float) -> np.ndarray:
    return np.concatenate([X, np.full((len(X), 1), k, dtype=X.dtype)], 1)


def kind_pool(dev: list) -> SimpleNamespace:
    """F's pool with the label-kind column appended: ``keys``, ``X`` (214 columns), ``o``, ``y``."""
    pools = F.FinalPools(dev)
    k = kind_rows(pools.keys).astype(pools.X.dtype)
    return SimpleNamespace(keys=pools.keys, o=pools.o, y=pools.y[T], n_base=pools.X.shape[1],
                           X=np.concatenate([pools.X, k[:, None]], 1))


# ---------------------------------------------------------------------------------------------
# K1

def fit_team(X: np.ndarray, o: np.ndarray, y: np.ndarray, n_trees: int):
    """``gbdt.fit_heads``'s team head alone: the same rows, target, parameters and number of trees."""
    rows = np.flatnonzero((y == 0) | (y == 1))
    t = (y[rows] == o[rows]).astype(np.float32)
    return lgb.train(gbdt.PARAMS, lgb.Dataset(X[rows], t, free_raw_data=True), n_trees)


def predict_team(team, X0: np.ndarray, X1: np.ndarray, n: int) -> np.ndarray:
    return (0.5 * (team.predict(X0, num_iteration=n) + 1.0 - team.predict(X1, num_iteration=n))).astype(np.float32)


def k1_check(dev: list, key: str, n: int) -> dict:
    """No score. ``fit_team`` on F's 213 columns gives the incumbent's cached team probabilities byte for byte, so
    the only difference between K1 and the incumbent is the column."""
    r = next(x for x in dev if x.key == key)
    kp = kind_pool(dev)
    m = ~np.isin(kp.keys, [key])
    team = fit_team(kp.X[m][:, :kp.n_base], kp.o[m], kp.y[m], n)
    old = Lm.load_oof(INC_G, T, r)
    same = []
    for c in F.GATE_CONDS:
        _, X0, X1 = TG.features(r, c)
        same.append(predict_team(team, X0, X1, n).tobytes() == old[f"ph_{c}"].tobytes())
    out = {"fold": key, "trees": n, "pool": list(kp.X.shape), "kind_1_share_of_rows": float(kind_rows(kp.keys).mean()),
           "team_head_equals_cache": bool(all(same))}
    log(f"k1-check: {out}")
    return out


def k1_gbdt(nm: Names, dev: list, folds: list, n: int) -> None:
    todo = [r for r in folds if not Lm.oof_path(nm.k1(n), T, r).exists()]
    if not todo:
        return
    kp = kind_pool(dev)
    log(f"pool {kp.X.shape}; kind-1 share of rows {kind_rows(kp.keys).mean():.3f}; LightGBM folds to do: {len(todo)}")
    for r in todo:
        t0 = time.time()
        m = ~np.isin(kp.keys, [r.key])
        team = fit_team(kp.X[m], kp.o[m], kp.y[m], n)
        t1 = time.time()
        old, k = Lm.load_oof(INC_G, T, r), kind_of(r)
        arrays = {}
        for c in G.worlds_here(r):
            _, X0, X1 = TG.features(r, c)
            arrays[f"pa_{c}"] = old[f"pa_{c}"]                      # the alive head is not refitted
            arrays[f"ph_{c}"] = predict_team(team, with_kind(X0, k), with_kind(X1, k), n)
            if c in F.GATE_CONDS:
                arrays[f"phx_{c}"] = predict_team(team, with_kind(X0, 1.0 - k), with_kind(X1, 1.0 - k), n)
        gain = team.feature_importance("gain")
        d3._save(Lm.oof_path(nm.k1(n), T, r), {
            **arrays, "model_sha": np.array(hashlib.sha256(team.model_to_string().encode()).hexdigest()),
            "trees": np.array(n), "kind": np.array(k), "kind_gain_share": np.array(gain[-1] / max(gain.sum(), 1e-12))})
        log(f"LightGBM fold {r.key}: fit {t1 - t0:.0f}s, predict {time.time() - t1:.0f}s, "
            f"share of split gain on the column {gain[-1] / max(gain.sum(), 1e-12):.4f}")


# ---------------------------------------------------------------------------------------------
# K2: the TCN member with the column

def predict_k2(model, sc, r: P.Ref, conds) -> dict:
    from katena.models import tcn
    E.set_deterministic(E.TCN_THREADS)
    k, out = kind_of(r), {}
    for c in conds:
        f, X0, X1 = TG.features(r, c)
        run = lambda v: tcn.predict(model, sc(with_kind(X0, v)), sc(with_kind(X1, v)), f["half"])  # noqa: E731
        pa0, ph0 = run(0.0)
        ph1 = run(1.0)[1] if (k == 1.0 or c in F.GATE_CONDS) else None
        out[f"pa_{c}"] = pa0                                         # alive: always read at kind 0
        out[f"ph_{c}"] = ph1 if k == 1.0 else ph0
        if c in F.GATE_CONDS:
            out[f"phx_{c}"] = ph0 if k == 1.0 else ph1
    return out


def k2_train(nm: Names, dev: list, folds: list, steps: int, reverse: bool = False, wait_s: float = 30.0) -> None:
    """Two processes may share the folds (``katena.eval.f3.stage_tcn``'s claim files): a fold's result does not
    depend on which process trained it."""
    var = nm.k2(steps)
    order = list(reversed(folds)) if reverse else list(folds)
    out_of = lambda r: Lm.oof_path(var, T, r)                        # noqa: E731
    claim_of = lambda r: out_of(r).with_suffix(".claim")             # noqa: E731
    if all(out_of(r).exists() for r in order):
        return
    t0 = time.time()
    kp = kind_pool(dev)
    n_cols = kp.X.shape[1]
    scs = {r.key: G1.scaler(kp, {r.key}) for r in order if not out_of(r).exists()}
    del kp
    seqs = F.tcn_seqs(dev, T, G.SEED)
    lost = G.add_touch(dev, seqs)
    for (key, _), s in seqs.items():
        s["X"] = np.concatenate([s["X"], np.full((len(s["X"]), 1), float(key.startswith(SC_PREFIX)), np.float16)], 1)
    if {s["X"].shape[1] for s in seqs.values()} != {n_cols}:
        raise RuntimeError(f"sequence widths differ from the pool's {n_cols} columns")
    log(f"K2: {len(scs)} scalers, {len(seqs)} sequences x {n_cols} columns, "
        f"{sum(s['X'].nbytes for s in seqs.values()) / 2 ** 30:.2f} GB, {steps} steps; finite values lost to "
        f"float16: {lost} ({time.time() - t0:.0f}s)")
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
            arrays = predict_k2(model, sc, r, G.worlds_here(r))
            arrays["model_sha"] = np.array(E.model_hash(model))
            arrays["n_in"] = np.array(model.inp.in_channels)
            arrays["kind"] = np.array(kind_of(r))
            d3._save(out, arrays)
            log(f"K2 fold {r.key}: train {t2 - t1:.0f}s, predict {time.time() - t2:.0f}s")
        finally:
            claim.unlink(missing_ok=True)
    while any(claim_of(r).exists() for r in order):                  # the other process is finishing its last fold
        time.sleep(wait_s)


# ---------------------------------------------------------------------------------------------
# gates

def _other(o: dict) -> dict:
    """The same probabilities with the team head read at the other value of the column (gate worlds only)."""
    return {**o, **{f"ph_{c}": o[f"phx_{c}"] for c in F.GATE_CONDS}}


def score_all(nm: Names, dev: list, folds: list, n: int, steps: int) -> dict:
    src = F.PriorSource(dev)
    sc = {}
    for r in folds:
        if not all(Lm.oof_path(v, T, r).exists() for v in (INC_G, INC_T, F_G, F_T, nm.k1(n))):
            continue
        g0, t0, fg, ft, g1 = (Lm.load_oof(v, T, r) for v in (INC_G, INC_T, F_G, F_T, nm.k1(n)))
        t1 = Lm.load_oof(nm.k2(steps), T, r) if Lm.oof_path(nm.k2(steps), T, r).exists() else None
        pri, gt = B.priors_for(src, {r.key}, T), C.load_gt(r, T)
        for c in G.worlds_here(r):
            rows = [("F3", g0, t0, F.W_F), ("K1", g1, t0, F.W_F)]
            if t1 is not None:
                rows.append(("K", g1, t1, F.W_F))
            if c in F.GATE_CONDS:
                rows += [("F3G", g0, t0, 0.0), ("K1G", g1, t0, 0.0), ("K1x", _other(g1), t0, F.W_F), ("F", fg, ft, F.W_F)]
                if t1 is not None:
                    rows += [("K2", g0, t1, F.W_F), ("F3T", g0, t0, 1.0), ("K2T", g0, t1, 1.0),
                             ("Kx", _other(g1), _other(t1), F.W_F)]
            for name, g, t, w in rows:
                sc[(name, c, r.key)] = G.blend_score(r, c, g, t, w, pri, gt)
    with open(nm.file("k_scores.pkl"), "wb") as fh:
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


def gate(nm: Names, dev: list, folds: list, n: int, steps: int, cand: str) -> dict:
    """``cand`` ("K1" or "K") against F3 under F's rule, with the SkillCorner guard of ``run_g12.g1_decision``."""
    sc = score_all(nm, dev, folds, n, steps)
    get = lambda name, cond, r: sc.get((name, cond, r.key))  # noqa: E731
    done = [r for r in folds if get(cand, "target", r) is not None]
    sets = (("all", done), ("SkillCorner", [r for r in done if r.is_sc]), ("IDSSE + Metrica", [r for r in done if not r.is_sc]))
    registered = not nm.p and n == N_TREES and steps == STEPS and len(done) == len(dev) == 28
    out = {"candidate": cand, "registered_run": registered, "folds": len(done), "trees": n, "steps": steps,
           "gate": {}, "info": {}}
    print(f"\n## {cand} gate: {cand} vs F3, {len(done)} of {len(dev)} folds\n" + HEAD)
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            g = F.compare(get, refs, cand, "F3", c)
            if g is not None:
                out["gate"][f"{tag}/{c}"] = g
                print(_row(f"{cand} vs F3, {c}, {tag}", g))
    gt_, gs_, gsc = (out["gate"].get(k) for k in ("all/target", "all/severe", "SkillCorner/target"))
    out["passes"] = bool(gt_ and gs_ and G.g1_decision(gt_, gs_, gsc))
    if gt_ and gs_:
        print(f"\nDECISION ({'registered run' if registered else 'NOT the registered run: partial or tagged'}): adopted at "
              f"target = {gt_['adopted']}, worse at severe = {gs_['worse']}, worse on SkillCorner at target = "
              f"{gsc['worse'] if gsc else 'n/a'} -> {cand + ' PASSES' if out['passes'] else cand + ' fails'}")
    pairs = [("K1G", "F3G", "LightGBM member alone"), ("K1x", "F3", "K1 read with the other label kind, vs F3"),
             ("K1x", "K1", "K1 read with the other label kind, vs K1"), ("K1", "F", "K1 vs F")]
    if cand == "K":
        pairs = [("K", "K1", "K vs K1"), ("K2", "F3", "the column in the TCN only, vs F3"),
                 ("K2T", "F3T", "TCN member alone"), ("Kx", "F3", "K read with the other label kind, vs F3"),
                 ("Kx", "K", "K read with the other label kind, vs K"), ("K", "F", "K vs F")] + pairs[:1]
    print("\n## Information\n" + HEAD)
    for a, b, label in pairs:
        for tag, refs in sets:
            for c in F.GATE_CONDS:
                g = F.compare(get, refs, a, b, c)
                if g is not None:
                    out["info"][f"{label}/{tag}/{c}"] = g
                    print(_row(f"{label}, {c}, {tag}", g))
    print("\n## Information: every other evaluation world (coupled worlds: IDSSE + Metrica only)\n" + HEAD)
    for wld in G.W.EVAL_WORLDS:
        if wld not in F.GATE_CONDS:
            g = F.compare(get, done, cand, "F3", wld)
            if g is not None:
                out["info"][f"world/{wld}"] = g
                print(_row(f"{cand} vs F3, {wld}", g))
    print(f"\n## Information: predicted possessions per true one\n| cond | folds | F3 | {cand} |\n|---|---|---|---|")
    for tag, refs in sets:
        for c in F.GATE_CONDS:
            if refs:
                v = {k: G.per_true(get, refs, k, c) for k in ("F3", cand)}
                out["info"][f"per_true/{tag}/{c}"] = v
                print(f"| {c} | {tag} ({len(refs)}) | {v['F3']:.3f} | {v[cand]:.3f} |")
    shares = [float(Lm.load_oof(nm.k1(n), T, r)["kind_gain_share"]) for r in done]
    out["info"]["kind_gain_share"] = [min(shares), max(shares)] if shares else []
    print(f"\nshare of the LightGBM team head's split gain on the column, per fold: "
          f"{min(shares):.4f} to {max(shares):.4f}" if shares else "")
    out["checks"] = {"F3_equals_stored_scores": stored_check(sc, "F3", done)}
    print(f"checks: {out['checks']}")
    G._dump(nm.file(f"{cand.lower()}_gate.json"), out)
    return out


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help=", ".join(STAGES) + " (run in the order given)")
    ap.add_argument("--folds", default="all", help="all (28), 8 (IDSSE + Metrica), or match keys")
    ap.add_argument("--tag", default="", help="smoke runs: keeps every file apart from the registered run")
    ap.add_argument("--trees", type=int, default=N_TREES, help="K1 size (needs --tag)")
    ap.add_argument("--steps", type=int, default=STEPS, help="K2 steps (needs --tag)")
    ap.add_argument("--reverse", action="store_true", help="k2-train: walk the folds from the other end")
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
    for stage in a.stages:
        t0 = time.time()
        if stage == "k1-check":
            k1_check(dev, a.check_fold, a.trees)
        elif stage == "k1-gbdt":
            k1_gbdt(nm, dev, folds, a.trees)
        elif stage == "k1-gate":
            gate(nm, dev, folds, a.trees, a.steps, "K1")
        elif stage == "k2-train":
            k2_train(nm, dev, folds, a.steps, a.reverse)
        elif stage == "k-gate":
            gate(nm, dev, folds, a.trees, a.steps, "K")
        log(f"{stage}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
