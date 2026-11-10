"""Gate G2: candidate F plus set-piece tag columns in its LightGBM member (J5-lite), against F on the 28 external
development folds. Fixed in docs/plan-final-week.md and pre-registered in docs/decision-log.md (2026-11-10,
"Gate G2: set-piece tag features (pre-registration)") before any G2 score was computed.

G2 is F with one change: the LightGBM member reads F's 213 columns plus the 9 ``katena.featurize.tags`` columns
(222), from imitation Smart Tagging (``katena.external.setpieces``) on IDSSE and Metrica and NaN on SkillCorner.
The TCN member, the blend (w = 0.5), the decoder (Viterbi, tau 2, T = 2) and the priors are F's. The TCN member is
not retrained: its out-of-fold probabilities are read from F's cache.

The cache is ``KATENA_WORK`` (its own). F's cache ``f_root`` (``D:/katena/work-final``) is only read:

    copied by ``seed_cache``   gt/, pool_d2j/, touch_pool_d2j/, feat/ and touch/ at target and severe
    read in place              oof/final_tcn/T2/ (F's TCN member), oof/final_gbdt/T2/ (the reproduction checks),
                               final_results.pkl (F's per-fold scores)

and under ``KATENA_WORK``:

    g2_tags/<key>.npz          imitation set pieces and their tag columns, for the pool rows and the evaluation rows
    oof/g2_gbdt/T2/<key>.npz   out-of-fold p_alive / p_home of G2's LightGBM member at target and severe
    g2_results.pkl             scores: G2, G2G (its LightGBM member alone), F_check (F re-decoded here)
    g2_gate.json, final_cost.jsonl

Never reads ``data/`` or ``evaluation/gt/``, never builds or scores a locked bench. Logs hold aggregates of
external matches only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from katena.eval import bakeoff as B
from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import d3
from katena.eval import final as F
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.external import setpieces as SP
from katena.featurize import tags as TAGS
from katena.models import ensemble as E

T = F.T
MIX = F.MIX
VAR = "g2_gbdt"
CONDS = F.GATE_CONDS                         # ("target", "severe"): the gate's conditions only
RES_NAME = "g2_results.pkl"
GATE_NAME = "g2_gate.json"
# tag columns are rebuilt when the code that makes them changes
SRC = hashlib.sha256(Path(TAGS.__file__).read_bytes() + Path(SP.__file__).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# inputs copied from F's cache

def seed_plan(dev: list, f_root: Path) -> list:
    out = []
    for r in dev:
        name = f"{r.dataset}_{r.id}.npz"
        pth = C.paths(r)
        out.append((Path(f_root) / "gt" / name, pth["gt"]))
        out.append((Path(f_root) / f"pool_{MIX}" / name, d2.pool_path(r, MIX)))
        out.append((Path(f_root) / f"touch_pool_{MIX}" / name, F.touch_pool_path(r)))
        for c in CONDS:
            out.append((Path(f_root) / "feat" / f"{r.dataset}_{r.id}__{c}.npz", pth["feat"](c)))
            out.append((Path(f_root) / "touch" / f"{r.dataset}_{r.id}__{c}.npz", TG.tpaths(r)["feat"](c)))
    return out


def seed_cache(dev: list, f_root: Path) -> dict:
    """Copy F's inputs into ``KATENA_WORK`` (never the other way). Development matches only."""
    if Path(f_root).resolve() == P.WORK.resolve():
        raise RuntimeError("the G2 cache must not be F's cache")
    locked = {r.key for r in P.split(P.all_refs())[1]}
    assert not locked & {r.key for r in dev}
    n, nb = 0, 0
    plan = seed_plan(dev, f_root)
    for src, dst in plan:
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        n += 1
        nb += dst.stat().st_size
    return {"planned": len(plan), "copied": n, "gb": round(nb / 2 ** 30, 2)}


# ---------------------------------------------------------------------------------------------
# imitation tags and their columns

def tags_path(ref: P.Ref) -> Path:
    return P.WORK / "g2_tags" / f"{ref.dataset}_{ref.id}.npz"


def _fresh(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return "src" in z.files and str(z["src"]) == SRC


def _pool_rows(ref: P.Ref) -> tuple[np.ndarray, np.ndarray]:
    with np.load(d2.pool_path(ref, MIX)) as z:
        return z["o"], z["idx"]


def build_tags(ref: P.Ref) -> dict:
    """Imitation set pieces of ``ref`` (None on SkillCorner) and their tag columns for the ``d2j`` pool rows (the
    row's own focal orientation) and for the evaluation rows (both focals). Returns aggregate counts."""
    out = tags_path(ref)
    g = C.load_gt(ref, T)
    if not _fresh(out):
        sps, st = SP.imitate(SP.events_for(ref, P.EXTERNAL), ref.id, g["fps"])
        o, idx = _pool_rows(ref)
        pool, names = TAGS.tag_columns(sps, idx, g["step"], g["fps"])
        A = np.where(o[:, None] == 1, pool[1], pool[0]).astype(np.float32)
        with np.load(C.paths(ref)["feat"](CONDS[0])) as z:
            eidx = z["idx"]
        for c in CONDS[1:]:
            with np.load(C.paths(ref)["feat"](c)) as z:
                if not np.array_equal(z["idx"], eidx):
                    raise RuntimeError(f"{ref.key}: evaluation rows differ between {CONDS[0]} and {c}")
        ev, _ = TAGS.tag_columns(sps, eidx, g["step"], g["fps"])
        sp = sps or []
        d3._save(out, {"A": A, "o": o, "idx": idx, "A0": ev[0], "A1": ev[1], "eidx": eidx, "names": np.array(names),
                       "start": np.array([s.start for s in sp], dtype=np.int64),
                       "end": np.array([s.end for s in sp], dtype=np.int64),
                       "kind": np.array([s.kind for s in sp], dtype="<U32"),
                       "team": np.array([-1 if s.team is None else s.team for s in sp], dtype=np.int64),
                       "has_tags": np.array(sps is not None), "stats": np.array(json.dumps(st)),
                       "src": np.array(SRC)})
    with np.load(out) as z:
        st = json.loads(str(z["stats"]))
        inside = z["A0"][:, TAGS.NAMES.index("tag_inside")]
        return {"key": ref.key, "has_tags": bool(z["has_tags"]), "set_pieces": int(len(z["start"])),
                "events": st.get("events", {}), "kept": st.get("kept", {}),
                "pool_rows": int(len(z["A"])), "eval_rows": int(len(z["A0"])),
                "nan_share_pool": round(float(np.isnan(z["A"]).mean()), 4),
                "inside_share_eval": round(float(np.nanmean(inside)), 4) if bool(z["has_tags"]) else None}


def load_tags(ref: P.Ref) -> dict:
    p = tags_path(ref)
    if not _fresh(p):
        raise RuntimeError(f"{ref.key}: tag columns missing or stale (built by other code); run the tags stage")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def features(ref: P.Ref, cond: str) -> tuple[np.ndarray, np.ndarray]:
    """(X0, X1) of G2's LightGBM member: F's 213 columns (``touch_gate.features``) then the 9 tag columns."""
    f, X0, X1 = TG.features(ref, cond)
    t = load_tags(ref)
    if not np.array_equal(t["eidx"], f["idx"]):
        raise RuntimeError(f"{ref.key} {cond}: tag rows do not line up")
    return np.concatenate([X0, t["A0"]], 1), np.concatenate([X1, t["A1"]], 1)


class G2Pools(F.FinalPools):
    """F's LightGBM pool (``final.FinalPools``: the ``d2j`` rows, M2's 160 columns and the 53 touch columns) with
    the 9 tag columns of the same rows appended (orientations and sample indices checked)."""

    def __init__(self, refs: list):
        super().__init__(refs, touch=True)
        parts = []
        for r in refs:
            t = load_tags(r)
            o, idx = _pool_rows(r)
            if not (np.array_equal(t["o"], o) and np.array_equal(t["idx"], idx)):
                raise RuntimeError(f"{r.key}: tag columns do not line up with the d2j pool")
            parts.append(t["A"])
        A = np.concatenate(parts)
        if len(A) != len(self.X):
            raise RuntimeError("tag columns do not line up with the pool")
        self.n_f = self.X.shape[1]
        self.X = np.concatenate([self.X, A.astype(self.X.dtype)], 1)
        self.names = self.names + list(TAGS.NAMES)
        self.groups = np.concatenate([self.groups, np.full(A.shape[1], TAGS.GROUP)])


# ---------------------------------------------------------------------------------------------
# LightGBM member

def f_oof_path(f_root: Path, variant: str, ref: P.Ref) -> Path:
    return Path(f_root) / "oof" / variant / f"T{T}" / f"{ref.dataset}_{ref.id}.npz"


def load_f_oof(f_root: Path, variant: str, ref: P.Ref) -> dict:
    with np.load(f_oof_path(f_root, variant, ref)) as z:
        return {k: z[k] for k in z.files}


def check_fg(pools: G2Pools, ref: P.Ref, f_root: Path) -> dict:
    """The pool here, without its tag columns, must give F's LightGBM member for fold ``ref`` byte for byte: the
    model hash stored with F's out-of-fold file, and its probabilities at target and severe."""
    heads = pools.fit({ref.key}, T, pools.cols(drop=(TAGS.GROUP,)))
    st = load_f_oof(f_root, F.VAR_G, ref)
    out = {"fold": ref.key, "model_equal": F.gbdt_hash(heads) == str(st["model_sha"])}
    for c in CONDS:
        _, X0, X1 = TG.features(ref, c)
        pa, ph = heads.predict(X0, X1)
        out[f"probs_equal_{c}"] = pa.tobytes() == st[f"pa_{c}"].tobytes() and ph.tobytes() == st[f"ph_{c}"].tobytes()
    return out


def stage_gbdt(pools: G2Pools, folds: list) -> None:
    cols = pools.cols()
    for r in folds:
        out = Lm.oof_path(VAR, T, r)
        if out.exists():
            continue
        t0 = time.time()
        heads = pools.fit({r.key}, T, cols)
        t1 = time.time()
        arrays = {}
        for c in CONDS:
            X0, X1 = features(r, c)
            arrays[f"pa_{c}"], arrays[f"ph_{c}"] = heads.predict(X0, X1)
        arrays["model_sha"] = np.array(F.gbdt_hash(heads))
        d3._save(out, arrays)
        print(f"[{time.strftime('%H:%M:%S')}] G2 LightGBM fold {r.key}: fit {t1 - t0:.0f}s, "
              f"predict {time.time() - t1:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------
# decode and score

def results() -> Lm.Results:
    return Lm.Results(P.WORK / RES_NAME)


def stage_score(res: Lm.Results, dev: list, folds: list, f_root: Path, f_store: dict) -> dict:
    """G2 (its LightGBM member blended with F's stored TCN member at w = 0.5), G2G (the member alone) and F_check
    (F's two stored members, re-decoded here, which must equal F's stored scores) at target and severe."""
    src = F.PriorSource(dev)
    chk = {"n": 0, "f_check_equal": 0}
    for r in folds:
        og = Lm.load_oof(VAR, T, r)
        ot = load_f_oof(f_root, F.VAR_T, r)
        of = load_f_oof(f_root, F.VAR_G, r)
        pri = B.priors_for(src, {r.key}, T)
        gt = C.load_gt(r, T)
        for c in CONDS:
            tcn = (ot[f"pa_{c}"], ot[f"ph_{c}"])
            pa, ph = E.blend_heads((og[f"pa_{c}"], og[f"ph_{c}"]), tcn, F.W_F)
            res.put(("G2", T, c, r.key), F.decode_score(r, c, pa, ph, pri, gt))
            res.put(("G2G", T, c, r.key), F.decode_score(r, c, og[f"pa_{c}"], og[f"ph_{c}"], pri, gt))
            pa, ph = E.blend_heads((of[f"pa_{c}"], of[f"ph_{c}"]), tcn, F.W_F)
            s = F.decode_score(r, c, pa, ph, pri, gt)
            res.put(("F_check", T, c, r.key), s)
            chk["n"] += 1
            chk["f_check_equal"] += int(F.same_score(s, f_store.get(("F", T, c, r.key))))
    res.save()
    return chk
