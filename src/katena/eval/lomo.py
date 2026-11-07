"""Leave-one-match-out machinery: pooled training rows, head fitting per fold, out-of-fold probabilities,
decoding, scoring and result storage."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from katena.decode import hmm
from katena.eval import corpus as C
from katena.eval import protocol as P
from katena.eval.score import score, to_native_codes
from katena.models import gbdt
from katena.models.rules import m1_predict

OOF_DIR = P.WORK / "oof"
RES_PATH = P.WORK / "results.pkl"


class Pools:
    """All development-match training rows in memory (about 0.4 GB)."""

    def __init__(self, refs: list):
        parts = [C.load_pool(r) for r in refs]
        self.refs = refs
        self.keys = np.concatenate([[r.key] * len(p["o"]) for r, p in zip(refs, parts)])
        self.X = np.concatenate([p["X"] for p in parts])
        self.o = np.concatenate([p["o"] for p in parts])
        self.y = {T: np.concatenate([p[f"y_T{T}"] for p in parts]) for T in P.T_SWEEP}
        self.dead_ok = np.concatenate([np.full(len(p["o"]), r.dataset != "skillcorner") for r, p in zip(refs, parts)])
        self.names = list(parts[0]["names"])
        self.groups = parts[0]["groups"]

    def cols(self, drop=(), keep=None) -> np.ndarray:
        ok = np.ones(len(self.groups), bool) if keep is None else np.isin(self.groups, keep)
        return np.flatnonzero(ok & ~np.isin(self.groups, drop))

    def fit(self, exclude: set, T: int, cols: np.ndarray, only_keys: set | None = None) -> gbdt.Heads:
        m = ~np.isin(self.keys, list(exclude))
        if only_keys is not None:
            m &= np.isin(self.keys, list(only_keys))
        return gbdt.fit_heads(self.X[m], self.o[m], self.y[T][m], self.dead_ok[m], cols)

    def label_seqs(self, exclude: set, T: int) -> list:
        """Contiguous work-grid label arrays (IDSSE and Metrica only; SkillCorner has no D and
        its X-broken runs would bias duration priors)."""
        out = []
        for r in self.refs:
            if r.key in exclude or r.is_sc:
                continue
            g = C.load_gt(r, T)
            out.append(g["y"][::g["step"]])
        return out


def oof_path(variant: str, T: int, ref: P.Ref) -> Path:
    return OOF_DIR / variant / f"T{T}" / f"{ref.dataset}_{ref.id}.npz"


def run_fold(pools: Pools, variant: str, T: int, ref: P.Ref, cols: np.ndarray, exclude: set,
             conds: list | None = None) -> None:
    """Train on every pooled match not in ``exclude`` and store p_alive / p_home for ``ref``."""
    out = oof_path(variant, T, ref)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    heads = pools.fit(exclude, T, cols)
    arrays = {}
    for c in (conds or C.conds_for(ref)):
        f = C.load_feat(ref, c)
        pa, ph = heads.predict(f["X0"], f["X1"])
        arrays[f"pa_{c}"], arrays[f"ph_{c}"] = pa, ph
    np.savez(out, **arrays)


def load_oof(variant: str, T: int, ref: P.Ref) -> dict:
    with np.load(oof_path(variant, T, ref)) as z:
        return {k: z[k] for k in z.files}


# ---- decode and score ------------------------------------------------------------------------

def decode_rows(oof: dict, cond: str, f: dict, pri: hmm.Priors, tau: float, kind: str) -> np.ndarray:
    return hmm.decode(oof[f"pa_{cond}"], oof[f"ph_{cond}"], f["half"], pri, tau, kind)


def score_rows(ref: P.Ref, T: int, cond: str, rows: np.ndarray, idx: np.ndarray, gt=None) -> dict:
    g = gt or C.load_gt(ref, T)
    pred = to_native_codes(rows, idx, g["step"], g["n_native"])
    return score(g["y"], pred, sc_mode=ref.is_sc)


def m0_rows(f: dict) -> np.ndarray:
    return f["m0"][f["idx"]]


def m1_rows(f: dict) -> np.ndarray:
    return m1_predict(f["X0"], list(f["names"]), f["half"])


class Results:
    """Persistent {(model, T, cond, match_key): metrics} store."""

    def __init__(self, path: Path = RES_PATH):
        self.path = path
        self.d = pickle.load(open(path, "rb")) if path.exists() else {}

    def put(self, key: tuple, val: dict) -> None:
        self.d[key] = val

    def get(self, key: tuple):
        return self.d.get(key)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        pickle.dump(self.d, open(tmp, "wb"))
        tmp.replace(self.path)


def timed(label: str, t0: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {label} ({time.time() - t0:.0f}s)", flush=True)
