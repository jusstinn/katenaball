"""J2: the final pipeline. Train candidate F once on all 30 external matches, then label the real match.

F is defined in ``katena.eval.final`` (docs/decision-log.md, "Final candidate F"): M2's LightGBM heads on M2's
160 columns plus D1's 53 touch columns, trained on D2's ``d2j`` world mix; the M5 TCN on M2's columns, same
mix; the two blended at w = 0.5 in logit space; Viterbi at tau = 2 with priors from the external labels.
Nothing here is chosen on the real match: every constant comes from ``katena.eval.final`` and its imports.

Stages, each cached under ``KATENA_WORK`` (``scripts/make_submission.py`` sets it to ``work/final``):

    corpus   ground truth, base features and pools of every external match: copied from a finished
             bake-off cache when one is given (byte-identical inputs), otherwise built
    worlds   the ``d2j`` worlds' features, their touch columns, and the ``d2j`` pools with touch columns
    train    the LightGBM member, the TCN member and the priors, saved to ``model/`` with sha256 hashes
    predict  the real match from ``data/``: features at 5 Hz from the real ATD, touch columns from the real
             25 Hz ball, blend, Viterbi, one label per video frame, the submission file, validation

Training reads external matches only. ``predict`` is the only stage that opens ``data/`` and it reports
aggregates only (counts, shares, shapes, hashes, run times), never a frame-level value (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import d3
from katena.eval import final as F
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.featurize import touch as TF

T = F.T
TAU = F.TAU
W_TCN = F.W_F
SEED = 0
MODEL_DIR = "model"
BUNDLE = "bundle.pkl"
MANIFEST = "manifest.json"


def model_dir() -> Path:
    return P.WORK / MODEL_DIR


def training_refs() -> list:
    """Every external match: the 28 development matches and the two benches. The benches were scored once,
    for a sanity read, and choose nothing; the final model may train on them (PLAN.md, D5)."""
    refs = P.all_refs()
    if not refs:
        raise RuntimeError(f"no external matches under {P.CANON}; run scripts/build_external.py first")
    return refs


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# corpus

def _corpus_files(ref: P.Ref) -> list:
    pth = C.paths(ref)
    return [pth["gt"], pth["pool"]] + [pth["feat"](c) for c in C.POOL_CONDS]


def stage_corpus(refs: list, seed_from: Path | None = None, workers: int = 1) -> dict:
    """Ground truth, base pool and mild/target/severe features of every match. Files from a bake-off cache
    are copied (they are what ``corpus.build_one`` writes); anything missing is built here, in ``workers``
    processes. Each match's files depend only on that match, so the worker count does not change them."""
    if seed_from is not None and Path(seed_from).resolve() == P.WORK.resolve():
        raise RuntimeError("the final cache must not be the bake-off cache")
    st = {"copied": 0, "built": 0, "present": 0}
    build = []
    for r in refs:
        files = _corpus_files(r)
        if all(f.exists() for f in files):
            st["present"] += 1
            continue
        rel = [f.relative_to(P.WORK) for f in files]
        if seed_from is not None and all((Path(seed_from) / x).exists() for x in rel):
            for x in rel:
                dst = P.WORK / x
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
                shutil.copyfile(Path(seed_from) / x, tmp)
                tmp.replace(dst)
            st["copied"] += 1
        else:
            build.append(r)
    if workers <= 1 or len(build) <= 1:
        for r in build:
            C.build_one(r)
            log(f"corpus built {r.key}")
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for key in pool.imap_unordered(C.build_one, build):
                log(f"corpus built {key}")
    st["built"] = len(build)
    return st


# ---------------------------------------------------------------------------------------------
# worlds: d2j features, touch columns, pools

def build_worlds(ref: P.Ref) -> dict:
    """Features and touch columns of every ``d2j`` training world of ``ref``, then its ``d2j`` pool and the
    touch columns of the same rows. Each world's 5 Hz corruption is re-run and checked against its feature
    file before touch columns are computed from the lifted native ball (as ``final.build_touch`` does)."""
    from katena.grid import downsample
    from katena.types import load_match
    t0 = time.time()
    worlds = F.train_worlds(ref.dataset)
    d2.build_ref(ref, [w for w in worlds if w not in C.CONFIG])
    todo = [w for w in worlds if not TG._fresh(TG.tpaths(ref)["feat"](w))]
    if todo:
        m = load_match(ref.path)
        m5 = downsample(m)
        for w in todo:
            mc, lifted = TG.corrupted(ref, m, m5, w)
            if not F.recovery_ok(ref, w, mc):
                raise RuntimeError(f"{ref.key} {w}: the re-run corruption does not match the cached features")
            X, names, idx = TF.touch_features(mc, TG._ball("touch", m, m5, mc, lifted))
            out = TG.tpaths(ref)["feat"](w)
            out.parent.mkdir(parents=True, exist_ok=True)
            d3._save(out, {"A0": X[0], "A1": X[1], "idx": idx, "names": np.array(names), "src": np.array(TG.SRC)})
    d2.build_pool(ref, F.MIX)
    F.build_touch_pool(ref)
    return {"key": ref.key, "worlds": len(worlds), "touch_built": len(todo), "wall_s": round(time.time() - t0, 1)}


def _worlds_job(ref: P.Ref) -> dict:
    return build_worlds(ref)


def stage_worlds(refs: list, workers: int = 1) -> list:
    out = []
    if workers <= 1:
        for r in refs:
            out.append(build_worlds(r))
            log(f"worlds {out[-1]}")
        return out
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        for st in pool.imap_unordered(_worlds_job, refs):
            out.append(st)
            log(f"worlds {st}")
    return out


# ---------------------------------------------------------------------------------------------
# train

@dataclass
class Model:
    """Everything ``predict`` needs, in plain types: LightGBM model strings, the TCN's state dict as numpy
    arrays, the TCN scaler, the decoder priors and the feature names both members expect."""
    gbdt_alive: str
    gbdt_team: str
    gbdt_cols: np.ndarray
    names: list            # M2 columns, then touch columns (LightGBM member)
    n_base: int            # the TCN reads the first n_base columns
    tcn_state: dict
    tcn_n_in: int
    scaler_mu: np.ndarray
    scaler_sd: np.ndarray
    priors: object
    touch_src: str

    def heads(self):
        import lightgbm as lgb

        from katena.models.gbdt import Heads
        return Heads(alive=lgb.Booster(model_str=self.gbdt_alive), team=lgb.Booster(model_str=self.gbdt_team),
                     cols=self.gbdt_cols)

    def tcn(self):
        import torch

        from katena.models import tcn
        net = tcn.TCN(self.tcn_n_in)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.tcn_state.items()})
        net.eval()
        sc = tcn.Scaler.__new__(tcn.Scaler)
        sc.mu, sc.sd = self.scaler_mu, self.scaler_sd
        return net, sc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_train(refs: list, threads: int | None = None, steps: int | None = None) -> dict:
    """Fit both members on every row of every match (nothing held out) and save ``model/bundle.pkl``."""
    from katena.eval import bakeoff as B
    from katena.models import ensemble as E
    threads = threads or E.TCN_THREADS
    E.set_deterministic(threads)
    t0 = time.time()
    pools = F.FinalPools(refs)
    log(f"pool: {pools.X.shape[0]} rows x {pools.X.shape[1]} columns ({pools.n_base} base)")
    heads = pools.fit(set(), T, pools.cols())
    t1 = time.time()
    log(f"LightGBM member: {t1 - t0:.0f}s")
    seqs = F.tcn_seqs(refs, T, SEED)
    net, sc = F.fit_tcn(pools, seqs, set(), SEED, threads, steps)
    del seqs
    t2 = time.time()
    log(f"TCN member: {t2 - t1:.0f}s")
    pri = B.priors_for(F.PriorSource(refs), set(), T)
    model = Model(gbdt_alive=heads.alive.model_to_string(), gbdt_team=heads.team.model_to_string(),
                  gbdt_cols=np.asarray(heads.cols), names=list(pools.names), n_base=int(pools.n_base),
                  tcn_state={k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()},
                  tcn_n_in=int(pools.n_base), scaler_mu=np.asarray(sc.mu), scaler_sd=np.asarray(sc.sd),
                  priors=pri, touch_src=TG.SRC)
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (BUNDLE + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(model, fh, protocol=4)
    tmp.replace(out / BUNDLE)
    manifest = {
        "matches": [r.key for r in refs], "rows": int(pools.X.shape[0]), "columns": int(pools.X.shape[1]),
        "mix": F.MIX, "T": T, "tau": TAU, "w_tcn": W_TCN, "seed": SEED, "tcn_threads": threads,
        "sha256": {"gbdt": F.gbdt_hash(heads), "tcn": E.model_hash(net), BUNDLE: sha256_file(out / BUNDLE)},
        "seconds": {"gbdt": round(t1 - t0, 1), "tcn": round(t2 - t1, 1)},
    }
    (out / MANIFEST).write_bytes((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return manifest


def load_model() -> Model:
    with open(model_dir() / BUNDLE, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------------------------
# predict

def submission_bytes(labels) -> bytes:
    """'frame,label' and one LF-terminated row per frame, built as bytes so no OS turns LF into CRLF."""
    return ("frame,label\n" + "".join(f"{i},{v}\n" for i, v in enumerate(labels))).encode("ascii")


def label_match(model: Model, m, threads: int | None = None) -> tuple[np.ndarray, dict]:
    """int8 codes (0 H, 1 A, 2 D, 3 X) for every native frame of ``m``, and an aggregate-only report.

    ``m`` is a native-rate Match without labels. It is not corrupted: the real ATD is the noisy input
    the training worlds stand in for."""
    from katena.decode import hmm
    from katena.eval.score import to_native_codes
    from katena.featurize.build import featurize, infer_dirs
    from katena.grid import downsample, mask_outside_blocks
    from katena.models import ensemble as E
    from katena.models import tcn
    threads = threads or E.TCN_THREADS
    if model.touch_src != TG.SRC:
        raise RuntimeError("the model was trained with other touch-feature code; retrain")
    t0 = time.time()
    m5 = downsample(m)
    dirs = infer_dirs(m5)
    f0, f1 = featurize(m5, 0, dirs), featurize(m5, 1, dirs)
    A, tnames, tidx = TF.touch_features(m5, TF.native_ball(m))
    if not np.array_equal(tidx, f0.idx):
        raise RuntimeError("touch rows do not line up with the base features")
    names = list(f0.names) + list(tnames)
    if names != model.names:
        raise RuntimeError("feature columns differ from the trained model's")
    X0, X1 = np.concatenate([f0.X, A[0]], 1), np.concatenate([f1.X, A[1]], 1)
    t1 = time.time()
    pa_g, ph_g = model.heads().predict(X0, X1)
    E.set_deterministic(threads)
    net, sc = model.tcn()
    import torch
    with torch.no_grad():
        pa_t, ph_t = tcn.predict(net, sc(f0.X), sc(f1.X), f0.half)
    pa, ph = E.blend_heads((pa_g, ph_g), (pa_t, ph_t), W_TCN)
    rows = hmm.decode(pa, ph, f0.half, model.priors, TAU, "viterbi")
    step = int(m5.meta["step"])
    codes = mask_outside_blocks(to_native_codes(rows, f0.idx, step, m.n, blocks=m.half_blocks), m)
    inside = np.zeros(m.n, dtype=bool)
    for _, a, b in m.half_blocks:
        inside[a:b + 1] = True
    edges = np.flatnonzero(np.diff(rows.astype(np.int16)) != 0)
    report = {
        "frames": int(m.n), "frames_in_halves": int(inside.sum()), "samples_5hz": int(len(rows)),
        "features": [int(X0.shape[0]), int(X0.shape[1])],
        "ball_missing_in_halves": round(float(np.isnan(m.ball[inside, 0]).mean()), 4),
        "label_share_in_halves": {k: round(float((codes[inside] == i).mean()), 4) for i, k in enumerate("HADX")},
        "spells_5hz": int(len(edges) + len(np.unique(f0.half))),
        "touch_events": TF.count_touches(m5, TF.native_ball(m)),
        "seconds": {"features": round(t1 - t0, 1), "models_and_decode": round(time.time() - t1, 1)},
    }
    return codes, report


def write_submission(codes: np.ndarray, path: Path, n_expected: int) -> dict:
    """Write the file, then run the official checks on the bytes written. Raises if they fail."""
    root = P.REPO / "evaluation"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import possession_metrics as pm

    from katena.types import LABEL_NAMES
    data = submission_bytes(np.array(LABEL_NAMES)[codes])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    reason = pm.validate_submission(pm.read_label_csv(path), n_expected)
    if reason:
        raise RuntimeError(f"the submission fails the official checks: {reason}")
    return {"path": str(path), "rows": int(len(codes)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def stage_predict(out: Path, data_dir: Path | None = None, threads: int | None = None) -> dict:
    """Label the real match and write the submission. Prints aggregates only."""
    from katena.target.atd import load_target
    m = load_target(data_dir)
    rep = m.meta.get("read_report", {})
    model = load_model()
    codes, report = label_match(model, m, threads)
    sub = write_submission(codes, out, m.n)
    report["read"] = {k: rep.get(k) for k in ("n_lines", "n_malformed", "n_out_of_range", "n_duplicate",
                                              "n_frames_present", "contiguous_from_one")}
    report["submission"] = sub
    report["model_sha256"] = sha256_file(model_dir() / BUNDLE)
    return report


def env_report() -> dict:
    import platform

    import lightgbm
    import torch
    return {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "lightgbm": lightgbm.__version__, "torch": torch.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random")}
