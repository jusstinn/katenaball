"""J2: the final pipeline. Train the submission model once on all 30 external matches, then label the real match.

F is defined in ``katena.eval.final`` (docs/decision-log.md, "Final candidate F"): M2's LightGBM heads on M2's
160 columns plus D1's 53 touch columns, trained on D2's ``d2j`` world mix; the M5 TCN on M2's columns, same
mix; the two blended at w = 0.5 in logit space; Viterbi at tau = 2 with priors from the external labels.
Nothing here is chosen on the real match: every constant comes from ``katena.eval.final`` and its imports.

``SPECS`` names the models ``train`` can fit. ``F3`` is the submission model (docs/decision-log.md, 2026-11-11):
F with a 1,200-tree LightGBM member and a 4,800-step TCN that also reads the touch columns (``katena.eval.f3``).
``F`` is the model it replaced, kept so its checks can be re-run. Each spec has its own folder (``model/`` for F,
``model_<spec>/`` otherwise), so one can never be taken for the other.

Stages, each cached under ``KATENA_WORK`` (``scripts/make_submission.py`` sets it to ``work/final``):

    corpus   ground truth, base features and pools of every external match: copied from a finished
             bake-off cache when one is given (byte-identical inputs), otherwise built
    worlds   the ``d2j`` worlds' features, their touch columns, and the ``d2j`` pools with touch columns
    train    the LightGBM member, the TCN member and the priors, saved with sha256 hashes to ``model_dir()``:
             ``model/`` for F, ``model_<spec>/`` for another spec
    predict  the real match from ``data/``: features at 5 Hz from the real ATD, touch columns from the real
             25 Hz ball, blend, Viterbi, one label per video frame, the kickoff rule (frames of a graded half
             before its Smart Tagging ``Kick Off (Start)`` become D), the submission file, validation

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
from katena.eval import f3 as F3
from katena.eval import final as F
from katena.eval import g1 as G1
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


@dataclass(frozen=True)
class Spec:
    """What ``train`` fits. Everything not listed here (pool, mix, parameters, seed, blend, decoder) is F's."""
    name: str
    gbdt_trees: int        # LightGBM rounds of both heads
    tcn_steps: int         # TCN training steps (the one-cycle schedule spans them)
    tcn_touch: bool        # the TCN also reads the touch columns (213 columns instead of 160)


SPECS = {"F": Spec("F", 150, 1200, False),
         "F3": Spec("F3", F3.N_TREES, F3.STEPS, True)}
SUBMISSION_SPEC = "F3"      # the model `make submit` trains and predicts with; changed only by an adoption


def spec_of(name: str | None = None) -> Spec:
    name = name or SUBMISSION_SPEC
    if name not in SPECS:
        raise ValueError(f"unknown model spec {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


def model_dir(hold_out=(), spec: str | None = None) -> Path:
    """``model/`` for F and ``model_<spec>/`` for any other spec, whichever is the submission model, so a cache that
    holds F's ``model/`` can never serve it as another model. A model with matches held out (a check, never the
    submission) lives in its own ``..._without_<keys>/``."""
    sp = spec_of(spec)
    base = MODEL_DIR if sp.name == "F" else f"{MODEL_DIR}_{sp.name}"
    if not hold_out:
        return P.WORK / base
    return P.WORK / (base + "_without_" + "+".join(sorted(k.replace("/", "_") for k in hold_out)))


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
    n_base: int            # the M2 columns come first
    tcn_state: dict
    tcn_n_in: int          # the TCN reads the first tcn_n_in columns: n_base (F) or all of them (F3)
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


def stage_train(refs: list, threads: int | None = None, steps: int | None = None, hold_out=(),
                spec: str | None = None) -> dict:
    """Fit both members on every row of every match (nothing held out) and save ``bundle.pkl`` in
    ``model_dir(hold_out, spec)``. ``spec`` names the model (``SPECS``; default the submission model).

    ``hold_out`` (match keys, for the dressed-match check only) leaves those matches out of the LightGBM rows,
    the TCN sequences and scaler, and the priors, exactly as a gate fold does. ``steps`` overrides the spec's
    TCN steps (tests only)."""
    from katena.eval import bakeoff as B
    from katena.models import ensemble as E
    from katena.models import gbdt
    sp = spec_of(spec)
    exclude = set(hold_out)
    if not exclude <= {r.key for r in refs}:
        raise ValueError(f"cannot hold out {sorted(exclude - {r.key for r in refs})}: not an external match")
    threads = threads or E.TCN_THREADS
    E.set_deterministic(threads)
    t0 = time.time()
    pools = F.FinalPools(refs)
    n_rows, n_cols = pools.X.shape
    log(f"spec {sp.name}; pool: {n_rows} rows x {n_cols} columns ({pools.n_base} base)")
    used = ~np.isin(pools.keys, list(exclude))
    heads = gbdt.fit_heads(pools.X[used], pools.o[used], pools.y[T][used], pools.dead_ok[used], pools.cols(),
                           n_trees=sp.gbdt_trees)
    t1 = time.time()
    log(f"LightGBM member ({sp.gbdt_trees} trees): {t1 - t0:.0f}s")
    tcn_steps = steps or sp.tcn_steps
    if sp.tcn_touch:                  # as the gate's folds: katena.eval.g1 at the spec's steps (katena.eval.f3)
        sc = G1.scaler(pools, exclude)
        seqs = G1.tcn_seqs(refs, T, SEED)
        net, sc = G1.fit_tcn(sc, seqs, exclude, SEED, threads, tcn_steps)
        tcn_n_in = n_cols
    else:
        seqs = F.tcn_seqs(refs, T, SEED)
        net, sc = F.fit_tcn(pools, seqs, exclude, SEED, threads, tcn_steps)
        tcn_n_in = int(pools.n_base)
    del seqs
    t2 = time.time()
    log(f"TCN member ({tcn_steps} steps, {tcn_n_in} columns): {t2 - t1:.0f}s")
    pri = B.priors_for(F.PriorSource(refs), exclude, T)
    model = Model(gbdt_alive=heads.alive.model_to_string(), gbdt_team=heads.team.model_to_string(),
                  gbdt_cols=np.asarray(heads.cols), names=list(pools.names), n_base=int(pools.n_base),
                  tcn_state={k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()},
                  tcn_n_in=int(tcn_n_in), scaler_mu=np.asarray(sc.mu), scaler_sd=np.asarray(sc.sd),
                  priors=pri, touch_src=TG.SRC)
    out = model_dir(hold_out, sp.name)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (BUNDLE + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(model, fh, protocol=4)
    tmp.replace(out / BUNDLE)
    manifest = {
        "matches": [r.key for r in refs if r.key not in exclude], "rows": int(n_rows),
        "columns": int(n_cols),
        "spec": {"name": sp.name, "gbdt_trees": sp.gbdt_trees, "tcn_steps": tcn_steps, "tcn_columns": int(tcn_n_in)},
        "mix": F.MIX, "T": T, "tau": TAU, "w_tcn": W_TCN, "seed": SEED, "tcn_threads": threads,
        "sha256": {"gbdt": F.gbdt_hash(heads), "tcn": E.model_hash(net), BUNDLE: sha256_file(out / BUNDLE)},
        "seconds": {"gbdt": round(t1 - t0, 1), "tcn": round(t2 - t1, 1)},
    }
    if exclude:
        manifest["held_out"] = sorted(exclude)
        manifest["rows_used"] = int(used.sum())
    (out / MANIFEST).write_bytes((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return manifest


def load_model(hold_out=(), spec: str | None = None) -> Model:
    with open(model_dir(hold_out, spec) / BUNDLE, "rb") as fh:
        return pickle.load(fh)


def tcn_columns(model: Model, base: np.ndarray, full: np.ndarray) -> np.ndarray:
    """The matrix the TCN member reads: the base columns (F) or the base and touch columns (F3). ``base`` and
    ``full`` are one orientation's M2 columns and the same with the touch columns appended."""
    if model.tcn_n_in == base.shape[1]:
        return base
    if model.tcn_n_in == full.shape[1]:
        return full
    raise RuntimeError(f"the TCN member reads {model.tcn_n_in} columns; the features have {base.shape[1]} "
                       f"base and {full.shape[1]} with the touch columns")


# ---------------------------------------------------------------------------------------------
# predict

def submission_bytes(labels) -> bytes:
    """'frame,label' and one LF-terminated row per frame, built as bytes so no OS turns LF into CRLF."""
    return ("frame,label\n" + "".join(f"{i},{v}\n" for i, v in enumerate(labels))).encode("ascii")


def label_match(model: Model, m, threads: int | None = None, set_pieces=None) -> tuple[np.ndarray, dict]:
    """int8 codes (0 H, 1 A, 2 D, 3 X) for every native frame of ``m``, and an aggregate-only report.

    ``m`` is a native-rate Match without labels. It is not corrupted: the real ATD is the noisy input
    the training worlds stand in for.

    ``set_pieces`` (``katena.featurize.tags.SetPiece`` list, from the real Smart Tagging) appends the G2 tag columns
    to the LightGBM member's input. Off by default: F has no tag columns, so a model must have been trained with them
    (G2, not adopted or integrated as of 2026-11-10), or the column check below refuses."""
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
    X0, X1 = np.concatenate([f0.X, A[0]], 1), np.concatenate([f1.X, A[1]], 1)
    T0, T1 = tcn_columns(model, f0.X, X0), tcn_columns(model, f1.X, X1)
    if set_pieces is not None:
        from katena.featurize import tags as TAGS
        G, gnames = TAGS.tag_columns(set_pieces, f0.idx, int(m5.meta["step"]), m.fps)
        names += gnames
        X0, X1 = np.concatenate([X0, G[0]], 1), np.concatenate([X1, G[1]], 1)
    if names != model.names:
        raise RuntimeError("feature columns differ from the trained model's")
    t1 = time.time()
    pa_g, ph_g = model.heads().predict(X0, X1)
    E.set_deterministic(threads)
    net, sc = model.tcn()
    import torch
    with torch.no_grad():
        pa_t, ph_t = tcn.predict(net, sc(T0), sc(T1), f0.half)
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
        # Was the start of each half read as dead ball? A half's graded frames can begin before its
        # kickoff (they are D in the ground truth). Shares over the opening seconds, aggregates only.
        "d_share_half_opening": {f"half{h}_first{s}s": round(float((codes[a:min(b + 1, a + int(s * m.fps))] == 2)
                                                                    .mean()), 3)
                                 for h, a, b in m.half_blocks for s in (60, 120)},
        "touch_events": TF.count_touches(m5, TF.native_ball(m)),
        "seconds": {"features": round(t1 - t0, 1), "models_and_decode": round(time.time() - t1, 1)},
    }
    return codes, report


# ---------------------------------------------------------------------------------------------
# kickoff from Smart Tagging (W4a, docs/plan-final-week.md)
#
# A graded half can begin before its kickoff, and the frames before the kickoff are dead ball (FAQ.md: the second
# half is graded from before its kickoff). The kickoff is read from the provided inputs, never from a frame constant:
# the half's Smart Tagging instance whose Type is "Kick Off (Start)" (Metrica's automatic tagging). Its start time
# maps to a video frame with round(seconds * 25), as DATA.md documents; frames of the block before it become D.

KICKOFF_TYPE = "kick off (start)"            # compared lower-case and stripped; "Kick Off (After Goal)" never matches
# How far into a graded half block a tagged kickoff may lie and still be used. A half block can open while the
# teams line up, so a kickoff a minute or two in is plausible. Much later, a "Kick Off (Start)" tag is more likely a
# tagging error than a long wait, and acting on it would turn minutes of play into D. Fixed on 2026-11-10 before the
# rule ever ran on the match, from no frame of it: 3 minutes. Past it, or outside the block, nothing changes.
KICKOFF_MAX_S = 180.0


def tagged_kickoffs(tags) -> dict:
    """{half: [video frame of the start of each "Kick Off (Start)" instance of that half]}. ``tags`` are
    ``katena.target.tagging.Tag`` (Type and Half matched case-insensitively there and here)."""
    out = {}
    for t in tags:
        if (t.kind or "").strip().lower() == KICKOFF_TYPE and t.half is not None:
            out.setdefault(int(t.half), []).append(int(t.start_frame))
    return out


def kickoff_status(frames: list, first: int, last: int, fps: float) -> tuple[str, int | None]:
    """(status, kickoff frame or None) of one graded half block ``[first, last]`` given its tagged kickoffs.
    The status is "applied" only when exactly one tag lies in ``[first, min(last, first + KICKOFF_MAX_S)]``."""
    if not frames:
        return "no_tag", None
    if len(frames) > 1:
        return "several_tags", None
    k = frames[0]
    if k < first:
        return "before_block", None
    if k > last:
        return "after_block", None
    if k > first + int(round(KICKOFF_MAX_S * fps)):
        return "beyond_bound", None
    return "applied", k


def apply_kickoff_rule(codes: np.ndarray, half_blocks, tags, fps: float,
                       missing: str = "no_tagging_file") -> tuple[np.ndarray, dict]:
    """``codes`` with the frames of each graded half before that half's tagged kickoff set to D (code 2), and a
    report of counts and flags only: per half, whether the rule applied, how many frames it changed, and why not.
    ``tags`` is the list read from the Smart Tagging file, or None when there is none (``missing`` says why); then
    nothing changes."""
    from katena.types import D
    out = np.array(codes, copy=True)
    kick = tagged_kickoffs(tags) if tags is not None else {}
    rep = {"bound_s": KICKOFF_MAX_S}
    for h, a, b in half_blocks:
        status, k = (missing, None) if tags is None else kickoff_status(kick.get(int(h), []), a, b, fps)
        changed = 0
        if k is not None:
            changed = int((out[a:k] != D).sum())
            out[a:k] = D
        rep[f"half{int(h)}"] = {"applied": k is not None, "frames_changed": changed, "status": status}
    return out, rep


def read_target_tags(data_dir: Path | None = None, fps: float = 25.0):
    """(tags, None) from the bundle's Smart Tagging file; (None, reason) when it is missing or unreadable. The
    reason holds an exception's type name only, never its message (messages can quote the file)."""
    from katena.target.atd import _games
    from katena.target.tagging import read_tags
    path = Path(_games().asset_paths(data_dir=str(data_dir) if data_dir else None)["smart_tagging"])
    if not path.is_file():
        return None, "no_tagging_file"
    try:
        return read_tags(path, fps=fps), None
    except Exception as e:  # noqa: BLE001  the type alone
        return None, f"unreadable_{type(e).__name__}"


def write_submission(codes: np.ndarray, path: Path, n_expected: int) -> dict:
    """Write the file, then run the official checks on the bytes written. Raises if they fail, and then leaves no
    file at ``path`` from this run (the bytes are checked in a temporary file and moved into place only if valid)."""
    root = P.REPO / "evaluation"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import possession_metrics as pm

    from katena.types import LABEL_NAMES
    data = submission_bytes(np.array(LABEL_NAMES)[codes])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    reason = pm.validate_submission(pm.read_label_csv(tmp), n_expected)
    if reason:
        tmp.unlink()
        raise RuntimeError(f"the submission fails the official checks: {reason}")
    tmp.replace(path)
    return {"path": str(path), "rows": int(len(codes)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def stage_predict(out: Path, data_dir: Path | None = None, threads: int | None = None, hold_out=(),
                  timeline=None, kickoff: bool = True, tag_columns: bool = False, spec: str | None = None) -> dict:
    """Label the real match with the model of ``spec`` (default the submission model) and write the submission.
    Prints aggregates only.

    After decoding, the kickoff rule (``apply_kickoff_rule``) sets each graded half's frames before its tagged
    ``Kick Off (Start)`` to D; ``kickoff=False`` turns it off. A bundle without a Smart Tagging file (an exported
    external match) is left unchanged, and the report says so.

    ``tag_columns=True`` reads the bundle's Smart Tagging ``SET PIECES`` intervals and gives them to ``label_match``
    as G2's tag columns. Off by default, and F refuses it (see ``label_match``).

    ``hold_out`` and ``timeline`` are for the dressed-match check only (an external match exported in the ATD
    format, ``scripts/dressed_check.py``): the held-out model, and the exported match's own frame count, half
    blocks, pitch and frame rate in place of ``games.py``'s."""
    from katena.target.atd import _games, load_target
    if hold_out or timeline is not None:
        competition = {Path(_games().DATA_DIR).resolve(), (P.REPO / "data").resolve()}
        if data_dir is None or Path(data_dir).resolve() in competition:
            raise ValueError("hold_out and timeline are for an exported external match, never the competition data")
    m = load_target(data_dir, timeline=timeline)
    rep = m.meta.get("read_report", {})
    sp = spec_of(spec)
    model = load_model(hold_out, sp.name)
    set_pieces = None
    if tag_columns:
        from katena.featurize import tags as TAGS
        tags, why = read_target_tags(data_dir, m.fps)
        if tags is None:
            raise RuntimeError(f"tag columns need the Smart Tagging file ({why})")
        set_pieces = TAGS.from_tagging(tags)
    codes, report = label_match(model, m, threads, set_pieces=set_pieces)
    if set_pieces is not None:
        report["tag_columns"] = {"set_pieces_by_type": TAGS.counts(set_pieces)}
    if kickoff:
        tags, why = read_target_tags(data_dir, m.fps)
        codes, report["kickoff_rule"] = apply_kickoff_rule(codes, m.half_blocks, tags, m.fps, missing=why or "")
    sub = write_submission(codes, out, m.n)
    report["read"] = {k: rep.get(k) for k in ("n_lines", "n_malformed", "n_out_of_range", "n_duplicate",
                                              "n_frames_present", "contiguous_from_one")}
    report["submission"] = sub
    report["model_spec"] = sp.name
    report["model_sha256"] = sha256_file(model_dir(hold_out, sp.name) / BUNDLE)
    return report


def env_report() -> dict:
    import platform

    import lightgbm
    import torch
    return {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "lightgbm": lightgbm.__version__, "torch": torch.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random")}
