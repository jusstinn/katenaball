"""The dressed-match check: an external match exported in the competition's ATD format (``katena.target.dress``)
and labelled by the submission path (``katena.pipeline.stage_predict``) must score what the evaluation path
scores for the same model at the clean condition (``native``).

Evaluation path, as the gate scores a fold (``katena.eval.final``): the cache's ``native`` base features
(``corpus.build_one``) and the ``native`` touch columns as ``touch_gate`` computes them (5 Hz players, the native
ball), the LightGBM member on both, the TCN member on the base columns through its scaler, ``ensemble.blend_heads``
at w = 0.5, the Viterbi decode at tau = 2 with the model's priors, and ``lomo.score_rows`` (``to_native_codes`` with
the half blocks, then the official ``possession_metrics``).

``submission_features`` recomputes what ``pipeline.label_match`` computes from a loaded Match, for diagnosis when
the two paths disagree. External matches only.
"""

from __future__ import annotations

import numpy as np

from katena.decode import hmm
from katena.eval import corpus as C
from katena.eval import final as F
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.eval import touch_gate as TG
from katena.featurize import touch as TF

KEYS = ("X0", "X1", "A0", "A1", "idx", "half")


def eval_features(ref: P.Ref) -> dict:
    from katena.grid import downsample
    from katena.types import load_match
    f = C.load_feat(ref, "native")
    m = load_match(ref.path)
    m5 = downsample(m)
    mc, lifted = TG.corrupted(ref, m, m5, "native")
    A, names, idx = TF.touch_features(mc, TG._ball("touch", m, m5, mc, lifted))
    if not np.array_equal(idx, f["idx"]):
        raise RuntimeError(f"{ref.key}: touch rows do not line up with the cached native features")
    return {"X0": f["X0"], "X1": f["X1"], "A0": A[0], "A1": A[1], "idx": f["idx"], "half": f["half"],
            "names": list(f["names"]) + list(names)}


def submission_features(m) -> dict:
    """The features ``pipeline.label_match`` builds from a native-rate Match (same calls, same order)."""
    from katena.featurize.build import featurize, infer_dirs
    from katena.grid import downsample
    m5 = downsample(m)
    dirs = infer_dirs(m5)
    f0, f1 = featurize(m5, 0, dirs), featurize(m5, 1, dirs)
    A, names, idx = TF.touch_features(m5, TF.native_ball(m))
    return {"X0": f0.X, "X1": f1.X, "A0": A[0], "A1": A[1], "idx": f0.idx, "half": f0.half,
            "names": list(f0.names) + list(names)}


def predict(model, f: dict, threads: int | None = None) -> dict:
    """Both members, the blend and the work-grid labels of a pipeline ``Model`` on features ``f``."""
    import torch

    from katena.models import ensemble as E
    from katena.models import tcn
    threads = threads or E.TCN_THREADS
    if f["names"] != model.names:
        raise RuntimeError("feature columns differ from the model's")
    pa_g, ph_g = model.heads().predict(np.concatenate([f["X0"], f["A0"]], 1), np.concatenate([f["X1"], f["A1"]], 1))
    E.set_deterministic(threads)
    net, sc = model.tcn()
    with torch.no_grad():
        pa_t, ph_t = tcn.predict(net, sc(f["X0"]), sc(f["X1"]), f["half"])
    pa, ph = E.blend_heads((pa_g, ph_g), (pa_t, ph_t), F.W_F)
    rows = hmm.decode(pa, ph, f["half"], model.priors, F.TAU, "viterbi")
    return {"pa_g": pa_g, "ph_g": ph_g, "pa_t": pa_t, "ph_t": ph_t, "pa": pa, "ph": ph, "rows": rows}


def eval_path(model, ref: P.Ref, T: int = P.T_MAIN, threads: int | None = None) -> dict:
    """The evaluation path for ``ref`` at ``native``: features, probabilities, labels, native codes and score."""
    from katena.eval.score import to_native_codes
    f = eval_features(ref)
    out = predict(model, f, threads)
    g = C.load_gt(ref, T)
    out["native"] = to_native_codes(out["rows"], f["idx"], g["step"], g["n_native"], g["blocks"])
    out["score"] = Lm.score_rows(ref, T, "native", out["rows"], f["idx"], g)
    out["features"] = f
    return out
