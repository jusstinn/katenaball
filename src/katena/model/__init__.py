"""D4 model: per-frame log P(H, A, D) from D3 features, plus a placeholder decoder for scoring.

    from katena.model import Model, decode_simple
    model = Model.load("artifacts/katena_model.json")
    logp = model.predict_logp(feats)                       # float32[T, 3], columns H, A, D
    codes = decode_simple(logp, feats.t_frame, m.half_blocks, m.n, model.info["decode"])

`decode_simple` is a PLACEHOLDER (Viterbi with a switch penalty and a minimum dwell); J3 owns the
real semi-Markov decoder. Training data, the locked split and the leave-one-match-out study live in
`katena.model.data` and `katena.model.lomo`.
"""
from .decode import DEFAULT_DECODE, argmax_frames, decode_simple, segment_viterbi, upsample_logp
from .lgbm import Model

__all__ = ["Model", "decode_simple", "segment_viterbi", "upsample_logp", "argmax_frames",
           "DEFAULT_DECODE"]
