"""External matches only. Is the ceiling set by noise, or by our features and 5 Hz grid?
Needs the bake-off cache (scripts/run_bakeoff.py). Run: uv run python scripts/probe_clean_ceiling.py

Train and test on native (uncorrupted) inputs, same features, same LOMO folds, same decoder as M2.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

from katena.decode import hmm  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
T, STRIDE = 2, 4


def rows_for(ref):
    f = C.load_feat(ref, "native")
    g = C.load_gt(ref, T)
    sel = np.arange(0, len(f["idx"]), STRIDE)
    o = np.random.default_rng(P.seed_for(ref.id, "clean")).integers(0, 2, len(sel)).astype(np.int8)
    X = np.where(o[:, None] == 1, f["X1"][sel], f["X0"][sel]).astype(np.float32)
    y = g["y"][::g["step"]][f["idx"][sel]]
    return X, o, y, np.full(len(sel), not ref.is_sc)


def main():
    dev, _ = P.split(P.all_refs())
    pool = {r.key: rows_for(r) for r in dev}
    labelled = [r for r in dev if not r.is_sc]
    seqs = {r.key: (lambda g: g["y"][::g["step"]])(C.load_gt(r, T)) for r in labelled}
    res = Lm.Results().d
    out = {}
    for r in labelled:
        keys = [k for k in pool if k != r.key]
        X = np.concatenate([pool[k][0] for k in keys])
        o = np.concatenate([pool[k][1] for k in keys])
        y = np.concatenate([pool[k][2] for k in keys])
        ok = np.concatenate([pool[k][3] for k in keys])
        heads = gbdt.fit_heads(X, o, y, ok, np.arange(X.shape[1]))
        f = C.load_feat(r, "native")
        pa, ph = heads.predict(f["X0"], f["X1"])
        pri = hmm.fit_priors([s for k, s in seqs.items() if k != r.key])
        g = C.load_gt(r, T)
        sc = {tau: Lm.score_rows(r, T, "native", hmm.decode(pa, ph, f["half"], pri, tau, "viterbi"), f["idx"], g)
              for tau in (1.0, 2.0, 4.0)}
        robust = res[("M2", T, "native", r.key)]
        out[r.key] = (r.dataset, sc[2.0]["spell_f1"], sc[1.0]["spell_f1"], sc[4.0]["spell_f1"],
                      robust["spell_f1"], sc[2.0]["team_acc"], robust["team_acc"],
                      sc[2.0]["n_pred"] / sc[2.0]["n_gt"])
        print(r.key, *[f"{v:.3f}" if isinstance(v, float) else v for v in out[r.key]], flush=True)
    for d in ("idsse", "metrica"):
        v = np.array([x[1:] for x in out.values() if x[0] == d], dtype=float).mean(0)
        print(f"{d}: clean->clean spellF1 tau2 {v[0]:.3f} tau1 {v[1]:.3f} tau4 {v[2]:.3f} | "
              f"robust->clean {v[3]:.3f} | team acc clean {v[4]:.3f} robust {v[5]:.3f} | pred/GT spells {v[6]:.2f}")


if __name__ == "__main__":
    main()
