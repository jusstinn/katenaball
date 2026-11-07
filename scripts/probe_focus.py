"""Focus probe on external matches only. Needs the bake-off cache (scripts/run_bakeoff.py).

    uv run python scripts/probe_focus.py

1. Re-score M2 and compare with the stored table (independent check of the pipeline).
2. Which true possessions do we lose, by duration?
3. How much could noisy phase tags with a team be worth? Tags are simulated from the external labels,
   at the published density (602 team tags over 158857 graded frames), across tag accuracy and fusion weight.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

import possession_metrics as pm  # noqa: E402
from katena.decode import hmm  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval.score import score, to_native_codes  # noqa: E402

T, COND, TAU = 2, "target", 2.0
SECONDS_PER_TAG = (158857 / 25) / 602
EDGES = [0, 2, 4, 8, 16, 32, 1e9]
ACCS = (0.6, 0.7, 0.8, 0.9)
WEIGHTS = (0.1, 0.25, 0.5, 1.0)


def make_tags(y5, model_team, acc, correlated, median_s, seed):
    """[(a, b, sign)] on the 5 Hz rows. sign +1 = home in possession."""
    rng = np.random.default_rng(seed)
    alive = np.flatnonzero((y5 == 0) | (y5 == 1))
    n = int(len(alive) / 5.0 / SECONDS_PER_TAG)
    # Overall error rate stays 1 - acc. In the correlated world errors land where the model errs.
    truth, wrong_model, spans = [], [], []
    for start in rng.choice(alive, size=n, replace=False):
        dur = rng.lognormal(np.log(median_s), 0.6)
        a = int(start + rng.normal(0, 5))
        b = int(a + dur * 5)
        a, b = max(a, 0), min(b, len(y5))
        seg = y5[a:b]
        h, w = int((seg == 0).sum()), int((seg == 1).sum())
        if h + w == 0:
            continue
        tr = 1 if h >= w else -1
        truth.append(tr)
        wrong_model.append(np.sign(model_team[a:b].mean() - 0.5 + 1e-9) != tr)
        spans.append((a, b))
    truth, wrong_model = np.array(truth), np.array(wrong_model)
    e = 1 - acc
    if correlated:
        wm = wrong_model.mean()
        e_w = min(1.0, 3 * e)
        e_r = max(0.0, (e - wm * e_w) / max(1 - wm, 1e-9))
        p_err = np.where(wrong_model, e_w, e_r)
    else:
        p_err = np.full(len(truth), e)
    flip = rng.random(len(truth)) < p_err
    signs = np.where(flip, -truth, truth)
    return [(a, b, int(s)) for (a, b), s in zip(spans, signs)]


def apply_tags(ph, tags, weight):
    lo = np.log(np.clip(ph, 1e-4, 1 - 1e-4)) - np.log(1 - np.clip(ph, 1e-4, 1 - 1e-4))
    for a, b, s in tags:
        lo[a:b] += s * weight
    return 1 / (1 + np.exp(-lo))


def main():
    dev, _ = P.split(P.all_refs())
    res = Lm.Results().d
    labelled = [r for r in dev if not r.is_sc]
    seqs = {r.key: (lambda g: g["y"][::g["step"]])(C.load_gt(r, T)) for r in labelled}

    table_scores, rescored = [], []
    by_bucket = {d: np.zeros((len(EDGES) - 1, 3)) for d in ("idsse", "metrica", "skillcorner")}
    base, cache = {}, {}
    for r in dev:
        pri = hmm.fit_priors([s for k, s in seqs.items() if k != r.key])
        f = C.load_feat(r, COND)
        oof = Lm.load_oof("full", T, r)
        g = C.load_gt(r, T)
        taus = res[("tau_table", T, "M2", "all")]["taus"]
        rows = hmm.decode(oof[f"pa_{COND}"], oof[f"ph_{COND}"], f["half"], pri, taus[r.key], "viterbi")
        s = Lm.score_rows(r, T, COND, rows, f["idx"], g)
        rescored.append(s["spell_f1"])
        table_scores.append(res[("M2", T, COND, r.key)]["spell_f1"])

        rows2 = hmm.decode(oof[f"pa_{COND}"], oof[f"ph_{COND}"], f["half"], pri, TAU, "viterbi")
        pred = to_native_codes(rows2, f["idx"], g["step"], g["n_native"], g["blocks"])
        if r.is_sc:
            pred = np.where(pred == 2, 3, pred).astype(np.int8)
        runs = pm.valid_runs(g["y"])
        gt_sp = pm.possession_spells(g["y"], runs)
        pr_sp = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
        flags = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
        fps = float(g["fps"])
        dur = np.array([(b - a) / fps for a, b, _ in gt_sp])
        for i in range(len(EDGES) - 1):
            m = (dur >= EDGES[i]) & (dur < EDGES[i + 1])
            by_bucket[r.dataset][i] += (m.sum(), flags[m].sum(), dur[m].sum())

        y_rows = g["y"][::g["step"]][f["idx"]]
        base[r.key] = score(g["y"], pred, sc_mode=r.is_sc)["spell_f1"]
        cache[r.key] = (r, f, oof, g, pri, y_rows, (oof[f"ph_{COND}"] > 0.5).astype(float))

    table_scores, rescored = np.array(table_scores), np.array(rescored)
    print(f"[check] M2 rescored {rescored.mean():.4f} vs stored table {table_scores.mean():.4f}; "
          f"max per-match abs diff {np.abs(rescored - table_scores).max():.6f}")
    print(f"[check] fixed tau={TAU}: mean {np.mean(list(base.values())):.4f}")

    print("\nTrue possessions by duration, matched at IoU 0.5 (M2, target-like, T=2)")
    print("dataset      bucket_s    n  share_spells share_time matched")
    for d, tab in by_bucket.items():
        for i in range(len(EDGES) - 1):
            n, hit, t = tab[i]
            print(f"{d:<12} {EDGES[i]:>3.0f}-{EDGES[i+1]:<5.0f} {int(n):>5} {n/tab[:,0].sum():>9.1%} "
                  f"{t/tab[:,2].sum():>10.1%} {hit/max(n,1):>7.1%}")

    print(f"\nSimulated team tags, one per {SECONDS_PER_TAG:.1f} s of graded time, median length 6 s")
    print("spellF1 gain over the same model without tags, mean over 28 folds (8 labelled folds in brackets)")
    keys = [r.key for r in dev]
    lab = [r.key for r in labelled]
    for correlated in (False, True):
        print("errors " + ("concentrated where the model errs (3x)" if correlated else "independent of the model"))
        print("acc \\ weight " + "".join(f"{w:>16}" for w in WEIGHTS))
        for acc in ACCS:
            line = f"{acc:<12}"
            for w in WEIGHTS:
                gains = {}
                for k in keys:
                    r, f, oof, g, pri, y_rows, mt = cache[k]
                    tags = make_tags(y_rows, mt, acc, correlated, 6.0, P.seed_for(k, f"tag{acc}"))
                    ph = apply_tags(oof[f"ph_{COND}"], tags, w)
                    rows = hmm.decode(oof[f"pa_{COND}"], ph, f["half"], pri, TAU, "viterbi")
                    gains[k] = Lm.score_rows(r, T, COND, rows, f["idx"], g)["spell_f1"] - base[k]
                line += f"{np.mean(list(gains.values())):>+9.3f}({np.mean([gains[k] for k in lab]):+.3f})"
            print(line, flush=True)


if __name__ == "__main__":
    main()
