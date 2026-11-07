"""Leave-one-match-out (LOMO) study on the external development matches, with nested calibration.

For each held-out development match h and each model variant:

1. Train on the other development matches: their training copies (severity grid or default
   config), plus team-swapped copies with H and A swapped.
2. Nested, inside the training matches only: split them into `k_inner` groups (sorted names, round
   robin), train on all but one group, predict the left-out group's evaluation copies. These
   out-of-fold (OOF) raw scores fit the temperatures and choose the placeholder decoder's
   (switch_penalty, min_dwell_s) on a grid by mean spellF1 against harmonised labels.
3. Predict h's evaluation copy (default noise config, a seed range disjoint from training), apply
   the temperatures, decode with the chosen parameters, and score with the official
   `possession_metrics.evaluate` against h's harmonised labels.

The held-out match never touches training, calibration or decoder choice. The two locked matches
are never loaded (`data.check_names`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..external.common import possession_metrics
from ..match import X
from . import data as D
from .decode import DEFAULT_DECODE, argmax_frames, decode_frames, decode_simple, prepare_emissions
from .lgbm import DEFAULT_ROUNDS, Model

DECODE_GRID = {"min_dwell_s": (0.25, 0.5, 1.0, 2.0),
               "switch_penalty": (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)}
K_INNER = 2

REPORT_KEYS = ("spell_f1_mean", "macro_f1", "spell_f1_at_headline", "n_gt_spells", "n_pred_spells",
               "mean_spell_duration_s", "gt_mean_spell_duration_s", "accuracy", "alive_dead_f1",
               "team_acc_gt_alive", "team_acc_both_alive")


@dataclass
class Variant:
    name: str
    kind: str = "twohead"
    features: str = "all"
    aug: str = "severity"
    rounds: int = DEFAULT_ROUNDS
    params: dict = field(default_factory=dict)

    def model(self) -> Model:
        return Model(kind=self.kind, features=self.features, params=self.params, rounds=self.rounds)


VARIANTS = {
    "twohead": Variant("twohead", "twohead", "all", "severity"),
    "3class": Variant("3class", "3class", "all", "severity"),
    "twohead_ballfree": Variant("twohead_ballfree", "twohead", "ball_free", "severity"),
    "3class_ballfree": Variant("3class_ballfree", "3class", "ball_free", "severity"),
    "twohead_noaug": Variant("twohead_noaug", "twohead", "all", "default"),
    "3class_noaug": Variant("3class_noaug", "3class", "all", "default"),
}


# -- data access ---------------------------------------------------------------------------------

@lru_cache(maxsize=16)
def gt_full(name: str) -> tuple:
    y, hb = D.full_labels(name)
    y.setflags(write=False)
    return y, tuple(hb)


@lru_cache(maxsize=16)
def eval_copy(name: str) -> dict:
    return D.get_copy(D.eval_spec(name))


def training_set(names, aug: str, swap: bool = True):
    copies = [D.get_copy(s) for n in names for s in D.train_plan(n, aug)]
    X_, y, cols = D.stack_training(copies, swap=swap)
    del copies
    return X_, y, cols


def fit_variant(v: Variant, names) -> Model:
    X_, y, cols = training_set(D.check_names(names), v.aug)
    model = v.model().fit(X_, y, cols)
    model.info["train_matches"] = list(names)
    model.info["train_aug"] = v.aug
    del X_, y
    return model


# -- scoring -------------------------------------------------------------------------------------

def quick_scores(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(spellF1, macroF1) exactly as `possession_metrics.evaluate` computes them, minus the extras."""
    pm = possession_metrics()
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return float(spell), float(macro)


def full_scores(gt: np.ndarray, pred: np.ndarray) -> dict:
    res = possession_metrics().evaluate(gt, pred)
    return {k: (float(res[k]) if isinstance(res[k], (float, np.floating)) else res[k]) for k in REPORT_KEYS}


# -- nested calibration and decoder choice ---------------------------------------------------------

def inner_groups(names, k: int = K_INNER) -> list[list]:
    names = sorted(names)
    k = max(2, min(k, len(names)))
    return [names[i::k] for i in range(k)]


def oof_raw(v: Variant, names, k: int = K_INNER, log=print) -> dict:
    """{match: raw scores on its evaluation copy} from models that never saw that match."""
    out = {}
    for g in inner_groups(names, k):
        rest = [n for n in names if n not in g]
        t = time.time()
        mdl = fit_variant(v, rest)
        for n in g:
            ev = eval_copy(n)
            out[n] = mdl.raw_scores(ev["X"], ev["names"])
        log(f"    inner fit on {len(rest)} -> predict {g} ({time.time() - t:.0f} s)")
        del mdl
    return out


def choose_decoder(model: Model, raw: dict, temperature: dict, grid: dict = DECODE_GRID) -> dict:
    """Grid search of (min_dwell_s, switch_penalty) on OOF predictions. Highest mean spellF1 wins,
    then highest mean macroF1, then the earlier grid point."""
    pens = list(grid["switch_penalty"])
    table = []
    for d in grid["min_dwell_s"]:
        sums = np.zeros((len(pens), 2))
        for n, r in raw.items():
            ev = eval_copy(n)
            gt, hb = gt_full(n)
            lp = model.logp_from_raw(r, temperature)
            E = prepare_emissions(lp, ev["t_frame"], hb, len(gt))
            preds = decode_frames(E, hb, {"min_dwell_s": d}, penalties=pens)
            for i in range(len(pens)):
                sums[i] += quick_scores(gt, preds[i])
        for i, p in enumerate(pens):
            s, mf = sums[i] / len(raw)
            table.append({"min_dwell_s": float(d), "switch_penalty": float(p),
                          "spell_f1": float(s), "macro_f1": float(mf)})
    best = max(enumerate(table), key=lambda it: (round(it[1]["spell_f1"], 6),
                                                 round(it[1]["macro_f1"], 6), -it[0]))[1]
    return {"decode": {"min_dwell_s": best["min_dwell_s"], "switch_penalty": best["switch_penalty"]},
            "inner_spell_f1": best["spell_f1"], "inner_macro_f1": best["macro_f1"], "table": table}


def calibrate_and_choose(v: Variant, model: Model, names, k: int = K_INNER, log=print) -> dict:
    raw = oof_raw(v, names, k, log)
    ys = [eval_copy(n)["y"] for n in raw]
    temperature = model.fit_temperature(np.concatenate([raw[n] for n in raw]), np.concatenate(ys))
    choice = choose_decoder(model, raw, temperature)
    choice["temperature"] = temperature
    choice["oof_nll_raw"] = _nll(model, raw, None)
    choice["oof_nll_cal"] = _nll(model, raw, temperature)
    return choice


def _nll(model: Model, raw: dict, temperature) -> float:
    tot, n = 0.0, 0
    unit = model._unit_temperature()
    for k, r in raw.items():
        y = eval_copy(k)["y"].astype(np.int64)
        lp = model.logp_from_raw(r, temperature or unit).astype(np.float64)
        tot += -float(lp[np.arange(len(y)), y].sum())
        n += len(y)
    return tot / max(1, n)


# -- one fold ------------------------------------------------------------------------------------

def run_fold(v: Variant, held_out: str, dev=D.DEV_NAMES, k: int = K_INNER, log=print) -> dict:
    D.check_names([held_out] + list(dev))
    train = [n for n in dev if n != held_out]
    t0 = time.time()
    model = fit_variant(v, train)
    t_fit = time.time() - t0
    log(f"  [{v.name}] outer fit on {len(train)} matches, {model.info['n_train_rows']} rows "
        f"({t_fit:.0f} s)")
    t1 = time.time()
    choice = calibrate_and_choose(v, model, train, k, log)
    t_inner = time.time() - t1
    model.set_temperature(choice["temperature"])
    ev = eval_copy(held_out)
    gt, hb = gt_full(held_out)
    lp = model.predict_logp(D.as_feats(ev, held_out))
    pred = decode_simple(lp, ev["t_frame"], hb, len(gt), choice["decode"])
    pred_argmax = argmax_frames(lp, ev["t_frame"], hb, len(gt))
    res = {"variant": asdict(v), "held_out": held_out, "train": train,
           "decode": choice["decode"], "temperature": choice["temperature"],
           "inner_spell_f1": choice["inner_spell_f1"], "inner_macro_f1": choice["inner_macro_f1"],
           "oof_nll_raw": choice["oof_nll_raw"], "oof_nll_cal": choice["oof_nll_cal"],
           "decode_table": choice["table"],
           "scores": full_scores(gt, pred), "scores_argmax": full_scores(gt, pred_argmax),
           "frame_nll": _frame_nll(lp, ev["y"]),
           "time_fit_s": t_fit, "time_inner_s": t_inner, "time_total_s": time.time() - t0,
           "n_train_rows": model.info["n_train_rows"], "pred": pred}
    log(f"  [{v.name}] {held_out}: spellF1 {res['scores']['spell_f1_mean']:.4f} macroF1 "
        f"{res['scores']['macro_f1']:.4f} spells {res['scores']['n_pred_spells']}/"
        f"{res['scores']['n_gt_spells']} decode {choice['decode']} T {choice['temperature']} "
        f"({res['time_total_s']:.0f} s)")
    return res


def robustness_fold(v: Variant, held_out: str, fold: dict, severity_indices=(0, 4),
                    dev=D.DEV_NAMES, log=print) -> dict:
    """Refit the outer model of a finished fold (deterministic, so it is the same model), reuse its
    nested temperatures and decoder parameters, and score it on evaluation copies whose noise is
    milder or harsher than the default. Also checks that the refit reproduces the fold's default
    prediction exactly."""
    train = [n for n in dev if n != held_out]
    t0 = time.time()
    model = fit_variant(v, train).set_temperature(fold["temperature"])
    ev = eval_copy(held_out)
    gt, hb = gt_full(held_out)
    pred0 = decode_simple(model.predict_logp(D.as_feats(ev, held_out)), ev["t_frame"], hb, len(gt),
                          fold["decode"])
    out = {"variant": v.name, "held_out": held_out, "reproduced_default": bool(
        np.array_equal(pred0, fold["pred"])) if "pred" in fold else None, "by_severity": {}}
    for si in severity_indices:
        spec = D.eval_spec(held_out, severity_index=si)
        d = D.get_copy(spec)
        pred = decode_simple(model.predict_logp(D.as_feats(d, held_out)), d["t_frame"], hb, len(gt),
                             fold["decode"])
        out["by_severity"][f"{spec.severity:.2f}"] = {
            "model": full_scores(gt, pred), "baseline": full_scores(gt, d["baseline"]),
            "ball_present_graded": d["ball_present_graded"]}
    out["time_s"] = time.time() - t0
    log(f"  [{v.name}] {held_out} robustness: reproduced={out['reproduced_default']} " + ", ".join(
        f"sev {k}: {r['model']['spell_f1_mean']:.4f}" for k, r in out["by_severity"].items())
        + f" ({out['time_s']:.0f} s)")
    return out


def _frame_nll(lp: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    return -float(np.mean(lp.astype(np.float64)[np.arange(len(y)), y]))


def baseline_fold(held_out: str) -> dict:
    ev = eval_copy(held_out)
    gt, _ = gt_full(held_out)
    return {"variant": {"name": "baseline"}, "held_out": held_out, "scores": full_scores(gt, ev["baseline"])}


# -- comparisons ---------------------------------------------------------------------------------

def spell_flags(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, int]:
    """Matched-or-not per GT spell at each IoU threshold (bool[5, n_gt]) and the predicted count."""
    pm = possession_metrics()
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(g, p)
    return np.vstack([pm.matched_gt_flags(ov, len(g), t) for t in pm.SPELL_IOUS]), len(p)


def _f1_from(flags: np.ndarray, picks: np.ndarray, n_pred: int) -> float:
    n_gt = len(picks)
    tp = flags[:, picks].sum(axis=1)
    fp = np.maximum(0, n_pred - tp)
    fn = n_gt - tp
    f1 = np.where(tp > 0, 2.0 * tp / np.maximum(1, 2 * tp + fp + fn), 0.0)
    return float(f1.mean())


def paired_compare(per_match_a: dict, per_match_b: dict, preds_a: dict | None = None,
                   preds_b: dict | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """A minus B on spellF1 over matches: win count, mean difference, and a 90% paired bootstrap.

    `per_match_*` map match -> spellF1. With per-frame predictions (`preds_*`: match -> codes), the
    bootstrap is hierarchical: resample matches, then the GT possessions inside each (shared by A
    and B), as the scorer's own spell bootstrap does. Without them it resamples matches only.
    """
    names = sorted(set(per_match_a) & set(per_match_b))
    diff = np.array([per_match_a[n] - per_match_b[n] for n in names])
    wins = int((diff > 0).sum())
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    if preds_a is not None and preds_b is not None:
        fl = {}
        for n in names:
            gt, _ = gt_full(n)
            fl[n] = (spell_flags(gt, preds_a[n]), spell_flags(gt, preds_b[n]))
        for b in range(n_boot):
            ms = rng.randint(0, len(names), len(names))
            tot = 0.0
            for mi in ms:
                (fa, na), (fb, nb) = fl[names[mi]]
                picks = rng.randint(0, fa.shape[1], fa.shape[1])
                tot += _f1_from(fa, picks, na) - _f1_from(fb, picks, nb)
            boots[b] = tot / len(names)
        kind = "hierarchical (matches, then GT possessions)"
    else:
        for b in range(n_boot):
            boots[b] = diff[rng.randint(0, len(diff), len(diff))].mean()
        kind = "matches"
    lo, hi = np.percentile(boots, [5, 95])
    need = int(np.ceil(0.7 * len(names)))
    return {"n": len(names), "wins": wins, "mean_diff": float(diff.mean()),
            "ci90": [float(lo), float(hi)], "bootstrap": kind,
            "passes_gate": bool(wins >= need and lo > 0), "wins_needed": need,
            "per_match": {n: float(d) for n, d in zip(names, diff)}}


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_np_default), encoding="utf-8")


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
