#!/usr/bin/env python
"""An external match dressed as the competition target: does the submission path score what the evaluation path
scores for the same held-out model?

    export KATENA_EXTERNAL=D:/katena/external_data KATENA_WORK=D:/katena/work-submit
    PY=.venv/Scripts/python.exe; D=D:/katena/dressed/J03WMX; K=idsse/DFL-MAT-J03WMX
    $PY scripts/dressed_check.py export --match $K --dir $D        # bundle, timeline, labels; read-back check
    $PY scripts/make_submission.py train --hold-out $K
    $PY scripts/make_submission.py predict --hold-out $K --data-dir $D/bundle --timeline $D/timeline.json \
        --out $D/run1/dressed.csv
    $PY scripts/dressed_check.py compare --match $K --dir $D --csv $D/run1/dressed.csv
    $PY scripts/dressed_check.py diff --gt $D/gt_T2.csv $D/run1/dressed.csv $D/run2/dressed.csv
    $PY scripts/dressed_check.py label --bundle other_machine/bundle.pkl --dir $D --out $D/other.csv

``export`` writes the canonical match in the competition's ATD format (``katena.target.dress``) with the timeline
``load_target`` would otherwise take from games.py, and the match's own labels at T = 2 as a ``frame,label`` CSV,
then reads the bundle back with ``load_target`` and compares every field with the canonical match.

``compare`` runs the held-out model down both paths (``katena.eval.dressed``): the evaluation path on the cache's
``native`` features, and the submission path's features recomputed from the bundle; it prints whether features,
probabilities and labels agree, and scores the CSV ``predict`` wrote with the official scorer
(``run_evaluation.score_submission``) beside the evaluation path's ``lomo.score_rows``. SkillCorner has no dead-ball
label, so the evaluation path turns a predicted D into X (``score(sc_mode=True)``); the CSV is scored as written and
with that mapping.

``diff`` counts graded frames on which two CSVs differ and scores both. ``label`` labels the exported match with any
``bundle.pkl`` (``pipeline.label_match``), e.g. one trained on another machine, to compare labels across machines. External matches only; output is
aggregates and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()

from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()

import numpy as np  # noqa: E402

import possession_metrics as pm  # noqa: E402
from katena.eval import corpus as C  # noqa: E402
from katena.eval import protocol as P  # noqa: E402

T = P.T_MAIN


def _ref(key: str) -> P.Ref:
    refs = {r.key: r for r in P.all_refs()}
    if key not in refs:
        sys.exit(f"{key} is not in the corpus under {P.CANON}")
    return refs[key]


def export(a) -> None:
    from katena import pipeline as PL
    from katena.target import dress
    from katena.target.atd import load_target
    from katena.types import LABEL_NAMES, load_match
    ref = _ref(a.match)
    d = Path(a.dir)
    m = load_match(ref.path)
    tl = dress.write_bundle(m, d / "bundle", name=f"dressed_{ref.dataset}_{ref.id}")
    (d / "timeline.json").write_text(json.dumps(tl.to_json(), indent=1) + "\n", encoding="utf-8", newline="\n")
    gt = C.load_gt(ref, T)["y"]
    (d / "gt_T2.csv").write_bytes(PL.submission_bytes(np.array(LABEL_NAMES)[gt]))
    back = load_target(d / "bundle", timeline=tl)
    same = dress.compare(m, back)
    rep = back.meta["read_report"]
    paths = dress.bundle_paths(d / "bundle")
    print(f"exported {ref.key}: {m.n} frames x {m.n_slots} slots at {m.fps:g} fps, pitch {m.pitch}, "
          f"{len(m.half_blocks)} half blocks; raw file {paths['atd_raw'].stat().st_size / 2 ** 20:.0f} MB")
    print(f"read back: contiguous 1..n {rep['contiguous_from_one']}, malformed {rep['n_malformed']}")
    for k, v in same.items():
        print(f"  {k:12s} {'identical' if v else 'DIFFERENT'}")
    bitwise = {k: getattr(m, k).tobytes() == getattr(back, k).tobytes() for k in ("P", "speed", "ball", "team")}
    print(f"  bitwise, NaN payloads included: {bitwise}")
    print(f"labels at T = {T}: {int((gt < 3).sum())} graded frames -> {d / 'gt_T2.csv'}")
    if not all(same.values()):
        sys.exit(1)


def compare(a) -> None:
    import run_evaluation as RE

    from katena import pipeline as PL
    from katena.eval import dressed as Dr
    from katena.eval.score import score
    from katena.grid import mask_outside_blocks
    from katena.target.atd import Timeline, load_target
    from katena.types import D, X, load_match
    ref = _ref(a.match)
    d = Path(a.dir)
    man = json.loads((PL.model_dir([a.match]) / PL.MANIFEST).read_text(encoding="utf-8"))
    model = PL.load_model([a.match])
    print(f"model {PL.model_dir([a.match]).name}: {len(man['matches'])} training matches, held out {man['held_out']}; "
          f"gbdt {man['sha256']['gbdt'][:16]}, tcn {man['sha256']['tcn'][:16]}")
    ev = Dr.eval_path(model, ref)
    tl = Timeline.from_json(json.loads((d / "timeline.json").read_text(encoding="utf-8")))
    sub = Dr.submission_features(load_target(d / "bundle", timeline=tl))
    print("features, evaluation path (cache, native) vs submission path (bundle through load_target):")
    for k in Dr.KEYS:
        e, s = ev["features"][k], sub[k]
        print(f"  {k:4s} {e.shape}: equal {np.array_equal(e, s, equal_nan=True)}, bitwise {e.tobytes() == s.tobytes()}")
    print(f"  column names equal {ev['features']['names'] == sub['names']}")
    ps = Dr.predict(model, sub)
    for k in ("pa_g", "ph_g", "pa_t", "ph_t", "pa", "ph"):
        print(f"  {k:4s} bitwise equal {ev[k].tobytes() == ps[k].tobytes()}, "
              f"max |diff| {float(np.max(np.abs(ev[k].astype(np.float64) - ps[k]))):.3g}")
    print(f"  work-grid labels: {int((ev['rows'] != ps['rows']).sum())} of {len(ev['rows'])} differ")

    m = load_match(ref.path)
    codes = pm.label_codes(pm.read_label_csv(str(a.csv)))
    ev_native = mask_outside_blocks(ev["native"], m)
    graded = np.zeros(m.n, bool)
    for _, lo, hi in m.half_blocks:
        graded[lo:hi + 1] = True
    print(f"  CSV vs evaluation path: {int((codes != ev_native).sum())} frames differ ({int((codes[graded] != ev_native[graded]).sum())} "
          f"in the halves); the evaluation path labels {int((ev['native'] != ev_native).sum())} frames just outside the "
          "halves that predict writes X (not graded)")

    gt = pm.label_codes(pm.read_label_csv(str(d / "gt_T2.csv")))
    assert np.array_equal(gt, C.load_gt(ref, T)["y"])
    res, reason = RE.score_submission(str(a.csv), gt, tuple(pm.TAUS), 0)
    if res is None:
        sys.exit(f"the official scorer skipped the CSV: {reason}")
    s = ev["score"]
    print(f"\nscores at T = {T}, clean input, {ref.key}, model without it:")
    print(f"  evaluation path (lomo.score_rows):  spellF1 {s['spell_f1']:.6f}  macroF1 {s['macro_f1']:.6f}  "
          f"possessions {s['n_pred']} predicted / {s['n_gt']} true")
    print(f"  CSV, official scorer:               spellF1 {res['spell_f1_mean']:.6f}  macroF1 {res['macro_f1']:.6f}  "
          f"possessions {res['n_pred_spells']} predicted / {res['n_gt_spells']} true")
    out = {"key": ref.key, "eval": {k: s[k] for k in ("spell_f1", "macro_f1", "n_pred", "n_gt")},
           "csv": {"spell_f1": res["spell_f1_mean"], "macro_f1": res["macro_f1"], "n_pred": res["n_pred_spells"],
                   "n_gt": res["n_gt_spells"]}, "csv_sha256": hashlib.sha256(Path(a.csv).read_bytes()).hexdigest()}
    if ref.is_sc:
        r2 = pm.evaluate(gt, np.where(codes == D, X, codes).astype(np.int8))
        s2 = score(gt, codes, sc_mode=True)
        print(f"  CSV, D -> X as the evaluation path: spellF1 {r2['spell_f1_mean']:.6f}  macroF1 {r2['macro_f1']:.6f}  "
              f"(SkillCorner has no dead-ball label; score(sc_mode=True) gives {s2['spell_f1']:.6f})")
        out["csv_sc_mode"] = {"spell_f1": r2["spell_f1_mean"], "macro_f1": r2["macro_f1"]}
    print(f"  difference (CSV - evaluation path): spellF1 {out['csv']['spell_f1'] - s['spell_f1']:+.6f}, "
          f"macroF1 {out['csv']['macro_f1'] - s['macro_f1']:+.6f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


def diff(a) -> None:
    gt = pm.label_codes(pm.read_label_csv(a.gt))
    graded = gt < 3
    codes = [pm.label_codes(pm.read_label_csv(p)) for p in a.csvs]
    for p, c in zip(a.csvs, codes):
        r = pm.evaluate(gt, c)
        print(f"{p}: sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]}, spellF1 {r['spell_f1_mean']:.6f}, "
              f"macroF1 {r['macro_f1']:.6f}, {r['n_pred_spells']} possessions")
    for p, c in zip(a.csvs[1:], codes[1:]):
        print(f"{a.csvs[0]} vs {p}: {int((c != codes[0]).sum())} frames differ, {int((c[graded] != codes[0][graded]).sum())} "
              f"of {int(graded.sum())} graded")


def label(a) -> None:
    import pickle

    from katena import pipeline as PL
    from katena.target.atd import Timeline, load_target
    from katena.types import LABEL_NAMES
    d = Path(a.dir)
    with open(a.bundle, "rb") as fh:
        model = pickle.load(fh)
    tl = Timeline.from_json(json.loads((d / "timeline.json").read_text(encoding="utf-8")))
    codes, rep = PL.label_match(model, load_target(d / "bundle", timeline=tl))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(PL.submission_bytes(np.array(LABEL_NAMES)[codes]))
    print(f"bundle {hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest()[:16]} -> {a.out} "
          f"(sha256 {hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--match", required=True)
    e.add_argument("--dir", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--match", required=True)
    c.add_argument("--dir", required=True)
    c.add_argument("--csv", required=True)
    c.add_argument("--out", default="")
    f = sub.add_parser("diff")
    f.add_argument("--gt", required=True)
    f.add_argument("csvs", nargs="+")
    lb = sub.add_parser("label")
    lb.add_argument("--bundle", required=True)
    lb.add_argument("--dir", required=True)
    lb.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd not in ("diff", "label") and os.environ.get("KATENA_WORK") is None:
        sys.exit("set KATENA_WORK to the submission cache")
    {"export": export, "compare": compare, "diff": diff, "label": label}[a.cmd](a)


if __name__ == "__main__":
    main()
