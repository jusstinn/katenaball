#!/usr/bin/env python3
"""Regenerate the submission: train the submission model (F3) on the 30 external matches, then label the real match.

    uv run python scripts/make_submission.py all  --out submissions/<team>.csv   # every stage, in order
    uv run python scripts/make_submission.py corpus [--seed-from work/bakeoff]  # external corpus features
    uv run python scripts/make_submission.py worlds [--workers N]               # d2j worlds and touch columns
    uv run python scripts/make_submission.py train                              # model_F3/bundle.pkl + manifest
    uv run python scripts/make_submission.py predict --out submissions/<team>.csv

Needs the external corpus (``scripts/build_external.py``) and, for ``predict``, the data bundle in ``data/``.
The cache is ``KATENA_WORK`` (default ``work/final``). ``predict`` prints aggregates only: counts, shares,
shapes, hashes and run times of the real match, never a frame-level value. See ``katena.pipeline``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("KATENA_WORK", str(REPO / "work" / "final"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]

from katena.models import ensemble as E  # noqa: E402

E.pin_cpu_env()                                    # before torch is imported
from katena.models import gbdt  # noqa: E402

gbdt.ensure_openmp()                               # macOS without Homebrew libomp: re-exec once

from katena import pipeline as PL  # noqa: E402
from katena.eval import protocol as P  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("all", "corpus", "worlds", "train", "predict"))
    ap.add_argument("--out", type=Path, default=REPO / "submissions" / "submission.csv")
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="a finished bake-off cache to copy the corpus files from (default: build them)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the corpus and worlds stages")
    ap.add_argument("--data-dir", type=Path, default=None, help="the data bundle (default: data/)")
    ap.add_argument("--spec", choices=sorted(PL.SPECS), default=PL.SUBMISSION_SPEC,
                    help=f"the model to train or predict with (default: the submission model, {PL.SUBMISSION_SPEC})")
    chk = ap.add_argument_group("dressed-match check only (scripts/dressed_check.py), never for the submission")
    chk.add_argument("--hold-out", action="append", default=[], metavar="KEY",
                     help="train / predict with the model that leaves this external match out (model_without_*/)")
    chk.add_argument("--timeline", type=Path, default=None,
                     help="predict: the exported match's timeline JSON in place of games.py's")
    a = ap.parse_args(argv)
    t0 = time.time()
    PL.log(f"cache {P.WORK}; spec {a.spec}; env {json.dumps(PL.env_report(), sort_keys=True)}")
    refs = PL.training_refs() if a.stage != "predict" else None
    if a.stage in ("all", "corpus"):
        PL.log(f"corpus {PL.stage_corpus(refs, a.seed_from, a.workers)}")
    if a.stage in ("all", "worlds"):
        st = PL.stage_worlds(refs, a.workers)
        PL.log(f"worlds: {len(st)} matches, {sum(s['touch_built'] for s in st)} touch files built")
    if a.stage in ("all", "train"):
        PL.log(f"train {json.dumps(PL.stage_train(refs, hold_out=a.hold_out, spec=a.spec), sort_keys=True)}")
    if a.stage in ("all", "predict"):
        try:
            timeline = None
            if a.timeline is not None:
                from katena.target.atd import Timeline
                timeline = Timeline.from_json(json.loads(a.timeline.read_text(encoding="utf-8")))
            rep = PL.stage_predict(a.out, a.data_dir, hold_out=a.hold_out, timeline=timeline, spec=a.spec)
        except Exception as e:     # an exception message can quote the match file: type and code lines only
            import traceback
            where = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(e.__traceback__)]
            print(f"predict failed: {type(e).__name__} at " + " <- ".join(reversed(where)), flush=True)
            return 1
        print(json.dumps(rep, indent=1, sort_keys=True), flush=True)
    PL.log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
