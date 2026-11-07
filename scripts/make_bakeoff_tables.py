#!/usr/bin/env python
"""Regenerate the AUTO tables block of docs/bakeoff.md from work/bakeoff/results.pkl."""

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation")]

from katena.eval import bakeoff as B  # noqa: E402
from katena.eval import lomo as Lm  # noqa: E402
from katena.eval import protocol as P  # noqa: E402
from katena.eval import report as R  # noqa: E402
from katena.eval import stages as S  # noqa: E402

dev, locked = B.dev_and_locked()
res = Lm.Results()
census = pickle.load(open(P.WORK / "census.pkl", "rb"))
tables = R.build_tables(res, dev, locked, census, B.ablation_folds(dev), S.FOLDS8(dev))
(P.WORK / "tables.md").write_text(tables)
R.splice(REPO / "docs" / "bakeoff.md", tables)
print(tables)
