"""Training labels of the SkillCorner matches from their own action events (candidates A and AW), in the form
the training code reads: the per-match label file, the LightGBM pool's labels, the TCN sequences' labels, and the
possession-length weights of AW.

The rule is ``katena.external.skillcorner_actions`` (decision log, 2026-11-12, "A (pre-registration)"). This
module only carries its labels to the rows the members train on. ``scripts/run_a.py`` and ``scripts/run_aw.py``
(the gates) and ``katena.pipeline.stage_train`` (the final model) must train on the same labels, so both take
them from here. External matches only.

* An IDSSE or Metrica match is never touched: ``load`` returns its standing truth.
* A SkillCorner match gets the action labels, then the sandwich merge of every other label (``labels.relabel``),
  and counts as a source with a dead-ball label (``dead_ok``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from katena.eval import corpus as C
from katena.eval import d2
from katena.eval import final as F
from katena.eval import protocol as P
from katena.external import skillcorner_actions as SA
from katena.labels import relabel
from katena.models import lenweight as LW
from katena.types import load_match

T = P.T_MAIN
DIR = "gt_act"


def raw_dir(r: P.Ref) -> Path:
    """The raw SkillCorner folder of a match (``scripts/build_external.py`` puts the pinned repository there)."""
    return P.EXTERNAL / "skillcorner" / "repo" / "data" / "matches" / r.id


def path(r: P.Ref) -> Path:
    return P.WORK / DIR / f"{r.dataset}_{r.id}.npz"


def build(r: P.Ref) -> None:
    """The action labels of one SkillCorner match under every T, in the layout of ``corpus``'s ground-truth file.
    Skipped when the file exists."""
    out = path(r)
    if out.exists():
        return
    m = load_match(r.path)
    m2 = dataclasses.replace(m, y=SA.build_labels(raw_dir(r), m))
    gt = {"fps": m.fps, "step": int(round(m.fps / 5.0)), "n_native": m.n,
          "blocks": np.array(m.half_blocks, dtype=np.int64), "y_raw": m2.y}
    for t in P.T_SWEEP:
        gt[f"y_T{t}"] = relabel(m2, t).y
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **gt)
    tmp.replace(out)


def build_all(refs: list) -> int:
    """Build the label file of every SkillCorner match in ``refs``; returns how many matches that is."""
    sc = [r for r in refs if r.is_sc]
    for r in sc:
        build(r)
    return len(sc)


def load(r: P.Ref, t: int = T) -> dict:
    """The labels a match trains on: the action labels of a SkillCorner match, the one truth of any other."""
    if not r.is_sc:
        return C.load_gt(r, t)
    with np.load(path(r)) as z:
        return {"y": z[f"y_T{t}"], "step": int(z["step"]), "fps": float(z["fps"]), "n_native": int(z["n_native"]),
                "blocks": z["blocks"]}


def pool_y(refs: list, pools: F.FinalPools, t: int = T) -> np.ndarray:
    """``pools.y[t]`` with the SkillCorner rows relabelled. A pool row's label is the truth at its work-grid
    sample, so with the standing truth this gives ``pools.y[t]`` back. ``refs`` must be the pool's matches, in
    its order."""
    y, off = pools.y[t].copy(), 0
    for r in refs:
        idx = d2.load_pool(r, F.MIX)["idx"]
        if r.is_sc:
            g = load(r, t)
            y[off:off + len(idx)] = g["y"][::g["step"]][idx]
        off += len(idx)
    if off != len(y):
        raise RuntimeError("pool rows do not line up with the matches")
    return y


def _seq_idx(r: P.Ref, world: str, n: int) -> np.ndarray:
    with np.load(C.paths(r)["feat"](world)) as z:
        idx = z["idx"]
    if len(idx) != n:
        raise RuntimeError(f"{r.key} {world}: sequence rows do not line up")
    return idx


def relabel_seqs(refs: list, seqs: dict, t: int = T) -> int:
    """Give the SkillCorner sequences of ``seqs`` (``final.tcn_seqs`` / ``g1.tcn_seqs``: keyed (match, world))
    their action labels and ``dead_ok``, in place. Returns how many sequences changed."""
    by_key = {r.key: r for r in refs}
    n = 0
    for (key, world), s in seqs.items():
        r = by_key[key]
        if not r.is_sc:
            continue
        g = load(r, t)
        s["y"], s["dead_ok"] = g["y"][::g["step"]][_seq_idx(r, world, len(s["y"]))], True
        n += 1
    return n


def native_weights(refs: list, t: int = T) -> dict:
    """match key -> (``lenweight.inv_length`` of the label the match trains on, for every native frame; the
    5 Hz step). AW's weights."""
    out = {}
    for r in refs:
        g = load(r, t)
        out[r.key] = (LW.inv_length(g["y"], g["fps"]), g["step"])
    return out


def pool_weights(refs: list, wn: dict) -> np.ndarray:
    """One weight per row of ``FinalPools(refs)`` (same matches, same order)."""
    return np.concatenate([LW.at_rows(wn[r.key][0], d2.load_pool(r, F.MIX)["idx"], wn[r.key][1]) for r in refs])


def weight_seqs(refs: list, seqs: dict, wn: dict) -> None:
    """Put each sequence's per-row weight of the team loss under ``w`` (float32), in place."""
    by_key = {r.key: r for r in refs}
    for (key, world), s in seqs.items():
        idx = _seq_idx(by_key[key], world, len(s["y"]))
        s["w"] = LW.at_rows(wn[key][0], idx, wn[key][1]).astype(np.float32)
