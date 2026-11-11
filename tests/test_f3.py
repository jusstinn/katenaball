"""F3's second-machine gate (katena.eval.f3, scripts/run_f3.py) on a tiny synthetic corpus: the larger LightGBM
member continues F's, the TCN stage equals ``g1.stage_tcn`` whichever process trains a fold, and the report follows
the registered rule. No external data."""

from __future__ import annotations

import importlib.util
import os
import shutil
import threading
from pathlib import Path

import pytest
import test_final as TFN
import test_g1 as TG1

from katena.eval import f3 as F3
from katena.eval import final as F
from katena.eval import g1 as G
from katena.eval import lomo as Lm
from katena.eval import protocol as P
from katena.models import ensemble as E
from katena.sim import worlds as W

synth_final = TFN.synth_final
tiny_tcn = TFN.tiny_tcn


def _probs(variant: str, r) -> dict:
    return {k: v.tobytes() for k, v in Lm.load_oof(variant, P.T_MAIN, r).items() if k[:3] in ("pa_", "ph_")}


def test_f3_gate_end_to_end(synth_final, tiny_tcn, tmp_path, monkeypatch):
    refs = synth_final
    T = P.T_MAIN
    TG1._f_cache(refs)
    f_root = tmp_path / "f_cache"                       # the F cache, read only from here on
    shutil.copytree(P.WORK, f_root)
    monkeypatch.setattr(P, "WORK", tmp_path / "f3")
    monkeypatch.setattr(Lm, "OOF_DIR", tmp_path / "f3" / "oof")
    monkeypatch.setattr(E, "TCN_THREADS", 2)
    monkeypatch.setattr(F3, "N_TREES", 170)             # more than F's 150, small enough for a test
    G.seed_cache(refs, f_root)

    # the larger member: its first 150 trees are F's member, the full model is another one
    info = F3.stage_gbdt(F.FinalPools(refs), refs)
    assert info == {"folds": 3, "first150_is_F": 3, "of": 3}
    assert all(_probs(F3.VAR_G, r) != _probs(F.VAR_G, r) for r in refs)
    assert F3.stage_gbdt(F.FinalPools(refs), refs)["folds"] == 0                   # cached

    # the TCN stage: a claimed fold is left to its owner, and the stage waits until the claim is gone
    claim = Lm.oof_path(F3.VAR_T, T, refs[1]).with_suffix(".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.write_bytes(b"")
    threading.Timer(1.0, claim.unlink).start()
    info = F3.stage_tcn(refs, refs, steps=2, wait_s=0.2)
    assert info["folds"] == 2 and not claim.exists() and not Lm.oof_path(F3.VAR_T, T, refs[1]).exists()
    assert F3.stage_tcn(refs, refs, steps=2, reverse=True, wait_s=0.2)["folds"] == 1   # the other end takes it
    assert F3.stage_tcn(refs, refs, steps=2) == {"folds": 0}
    # every fold equals g1.stage_tcn's, whichever call trained it
    G.stage_tcn(refs, refs, threads=2, steps=2, variant="chk")
    for r in refs:
        a, b = Lm.load_oof(F3.VAR_T, T, r), Lm.load_oof("chk", T, r)
        assert str(a["model_sha"]) == str(b["model_sha"]) and int(a["n_in"]) == int(b["n_in"])
        assert _probs(F3.VAR_T, r) == _probs("chk", r)

    store = F3.f_store()
    res = F3.results()
    chk = F3.stage_score(res, refs, refs, store)
    n = 3 * len(W.EVAL_WORLDS)
    assert chk == {"n": n, "F_equal": n, "FG_equal": n, "FT_equal": n}             # F re-decoded = F stored
    get = F.getter(res, store)
    for r in refs:
        assert all(get(m, "target", r) is not None for m in list(F3.BLENDS) + ["F", "FG", "FT"])

    for k in E.CPU_ENV:                                   # the script's import pins CPU_ENV: restored afterwards
        old = os.environ.get(k)
        monkeypatch.setenv(k, old if old is not None else "")
        if old is None:
            monkeypatch.delenv(k)
    spec = importlib.util.spec_from_file_location("run_f3", Path(__file__).resolve().parents[1] / "scripts" / "run_f3.py")
    run_f3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_f3)
    out = run_f3.gate_report(refs)
    g = out["gate"]
    assert out["registered_run"] is False and g["all/target"]["n"] == 3 and (P.WORK / "f3_gate.json").exists()
    assert out["f3_passes"] == F3.passes(g["all/target"], g["all/severe"], g.get("SkillCorner/target"))
    assert {"T1200-F/all/target", "LT-F/all/target", "F3G-FG/all/target", "world/target"} <= set(out["info"])


@pytest.mark.parametrize("adopted, worse_severe, worse_sc, want", [
    (True, False, False, True), (True, False, None, True), (False, False, False, False),
    (True, True, False, False), (True, False, True, False)])
def test_the_registered_rule(adopted, worse_severe, worse_sc, want):
    sc = None if worse_sc is None else {"worse": worse_sc}
    assert F3.passes({"adopted": adopted}, {"worse": worse_severe}, sc) is want
