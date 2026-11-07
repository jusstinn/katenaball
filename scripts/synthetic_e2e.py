#!/usr/bin/env python3
"""Synthetic end-to-end run of the locked M2 baseline, for CI's determinism check.

No match data and no external data. Synthetic matches go through the real modules: katena.sim.degrade
(mild, target-like and severe noise) -> katena.featurize -> katena.models.gbdt heads -> katena.decode.hmm
Viterbi -> labels -> a frame,label CSV in the submission format, with the real match's row count and
LF line endings on every OS.

    uv run python scripts/synthetic_e2e.py run --out DIR [--label NAME]   # DIR/submission.csv, DIR/hashes.json
    python scripts/synthetic_e2e.py compare A.json B.json                 # exit 1 unless every hash matches
    python scripts/synthetic_e2e.py compare --report A.json B.json ...    # table across OSes; never fails

CI runs it twice in separate processes on each OS and fails if the two runs differ, then compares the
OSes as an early warning for the two-machine hash test (docs/PLAN.md, J3). The compare command needs
only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TRAIN_SEEDS = tuple(range(6))
TARGET_SEED = 1000
T_LABEL = 2             # locked baseline: merge opponent flips under 2 s (docs/PLAN.md)
TAU = 2.0               # locked Viterbi temperature
POOL_STRIDE = 4         # as katena.eval.corpus builds its training pool
FPS = 25
PITCH = (105.0, 68.0)
HALF_S = 150.0
# Home's shape when defending the left goal (attacking +x); away is the 180-degree mirror.
FORMATION = ((4.0, 34.0), (16.0, 10.0), (16.0, 26.0), (16.0, 42.0), (16.0, 58.0), (33.0, 12.0),
             (33.0, 28.0), (33.0, 40.0), (33.0, 56.0), (46.0, 24.0), (46.0, 44.0))


# -- synthetic matches -------------------------------------------------------------------------------

def _label_runs(rng, n: int):
    """H/A/D runs with log-normal lengths: possessions of a few seconds, stoppages between some."""
    import numpy as np
    out = np.empty(n, dtype=np.int8)
    i, state = 0, int(rng.integers(0, 2))
    while i < n:
        seconds = rng.lognormal(np.log(6.0), 0.5) if state == 2 else rng.lognormal(np.log(7.0), 0.9)
        k = max(1, int(min(max(seconds, 0.4), 60.0) * FPS))
        out[i:i + k] = state
        i += k
        state = int(rng.integers(0, 2)) if state == 2 else (2 if rng.random() < 0.3 else 1 - state)
    return out


def synth_match(seed: int, n_frames: int = 0):
    """Two halves of HALF_S seconds at 25 fps: 11 v 11 players drifting around a formation, the block
    shifting towards the goal being attacked, a ball that is carried and passed within the team in
    possession, and stopped off the pitch (or untracked) while dead. Frames outside the halves are X."""
    import numpy as np

    from katena.external.common import derive_speed
    from katena.types import Match

    rng = np.random.default_rng(seed)
    L, W = PITCH
    pre, gap, nh = 4 * FPS, 12 * FPS, int(HALF_S * FPS)
    blocks = [(1, pre, pre + nh - 1), (2, pre + nh + gap, pre + 2 * nh + gap - 1)]
    n = max(n_frames, blocks[-1][2] + 1 + 4 * FPS)
    S = 22
    team = np.repeat(np.array([0, 1], dtype=np.int8), 11)
    P = np.full((n, S, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    y = np.full(n, 3, dtype=np.int8)
    base = np.array(FORMATION)
    for half, a, b in blocks:
        m = b - a + 1
        lab = _label_runs(rng, m)
        y[a:b + 1] = lab
        t = np.arange(m) / FPS
        attack = np.where(lab == 0, 1.0, np.where(lab == 1, -1.0, 0.0))
        push = np.convolve(attack, np.ones(4 * FPS) / (4 * FPS), mode="same")
        xy = np.empty((m, S, 2))
        for s in range(S):
            fx, fy = base[s % 11]
            if s >= 11:
                fx, fy = L - fx, W - fy
            phase, freq = rng.uniform(0, 2 * np.pi, 4), rng.uniform(0.01, 0.08, 4)
            wave = [3.0 * np.sin(2 * np.pi * freq[k] * t + phase[k]) for k in range(4)]
            xy[:, s, 0] = fx + 14.0 * push + wave[0] + 0.5 * wave[1] + rng.normal(0, 0.05, m)
            xy[:, s, 1] = fy + wave[2] + 0.5 * wave[3] + rng.normal(0, 0.05, m)
        bxy = np.full((m, 2), np.nan)
        edges = np.flatnonzero(np.diff(lab)) + 1
        for start, stop in zip(np.r_[0, edges], np.r_[edges, m]):
            if lab[start] == 2:
                prev = bxy[start - 1] if start and np.isfinite(bxy[start - 1, 0]) else np.array([L / 2, W / 2])
                spot = (prev[0], -0.8 if prev[1] < W / 2 else W + 0.8)     # out over the nearer touchline
                bxy[start:stop] = spot if rng.random() < 0.6 else np.nan
                continue
            mates = np.arange(11) if lab[start] == 0 else np.arange(11, 22)
            carrier, k = int(rng.choice(mates)), start
            while k < stop:
                hold = min(stop, k + int(rng.uniform(1.0, 4.0) * FPS))
                bxy[k:hold] = xy[k:hold, carrier] + 0.4
                receiver = int(rng.choice(mates[mates != carrier]))
                land = min(stop, hold + int(0.6 * FPS))
                if land > hold:
                    w = np.linspace(0.0, 1.0, land - hold + 2)[1:-1, None]
                    bxy[hold:land] = (1 - w) * xy[hold, carrier] + w * xy[land - 1, receiver]
                carrier, k = receiver, land
        for _ in range(int(0.08 * m / (0.6 * FPS))):                       # tracking dropouts
            s0 = int(rng.integers(0, m))
            bxy[s0:s0 + max(1, int(rng.lognormal(np.log(0.6), 0.6) * FPS))] = np.nan
        if half == 2:                                                       # home attacks -x after the break
            xy[..., 0], xy[..., 1] = L - xy[..., 0], W - xy[..., 1]
            bxy[:, 0], bxy[:, 1] = L - bxy[:, 0], W - bxy[:, 1]
        P[a:b + 1] = xy
        ball[a:b + 1] = bxy
    match = Match(P=P, team=team, speed=derive_speed(P, FPS), ball=ball, half_blocks=blocks,
                  attack_dir={1: 1, 2: -1}, y=y, pitch=PITCH, fps=float(FPS), name="synthetic-{}".format(seed),
                  source="synthetic", meta={"label_source": "synthetic"})
    match.validate()
    return match


# -- the pipeline --------------------------------------------------------------------------------------

def _digest(*arrays) -> str:
    """sha256 over arrays as little-endian bytes, with every NaN in one canonical bit pattern."""
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        if a.dtype.kind == "f":
            a = np.where(np.isnan(a), np.array(np.nan, dtype=a.dtype), a)
        h.update(str(a.dtype.str).encode() + str(a.shape).encode())
        h.update(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


def submission_bytes(labels) -> bytes:
    """The submission file, byte for byte: 'frame,label' and one LF-terminated row per frame.
    Built as bytes and written in binary mode, so no OS can turn the LF into CRLF."""
    return ("frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(labels))).encode("ascii")


def run(out_dir: Path, label: str) -> dict:
    sys.path[:0] = [str(REPO / "src"), str(REPO / "evaluation"), str(REPO)]
    from katena.models import gbdt
    gbdt.ensure_openmp()                     # macOS without Homebrew libomp: re-exec once

    import lightgbm
    import numba
    import numpy as np

    import games
    import possession_metrics as pm
    from katena.decode import hmm
    from katena.eval.score import to_native_codes
    from katena.featurize.build import featurize, infer_dirs
    from katena.grid import downsample, mask_outside_blocks
    from katena.labels import relabel
    from katena.sim import degrade as dg
    from katena.types import LABEL_NAMES

    t0 = time.time()
    conds = (dg.MILD, dg.TARGET, dg.SEVERE)
    rng = np.random.default_rng(12345)
    pool_X, pool_o, pool_y, label_seqs = [], [], [], []
    for seed in TRAIN_SEEDS:
        m = synth_match(seed)
        m5 = downsample(m)
        y5 = relabel(m, T_LABEL).y[::m5.meta["step"]]
        label_seqs.append(y5)
        for k, cfg in enumerate(conds):
            mc = dg.corrupt(m5, cfg, 100 * seed + k)
            dirs = infer_dirs(mc)
            f0, f1 = featurize(mc, 0, dirs), featurize(mc, 1, dirs)
            sel = np.flatnonzero((np.arange(len(f0.idx)) + k) % POOL_STRIDE == 0)
            o = rng.integers(0, 2, len(sel)).astype(np.int8)
            pool_X.append(np.where(o[:, None] == 1, f1.X[sel], f0.X[sel]))
            pool_o.append(o)
            pool_y.append(y5[f0.idx[sel]])
    X, o, y = np.concatenate(pool_X), np.concatenate(pool_o), np.concatenate(pool_y)
    heads = gbdt.fit_heads(X, o, y, np.ones(len(y), dtype=bool), np.arange(X.shape[1]))
    priors = hmm.fit_priors(label_seqs)
    t_train = time.time() - t0

    n_frames = games.GAMES[games.GAME]["n_video_frames"]
    target = synth_match(TARGET_SEED, n_frames=n_frames)
    t5 = downsample(target)
    tc = dg.corrupt(t5, dg.TARGET, 7)
    dirs = infer_dirs(tc)
    f0, f1 = featurize(tc, 0, dirs), featurize(tc, 1, dirs)
    p_alive, p_home = heads.predict(f0.X, f1.X)
    rows = hmm.decode(p_alive, p_home, f0.half, priors, TAU, "viterbi")
    codes = mask_outside_blocks(to_native_codes(rows, f0.idx, t5.meta["step"], target.n), target)
    labels = np.array(LABEL_NAMES)[codes]

    out_dir.mkdir(parents=True, exist_ok=True)
    data = submission_bytes(labels)
    csv_path = out_dir / "submission.csv"
    csv_path.write_bytes(data)
    reason = pm.validate_submission(pm.read_label_csv(csv_path), n_frames)
    if reason:
        raise SystemExit("synthetic submission is invalid: {}".format(reason))
    truth = relabel(target, T_LABEL).y
    result = pm.evaluate(truth, codes)
    inside = np.zeros(target.n, dtype=bool)
    for _, a, b in target.half_blocks:
        inside[a:b + 1] = True

    models = (heads.alive.model_to_string() + heads.team.model_to_string()).encode()
    record = {
        "sha256": {
            "submission_csv": hashlib.sha256(data).hexdigest(),
            "decoded_rows": _digest(rows),
            "probabilities": _digest(p_alive, p_home),
            "models": hashlib.sha256(models).hexdigest(),
            "training_pool": _digest(X, o, y),
            "target_features": _digest(f0.X, f1.X),
        },
        "env": {
            "label": label, "os": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__,
            "lightgbm": lightgbm.__version__, "numba": numba.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"),
        },
        "summary": {
            "rows": int(len(labels)), "training_rows": int(len(y)), "features": int(X.shape[1]),
            "label_counts": {name: int((codes == i).sum()) for i, name in enumerate(LABEL_NAMES)},
            "x_on_synthetic_graded_frames": int(((codes == 3) & inside & (truth < 3)).sum()),
            "spell_f1_vs_synthetic_truth": round(float(result["spell_f1_mean"]), 6),
            "macro_f1_vs_synthetic_truth": round(float(result["macro_f1"]), 6),
            "seconds_train": round(t_train, 1), "seconds_total": round(time.time() - t0, 1),
        },
    }
    (out_dir / "hashes.json").write_bytes((json.dumps(record, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return record


# -- comparing runs --------------------------------------------------------------------------------------

def compare(paths, report: bool = False, summary_path: str = "") -> int:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [r["env"]["label"] for r in runs]
    keys = sorted(set().union(*(r["sha256"] for r in runs)))
    differ = [k for k in keys if len({r["sha256"].get(k) for r in runs}) > 1]
    width = max([18] + [len(n) + 2 for n in names])
    print("{:<18}".format("sha256") + "".join("{:<{w}}".format(n, w=width) for n in names) + "same")
    for k in keys:
        cells = [(r["sha256"].get(k) or "-")[:16] for r in runs]
        print("{:<18}".format(k) + "".join("{:<{w}}".format(c, w=width) for c in cells)
              + ("yes" if k not in differ else "NO"))
    for field in ("os", "machine", "python", "numpy", "lightgbm", "numba"):
        print("{:<18}".format(field) + "".join("{:<{w}}".format(str(r["env"].get(field)), w=width) for r in runs))
    print("{:<18}".format("spellF1 (synth)") + "".join(
        "{:<{w}}".format(str(r["summary"]["spell_f1_vs_synthetic_truth"]), w=width) for r in runs))
    if summary_path:
        lines = ["### Synthetic determinism hashes", "", "| sha256 | " + " | ".join(names) + " | same |",
                 "|---|" + "---|" * len(names) + "---|"]
        lines += ["| {} | {} | {} |".format(k, " | ".join("`{}`".format((r["sha256"].get(k) or "-")[:16]) for r in runs),
                                           "yes" if k not in differ else "**no**") for k in keys]
        with open(summary_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n\n")
    if not differ:
        print("\nall {} hashes identical across {} run(s)".format(len(keys), len(runs)))
        return 0
    print("\n{} of {} hashes differ: {}".format(len(differ), len(keys), ", ".join(differ)))
    if report:
        if os.environ.get("GITHUB_ACTIONS"):
            print("::warning title=Cross-OS hashes differ::{} differ across {} (non-blocking early warning "
                  "for the two-machine hash test)".format(", ".join(differ), ", ".join(names)))
        return 0
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the synthetic pipeline once")
    p_run.add_argument("--out", required=True, type=Path, help="directory for submission.csv and hashes.json")
    p_run.add_argument("--label", default="{}-{}".format(platform.system(), platform.machine()),
                       help="name for this run in comparisons (CI passes the runner OS)")
    p_cmp = sub.add_parser("compare", help="compare hashes.json files")
    p_cmp.add_argument("files", nargs="+")
    p_cmp.add_argument("--report", action="store_true", help="print and warn, but exit 0 (cross-OS)")
    p_cmp.add_argument("--summary", default="", help="append a markdown table here (GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare(args.files, args.report, args.summary)
    record = run(args.out, args.label)
    s = record["summary"]
    print("synthetic e2e: {rows} rows, spellF1 {spell_f1_vs_synthetic_truth} vs synthetic truth, "
          "{seconds_total} s".format(**s))
    for k, v in sorted(record["sha256"].items()):
        print("  {:<16} {}".format(k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
