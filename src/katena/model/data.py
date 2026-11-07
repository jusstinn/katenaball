"""D4 training data: locked split, clean-match preparation, corrupted copies, feature cache.

External matches only. Nothing here reads `data/` or the competition match.

Pipeline for one clean external match (`prepare_clean` then `build_copy`):

1. Labels harmonised with D1's rule at `min_control_s = 3.5 s` (`HARMONISE`).
2. Ball gaps filled (`fill_missing_ball`). Metrica Game 3's open tracking removes the ball while it is
   dead, so "ball missing" would be an almost perfect dead-ball cue there and nowhere else. The fill
   is label-free (linear in time between the last and next seen positions, held at the ends of a
   half), so after it every clean match has a ball on every graded frame and ball loss comes only
   from D2's simulator. DFL already has the ball on 100% of frames, so the fill is a no-op there.
3. `corrupt` with a D2 config and a seed (`CopySpec`). Training copies span `severity_grid(cfg, 5)`
   with a different seed per copy; the evaluation copy uses the default config and a seed range
   disjoint from training.
4. `featurize` with a row stride (10 for training copies, 5 for the evaluation copy).

Team-swapped rows are not cached: `swap_teams` is a column permutation, applied when a training set
is assembled (`stack_training`).
"""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..external.corpus import CORPUS_NAMES, EXTERNAL_DATA, IDSSE_NAMES, cache_path, load_corpus
from ..feats import FEATURE_VERSION, featurize, swap_teams
from ..feats.core import Feats
from ..match import A, D, H, X, Match
from ..noise import DEFAULT_CFG, corrupt, resolve_cfg, severity_grid
from .baseline import starter_baseline

# -- locked holdout ------------------------------------------------------------------------------
# Rule, fixed before any model was trained: the LAST TWO DFL match ids in sorted order. They are
# never loaded for training, tuning or scoring during development (see `check_names`).
LOCKED_NAMES: tuple = tuple(sorted(IDSSE_NAMES)[-2:])
assert LOCKED_NAMES == ("idsse-J03WQQ", "idsse-J03WR9"), LOCKED_NAMES
DEV_NAMES: tuple = tuple(sorted(n for n in CORPUS_NAMES if n not in LOCKED_NAMES))

HARMONISE = {"min_control_s": 3.5}
FILL_VERSION = "linear-hold-v1"
DATA_VERSION = "d4-data-v1"

TRAIN_STRIDE = 10
EVAL_STRIDE = 5
N_TRAIN_COPIES = 5


def check_names(names, allow_locked: bool = False) -> list:
    names = list(names)
    bad = [n for n in names if n in LOCKED_NAMES]
    if bad and not allow_locked:
        raise PermissionError(f"{bad} are locked holdout matches; pass allow_locked=True only for "
                              "the one final locked-match score")
    unknown = [n for n in names if n not in CORPUS_NAMES]
    if unknown:
        raise KeyError(f"unknown external matches {unknown}")
    return names


# -- clean match preparation ---------------------------------------------------------------------

def fill_missing_ball(m: Match) -> tuple[Match, dict]:
    """Copy of `m` with every NaN ball frame inside a half filled, label-free.

    Inside a half, a gap with a seen position on both sides is filled linearly in time; a gap at the
    start (pre-kickoff) takes the first seen position, a gap at the end the last seen one. Frames
    outside the halves are left as they are. Returns (match, diagnostics) where the diagnostics are
    counts and shares only.
    """
    ball = m.ball.astype(np.float32, copy=True)
    n_filled = n_filled_dead = 0
    for _, first, last in m.half_blocks:
        b = ball[first:last + 1]
        ok = np.isfinite(b).all(axis=1)
        if ok.all() or not ok.any():
            continue
        t = np.arange(len(b), dtype=np.float64)
        miss = ~ok
        for k in range(2):
            b[miss, k] = np.interp(t[miss], t[ok], b[ok, k].astype(np.float64)).astype(np.float32)
        n_filled += int(miss.sum())
        if m.y is not None:
            n_filled_dead += int((m.y[first:last + 1][miss] == D).sum())
    out = replace(m, ball=ball, meta=dict(m.meta))
    g = m.graded_mask()
    diag = {"ball_fill": FILL_VERSION, "frames_filled": n_filled,
            "share_of_graded_filled": n_filled / max(1, int(g.sum())),
            "share_of_filled_that_is_dead": (n_filled_dead / n_filled) if n_filled else 0.0}
    out.meta["ball_fill"] = diag
    return out, diag


def prepare_clean(name: str, root=EXTERNAL_DATA, allow_locked: bool = False) -> Match:
    """Clean external match with harmonised labels (raw kept in meta['y_raw']) and the ball filled."""
    check_names([name], allow_locked)
    ms = load_corpus(root, names=[name], harmonise_cfg=HARMONISE)
    if not ms:
        raise FileNotFoundError(f"{name} is not cached under {root}; run scripts/external_corpus.py build")
    m, _ = fill_missing_ball(ms[0])
    return m


# -- corrupted copies ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CopySpec:
    """One corrupted, featurized copy of a clean match."""
    match: str
    role: str                 # "train" or "eval"
    index: int                # copy number within its plan
    cfg: dict = field(hash=False, compare=False, default_factory=dict)
    seed: int = 0
    stride: int = TRAIN_STRIDE
    offset: int = 0
    severity: float = 1.0

    def resolved_cfg(self) -> dict:
        return {k: v for k, v in resolve_cfg(self.cfg).items() if not str(k).startswith("_")}

    def key(self) -> str:
        payload = {"data": DATA_VERSION, "feats": FEATURE_VERSION, "fill": FILL_VERSION,
                   "harmonise": HARMONISE, "match": self.match, "cfg": self.resolved_cfg(),
                   "seed": int(self.seed), "stride": int(self.stride), "offset": int(self.offset)}
        s = json.dumps(payload, sort_keys=True, default=float)
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def label(self) -> str:
        return f"{self.role}{self.index}"


def _match_code(name: str) -> int:
    return zlib.crc32(name.encode()) % 100_000


def train_seed(name: str, i: int) -> int:
    return 1_000_000 + _match_code(name) * 10 + int(i)


def eval_seed(name: str) -> int:
    # disjoint from every training seed: training seeds lie in [1e6, 2e6)
    return 9_000_000 + _match_code(name)


def train_plan(name: str, aug: str = "severity", n_copies: int = N_TRAIN_COPIES,
               stride: int = TRAIN_STRIDE) -> list[CopySpec]:
    """Training copies of one match.

    aug="severity": one copy per config of `severity_grid(DEFAULT_CFG, n_copies)` (log-spaced
    severity 0.5 .. 2.0, the default config in the middle), each with its own seed.
    aug="default": the same number of copies, all with the default config and different seeds. The
    middle copy is shared with the severity plan, and row offsets match, so the two plans differ
    only in the noise configs.
    """
    grid = severity_grid(DEFAULT_CFG, n_copies)
    mid = n_copies // 2
    out = []
    for i in range(n_copies):
        off = (i * stride // n_copies) % stride
        if aug == "severity":
            cfg, seed, sev = grid[i], train_seed(name, i), float(grid[i]["_severity"])
        elif aug == "default":
            cfg, sev = (grid[mid] if i == mid else {}), 1.0
            seed = train_seed(name, mid) if i == mid else train_seed(name, n_copies + i)
        else:
            raise ValueError(f"unknown aug {aug!r}")
        out.append(CopySpec(name, "train", i, cfg, seed, stride, off, sev))
    return out


def eval_spec(name: str, stride: int = EVAL_STRIDE, severity_index: int | None = None) -> CopySpec:
    """The evaluation copy: default config. With `severity_index` (0..4 of `severity_grid(cfg, 5)`),
    a mismatched-noise evaluation copy instead, for robustness checks; its seed is also disjoint
    from every training seed."""
    if severity_index is None:
        return CopySpec(name, "eval", 0, {}, eval_seed(name), stride, 0, 1.0)
    grid = severity_grid(DEFAULT_CFG, N_TRAIN_COPIES)
    c = grid[int(severity_index)]
    return CopySpec(name, "eval", 1 + int(severity_index), c, eval_seed(name) + 100_000 * (1 + int(severity_index)),
                    stride, 0, float(c["_severity"]))


def build_copy(m_clean: Match, spec: CopySpec) -> dict:
    """corrupt + featurize. Returns {X, t_frame, y, names, spec} (y at t_frame, harmonised)."""
    mc = corrupt(m_clean, spec.cfg, seed=spec.seed)
    f = featurize(mc, {"stride": int(spec.stride), "stride_offset": int(spec.offset)})
    y = mc.y[f.t_frame].astype(np.int8)
    if np.isin(y, (X,)).any():
        raise RuntimeError("featurize emitted a row outside the graded halves")
    g = mc.graded_mask()
    out = {"X": f.X, "t_frame": f.t_frame, "y": y, "names": list(f.names),
           "ball_present_graded": float(np.isfinite(mc.ball[g, 0]).mean()),
           "half_blocks": [tuple(int(v) for v in b) for b in mc.half_blocks],
           "n_frames": int(mc.n)}
    if spec.role == "eval":
        # the starter baseline on exactly the same corrupted input, for the comparison table
        out["baseline"] = starter_baseline(mc, 50)
    return out


# -- cache ---------------------------------------------------------------------------------------

def default_cache_dir() -> Path:
    return Path(os.environ.get("KATENA_MODEL_CACHE", str(Path(EXTERNAL_DATA) / "derived" / "d4_feats")))


def _cache_file(spec: CopySpec, cache_dir) -> Path:
    return Path(cache_dir) / f"{spec.match}__{spec.label()}__{spec.key()}.npz"


def save_copy(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    extra = {"baseline": d["baseline"]} if "baseline" in d else {}
    np.savez(tmp, X=d["X"], t_frame=d["t_frame"], y=d["y"], names=np.array(d["names"]),
             info=np.array(json.dumps({"ball_present_graded": d["ball_present_graded"],
                                       "half_blocks": d["half_blocks"], "n_frames": d["n_frames"]})),
             **extra)
    os.replace(tmp, path)


def load_copy(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z["info"]))
        out = {"X": z["X"], "t_frame": z["t_frame"], "y": z["y"], "names": [str(s) for s in z["names"]],
               "ball_present_graded": info["ball_present_graded"],
               "half_blocks": [tuple(b) for b in info["half_blocks"]], "n_frames": info["n_frames"]}
        if "baseline" in z.files:
            out["baseline"] = z["baseline"]
        return out


def ensure_copies(name: str, specs: list[CopySpec], cache_dir=None, root=EXTERNAL_DATA,
                  log=print) -> list[Path]:
    """Build whatever is not cached yet for one match. Loads the clean match at most once."""
    cache_dir = Path(cache_dir or default_cache_dir())
    paths = [_cache_file(s, cache_dir) for s in specs]
    todo = [(s, p) for s, p in zip(specs, paths) if not p.exists()]
    if todo:
        m = prepare_clean(name, root)
        for s, p in todo:
            save_copy(build_copy(m, s), p)
            log(f"  built {name} {s.label()} sev {s.severity:.2f} seed {s.seed} -> {p.name}")
        del m
    return paths


def _ensure_worker(args):
    name, specs, cache_dir, root = args
    import time
    t = time.time()
    ensure_copies(name, specs, cache_dir, root, log=lambda *_: None)
    return name, time.time() - t


def ensure_all(plan: dict, cache_dir=None, root=EXTERNAL_DATA, workers: int = 1, log=print) -> None:
    """plan: {match: [CopySpec]}. Builds missing copies, one process per match (`workers` at once)."""
    cache_dir = Path(cache_dir or default_cache_dir())
    jobs = []
    for name, specs in plan.items():
        missing = [s for s in specs if not _cache_file(s, cache_dir).exists()]
        if missing:
            jobs.append((name, missing, str(cache_dir), str(root)))
    if not jobs:
        return
    log(f"building {sum(len(j[1]) for j in jobs)} corrupted copies of {len(jobs)} matches "
        f"({workers} workers) -> {cache_dir}")
    if workers <= 1:
        for j in jobs:
            name, dt = _ensure_worker(j)
            log(f"  {name}: {dt:.0f} s")
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, dt in ex.map(_ensure_worker, jobs):
            log(f"  {name}: {dt:.0f} s")


def get_copy(spec: CopySpec, cache_dir=None, root=EXTERNAL_DATA) -> dict:
    p = _cache_file(spec, cache_dir or default_cache_dir())
    if not p.exists():
        ensure_copies(spec.match, [spec], cache_dir, root, log=lambda *_: None)
    return load_copy(p)


# -- training sets -------------------------------------------------------------------------------

def swap_labels(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == H], out[y == A] = A, H
    return out


def as_feats(d: dict, name: str = "") -> Feats:
    return Feats(X=d["X"], t_frame=d["t_frame"], mask=np.ones(len(d["t_frame"]), dtype=bool),
                 names=list(d["names"]), meta={"match": name, "swapped": False})


def stack_training(copies: list[dict], swap: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """Concatenate copies (and their team-swapped versions) into one (X, y, names)."""
    names = copies[0]["names"]
    n = sum(len(c["y"]) for c in copies) * (2 if swap else 1)
    Xo = np.empty((n, len(names)), dtype=np.float32)
    yo = np.empty(n, dtype=np.int8)
    r = 0
    for c in copies:
        if c["names"] != names:
            raise ValueError("copies have different feature columns")
        k = len(c["y"])
        Xo[r:r + k], yo[r:r + k] = c["X"], c["y"]
        r += k
        if swap:
            Xo[r:r + k] = swap_teams(as_feats(c)).X
            yo[r:r + k] = swap_labels(c["y"])
            r += k
    return Xo, yo, list(names)


def full_labels(name: str, root=EXTERNAL_DATA, allow_locked: bool = False) -> tuple[np.ndarray, list]:
    """Harmonised labels over the whole timeline (X outside halves) and the half blocks."""
    check_names([name], allow_locked)
    m = load_corpus(root, names=[name], harmonise_cfg=HARMONISE)[0]
    return m.y.astype(np.int8), [tuple(int(v) for v in b) for b in m.half_blocks]


def corpus_available(names=DEV_NAMES, root=EXTERNAL_DATA) -> bool:
    return all(cache_path(n, root).exists() for n in names)
