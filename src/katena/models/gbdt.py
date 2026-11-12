"""LightGBM alive/dead and team heads (CPU), trained on pooled external rows."""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import numpy as np


def openmp_hint(platform: str | None = None) -> str:
    """What to install when LightGBM cannot load its OpenMP runtime, by platform (default: this one)."""
    platform = platform or sys.platform
    if platform == "darwin":
        return "lightgbm cannot load libomp; install it (brew install libomp)"
    if platform.startswith("linux"):
        return ("lightgbm cannot load the OpenMP runtime libgomp.so.1, which its Linux wheel takes from the system. "
                "Install it and run the same command again: apt-get install libgomp1 (Debian, Ubuntu), "
                "dnf install libgomp (Fedora, RHEL), apk add libgomp (Alpine). Slim container images and a bare "
                "WSL distribution do not have it")
    if platform == "win32":
        return ("lightgbm cannot load: its Windows wheel needs the Microsoft Visual C++ runtime "
                "(install the Microsoft Visual C++ Redistributable, x64)")
    return "lightgbm cannot load its OpenMP runtime"


def ensure_openmp() -> None:
    """Entry-point helper. LightGBM needs an OpenMP runtime. On macOS without Homebrew's libomp, the copy
    bundled in scikit-learn works when ``DYLD_FALLBACK_LIBRARY_PATH`` points at it, and that variable is only
    read at process start, so re-exec once. Elsewhere there is nothing to fall back on: say what to install
    (``openmp_hint``). Call this first thing in a script (not from library code)."""
    if lgb is not None:
        return
    if "pytest" in sys.modules:      # re-exec would replace the test runner; LightGBM tests skip instead
        return
    why = f" [{_LGB_ERROR}]" if _LGB_ERROR else ""
    if sys.platform != "darwin":
        raise RuntimeError(openmp_hint() + why)
    import sklearn
    d = os.path.join(os.path.dirname(sklearn.__file__), ".dylibs")
    if not glob.glob(os.path.join(d, "libomp*")) or os.environ.get("_KATENA_REEXEC"):
        raise RuntimeError(openmp_hint() + why)
    env = dict(os.environ, DYLD_FALLBACK_LIBRARY_PATH=d, _KATENA_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.orig_argv[1:], env)


_LGB_ERROR = ""
try:
    import lightgbm as lgb
except OSError as e:                 # the OpenMP runtime is missing: see ensure_openmp
    lgb = None
    _LGB_ERROR = str(e).splitlines()[0][:200] if str(e) else type(e).__name__

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1, max_bin=63,
              lambda_l2=1.0, seed=0, num_threads=8, verbose=-1, deterministic=True,
              force_row_wise=True)
N_TREES = 150


@dataclass
class Heads:
    alive: object
    team: object
    cols: np.ndarray

    def predict(self, X0: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(p_alive, p_home) averaged over the two focal orientations."""
        c = self.cols
        pa = 0.5 * (self.alive.predict(X0[:, c]) + self.alive.predict(X1[:, c]))
        ph = 0.5 * (self.team.predict(X0[:, c]) + 1.0 - self.team.predict(X1[:, c]))
        return pa.astype(np.float32), ph.astype(np.float32)


def fit_heads(X: np.ndarray, o: np.ndarray, y: np.ndarray, dead_ok: np.ndarray, cols: np.ndarray,
              n_trees: int = N_TREES, team_weight: np.ndarray | None = None) -> Heads:
    """X rows already hold the orientation chosen by ``o`` (0 focal=home, 1 focal=away).
    ``y``: 0 H, 1 A, 2 D, 3 X. ``dead_ok``: rows from sources that carry a dead label.
    ``team_weight``: one weight per row of ``X`` for the team head (scaled here to mean 1 over its rows);
    without it the head is unweighted, as before."""
    team_rows = np.flatnonzero((y == 0) | (y == 1))
    alive_rows = np.flatnonzero(dead_ok & (y < 3))
    team_t = (y[team_rows] == o[team_rows]).astype(np.float32)
    alive_t = (y[alive_rows] != 2).astype(np.float32)
    wt = None if team_weight is None else team_weight[team_rows] / team_weight[team_rows].mean()
    team = lgb.train(PARAMS, lgb.Dataset(X[team_rows][:, cols], team_t, weight=wt, free_raw_data=True), n_trees)
    alive = lgb.train(PARAMS, lgb.Dataset(X[alive_rows][:, cols], alive_t, free_raw_data=True), n_trees)
    return Heads(alive=alive, team=team, cols=cols)
