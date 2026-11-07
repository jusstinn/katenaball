"""D3: TCN + LightGBM ensemble, deterministic on CPU.

Both members give (p_alive, p_home) per 5 Hz sample: the M2 LightGBM heads (``katena.models.gbdt``)
and the M5 TCN (``katena.models.tcn``, used unchanged). The ensemble averages them in logit space, for
each head separately,

    logit p = (1 - w) * logit p_gbdt + w * logit p_tcn

and decodes with the unchanged 3-state Viterbi (``katena.decode.hmm``). ``w = 0`` returns the LightGBM
probabilities bit for bit, so it decodes exactly like M2, and ``w = 1`` returns the TCN's. The gated
candidate uses ``W_MAIN = 0.5``, fixed before any result, so no ensemble weight is fitted on any match.

Determinism: ``fit_tcn`` turns on deterministic algorithms, fixes the intra-op thread count, and
seeds torch; the batch order comes from the seeded NumPy generator inside ``tcn.train``. Results also
depend on the CPU kernels torch picks, so ``pin_cpu_env`` fixes those when called before torch is
imported. ``scripts/d3_determinism.py`` trains and predicts in separate processes and compares the
SHA-256 of every output: on one machine every byte repeats. Other thread counts or kernels change the
probabilities in about the sixth decimal and, on the folds probed, no decoded label (decision log,
2026-11-07). ``quantize`` / ``quantize_logit`` round probabilities before decoding, for that experiment
only: rounding cannot undo a different training run, and the decode is already insensitive to last bits.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

EPS = 1e-4                       # same clip as katena.decode.hmm.emissions
W_MAIN = 0.5                     # TCN weight of the gated candidate (fixed a priori)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TCN_THREADS = 4                  # torch intra-op threads; a different count gives different bits
QBITS = 16                       # quantize: probabilities rounded to multiples of 2**-QBITS

# Environment that fixes which CPU kernels torch and MKL use. It must be in the environment before
# torch is imported (oneDNN, MKL and ATen read it once). AVX2 is this machine's ISA, so pinning it
# changes nothing here; it keeps an AVX-512 x86-64 machine on the same kernel family. On Apple silicon
# these variables do nothing: other kernels run there.
CPU_ENV = {"ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2", "MKL_CBWR": "AVX2",
           "OMP_NUM_THREADS": str(TCN_THREADS), "MKL_NUM_THREADS": str(TCN_THREADS)}


# ---------------------------------------------------------------------------------------------
# blending

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def blend(p_gbdt: np.ndarray, p_tcn: np.ndarray, w: float) -> np.ndarray:
    """Logit-space average with TCN weight ``w``. The end points return the member unchanged."""
    if w == 0.0:
        return np.asarray(p_gbdt, dtype=np.float32)
    if w == 1.0:
        return np.asarray(p_tcn, dtype=np.float32)
    z = (1.0 - w) * logit(p_gbdt) + w * logit(p_tcn)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def blend_heads(gbdt: tuple, tcn: tuple, w: float) -> tuple:
    """``gbdt``, ``tcn``: (p_alive, p_home) of each member. Returns the blended pair."""
    return blend(gbdt[0], tcn[0], w), blend(gbdt[1], tcn[1], w)


def quantize(p: np.ndarray, bits: int = QBITS) -> np.ndarray:
    """Round to the nearest multiple of 2**-bits (exact in binary, ties to even)."""
    s = float(2 ** bits)
    return (np.rint(np.asarray(p, dtype=np.float64) * s) / s).astype(np.float32)


def quantize_logit(p: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Round logit(p) to a multiple of ``step`` (a power of two keeps the grid exact), then map back.
    The relative error of the odds is at most ``exp(step / 2) - 1`` everywhere, unlike ``quantize``,
    which is coarse near 0 and 1."""
    z = np.rint(logit(p) / step) * step
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# nested selection helpers (grouped folds)

def choose(table: np.ndarray, grid: list):
    """``table`` [len(grid), n]: a score per grid entry and inner match. Returns the entry with the best
    mean. The first entry wins ties, so order ``grid`` from the incumbent outwards."""
    return grid[int(np.argmax(np.asarray(table, dtype=np.float64).mean(1)))]


def pairs(n_groups: int) -> list:
    """Unordered group pairs. A model trained without groups g and h gives inner out-of-fold
    predictions on g for outer group h and on h for outer group g, so one fit serves both."""
    return [(g, h) for g in range(n_groups) for h in range(g + 1, n_groups)]


# ---------------------------------------------------------------------------------------------
# determinism

def pin_cpu_env(env: dict | None = None) -> None:
    """Put ``CPU_ENV`` into ``os.environ`` without overriding values already set. Call it before torch
    is imported; afterwards it has no effect on the kernels torch already chose."""
    for k, v in (env or CPU_ENV).items():
        os.environ.setdefault(k, v)


def set_deterministic(threads: int = TCN_THREADS) -> None:
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:            # only settable before the first parallel op; harmless then
        pass


def fit_tcn(seqs: list, seed: int = 0, threads: int = TCN_THREADS, steps: int | None = None):
    """``tcn.train`` under the deterministic settings. ``seqs`` as for ``tcn.train`` (already scaled)."""
    from katena.models import tcn
    set_deterministic(threads)
    return tcn.train(seqs, steps=steps or tcn.STEPS, seed=seed, threads=threads)


def sha256_arrays(*arrays) -> str:
    """Hash of dtype, shape and bytes of every array, in order."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def model_hash(model) -> str:
    """Hash of a torch module's parameters and buffers, in name order."""
    sd = model.state_dict()
    return sha256_arrays(*[sd[k].detach().cpu().numpy() for k in sorted(sd)])
