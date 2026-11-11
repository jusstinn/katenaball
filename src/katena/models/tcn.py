"""M5: small non-causal temporal convolution net on the same 5 Hz features (offline labelling may
look both ways). Two heads as in the GBDT models: alive (dead-labelled sources only) and team
(focal team has the ball). Trained on windows of whole-match feature sequences."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CH = 48
DILATIONS = (1, 2, 4, 8, 16, 32)
WIN = 256
BATCH = 32
STEPS = 1200
LR = 2e-3


class Block(nn.Module):
    def __init__(self, ch: int, dil: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.c2(self.drop(torch.nn.functional.gelu(self.norm(self.c1(x)))))
        return x + h


class TCN(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.inp = nn.Conv1d(n_in, CH, 1)
        self.blocks = nn.Sequential(*[Block(CH, d) for d in DILATIONS])
        self.out = nn.Conv1d(CH, 2, 1)

    def forward(self, x):                      # [B, F, T] -> [B, 2, T] (alive, team logits)
        return self.out(self.blocks(self.inp(x)))


class Scaler:
    def __init__(self, X: np.ndarray):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = np.clip((X - self.mu) / self.sd, -5, 5)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def train(seqs: list, steps: int = STEPS, seed: int = 0, threads: int = 6) -> tuple:
    """``seqs``: list of dicts X float32[T,F] (already scaled), y int8[T], o int8 scalar orientation,
    dead_ok bool. Returns (model, scaler-free). A sequence may carry ``w`` float32[T], per-frame weights of
    the team loss (``katena.models.lenweight``); without it every H or A frame weighs 1, as before."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TCN(seqs[0]["X"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    lens = np.array([len(s["y"]) for s in seqs], dtype=float)
    pr = lens / lens.sum()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for step in range(steps):
        xb = np.zeros((BATCH, seqs[0]["X"].shape[1], WIN), np.float32)
        ya = np.zeros((BATCH, WIN), np.float32); wa = np.zeros((BATCH, WIN), np.float32)
        yt = np.zeros((BATCH, WIN), np.float32); wt = np.zeros((BATCH, WIN), np.float32)
        for b in range(BATCH):
            s = seqs[rng.choice(len(seqs), p=pr)]
            T = len(s["y"])
            a = int(rng.integers(0, max(1, T - WIN)))
            sl = slice(a, a + WIN)
            k = len(s["y"][sl])
            xb[b, :, :k] = s["X"][sl].T
            y = s["y"][sl]
            if s["dead_ok"]:
                ya[b, :k] = (y != 2); wa[b, :k] = (y < 3)
            yt[b, :k] = (y == s["o"]); wt[b, :k] = (y < 2) if "w" not in s else (y < 2) * s["w"][sl]
        out = model(torch.from_numpy(xb))
        la = (bce(out[:, 0], torch.from_numpy(ya)) * torch.from_numpy(wa)).sum() / max(wa.sum(), 1)
        lt = (bce(out[:, 1], torch.from_numpy(yt)) * torch.from_numpy(wt)).sum() / max(wt.sum(), 1)
        opt.zero_grad()
        (la + lt).backward()
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: TCN, X0: np.ndarray, X1: np.ndarray, half: np.ndarray) -> tuple:
    """(p_alive, p_home) averaged over the two orientations; each half run as its own sequence."""
    pa = np.zeros(len(half), np.float32)
    ph = np.zeros(len(half), np.float32)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        o0 = torch.sigmoid(model(torch.from_numpy(X0[sel].T[None]))[0]).numpy()
        o1 = torch.sigmoid(model(torch.from_numpy(X1[sel].T[None]))[0]).numpy()
        pa[sel] = 0.5 * (o0[0] + o1[0])
        ph[sel] = 0.5 * (o0[1] + 1.0 - o1[1])
    return pa, ph
