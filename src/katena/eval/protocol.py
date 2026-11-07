"""Pre-registered protocol constants (docs/bakeoff-protocol.md). Change only via a dated amendment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CANON = REPO / "external_data" / "canonical"
WORK = REPO / "work" / "bakeoff"

DATASETS = ("idsse", "metrica", "skillcorner")
LOCKED_METRICA = "Sample_Game_2"
SEVERITIES = ("mild", "target", "severe")
HEADLINE = "target"
T_SWEEP = (0, 1, 2, 3, 5)
T_MAIN = 2
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)       # extended from (0.25, 0.5, 1.0): see Amendment A2
SEED = 0


@dataclass(frozen=True)
class Ref:
    dataset: str
    id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.id}"

    @property
    def is_sc(self) -> bool:
        return self.dataset == "skillcorner"


def all_refs() -> list[Ref]:
    out = []
    for ds in DATASETS:
        for p in sorted((CANON / ds).glob("*.npz")):
            out.append(Ref(ds, p.stem, p))
    return out


def idsse_hash(match_id: str) -> str:
    return hashlib.sha256(("bakeoff-v1/idsse/" + match_id).encode()).hexdigest()


def locked_idsse(ids: list[str]) -> str:
    """The IDSSE match with the lexicographically smallest hash. Chosen by hash, not by looking."""
    return min(ids, key=idsse_hash)


def split(refs: list[Ref]) -> tuple[list[Ref], list[Ref]]:
    """(development, locked benches)."""
    li = locked_idsse([r.id for r in refs if r.dataset == "idsse"])
    locked = [r for r in refs if (r.dataset == "idsse" and r.id == li)
              or (r.dataset == "metrica" and r.id == LOCKED_METRICA)]
    dev = [r for r in refs if r not in locked]
    return dev, locked


def seed_for(match_id: str, cond: str) -> int:
    return int(hashlib.sha256(f"{match_id}|{cond}".encode()).hexdigest()[:8], 16) % (2 ** 31)


def group_of(match_id: str, n_groups: int = 6) -> int:
    """Hash-fixed group for the grouped (nested) folds used by M6."""
    return int(hashlib.sha256(("group|" + match_id).encode()).hexdigest()[:8], 16) % n_groups
