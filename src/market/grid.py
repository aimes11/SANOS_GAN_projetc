from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, List


@dataclass(frozen=True)
class SurfaceGrid:
    """
    Standard grid for an equity index vol surface.

    We work in log-moneyness:
        k = log(K / F)
    """
    maturities: List[float]
    k_grid: List[float]

    def __post_init__(self):
        if any(T <= 0 for T in self.maturities):
            raise ValueError("All maturities must be positive")
        if sorted(self.maturities) != self.maturities:
            raise ValueError("Maturities must be in increasing order")
        if len(self.k_grid) < 2:
            raise ValueError("k_grid must have at least two points")
        if sorted(self.k_grid) != self.k_grid:
            raise ValueError("k_grid must be in increasing order")

    def strikes_from_forward(self, F: float) -> List[float]:
        """Return strikes K = F * exp(k) for all k in k_grid."""
        if F <= 0:
            raise ValueError("Forward F must be > 0")
        return [F * math.exp(k) for k in self.k_grid]


def make_default_grid(
    maturities: Sequence[float] | None = None,
    k_min: float = -0.3,
    k_max: float = 0.3,
    n_k: int = 61,
) -> SurfaceGrid:
    """
    Factory helper: creates a typical equity grid.
    """
    if maturities is None:
        maturities = [1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 2.0]

    if n_k < 2:
        raise ValueError("n_k must be >= 2")
    if k_max <= k_min:
        raise ValueError("k_max must be > k_min")

    step = (k_max - k_min) / (n_k - 1)
    k_grid = [k_min + i * step for i in range(n_k)]

    return SurfaceGrid(maturities=list(maturities), k_grid=k_grid)