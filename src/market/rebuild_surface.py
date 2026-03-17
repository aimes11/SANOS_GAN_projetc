# src/market/rebuild_surface.py
from __future__ import annotations

from typing import List, Dict
import numpy as np

from src.market.grid import SurfaceGrid


def rebuild_price_surface_from_snapshot(
    snapshot: List[Dict],
    grid: SurfaceGrid,
    field: str = "mid",
) -> List[List[float]]:
    """
    Rebuild a rectangular price surface on the standard k-grid
    from an irregular market snapshot.

    For each maturity:
      - collect available quotes
      - sort by k
      - linearly interpolate onto the full k_grid

    field can be: "mid", "bid", "ask", "true_price"
    """
    surface = []

    for T in grid.maturities:
        quotes_T = [q for q in snapshot if abs(q["maturity"] - T) < 1e-12]

        if len(quotes_T) < 2:
            raise ValueError(f"Not enough quotes to interpolate for maturity {T}")

        quotes_T = sorted(quotes_T, key=lambda q: q["k"])

        k_obs = np.array([q["k"] for q in quotes_T], dtype=float)
        p_obs = np.array([q[field] for q in quotes_T], dtype=float)

        k_target = np.array(grid.k_grid, dtype=float)

        # linear interpolation, flat extrapolation at boundaries
        p_interp = np.interp(k_target, k_obs, p_obs)

        surface.append(p_interp.tolist())

    return surface