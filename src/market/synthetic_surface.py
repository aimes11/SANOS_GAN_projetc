# src/market/synthetic_surface.py
from __future__ import annotations

import math
from typing import List

from src.market.grid import SurfaceGrid
from src.pricer.bs import BSInputs, price, forward


def smile_vol(k: float, T: float,
              sigma0: float = 0.2,
              skew: float = -0.2,
              curvature: float = 0.4,
              term_alpha: float = 0.0) -> float:
    """
    Simple parametric smile in log-moneyness.

    sigma(k,T) = sigma0
                 + skew * k
                 + curvature * k^2
                 + term_alpha * sqrt(T)

    Parameters chosen small to avoid obvious arbitrage in tests.
    """
    vol = sigma0 + skew * k + curvature * k * k + term_alpha * math.sqrt(T)
    return max(vol, 1e-4)  # avoid zero/negative vol


def build_vol_surface(grid: SurfaceGrid,
                      sigma0: float = 0.2,
                      skew: float = -0.2,
                      curvature: float = 0.4,
                      term_alpha: float = 0.0) -> List[List[float]]:
    """
    Returns vol_matrix[i][j] = sigma(k_j, T_i)
    """
    vol_matrix: List[List[float]] = []

    for T in grid.maturities:
        row = [smile_vol(k, T, sigma0, skew, curvature, term_alpha)
               for k in grid.k_grid]
        vol_matrix.append(row)

    return vol_matrix


def build_price_surface(grid: SurfaceGrid,
                        spot: float,
                        rate: float,
                        div: float,
                        sigma0: float = 0.2,
                        skew: float = -0.2,
                        curvature: float = 0.4,
                        term_alpha: float = 0.0) -> List[List[float]]:
    """
    Returns price_matrix[i][j] = Call price at (T_i, k_j)
    """
    price_matrix: List[List[float]] = []

    for T in grid.maturities:
        F = forward(spot, T, rate, div)
        Ks = grid.strikes_from_forward(F)

        row = []
        for k, K in zip(grid.k_grid, Ks):
            vol = smile_vol(k, T, sigma0, skew, curvature, term_alpha)
            inp = BSInputs(spot, K, T, rate, div, vol)
            row.append(price(inp, "call"))

        price_matrix.append(row)

    return price_matrix