from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def make_common_strike_grid(
    spot: float,
    k_min: float = -0.3,
    k_max: float = 0.3,
    n_k: int = 61,
) -> list[float]:
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if n_k < 2:
        raise ValueError("n_k must be >= 2")
    if k_max <= k_min:
        raise ValueError("k_max must be > k_min")

    step = (k_max - k_min) / (n_k - 1)
    k_grid = [k_min + i * step for i in range(n_k)]
    return [spot * math.exp(k) for k in k_grid]


def make_common_strike_grid_from_data(
    df: pd.DataFrame,
    maturities: Sequence[float],
    n_k: int = 41,
) -> list[float]:
    if n_k < 2:
        raise ValueError("n_k must be >= 2")
    if len(maturities) == 0:
        raise ValueError("maturities must not be empty")

    mins: list[float] = []
    maxs: list[float] = []

    for T in maturities:
        df_T = df[df["T"] == T]
        if df_T.empty:
            continue
        mins.append(float(df_T["strike"].min()))
        maxs.append(float(df_T["strike"].max()))

    if not mins or not maxs:
        raise ValueError("No strike coverage available to build common grid.")

    strike_min_common = max(mins)
    strike_max_common = min(maxs)

    if strike_max_common <= strike_min_common:
        raise ValueError("No overlapping strike range across maturities.")

    return np.linspace(strike_min_common, strike_max_common, n_k).tolist()


def make_strike_surface(
    maturities: Sequence[float],
    strike_grid: Sequence[float],
) -> list[list[float]]:
    if len(maturities) == 0:
        raise ValueError("maturities must not be empty")
    if len(strike_grid) < 2:
        raise ValueError("strike_grid must contain at least two points")

    return [list(strike_grid) for _ in maturities]

def find_compatible_maturities_for_common_grid(
    df: pd.DataFrame,
    maturities: Sequence[float],
    min_overlap_width: float = 10.0,
) -> list[float]:
    """
    Keep the largest ordered subset of maturities whose strike ranges
    still admit a non-empty common overlap of width >= min_overlap_width.
    """
    selected: list[float] = []
    current_min = -np.inf
    current_max = np.inf

    for T in sorted(maturities):
        df_T = df[df["T"] == T]
        if df_T.empty:
            continue

        t_min = float(df_T["strike"].min())
        t_max = float(df_T["strike"].max())

        new_min = max(current_min, t_min)
        new_max = min(current_max, t_max)

        if new_max - new_min >= min_overlap_width:
            selected.append(float(T))
            current_min = new_min
            current_max = new_max

    return selected