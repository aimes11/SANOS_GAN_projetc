from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence


def rebuild_surface_from_dataframe_on_strikes(
    df: pd.DataFrame,
    maturities: Sequence[float],
    strike_grid: Sequence[float],
    field: str = "mid",
    maturity_tol: float = 1e-10,
) -> list[list[float]]:
    if len(maturities) == 0:
        raise ValueError("maturities must not be empty")
    if len(strike_grid) < 2:
        raise ValueError("strike_grid must contain at least two points")

    strike_grid = np.asarray(strike_grid, dtype=float)
    if np.any(np.diff(strike_grid) <= 0):
        raise ValueError("strike_grid must be strictly increasing")

    if field not in df.columns:
        raise ValueError(f"Field '{field}' not found in DataFrame")

    # split once by maturity
    grouped = {float(T): g.sort_values("strike") for T, g in df.groupby("T")}

    surface: list[list[float]] = []

    for T in maturities:
        T = float(T)

        if T not in grouped:
            raise ValueError(f"No data for maturity {T}")

        df_T = grouped[T]

        K_obs = df_T["strike"].to_numpy(dtype=float)
        P_obs = df_T[field].to_numpy(dtype=float)

        if len(K_obs) < 2:
            raise ValueError(f"Not enough quotes to interpolate for maturity {T}")

        if np.any(np.diff(K_obs) <= 0):
            raise ValueError(f"Observed strikes must be strictly increasing at maturity {T}")

        if np.any(P_obs < 0):
            raise ValueError(f"Negative values found in field '{field}' at maturity {T}")

        P_interp = np.interp(strike_grid, K_obs, P_obs)
        surface.append(P_interp.tolist())

    return surface