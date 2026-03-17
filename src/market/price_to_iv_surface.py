# src/market/price_to_iv_surface.py
from __future__ import annotations

from typing import List

from src.pricer.bs import BSInputs
from src.pricer.implied_vol import implied_vol


def price_surface_to_iv_surface(
    price_surface: List[List[float]],
    strikes_surface: List[List[float]],
    maturities: List[float],
    spot: float,
    rate: float,
    div: float,
    opt_type: str = "call",
    fallback_nan: bool = True,
) -> List[List[float]]:
    """
    Convert a call price surface C(K,T) into an implied volatility surface sigma(K,T).

    price_surface[i][j]   = price at maturity i and strike j
    strikes_surface[i][j] = strike at maturity i and strike j
    maturities[i]         = maturity T_i
    """
    iv_surface = []

    for i, T in enumerate(maturities):
        iv_row = []
        for j, K in enumerate(strikes_surface[i]):
            price_ij = price_surface[i][j]

            inp = BSInputs(
                spot=spot,
                strike=K,
                ttm=T,
                rate=rate,
                div=div,
                vol=0.2,   # initial placeholder
            )

            iv = implied_vol(price_ij, inp, opt_type=opt_type)

            if iv is None:
                iv_row.append(float("nan") if fallback_nan else 0.0)
            else:
                iv_row.append(iv)

        iv_surface.append(iv_row)

    return iv_surface