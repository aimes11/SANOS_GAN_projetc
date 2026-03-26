from __future__ import annotations

from typing import Sequence

from src.pricer.bs import BSInputs
from src.pricer.implied_vol import implied_vol


def price_surface_to_iv_surface(
    price_surface: Sequence[Sequence[float]],
    strikes_surface: Sequence[Sequence[float]],
    maturities: Sequence[float],
    spot: float,
    rate: float,
    div: float,
    opt_type: str = "call",
    fallback_nan: bool = True,
) -> list[list[float]]:
    """
    Convert a European option price surface into an implied volatility surface.

    price_surface[i][j]   = price at maturity i and strike j
    strikes_surface[i][j] = strike at maturity i and strike j
    maturities[i]         = maturity T_i
    """
    if spot <= 0:
        raise ValueError("spot must be > 0")

    opt_type = opt_type.lower()
    if opt_type not in {"call", "put"}:
        raise ValueError("opt_type must be 'call' or 'put'")

    if len(price_surface) != len(strikes_surface) or len(price_surface) != len(maturities):
        raise ValueError("price_surface, strikes_surface and maturities must have matching outer dimensions")

    iv_surface: list[list[float]] = []

    for i, T in enumerate(maturities):
        if T <= 0:
            raise ValueError(f"Maturity at index {i} must be > 0")

        p_row = price_surface[i]
        k_row = strikes_surface[i]

        if len(p_row) != len(k_row):
            raise ValueError(f"Row {i}: price and strike row lengths do not match")

        iv_row: list[float] = []

        for j, K in enumerate(k_row):
            if K <= 0:
                raise ValueError(f"Strike at ({i}, {j}) must be > 0")

            price_ij = float(p_row[j])

            inp = BSInputs(
                spot=float(spot),
                strike=float(K),
                ttm=float(T),
                rate=float(rate),
                div=float(div),
                vol=0.2,  # placeholder required by BSInputs interface
            )

            iv = implied_vol(price_ij, inp, opt_type=opt_type)

            if iv is None:
                if fallback_nan:
                    iv_row.append(float("nan"))
                else:
                    raise ValueError(
                        f"Implied vol not found at maturity index {i}, strike index {j}"
                    )
            else:
                iv_row.append(float(iv))

        iv_surface.append(iv_row)

    return iv_surface