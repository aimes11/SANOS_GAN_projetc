from __future__ import annotations

from typing import Literal, Optional

from bs import BSInputs, price, forward, discount

OptionType = Literal["call", "put" ]

def no_arb_bound_call(F: float, K: float, D: float) -> float:
    """ No arbitrage bounds for a call under forward measure"""
    lower = D * max(F - K, 0)
    upper = D * F
    return lower, upper

def no_arb_bound_put(F: float, K: float, D: float) -> float:
    """ No arbitrage bounds for a put under forward measure"""
    lower = D * max(K - F, 0)
    upper = D * K
    return lower, upper

def implied_vol( mkt_price: float, inp: BSInputs, opt_type: OptionType = "call", vol_low: float = 1e-6, vol_high: float = 5.0, tol: float = 1e-8, max_iter: int = 200,) -> Optional[float]:
    """ 
    Implied volatility using bisection method.

    We search for sigma such that Call(sigma) = mkt_price

    Returns:
    - None if mkt_price is outside of the arbitrage bounds
    - impplied vol if solvable
    """

    if mkt_price < 0:
        return None
    
    S, K, T, r, q = inp.spot, inp.strike, inp.ttm, inp.rate, inp.div

    if T ==0 : 
        return 0.0
    
    D = discount(r, T)
    F = forward(S, r, q, T)

    if opt_type == "call":
        lower, upper = no_arb_bound_call(F, K, D)
    else:
        lower, upper = no_arb_bound_put(F, K, D)    

    if mkt_price < lower - tol or mkt_price > upper + tol:
        return None
    
    def f(vol: float) -> float:
        tmp = BSInputs(spot = S, strike = K, ttm = T, rate = r, div = q, vol = vol)
        return price(tmp, opt_type) - mkt_price
    
    a, b = vol_low, vol_high
    fa, fb = f(a), f(b)

    tries = 0
    while fa * fb > 0 and tries < 10:
        b *= 2.0
        fb = f(b)
        tries += 1

    if fa * fb > 0:
        raise ValueError("f(vol_low) and f(vol_high) must have opposite signs") 
    
    # Bisection algorithm
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        if f(m) == 0  or 0.5 * (b -a) < tol:
            return m
        
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)