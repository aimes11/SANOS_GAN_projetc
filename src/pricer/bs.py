from __future__ import annotations

import math
from typing import Literal

OptionType =  Literal['call', "put"]

def norm_cdf(x : float) -> float:
    """Standard normal CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def norm_pdf(x : float) -> float:
    """Standard normal PDF"""
    return math.exp(- x ** 2 / 2) / math.sqrt(2 * math.pi)

class BSInputs:
    """ Box tool for Black Scholes Model inputs"""

    def __init__(self, spot: float, strike: float, ttm: float, rate: float, div: float, vol: float):
        self.spot = float(spot)
        self.strike = float(strike)
        self.ttm = float(ttm)
        self.rate = float(rate)
        self.div = float(vol)
        self.vol = float(vol)

    def __repr__(self) -> str:
        #To facilitate debug we show directly the parameters rather then the adress
        return(
            f'BSInputs(spot = {self.spot}, strike = {self.strike}, ttm = {self.ttm}, rate = {self.rate}, div = {self.div}, vol = {self.rate})'
        )
    
def forward (spot: float, rate: float, div: float, ttm: float) -> float:
    """Forward price F = S * exp((r - d) * T)"""
    return spot * math.exp((rate - div) * ttm)

def discount(rate: float, div: float, ttm: float) -> float:
    """Discounting function D = exp(- (r - d) * T)"""
    return math.exp(-(rate - div) * ttm)

def d1_d2( F: float, strike: float, vol: float, ttm: float) -> tuple[float, float]:
    if F <= 0 or strike <= 0:
        raise ValueError("F and K must be > 0")
    if ttm <= 0:
        return float("nan"), float("nan")
    if vol <= 0:
        raise ValueError('vol must be > 0')
    srt = vol * math.sqrt(ttm)
    d1 = (math.log(F / strike) + 0.5 * vol ** 2 * ttm) / srt
    d2 = d1 - srt
    return d1, d2

def price(inp: BSInputs, opt_type: OptionType = "call") -> float:
    """
    BlackScholes under forward measure
    C = D * (F * N(d1) - K * N(d2))
    P = D * (K * N(-d2) - F * N(-d1))
    """
    S, K, T, r, d, sig = inp.spot, inp.strike, inp.ttm, inp.rate, inp.div, inp.vol

    if S <= 0 or K <= 0:
        raise ValueError('S and K must be > 0')
    if T < 0:
        raise ValueError('ttm must be >= 0')
    if sig < 0:
        raise ValueError('vol must be  >= 0')
    
    # price at maturity
    if T == 0:
        return max(S - K, 0) if opt_type == "call" else max(K - S, 0)
    
    D = discount(r, d, T)
    F = forward(S, r, d, T)

    if sig == 0:
        #deterministic payoff under forward measure
        return D * max(F - K, 0) if opt_type == "call" else D * max(K - F, 0)
    
    d1, d2 = d1_d2(F, K, sig, T)
    if opt_type == "call":
        return D * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return D * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
    
def vega(inp: BSInputs) -> float:
    """Vega = dPrice/dVol = D * F * N'(d1) * sqrt(T)"""
    S, K, T, r, d, sig = inp.spot, inp.strike, inp.ttm, inp.rate, inp.div, inp.vol
    if S <= 0 or K <= 0:
        raise ValueError('S and K must be > 0')
    if T < 0 or sig <= 0:
        return 0.0
    D = discount(r, d, T)
    F = forward(S, r, d, T)

    d1, _ = d1_d2(F, K, sig, T)
    return D * F * norm_pdf(d1) * math.sqrt(T)

    
    
    