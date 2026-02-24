# src/validation/butterfly.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ButterflyReport:
    n_violations: int
    worst_value: float          # most negative convexity
    worst_location: Tuple[int, int]  # (i_maturity, j_strike)


def _second_derivative_nonuniform(Km: float, K: float, Kp: float, Cm: float, C: float, Cp: float) -> float:
    """
    Approximate C''(K) on a non-uniform grid using 3 points:
    (Km,Cm), (K,C), (Kp,Cp)

    Formula:
      C''(K) ≈ 2 * [ (Cp - C)/(Kp - K) - (C - Cm)/(K - Km) ] / (Kp - Km)
    """
    left = (C - Cm) / (K - Km)
    right = (Cp - C) / (Kp - K)
    return 2.0 * (right - left) / (Kp - Km)


def check_butterfly(price_surface: List[List[float]], strikes_surface: List[List[float]], eps: float = 0.0) -> ButterflyReport:
    """
    price_surface[i][j] = call price at maturity i, strike index j
    strikes_surface[i][j] = corresponding strike K at maturity i, strike index j

    Condition (discrete): C''(K) >= -eps for all internal points j=1..n-2.
    """
    nT = len(price_surface)
    nK = len(price_surface[0]) if nT > 0 else 0

    n_viol = 0
    worst = 0.0
    worst_loc = (-1, -1)

    for i in range(nT):
        Ks = strikes_surface[i]
        Cs = price_surface[i]
        for j in range(1, nK - 1):
            Km, K, Kp = Ks[j - 1], Ks[j], Ks[j + 1]
            Cm, C, Cp = Cs[j - 1], Cs[j], Cs[j + 1]

            c2 = _second_derivative_nonuniform(Km, K, Kp, Cm, C, Cp)

            if c2 < -eps:
                n_viol += 1
                if c2 < worst:
                    worst = c2
                    worst_loc = (i, j)

    return ButterflyReport(n_violations=n_viol, worst_value=worst, worst_location=worst_loc)