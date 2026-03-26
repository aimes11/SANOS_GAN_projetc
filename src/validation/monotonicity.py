# src/validation/monotonicity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StrikeMonoReport:
    n_violations: int
    worst_value: float              # most positive dC (should be <= 0)
    worst_location: Tuple[int, int]  # (i_maturity, j_strike)


def check_call_decreasing_in_strike(price_surface: List[List[float]], eps: float = 0.0) -> StrikeMonoReport:
    """
    For each maturity i: check C(K_{j+1}) - C(K_j) <= eps (decreasing in strike).
    Assumes strikes are ordered increasingly along j (true for our grid).
    """
    nT = len(price_surface)
    nK = len(price_surface[0]) if nT > 0 else 0

    n_viol = 0
    worst = 0.0
    worst_loc = (-1, -1)

    for i in range(nT):
        row = price_surface[i]
        for j in range(nK - 1):
            dC = row[j + 1] - row[j]  # should be <= 0
            if dC > eps:
                n_viol += 1
                if dC > worst:
                    worst = dC
                    worst_loc = (i, j)

    return StrikeMonoReport(n_violations=n_viol, worst_value=worst, worst_location=worst_loc)