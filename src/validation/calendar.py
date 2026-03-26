# src/validation/calendar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CalendarReport:
    n_violations: int
    worst_value: float             # most negative dC (C(T_next)-C(T_prev))
    worst_location: Tuple[int, int]  # (i_maturity, j_k)


def check_calendar_kgrid(price_surface: List[List[float]], eps: float = 0.0) -> CalendarReport:
    """
    Approx calendar check on a fixed k-grid:
      for each k_j: C(T_{i+1}, k_j) >= C(T_i, k_j)

    This is a useful debugging check. Strict calendar no-arb is at fixed K,
    which we can implement later via interpolation.
    """
    nT = len(price_surface)
    nK = len(price_surface[0]) if nT > 0 else 0

    n_viol = 0
    worst = 0.0
    worst_loc = (-1, -1)

    for j in range(nK):
        for i in range(nT - 1):
            dC = price_surface[i + 1][j] - price_surface[i][j]
            if dC < -eps:
                n_viol += 1
                if dC < worst:
                    worst = dC
                    worst_loc = (i, j)

    return CalendarReport(n_violations=n_viol, worst_value=worst, worst_location=worst_loc)