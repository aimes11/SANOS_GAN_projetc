# src/validation/static_arbitrage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.validation.butterfly import ButterflyReport, check_butterfly
from src.validation.calendar import CalendarReport, check_calendar_kgrid
from src.validation.monotonicity import StrikeMonoReport, check_call_decreasing_in_strike


@dataclass
class StaticArbitrageReport:
    strike_mono: StrikeMonoReport
    butterfly: ButterflyReport
    calendar: CalendarReport

    def ok(self) -> bool:
        return (
            self.strike_mono.n_violations == 0
            and self.butterfly.n_violations == 0
            and self.calendar.n_violations == 0
        )


def check_static_no_arb(
    price_surface: List[List[float]],
    strikes_surface: List[List[float]],
    eps: float = 1e-12,
) -> StaticArbitrageReport:
    """
    Aggregates:
      - decreasing in strike (call monotonicity)
      - convex in strike (butterfly)
      - increasing in maturity (calendar on k-grid, debug check)
    """
    strike_mono = check_call_decreasing_in_strike(price_surface, eps=eps)
    butterfly = check_butterfly(price_surface, strikes_surface, eps=eps)
    calendar = check_calendar_kgrid(price_surface, eps=eps)

    return StaticArbitrageReport(
        strike_mono=strike_mono,
        butterfly=butterfly,
        calendar=calendar,
    )