# src/sanos/projection_slice.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cvxpy as cp


@dataclass
class SliceProjectionResult:
    C_clean: List[float]
    status: str
    objective_value: float


@dataclass
class SurfaceProjectionResult:
    C_clean: List[List[float]]  # (n_maturities, n_strikes)
    status: str
    objective_value: float


def project_slice_noarb(
    K: List[float],
    C_mkt: List[float],
    eps: float = 0.0,
    weights: Optional[List[float]] = None,
) -> SliceProjectionResult:
    """
    Project one maturity slice of call prices onto no-arbitrage constraints in strike:

    Constraints (for calls):
      1) C(K) decreasing in K
      2) C(K) convex in K  -> discrete butterfly >= 0
      3) C(K) >= 0

    Objective:
      minimize sum w_j (C_j - C_mkt_j)^2

    K must be strictly increasing.
    """
    K = np.asarray(K, dtype=float)
    C_mkt = np.asarray(C_mkt, dtype=float)
    n = len(K)

    if n < 3:
        raise ValueError("Need at least 3 strikes to enforce convexity.")
    if not np.all(np.diff(K) > 0):
        raise ValueError("Strikes K must be strictly increasing.")
    if np.any(C_mkt < -1e-12):
        raise ValueError("Market prices must be >= 0 (or close).")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("weights length mismatch.")
        if np.any(w <= 0):
            raise ValueError("weights must be > 0.")

    # Decision variable: cleaned call prices
    C = cp.Variable(n)

    constraints = []
    constraints.append(C >= 0.0)

    # 1) Monotonicity: C_{j+1} <= C_j + eps
    for j in range(n - 1):
        constraints.append(C[j + 1] - C[j] <= eps)

    # 2) Convexity (non-uniform grid):
    # c2_j = 2 * [ (C_{j+1}-C_j)/(K_{j+1}-K_j) - (C_j-C_{j-1})/(K_j-K_{j-1}) ] / (K_{j+1}-K_{j-1}) >= -eps
    for j in range(1, n - 1):
        Km, K0, Kp = K[j - 1], K[j], K[j + 1]
        denom = (Kp - Km)
        left = (C[j] - C[j - 1]) / (K0 - Km)
        right = (C[j + 1] - C[j]) / (Kp - K0)
        c2 = 2.0 * (right - left) / denom
        constraints.append(c2 >= -eps)

    objective = cp.Minimize(cp.sum(cp.multiply(w, cp.square(C - C_mkt))))
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.ECOS, verbose=False)

    C_clean = C.value
    if C_clean is None:
        return SliceProjectionResult(C_clean=[], status=str(prob.status), objective_value=float("nan"))

    return SliceProjectionResult(
        C_clean=C_clean.tolist(),
        status=str(prob.status),
        objective_value=float(prob.value),
    )



def project_surface_noarb(
    K_surface: list[list[float]],
    T: list[float],
    C_mkt: list[list[float]],
    spot: float,
    rate: float,
    div: float,
    weights: list[list[float]] | None = None,
    eps_strike: float = 0.0,
    eps_calendar: float = 0.0,
) -> SurfaceProjectionResult:
    """
    Global projection of call price surface onto no-arbitrage constraints.

    Constraints:
        - C >= 0
        - Monotonicity in strike
        - Convexity in strike
        - Calendar monotonicity
        - Financial bounds:
            max(S*e^{-qT} - K*e^{-rT}, 0) <= C <= S*e^{-qT}
    """

    # -----------------------------
    # Convert inputs
    # -----------------------------
    K_surface = [np.asarray(K, dtype=float) for K in K_surface]
    T = np.asarray(T, dtype=float)
    C_mkt = np.asarray(C_mkt, dtype=float)

    n_maturities = len(T)
    if n_maturities == 0:
        raise ValueError("Empty maturities")

    n_strikes = len(K_surface[0])
    if n_strikes < 3:
        raise ValueError("Need at least 3 strikes")

    # -----------------------------
    # Validation
    # -----------------------------
    if len(K_surface) != n_maturities:
        raise ValueError("K_surface length mismatch")

    if C_mkt.shape != (n_maturities, n_strikes):
        raise ValueError("C_mkt shape mismatch")

    if not np.all(np.diff(T) > 0):
        raise ValueError("T must be strictly increasing")

    if np.any(C_mkt < -1e-12):
        raise ValueError("Negative prices in market data")

    for i, K in enumerate(K_surface):
        if not np.all(np.diff(K) > 0):
            raise ValueError(f"K[{i}] not strictly increasing")

    # Check aligned strikes (CRITICAL for calendar constraints)
    K_ref = K_surface[0]
    for i in range(1, n_maturities):
        if not np.allclose(K_surface[i], K_ref, atol=1e-10):
            raise ValueError("Strike grids must be identical across maturities")

    # -----------------------------
    # Weights
    # -----------------------------
    if weights is None:
        w = np.ones((n_maturities, n_strikes))
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_maturities, n_strikes):
            raise ValueError("weights shape mismatch")
        if np.any(w <= 0):
            raise ValueError("weights must be positive")

    # -----------------------------
    # CVXPY variable
    # -----------------------------
    C = cp.Variable((n_maturities, n_strikes))

    constraints = []

    # -----------------------------
    # 1) Positivity
    # -----------------------------
    constraints.append(C >= 0.0)

    # -----------------------------
    # 2) Financial bounds
    # -----------------------------
    for i in range(n_maturities):
        T_i = T[i]

        df_r = np.exp(-rate * T_i)
        df_q = np.exp(-div * T_i)

        K_i = K_surface[i]

        lower = np.maximum(spot * df_q - K_i * df_r, 0.0)
        upper = spot * df_q

        constraints.append(C[i, :] >= lower)
        constraints.append(C[i, :] <= upper)

    # -----------------------------
    # 3) Strike constraints
    # -----------------------------
    for i in range(n_maturities):
        K = K_surface[i]

        # Monotonicity
        for j in range(n_strikes - 1):
            constraints.append(C[i, j + 1] - C[i, j] <= eps_strike)

        # Convexity
        for j in range(1, n_strikes - 1):
            Km, K0, Kp = K[j - 1], K[j], K[j + 1]

            left = (C[i, j] - C[i, j - 1]) / (K0 - Km)
            right = (C[i, j + 1] - C[i, j]) / (Kp - K0)

            c2 = 2.0 * (right - left) / (Kp - Km)

            constraints.append(c2 >= -eps_strike)

    # -----------------------------
    # 4) Calendar constraints
    # -----------------------------
    for i in range(n_maturities - 1):
        constraints.append(C[i, :] <= C[i + 1, :] + eps_calendar)

    # -----------------------------
    # Objective
    # -----------------------------
    objective = cp.Minimize(
        cp.sum(cp.multiply(w, cp.square(C - C_mkt)))
    )

    prob = cp.Problem(objective, constraints)

    # -----------------------------
    # Solve
    # -----------------------------
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)

    # -----------------------------
    # Output
    # -----------------------------
    C_clean = C.value

    if C_clean is None:
        return SurfaceProjectionResult(
            C_clean=[],
            status=str(prob.status),
            objective_value=float("nan"),
        )

    C_clean = np.maximum(np.asarray(C_clean, dtype=float), 0.0)

    return SurfaceProjectionResult(
        C_clean=C_clean.tolist(),
        status=str(prob.status),
        objective_value=float(prob.value),
    )