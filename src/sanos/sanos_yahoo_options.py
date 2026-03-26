from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.ndimage import gaussian_filter

from src.market.strike_grid import make_common_strike_grid_from_data, make_strike_surface, find_compatible_maturities_for_common_grid
from src.market.rebuild_surface_strike import rebuild_price_surface_from_snapshot_on_strikes
from src.market.projection_slice import project_surface_noarb
from src.market.price_to_iv_surface import price_surface_to_iv_surface
from src.validation.StaticArbitrageReport import check_static_no_arb


def download_yahoo_calls(symbol: str = "SPY", n_expiries: int = 365) -> tuple[pd.DataFrame, float]:
    tk = yf.Ticker(symbol)
    expiries = tk.options[:n_expiries]
    if not expiries:
        raise ValueError(f"No option expiries found for {symbol}")

    # asset-specific filters
    if symbol == "TSLA":
        k_min, k_max = -0.6, 0.6
        spread_max = 1.5
        iv_max = 5.0
    elif symbol == "NVDA":
        k_min, k_max = -0.5, 0.5
        spread_max = 1.0
        iv_max = 4.0
    else:  # SPY, QQQ, etc.
        k_min, k_max = -0.25, 0.25
        spread_max = 0.5
        iv_max = 3.0

    data = []
    for expiry in expiries:
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy()
        if calls.empty:
            continue
        calls["expiry"] = expiry
        calls["type"] = "call"
        data.append(calls)

    if not data:
        raise ValueError(f"No call data downloaded for {symbol}")

    df = pd.concat(data, ignore_index=True)

    df["expiry"] = pd.to_datetime(df["expiry"])
    today = pd.Timestamp.now().normalize()
    df["T"] = (df["expiry"] - today).dt.total_seconds() / (365.0 * 24 * 3600)

    spot = float(tk.history(period="1d")["Close"].iloc[-1])

    # basic filters
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] > df["bid"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]

    df = df[df["T"] > 0.01]
    df = df[df["mid"] > 0]

    df["k"] = np.log(df["strike"] / spot)
    df = df[(df["k"] > k_min) & (df["k"] < k_max)]

    df = df[df["spread"] / df["mid"] < spread_max]

    if "impliedVolatility" in df.columns:
        df = df[(df["impliedVolatility"] > 0.01) & (df["impliedVolatility"] < iv_max)]

    cols = [
        "expiry", "T", "strike", "bid", "ask", "mid",
        "spread", "impliedVolatility", "type", "k"
    ]
    df = df[cols].sort_values(["T", "strike"]).reset_index(drop=True)

    print(f"{symbol} filters -> k in [{k_min},{k_max}], spread/mid < {spread_max}, IV < {iv_max}")
    print("Rows after cleaning:", len(df))

    return df, spot


def dataframe_to_snapshot(df: pd.DataFrame) -> list[dict]:
    snapshot = []
    for _, row in df.iterrows():
        snapshot.append({
            "maturity": float(row["T"]),
            "k": float(row["k"]),
            "strike": float(row["strike"]),
            "mid": float(row["mid"]),
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "spread": float(row["spread"]),
        })
    return snapshot


def make_weights_from_spread_surface(
    spread_surface: list[list[float]],
    floor: float = 1e-3,
    cap: float = 1e3,
) -> list[list[float]]:
    S = np.array(spread_surface, dtype=float)
    W = 1.0 / np.maximum(S, floor) ** 2
    W = np.minimum(W, cap)
    return W.tolist()


def plot_surface_3d_strike(maturities, strike_grid, surface, title):
    T = np.array(maturities, dtype=float)
    K = np.array(strike_grid, dtype=float)
    Z = np.array(surface, dtype=float)

    Kmesh, Tmesh = np.meshgrid(K, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Kmesh, Tmesh, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Call price")
    ax.set_title(title)
    plt.show()


def plot_slice_strike(maturities, strike_grid, surface_before, surface_after, i=0):
    K = np.array(strike_grid, dtype=float)
    C_before = np.array(surface_before[i], dtype=float)
    C_after = np.array(surface_after[i], dtype=float)
    T = maturities[i]

    plt.figure()
    plt.plot(K, C_before, label="Rebuilt market surface", marker="o", markersize=3)
    plt.plot(K, C_after, label="Projected no-arb surface", marker="o", markersize=3)
    plt.xlabel("Strike K")
    plt.ylabel("Call price")
    plt.title(f"Slice projection at T={T:.3f}y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_iv_surface_3d_strike(
    maturities,
    strike_grid,
    iv_surface,
    title,
    smooth_sigma: float | None = None,
):
    T = np.array(maturities, dtype=float)
    K = np.array(strike_grid, dtype=float)
    Z = np.array(iv_surface, dtype=float)

    print(title)
    print("IV shape:", Z.shape)
    print("NaN count:", np.isnan(Z).sum(), "/", Z.size)

    if Z.size == 0:
        print("Empty IV surface.")
        return

    if np.all(np.isnan(Z)):
        print("All IV values are NaN, nothing to plot.")
        return

    if smooth_sigma is not None:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=smooth_sigma)

    Kmesh, Tmesh = np.meshgrid(K, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Kmesh, Tmesh, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Implied volatility")
    ax.set_title(title)
    plt.show()


def plot_iv_slice_strike(maturities, strike_grid, iv_surface_before, iv_surface_after, i=0):
    K = np.array(strike_grid, dtype=float)
    iv_before = np.array(iv_surface_before[i], dtype=float)
    iv_after = np.array(iv_surface_after[i], dtype=float)
    T = maturities[i]

    plt.figure()
    plt.plot(K, iv_before, label="IV before projection", marker="o", markersize=3)
    plt.plot(K, iv_after, label="IV after projection", marker="o", markersize=3)
    plt.xlabel("Strike K")
    plt.ylabel("Implied volatility")
    plt.title(f"Implied vol slice at T={T:.3f}y")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    symbol = "SPY"

    # 1) download + clean
    df, spot = download_yahoo_calls(symbol=symbol, n_expiries=365)
    print(f"{symbol} spot:", spot)
    print("Number of cleaned quotes:", len(df))
    print(df.head())

    # 2) keep maturities with enough quotes
    counts = df.groupby("T").size().reset_index(name="n_quotes")
    print(counts)

    candidate_maturities = counts[counts["n_quotes"] >= 5]["T"].tolist()
    print("Candidate maturities:", candidate_maturities)

    valid_maturities = find_compatible_maturities_for_common_grid(
        df=df,
        maturities=candidate_maturities,
        min_overlap_width=10.0,
    )
    print("Compatible maturities:", valid_maturities)

    df = df[df["T"].isin(valid_maturities)].copy()

    if df.empty:
        raise ValueError("No maturities left after filtering.")

    # 3) build snapshot
    snapshot = dataframe_to_snapshot(df)

    # 4) common strike grid across maturities
    maturities = sorted(df["T"].unique().tolist())

    strike_grid = make_common_strike_grid_from_data(
        df=df,
        maturities=maturities,
        n_k=31,
    )

    print(f"Common strike grid: [{strike_grid[0]:.2f}, {strike_grid[-1]:.2f}] with {len(strike_grid)} points")

    strikes_surface = make_strike_surface(
        maturities=maturities,
        strike_grid=strike_grid,
    )

    # diagnostics on observed ranges
    for T in maturities:
        df_T = df[df["T"] == T]
        print(
            f"T={T:.3f} | strike range = "
            f"[{df_T['strike'].min():.2f}, {df_T['strike'].max():.2f}] | "
            f"n={len(df_T)}"
        )

    # 5) rebuild rectangular price surface on common strikes
    surface_mid = rebuild_price_surface_from_snapshot_on_strikes(
        snapshot=snapshot,
        maturities=maturities,
        strike_grid=strike_grid,
        field="mid",
    )

    spread_surface = rebuild_price_surface_from_snapshot_on_strikes(
        snapshot=snapshot,
        maturities=maturities,
        strike_grid=strike_grid,
        field="spread",
    )

    weights_surface = make_weights_from_spread_surface(
        spread_surface=spread_surface,
        floor=1e-3,
        cap=1e3,
    )

    print("Surface shape:", len(surface_mid), "x", len(surface_mid[0]))

    W = np.array(weights_surface, dtype=float)
    print("Weights stats:")
    print("  min =", W.min())
    print("  max =", W.max())
    print("  mean =", W.mean())

    # 6) static no-arb check before
    rep_before = check_static_no_arb(surface_mid, strikes_surface, eps=1e-6)
    print("\nBefore projection:")
    print(rep_before)
    print("OK before:", rep_before.ok())

    # 7) full-surface projection with strike + calendar constraints
    res = project_surface_noarb(
        K_surface=strikes_surface,
        T=maturities,
        C_mkt=surface_mid,
        spot=spot,
        rate=0.0,
        div=0.0,
        weights=weights_surface,
    )
    projected = res.C_clean

    if not projected:
        raise ValueError(f"Projection failed with status={res.status}")

    print("Projection status:", res.status)
    print("Projection objective:", res.objective_value)

    # 8) static no-arb check after
    rep_after = check_static_no_arb(projected, strikes_surface, eps=1e-6)
    print("\nAfter projection:")
    print(rep_after)
    print("OK after:", rep_after.ok())
    print("calendar worst", rep_after.calendar.worst_value, "at", rep_after.calendar.worst_location)
    print("butterfly worst", rep_after.butterfly.worst_value, "at", rep_after.butterfly.worst_location)
    print("strike mono worst", rep_after.strike_mono.worst_value, "at", rep_after.strike_mono.worst_location)

    # 9) convert to implied vol surfaces
    iv_surface_before = price_surface_to_iv_surface(
        surface_mid,
        strikes_surface,
        maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    price_smoothed = gaussian_filter(np.array(projected), sigma=0.7)
    
    iv_surface_after = price_surface_to_iv_surface(
        price_smoothed.tolist(),
        strikes_surface,
        maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    # 10) plots
    if len(maturities) >= 2:
        plot_surface_3d_strike(
            maturities,
            strike_grid,
            surface_mid,
            f"{symbol} rebuilt market price surface",
        )
        plot_surface_3d_strike(
            maturities,
            strike_grid,
            projected,
            f"{symbol} projected no-arbitrage price surface",
        )
    else:
        print("Only one maturity available: skipping 3D price surface plots.")

    plot_slice_strike(
        maturities,
        strike_grid,
        surface_mid,
        projected,
        i=min(2, len(maturities) - 1),
    )

    if len(maturities) >= 2:
        plot_iv_surface_3d_strike(
            maturities,
            strike_grid,
            iv_surface_before,
            f"{symbol} implied vol surface (rebuilt)",
        )
        plot_iv_surface_3d_strike(
            maturities,
            strike_grid,
            iv_surface_after,
            f"{symbol} implied vol surface (projected)",
            smooth_sigma=0.5,
        )
    else:
        plot_iv_slice_strike(
            maturities,
            strike_grid,
            iv_surface_before,
            iv_surface_after,
            i=0,
        )


if __name__ == "__main__":
    main()