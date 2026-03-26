from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.ndimage import gaussian_filter

from src.market.strike_grid import make_common_strike_grid_from_data, make_strike_surface, find_compatible_maturities_for_common_grid
from data.code.build_csv import load_options_from_csv, select_one_quote_date
from src.market.rebuild_surface_strike import rebuild_surface_from_dataframe_on_strikes
from src.market.projection_slice import project_surface_noarb
from src.market.price_to_iv_surface import price_surface_to_iv_surface
from src.validation.StaticArbitrageReport import check_static_no_arb



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
    symbol = "SPX"

    # 1) load full dataset
    df, _ = load_options_from_csv("data/combined_options_data.csv")
    print("Total cleaned rows:", len(df))

    # 2) select one quote date only
    # Mets une date explicite si tu veux, sinon None pour choisir le jour le plus fourni
    df, quote_date, spot = select_one_quote_date(df, quote_date="15-08-2013")

    print(f"{symbol} quote date:", quote_date.date())
    print(f"{symbol} spot:", spot)
    print("Rows for selected day:", len(df))
    print(df.head())

    # 3) maturities with enough quotes
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

    maturities = sorted(df["T"].unique().tolist())

    # 4) common strike grid
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

    for T in maturities:
        df_T = df[df["T"] == T]
        print(
            f"T={T:.3f} | strike range = "
            f"[{df_T['strike'].min():.2f}, {df_T['strike'].max():.2f}] | "
            f"n={len(df_T)}"
        )

    # 5) rebuild surfaces directly from df
    surface_mid = rebuild_surface_from_dataframe_on_strikes(
        df=df,
        maturities=maturities,
        strike_grid=strike_grid,
        field="mid",
    )

    spread_surface = rebuild_surface_from_dataframe_on_strikes(
        df=df,
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

    # 6) check before
    rep_before = check_static_no_arb(surface_mid, strikes_surface, eps=1e-6)
    print("\nBefore projection:")
    print(rep_before)
    print("OK before:", rep_before.ok())

    # 7) projection
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

    # 8) check after
    rep_after = check_static_no_arb(projected, strikes_surface, eps=1e-6)
    print("\nAfter projection:")
    print(rep_after)
    print("OK after:", rep_after.ok())
    print("calendar worst", rep_after.calendar.worst_value, "at", rep_after.calendar.worst_location)
    print("butterfly worst", rep_after.butterfly.worst_value, "at", rep_after.butterfly.worst_location)
    print("strike mono worst", rep_after.strike_mono.worst_value, "at", rep_after.strike_mono.worst_location)

    # 9) IV
    iv_surface_before = price_surface_to_iv_surface(
        surface_mid,
        strikes_surface,
        maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    price_smoothed = gaussian_filter(np.array(projected), sigma=0.5)

    iv_surface_after = price_surface_to_iv_surface(
        price_smoothed.tolist(),
        strikes_surface,
        maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    # 10) plots
    title_prefix = f"{symbol} {quote_date.date()}"

    if len(maturities) >= 2:
        plot_surface_3d_strike(
            maturities,
            strike_grid,
            surface_mid,
            f"{title_prefix} rebuilt market price surface",
        )
        plot_surface_3d_strike(
            maturities,
            strike_grid,
            projected,
            f"{title_prefix} projected no-arbitrage price surface",
        )

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
            f"{title_prefix} implied vol surface (rebuilt)",
        )
        plot_iv_surface_3d_strike(
            maturities,
            strike_grid,
            iv_surface_after,
            f"{title_prefix} implied vol surface (projected)",
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