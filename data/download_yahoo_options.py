from __future__ import annotations

import datetime as dt
from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from src.market.grid import make_default_grid
from src.market.rebuild_surface import rebuild_price_surface_from_snapshot
from src.validation.StaticArbitrageReport import check_static_no_arb
from src.sanos.projection_slice import project_slice_noarb
from src.market.price_to_iv_surface import price_surface_to_iv_surface

def download_yahoo_calls(symbol: str = "QQQ", n_expiries: int = 365) -> tuple[pd.DataFrame, float]:
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
        k_min, k_max = -0.35, 0.35
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

    cols = ["expiry", "T", "strike", "bid", "ask", "mid", "spread", "impliedVolatility", "type", "k"]
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
        })
    return snapshot


def build_strikes_surface(grid, spot: float, rate: float = 0.0, div: float = 0.0):
    strikes_surface = []
    for T in grid.maturities:
        forward = spot * np.exp((rate - div) * T)
        strikes_surface.append(grid.strikes_from_forward(forward))
    return strikes_surface


def project_surface_slice_by_slice(price_surface, strikes_surface):
    clean_surface = []
    for C_row, K_row in zip(price_surface, strikes_surface):
        res = project_slice_noarb(K_row, C_row, eps=1e-12)
        clean_surface.append(res.C_clean)
    return clean_surface


def plot_surface_3d(grid, surface, title):
    T = np.array(grid.maturities, dtype=float)
    k = np.array(grid.k_grid, dtype=float)
    Z = np.array(surface, dtype=float)

    Kmesh, Tmesh = np.meshgrid(k, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Kmesh, Tmesh, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("maturity T")
    ax.set_zlabel("Call price")
    ax.set_title(title)
    plt.show()


def plot_slice(grid, strikes_surface, surface_before, surface_after, i=3):
    K = strikes_surface[i]
    C_before = surface_before[i]
    C_after = surface_after[i]
    T = grid.maturities[i]

    plt.figure()
    plt.plot(K, C_before, label="Rebuilt Yahoo surface", marker="o", markersize=3)
    plt.plot(K, C_after, label="Projected no-arb surface", marker="o", markersize=3)
    plt.xlabel("Strike K")
    plt.ylabel("Call price")
    plt.title(f"Yahoo slice projection at T={T:.3f}y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iv_surface_3d(grid, iv_surface, title):
    T = np.array(grid.maturities, dtype=float)
    k = np.array(grid.k_grid, dtype=float)
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

    if len(T) < 2:
        print("Only one maturity available: use a 2D IV slice instead of a 3D surface.")
        return

    Kmesh, Tmesh = np.meshgrid(k, T)

    # masque les NaN pour éviter les surfaces vides/cassées
    Z_masked = np.ma.masked_invalid(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Kmesh, Tmesh, Z_masked, linewidth=0, antialiased=True)

    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("maturity T")
    ax.set_zlabel("implied volatility")
    ax.set_title(title)

    plt.show()

def plot_iv_slice(grid, iv_surface_before, iv_surface_after, i=0):
    k = np.array(grid.k_grid, dtype=float)
    iv_before = np.array(iv_surface_before[i], dtype=float)
    iv_after = np.array(iv_surface_after[i], dtype=float)

    plt.figure()
    plt.plot(k, iv_before, label="IV before projection", marker="o", markersize=3)
    plt.plot(k, iv_after, label="IV after projection", marker="o", markersize=3)
    plt.xlabel("log-moneyness k")
    plt.ylabel("implied volatility")
    plt.title(f"Implied vol slice at T={grid.maturities[i]:.3f}y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    symbol = "QQQ"

    # 1) download + clean
    df, spot = download_yahoo_calls(symbol=symbol, n_expiries=365)
    print(f"{symbol} spot:", spot)
    print("Number of cleaned quotes:", len(df))
    print(df.head())

    # 2) build snapshot
    snapshot = dataframe_to_snapshot(df)

    counts = df.groupby("T").size().reset_index(name="n_quotes")
    print(counts)

    valid_maturities = counts[counts["n_quotes"] >= 10]["T"].tolist()
    print("Valid maturities:", valid_maturities)

    df = df[df["T"].isin(valid_maturities)].copy()
    snapshot = dataframe_to_snapshot(df)

    # 3) grid with Yahoo maturities
    market_maturities = sorted(df["T"].unique().tolist())
    grid = make_default_grid(
        maturities=market_maturities,
        k_min=-0.3, #0.30 pour SPY, QQQ, 0.60 pour TSLA
        k_max=0.3, #0.30 pour SPY, QQQ, 0.60 pour TSLA
        n_k=61,
    )

    # 4) rebuild surface from irregular quotes
    surface_mid = rebuild_price_surface_from_snapshot(snapshot, grid, field="mid")

    # 5) build strike surface
    strikes_surface = build_strikes_surface(grid, spot, rate=0.0, div=0.0)

    print("Surface shape:", len(surface_mid), "x", len(surface_mid[0]))

    # 6) static no-arb check before
    rep_before = check_static_no_arb(surface_mid, strikes_surface, eps=1e-08)
    print("\nBefore projection:")
    print(rep_before)
    print("OK before:", rep_before.ok())

    # 7) projection
    projected = project_surface_slice_by_slice(surface_mid, strikes_surface)

    # 8) static no-arb check after
    rep_after = check_static_no_arb(projected, strikes_surface, eps=1e-08)
    print("\nAfter projection:")
    print(rep_after)
    print("OK after:", rep_after.ok())

    # 9) convert to implied vol surfaces
    iv_surface_before = price_surface_to_iv_surface(
        surface_mid,
        strikes_surface,
        grid.maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    iv_surface_after = price_surface_to_iv_surface(
        projected,
        strikes_surface,
        grid.maturities,
        spot=spot,
        rate=0.0,
        div=0.0,
    )

    # 10) plots
        # price plots
    if len(grid.maturities) >= 2:
        plot_surface_3d(grid, surface_mid, f"{symbol} rebuilt market price surface")
        plot_surface_3d(grid, projected, f"{symbol} projected no-arbitrage price surface")
    else:
        print("Only one maturity available: skipping 3D price surface plots.")

    plot_slice(grid, strikes_surface, surface_mid, projected, i=min(2, len(grid.maturities) - 1))

    # IV plots
    if len(grid.maturities) >= 2:
        plot_iv_surface_3d(grid, iv_surface_before, f"{symbol} implied vol surface (rebuilt)")
        plot_iv_surface_3d(grid, iv_surface_after, f"{symbol} implied vol surface (projected)")
    else:
        plot_iv_slice(grid, iv_surface_before, iv_surface_after, i=0)
    
if __name__ == "__main__":
    main()