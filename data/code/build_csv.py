from __future__ import annotations

import numpy as np
import pandas as pd


def load_options_from_csv(filepath: str) -> tuple[pd.DataFrame, float]:
    usecols = [
        "QUOTE_DATE",
        "UNDERLYING_LAST",
        "DTE",
        "STRIKE",
        "C_BID",
        "C_ASK",
        "C_IV",
        "C_VOLUME",
    ]

    df = pd.read_csv(filepath, usecols=usecols)

    out = pd.DataFrame()
    out["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], dayfirst=True, errors="coerce")
    out["spot"] = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    out["strike"] = pd.to_numeric(df["STRIKE"], errors="coerce")
    out["bid"] = pd.to_numeric(df["C_BID"], errors="coerce")
    out["ask"] = pd.to_numeric(df["C_ASK"], errors="coerce")
    out["T"] = pd.to_numeric(df["DTE"], errors="coerce") / 365.0
    out["impliedVolatility"] = pd.to_numeric(df["C_IV"], errors="coerce")
    out["volume"] = pd.to_numeric(df["C_VOLUME"], errors="coerce")

    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = out["ask"] - out["bid"]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["QUOTE_DATE", "T", "strike", "bid", "ask", "mid", "spot"])

    out = out[(out["bid"] > 0) & (out["ask"] > out["bid"])]
    out = out[out["T"] > 0.01]
    out = out[out["mid"] > 0.5]
    out = out[out["spread"] / out["mid"] < 0.3]

    out["type"] = "call"

    print("Loaded rows:", len(out))
    print(out.head())

    # spot global unused later except convenience
    spot = float(out["spot"].median())
    return out, spot

def select_one_quote_date(df: pd.DataFrame, quote_date: str | None = None) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """
    Keep one market date only.
    If quote_date is None, use the most populated QUOTE_DATE.
    """
    if "QUOTE_DATE" not in df.columns:
        raise ValueError("DataFrame must contain QUOTE_DATE")

    if quote_date is None:
        counts = df.groupby("QUOTE_DATE").size().sort_values(ascending=False)
        selected_date = pd.Timestamp(counts.index[0])
    else:
        selected_date = pd.Timestamp(quote_date)

    df_day = df[df["QUOTE_DATE"] == selected_date].copy()

    if df_day.empty:
        raise ValueError(f"No data found for QUOTE_DATE={selected_date.date()}")

    # representative spot for that day
    spot_day = float(df_day["spot"].median())

    # recompute k using day-specific spot
    df_day["k"] = np.log(df_day["strike"] / spot_day)

    # aggregate duplicates within that day
    agg_map = {
        "bid": "mean",
        "ask": "mean",
        "mid": "mean",
        "spread": "mean",
        "impliedVolatility": "mean",
        "volume": "sum",
        "k": "mean",
        "spot": "median",
    }

    df_day = (
        df_day.groupby(["QUOTE_DATE", "T", "strike"], as_index=False)
        .agg(agg_map)
        .sort_values(["T", "strike"])
        .reset_index(drop=True)
    )

    return df_day, selected_date, spot_day

def load_aapl_options_one_day(path: str, quote_date: str):
    df = pd.read_csv(path)

    # clean columns
    df.columns = df.columns.str.strip().str.replace('[\\[\\]]', '', regex=True)

    # convert dates
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])

    # filter ONE day
    selected_date = pd.to_datetime(quote_date)
    df = df[df["QUOTE_DATE"] == selected_date].copy()

    if df.empty:
        raise ValueError(f"No data for date {quote_date}")

    print("Selected date:", selected_date.date())
    print("Rows:", len(df))

    # rename
    df = df.rename(columns={
        "STRIKE": "strike",
        "C_BID": "bid",
        "C_ASK": "ask",
        "C_IV": "impliedVolatility",
        "UNDERLYING_LAST": "spot"
    })

    # compute maturity (IMPORTANT: from dates, not DTE)
    df["T"] = (df["EXPIRE_DATE"] - df["QUOTE_DATE"]).dt.days / 365.0

    # mid / spread
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]

    # clean
    df = df[
        (df["bid"] > 0) &
        (df["ask"] > df["bid"]) &
        (df["mid"] > 0) &
        (df["T"] > 0.01)
    ]

    # spot
    spot = df["spot"].iloc[0]

    # log-moneyness
    df["k"] = np.log(df["strike"] / spot)

    # filters
    df = df[
        (df["k"].between(-0.5, 0.5)) &
        (df["spread"] / df["mid"] < 0.5)
    ]

    df = df.sort_values(["T", "strike"]).reset_index(drop=True)

    print("Clean rows:", len(df))
    print("Spot:", spot)

    return df, spot