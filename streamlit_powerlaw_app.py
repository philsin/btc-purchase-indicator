# streamlit_powerlaw_app.py
"""BTC Purchase Indicator – long‑term power‑law dashboard
--------------------------------------------------------
* Fetches daily BTC/USD closes (CoinMetrics free API – no key needed).
* Fits a log‑log OLS regression (price vs. days‑since‑genesis).
* Draws the mid‑line plus ±*σ* support / resistance bands.
* 21 million terminal supply option to display Market‑Cap.
* Highlights current zone: 🟢 Value, ⚪ Neutral, 🔴 Frothy.
* Works on Streamlit Cloud + mobile browsers.
"""

from __future__ import annotations
import datetime as dt
import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

################################################################################
# 1. CONFIG
################################################################################
CM_ENDPOINT = (
    "https://api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc"
    "&metrics=PriceUSD&frequency=1d&page_size=10000"
)
GENESIS = dt.date(2009, 1, 3)
SUPPLY_FULL = 21_000_000  # fully‑diluted supply for MCap

################################################################################
# 2. HELPERS
################################################################################

def days_since_genesis(d: pd.Timestamp) -> float:
    return (d.date() - GENESIS).days or 1  # avoid log10(0)


def fetch_prices() -> pd.DataFrame:
    """Download daily BTC/USD closes from CoinMetrics (≈ 12 k rows)."""
    r = requests.get(CM_ENDPOINT, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = js["data"]
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"PriceUSD": "price"})
    df = df[["time", "price"]].astype({"price": float})
    df = df.sort_values("time").reset_index(drop=True)
    return df


def fit_power_law(df: pd.DataFrame) -> Tuple[np.ndarray, float, float]:
    """Return fitted mid‑line and sigma of log‑residuals."""
    x = df["days"].values
    y = df["price"].values
    A, B = np.polyfit(np.log10(x), np.log10(y), 1)  # log10(y) = A + B*log10(x)
    mid = 10 ** (A + B * np.log10(x))
    sigma = (np.log10(y) - np.log10(mid)).std(ddof=1)
    return mid, A, B, sigma


def abbreviate(num: float, with_dollar: bool = False) -> str:
    prefix = "$" if with_dollar else ""
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"{prefix}{num/1e12:.1f}T"
    if abs_num >= 1e9:
        return f"{prefix}{num/1e9:.1f}B"
    if abs_num >= 1e6:
        return f"{prefix}{num/1e6:.1f}M"
    if abs_num >= 1e3:
        return f"{prefix}{num/1e3:.1f}K"
    return f"{prefix}{num:.2f}"

################################################################################
# 3. STREAMLIT APP
################################################################################

st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

# Sidebar controls
st.sidebar.header("Settings")
sigma_mult = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.5)
show_mcap = st.sidebar.toggle("Show fully‑diluted Market‑Cap (21 M)")

@st.cache_data(ttl=3600)
def load_data():
    df = fetch_prices()
    df["days"] = df["time"].apply(days_since_genesis)
    mid, A, B, sigma = fit_power_law(df)
    df["mid"] = mid
    df["support"] = 10 ** (np.log10(mid) - sigma_mult * sigma)
    df["resist"] = 10 ** (np.log10(mid) + sigma_mult * sigma)
    return df, sigma

df, sigma = load_data()

# Compute PL ratio and status
latest = df.iloc[-1]
pl_ratio = latest["price"] / latest["mid"]
if pl_ratio < 10 ** (-sigma_mult * sigma):
    zone = "🟢 Value"
elif pl_ratio > 10 ** (sigma_mult * sigma):
    zone = "🔴 Frothy"
else:
    zone = "⚪ Neutral"

st.markdown(f"## {zone} &nbsp;&nbsp; _({dt.date.today()})_")

# Build Plotly figure
hover_tmpl = "%{customdata[0]}<br>%{customdata[1]} | %{customdata[2]}<extra></extra>"
fig = go.Figure()
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")

# Bands
for name, col, width in [("support", "red", 1), ("resist", "purple", 1)]:
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[name],
            line=dict(color=col, width=width, dash="dash"),
            name="",
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Mid‑line
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["mid"],
        line=dict(color="green", width=1),
        name="",
        showlegend=False,
        hoverinfo="skip",
    )
)

# Price series
custom = np.column_stack(
    [df["time"].dt.strftime("%b %Y"),
     df["price"].apply(abbreviate, with_dollar=False if show_mcap else True),
     (df["price"]*SUPPLY_FULL if show_mcap else df["price"]).apply(lambda x: abbreviate(x, True))]
)
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["price"],
        mode="lines",
        line=dict(color="orange", width=1.5),
        customdata=custom,
        hovertemplate=hover_tmpl,
        name="",
        showlegend=False,
    )
)

fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    height=550,
    yaxis_title="USD" if not show_mcap else "Market Cap (USD)",
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("Data: CoinMetrics • Model: log‑log power‑law ±σ bands • Supply = 21 M BTC")
