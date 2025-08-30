#!/usr/bin/env python3
"""
BTC Purchase Indicator — Midline Percentiles (responsive with adjustable width)

This script builds a static website that plots bitcoin’s price relative to various denominators
(USD, gold, S&P 500, etc.) and overlays percentile rails derived from a robust power-law
relationship between price and time. The UI includes a panel with statistics, a denominator
drop‑down, a date lock, live hover, copy chart, and an adjustable width slider so you can
change the chart-to-panel ratio on the fly.  Rails and percentiles follow a symmetric
winsorized residual method, which produces midline and bands similar to well‑known
power‑law charts (e.g., Porkopolis) while controlling early-cycle outliers.

Key features:
- Fits the median (q50) regression line in log–log space (price vs. years since Genesis block).
- Computes 2.5/20/80/97.5 percentile rails as constant log offsets from that midline, using
  winsorized (clipped) residuals and symmetric quantiles to avoid extreme early outliers.
- All rails are parallel and straight in log–log space; p% is measured within the 2.5–97.5 band.
- Responsive layout: left/right panes flex using CSS variables. A slider adjusts the chart’s
  width percentage; a ResizeObserver ensures Plotly resizes properly on mobile and desktop.
- Handles any number of additional denominators found in data/denominator_*.csv.
- Outputs a standalone HTML file (docs/index.html) ready to publish on GitHub Pages.

Adjustable constants (top of script):
  EPS_LOG:     separates rails to avoid overlap (≈2.3% shift).
  RESID_WINSOR: fraction of tails clipped when computing residual quantiles (0.02 = 2%).
  SYMMETRIC_RAILS: if True, forces upper/lower percentiles to be equal distances from median.

Run this script from the root of the repository. It assumes BTC data in data/btc_usd.csv
and optional denominators in data/denominator_*.csv. If btc_usd.csv doesn’t exist, it
downloads daily BTC prices from Blockchain.com. The output site writes to docs/index.html.

Usage:
  python build_static.py

"""
import os
import glob
import json
import io
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
from statsmodels.regression.quantile_regression import QuantReg

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "data"
BTC_FILE = os.path.join(DATA_DIR, "btc_usd.csv")
OUTPUT_HTML = "docs/index.html"

GENESIS_DATE = datetime(2009, 1, 3)
END_PROJ = datetime(2040, 12, 31)

# Rails parameters
EPS_LOG = 0.010      # small positive offset to prevent rails from touching
RESID_WINSOR = 0.02  # winsorize top/bottom 2% of residuals
SYMMETRIC_RAILS = True

# Colour palette: Floor→20→50→80→Ceiling (red→orange→gold→green→darkgreen)
COL_FLOOR   = "#D32F2F"
COL_20      = "#F57C00"
COL_50      = "#FBC02D"
COL_80      = "#66BB6A"
COL_CEILING = "#2E7D32"
COL_BTC     = "#000000"

# --------------------------------------------------
# Data loading and preparation
# --------------------------------------------------
def years_since_genesis(dates: pd.Series) -> pd.Series:
    """Convert timestamps to fractional years since the Bitcoin genesis block.

    We add 1 day to avoid zero in the log-x transform.
    """
    delta = (pd.to_datetime(dates) - GENESIS_DATE) / np.timedelta64(1, "D")
    return (delta.astype(float) / 365.25) + (1.0 / 365.25)

def fetch_btc_csv() -> pd.DataFrame:
    """Load BTC price data from BTC_FILE or download from Blockchain.com.

    Returns a DataFrame with columns ["date","price"].
    """
    if os.path.exists(BTC_FILE):
        return pd.read_csv(BTC_FILE, parse_dates=["date"])
    os.makedirs(DATA_DIR, exist_ok=True)
    # Use blockchain.info daily close data as fallback
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    text = requests.get(url, timeout=30).text.strip()
    lines = text.splitlines()
    # Determine header presence
    if lines[0].lower().startswith("timestamp"):
        df = pd.read_csv(io.StringIO(text))
        date_col, val_col = df.columns[0], df.columns[1]
        df = df.rename(columns={date_col: "date", val_col: "price"})
    else:
        df = pd.read_csv(io.StringIO(text), header=None, names=["date", "price"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("date")
    df.to_csv(BTC_FILE, index=False)
    return df

def load_denominators() -> dict:
    """Scan data/denominator_*.csv and load each as a DataFrame.

    Returns a mapping from uppercase key to DataFrame with columns ["date","price"].
    """
    denoms = {}
    pattern = os.path.join(DATA_DIR, "denominator_*.csv")
    for path in glob.glob(pattern):
        key = os.path.splitext(os.path.basename(path))[0].replace("denominator_", "").upper()
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            price_col = [c for c in df.columns if c.lower() != "date"][0]
            df = df.rename(columns={"date": "date", price_col: "price"})
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.dropna().sort_values("date")
            denoms[key] = df
        except Exception as e:
            print(f"[warn] skipping {path}: {e}")
    return denoms

def quantile_fit(x: np.ndarray, y: np.ndarray, q: float = 0.5):
    """Quantile regression (log–log) returning intercept, slope, residuals and mask."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xlog = np.log10(x[mask])
    ylog = np.log10(y[mask])
    X = pd.DataFrame({"const": 1.0, "logx": xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0, b = float(res.params["const"]), float(res.params["logx"])
    resid = ylog - (a0 + b * xlog)
    return a0, b, resid, mask

def winsorize(arr: np.ndarray, p: float) -> np.ndarray:
    """Clip array tails at fraction p (both sides)."""
    lo, hi = np.nanquantile(arr, p), np.nanquantile(arr, 1 - p)
    return np.clip(arr, lo, hi)

def symmetric_quantiles(resid: np.ndarray, q: float):
    """Return symmetric residual offsets (lower, upper) such that ±d contains q.

    If SYMMETRIC_RAILS is True, ensures the distance from the median is the same
    for the lower and upper quantiles.
    """
    median = float(np.nanmedian(resid))
    d_up = float(np.nanquantile(resid - median, q))
    d_down = float(np.nanquantile(median - resid, q))
    d = max(d_up, d_down)
    return median - d, median + d

def defaults_for_series(x_years: pd.Series, y_vals: pd.Series) -> dict:
    """Compute default midline and residual offsets for 2.5, 20, 80, 97.5 percentiles."""
    a0, b, resid, _ = quantile_fit(x_years.values, y_vals.values, q=0.5)
    r = resid.copy()
    if RESID_WINSOR > 0:
        r = winsorize(r, RESID_WINSOR)
    if SYMMETRIC_RAILS:
        c025, c975 = symmetric_quantiles(r, 0.975)
        c200, c800 = symmetric_quantiles(r, 0.800)
    else:
        c025 = float(np.nanquantile(r, 0.025))
        c975 = float(np.nanquantile(r, 0.975))
        c200 = float(np.nanquantile(r, 0.200))
        c800 = float(np.nanquantile(r, 0.800))
    # separate rails slightly
    c025 -= EPS_LOG
    c975 += EPS_LOG
    return {"a0": a0, "b": b, "c025": c025, "c200": c200, "c800": c800, "c975": c975}

def build_payload(df: pd.DataFrame, denom_key: str | None) -> dict:
    """Prepare series and defaults for a given denominator key."""
    if denom_key is None:
        y = df["btc"]
        label = "BTC / USD"
    else:
        denom_col = denom_key.lower()
        y = df["btc"] / df[denom_col]
        label = f"BTC / {denom_key.upper()}"
    d = defaults_for_series(df["x_years"], y)
    return {
        "label": label,
        "x_main": df["x_years"].tolist(),
        "y_main": y.tolist(),
        "date_iso_main": df["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "defaults": d,
    }

def ensure_docs_dir():
    docs = os.path.dirname(OUTPUT_HTML)
    os.makedirs(docs, exist_ok=True)

def write_html(html_content: str):
    ensure_docs_dir()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Wrote {OUTPUT_HTML}")

# Create x_grid globally for use in payloads
x_start = float(base["x_years"].iloc[0])
x_end   = float(years_since_genesis(pd.Series([END_PROJ])).iloc[0])
x_grid = np.logspace(np.log10(max(1e-6, x_start)), np.log10(x_end), 700)

# Build PRECOMP data
btc_df = fetch_btc_csv()
base_df = btc_df.rename(columns={"price": "btc"}).sort_values("date").reset_index(drop=True)
denoms = load_denominators()
for k, df in denoms.items():
    base_df = base_df.merge(df.rename(columns={"price": k.lower()}), on="date", how="left")
base_df["x_years"] = years_since_genesis(base_df["date"])
base_df["date_iso"] = base_df["date"].dt.strftime("%Y-%m-%d")

PRECOMP = {"USD": build_payload(base_df, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base_df, k)

# Define plotly figure and compute rails placeholders
fig = go.Figure([
    # placeholder rails; updated by JS
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="Floor", line=dict(color=COL_FLOOR)),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="20%", line=dict(color=COL_20, dash="dot")),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="50%", line=dict(color=COL_50, width=3)),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="80%", line=dict(color=COL_80, dash="dot")),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="Ceiling", line=dict(color=COL_CEILING)),
    go.Scatter(x=PRECOMP["USD"]["x_main"], y=PRECOMP["USD"]["y_main"], name="BTC / USD",
               mode="lines", line=dict(color=COL_BTC)),
    go.Scatter(x=PRECOMP["USD"]["x_main"], y=PRECOMP["USD"]["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
])
fig.update_layout(
    template="plotly_white",
    hovermode="x",
    showlegend=True,
    xaxis=dict(type="log", title=None),
    yaxis=dict(type="log", title=PRECOMP["USD"]["label"]),
    margin=dict(l=70, r=420, t=70, b=70)
)
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

# Generate final HTML
html = html_template.safe_substitute(
    plot_html=plot_html,
    precomp_json=json.dumps(PRECOMP),
    genesis_iso=GENESIS_DATE.strftime("%Y-%m-%d"),
    COL_FLOOR=COL_FLOOR, COL_20=COL_20, COL_50=COL_50, COL_80=COL_80, COL_CEILING=COL_CEILING
)
write_html(html)
