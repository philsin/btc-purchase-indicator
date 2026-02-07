#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — stationary rails with composite signal (no cross-denom voting),
cursor-driven panel (future uses 50%), even-year x labels after 2020, halvings & liquidity toggles,
info modal, and user-defined horizontal Level lines (per denominator).

Writes: docs/index.html
"""

import os, io, glob, json, re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ───────────────────────────── Config ─────────────────────────────
DATA_DIR     = "data"
BTC_FILE     = os.path.join(DATA_DIR, "btc_usd.csv")
OUTPUT_HTML  = "docs/index.html"

GENESIS_DATE = datetime(2009, 1, 3)
END_PROJ     = datetime(2040, 12, 31)
# Chart starts from Genesis block, not an arbitrary later date
X_START_DATE = GENESIS_DATE

RESID_WINSOR     = 0.02
EPS_LOG_SPACING  = 0.010
COL_BTC          = "#000000"

# Halvings
PAST_HALVINGS   = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
FUTURE_HALVINGS = ["2028-04-20", "2032-04-20", "2036-04-19", "2040-04-19"]

# Liquidity (65-month wave)
LIQ_PEAK_ANCHOR_ISO = "2015-02-01"  # PEAK (red)
LIQ_PERIOD_MONTHS   = 65
LIQ_START_ISO       = "2009-01-03"

# Composite weights (tweakable)
W_P   = 1.00   # rails p (centered at 50)
W_LIQ = 0.60   # liquidity rising/falling (cosine of phase)
W_HALV= 0.60   # halving window (+1 pre, +0.5 post, −0.5 late)

# Auto denominators
AUTO_DENOMS = {
    "GOLD": {"path": os.path.join(DATA_DIR, "denominator_gold.csv"),
             "url":  "https://stooq.com/q/d/l/?s=xauusd&i=d", "parser":"stooq"},
    "SPX":  {"path": os.path.join(DATA_DIR, "denominator_spx.csv"),
             "url":  "https://stooq.com/q/d/l/?s=%5Espx&i=d", "parser":"stooq"},
    "ETH":  {"path": os.path.join(DATA_DIR, "denominator_eth.csv"),
             "url":  "https://stooq.com/q/d/l/?s=ethusd&i=d", "parser":"stooq"},
}
UA = {"User-Agent":"btc-indicator/1.0"}

# Smart caching: data older than this is considered stale and will be refreshed
CACHE_MAX_AGE_HOURS = 12  # Refresh data every 12 hours

def is_cache_stale(filepath, max_age_hours=CACHE_MAX_AGE_HOURS):
    """Check if a cached file is older than max_age_hours."""
    if not os.path.exists(filepath):
        return True
    file_mtime = os.path.getmtime(filepath)
    age_hours = (datetime.now().timestamp() - file_mtime) / 3600
    return age_hours > max_age_hours

def data_needs_update(filepath, df=None):
    """Check if data file is stale OR if the last data point is old."""
    if is_cache_stale(filepath):
        return True
    # Also check if the data itself is outdated (last row is old)
    if df is not None and "date" in df.columns:
        last_date = pd.to_datetime(df["date"]).max()
        days_old = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
        if days_old > 2:  # Data is more than 2 days old
            return True
    return False

# ───────────────────── Time Periods (for zoom presets) ─────────────────────
# Cycle dates based on halving schedule
CYCLE_1_START = "2009-01-03"  # Genesis
CYCLE_2_START = "2012-11-28"  # 1st halving
CYCLE_3_START = "2016-07-09"  # 2nd halving
CYCLE_4_START = "2020-05-11"  # 3rd halving
CYCLE_5_START = "2024-04-20"  # 4th halving

TIME_PERIODS = {
    "all": {"label": "All Data", "start": "2009-01-03"},
    "10y": {"label": "Last 10 Years", "days_back": 3652},
    "5y": {"label": "Last 5 Years", "days_back": 1826},
    "2y": {"label": "Last 2 Years", "days_back": 730},
    "1y": {"label": "Last 1 Year", "days_back": 365},
    "cycle": {"label": "Current Cycle", "start": CYCLE_5_START},
}

# ───────────────────── Helpers / fetchers ─────────────────────
def days_since_genesis(dates):
    """Days since Bitcoin Genesis Block (Jan 3, 2009), starting at day 1."""
    d = pd.to_datetime(dates)
    delta_days = (d - GENESIS_DATE) / np.timedelta64(1, "D")
    return delta_days.astype(float) + 1.0  # day 1 = genesis

def years_since_genesis(dates):
    d = pd.to_datetime(dates)
    delta_days = (d - GENESIS_DATE) / np.timedelta64(1, "D")
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)

def _fetch_btc_from_api() -> pd.DataFrame:
    """Fetch BTC price data from blockchain.info API."""
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    raw = r.text.strip()
    if raw.splitlines()[0].lower().startswith("timestamp"):
        df = pd.read_csv(io.StringIO(raw))
        ts = [c for c in df.columns if c.lower().startswith("timestamp")][0]
        val = [c for c in df.columns if c.lower().startswith("value")][0]
        df = df.rename(columns={ts:"date", val:"price"})
    else:
        df = pd.read_csv(io.StringIO(raw), header=None, names=["date","price"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.sort_values("date").dropna()

def _fetch_btc_from_coingecko() -> pd.DataFrame:
    """Fallback: Fetch BTC price data from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max"
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    js = r.json()
    rows = js.get("prices") or []
    if not rows: raise ValueError("CoinGecko returned no BTC data")
    df = pd.DataFrame(rows, columns=["ms", "price"])
    df["date"] = pd.to_datetime(df["ms"], unit="ms").dt.normalize()
    out = df[["date","price"]].sort_values("date").dropna()
    # CoinGecko can have multiple entries per day, take the last one
    out = out.groupby("date", as_index=False).last()
    return out

def fetch_btc_csv() -> pd.DataFrame:
    """
    Fetch BTC/USD price data with smart caching.
    - Returns cached data if fresh (< CACHE_MAX_AGE_HOURS old)
    - Refreshes from API if cache is stale or data is outdated
    - Falls back to CoinGecko if primary source fails
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if we have fresh cached data
    if os.path.exists(BTC_FILE):
        df = pd.read_csv(BTC_FILE, parse_dates=["date"])
        df = df.sort_values("date").dropna()
        if not data_needs_update(BTC_FILE, df):
            print(f"[cache] Using cached BTC data (last: {df['date'].max().date()})")
            return df
        print(f"[cache] BTC data is stale, refreshing...")

    # Fetch fresh data
    df = None
    for fetch_fn, name in [(_fetch_btc_from_api, "blockchain.info"),
                           (_fetch_btc_from_coingecko, "CoinGecko")]:
        try:
            df = fetch_fn()
            if df is not None and not df.empty:
                print(f"[fetch] Got {len(df)} rows from {name} (last: {df['date'].max().date()})")
                break
        except Exception as e:
            print(f"[warn] {name} failed: {e}")
            df = None

    if df is None or df.empty:
        # Last resort: use cached file even if stale
        if os.path.exists(BTC_FILE):
            print("[warn] All APIs failed, using stale cache")
            return pd.read_csv(BTC_FILE, parse_dates=["date"]).sort_values("date").dropna()
        raise ValueError("Could not fetch BTC data from any source")

    df.to_csv(BTC_FILE, index=False)
    return df

def _fetch_stooq_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)
    if "date" not in df.columns or "close" not in df.columns or df.empty:
        raise ValueError("stooq returned no usable columns")
    out = df[["date","close"]].rename(columns={"close":"price"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.sort_values("date").dropna()

def _fetch_eth_from_stooq() -> pd.DataFrame:
    return _fetch_stooq_csv("https://stooq.com/q/d/l/?s=ethusd&i=d")

def _fetch_eth_from_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=max"
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    js = r.json()
    rows = js.get("prices") or []
    if not rows: raise ValueError("coingecko empty ETH")
    df = pd.DataFrame(rows, columns=["ms", "price"])
    df["date"] = pd.to_datetime(df["ms"], unit="ms").dt.normalize()
    out = df[["date","price"]].sort_values("date").dropna()
    out = out.groupby("date", as_index=False).last()
    return out

def _fetch_eth_from_cryptocompare() -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&allData=true"
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    js = r.json()
    if js.get("Response")!="Success": raise ValueError("cryptocompare not success")
    rows = js.get("Data",{}).get("Data") or []
    if not rows: raise ValueError("cryptocompare empty")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df = df.rename(columns={"close":"price"})[["date","price"]].sort_values("date").dropna()
    return df

def _fetch_eth_from_binance() -> pd.DataFrame:
    base = "https://api.binance.com/api/v3/klines"
    params = {"symbol":"ETHUSDT","interval":"1d","limit":1000}
    out=[]; start=None
    while True:
        q=params.copy()
        if start is not None: q["startTime"]=start
        r=requests.get(base, params=q, timeout=30, headers=UA); r.raise_for_status()
        chunk=r.json()
        if not chunk: break
        out.extend(chunk)
        start=int(chunk[-1][6])+1
        if len(chunk)<params["limit"] or len(out)>100000: break
    if not out: raise ValueError("binance empty")
    cols=["openTime","open","high","low","close","volume","closeTime","qav","trades","takerBase","takerQuote","ignore"]
    df=pd.DataFrame(out, columns=cols)
    df["date"]=pd.to_datetime(df["closeTime"], unit="ms").dt.normalize()
    df["price"]=pd.to_numeric(df["close"], errors="coerce")
    return df[["date","price"]].sort_values("date").dropna()

def ensure_auto_denominators():
    """
    Ensure denominator data files exist and are fresh.
    Uses smart caching - refreshes if file is stale (> CACHE_MAX_AGE_HOURS old).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, info in AUTO_DENOMS.items():
        path = info["path"]

        # Check if we need to fetch (missing or stale)
        needs_fetch = not os.path.exists(path) or is_cache_stale(path)
        if not needs_fetch:
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                needs_fetch = data_needs_update(path, df)
                if not needs_fetch:
                    continue  # Cache is fresh
            except Exception:
                needs_fetch = True

        if needs_fetch:
            print(f"[cache] {key} data needs refresh...")

        try:
            if key == "ETH":
                df=None
                for fn in (_fetch_eth_from_stooq, _fetch_eth_from_coingecko,
                           _fetch_eth_from_cryptocompare, _fetch_eth_from_binance):
                    try:
                        df = fn()
                        if df is not None and not df.empty:
                            print(f"[fetch] Got {len(df)} rows for ETH")
                            break
                    except Exception:
                        df = None
                if df is None or df.empty: raise ValueError("ETH fetchers returned no data")
            else:
                df = _fetch_stooq_csv(info["url"]) if info["parser"]=="stooq" else None
                if df is not None and not df.empty:
                    print(f"[fetch] Got {len(df)} rows for {key}")
            if df is None or df.empty: raise ValueError(f"{key} returned no data")
            df.to_csv(path, index=False)
            print(f"[auto-denom] wrote {path} ({len(df)} rows, last: {df['date'].max().date()})")
        except Exception as e:
            print(f"[warn] could not fetch {key}: {e}")

def load_denominators():
    ensure_auto_denominators()
    out={}
    for p in glob.glob(os.path.join(DATA_DIR, "denominator_*.csv")):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            price_col = [c for c in df.columns if c.lower()!="date"][0]
            df = df.rename(columns={price_col:"price"})[["date","price"]]
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            out[key]=df.sort_values("date").dropna()
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    return out

# ───────────────────────── Fit / stationary rails ─────────────────────────
def winsorize(arr, p):
    lo, hi = np.nanquantile(arr, p), np.nanquantile(arr, 1-p)
    return np.clip(arr, lo, hi)

def quantile_fit_loglog(x_years, y_vals, q=0.5):
    x_years = np.asarray(x_years); y_vals = np.asarray(y_vals)
    mask = np.isfinite(x_years) & np.isfinite(y_vals) & (x_years>0) & (y_vals>0)
    xlog = np.log10(x_years[mask]); ylog = np.log10(y_vals[mask])
    X = pd.DataFrame({"const":1.0,"logx":xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = ylog - (a0 + b*xlog)
    return a0, b, resid

def build_support_constant_rails(x_years, y_vals):
    a0, b, resid = quantile_fit_loglog(x_years, y_vals, q=0.5)
    r = winsorize(resid, RESID_WINSOR) if RESID_WINSOR else resid
    med = float(np.nanmedian(r))
    q_grid = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(r, q_grid)
    off_grid = rq - med
    off_grid[0]  -= EPS_LOG_SPACING
    off_grid[-1] += EPS_LOG_SPACING
    # Power Law Oscillator: normalize raw residuals to [-1, +1]
    raw_resid = resid  # unwinsorized for oscillator
    resid_min = float(np.nanmin(raw_resid))
    resid_max = float(np.nanmax(raw_resid))
    return {"a0":a0, "b":b,
            "q_grid":[float(q) for q in q_grid],
            "off_grid":[float(v) for v in off_grid],
            "resid_min": resid_min,
            "resid_max": resid_max}

# ───────────────────────── Build model ─────────────────────────
btc = fetch_btc_csv().rename(columns={"price":"btc"})
denoms = load_denominators()
print("[denoms]", list(denoms.keys()))

base = btc.sort_values("date").reset_index(drop=True)
for key, df in denoms.items():
    base = base.merge(df.rename(columns={"price": key.lower()}), on="date", how="left")

base["x_years"]   = years_since_genesis(base["date"])
base["x_days"]    = days_since_genesis(base["date"])
base["date_iso"]  = base["date"].dt.strftime("%Y-%m-%d")

# Grid always starts from Genesis block for consistent power law visualization
# The actual data may start later, but the trend line extends to Genesis
first_data_dt = base["date"].iloc[0]
max_dt = END_PROJ

# Years-based: Genesis = ~0.00274 years (1/365.25) to avoid log(0)
x_genesis_years = 1.0 / 365.25  # Day 1 in years
x_end_years = float(years_since_genesis(pd.Series([max_dt])).iloc[0])
x_grid = np.logspace(np.log10(x_genesis_years), np.log10(x_end_years), 700)

# Days-based: Genesis = day 1
x_genesis_days = 1.0
x_end_days = float(days_since_genesis(pd.Series([max_dt])).iloc[0])
x_grid_days = np.logspace(np.log10(x_genesis_days), np.log10(x_end_days), 700)

# For chart initial range, start from first actual data point
x_start = float(years_since_genesis(pd.Series([first_data_dt])).iloc[0])

def year_ticks_log(last_dt):
    """Generate x-axis tick values and labels starting from Genesis year (2009)."""
    vals, labs = [], []
    for y in range(GENESIS_DATE.year, last_dt.year+1):
        d = datetime(y, 1, 1)
        if y > 2020 and (y % 2 == 1):  # hide odd years after 2020 for clarity
            continue
        vy = float(years_since_genesis(pd.Series([d])).iloc[0])
        if vy <= 0:
            continue
        vals.append(vy)
        labs.append(str(y))
    return vals, labs

def y_ticks():
    vals = [10**e for e in range(0,9)]
    labs = [f"{int(10**e):,}" for e in range(0,9)]
    return vals, labs

xtickvals, xticktext = year_ticks_log(max_dt)
ytickvals, yticktext = y_ticks()

def series_for_denom(df, key):
    if not key or key.lower() in ("usd", "none"):
        return df["btc"], "BTC / USD", "$", 2
    k = key.lower()
    if k in df.columns:
        return df["btc"]/df[k], f"BTC / {key.upper()}", "", 6
    return df["btc"], "BTC / USD", "$", 2

def resample_ohlc(df, y_col, freq='W'):
    """Resample to weekly or monthly OHLC candles."""
    df = df.set_index('date')
    ohlc = df[y_col].resample(freq).ohlc()
    ohlc = ohlc.dropna()
    ohlc = ohlc.reset_index()
    return ohlc

def build_payload(df, denom_key=None):
    y, label, unit, decimals = series_for_denom(df, denom_key)
    mask = np.isfinite(df["x_years"].values) & np.isfinite(y.values)
    xs_years = df["x_years"].values[mask]
    xs_days = df["x_days"].values[mask]
    ys = y.values[mask]
    dates = df["date_iso"].values[mask]
    support = build_support_constant_rails(xs_years, ys)

    # Also fit in days-space for display
    support_days = build_support_constant_rails(xs_days, ys)

    # Build OHLC for different candle intervals
    # CRITICAL: Use the properly denominated price series, not raw USD price
    df_ohlc = df.copy()
    df_ohlc['_denom_price'] = y.values  # This is already BTC/USD or BTC/GOLD etc.

    # Weekly OHLC (using correctly denominated price)
    ohlc_w = resample_ohlc(df_ohlc, '_denom_price', 'W')
    ohlc_w['x_years'] = years_since_genesis(ohlc_w['date'])
    ohlc_w['x_days'] = days_since_genesis(ohlc_w['date'])
    ohlc_w['date_iso'] = ohlc_w['date'].dt.strftime('%Y-%m-%d')

    # Monthly OHLC (using correctly denominated price)
    ohlc_m = resample_ohlc(df_ohlc, '_denom_price', 'ME')
    ohlc_m['x_years'] = years_since_genesis(ohlc_m['date'])
    ohlc_m['x_days'] = days_since_genesis(ohlc_m['date'])
    ohlc_m['date_iso'] = ohlc_m['date'].dt.strftime('%Y-%m-%d')

    return {
        "label": label, "unit": unit, "decimals": decimals,
        "x_main": xs_years.tolist(), "y_main": ys.tolist(),
        "x_days": xs_days.tolist(),
        "date_iso_main": dates.tolist(),
        "x_grid": x_grid.tolist(),
        "x_grid_days": x_grid_days.tolist(),
        "support": support,
        "support_days": support_days,
        # Weekly candles
        "ohlc_w": {
            "x_years": ohlc_w["x_years"].tolist(),
            "x_days": ohlc_w["x_days"].tolist(),
            "date_iso": ohlc_w["date_iso"].tolist(),
            "open": ohlc_w["open"].tolist(),
            "high": ohlc_w["high"].tolist(),
            "low": ohlc_w["low"].tolist(),
            "close": ohlc_w["close"].tolist()
        },
        # Monthly candles
        "ohlc_m": {
            "x_years": ohlc_m["x_years"].tolist(),
            "x_days": ohlc_m["x_days"].tolist(),
            "date_iso": ohlc_m["date_iso"].tolist(),
            "open": ohlc_m["open"].tolist(),
            "high": ohlc_m["high"].tolist(),
            "low": ohlc_m["low"].tolist(),
            "close": ohlc_m["close"].tolist()
        }
    }

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)
P0 = PRECOMP["USD"]
LAST_PRICE_ISO = str(pd.to_datetime(base["date"]).max().date())

# ───────────────────── Plot (rails stubs + series) ─────────────────────
MAX_RAIL_SLOTS = 12
IDX_MAIN  = MAX_RAIL_SLOTS
IDX_CLICK = MAX_RAIL_SLOTS + 1
IDX_CARRY = MAX_RAIL_SLOTS + 2
IDX_TT    = MAX_RAIL_SLOTS + 3

def add_stub(idx):
    return go.Scatter(x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
                      name=f"Rail {idx+1}",
                      line=dict(width=1.6, color="#999", dash="dot"),
                      visible=False, hoverinfo="skip")

traces = [add_stub(i) for i in range(MAX_RAIL_SLOTS)]
traces += [
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="_click", showlegend=False, hoverinfo="skip",
               line=dict(width=18, color="rgba(0,0,0,0.001)")),
    go.Scatter(x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
               name="_carry", showlegend=False, hoverinfo="x",
               line=dict(width=0.1, color="rgba(0,0,0,0.001)")),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="_tooltip", showlegend=False,
               hovertemplate="%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>",
               line=dict(width=0, color="rgba(0,0,0,0)")),
]

fig = go.Figure(traces)
# Genesis block x-coordinate for reference
x_genesis_ref = 1.0 / 365.25  # Years since Genesis at day 1

fig.update_layout(
    template="plotly_white",
    hovermode="x",
    showlegend=False,
    title=None,
    xaxis=dict(type="log", title=None, tickmode="array",
               tickvals=xtickvals, ticktext=xticktext,
               tickangle=0,
               range=[np.log10(x_start), np.log10(x_end_years)]),
    yaxis=dict(type="log", title=P0["label"],
               tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    margin=dict(l=60, r=20, t=20, b=50),
    autosize=True,
    # Add Genesis block reference line
    shapes=[dict(
        type="line", xref="x", yref="paper",
        x0=x_genesis_ref, x1=x_genesis_ref, y0=0, y1=1,
        line=dict(color="#F59E0B", width=1.5, dash="dot"),
        layer="below"
    )],
    annotations=[dict(
        x=np.log10(x_genesis_ref), y=1, xref="x", yref="paper",
        text="Genesis<br>Jan 3, 2009", showarrow=False,
        font=dict(size=10, color="#F59E0B"),
        xanchor="left", yanchor="top", xshift=5
    )]
)

# Ensure double-click resets axes (desktop + iOS Safari)
plot_html = fig.to_html(
    full_html=False,
    include_plotlyjs="cdn",
    config={
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage"],
        "doubleClick": "reset",      # double-click / double-tap resets view
    }
)

# ───────────────────────────── HTML ─────────────────────────────
HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"/>
<title>BTC Power Law</title>
<style>
:root{--sidebar:280px;--topbar:44px}
*{box-sizing:border-box}
html,body{height:100%;margin:0;font-family:system-ui,-apple-system,sans-serif;background:#0f172a}

/* Main layout */
.app{display:flex;flex-direction:column;height:100vh;height:100dvh;overflow:hidden}
.topbar{height:var(--topbar);background:#1e293b;display:flex;align-items:center;padding:0 12px;gap:12px;flex-shrink:0}
.topbar-title{font-weight:700;font-size:14px;color:#f8fafc;white-space:nowrap}
.topbar-controls{display:flex;align-items:center;gap:8px;margin-left:auto}
.main{display:flex;flex:1;overflow:hidden}
.chart-area{flex:1;min-width:0;padding:4px;display:flex;flex-direction:column;background:#0f172a}
.chart-area .js-plotly-plot,.chart-area .plotly-graph-div{width:100%!important;height:100%!important}
.sidebar{width:var(--sidebar);background:#1e293b;display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}

/* Controls */
select,input[type=date]{font-size:12px;padding:6px 8px;border-radius:6px;border:1px solid #475569;background:#334155;color:#f8fafc;cursor:pointer}
select{padding-right:24px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394a3b8' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 6px center;appearance:none}
select:focus,input:focus{outline:none;border-color:#3b82f6}
.btn{font-size:12px;padding:6px 10px;border-radius:6px;border:1px solid #475569;background:#334155;color:#f8fafc;cursor:pointer;transition:all 100ms}
.btn:hover{background:#475569}
.btn.active{background:#2563eb;border-color:#2563eb}
.btn-icon{width:28px;height:28px;padding:0;display:inline-flex;align-items:center;justify-content:center;font-weight:600;font-size:14px}

/* Sidebar sections */
.sidebar-section{padding:12px;border-bottom:1px solid #334155}
.section-header{display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none;padding:4px 0}
.section-title{font-weight:600;font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px}
.section-toggle{color:#64748b;font-size:10px}
.section-content{margin-top:8px}
.section-content.collapsed{display:none}

/* Signal box - prominent */
#predictionBox{padding:16px;border-radius:12px;text-align:center;transition:all 200ms}
#predSignal{font-weight:800;font-size:20px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px}
#predReason{font-size:11px;color:#94a3b8;line-height:1.4}
.signal-bar{height:8px;border-radius:4px;margin-top:12px;background:linear-gradient(to right,#7C3AED 0%,#2563EB 15%,#0EA5E9 25%,#16A34A 40%,#EAB308 60%,#F97316 75%,#DC2626 100%)}
.signal-marker{width:12px;height:12px;background:#f8fafc;border-radius:50%;border:2px solid #0f172a;box-shadow:0 2px 4px rgba(0,0,0,0.3);position:relative;top:-10px;transition:margin-left 300ms}

/* Readout */
.readout-compact{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:#e2e8f0;margin-top:12px}
.readout-item{display:flex;flex-direction:column;min-width:70px}
.readout-label{color:#64748b;font-size:10px;text-transform:uppercase}
.readout-value{font-weight:600;font-family:ui-monospace,monospace;font-size:14px}
.readout-date{font-weight:600;font-size:13px;color:#94a3b8;margin-bottom:8px}

/* Settings panel items */
.setting-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.setting-row label{font-size:11px;color:#94a3b8;min-width:60px}
.setting-row select,.setting-row input{flex:1}

/* Info modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.7);display:none;align-items:center;justify-content:center;z-index:100;padding:16px}
.modal-overlay.open{display:flex}
.modal-card{background:#1e293b;border-radius:12px;max-width:500px;width:100%;max-height:80vh;overflow-y:auto;padding:20px;color:#e2e8f0}
.modal-card h3{margin:0 0 12px;font-size:16px;color:#f8fafc}
.modal-card p{margin:8px 0;font-size:12px;line-height:1.6;color:#94a3b8}
.modal-card code{display:block;margin:8px 0;padding:8px;background:#334155;border-radius:6px;font-size:11px;color:#e2e8f0}

.hidden{display:none}

/* Desktop */
@media (min-width:769px){
  .mobile-only{display:none!important}
}

/* Tablet portrait */
@media (max-width:768px){
  :root{--sidebar:100%;--topbar:40px}
  .main{flex-direction:column}
  .chart-area{flex:1;min-height:0}
  .sidebar{width:100%;max-height:35vh;border-top:1px solid #334155;overflow-y:auto}
  .topbar-title{font-size:13px}
  .desktop-only{display:none!important}
}

/* Mobile landscape - CRITICAL */
@media (max-height:500px) and (orientation:landscape){
  :root{--topbar:36px}
  .topbar{padding:0 8px;gap:8px}
  .topbar-title{font-size:12px}
  .topbar-controls select,.topbar-controls .btn{font-size:11px;padding:4px 6px}
  .main{flex-direction:row}
  .chart-area{flex:1;padding:2px}
  .sidebar{width:200px;max-height:none;border-top:none;border-left:1px solid #334155}
  .sidebar-section{padding:8px}
  #predSignal{font-size:14px}
  #predReason{font-size:10px}
  .readout-compact{gap:8px}
  .readout-value{font-size:12px}
  .signal-bar{height:6px;margin-top:8px}
  .signal-marker{width:10px;height:10px;top:-8px}
}

/* Mobile portrait */
@media (max-width:480px){
  .topbar{flex-wrap:wrap;height:auto;padding:6px 8px;gap:6px}
  .topbar-title{flex:1}
  .topbar-controls{width:100%;justify-content:space-between}
  .topbar-controls select{flex:1;max-width:none}
  #predSignal{font-size:16px}
}

/* Very small screens */
@media (max-width:360px){
  .topbar-controls .btn-icon{width:24px;height:24px}
  select{font-size:11px;padding:4px 6px}
}
</style>
</head><body>
<div class="app">
  <div class="topbar">
    <div class="topbar-title">BTC Power Law</div>
    <div class="topbar-controls">
      <select id="denomSel" title="Denominator"></select>
      <input type="date" id="datePick" class="desktop-only" title="Select date"/>
      <button id="todayBtn" class="btn desktop-only">Today</button>
      <button id="settingsBtn" class="btn btn-icon mobile-only" title="Settings">&#9881;</button>
      <button id="infoBtn" class="btn btn-icon" title="Info">?</button>
    </div>
  </div>

  <div class="main">
    <div class="chart-area" id="leftCol">__PLOT_HTML__</div>

    <div class="sidebar">
      <!-- Signal -->
      <div class="sidebar-section">
        <div id="predictionBox">
          <div id="predSignal">\u2014</div>
          <div id="predReason"></div>
        </div>
        <div class="signal-bar"></div>
        <div class="signal-marker" id="oscMarker"></div>
        <div class="readout-compact">
          <div class="readout-item"><span class="readout-label">Date</span><span class="readout-value" id="readoutDate">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Price</span><span class="readout-value" id="mainVal">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Position</span><span class="readout-value" id="pPct">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Oscillator</span><span class="readout-value" id="oscVal">\u2014</span></div>
        </div>
      </div>

      <!-- Settings (collapsible) -->
      <div class="sidebar-section">
        <div class="section-header" data-target="settingsContent">
          <span class="section-title">Settings</span>
          <span class="section-toggle">\u25B6</span>
        </div>
        <div class="section-content collapsed" id="settingsContent">
          <div class="setting-row mobile-only">
            <label>Date</label>
            <input type="date" id="datePickMobile"/>
            <button id="todayBtnMobile" class="btn">Today</button>
          </div>
          <div class="setting-row">
            <label>Period</label>
            <select id="periodSel">
              <option value="all">All Data</option>
              <option value="10y">Last 10 Years</option>
              <option value="5y">Last 5 Years</option>
              <option value="2y">Last 2 Years</option>
              <option value="1y">Last 1 Year</option>
              <option value="cycle">Current Cycle</option>
            </select>
          </div>
          <div class="setting-row">
            <label>X-Axis</label>
            <select id="xAxisSel">
              <option value="years">Years</option>
              <option value="days">Days</option>
              <option value="date">Calendar</option>
            </select>
          </div>
          <div class="setting-row">
            <label>Candles</label>
            <select id="candleSel">
              <option value="D">Daily</option>
              <option value="W">Weekly</option>
              <option value="M">Monthly</option>
            </select>
          </div>
          <div class="setting-row">
            <button id="halvingsBtn" class="btn" style="flex:1">Halvings</button>
            <button id="liquidityBtn" class="btn" style="flex:1">Liquidity</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Info modal -->
<div id="infoModal" class="modal-overlay">
  <div class="modal-card">
    <h3>Bitcoin Power Law</h3>
    <p>Bitcoin's price follows a power law: <b>P = 10^a x t^b</b> where b is approximately 5.8. On a log-log chart, this creates a straight line with R squared of about 0.95.</p>
    <code>log(Price) = a + b x log(years_since_genesis)</code>
    <p><b>Position:</b> Where price sits within the power law corridor (0-100%). Low = undervalued, high = overvalued.</p>
    <p><b>Oscillator:</b> Normalized deviation from trend (-1 to +1). Historical tops occur at 0.8-0.9.</p>
    <p style="font-size:10px;color:#64748b;margin-top:16px">Sources: Santostasi, Burger, Perrenod</p>
    <div style="text-align:right;margin-top:12px"><button id="infoClose" class="btn">Close</button></div>
  </div>
</div>

<script>
const PRECOMP = __PRECOMP__;
const TIME_PERIODS = __TIME_PERIODS__;
const GENESIS = new Date('__GENESIS__T00:00:00Z');
const END_ISO = '__END_ISO__';
const LAST_PRICE_ISO = '__LAST_PRICE_ISO__';

const MAX_SLOTS = __MAX_RAIL_SLOTS__;
const IDX_MAIN  = __IDX_MAIN__;
const IDX_CLICK = __IDX_CLICK__;
const IDX_CARRY = __IDX_CARRY__;
const IDX_TT    = __IDX_TT__;
const EPS_LOG_SPACING = __EPS_LOG_SPACING__;
const PAST_HALVINGS = __PAST_HALVINGS__;
const FUTURE_HALVINGS = __FUTURE_HALVINGS__;

const LIQ_PEAK_ANCHOR_ISO = '__LIQ_PEAK_ANCHOR_ISO__';
const LIQ_PERIOD_MONTHS   = __LIQ_PERIOD_MONTHS__;
const LIQ_START_ISO       = '__LIQ_START_ISO__';

const W_P   = __W_P__;
const W_LIQ = __W_LIQ__;
const W_HALV= __W_HALV__;

// X-axis mode: 'years' or 'days'
let xAxisMode = 'days';
let candleMode = 'D';  // D=daily, W=weekly, M=monthly

const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }
function daysFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000) + 1; }
function shortDateFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function shortDateFromDays(days){ const ms=(days-1)*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function isoFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`; }
function isoFromDays(days){ const ms=(days-1)*86400000; const d=new Date(GENESIS.getTime()+ms); return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`; }
function timestampFromISO(iso){ return new Date(iso+'T00:00:00Z').getTime(); }
function isoFromTimestamp(ts){ const d=new Date(ts); return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`; }
function shortDateFromTimestamp(ts){ const d=new Date(ts); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function interp(xs, ys, x){ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function daysBetweenISO(a,b){ return Math.round((new Date(b+'T00:00:00Z') - new Date(a+'T00:00:00Z'))/86400000); }
function monthsBetweenISO(a,b){ return (new Date(b+'T00:00:00Z') - new Date(a+'T00:00:00Z'))/(86400000*30.4375); }

// Get x-values based on current mode
function getXMain(P){
  // For candle modes, use OHLC data
  if(candleMode==='W' && P.ohlc_w) return xAxisMode==='days' ? P.ohlc_w.x_days : (xAxisMode==='years' ? P.ohlc_w.x_years : P.ohlc_w.date_iso.map(timestampFromISO));
  if(candleMode==='M' && P.ohlc_m) return xAxisMode==='days' ? P.ohlc_m.x_days : (xAxisMode==='years' ? P.ohlc_m.x_years : P.ohlc_m.date_iso.map(timestampFromISO));
  // Daily
  if(xAxisMode==='date') return P.date_iso_main.map(timestampFromISO);
  return xAxisMode==='days' ? P.x_days : P.x_main;
}
function getYMain(P){
  if(candleMode==='W' && P.ohlc_w) return P.ohlc_w.close;
  if(candleMode==='M' && P.ohlc_m) return P.ohlc_m.close;
  return P.y_main;
}
function getOHLC(P){
  if(candleMode==='W' && P.ohlc_w) return P.ohlc_w;
  if(candleMode==='M' && P.ohlc_m) return P.ohlc_m;
  return null;
}
function getXGrid(P){ return xAxisMode==='days' ? P.x_grid_days : P.x_grid; }
function getSupport(P){ return xAxisMode==='days' ? P.support_days : P.support; }
function xToISO(x){
  if(xAxisMode==='date') return isoFromTimestamp(x);
  return xAxisMode==='days' ? isoFromDays(x) : isoFromYears(x);
}
function xToShortDate(x){
  if(xAxisMode==='date') return shortDateFromTimestamp(x);
  return xAxisMode==='days' ? shortDateFromDays(x) : shortDateFromYears(x);
}
function isoToX(iso){
  if(xAxisMode==='date') return timestampFromISO(iso);
  return xAxisMode==='days' ? daysFromISO(iso) : yearsFromISO(iso);
}

// ───────── Cycle Context & Projections ─────────
function getNextHalving(fromISO){
  const all = PAST_HALVINGS.concat(FUTURE_HALVINGS).sort();
  for(const h of all){ if(h > fromISO) return h; }
  return null;
}
function getPrevHalving(fromISO){
  const all = PAST_HALVINGS.concat(FUTURE_HALVINGS).sort().reverse();
  for(const h of all){ if(h <= fromISO) return h; }
  return PAST_HALVINGS[0];
}
function daysBetween(iso1, iso2){
  return Math.round((new Date(iso2+'T00:00:00Z') - new Date(iso1+'T00:00:00Z')) / 86400000);
}
function addDaysISO(iso, days){
  const d = new Date(iso+'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0,10);
}

function getCycleContext(iso){
  const prev = getPrevHalving(iso);
  const next = getNextHalving(iso);
  const daysSince = daysBetween(prev, iso);
  const daysUntil = next ? daysBetween(iso, next) : null;
  const cycleLength = next && prev ? daysBetween(prev, next) : 1460; // ~4 years
  const progress = daysSince / cycleLength;

  // Cycle phases based on typical halving cycle behavior
  let phase, phaseColor;
  if(progress < 0.25){
    phase = 'Accumulation'; phaseColor = '#16A34A';
  } else if(progress < 0.50){
    phase = 'Bull Run'; phaseColor = '#2563EB';
  } else if(progress < 0.65){
    phase = 'Euphoria/Top'; phaseColor = '#DC2626';
  } else if(progress < 0.85){
    phase = 'Bear Market'; phaseColor = '#F97316';
  } else {
    phase = 'Recovery'; phaseColor = '#0EA5E9';
  }

  return {
    prevHalving: prev, nextHalving: next,
    daysSince, daysUntil, cycleLength, progress,
    phase, phaseColor
  };
}

function getProjectedPrice(P, futureISO){
  const sup = getSupport(P);
  const xFuture = xAxisMode==='days' ? daysFromISO(futureISO) : yearsFromISO(futureISO);
  const logMid = sup.a0 + sup.b * Math.log10(xFuture);
  const mid = Math.pow(10, logMid);
  // Use quantile offsets for range
  const q25 = mid * Math.pow(10, sup.off_grid[Math.floor(sup.off_grid.length*0.25)]);
  const q75 = mid * Math.pow(10, sup.off_grid[Math.floor(sup.off_grid.length*0.75)]);
  return { mid, low: q25, high: q75 };
}

function getHistoricalContext(osc, p){
  // Historical patterns based on oscillator and p-value zones
  if(osc >= 0.8){
    return '\ud83d\udcca Historical: All 4 cycle tops occurred at oscillator >0.8. Average decline after: -75% to -85%. Consider defensive positioning.';
  } else if(osc >= 0.6){
    return '\ud83d\udcca Historical: Oscillator 0.6-0.8 preceded tops by 1-6 months in past cycles. Volatility increases significantly in this zone.';
  } else if(osc >= 0.3){
    return '\ud83d\udcca Historical: Mid-cycle consolidation zone. Price often trades sideways for extended periods before next major move.';
  } else if(osc >= -0.2){
    return '\ud83d\udcca Historical: Fair value zone. DCA strategies have performed well historically. Average time in this zone: 40% of cycle.';
  } else if(osc >= -0.5){
    return '\ud83d\udcca Historical: Below-trend zone. Past cycles showed 2-4x returns within 18 months from similar positions.';
  } else if(osc >= -0.8){
    return '\ud83d\udcca Historical: Deep value zone. Only ~5% of Bitcoin\u2019s history spent here. 3-10x returns typical over following 2 years.';
  } else {
    return '\ud83d\udcca Historical: Extreme undervaluation (bottom 2%). Occurred during 2011, 2015, 2019, 2022 capitulation events. 10x+ returns followed.';
  }
}

// Prediction signal based on oscillator AND p-value (position in corridor)
// More refined signals using both metrics
function getPredictionSignal(osc, p){
  // SELL ZONES - oscillator-driven but confirmed by p-value
  if (osc >= 0.85 || (osc >= 0.7 && p >= 90)) {
    return {signal: '\u26a0\ufe0f SELL ZONE', color: '#DC2626', bg: '#FEE2E2',
      reason: 'Oscillator \u2265'+(osc>=0.85?'0.85':'0.70')+' + Position '+(p>=90?'\u226590%':p.toFixed(0)+'%')+': All historical ATHs. Take profits.'};
  }
  if (osc >= 0.6 || p >= 85) {
    return {signal: '\u2b06\ufe0f PRE-SELL', color: '#F97316', bg: '#FFF7ED',
      reason: 'Oscillator '+osc.toFixed(2)+', Position '+p.toFixed(0)+'%: Approaching top zone. Prepare exit strategy.'};
  }
  // AVOID ZONE - poor entry points
  if (p >= 75 && osc >= 0.3) {
    return {signal: '\u26d4 AVOID', color: '#EF4444', bg: '#FEF2F2',
      reason: 'Position '+p.toFixed(0)+'% (upper corridor): Historically poor entry. Wait for better prices.'};
  }
  if (osc >= 0.4) {
    return {signal: '\u2197\ufe0f ELEVATED', color: '#EAB308', bg: '#FEFCE8',
      reason: 'Oscillator '+osc.toFixed(2)+': Above fair value. Hold positions, be cautious adding.'};
  }
  // FAIR VALUE - good for DCA
  if (osc >= -0.2) {
    return {signal: '\u2705 FAIR VALUE', color: '#16A34A', bg: '#F0FDF4',
      reason: 'Oscillator '+osc.toFixed(2)+', Position '+p.toFixed(0)+'%: Near trend. Good DCA zone.'};
  }
  // ACCUMULATION ZONES - p-value driven
  if (p <= 25 || osc <= -0.5) {
    if (osc <= -0.8 || p <= 10) {
      return {signal: '\ud83d\udea8 EXTREME BUY', color: '#7C3AED', bg: '#F5F3FF',
        reason: 'Position '+p.toFixed(0)+'%, Oscillator '+osc.toFixed(2)+': Extreme undervaluation. Rare opportunity!'};
    }
    if (osc <= -0.5 || p <= 15) {
      return {signal: '\ud83d\udcb0 STRONG BUY', color: '#2563EB', bg: '#EFF6FF',
        reason: 'Position '+p.toFixed(0)+'% (lower corridor): Historically excellent entry. Aggressive accumulation.'};
    }
    return {signal: '\u2b07\ufe0f ACCUMULATE', color: '#0EA5E9', bg: '#F0F9FF',
      reason: 'Position '+p.toFixed(0)+'%, Oscillator '+osc.toFixed(2)+': Below trend. Good buying opportunity.'};
  }
  return {signal: '\u2b07\ufe0f ACCUMULATE', color: '#0EA5E9', bg: '#F0F9FF',
    reason: 'Oscillator '+osc.toFixed(2)+': Below fair value. Consider adding.'};
}

function fmtVal(P, v){
  if(!isFinite(v)) return '—';
  const dec = Math.max(0, Math.min(10, P.decimals||2));
  if (P.unit === '$') return '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:dec,maximumFractionDigits:dec});
  const maxd=Math.max(dec,6);
  return Number(v).toLocaleString(undefined,{minimumFractionDigits:Math.min(6,maxd),maximumFractionDigits:maxd});
}

// Colors & classes
function colorForPercent(p){ const t=clamp(p/100,0,1);
  function hx(v){return Math.max(0,Math.min(255,Math.round(v))); }
  function toHex(r,g,b){return '#'+[r,g,b].map(v=>hx(v).toString(16).padStart(2,'0')).join('');}
  if(t<=0.5){const u=t/0.5;return toHex(0xD3+(0xFB-0xD3)*u,0x2F+(0xC0-0x2F)*u,0x2F+(0x2D-0x2F)*u);}
  const u=(t-0.5)/0.5; return toHex(0xFB+(0x2E-0xFB)*u,0xC0+(0x7D-0xC0)*u,0x2D+(0x32-0x2D)*u);
}
function classFromScore(s){
  if (s >= 1.25) return 'SELL THE HOUSE';
  if (s >= 0.75) return 'Strong Buy';
  if (s >= 0.25) return 'Buy';
  if (s >  -0.25) return 'DCA';
  if (s >  -0.75) return 'Hold On';
  if (s >  -1.50) return 'Frothy';
  return 'Top Inbound';
}

// DOM - Simplified
const leftCol=document.getElementById('leftCol');
const sidebar=document.querySelector('.sidebar');
const plotDiv=(function(){ return document.querySelector('.chart-area .js-plotly-plot') || document.querySelector('.chart-area .plotly-graph-div'); })();
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const datePickMobile=document.getElementById('datePickMobile');
const btnToday=document.getElementById('todayBtn');
const btnTodayMobile=document.getElementById('todayBtnMobile');
const btnHalvings=document.getElementById('halvingsBtn');
const btnLiquidity=document.getElementById('liquidityBtn');
const infoBtn=document.getElementById('infoBtn');
const infoModal=document.getElementById('infoModal');
const infoClose=document.getElementById('infoClose');
const settingsBtn=document.getElementById('settingsBtn');

const elDate=document.getElementById('readoutDate');
const elMain=document.getElementById('mainVal');
const elP=document.getElementById('pPct');
const oscValEl=document.getElementById('oscVal');
const periodSel=document.getElementById('periodSel');
const xAxisSel=document.getElementById('xAxisSel');
const candleSel=document.getElementById('candleSel');
const predSignal=document.getElementById('predSignal');
const predReason=document.getElementById('predReason');
const predBox=document.getElementById('predictionBox');
const oscMarker=document.getElementById('oscMarker');

// Collapsible sections
document.querySelectorAll('.section-header').forEach(header=>{
  header.addEventListener('click',()=>{
    const target=document.getElementById(header.dataset.target);
    if(!target) return;
    const toggle=header.querySelector('.section-toggle');
    if(target.classList.contains('collapsed')){
      target.classList.remove('collapsed');
      if(toggle) toggle.textContent='\u25BC';
    } else {
      target.classList.add('collapsed');
      if(toggle) toggle.textContent='\u25B6';
    }
  });
});

// Settings button (mobile) - opens settings section
if(settingsBtn){
  settingsBtn.addEventListener('click',()=>{
    const settingsContent=document.getElementById('settingsContent');
    if(settingsContent){
      settingsContent.classList.toggle('collapsed');
      const toggle=document.querySelector('[data-target="settingsContent"] .section-toggle');
      if(toggle) toggle.textContent=settingsContent.classList.contains('collapsed')?'\u25B6':'\u25BC';
    }
  });
}

// Sync date pickers (desktop and mobile)
function syncDatePickers(val){
  if(datePick) datePick.value=val;
  if(datePickMobile) datePickMobile.value=val;
}

// Denominators
(function initDenoms(){
  const keys=Object.keys(PRECOMP); const order=['USD'].concat(keys.filter(k=>k!=='USD'));
  denomSel.innerHTML=''; order.forEach(k=>{ const o=document.createElement('option'); o.value=k; o.textContent=k; denomSel.appendChild(o); });
  denomSel.value='USD';
})();

// Rails state (simplified - fixed set)
let rails=[97.5,90,75,50,25,2.5];
function sortRails(){ rails=rails.filter(p=>isFinite(p)).map(Number).map(p=>clamp(p,0.1,99.9)).filter((p,i,a)=>a.indexOf(p)===i).sort((a,b)=>b-a); }
function idFor(p){ return 'v'+String(p).replace('.','_'); }

// Sizing - chart fills available space
function triggerResize(){
  if(window.Plotly&&plotDiv){
    Plotly.Plots.resize(plotDiv);
    requestAnimationFrame(()=>Plotly.Plots.resize(plotDiv));
  }
}

// Auto-resize on window/container changes
if(window.ResizeObserver){
  new ResizeObserver(()=>triggerResize()).observe(leftCol);
}
window.addEventListener('resize',()=>triggerResize());
// Trigger resize on orientation change (important for mobile)
window.addEventListener('orientationchange',()=>{ setTimeout(triggerResize,100); });

// ───────── Rails math ─────────
function logMidline(P){
  const d=getSupport(P);
  const xg=getXGrid(P);
  return xg.map(x=> (d.a0 + d.b*Math.log10(x)) );
}
function offsetForPercent(P, percent){
  const d=getSupport(P); const q=d.q_grid, off=d.off_grid;
  const p01=clamp(percent/100, q[0], q[q.length-1]);
  let lo=0, hi=q.length-1;
  while(hi-lo>1){ const m=(hi+lo)>>1; if(q[m]<=p01) lo=m; else hi=m; }
  const t=(p01-q[lo])/(q[hi]-q[lo]);
  return off[lo] + t*(off[hi]-off[lo]);
}
function seriesForPercent(P, percent){
  const logM=logMidline(P), off=offsetForPercent(P,percent);
  const eps=(percent>=97.5||percent<=2.5)?EPS_LOG_SPACING:0.0;
  return logM.map(v=>Math.pow(10, v+off+(percent>=50?eps:-eps)));
}
function percentFromOffset(P, off){
  const d=getSupport(P); const q=d.q_grid, offg=d.off_grid;
  if (off<=offg[0]) return 100*q[0];
  if (off>=offg[offg.length-1]) return 100*q[offg.length-1];
  let lo=0, hi=offg.length-1;
  while(hi-lo>1){ const m=(hi+lo)>>1; if(offg[m]<=off) lo=m; else hi=m; }
  const t=(off-offg[lo])/(offg[hi]-offg[lo]);
  return 100*(q[lo] + t*(q[hi]-q[lo]));
}

// Liquidity & halving utilities
function liqCosAtISO(iso){
  const m = monthsBetweenISO(LIQ_PEAK_ANCHOR_ISO, iso);
  const ang = 2*Math.PI*m/LIQ_PERIOD_MONTHS;
  return Math.cos(ang); // >0 rising; <0 falling
}
function halvingScoreAtISO(iso){
  let last = null, next = null;
  for (const h of PAST_HALVINGS.concat(FUTURE_HALVINGS)){
    if (h <= iso) last = h;
    if (h > iso){ next = h; break; }
  }
  let s = 0;
  if (next){
    const d = daysBetweenISO(iso, next);
    if (d >= 0 && d <= 365) s += 1.0; // pre-halving year
  }
  if (last){
    const d = daysBetweenISO(last, iso);
    if (d > 0 && d <= 540) s += 0.5;   // early post-halving
    if (d > 540)           s -= 0.5;   // late post-halving
  }
  return s;
}

// Composite
function zFromP(p){ return (50 - p)/25; } // +1 at p=25, 0 at p=50, −1 at p=75
function compositeFrom(p, iso){
  const liq = liqCosAtISO(iso);       // −1..+1
  const halv= halvingScoreAtISO(iso); // −0.5..+1.5
  const score = W_P*zFromP(p) + W_LIQ*liq + W_HALV*halv;
  return score;
}
// Power Law Oscillator: normalize log-deviation from trend to [-1, +1]
function oscillatorFromDeviation(P, logDev){
  const sup = getSupport(P);
  const rmin = sup.resid_min;
  const rmax = sup.resid_max;
  if (rmax === rmin) return 0;
  return (2 * (logDev - rmin) / (rmax - rmin)) - 1;
}
function oscillatorLabel(osc){
  if (osc >= 0.8)  return 'Extreme Overvalued';
  if (osc >= 0.5)  return 'Overvalued';
  if (osc >= 0.2)  return 'Slightly Above Trend';
  if (osc > -0.2)  return 'Fair Value';
  if (osc > -0.5)  return 'Slightly Below Trend';
  if (osc > -0.8)  return 'Undervalued';
  return 'Extreme Undervalued';
}
function oscillatorColor(osc){
  // Red at +1, yellow at 0, green at -1
  const t = clamp((osc + 1) / 2, 0, 1); // 0=green, 0.5=yellow, 1=red
  function hx(v){return Math.max(0,Math.min(255,Math.round(v)));}
  function toHex(r,g,b){return '#'+[r,g,b].map(v=>hx(v).toString(16).padStart(2,'0')).join('');}
  if (t <= 0.5){ const u=t/0.5; return toHex(0x2E+(0xFB-0x2E)*u, 0x7D+(0xC0-0x7D)*u, 0x32+(0x2D-0x32)*u); }
  const u=(t-0.5)/0.5; return toHex(0xFB+(0xD3-0xFB)*u, 0xC0+(0x2F-0xC0)*u, 0x2D+(0x2F-0x2D)*u);
}

function titleForScore(s){ return 'BTC Purchase Indicator \u2014 ' + classFromScore(s); }
function setTitle(s){ Plotly.relayout(plotDiv, {'title.text': titleForScore(s)}); }

// Render rails
function renderRails(P){
  const xg = getXGrid(P);
  const n=Math.min(rails.length, MAX_SLOTS);
  for(let i=0;i<MAX_SLOTS;i++){
    const visible=(i<n); let restyle={visible};
    if(visible){
      const p=rails[i], color=colorForPercent(p);
      const nm=(p===2.5?'Floor':(p===97.5?'Ceiling':(p+'%')));
      const is50 = (Math.abs(p - 50) < 0.01);
      restyle=Object.assign(restyle,{x:[xg], y:[seriesForPercent(P,p)], name:nm,
        line:{color: is50 ? '#4B5563' : color, width: is50 ? 2.2 : 1.6, dash: is50 ? 'solid' : 'dot'}});
    }
    Plotly.restyle(plotDiv, restyle, [i]);
  }
  Plotly.restyle(plotDiv, {x:[xg], y:[seriesForPercent(P,50)]}, [IDX_CARRY]);
}

// Panel update (simplified)
function updatePanel(P, xVal){
  const iso = xToISO(xVal);
  elDate.textContent = xToShortDate(xVal);

  const xg = getXGrid(P);
  const v50series = seriesForPercent(P,50);
  const v50 = interp(xg, v50series, xVal);

  const xMain = getXMain(P);
  const lastX = xMain[xMain.length-1];
  let usedP=null, mainTxt='', logDev=0;

  if (xVal > lastX){
    mainTxt = fmtVal(P, v50) + ' (50%)';
    elP.textContent = '\u2014';
    usedP = 50;
    logDev = 0;
  } else {
    const yMain = getYMain(P);
    let idx=0,best=1e99; for(let i=0;i<xMain.length;i++){ const d=Math.abs(xMain[i]-xVal); if(d<best){best=d; idx=i;} }
    const y=yMain[idx]; mainTxt = fmtVal(P,y);
    const sup=getSupport(P), z=Math.log10(xMain[idx]);
    logDev = Math.log10(y) - (sup.a0 + sup.b*z);
    usedP=clamp(percentFromOffset(P, logDev), 0, 100);
    elP.textContent=usedP.toFixed(1)+'%';
  }
  elMain.textContent = mainTxt;

  // Power Law Oscillator
  const osc = oscillatorFromDeviation(P, logDev);
  const oscCol = oscillatorColor(osc);
  if(oscValEl){ oscValEl.textContent = osc.toFixed(3); oscValEl.style.color = oscCol; }

  // Prediction Signal
  const pred = getPredictionSignal(osc, usedP);
  predSignal.textContent = pred.signal;
  predSignal.style.color = pred.color;
  predReason.textContent = pred.reason;
  predBox.style.borderColor = pred.color;
  predBox.style.background = pred.bg;

  // Oscillator position marker (map -1..1 to 0..100%)
  const oscPct = clamp((osc + 1) / 2 * 100, 0, 100);
  if(oscMarker) oscMarker.style.marginLeft = `calc(${oscPct}% - 6px)`;
}

// Hover drives panel
plotDiv.on('plotly_hover', ev=>{
  if(!(ev.points && ev.points.length)) return;
  updatePanel(PRECOMP[denomSel.value], ev.points[0].x);
});

// Date picker initialization
function getTodayISO(){
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

// Initialize both date pickers
const todayISO = getTodayISO();
if(datePick){ datePick.value = todayISO; datePick.max = todayISO; }
if(datePickMobile){ datePickMobile.value = todayISO; datePickMobile.max = todayISO; }

// Date change handler
function handleDateChange(val){
  if(!val) return;
  syncDatePickers(val);
  const P=PRECOMP[denomSel.value];
  const xVal = isoToX(val);
  updatePanel(P, xVal);
}

// Today button handler
function handleToday(){
  const P=PRECOMP[denomSel.value];
  const today = getTodayISO();
  syncDatePickers(today);
  const todayX = isoToX(today);
  const xMain = getXMain(P);
  const lastDataX = xMain[xMain.length-1];
  updatePanel(P, Math.min(todayX, lastDataX));
}

// Bind date picker events
if(datePick) datePick.addEventListener('change',()=>handleDateChange(datePick.value));
if(datePickMobile) datePickMobile.addEventListener('change',()=>handleDateChange(datePickMobile.value));
if(btnToday) btnToday.addEventListener('click',handleToday);
if(btnTodayMobile) btnTodayMobile.addEventListener('click',handleToday);

// ───────── X-Axis mode switching ─────────
function fullRedraw(){
  const P = PRECOMP[denomSel.value];
  delete P._classSeries; delete P._pSeries; delete P._scoreSeries;
  const xMain = getXMain(P);
  const yMain = getYMain(P);
  const xg = getXGrid(P);

  // Axis labels and type based on mode
  let xLabel, xType;
  if(xAxisMode==='date'){
    xLabel = 'Date';
    xType = 'date';
  } else if(xAxisMode==='days'){
    xLabel = 'Log\u2081\u2080 Days from Genesis';
    xType = 'log';
  } else {
    xLabel = 'Log\u2081\u2080 Years from Genesis';
    xType = 'log';
  }

  Plotly.restyle(plotDiv,{x:[xMain], y:[yMain], name:[P.label]}, [IDX_MAIN]);
  Plotly.restyle(plotDiv,{x:[xMain], y:[yMain]},                 [IDX_CLICK]);
  Plotly.restyle(plotDiv,{x:[xg], y:[seriesForPercent(P,50)]},   [IDX_CARRY]);
  Plotly.relayout(plotDiv,{
    'xaxis.title': xLabel,
    'xaxis.type': xType,
    'yaxis.title': P.label,
    'xaxis.autorange': true,
    'yaxis.autorange': true
  });
  renderRails(P);
  updatePanel(P, xMain[xMain.length-1]);
}

xAxisSel.addEventListener('change',()=>{
  xAxisMode = xAxisSel.value;
  fullRedraw();
});

candleSel.addEventListener('change',()=>{
  candleMode = candleSel.value;
  fullRedraw();
});

// ───────── Period (zoom preset) switching ─────────
periodSel.addEventListener('change',()=>{
  const key = periodSel.value;
  const per = TIME_PERIODS[key];
  if (!per) return;
  const P = PRECOMP[denomSel.value];
  const xMain = getXMain(P);
  let xlo, xhi;
  if (per.days_back) {
    xhi = xMain[xMain.length-1];
    const isoStart = isoFromDays(daysFromISO(LAST_PRICE_ISO) - per.days_back);
    xlo = isoToX(isoStart);
  } else {
    xlo = isoToX(per.start);
    xhi = per.end ? isoToX(per.end) : (key==='all' ? undefined : xMain[xMain.length-1] * 1.05);
  }
  if (key==='all'){
    Plotly.relayout(plotDiv,{'xaxis.autorange':true,'yaxis.autorange':true});
  } else if (xAxisMode==='date') {
    // For calendar date mode, use timestamps directly
    Plotly.relayout(plotDiv,{'xaxis.range':[xlo, xhi], 'yaxis.autorange':true});
  } else {
    Plotly.relayout(plotDiv,{'xaxis.range':[Math.log10(Math.max(1e-6,xlo)), Math.log10(xhi)], 'yaxis.autorange':true});
  }
});

// Double-click reset
plotDiv.on('plotly_doubleclick', ()=>{
  Plotly.relayout(plotDiv, {'xaxis.autorange': true, 'yaxis.autorange': true});
  return false;
});

// Denominator change
denomSel.addEventListener('change',()=>{
  fullRedraw();
});

// Sync all (simplified)
function syncAll(){
  sortRails();
  const P=PRECOMP[denomSel.value];
  const xMain = getXMain(P);
  renderRails(P);
  updatePanel(P, xMain[xMain.length-1]);
}

// Init (simplified)
(function init(){
  const P=PRECOMP['USD'];
  const xMain = getXMain(P);
  const yMain = getYMain(P);
  const xg = getXGrid(P);
  renderRails(P);
  Plotly.restyle(plotDiv,{x:[xg], y:[seriesForPercent(P,50)]},[IDX_CARRY]);
  const tt='%{customdata}<br>$%{y:.2f}<extra></extra>';
  Plotly.restyle(plotDiv,{x:[xMain], y:[yMain], customdata:[P.date_iso_main], hovertemplate:[tt], line:[{width:0}]},[IDX_TT]);
  updatePanel(P, xMain[xMain.length-1]);
})();

// Halvings toggle
let halvingsOn=false;
function makeHalvingShapes(){
  const shapes=[];
  function lineAt(iso, dashed){
    const xVal = isoToX(iso);
    return {type:'line', xref:'x', yref:'paper', x0:xVal, x1:xVal, y0:0, y1:1,
            line:{color:'#9CA3AF', width:1.2, dash:(dashed?'dash':'solid')}, layer:'below', meta:'halving'};
  }
  PAST_HALVINGS.forEach(iso=>{ shapes.push(lineAt(iso, false)); });
  FUTURE_HALVINGS.forEach(iso=>{ shapes.push(lineAt(iso, true)); });
  return shapes;
}
btnHalvings.onclick=()=>{ halvingsOn=!halvingsOn;
  const curr=plotDiv.layout.shapes||[];
  if(halvingsOn){
    Plotly.relayout(plotDiv, {shapes:[].concat(curr, makeHalvingShapes())});
    btnHalvings.classList.add('active');
    btnHalvings.setAttribute('aria-pressed', 'true');
  }else{
    const remain=(plotDiv.layout.shapes||[]).filter(s=>s.meta!=='halving');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnHalvings.classList.remove('active');
    btnHalvings.setAttribute('aria-pressed', 'false');
  }
};

// Liquidity toggle (peaks red, troughs green; future dashed)
let liquidityOn = false;
function addMonthsISO(iso, months){
  const d  = new Date(iso + 'T00:00:00Z');
  const y0 = d.getUTCFullYear(); const m0 = d.getUTCMonth(); const day= d.getUTCDate();
  const nd = new Date(Date.UTC(y0, m0 + months, 1));
  const lastDay = new Date(Date.UTC(nd.getUTCFullYear(), nd.getUTCMonth()+1, 0)).getUTCDate();
  nd.setUTCDate(Math.min(day, lastDay));
  return nd.toISOString().slice(0,10);
}
function compareISO(a,b){ return a<b?-1:(a>b?1:0); }
function makeLiquidityShapes(){
  const shapes = [];
  const startIso = LIQ_START_ISO;
  const endIso   = END_ISO;
  const lastIso  = LAST_PRICE_ISO;

  let peaks = [];
  let iso = LIQ_PEAK_ANCHOR_ISO;
  while (compareISO(iso, startIso) > 0){ peaks.push(iso); iso = addMonthsISO(iso, -LIQ_PERIOD_MONTHS); }
  peaks = peaks.reverse();
  iso = addMonthsISO(peaks[peaks.length-1], LIQ_PERIOD_MONTHS);
  while (compareISO(iso, endIso) <= 0){ peaks.push(iso); iso = addMonthsISO(iso, LIQ_PERIOD_MONTHS); }

  const half = Math.round(LIQ_PERIOD_MONTHS/2);
  const troughs = peaks.map(p => addMonthsISO(p, half));

  function vline(iso, color){
    const dashed = compareISO(iso, lastIso) > 0;
    const xVal = isoToX(iso);
    return { type:'line', xref:'x', yref:'paper',
      x0: xVal, x1: xVal, y0:0, y1:1,
      line:{color:color, width:1.4, dash: dashed ? 'dash' : 'solid'},
      layer:'below', meta:'liquidity' };
  }

  troughs.forEach(tIso => { if (compareISO(tIso, startIso)>=0 && compareISO(tIso, endIso)<=0) shapes.push(vline(tIso, '#22C55E')); });
  peaks.forEach(pIso =>   { if (compareISO(pIso, startIso)>=0 && compareISO(pIso, endIso)<=0) shapes.push(vline(pIso, '#EF4444')); });
  return shapes;
}
btnLiquidity.onclick = () => {
  liquidityOn = !liquidityOn;
  const curr = plotDiv.layout.shapes || [];
  if (liquidityOn){
    Plotly.relayout(plotDiv, {shapes:[].concat(curr, makeLiquidityShapes())});
    btnLiquidity.classList.add('active');
    btnLiquidity.setAttribute('aria-pressed', 'true');
  } else {
    const remain = (plotDiv.layout.shapes||[]).filter(s => s.meta!=='liquidity');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnLiquidity.classList.remove('active');
    btnLiquidity.setAttribute('aria-pressed', 'false');
  }
};

// Info modal
infoBtn.onclick = ()=>{ infoModal.classList.add('open'); };
infoClose.onclick = ()=>{ infoModal.classList.remove('open'); };
infoModal.addEventListener('click', (e)=>{ if(e.target===infoModal) infoModal.classList.remove('open'); });
</script>
</body></html>
"""

# ───────────────────────────── Fill placeholders ─────────────────────────────
HTML = (HTML
    .replace("__PLOT_HTML__", plot_html)
    .replace("__PRECOMP__", json.dumps(PRECOMP))
    .replace("__TIME_PERIODS__", json.dumps(TIME_PERIODS))
    .replace("__GENESIS__", GENESIS_DATE.strftime("%Y-%m-%d"))
    .replace("__END_ISO__", END_PROJ.strftime("%Y-%m-%d"))
    .replace("__LAST_PRICE_ISO__", LAST_PRICE_ISO)
    .replace("__MAX_RAIL_SLOTS__", str(MAX_RAIL_SLOTS))
    .replace("__IDX_MAIN__",  str(IDX_MAIN))
    .replace("__IDX_CLICK__", str(IDX_CLICK))
    .replace("__IDX_CARRY__", str(IDX_CARRY))
    .replace("__IDX_TT__",    str(IDX_TT))
    .replace("__EPS_LOG_SPACING__", str(EPS_LOG_SPACING))
    .replace("__PAST_HALVINGS__", json.dumps(PAST_HALVINGS))
    .replace("__FUTURE_HALVINGS__", json.dumps(FUTURE_HALVINGS))
    .replace("__LIQ_PEAK_ANCHOR_ISO__", LIQ_PEAK_ANCHOR_ISO)
    .replace("__LIQ_PERIOD_MONTHS__", str(LIQ_PERIOD_MONTHS))
    .replace("__LIQ_START_ISO__", LIQ_START_ISO)
    .replace("__W_P__", str(W_P))
    .replace("__W_LIQ__", str(W_LIQ))
    .replace("__W_HALV__", str(W_HALV))
)

# ───────────────────────────── Write site ─────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
# Remove any surrogate characters that can't be encoded to UTF-8
HTML_clean = re.sub(r'[\ud800-\udfff]', '', HTML)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML_clean)
print("Wrote", OUTPUT_HTML)
