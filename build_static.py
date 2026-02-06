#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — stationary rails with composite signal (no cross-denom voting),
cursor-driven panel (future uses 50%), even-year x labels after 2020, halvings & liquidity toggles,
info modal, and user-defined horizontal Level lines (per denominator).

Writes: docs/index.html
"""

import os, io, glob, json
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
    y_col = "price" if denom_key is None else f"price_{denom_key.lower()}"
    if y_col not in df.columns:
        y_col = "price"

    # Weekly OHLC
    ohlc_w = resample_ohlc(df.copy(), y_col, 'W')
    ohlc_w['x_years'] = years_since_genesis(ohlc_w['date'])
    ohlc_w['x_days'] = days_since_genesis(ohlc_w['date'])
    ohlc_w['date_iso'] = ohlc_w['date'].dt.strftime('%Y-%m-%d')

    # Monthly OHLC
    ohlc_m = resample_ohlc(df.copy(), y_col, 'ME')
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
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Power Law Indicator</title>
<style>
:root{--sidebar:300px;--topbar:48px}
*{box-sizing:border-box}
html,body{height:100%;margin:0;font-family:Inter,system-ui,-apple-system,sans-serif;background:#f8fafc}

/* Main layout */
.app{display:flex;flex-direction:column;height:100vh;overflow:hidden}
.topbar{height:var(--topbar);background:#fff;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;padding:0 12px;gap:16px;flex-shrink:0}
.topbar-title{font-weight:700;font-size:15px;color:#1e293b;white-space:nowrap}
.topbar-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.main{display:flex;flex:1;overflow:hidden}
.chart-area{flex:1;min-width:0;padding:8px;display:flex;flex-direction:column}
.chart-area .js-plotly-plot,.chart-area .plotly-graph-div{width:100%!important;height:100%!important}
.sidebar{width:var(--sidebar);border-left:1px solid #e2e8f0;background:#fff;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}

/* Controls */
select{font-size:13px;padding:6px 24px 6px 8px;border-radius:6px;border:1px solid #cbd5e1;background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23475569' d='M2 4l4 4 4-4'/%3E%3C/svg%3E") no-repeat right 8px center;appearance:none;cursor:pointer}
select:hover{border-color:#94a3b8}
select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,0.15)}
input[type=date],input[type=number],input[type=text]{font-size:13px;padding:6px 8px;border-radius:6px;border:1px solid #cbd5e1;background:#fff}
input:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,0.15)}
.btn{font-size:13px;padding:6px 10px;border-radius:6px;border:1px solid #cbd5e1;background:#fff;cursor:pointer;transition:all 100ms;white-space:nowrap}
.btn:hover{background:#f1f5f9;border-color:#94a3b8}
.btn:active{background:#e2e8f0}
.btn.active{background:#2563eb;color:#fff;border-color:#2563eb}
.btn.active:hover{background:#1d4ed8}
.btn-icon{width:32px;height:32px;padding:0;display:inline-flex;align-items:center;justify-content:center;font-weight:600}
.btn-sm{font-size:12px;padding:4px 8px}

/* Sidebar sections */
.sidebar-section{padding:12px;border-bottom:1px solid #e2e8f0}
.sidebar-section:last-child{border-bottom:none}
.section-header{display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none}
.section-title{font-weight:600;font-size:13px;color:#475569}
.section-toggle{color:#94a3b8;font-size:11px}
.section-content{margin-top:8px}
.section-content.collapsed{display:none}

/* Prediction box - prominent */
#predictionBox{padding:12px;border-radius:10px;border:2px solid #e2e8f0;text-align:center;transition:all 200ms}
#predSignal{font-weight:700;font-size:18px;margin-bottom:4px}
#predReason{font-size:11px;color:#64748b;line-height:1.4}

/* Readout */
.readout-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:12px}
.readout-item{display:flex;flex-direction:column}
.readout-label{color:#64748b;font-size:11px}
.readout-value{font-weight:600;font-family:ui-monospace,monospace;font-size:13px}
.readout-date{font-weight:600;font-size:14px;color:#1e293b;margin-bottom:8px}

/* Rails compact */
.rails-list{font-size:11px;color:#64748b;margin-top:4px}
.rail-row{display:flex;align-items:center;gap:6px;margin:3px 0}
.rail-row input[type=number]{width:70px;font-size:12px;padding:4px 6px}
.rail-row button{font-size:11px;padding:2px 6px}

/* Advanced section */
.adv-row{display:flex;align-items:center;gap:6px;margin:6px 0;flex-wrap:wrap}
.adv-row label{font-size:12px;color:#64748b;min-width:50px}
.adv-row input[type=range]{flex:1;min-width:80px}
.adv-row input[type=number]{width:60px}

/* Indicator dropdown */
.indicator-wrap{position:relative}
.indicator-menu{position:absolute;top:100%;right:0;z-index:50;min-width:180px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.1);padding:6px;display:none;margin-top:4px}
.indicator-menu.open{display:block}
.indicator-item{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;cursor:pointer;font-size:12px}
.indicator-item:hover{background:#f1f5f9}
.indicator-actions{display:flex;gap:4px;padding-top:6px;border-top:1px solid #e2e8f0;margin-top:4px}
.indicator-actions button{flex:1}

/* Info modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.4);display:none;align-items:center;justify-content:center;z-index:100;padding:20px}
.modal-overlay.open{display:flex}
.modal-card{background:#fff;border-radius:12px;max-width:640px;width:100%;max-height:80vh;overflow-y:auto;padding:20px;box-shadow:0 20px 40px rgba(0,0,0,0.15)}
.modal-card h3{margin:0 0 12px;font-size:18px}
.modal-card p{margin:8px 0;font-size:13px;line-height:1.6;color:#334155}
.modal-card code{display:block;margin:8px 0;padding:10px;background:#f1f5f9;border-radius:6px;font-size:12px}

/* Responsive */
@media (max-width:768px){
  .main{flex-direction:column}
  .sidebar{width:100%;max-height:40vh;border-left:none;border-top:1px solid #e2e8f0}
  .topbar{flex-wrap:wrap;height:auto;padding:8px 12px;gap:8px}
  .topbar-title{width:100%}
}
.hidden{display:none}
.smallnote{font-size:11px;color:#64748b}
</style>
</head><body>
<div class="app">
  <!-- Top control bar -->
  <div class="topbar">
    <div class="topbar-title">BTC Power Law</div>
    <div class="topbar-controls">
      <select id="denomSel" title="Denominator"></select>
      <select id="periodSel" title="Time Period">
        <option value="all">All Data</option>
        <option value="10y">Last 10 Years</option>
        <option value="5y">Last 5 Years</option>
        <option value="2y">Last 2 Years</option>
        <option value="1y">Last 1 Year</option>
        <option value="cycle">Current Cycle</option>
      </select>
      <select id="xAxisSel" title="X-Axis Mode">
        <option value="days">Days Since Genesis</option>
        <option value="years">Years Since Genesis</option>
        <option value="date">Calendar Date</option>
      </select>
      <select id="candleSel" title="Candle Interval">
        <option value="D">Daily</option>
        <option value="W">Weekly</option>
        <option value="M">Monthly</option>
      </select>
      <button id="halvingsBtn" class="btn" title="Toggle halvings">Halvings</button>
      <button id="liquidityBtn" class="btn" title="Toggle liquidity cycle">Liquidity</button>
      <div class="indicator-wrap">
        <button id="indicatorBtn" class="btn">Filter</button>
        <div id="indicatorMenu" class="indicator-menu">
          <label class="indicator-item"><input type="checkbox" value="SELL THE HOUSE"> SELL THE HOUSE</label>
          <label class="indicator-item"><input type="checkbox" value="Strong Buy"> Strong Buy</label>
          <label class="indicator-item"><input type="checkbox" value="Buy"> Buy</label>
          <label class="indicator-item"><input type="checkbox" value="DCA"> DCA</label>
          <label class="indicator-item"><input type="checkbox" value="Hold On"> Hold On</label>
          <label class="indicator-item"><input type="checkbox" value="Frothy"> Frothy</label>
          <label class="indicator-item"><input type="checkbox" value="Top Inbound"> Top Inbound</label>
          <div class="indicator-actions">
            <button id="indicatorClear" class="btn btn-sm">Clear</button>
            <button id="indicatorApply" class="btn btn-sm">Apply</button>
          </div>
        </div>
      </div>
      <button id="infoBtn" class="btn btn-icon" title="How it works">?</button>
    </div>
  </div>

  <div class="main">
    <!-- Chart area -->
    <div class="chart-area" id="leftCol">__PLOT_HTML__</div>

    <!-- Sidebar -->
    <div class="sidebar">
      <!-- Prediction Signal -->
      <div class="sidebar-section">
        <div id="predictionBox">
          <div id="predSignal">\u2014</div>
          <div id="predReason"></div>
        </div>
      </div>

      <!-- Current Values -->
      <div class="sidebar-section">
        <div class="readout-date" id="readoutDate">\u2014</div>
        <div class="readout-grid">
          <div class="readout-item"><span class="readout-label">Price</span><span class="readout-value" id="mainVal">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Position</span><span class="readout-value" id="pPct">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Oscillator</span><span class="readout-value" id="oscVal">\u2014</span></div>
          <div class="readout-item"><span class="readout-label">Composite</span><span class="readout-value" id="compLine">\u2014</span></div>
        </div>
        <div id="readoutRows" style="margin-top:8px;font-size:11px"></div>
      </div>

      <!-- Date Navigation -->
      <div class="sidebar-section">
        <div class="section-header" data-target="dateContent">
          <span class="section-title">Navigate</span>
          <span class="section-toggle">\u25BC</span>
        </div>
        <div class="section-content" id="dateContent">
          <div class="adv-row">
            <input type="date" id="datePick" style="flex:1"/>
            <button id="setDateBtn" class="btn btn-sm">Go</button>
            <button id="todayBtn" class="btn btn-sm">Today</button>
          </div>
        </div>
      </div>

      <!-- Tools (collapsible) -->
      <div class="sidebar-section">
        <div class="section-header" data-target="toolsContent">
          <span class="section-title">Tools</span>
          <span class="section-toggle">\u25BC</span>
        </div>
        <div class="section-content" id="toolsContent">
          <div class="adv-row">
            <label>Level</label>
            <input type="text" id="levelInput" placeholder="e.g. 50000" style="flex:1;min-width:80px"/>
            <button id="addLevelBtn" class="btn btn-sm">Add</button>
            <button id="clearLevelsBtn" class="btn btn-sm">Clear</button>
          </div>
          <div class="adv-row" style="margin-top:4px">
            <span class="smallnote">Right-click twice to draw trend line</span>
            <button id="clearTrendBtn" class="btn btn-sm">Clear</button>
          </div>
          <div class="adv-row" style="margin-top:4px">
            <button id="copyBtn" class="btn btn-sm" style="flex:1">Copy Chart</button>
          </div>
        </div>
      </div>

      <!-- Rails (collapsible) -->
      <div class="sidebar-section">
        <div class="section-header" data-target="railsContent">
          <span class="section-title">Rails</span>
          <span class="section-toggle">\u25B6</span>
        </div>
        <div class="section-content collapsed" id="railsContent">
          <div class="rails-list">Current: <span id="railsListText"></span></div>
          <div id="railsEditor" class="hidden" style="margin-top:8px">
            <div id="railItems"></div>
            <div class="rail-row" style="margin-top:6px">
              <input type="number" id="addPct" placeholder="%" step="0.1" min="0.1" max="99.9"/>
              <button id="addBtn" class="btn btn-sm">Add</button>
            </div>
          </div>
          <button id="editBtn" class="btn btn-sm" style="margin-top:6px">Edit</button>
        </div>
      </div>

      <!-- Advanced (collapsible) -->
      <div class="sidebar-section">
        <div class="section-header" data-target="advContent">
          <span class="section-title">Advanced</span>
          <span class="section-toggle">\u25B6</span>
        </div>
        <div class="section-content collapsed" id="advContent">
          <div class="adv-row">
            <label>Width</label>
            <input type="range" id="chartWslider" min="400" max="2400" step="10" value="1100"/>
            <input type="number" id="chartWpx" min="400" max="2400" step="10" value="1100" style="width:60px"/>
            <button id="fitBtn" class="btn btn-sm">Fit</button>
          </div>
          <div class="smallnote" style="margin-top:6px">Denominators: <span id="denomsDetected"></span></div>
          <div id="plParams" class="smallnote" style="margin-top:6px"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Info modal -->
<div id="infoModal" class="modal-overlay" role="dialog" aria-modal="true">
  <div class="modal-card">
    <h3>Bitcoin Power Law Indicator</h3>
    <p><b>The Power Law:</b> Bitcoin's price follows P = 10<sup>a</sup> \u00d7 t<sup>b</sup>, where b \u2248 5.8. On a log-log chart, this is a straight line (R\u00b2 \u2248 0.95).</p>
    <code>log\u2081\u2080(Price) = a + b \u00d7 log\u2081\u2080(years_since_genesis)</code>
    <p><b>Rails:</b> Quantile bands from median regression residuals. Floor (\u22482.5%) to Ceiling (\u224897.5%).</p>
    <p><b>Position (p):</b> Your position within the corridor (0-100%). Low = undervalued, high = overvalued.</p>
    <p><b>Oscillator:</b> Log-deviation normalized to [-1, +1]. ATHs occurred at 0.8-0.9. Values below -0.5 are accumulation zones.</p>
    <p><b>Composite:</b> Blends position, 65-month liquidity cycle, and halving window signals.</p>
    <p class="smallnote" style="margin-top:12px;padding-top:8px;border-top:1px solid #e2e8f0"><b>Sources:</b> Santostasi (2014), Burger (2019), Perrenod (2024), Mezinskis/Porkopolis</p>
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

// Prediction signal based on oscillator thresholds (Burger's oscillator zones)
function getPredictionSignal(osc, p){
  if (osc >= 0.85) return {signal: '\u26a0\ufe0f SELL ZONE', color: '#DC2626', bg: '#FEE2E2', reason: 'Oscillator >=0.85: Historically all ATHs occurred here. Consider taking profits.'};
  if (osc >= 0.7)  return {signal: '\u2b06\ufe0f FROTHY', color: '#F97316', bg: '#FFF7ED', reason: 'Oscillator 0.7-0.85: Approaching bubble territory. Reduce exposure.'};
  if (osc >= 0.4)  return {signal: '\u2197\ufe0f ELEVATED', color: '#EAB308', bg: '#FEFCE8', reason: 'Oscillator 0.4-0.7: Above fair value. Hold, be cautious adding.'};
  if (osc >= -0.2) return {signal: '\u2705 FAIR VALUE', color: '#16A34A', bg: '#F0FDF4', reason: 'Oscillator -0.2 to 0.4: Near trend line. Good for DCA.'};
  if (osc >= -0.5) return {signal: '\u2b07\ufe0f ACCUMULATE', color: '#0EA5E9', bg: '#F0F9FF', reason: 'Oscillator -0.5 to -0.2: Below trend. Good buying opportunity.'};
  if (osc >= -0.8) return {signal: '\ud83d\udcb0 STRONG BUY', color: '#2563EB', bg: '#EFF6FF', reason: 'Oscillator -0.8 to -0.5: Significantly undervalued. Aggressive accumulation.'};
  return {signal: '\ud83d\udea8 EXTREME BUY', color: '#7C3AED', bg: '#F5F3FF', reason: 'Oscillator <-0.8: Extreme undervaluation. Rare opportunity.'};
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

// DOM
const leftCol=document.getElementById('leftCol');
const sidebar=document.querySelector('.sidebar');
const plotDiv=(function(){ return document.querySelector('.chart-area .js-plotly-plot') || document.querySelector('.chart-area .plotly-graph-div'); })();
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const btnSet=document.getElementById('setDateBtn');
const btnToday=document.getElementById('todayBtn');
const btnCopy=document.getElementById('copyBtn');
const btnFit=document.getElementById('fitBtn');
const btnHalvings=document.getElementById('halvingsBtn');
const btnLiquidity=document.getElementById('liquidityBtn');
const infoBtn=document.getElementById('infoBtn');
const infoModal=document.getElementById('infoModal');
const infoClose=document.getElementById('infoClose');

const elDate=document.getElementById('readoutDate');
const elRows=document.getElementById('readoutRows');
const elMain=document.getElementById('mainVal');
const elP=document.getElementById('pPct');
const elComp=document.getElementById('compLine');
const chartWpx=document.getElementById('chartWpx');
const periodSel=document.getElementById('periodSel');
const xAxisSel=document.getElementById('xAxisSel');
const candleSel=document.getElementById('candleSel');
const predSignal=document.getElementById('predSignal');
const predReason=document.getElementById('predReason');
const predBox=document.getElementById('predictionBox');

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

const editBtn=document.getElementById('editBtn');
const railsView=document.getElementById('railsView');
const railsListText=document.getElementById('railsListText');
const railsEditor=document.getElementById('railsEditor');
const railItems=document.getElementById('railItems');
const addPct=document.getElementById('addPct');
const indicatorBtn=document.getElementById('indicatorBtn');
const indicatorMenu=document.getElementById('indicatorMenu');
const indicatorClear=document.getElementById('indicatorClear');
const indicatorApply=document.getElementById('indicatorApply');
const denomsDetected=document.getElementById('denomsDetected');

// Levels (horizontal lines) — state per denominator
const levelInput      = document.getElementById('levelInput');
const addLevelBtn     = document.getElementById('addLevelBtn');
const clearLevelsBtn  = document.getElementById('clearLevelsBtn');
const levelsByDenom = {};  // { USD:[numbers], GOLD:[...], ... }
Object.keys(PRECOMP).forEach(k => levelsByDenom[k] = []);

// Indicator UI
function getCheckedIndicators(){ const boxes=indicatorMenu.querySelectorAll('input[type=checkbox]'); const out=[]; boxes.forEach(b=>{if(b.checked) out.push(b.value)}); return out;}
function setCheckedIndicators(arr){ const set=new Set(arr); const boxes=indicatorMenu.querySelectorAll('input[type=checkbox]'); boxes.forEach(b=>{b.checked=set.has(b.value)}); }
function labelForIndicatorBtn(arr){ return (!arr.length)?'Indicator: All':(arr.length===1?'Indicator: '+arr[0]:'Indicator: '+arr.length+' selected'); }
let selectedIndicators=[];
indicatorBtn.addEventListener('click',e=>{e.stopPropagation(); indicatorMenu.classList.toggle('open');});
indicatorClear.addEventListener('click',e=>{e.preventDefault(); setCheckedIndicators([]);});
indicatorApply.addEventListener('click',e=>{e.preventDefault(); indicatorMenu.classList.remove('open'); selectedIndicators=getCheckedIndicators(); indicatorBtn.textContent=labelForIndicatorBtn(selectedIndicators); if(selectedIndicators.length>0){indicatorBtn.classList.add('active');}else{indicatorBtn.classList.remove('active');} applyIndicatorMask(PRECOMP[denomSel.value]);});
document.addEventListener('click',e=>{ if(!indicatorMenu.contains(e.target) && !indicatorBtn.contains(e.target)) indicatorMenu.classList.remove('open'); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape') indicatorMenu.classList.remove('open'); });

// Denominators
(function initDenoms(){
  const keys=Object.keys(PRECOMP); const order=['USD'].concat(keys.filter(k=>k!=='USD'));
  denomSel.innerHTML=''; order.forEach(k=>{ const o=document.createElement('option'); o.value=k; o.textContent=k; denomSel.appendChild(o); });
  denomSel.value='USD';
  if (denomsDetected) denomsDetected.textContent = order.filter(k=>k!=='USD').join(', ') || '(none)';
})();

// Rails state & UI
let rails=[97.5,90,75,50,25,2.5];
function sortRails(){ rails=rails.filter(p=>isFinite(p)).map(Number).map(p=>clamp(p,0.1,99.9)).filter((p,i,a)=>a.indexOf(p)===i).sort((a,b)=>b-a); }
function railsText(){ return rails.map(p=>String(p).replace(/\.0$/,'')+'%').join(', '); }
function idFor(p){ return 'v'+String(p).replace('.','_'); }

function rebuildReadoutRows(){
  elRows.innerHTML='';
  rails.forEach(p=>{
    const row=document.createElement('div'); row.className='row';
    const lab=document.createElement('div'); const val=document.createElement('div'); val.className='num'; val.id=idFor(p);
    const color=colorForPercent(p); const nm=(p===2.5?'Floor':(p===97.5?'Ceiling':(p+'%')));
    const is50 = (Math.abs(p - 50) < 0.01);
    lab.innerHTML='<span style="color:'+color+';'+(is50?'font-weight:700;':'')+'">'+nm+'</span>';
    row.appendChild(lab); row.appendChild(val); elRows.appendChild(row);
  });
}
function rebuildEditor(){
  railItems.innerHTML='';
  rails.forEach((p,idx)=>{
    const row=document.createElement('div'); row.className='rail-row';
    const color=colorForPercent(p);
    const labelTxt=(p===2.5?'Floor':(p===97.5?'Ceiling':(p+'%')));
    const lab=document.createElement('span'); lab.style.minWidth='48px'; lab.style.color=color; lab.textContent=labelTxt;
    const inp=document.createElement('input'); inp.type='number'; inp.step='0.1'; inp.min='0.1'; inp.max='99.9'; inp.value=String(p);
    const rm=document.createElement('button'); rm.textContent='Remove'; rm.className='btn';
    inp.addEventListener('change',()=>{ const v=Number(inp.value); rails[idx]=isFinite(v)?v:p; sortRails(); syncAll(); });
    rm.addEventListener('click',()=>{ rails.splice(idx,1); syncAll(); });
    row.appendChild(lab); row.appendChild(inp); row.appendChild(rm); railItems.appendChild(row);
  });
  railsListText.textContent=railsText();
}
document.getElementById('addBtn').addEventListener('click',()=>{ const v=Number(addPct.value); if(!isFinite(v)) return; rails.push(v); addPct.value=''; sortRails(); syncAll(); });
editBtn.addEventListener('click',()=>{ railsEditor.classList.toggle('hidden'); railsView.classList.toggle('hidden'); editBtn.textContent = railsEditor.classList.contains('hidden') ? 'Edit Rails' : 'Done'; });

// Sizing - chart now fills available space by default
const chartWslider=document.getElementById('chartWslider');
function triggerResize(){
  if(window.Plotly&&plotDiv){
    Plotly.Plots.resize(plotDiv);
    requestAnimationFrame(()=>Plotly.Plots.resize(plotDiv));
  }
}
chartWpx.addEventListener('input',()=>{ chartWslider.value=chartWpx.value; });
chartWpx.addEventListener('change',()=>{ chartWslider.value=chartWpx.value; });
chartWslider.addEventListener('input',()=>{ chartWpx.value=chartWslider.value; });

// Fit button triggers a resize
btnFit.addEventListener('click',()=>{ triggerResize(); });

// Auto-resize on window/container changes
if(window.ResizeObserver){
  new ResizeObserver(()=>triggerResize()).observe(leftCol);
}
window.addEventListener('resize',()=>triggerResize());

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
  railsListText.textContent=railsText();
}

// Indicator mask based on COMPOSITE classes
function computeSeriesForMask(P){
  if (P._classSeries) return;
  const d=P.support;
  const pSeries=new Array(P.x_main.length);
  for (let i=0;i<P.x_main.length;i++){
    const z=Math.log10(P.x_main[i]); const ly=Math.log10(P.y_main[i]);
    const mid=d.a0 + d.b*z; const off=ly-mid;
    pSeries[i]=clamp(percentFromOffset(P, off), 0, 100);
  }
  const classes=new Array(P.x_main.length);
  const scores=new Array(P.x_main.length);
  for (let i=0;i<P.x_main.length;i++){
    const iso = PRECOMP[denomSel.value].date_iso_main[i];
    const sc = compositeFrom(pSeries[i], iso);
    scores[i]=sc; classes[i]=classFromScore(sc);
  }
  P._pSeries=pSeries; P._scoreSeries=scores; P._classSeries=classes;
}
function applyIndicatorMask(P){
  const xMain = getXMain(P);
  const yMain = getYMain(P);
  if (!selectedIndicators || selectedIndicators.length===0){
    Plotly.restyle(plotDiv, {x:[xMain], y:[yMain], name:[P.label]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[xMain], y:[yMain]},                [IDX_CLICK]);
  } else {
    computeSeriesForMask(P);
    const set=new Set(selectedIndicators);
    const ym = yMain.map((v,i)=> set.has(P._classSeries[i]) ? v : null);
    const nm = P.label+' \u2014 '+Array.from(set).join(' + ');
    Plotly.restyle(plotDiv, {x:[xMain], y:[ym], name:[nm]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[xMain], y:[ym]},            [IDX_CLICK]);
  }
  const tt = (P.unit === '$') ? '%{customdata}<br>$%{y:.2f}<extra></extra>' : '%{customdata}<br>%{y:.6f}<extra></extra>';
  Plotly.restyle(plotDiv, {x:[xMain], y:[yMain], customdata:[P.date_iso_main], hovertemplate:[tt], line:[{width:0}]}, [IDX_TT]);
}

// Panel update (future uses 50% for main value; p hidden)
const elOsc = document.getElementById('oscVal');
const elPlParams = document.getElementById('plParams');

function updatePanel(P, xVal){
  const iso = xToISO(xVal);
  elDate.textContent = xToShortDate(xVal);

  const xg = getXGrid(P);
  const v50series = seriesForPercent(P,50);
  const v50 = interp(xg, v50series, xVal);

  rails.forEach(p=>{
    const v=interp(xg, seriesForPercent(P,p), xVal);
    const el=document.getElementById(idFor(p)); if(!el) return;
    const mult=(v50>0&&isFinite(v50))?` (${(v/v50).toFixed(1)}x)`:''; el.textContent=fmtVal(P,v)+mult;
  });

  const xMain = getXMain(P);
  const lastX = xMain[xMain.length-1];
  let usedP=null, mainTxt='', compScore=null, logDev=0;

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
    const off = logDev;
    usedP=clamp(percentFromOffset(P, off), 0, 100);
    elP.textContent=usedP.toFixed(1)+'%';
  }
  elMain.textContent = mainTxt;

  // Power Law Oscillator
  const osc = oscillatorFromDeviation(P, logDev);
  const oscCol = oscillatorColor(osc);
  elOsc.textContent = osc.toFixed(3);
  elOsc.style.color = oscCol;

  // Prediction Signal
  const pred = getPredictionSignal(osc, usedP);
  predSignal.textContent = pred.signal;
  predSignal.style.color = pred.color;
  predReason.textContent = pred.reason;
  predBox.style.borderColor = pred.color;
  predBox.style.background = pred.bg;

  compScore = compositeFrom(usedP, iso);
  elComp.textContent = compScore.toFixed(2) + ' \u2014 ' + classFromScore(compScore);
  setTitle(compScore);

  // Show power law parameters
  const supP = getSupport(P);
  const slope = supP.b.toFixed(4);
  const intercept = supP.a0.toFixed(4);
  const fairVal = Math.pow(10, supP.a0 + supP.b * Math.log10(xVal));
  const xLabel = xAxisMode==='days' ? 'days' : 't';
  elPlParams.innerHTML = 'Power Law: log\u2081\u2080(P) = '+intercept+' + '+slope+' \u00d7 log\u2081\u2080('+xLabel+')<br>Fair Value (50%): '+fmtVal(P, fairVal) + (xVal<=lastX ? '<br>Deviation: '+(logDev>=0?'+':'')+logDev.toFixed(4)+' log units' : '');
}

// NOTE: single-click should do NOTHING → no plotly_click handler

// Hover drives panel everywhere (carrier trace active in future, too)
plotDiv.on('plotly_hover', ev=>{
  if(!(ev.points && ev.points.length)) return;
  updatePanel(PRECOMP[denomSel.value], ev.points[0].x);
});

// Date lock
// Initialize date picker to actual today (user's local date)
function getTodayISO(){
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}
datePick.value = getTodayISO();
datePick.max = getTodayISO();  // Can't select future dates

btnSet.addEventListener('click',()=>{
  if(!datePick.value) return;
  const P=PRECOMP[denomSel.value];
  const xVal = isoToX(datePick.value);
  updatePanel(P, xVal);
});
datePick.addEventListener('keydown',(e)=>{
  if(e.key==='Enter'){ e.preventDefault(); btnSet.click(); }
});
btnToday.addEventListener('click',()=>{
  const P=PRECOMP[denomSel.value];
  const today = getTodayISO();
  datePick.value = today;
  // Use actual today or latest data point, whichever is earlier
  const todayX = isoToX(today);
  const xMain = getXMain(P);
  const lastDataX = xMain[xMain.length-1];
  updatePanel(P, Math.min(todayX, lastDataX));
});

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
  renderRails(P); applyIndicatorMask(P);
  redrawLevels(P); redrawTrendLines();
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

// ───────── Right-click trend line drawing ─────────
let trendLinePoints = [];
let trendLines = [];
const TREND_LINE_META = 'user-trend';

plotDiv.addEventListener('contextmenu', (e)=>{
  e.preventDefault();
  const bb = plotDiv.getBoundingClientRect();
  const fl = plotDiv._fullLayout;
  if (!fl || !fl.xaxis || !fl.yaxis) return;
  const xa = fl.xaxis, ya = fl.yaxis;
  const px = e.clientX - bb.left - fl.margin.l;
  const py = e.clientY - bb.top - fl.margin.t;
  if (px < 0 || py < 0 || px > xa._length || py > ya._length) return;
  const logx = xa.range[0] + (px / xa._length) * (xa.range[1] - xa.range[0]);
  const logy = ya.range[0] + ((ya._length - py) / ya._length) * (ya.range[1] - ya.range[0]);
  const xVal = Math.pow(10, logx);
  const yVal = Math.pow(10, logy);
  trendLinePoints.push({x: xVal, y: yVal, logx: logx, logy: logy});
  if (trendLinePoints.length === 2) {
    const p1 = trendLinePoints[0], p2 = trendLinePoints[1];
    const slope = (p2.logy - p1.logy) / (p2.logx - p1.logx);
    const intercept = p1.logy - slope * p1.logx;
    const xRange = xa.range;
    const x0log = xRange[0] - 0.5;
    const x1log = xRange[1] + 0.5;
    const y0log = intercept + slope * x0log;
    const y1log = intercept + slope * x1log;
    const line = {
      type:'line', xref:'x', yref:'y',
      x0: Math.pow(10, x0log), x1: Math.pow(10, x1log),
      y0: Math.pow(10, y0log), y1: Math.pow(10, y1log),
      line:{color:'#6366F1', width:1.8, dash:'solid'},
      layer:'above', meta: TREND_LINE_META,
      _slope: slope, _intercept: intercept
    };
    trendLines.push(line);
    const shapes = (plotDiv.layout.shapes||[]).concat([line]);
    Plotly.relayout(plotDiv, {shapes: shapes});
    trendLinePoints = [];
  }
});

function redrawTrendLines(){
  const shapes = (plotDiv.layout.shapes||[]).filter(s => s.meta !== TREND_LINE_META);
  const fl = plotDiv._fullLayout;
  if (!fl || !fl.xaxis) { Plotly.relayout(plotDiv, {shapes: shapes}); return; }
  const xRange = fl.xaxis.range;
  const extended = trendLines.map(tl => {
    const x0log = xRange[0] - 0.5;
    const x1log = xRange[1] + 0.5;
    const y0log = tl._intercept + tl._slope * x0log;
    const y1log = tl._intercept + tl._slope * x1log;
    return Object.assign({}, tl, {
      x0: Math.pow(10, x0log), x1: Math.pow(10, x1log),
      y0: Math.pow(10, y0log), y1: Math.pow(10, y1log)
    });
  });
  Plotly.relayout(plotDiv, {shapes: shapes.concat(extended)});
}

// Re-extend trend lines on pan/zoom
plotDiv.on('plotly_relayout', (e)=>{
  if (!e) return;
  const touchedX = Object.keys(e).some(k => k.startsWith('xaxis.'));
  if (!touchedX || trendLines.length === 0) return;
  requestAnimationFrame(()=> redrawTrendLines());
});

// Clear trend lines button
document.getElementById('clearTrendBtn').addEventListener('click',()=>{
  trendLines = [];
  trendLinePoints = [];
  const shapes = (plotDiv.layout.shapes||[]).filter(s => s.meta !== TREND_LINE_META);
  Plotly.relayout(plotDiv, {shapes: shapes});
});

// Copy chart
btnCopy.addEventListener('click', async ()=>{
  btnCopy.textContent='Copying...'; btnCopy.disabled=true;
  try{
    const url=await Plotly.toImage(plotDiv,{format:'png',scale:2});
    if(navigator.clipboard && window.ClipboardItem){
      const blob=await (await fetch(url)).blob();
      await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
      btnCopy.textContent='Copied!';
    } else {
      const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png';
      document.body.appendChild(a); a.click(); a.remove();
      btnCopy.textContent='Downloaded!';
    }
  }catch(e){
    console.error('Copy Chart failed:', e);
    btnCopy.textContent='Failed';
  }
  setTimeout(()=>{ btnCopy.textContent='Copy Chart'; btnCopy.disabled=false; }, 1500);
});

// ─────────── Level Lines: no axis stretch + always span visible x-range ───────────
function currentXRange(){
  const fl = (plotDiv && plotDiv._fullLayout) ? plotDiv._fullLayout : null;
  if (fl && fl.xaxis && Array.isArray(fl.xaxis.range)) return [fl.xaxis.range[0], fl.xaxis.range[1]];
  const P = PRECOMP[denomSel.value];
  return [P.x_main[0], P.x_main[P.x_main.length-1]];
}

function parseLevelInput(str){
  if(!str) return NaN;
  const cleaned = String(str).replace(/\$/g,'').replace(/,/g,'').trim();
  return Number(cleaned);
}

function levelShape(y){
  const xr = currentXRange();
  return {
    type:'line', xref:'x', yref:'y',
    x0:xr[0], x1:xr[1], y0:y, y1:y,
    line:{color:'#6B7280', width:1.6, dash:'dot'},
    layer:'above', meta:'level'
  };
}

function levelLabel(P, y){
  // Pin to right edge of plotting area in paper coords (prevents axis stretch)
  return {
    xref:'paper', x:1.005,
    yref:'y',     y:y,
    text: fmtVal(P, y), showarrow:false, xanchor:'left',
    bgcolor:'rgba(0,0,0,0.03)', bordercolor:'#9CA3AF',
    font:{size:11}, meta:'level'
  };
}

function redrawLevels(P){
  const keepShapes = (plotDiv.layout.shapes||[]).filter(s => s.meta!=='level');
  const keepAnn    = (plotDiv.layout.annotations||[]).filter(a => a.meta!=='level');

  const addShapes=[], addAnn=[];
  (levelsByDenom[denomSel.value]||[]).forEach(v => {
    addShapes.push(levelShape(v));
    addAnn.push(levelLabel(P, v));
  });

  Plotly.relayout(plotDiv, {
    shapes: keepShapes.concat(addShapes),
    annotations: keepAnn.concat(addAnn)
  });
}

addLevelBtn.addEventListener('click', () => {
  const P = PRECOMP[denomSel.value];
  const v = parseLevelInput(levelInput.value);
  if(!isFinite(v) || v <= 0){
    alert('Please enter a positive number (price/ratio).');
    return;
  }
  const arr = levelsByDenom[denomSel.value];
  if(!arr.some(x => Math.abs(x - v) <= (Math.abs(v)*1e-12))) arr.push(v);
  redrawLevels(P);
});
clearLevelsBtn.addEventListener('click', () => {
  levelsByDenom[denomSel.value] = [];
  redrawLevels(PRECOMP[denomSel.value]);
});

// Keep level lines spanning current x-range after any pan/zoom/autorange
plotDiv.on('plotly_relayout', (e)=>{
  if (!e) return;
  const touchedX = Object.keys(e).some(k => k.startsWith('xaxis.'));
  if (!touchedX) return;
  const xr = currentXRange();
  const shapes = (plotDiv.layout.shapes||[]).map(s=>{
    if (s.meta === 'level') return Object.assign({}, s, { x0: xr[0], x1: xr[1] });
    return s;
  });
  Plotly.relayout(plotDiv, { shapes });
});

// Robust double-click reset that also realigns level lines
plotDiv.on('plotly_doubleclick', ()=>{
  Plotly.relayout(plotDiv, {'xaxis.autorange': true, 'yaxis.autorange': true});
  requestAnimationFrame(()=>{
    const xr = currentXRange();
    const shapes = (plotDiv.layout.shapes||[]).map(s=>{
      if (s.meta === 'level') return Object.assign({}, s, { x0: xr[0], x1: xr[1] });
      return s;
    });
    Plotly.relayout(plotDiv, { shapes });
  });
  return false; // prevents race with built-in handler on some mobile Safari versions
});

// Denominator change
denomSel.addEventListener('change',()=>{
  fullRedraw();
});

// Sync all
function syncAll(){
  sortRails(); rebuildEditor();
  const P=PRECOMP[denomSel.value];
  const xMain = getXMain(P);
  renderRails(P); applyIndicatorMask(P); redrawLevels(P);
  updatePanel(P, xMain[xMain.length-1]);
}

// Init
(function init(){
  rebuildReadoutRows(); rebuildEditor();
  const P=PRECOMP['USD'];
  const xMain = getXMain(P);
  const yMain = getYMain(P);
  const xg = getXGrid(P);
  renderRails(P); applyIndicatorMask(P);
  Plotly.restyle(plotDiv,{x:[xg], y:[seriesForPercent(P,50)]},[IDX_CARRY]);
  const tt='%{customdata}<br>$%{y:.2f}<extra></extra>';
  Plotly.restyle(plotDiv,{x:[xMain], y:[yMain], customdata:[P.date_iso_main], hovertemplate:[tt], line:[{width:0}]},[IDX_TT]);
  updatePanel(P, xMain[xMain.length-1]);
  redrawLevels(P);
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
    btnHalvings.textContent='Halvings';
  }else{
    const remain=(plotDiv.layout.shapes||[]).filter(s=>s.meta!=='halving');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnHalvings.classList.remove('active');
    btnHalvings.textContent='Halvings';
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
    btnLiquidity.textContent = 'Liquidity';
  } else {
    const remain = (plotDiv.layout.shapes||[]).filter(s => s.meta!=='liquidity');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnLiquidity.classList.remove('active');
    btnLiquidity.textContent = 'Liquidity';
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
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print("Wrote", OUTPUT_HTML)
