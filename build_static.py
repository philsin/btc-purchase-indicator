#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — median power-law with *smooth* residual quantiles,
future-stable rails, interactive dashboard, halvings & liquidity toggles.

This version models dispersion as *time-varying*:
- Fit median line in log–log space (QuantReg @ q=0.5).
- Compute residuals r = log10(y) - (a0 + b*log10(x)).
- For a grid of percentiles q in [1%, 99%], estimate *rolling* residual
  quantiles along log10(x) (nearest-neighbor window), then linearly
  interpolate to the plotting grid (x_grid).
- Rails use c_q(z) offsets so the multiples vs the midline can vary over
  history, but are *frozen* after the last observed date (stable for planning).

UI:
- Denominators: USD (or None), GOLD, SPX, ETH (auto-fetched with fallbacks).
- Edit/select percentile rails (sorted, dashed lines; 50% dashed too).
- Indicator multi-select (SELL THE HOUSE … Top Inbound) filters the BTC line.
- Click/hover panel shows values + (multiple vs 50%).
- “Copy Chart” copies current view (zoom respected).
- Halvings (past solid, future dashed).
- Liquidity cycle: 65-month peak-to-peak, PEAK anchored at 2015-02-01 (red),
  TROUGHS half-cycle later (green); drawn back to 2009-01-03.

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
X_START_DATE = datetime(2011, 1, 1)

RESID_WINSOR     = 0.02           # clip tails of residuals before smoothing
WINDOW_FRAC      = 0.18           # rolling window (fraction of samples) for smooth quantiles
Q_MIN, Q_MAX, Q_STEP = 0.01, 0.99, 0.01  # grid of residual quantiles
EPS_LOG_SPACING  = 0.010          # small spread guard at floor/ceiling

COL_BTC          = "#000000"

# Halvings: past + projected
PAST_HALVINGS   = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
FUTURE_HALVINGS = ["2028-04-20", "2032-04-20", "2036-04-19", "2040-04-19"]

# Liquidity cycle (peak anchored)
LIQ_PEAK_ANCHOR_ISO = "2015-02-01"  # PEAK (red)
LIQ_PERIOD_MONTHS   = 65            # peak-to-peak

# Draw liquidity lines back to genesis so pre-2015 peaks/troughs show up
LIQ_START_ISO = "2009-01-03"

# Auto denominators (daily closes). Multiple fallbacks for ETH.
AUTO_DENOMS = {
    "GOLD": {"path": os.path.join(DATA_DIR, "denominator_gold.csv"),
             "url":  "https://stooq.com/q/d/l/?s=xauusd&i=d", "parser":"stooq"},
    "SPX":  {"path": os.path.join(DATA_DIR, "denominator_spx.csv"),
             "url":  "https://stooq.com/q/d/l/?s=%5Espx&i=d", "parser":"stooq"},
    "ETH":  {"path": os.path.join(DATA_DIR, "denominator_eth.csv"),
             "url":  "https://stooq.com/q/d/l/?s=ethusd&i=d", "parser":"stooq"},
}
UA = {"User-Agent":"btc-indicator/1.0"}

# ─────────────────────── Data helpers / fetchers ───────────────────────
def years_since_genesis(dates):
    d = pd.to_datetime(dates)
    delta_days = (d - GENESIS_DATE) / np.timedelta64(1, "D")
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)

def fetch_btc_csv() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(BTC_FILE):
        df = pd.read_csv(BTC_FILE, parse_dates=["date"])
        return df.sort_values("date").dropna()
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
    df = df.sort_values("date").dropna()
    df.to_csv(BTC_FILE, index=False)
    return df

def _fetch_stooq_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers=UA); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)
    if "date" not in df.columns or "close" not in df.columns or df.empty:
        raise ValueError("stooq returned no usable columns")
    out = df[["date","close"]].rename(columns={"date":"date","close":"price"})
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
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, info in AUTO_DENOMS.items():
        path = info["path"]
        if os.path.exists(path): continue
        try:
            if key == "ETH":
                df=None
                for fn in (_fetch_eth_from_stooq, _fetch_eth_from_coingecko,
                           _fetch_eth_from_cryptocompare, _fetch_eth_from_binance):
                    try:
                        df = fn()
                        if df is not None and not df.empty: break
                    except Exception:
                        df = None
                if df is None or df.empty: raise ValueError("ETH fetchers returned no data")
            else:
                df = _fetch_stooq_csv(info["url"]) if info["parser"]=="stooq" else None
            if df is None or df.empty: raise ValueError(f"{key} returned no data")
            df.to_csv(path, index=False)
            print(f"[auto-denom] wrote {path} ({len(df)} rows)")
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

# ───────────────────────── Fit / rails support ─────────────────────────
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
    return a0, b, resid, xlog

def smooth_residual_quantiles(xlog, resid, x_grid):
    """Return q_grid, z_grid, off_qz (shape: [len(q_grid), len(z_grid)]).
       Uses nearest-neighbor rolling window along xlog (sorted), per-quantile."""
    q_grid = np.arange(Q_MIN, Q_MAX + 1e-9, Q_STEP)
    # sort by xlog
    order = np.argsort(xlog)
    xs = xlog[order]
    rs = resid[order]
    n = len(xs)
    # rolling window size
    k = int(max(101, round(WINDOW_FRAC * n)))
    if k % 2 == 0: k += 1
    m = k // 2
    # compute quantiles at each observed xs
    Qmat = np.zeros((n, len(q_grid)), dtype=float)
    for j in range(n):
        lo = max(0, j - m)
        hi = min(n - 1, j + m)
        window = rs[lo:hi+1]
        # robust fallback if window tiny
        if len(window) < 8:
            Qmat[j, :] = np.quantile(rs, q_grid)
        else:
            Qmat[j, :] = np.quantile(window, q_grid)
        # enforce monotone across q to avoid tiny inversions
        Qmat[j, :] = np.maximum.accumulate(Qmat[j, :])
    # interpolate to plotting grid (z_grid)
    z_grid = np.log10(x_grid)
    off_qz = np.zeros((len(q_grid), len(z_grid)), dtype=float)
    for i_q in range(len(q_grid)):
        off_qz[i_q, :] = np.interp(z_grid, xs, Qmat[:, i_q], left=Qmat[0, i_q], right=Qmat[-1, i_q])
    return q_grid, z_grid, off_qz

def build_payload(df, denom_key=None):
    # y series for given denominator
    if not denom_key or denom_key.lower() in ("usd", "none"):
        y = df["btc"].copy()
        label, unit, decimals = "BTC / USD", "$", 2
    else:
        k = denom_key.lower()
        if k in df.columns:
            y = df["btc"]/df[k]
            label, unit, decimals = f"BTC / {denom_key.upper()}", "", 6
        else:
            y = df["btc"].copy()
            label, unit, decimals = "BTC / USD", "$", 2

    mask = np.isfinite(df["x_years"].values) & np.isfinite(y.values)
    xs = df["x_years"].values[mask]
    ys = y.values[mask]
    dates = df["date_iso"].values[mask]

    # median fit + residuals (winsorized)
    a0, b, resid, xlog = quantile_fit_loglog(xs, ys, q=0.5)
    if RESID_WINSOR:
        resid = winsorize(resid, RESID_WINSOR)

    # grid and smooth residual quantiles
    q_grid, z_grid, off_qz = smooth_residual_quantiles(xlog, resid, x_grid)

    support = {
        "a0": float(a0), "b": float(b),
        "q_grid": [float(q) for q in q_grid],
        "z_grid": [float(z) for z in z_grid],
        "off_qz": [[float(v) for v in row] for row in off_qz]  # list of lists [q][z]
    }
    return {
        "label": label, "unit": unit, "decimals": decimals,
        "x_main": xs.tolist(), "y_main": ys.tolist(),
        "date_iso_main": dates.tolist(),
        "x_grid": x_grid.tolist(),
        "support": support
    }

# ───────────────────────────── Build model ─────────────────────────────
btc = fetch_btc_csv().rename(columns={"price":"btc"})
denoms = load_denominators()
print("[denoms]", list(denoms.keys()))

base = btc.sort_values("date").reset_index(drop=True)
for key, df in denoms.items():
    base = base.merge(df.rename(columns={"price": key.lower()}), on="date", how="left")

base["x_years"]   = years_since_genesis(base["date"])
base["date_iso"]  = base["date"].dt.strftime("%Y-%m-%d")

first_dt = max(base["date"].iloc[0], X_START_DATE)
max_dt   = END_PROJ

x_start = float(years_since_genesis(pd.Series([first_dt])).iloc[0])
x_end   = float(years_since_genesis(pd.Series([max_dt])).iloc[0])
x_grid  = np.logspace(np.log10(max(1e-6, x_start)), np.log10(x_end), 700)

# y-axis ticks and x ticks
def year_ticks_log(first_dt, last_dt):
    vals, labs = [], []
    for y in range(first_dt.year, last_dt.year+1):
        d = datetime(y,1,1)
        if d < first_dt or d > last_dt: continue
        vy = float(years_since_genesis(pd.Series([d])).iloc[0])
        if vy <= 0: continue
        vals.append(vy); labs.append(str(y))
    return vals, labs

def y_ticks():
    vals = [10**e for e in range(0,9)]
    labs = [f"{int(10**e):,}" for e in range(0,9)]
    return vals, labs

xtickvals, xticktext = year_ticks_log(first_dt, max_dt)
ytickvals, yticktext = y_ticks()

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)
P0 = PRECOMP["USD"]

LAST_PRICE_ISO = str(pd.to_datetime(base["date"]).max().date())

# ───────────────────── Plot base figure and traces ─────────────────────
MAX_RAIL_SLOTS = 12
IDX_MAIN  = MAX_RAIL_SLOTS
IDX_CLICK = MAX_RAIL_SLOTS + 1
IDX_CURSR = MAX_RAIL_SLOTS + 2
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
               name="_cursor", showlegend=False, hoverinfo="skip",
               line=dict(width=0.1, color="rgba(0,0,0,0.001)")),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="_tooltip", showlegend=False,
               hovertemplate="%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>",
               line=dict(width=0, color="rgba(0,0,0,0)")),
]

fig = go.Figure(traces)
fig.update_layout(
    template="plotly_white",
    hovermode="x",
    showlegend=True,
    title="BTC Purchase Indicator — ",
    xaxis=dict(type="log", title=None, tickmode="array",
               tickvals=xtickvals, ticktext=xticktext,
               tickangle=45,
               range=[np.log10(x_start), np.log10(x_end)]),
    yaxis=dict(type="log", title=P0["label"],
               tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),
)

plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive":True,"displayModeBar":True,"modeBarButtonsToRemove":["toImage"]})

# ───────────────────────────────── HTML (placeholders replaced) ─────────────────────────────────
HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
:root{--panelW:420px;}
html,body{height:100%} body{margin:0;font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
.layout{display:flex;min-height:100vh;width:100vw}
.left{flex:0 0 auto;width:1100px;min-width:280px;padding:8px 0 8px 8px}
.left .js-plotly-plot,.left .plotly-graph-div{width:100%!important}
.right{flex:0 0 var(--panelW);border-left:1px solid #e5e7eb;padding:12px;display:flex;flex-direction:column;gap:12px;overflow:auto}
#controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,input[type=date],input[type=number]{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}
.btn{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;cursor:pointer;transition:background 120ms, box-shadow 120ms}
.btn:hover{background:#f3f4f6}
.btn:active{background:#e5e7eb; box-shadow:inset 0 1px 2px rgba(0,0,0,0.08)}
#readout{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fafafa;font-size:14px}
#readout .date{font-weight:700;margin-bottom:6px}
#readout .row{display:grid;grid-template-columns:auto 1fr auto;column-gap:8px;align-items:baseline}
#readout .num{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;text-align:right;min-width:12ch;white-space:pre}
fieldset{border:1px solid #e5e7eb;border-radius:8px;padding:8px 10px}
legend{padding:0 6px;color:#374151;font-weight:600;font-size:13px}
.rail-row{display:flex;align-items:center;gap:8px;margin:2px 0}
.rail-row input[type=number]{width:90px}
.rail-row button{padding:4px 8px;font-size:12px}
.smallnote{font-size:12px;color:#6b7280}
@media (max-width:900px){
  .layout{flex-direction:column}
  .right{flex:0 0 auto;border-left:none;border-top:1px solid #e5e7eb}
  .left{flex:0 0 auto;width:100%;padding:8px}
}
#chartWidthBox{display:flex;align-items:center;gap:8px}
#chartWpx{width:120px}
.hidden{display:none}

/* Indicator multi-select */
.indicator-wrap{position:relative; display:inline-block;}
.indicator-btn{padding:8px 10px; border:1px solid #d1d5db; border-radius:8px; background:#fff; cursor:pointer; font-size:14px; transition:background 120ms}
.indicator-btn:hover{background:#f3f4f6}
.indicator-btn:active{background:#e5e7eb}
.indicator-menu{position:absolute; top:100%; left:0; z-index:50; min-width:220px; background:white; border:1px solid #e5e7eb; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.08); padding:8px; display:none;}
.indicator-menu.open{display:block;}
.indicator-item{display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:6px; cursor:pointer;}
.indicator-item:hover{background:#f3f4f6;}
.indicator-actions{display:flex; justify-content:space-between; gap:8px; padding-top:6px;}
.indicator-actions button{padding:6px 10px; font-size:12px;}
</style>
</head><body>
<div id="capture" class="layout">
  <div class="left" id="leftCol">__PLOT_HTML__</div>
  <div class="right">
    <div id="controls">
      <label for="denomSel"><b>Denominator:</b></label>
      <select id="denomSel"></select>
      <input type="date" id="datePick"/>
      <button id="setDateBtn" class="btn">Set Date</button>
      <button id="todayBtn" class="btn">Today</button>

      <div class="indicator-wrap">
        <button id="indicatorBtn" class="indicator-btn">Indicator: All</button>
        <div id="indicatorMenu" class="indicator-menu">
          <label class="indicator-item"><input type="checkbox" value="SELL THE HOUSE"> SELL THE HOUSE</label>
          <label class="indicator-item"><input type="checkbox" value="Strong Buy"> Strong Buy</label>
          <label class="indicator-item"><input type="checkbox" value="Buy"> Buy</label>
          <label class="indicator-item"><input type="checkbox" value="DCA"> DCA</label>
          <label class="indicator-item"><input type="checkbox" value="Hold On"> Hold On</label>
          <label class="indicator-item"><input type="checkbox" value="Frothy"> Frothy</label>
          <label class="indicator-item"><input type="checkbox" value="Top Inbound"> Top Inbound</label>
          <div class="indicator-actions">
            <button id="indicatorClear" class="btn">Clear</button>
            <button id="indicatorApply" class="btn">Apply</button>
          </div>
        </div>
      </div>

      <button id="halvingsBtn" class="btn" title="Toggle halving lines">Halvings</button>
      <button id="liquidityBtn" class="btn" title="Toggle liquidity cycle">Liquidity</button>
      <button id="copyBtn" class="btn" title="Copy current view to clipboard">Copy Chart</button>
    </div>

    <div id="chartWidthBox">
      <b>Chart Width (px):</b>
      <input type="number" id="chartWpx" min="400" max="2400" step="10" value="1100"/>
      <button id="fitBtn" class="btn" title="Make chart fill remaining space">Fit</button>
    </div>

    <fieldset id="railsBox">
      <legend>Rails</legend>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
        <button id="editBtn" class="btn">Edit Rails</button>
        <span class="smallnote">Add/remove/change percents. Sorted automatically (high→low). 50% is dashed.</span>
      </div>
      <div id="railsView" class="smallnote">Current: <span id="railsListText"></span></div>

      <div id="railsEditor" class="hidden">
        <div id="railItems"></div>
        <div class="rail-row" style="margin-top:6px;">
          <input type="number" id="addPct" placeholder="Add % (e.g. 92.5)" step="0.1" min="0.1" max="99.9"/>
          <button id="addBtn" class="btn">Add</button>
        </div>
      </div>
    </fieldset>

    <div class="smallnote">Detected denominators: <span id="denomsDetected"></span></div>

    <div id="readout">
      <div class="date">—</div>
      <div id="readoutRows"></div>
      <div style="margin-top:10px;"><b id="mainLabel">BTC Price:</b> <span id="mainVal" class="num">—</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>

<script>
const PRECOMP = __PRECOMP__;
const GENESIS = new Date('__GENESIS__T00:00:00Z');
const END_ISO = '__END_ISO__';
const LAST_PRICE_ISO = '__LAST_PRICE_ISO__';

const MAX_SLOTS = __MAX_RAIL_SLOTS__;
const IDX_MAIN  = __IDX_MAIN__;
const IDX_CLICK = __IDX_CLICK__;
const IDX_CURSR = __IDX_CURSR__;
const IDX_TT    = __IDX_TT__;
const EPS_LOG_SPACING = __EPS_LOG_SPACING__;
const PAST_HALVINGS = __PAST_HALVINGS__;
const FUTURE_HALVINGS = __FUTURE_HALVINGS__;

const LIQ_PEAK_ANCHOR_ISO = '__LIQ_PEAK_ANCHOR_ISO__'; // PEAK (red)
const LIQ_PERIOD_MONTHS   = __LIQ_PERIOD_MONTHS__;     // peak-to-peak
const LIQ_START_ISO       = '__LIQ_START_ISO__';

const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }
function shortDateFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function interp(xs, ys, x){ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }

// value formatting
function fmtVal(P, v){
  if(!isFinite(v)) return '—';
  const dec = Math.max(0, Math.min(10, P.decimals||2));
  if (P.unit === '$') return '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:dec,maximumFractionDigits:dec});
  const maxd=Math.max(dec,6);
  return Number(v).toLocaleString(undefined,{minimumFractionDigits:Math.min(6,maxd),maximumFractionDigits:maxd});
}

// Colors and Indicator ranges
function colorForPercent(p){ const t=clamp(p/100,0,1);
  function hx(v){return Math.max(0,Math.min(255,Math.round(v))); }
  function toHex(r,g,b){return '#'+[r,g,b].map(v=>hx(v).toString(16).padStart(2,'0')).join('');}
  if(t<=0.5){const u=t/0.5;return toHex(0xD3+(0xFB-0xD3)*u,0x2F+(0xC0-0x2F)*u,0x2F+(0x2D-0x2F)*u);}
  const u=(t-0.5)/0.5; return toHex(0xFB+(0x2E-0xFB)*u,0xC0+(0x7D-0xC0)*u,0x2D+(0x32-0x2D)*u);
}
function indicatorFromP(p){
  if (p < 2.5) return 'SELL THE HOUSE';
  if (p < 25)  return 'Strong Buy';
  if (p < 50)  return 'Buy';
  if (p < 75)  return 'DCA';
  if (p < 90)  return 'Hold On';
  if (p <= 97.5) return 'Frothy';
  return 'Top Inbound';
}
function rangeForIndicator(name){
  switch(name){
    case 'SELL THE HOUSE': return [-Infinity, 2.5];
    case 'Strong Buy':     return [2.5, 25];
    case 'Buy':            return [25, 50];
    case 'DCA':            return [50, 75];
    case 'Hold On':        return [75, 90];
    case 'Frothy':         return [90, 97.5];
    case 'Top Inbound':    return [97.5, Infinity];
    default:               return [-Infinity, Infinity];
  }
}

// DOM refs
const leftCol=document.getElementById('leftCol');
const plotDiv=document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div');
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const btnSet=document.getElementById('setDateBtn');
const btnToday=document.getElementById('todayBtn');
const btnCopy=document.getElementById('copyBtn');
const btnFit=document.getElementById('fitBtn');
const btnHalvings=document.getElementById('halvingsBtn');
const btnLiquidity=document.getElementById('liquidityBtn');
const elDate=document.querySelector('#readout .date');
const elRows=document.getElementById('readoutRows');
const elMain=document.getElementById('mainVal');
const elMainLabel=document.getElementById('mainLabel');
const elP=document.getElementById('pPct');
const chartWpx=document.getElementById('chartWpx');
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

// Indicator UI
function getCheckedIndicators(){ const boxes=indicatorMenu.querySelectorAll('input[type=checkbox]'); const out=[]; boxes.forEach(b=>{if(b.checked) out.push(b.value)}); return out;}
function setCheckedIndicators(arr){ const set=new Set(arr); const boxes=indicatorMenu.querySelectorAll('input[type=checkbox]'); boxes.forEach(b=>{b.checked=set.has(b.value)}); }
function labelForIndicatorBtn(arr){ return (!arr.length)?'Indicator: All':(arr.length===1?'Indicator: '+arr[0]:'Indicator: '+arr.length+' selected'); }
let selectedIndicators=[];
indicatorBtn.addEventListener('click',e=>{e.stopPropagation(); indicatorMenu.classList.toggle('open');});
indicatorClear.addEventListener('click',e=>{e.preventDefault(); setCheckedIndicators([]);});
indicatorApply.addEventListener('click',e=>{e.preventDefault(); indicatorMenu.classList.remove('open'); selectedIndicators=getCheckedIndicators(); indicatorBtn.textContent=labelForIndicatorBtn(selectedIndicators); applyIndicatorMask(PRECOMP[denomSel.value]);});
document.addEventListener('click',e=>{ if(!indicatorMenu.contains(e.target) && !indicatorBtn.contains(e.target)) indicatorMenu.classList.remove('open'); });

// Denominator init
(function initDenoms(){
  const keys=Object.keys(PRECOMP); const order=['USD'].concat(keys.filter(k=>k!=='USD'));
  denomSel.innerHTML=''; order.forEach(k=>{ const o=document.createElement('option'); o.value=k; o.textContent=k; denomSel.appendChild(o); });
  denomSel.value='USD';
  if (denomsDetected) denomsDetected.textContent = order.filter(k=>k!=='USD').join(', ') || '(none)';
})();

// Rails state & UI
let rails=[97.5,90,75,50,25,2.5];
function sortRails(){ rails=rails.filter(p=>isFinite(p)).map(Number).map(p=>clamp(p,0.1,99.9)).filter((p,i,a)=>a.indexOf(p)===i).sort((a,b)=>b-a); }
function railsText(){ return rails.map(p=>String(p).replace(/\\.0$/,'')+'%').join(', '); }
function idFor(p){ return 'v'+String(p).replace('.','_'); }

function rebuildReadoutRows(){
  elRows.innerHTML='';
  rails.forEach(p=>{
    const row=document.createElement('div'); row.className='row';
    const lab=document.createElement('div'); const val=document.createElement('div'); val.className='num'; val.id=idFor(p);
    const color=colorForPercent(p); const nm=(p===2.5?'Floor':(p===97.5?'Ceiling':(p+'%')));
    lab.innerHTML=`<span style="color:${color};">${nm}</span>`;
    row.appendChild(lab); row.appendChild(val); row.appendChild(document.createElement('div')); elRows.appendChild(row);
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

// Sizing
function applyChartWidthPx(px){
  const v=clamp(Number(px)||1100, 400, 2400);
  leftCol.style.flex='0 0 auto'; leftCol.style.width=v+'px';
  if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv);
}
chartWpx.addEventListener('change',()=>applyChartWidthPx(chartWpx.value));
btnFit.addEventListener('click',()=>{
  const total=document.documentElement.clientWidth||window.innerWidth, panel=420, pad=32;
  const target=Math.max(400, total-panel-pad); chartWpx.value=target; applyChartWidthPx(target);
});
if(window.ResizeObserver) new ResizeObserver(()=>{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }).observe(leftCol);

// ─────────── Rails math (time-varying residual quantiles, future-frozen) ───────────
function logMidline(P){ const d=P.support; return d.z_grid.map(z=> d.a0 + d.b*z ); }

// offsets for given percent across x_grid (z_grid aligned)
function offsetsForPercent(P, percent){
  const d=P.support, qGrid=d.q_grid, off=d.off_qz; // off[q][z]
  const p01=clamp(percent/100, qGrid[0], qGrid[qGrid.length-1]);
  // for each z index, interpolate in q dimension
  const out=new Array(d.z_grid.length);
  for (let zi=0; zi<d.z_grid.length; zi++){
    // build column (off vs q) for this z
    // offCol is monotone
    let lo=0, hi=qGrid.length-1;
    while(hi-lo>1){ const m=(hi+lo)>>1; if(qGrid[m]<=p01) lo=m; else hi=m; }
    const t=(p01-qGrid[lo])/(qGrid[hi]-qGrid[lo]);
    out[zi]= off[lo][zi] + t*(off[hi][zi]-off[lo][zi]);
  }
  return out;
}

// produce y-series for a given percent (guard spacing at floor/ceiling)
function seriesForPercent(P, percent){
  const logM = logMidline(P);
  const offs = offsetsForPercent(P, percent);
  const out  = new Array(P.x_grid.length);
  const eps  = (percent>=97.5||percent<=2.5)?EPS_LOG_SPACING:0.0;
  for (let i=0;i<out.length;i++){ out[i]=Math.pow(10, logM[i] + offs[i] + (percent>=50?eps:-eps)); }
  return out;
}

// compute percentile p of a (x,y) point from local residual curve
function percentAtPoint(P, xYears, yVal){
  const d=P.support, z=Math.log10(xYears), logy=Math.log10(yVal);
  const mid = d.a0 + d.b*z;
  const off = logy - mid;
  // choose nearest z index
  const zArr=d.z_grid; let lo=0, hi=zArr.length-1;
  if (z<=zArr[0]){ lo=hi=0; }
  else if (z>=zArr[hi]){ lo=hi=hi; }
  else { while(hi-lo>1){ const m=(hi+lo)>>1; if(zArr[m]<=z) lo=m; else hi=m; } }
  // access off vs q at column lo (nearest)
  const qGrid=d.q_grid; const offCol = d.off_qz.map(row=>row[lo]);
  // invert offCol(q) to get q(off)
  let a=0, b=qGrid.length-1;
  if (off<=offCol[0]) return 100*qGrid[0];
  if (off>=offCol[b]) return 100*qGrid[b];
  while(b-a>1){ const m=(a+b)>>1; if(offCol[m]<=off) a=m; else b=m; }
  const t=(off-offCol[a])/(offCol[b]-offCol[a]);
  const q = qGrid[a] + t*(qGrid[b]-qGrid[a]);
  return 100*clamp(q,0,1);
}

// Render rails
function renderRails(P){
  const n=Math.min(rails.length, MAX_SLOTS);
  for(let i=0;i<MAX_SLOTS;i++){
    const visible=(i<n); let restyle={visible};
    if(visible){
      const p=rails[i], color=colorForPercent(p);
      const nm=(p===2.5?'Floor':(p===97.5?'Ceiling':(p+'%')));
      restyle=Object.assign(restyle,{x:[P.x_grid], y:[seriesForPercent(P,p)], name:nm, line:{color:color, width:1.6, dash:'dot'}});
    }
    Plotly.restyle(plotDiv, restyle, [i]);
  }
  railsListText.textContent=railsText(); rebuildReadoutRows();
}

let locked=false, lockedX=null;
function setTitleForP(p){ Plotly.relayout(plotDiv, {'title.text': 'BTC Purchase Indicator — '+indicatorFromP(p)}); }

// Indicator mask
function computePSeries(P){
  const out=new Array(P.x_main.length);
  for (let i=0;i<P.x_main.length;i++){
    out[i]=percentAtPoint(P, P.x_main[i], P.y_main[i]);
  }
  P._pSeries = out;
}
function applyIndicatorMask(P){
  if (!selectedIndicators || selectedIndicators.length===0){
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]},                [IDX_CLICK]);
  } else {
    if (!P._pSeries) computePSeries(P);
    const ranges = selectedIndicators.map(rangeForIndicator);
    const ym = P.y_main.map((v,i)=>{ const p=P._pSeries[i]; for (let r of ranges){ if (p>=r[0] && p<r[1]) return v; } return null; });
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym], name:[P.label+' — '+selectedIndicators.join(' + ')]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym]},                                                           [IDX_CLICK]);
  }
  const tt = (P.unit === '$') ? "%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>" : "%{x|%b-%d-%Y}<br>%{y:.6f}<extra></extra>";
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], hovertemplate:[tt], line:[{width:0}]}, [IDX_TT]);
}

function updatePanel(P,xYears){
  elDate.textContent=shortDateFromYears(xYears);
  const v50 = interp(P.x_grid, seriesForPercent(P,50), xYears);
  rails.forEach(p=>{ const v=interp(P.x_grid, seriesForPercent(P,p), xYears);
    const el=document.getElementById(idFor(p)); if(!el) return;
    const mult=(v50>0&&isFinite(v50))?` (${(v/v50).toFixed(1)}x)`:''; el.textContent=fmtVal(P,v)+mult;
  });

  const lastX=P.x_main[P.x_main.length-1];
  if (xYears>lastX){ elMain.textContent='—'; elP.textContent='(p≈—)'; }
  else{
    let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
    const y=P.y_main[idx]; elMain.textContent=fmtVal(P,y); elMainLabel.textContent=(P.unit==='$'?'BTC Price:':'BTC Ratio:');
    const pVal=percentAtPoint(P, P.x_main[idx], P.y_main[idx]);
    elP.textContent=`(p≈${pVal.toFixed(1)}%)`; setTitleForP(pVal);
  }
  Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});
}

// Click inserts small label above midline and locks panel
plotDiv.on('plotly_click', ev=>{
  if(!(ev.points && ev.points.length) || locked) return;
  const xYears=ev.points[0].x; const P=PRECOMP[denomSel.value];
  const yMid=interp(P.x_grid, seriesForPercent(P,50), xYears); const yAbove=yMid*1.2;
  const text=shortDateFromYears(xYears);
  const ann={x:xYears,y:yAbove,xref:'x',yref:'y',text,showarrow:true,arrowhead:2,ax:0,ay:-20,bgcolor:'rgba(255,255,255,0.95)',bordercolor:'#94a3b8',font:{size:12}};
  Plotly.relayout(plotDiv,{annotations:[ann]}); updatePanel(P,xYears);
});

// Hover drives panel when unlocked
plotDiv.on('plotly_hover', ev=>{
  if(!(ev.points && ev.points.length) || locked) return;
  updatePanel(PRECOMP[denomSel.value], ev.points[0].x);
});

// Date buttons
btnSet.onclick=()=>{ if(!datePick.value) return; locked=true; lockedX=yearsFromISO(datePick.value); updatePanel(PRECOMP[denomSel.value], lockedX); };
btnToday.onclick=()=>{ const P=PRECOMP[denomSel.value]; locked=true; lockedX=P.x_main[P.x_main.length-1]; updatePanel(P, lockedX); };

// Copy chart
btnCopy.onclick=async ()=>{ try{ const url=await Plotly.toImage(plotDiv,{format:'png',scale:2}); if(navigator.clipboard && window.ClipboardItem){ const blob=await (await fetch(url)).blob(); await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]); } else { const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png'; document.body.appendChild(a); a.click(); a.remove(); } }catch(e){ console.error('Copy Chart failed:', e); alert('Copy failed.'); } };

// Denominator change
denomSel.onchange=()=>{ const key=denomSel.value, P=PRECOMP[key];
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main]},                 [IDX_CLICK]);
  Plotly.relayout(plotDiv,{annotations:[]}); renderRails(P); applyIndicatorMask(P);
  const x=locked?lockedX:P.x_main[P.x_main.length-1]; updatePanel(P,x);
};

// Sync all
function syncAll(){ sortRails(); rebuildEditor(); const P=PRECOMP[denomSel.value]; renderRails(P); applyIndicatorMask(P); const x=locked?lockedX:P.x_main[P.x_main.length-1]; updatePanel(P,x); }

// Init
(function init(){
  rebuildReadoutRows(); rebuildEditor();
  renderRails(PRECOMP['USD']); applyIndicatorMask(PRECOMP['USD']);
  const P=PRECOMP['USD']; const tt="%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>";
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main], hovertemplate:[tt], line:[{width:0}]},[IDX_TT]);
  updatePanel(P, P.x_main[P.x_main.length-1]);
})();

// ─────────────────────── Halvings toggle ───────────────────────
let halvingsOn=false;
function makeHalvingShapes(){
  const shapes=[];
  function lineAt(xYears, dashed){
    return {type:'line', xref:'x', yref:'paper', x0:xYears, x1:xYears, y0:0, y1:1,
            line:{color:'#9CA3AF', width:1.2, dash:(dashed?'dash':'solid')}, layer:'below', meta:'halving'};
  }
  PAST_HALVINGS.forEach(iso=>{ shapes.push(lineAt(yearsFromISO(iso), false)); });
  FUTURE_HALVINGS.forEach(iso=>{ shapes.push(lineAt(yearsFromISO(iso), true)); });
  return shapes;
}
btnHalvings.onclick=()=>{ halvingsOn=!halvingsOn;
  const curr=plotDiv.layout.shapes||[];
  if(halvingsOn){
    Plotly.relayout(plotDiv, {shapes:[].concat(curr, makeHalvingShapes())});
    btnHalvings.textContent='Halvings ✓';
  }else{
    const remain=(plotDiv.layout.shapes||[]).filter(s=>s.meta!=='halving');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnHalvings.textContent='Halvings';
  }
};

// ─────────────────────── Liquidity toggle (65-month peaks) ───────────────────────
let liquidityOn = false;

function addMonthsISO(iso, months){
  const d  = new Date(iso + 'T00:00:00Z');
  const y0 = d.getUTCFullYear();
  const m0 = d.getUTCMonth();
  const day= d.getUTCDate();
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

  // build peak sequence (red) back and forward from anchor
  let peaks = [];
  let iso = LIQ_PEAK_ANCHOR_ISO;
  while (compareISO(iso, startIso) > 0){
    peaks.push(iso);
    iso = addMonthsISO(iso, -LIQ_PERIOD_MONTHS);
  }
  peaks = peaks.reverse();
  iso = addMonthsISO(peaks[peaks.length-1], LIQ_PERIOD_MONTHS);
  while (compareISO(iso, endIso) <= 0){
    peaks.push(iso);
    iso = addMonthsISO(iso, LIQ_PERIOD_MONTHS);
  }

  // troughs (green) half-cycle later
  const half = Math.round(LIQ_PERIOD_MONTHS/2);
  const troughs = peaks.map(p => addMonthsISO(p, half));

  function vline(iso, color){
    const dashed = compareISO(iso, lastIso) > 0;
    return {
      type:'line', xref:'x', yref:'paper',
      x0: yearsFromISO(iso), x1: yearsFromISO(iso), y0:0, y1:1,
      line:{color:color, width:1.4, dash: dashed ? 'dash' : 'solid'},
      layer:'below', meta:'liquidity'
    };
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
    btnLiquidity.textContent = 'Liquidity ✓';
  } else {
    const remain = (plotDiv.layout.shapes||[]).filter(s => s.meta!=='liquidity');
    Plotly.relayout(plotDiv, {shapes:remain});
    btnLiquidity.textContent = 'Liquidity';
  }
};
</script>
</body></html>
"""

# ───────────────────────── Fill placeholders ─────────────────────────
HTML = (HTML
    .replace("__PLOT_HTML__", plot_html)
    .replace("__PRECOMP__", json.dumps(PRECOMP))
    .replace("__GENESIS__", GENESIS_DATE.strftime("%Y-%m-%d"))
    .replace("__END_ISO__", END_PROJ.strftime("%Y-%m-%d"))
    .replace("__LAST_PRICE_ISO__", LAST_PRICE_ISO)
    .replace("__MAX_RAIL_SLOTS__", str(MAX_RAIL_SLOTS))
    .replace("__IDX_MAIN__",  str(IDX_MAIN))
    .replace("__IDX_CLICK__", str(IDX_CLICK))
    .replace("__IDX_CURSR__", str(IDX_CURSR))
    .replace("__IDX_TT__",    str(IDX_TT))
    .replace("__EPS_LOG_SPACING__", str(EPS_LOG_SPACING))
    .replace("__PAST_HALVINGS__", json.dumps(PAST_HALVINGS))
    .replace("__FUTURE_HALVINGS__", json.dumps(FUTURE_HALVINGS))
    .replace("__LIQ_PEAK_ANCHOR_ISO__", LIQ_PEAK_ANCHOR_ISO)
    .replace("__LIQ_PERIOD_MONTHS__", str(LIQ_PERIOD_MONTHS))
    .replace("__LIQ_START_ISO__", LIQ_START_ISO)
)

# ───────────────────────────── Write site ─────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print("Wrote", OUTPUT_HTML)
