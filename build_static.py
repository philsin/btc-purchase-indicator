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
X_START_DATE = datetime(2011, 1, 1)

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

# ───────────────────── Helpers / fetchers ─────────────────────
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
    return {"a0":a0, "b":b,
            "q_grid":[float(q) for q in q_grid],
            "off_grid":[float(v) for v in off_grid]}

# ───────────────────────── Build model ─────────────────────────
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

def year_ticks_log(first_dt, last_dt):
    vals, labs = [], []
    for y in range(first_dt.year, last_dt.year+1):
        d = datetime(y,1,1)
        if y > 2020 and (y % 2 == 1):  # hide odd years after 2020
            continue
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

def series_for_denom(df, key):
    if not key or key.lower() in ("usd", "none"):
        return df["btc"], "BTC / USD", "$", 2
    k = key.lower()
    if k in df.columns:
        return df["btc"]/df[k], f"BTC / {key.upper()}", "", 6
    return df["btc"], "BTC / USD", "$", 2

def build_payload(df, denom_key=None):
    y, label, unit, decimals = series_for_denom(df, denom_key)
    mask = np.isfinite(df["x_years"].values) & np.isfinite(y.values)
    xs = df["x_years"].values[mask]
    ys = y.values[mask]
    dates = df["date_iso"].values[mask]
    support = build_support_constant_rails(xs, ys)
    return {
        "label": label, "unit": unit, "decimals": decimals,
        "x_main": xs.tolist(), "y_main": ys.tolist(),
        "date_iso_main": dates.tolist(),
        "x_grid": x_grid.tolist(),
        "support": support
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
.info-btn{width:28px;height:28px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700}
.info-modal{position:fixed;inset:0;background:rgba(0,0,0,0.35);display:none;align-items:center;justify-content:center;z-index:100}
.info-card{background:#fff;border-radius:12px;max-width:680px;width:92%;padding:16px;border:1px solid #e5e7eb;box-shadow:0 12px 28px rgba(0,0,0,0.18)}
.info-card h3{margin:0 0 8px 0}
.info-card p{margin:6px 0}
.info-modal.open{display:flex}
@media (max-width:900px){
  .layout{flex-direction:column}
  .right{flex:0 0 auto;border-left:none;border-top:1px solid #e5e7eb}
  .left{flex:0 0 auto;width:100%;padding:8px}
}
#chartWidthBox{display:flex;align-items:center;gap:8px}
#chartWpx{width:120px}
#levelsBox{display:flex;align-items:center;gap:8px}
#levelsBox input[type=text]{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;min-width:150px}
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

      <button id="infoBtn" class="btn info-btn" title="How it works">i</button>
    </div>

    <div id="chartWidthBox">
      <b>Chart Width (px):</b>
      <input type="number" id="chartWpx" min="400" max="2400" step="10" value="1100"/>
      <button id="fitBtn" class="btn" title="Make chart fill remaining space">Fit</button>
    </div>

    <!-- Levels: user-defined horizontal lines -->
    <div id="levelsBox">
      <b>Level:</b>
      <input type="text" id="levelInput" placeholder="e.g. 50000 or 1.2"/>
      <button id="addLevelBtn" class="btn">Add Level</button>
      <button id="clearLevelsBtn" class="btn">Clear</button>
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
      <div><b>Composite:</b> <span id="compLine" class="smallnote">—</span></div>
    </div>
  </div>
</div>

<!-- Info modal -->
<div id="infoModal" class="info-modal" role="dialog" aria-modal="true">
  <div class="info-card">
    <h3>How the Indicator Works</h3>
    <p><b>Rails:</b> We fit a power-law (log-log) median to BTC and measure residuals. Fixed quantiles of residuals become % rails (Floor≈2.5% … Ceiling≈97.5%).</p>
    <p><b>p-value:</b> Your position within Floor⇢Ceiling on a log scale (0–100%).</p>
    <p><b>Composite score:</b> We blend (a) centered p, (b) the 65-month liquidity wave (rising=tailwind, falling=headwind), and (c) the halving window (+1 pre-halving year, +0.5 in first ~18 months after, −0.5 late). No cross-denominator voting.</p>
    <p><b>Labels:</b> The title (SELL THE HOUSE → Top Inbound) comes from the composite score. The Indicator filter highlights dates that matched those labels historically.</p>
    <div style="display:flex;justify-content:flex-end;margin-top:10px;">
      <button id="infoClose" class="btn">Close</button>
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

const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }
function shortDateFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function isoFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`; }
function interp(xs, ys, x){ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function daysBetweenISO(a,b){ return Math.round((new Date(b+'T00:00:00Z') - new Date(a+'T00:00:00Z'))/86400000); }
function monthsBetweenISO(a,b){ return (new Date(b+'T00:00:00Z') - new Date(a+'T00:00:00Z'))/(86400000*30.4375); }

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
  if (s >  -1.50) return 'Frothy';   # extended to −1.50 per your request
  return 'Top Inbound';
}

// DOM
const leftCol=document.getElementById('leftCol');
const rightCol=document.querySelector('.right');
const plotDiv=(function(){ return document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div'); })();
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

const elDate=document.querySelector('#readout .date');
const elRows=document.getElementById('readoutRows');
const elMain=document.getElementById('mainVal');
const elMainLabel=document.getElementById('mainLabel');
const elP=document.getElementById('pPct');
const elComp=document.getElementById('compLine');
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
indicatorApply.addEventListener('click',e=>{e.preventDefault(); indicatorMenu.classList.remove('open'); selectedIndicators=getCheckedIndicators(); indicatorBtn.textContent=labelForIndicatorBtn(selectedIndicators); applyIndicatorMask(PRECOMP[denomSel.value]);});
document.addEventListener('click',e=>{ if(!indicatorMenu.contains(e.target) && !indicatorBtn.contains(e.target)) indicatorMenu.classList.remove('open'); });

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
  // iOS Safari can require two-step resize to reflow canvas properly
  if(window.Plotly&&plotDiv){
    Plotly.Plots.resize(plotDiv);
    requestAnimationFrame(()=>Plotly.Plots.resize(plotDiv));
  }
}
chartWpx.addEventListener('change',()=>applyChartWidthPx(chartWpx.value));

// Robust "Fit" that measures the right panel (fixes mobile Safari)
btnFit.addEventListener('click',()=>{
  const totalW = document.documentElement.clientWidth || window.innerWidth || screen.width || 1200;
  const rightW = (rightCol && rightCol.getBoundingClientRect ? rightCol.getBoundingClientRect().width : 420);
  const padding = 16; // small gutter
  const target = Math.max(400, Math.floor(totalW - rightW - padding));
  chartWpx.value = target;
  applyChartWidthPx(target);
});

if(window.ResizeObserver){
  // keep plot snug on orientation/keyboard changes on iOS
  new ResizeObserver(()=>{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }).observe(leftCol);
}

// ───────── Rails math ─────────
function logMidline(P){ const d=P.support; return P.x_grid.map(x=> (d.a0 + d.b*Math.log10(x)) ); }
function offsetForPercent(P, percent){
  const d=P.support; const q=d.q_grid, off=d.off_grid;
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
  const d=P.support; const q=d.q_grid, offg=d.off_grid;
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
function titleForScore(s){ return 'BTC Purchase Indicator — ' + classFromScore(s); }
function setTitle(s){ Plotly.relayout(plotDiv, {'title.text': titleForScore(s)}); }

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
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[seriesForPercent(P,50)]}, [IDX_CARRY]);
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
  if (!selectedIndicators || selectedIndicators.length===0){
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]},                [IDX_CLICK]);
  } else {
    computeSeriesForMask(P);
    const set=new Set(selectedIndicators);
    const ym = P.y_main.map((v,i)=> set.has(P._classSeries[i]) ? v : null);
    const nm = P.label+' — '+Array.from(set).join(' + ');
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym], name:[nm]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym]},            [IDX_CLICK]);
  }
  const tt = (P.unit === '$') ? "%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>" : "%{x|%b-%d-%Y}<br>%{y:.6f}<extra></extra>";
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], hovertemplate:[tt], line:[{width:0}]}, [IDX_TT]);
}

// Panel update (future uses 50% for main value; p hidden)
function updatePanel(P,xYears){
  const iso = isoFromYears(xYears);
  elDate.textContent=shortDateFromYears(xYears);

  const v50series = seriesForPercent(P,50);
  const v50 = interp(P.x_grid, v50series, xYears);

  rails.forEach(p=>{
    const v=interp(P.x_grid, seriesForPercent(P,p), xYears);
    const el=document.getElementById(idFor(p)); if(!el) return;
    const mult=(v50>0&&isFinite(v50))?` (${(v/v50).toFixed(1)}x)`:''; el.textContent=fmtVal(P,v)+mult;
  });

  const lastX=P.x_main[P.x_main.length-1];
  let usedP=null, mainTxt='', compScore=null;

  if (xYears>lastX){
    mainTxt = fmtVal(P, v50) + " (50%)";
    elP.textContent = "(p≈—)";
    elMainLabel.textContent=(P.unit==='$'?'BTC Price:':'BTC Ratio:');
    usedP = 50; // neutral p for future composite
  } else {
    let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
    const y=P.y_main[idx]; mainTxt = fmtVal(P,y); elMainLabel.textContent=(P.unit==='$'?'BTC Price:':'BTC Ratio:');
    const d=P.support, z=Math.log10(P.x_main[idx]), off=Math.log10(y)-(d.a0+d.b*z);
    usedP=clamp(percentFromOffset(P, off), 0, 100);
    elP.textContent=`(p≈${usedP.toFixed(1)}%)`;
  }
  elMain.textContent = mainTxt;

  compScore = compositeFrom(usedP, iso);
  elComp.textContent = `${compScore.toFixed(2)} — ${classFromScore(compScore)}`;
  setTitle(compScore);
}

// NOTE: single-click should do NOTHING → no plotly_click handler

// Hover drives panel everywhere (carrier trace active in future, too)
plotDiv.on('plotly_hover', ev=>{
  if(!(ev.points && ev.points.length)) return;
  updatePanel(PRECOMP[denomSel.value], ev.points[0].x);
});

// Date lock
document.getElementById('setDateBtn').onclick=()=>{ if(!datePick.value) return; updatePanel(PRECOMP[denomSel.value], yearsFromISO(datePick.value)); };
document.getElementById('todayBtn').onclick=()=>{ const P=PRECOMP[denomSel.value]; updatePanel(P, P.x_main[P.x_main.length-1]); };

// Copy chart
btnCopy.onclick=async ()=>{ try{ const url=await Plotly.toImage(plotDiv,{format:'png',scale:2}); if(navigator.clipboard && window.ClipboardItem){ const blob=await (await fetch(url)).blob(); await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]); } else { const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png'; document.body.appendChild(a); a.click(); a.remove(); } }catch(e){ console.error('Copy Chart failed:', e); alert('Copy failed.'); } };

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
denomSel.onchange=()=>{ const key=denomSel.value, P=PRECOMP[key];
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main]},                 [IDX_CLICK]);
  Plotly.restyle(plotDiv,{x:[P.x_grid], y:[seriesForPercent(P,50)]},   [IDX_CARRY]);
  Plotly.relayout(plotDiv,{annotations:(plotDiv.layout.annotations||[]).filter(a=>a.meta!=='halving' && a.meta!=='liquidity')});
  renderRails(P); applyIndicatorMask(P);
  redrawLevels(P);
  updatePanel(P, P.x_main[P.x_main.length-1]);
};

// Sync all
function syncAll(){ sortRails(); rebuildEditor(); const P=PRECOMP[denomSel.value]; renderRails(P); applyIndicatorMask(P); redrawLevels(P); updatePanel(P, P.x_main[P.x_main.length-1]); }

// Init
(function init(){
  rebuildReadoutRows(); rebuildEditor();
  renderRails(PRECOMP['USD']); applyIndicatorMask(PRECOMP['USD']);
  const P=PRECOMP['USD'];
  Plotly.restyle(plotDiv,{x:[P.x_grid], y:[seriesForPercent(P,50)]},[IDX_CARRY]);
  const tt="%{x|%b-%d-%Y}<br>$%{y:.2f}<extra></extra>";
  Plotly.restyle(plotDiv,{x:[P.x_main], y:[P.y_main], hovertemplate:[tt], line:[{width:0}]},[IDX_TT]);
  updatePanel(P, P.x_main[P.x_main.length-1]);
  redrawLevels(P);
})();

// Halvings toggle
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
    return { type:'line', xref:'x', yref:'paper',
      x0: yearsFromISO(iso), x1: yearsFromISO(iso), y0:0, y1:1,
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
    btnLiquidity.textContent = 'Liquidity ✓';
  } else {
    const remain = (plotDiv.layout.shapes||[]).filter(s => s.meta!=='liquidity');
    Plotly.relayout(plotDiv, {shapes:remain});
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
    .replace("__W_HALV__", str(W_HALV))   # fixed typo here
)

# ───────────────────────────── Write site ─────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print("Wrote", OUTPUT_HTML)
