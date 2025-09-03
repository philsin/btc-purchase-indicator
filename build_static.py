#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — dynamic percentile rails with editable dashboard.

What's new:
- Hover/click drives the readout unless locked by Today / Set Date.
- Denominators: USD, GOLD (XAUUSD), SPX (^SPX), ETH (ETHUSD).
- 50% line is dashed (same style as others).
- Indicator filter dropdown to show only periods matching a band (SELL THE HOUSE…Top Inbound).
- Tap/click places a small two-line annotation above the BTC line.
- Copy Chart grabs the CURRENT zoom/pan via Plotly.toImage() (clipboard + fallback).
- No f-strings in HTML (placeholders replaced after).
"""

import os, io, glob, json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
BTC_FILE     = os.path.join(DATA_DIR, "btc_usd.csv")
OUTPUT_HTML  = "docs/index.html"

GENESIS_DATE = datetime(2009, 1, 3)
END_PROJ     = datetime(2040, 12, 31)
X_START_DATE = datetime(2011, 1, 1)

RESID_WINSOR     = 0.02
EPS_LOG_SPACING  = 0.010
COL_BTC          = "#000000"

# Auto-fetch denominators if not present (Stooq daily CSVs)
AUTO_DENOMS = {
    "GOLD": {"path": os.path.join(DATA_DIR, "denominator_gold.csv"),
             "url":  "https://stooq.com/q/d/l/?s=xauusd&i=d", "parser":"stooq"},
    "SPX":  {"path": os.path.join(DATA_DIR, "denominator_spx.csv"),
             "url":  "https://stooq.com/q/d/l/?s=%5Espx&i=d", "parser":"stooq"},
    "ETH":  {"path": os.path.join(DATA_DIR, "denominator_eth.csv"),
             "url":  "https://stooq.com/q/d/l/?s=ethusd&i=d", "parser":"stooq"},
}

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
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
    r = requests.get(url, timeout=30); r.raise_for_status()
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
    r = requests.get(url, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Date" not in df.columns or "Close" not in df.columns or df.empty:
        raise ValueError("stooq returned no data")
    out = df[["Date","Close"]].rename(columns={"Date":"date","Close":"price"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.sort_values("date").dropna()

def ensure_auto_denominators():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, info in AUTO_DENOMS.items():
        if os.path.exists(info["path"]): continue
        try:
            df = _fetch_stooq_csv(info["url"]) if info["parser"]=="stooq" else None
            if df is not None: df.to_csv(info["path"], index=False)
            print(f"[auto-denom] wrote {info['path']}")
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

# ──────────────────────────────────────────────────────────────────────────────
# Fitting / rails support
# ──────────────────────────────────────────────────────────────────────────────
def quantile_fit_loglog(x_years, y_vals, q=0.5):
    x_years = np.asarray(x_years); y_vals = np.asarray(y_vals)
    mask = np.isfinite(x_years) & np.isfinite(y_vals) & (x_years>0) & (y_vals>0)
    xlog = np.log10(x_years[mask]); ylog = np.log10(y_vals[mask])
    X = pd.DataFrame({"const":1.0,"logx":xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = ylog - (a0 + b*xlog)
    return a0, b, resid

def build_support_for_dynamic_rails(x_years, y_vals):
    a0, b, resid = quantile_fit_loglog(x_years, y_vals, q=0.5)
    r = np.copy(resid)
    if RESID_WINSOR:
        lo, hi = np.nanquantile(r, RESID_WINSOR), np.nanquantile(r, 1-RESID_WINSOR)
        r = np.clip(r, lo, hi)
    med = float(np.nanmedian(r))
    q_grid = np.linspace(0.001, 0.999, 999)
    rq = np.quantile(r, q_grid)
    off_grid = rq - med
    off_grid[0]  -= EPS_LOG_SPACING
    off_grid[-1] += EPS_LOG_SPACING
    return {"a0":a0, "b":b,
            "q_grid":[float(q) for q in q_grid],
            "off_grid":[float(v) for v in off_grid]}

# ──────────────────────────────────────────────────────────────────────────────
# Axis ticks
# ──────────────────────────────────────────────────────────────────────────────
def year_ticks_log(first_dt, last_dt):
    vals, labs = [], []
    for y in range(first_dt.year, last_dt.year+1):
        d = datetime(y,1,1)
        if d < first_dt or d > last_dt: continue
        vy = float(years_since_genesis(pd.Series([d])).iloc[0])
        if vy <= 0: continue
        if y > 2026 and (y % 2 == 1):  # hide odd after 2026
            continue
        vals.append(vy); labs.append(str(y))
    return vals, labs

def y_ticks():
    vals = [10**e for e in range(0,9)]
    labs = [f"{int(10**e):,}" for e in range(0,9)]
    return vals, labs

# ──────────────────────────────────────────────────────────────────────────────
# Build model
# ──────────────────────────────────────────────────────────────────────────────
btc = fetch_btc_csv().rename(columns={"price":"btc"})
denoms = load_denominators()

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
    support = build_support_for_dynamic_rails(xs, ys)
    return {
        "label": label, "unit": unit, "decimals": decimals,
        "x_main": xs.tolist(), "y_main": ys.tolist(),
        "date_iso_main": dates.tolist(), "x_grid": x_grid.tolist(),
        "support": support
    }

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)
P0 = PRECOMP["USD"]

# ──────────────────────────────────────────────────────────────────────────────
# Base figure (rails + main + click catcher + cursor)
# ──────────────────────────────────────────────────────────────────────────────
MAX_RAIL_SLOTS = 12
IDX_MAIN  = MAX_RAIL_SLOTS
IDX_CLICK = MAX_RAIL_SLOTS + 1
IDX_CURSR = MAX_RAIL_SLOTS + 2  # invisible hover cursor

def add_stub(idx):
    return go.Scatter(x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
                      name=f"Rail {idx+1}", line=dict(width=1.6, color="#999", dash="dot"),
                      visible=False, hoverinfo="skip")

traces = [add_stub(i) for i in range(MAX_RAIL_SLOTS)]
traces += [
    # visible BTC line
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    # wide transparent click-catcher overlay (for taps)
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="_click", showlegend=False, hoverinfo="skip",
               line=dict(width=18, color="rgba(0,0,0,0.001)")),
    # ultra-thin almost invisible cursor trace to make hover work everywhere
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="_cursor", showlegend=False, hoverinfo="x",
               line=dict(width=0.1, color="rgba(0,0,0,0.001)")),
]

fig = go.Figure(traces)
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    title="BTC Purchase Indicator — ",
    xaxis=dict(type="log", title=None, tickmode="array",
               tickvals=xtickvals, ticktext=xticktext,
               range=[np.log10(x_start), np.log10(x_end)], showspikes=False),
    yaxis=dict(type="log", title=P0["label"],
               tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),
)

plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive":True,"displayModeBar":True,"modeBarButtonsToRemove":["toImage"]})

# ──────────────────────────────────────────────────────────────────────────────
# HTML (placeholders; NO f-strings)
# ──────────────────────────────────────────────────────────────────────────────
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
select,button,input[type=date],input[type=number]{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}
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
</style>
</head><body>
<div id="capture" class="layout">
  <div class="left" id="leftCol">
    __PLOT_HTML__
  </div>
  <div class="right">
    <div id="controls">
      <label for="denomSel"><b>Denominator:</b></label>
      <select id="denomSel"></select>
      <input type="date" id="datePick"/>
      <button id="setDateBtn">Set Date</button>
      <button id="todayBtn">Today</button>

      <label for="indicatorSel"><b>Indicator:</b></label>
      <select id="indicatorSel">
        <option value="ALL">All</option>
        <option value="SELL THE HOUSE">SELL THE HOUSE</option>
        <option value="Strong Buy">Strong Buy</option>
        <option value="Buy">Buy</option>
        <option value="DCA">DCA</option>
        <option value="Hold On">Hold On</option>
        <option value="Frothy">Frothy</option>
        <option value="Top Inbound">Top Inbound</option>
      </select>

      <button id="copyBtn">Copy Chart</button>
    </div>

    <div id="chartWidthBox">
      <b>Chart Width (px):</b>
      <input type="number" id="chartWpx" min="400" max="2400" step="10" value="1100"/>
      <button id="fitBtn" title="Make chart fill remaining space">Fit</button>
    </div>

    <fieldset id="railsBox">
      <legend>Rails</legend>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
        <button id="editBtn">Edit Rails</button>
        <span class="smallnote">Add/remove/change percents. Sorted automatically (high→low). 50% is dashed.</span>
      </div>
      <div id="railsView" class="smallnote">Current: <span id="railsListText"></span></div>

      <div id="railsEditor" class="hidden">
        <div id="railItems"></div>
        <div class="rail-row" style="margin-top:6px;">
          <input type="number" id="addPct" placeholder="Add % (e.g. 92.5)" step="0.1" min="0.1" max="99.9"/>
          <button id="addBtn">Add</button>
        </div>
      </div>
    </fieldset>

    <div style="font-size:12px;color:#6b7280;">Detected denominators: <span id="denomsDetected"></span></div>

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
const MAX_SLOTS = __MAX_RAIL_SLOTS__;
const IDX_MAIN  = __IDX_MAIN__;
const IDX_CLICK = __IDX_CLICK__;
const IDX_CURSR = __IDX_CURSR__;
const EPS_LOG_SPACING = __EPS_LOG_SPACING__;

const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }
function shortDateFromYears(y){ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function interp(xs, ys, x){ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }

// Units
function fmtVal(P, v){
  if(!isFinite(v)) return '—';
  const dec = Math.max(0, Math.min(10, P.decimals||2));
  if (P.unit === '$') {
    return '$'+Number(v).toLocaleString(undefined, {minimumFractionDigits: dec, maximumFractionDigits: dec});
  } else {
    const maxd = Math.max(dec, 6);
    return Number(v).toLocaleString(undefined, {minimumFractionDigits: Math.min(6, maxd), maximumFractionDigits: maxd});
  }
}

// Color + indicator mapping
function colorForPercent(p){ const t=Math.max(0,Math.min(1,p/100));
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

// DOM
const leftCol=document.getElementById('leftCol');
const plotDiv=document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div');
const denomSel=document.getElementById('denomSel');
const indicatorSel=document.getElementById('indicatorSel');
const datePick=document.getElementById('datePick');
const btnSet=document.getElementById('setDateBtn');
const btnToday=document.getElementById('todayBtn');
const btnCopy=document.getElementById('copyBtn');
const btnFit=document.getElementById('fitBtn');
const elDenoms=document.getElementById('denomsDetected');
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
const addBtn=document.getElementById('addBtn');

// Denominator dropdown (USD, GOLD, SPX, ETH)
const denomKeys = Object.keys(PRECOMP);
['USD', ...denomKeys.filter(k=>k!=='USD')].forEach(k=>{ const o=document.createElement('option'); o.value=k; o.textContent=k; denomSel.appendChild(o); });
document.getElementById('denomsDetected').textContent = denomKeys.filter(k=>k!=='USD').join(', ') || '(none)';

// Rails state (50% stays dashed like others)
let rails = [97.5, 90, 75, 50, 25, 2.5];
function sortRails(){ rails = rails.filter(p=>isFinite(p)).map(Number).map(p=>Math.max(0.1,Math.min(99.9,p))).filter((p,i,a)=>a.indexOf(p)===i).sort((a,b)=>b-a); }
function railsText(){ return rails.map(p=>String(p).replace(/\.0$/,'')+'%').join(', '); }
function idFor(p){ return 'v'+String(p).replace('.','_'); }

function rebuildReadoutRows(){
  elRows.innerHTML='';
  rails.forEach(p=>{
    const row=document.createElement('div'); row.className='row';
    const lab=document.createElement('div');
    const val=document.createElement('div'); val.className='num'; val.id=idFor(p);
    const color=colorForPercent(p);
    const name=Math.abs(p-2.5)<1e-9?'Floor':Math.abs(p-97.5)<1e-9?'Ceiling':(p+'%');
    lab.innerHTML=`<span style="color:${color};">${name}</span>`;
    row.appendChild(lab); row.appendChild(val); row.appendChild(document.createElement('div'));
    elRows.appendChild(row);
  });
}
function rebuildEditor(){
  railItems.innerHTML='';
  rails.forEach((p,idx)=>{
    const row=document.createElement('div'); row.className='rail-row';
    const color=colorForPercent(p);
    const labelTxt=Math.abs(p-2.5)<1e-9?'Floor':Math.abs(p-97.5)<1e-9?'Ceiling':(p+'%');
    const lab=document.createElement('span'); lab.style.minWidth='48px'; lab.style.color=color; lab.textContent=labelTxt;
    const inp=document.createElement('input'); inp.type='number'; inp.step='0.1'; inp.min='0.1'; inp.max='99.9'; inp.value=String(p);
    const rm=document.createElement('button'); rm.textContent='Remove';
    inp.addEventListener('change',()=>{ const v=Number(inp.value); rails[idx]=isFinite(v)?v:p; sortRails(); syncAll(); });
    rm.addEventListener('click',()=>{ rails.splice(idx,1); syncAll(); });
    row.appendChild(lab); row.appendChild(inp); row.appendChild(rm); railItems.appendChild(row);
  });
  railsListText.textContent=railsText();
}
addBtn.addEventListener('click',()=>{ const v=Number(addPct.value); if(!isFinite(v)) return; rails.push(v); addPct.value=''; sortRails(); syncAll(); });
editBtn.addEventListener('click',()=>{ railsEditor.classList.toggle('hidden'); railsView.classList.toggle('hidden'); editBtn.textContent = railsEditor.classList.contains('hidden') ? 'Edit Rails' : 'Done'; });

// Sizing
function applyChartWidthPx(px){
  const v=Math.max(400, Math.min(2400, Number(px)||1100));
  leftCol.style.flex='0 0 auto'; leftCol.style.width=v+'px';
  if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv);
}
chartWpx.addEventListener('change',()=>applyChartWidthPx(chartWpx.value));
btnFit.addEventListener('click',()=>{
  const total=document.documentElement.clientWidth||window.innerWidth, panel=420, pad=32;
  const target=Math.max(400, total-panel-pad); chartWpx.value=target; applyChartWidthPx(target);
});
if(window.ResizeObserver) new ResizeObserver(()=>{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }).observe(leftCol);

// Rails math
function logMidline(P){ const d=P.support; return P.x_grid.map(x=> (d.a0 + d.b*Math.log10(x)) ); }
function offsetForPercent(P, percent){
  const d=P.support; const p01=Math.max(d.q_grid[0], Math.min(d.q_grid[d.q_grid.length-1], percent/100));
  return interp(d.q_grid, d.off_grid, p01);
}
function seriesForPercent(P, percent){
  const logM=logMidline(P), off=offsetForPercent(P,percent);
  const eps=(percent>=97.5||percent<=2.5)?EPS_LOG_SPACING:0.0;
  return logM.map(v=>Math.pow(10, v+off+(percent>=50?eps:-eps)));
}
function percentFromOffset(P, off){
  const d=P.support;
  const q=Math.max(d.q_grid[0], Math.min(d.q_grid[d.q_grid.length-1], interp(d.off_grid, d.q_grid, off)));
  return q;
}

// Render rails (50% dashed)
function renderRails(P){
  const n=Math.min(rails.length, MAX_SLOTS);
  for(let i=0;i<MAX_SLOTS;i++){
    const visible=(i<n); let restyle={visible};
    if(visible){
      const p=rails[i], color=colorForPercent(p), dash='dot', width=1.6;
      restyle=Object.assign(restyle,{
        x:[P.x_grid], y:[seriesForPercent(P,p)],
        name:(Math.abs(p-2.5)<1e-9?'Floor':(Math.abs(p-97.5)<1e-9?'Ceiling':(p+'%'))),
        line:{color,width,dash}
      });
    }
    Plotly.restyle(plotDiv, restyle, [i]);
  }
  railsListText.textContent=railsText();
  rebuildReadoutRows();
}

let locked=false, lockedX=null;

function setTitleForP(p){ Plotly.relayout(plotDiv, {'title.text': 'BTC Purchase Indicator — '+indicatorFromP(p)}); }

function computePSeries(P){
  const d=P.support;
  const out = new Array(P.x_main.length);
  for (let i=0;i<P.x_main.length;i++){
    const logx=Math.log10(P.x_main[i]);
    const ly=Math.log10(P.y_main[i]);
    const mid=d.a0 + d.b*logx;
    const off=ly-mid;
    const q = percentFromOffset(P, off);
    out[i] = Math.max(0, Math.min(100, 100*q));
  }
  P._pSeries = out;
}

function applyIndicatorMask(P){
  const mode = indicatorSel.value || 'ALL';
  if (mode === 'ALL'){
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
    Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]},                [IDX_CLICK]);
    return;
  }
  if (!P._pSeries) computePSeries(P);
  const [lo,hi] = rangeForIndicator(mode);
  const ym = P.y_main.map((v,i)=> (P._pSeries[i]>=lo && P._pSeries[i]<hi) ? v : null);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym], name:[P.label+' — '+mode]}, [IDX_MAIN]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[ym]},                            [IDX_CLICK]);
}

function updatePanel(P,xYears){
  elDate.textContent=shortDateFromYears(xYears);
  rails.forEach(p=>{
    const v=interp(P.x_grid, seriesForPercent(P,p), xYears);
    const el=document.getElementById(idFor(p)); if(el) el.textContent=fmtVal(P, v);
  });
  // nearest main point (unfiltered, so readouts always available)
  let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx];
  elMain.textContent=fmtVal(P,y);
  elMainLabel.textContent=(P.unit==='$'?'BTC Price:':'BTC Ratio:');

  const d=P.support, logx=Math.log10(xYears), ly=Math.log10(y), mid=d.a0+d.b*logx, off=ly-mid;
  const pVal=Math.max(0, Math.min(100, 100*percentFromOffset(P, off)));
  elP.textContent=`(p≈${pVal.toFixed(1)}%)`; setTitleForP(pVal);

  Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});
}

// Click annotation (above line)
plotDiv.on('plotly_click', ev=>{
  if(!(ev.points && ev.points.length)) return;
  if (locked) return; // ignore clicks when locked to Today/Set Date
  const xYears = ev.points[0].x;
  const P = PRECOMP[denomSel.value];
  let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx], yAbove=y*1.2;
  const text=shortDateFromYears(xYears)+"<br>"+fmtVal(P,y);
  const ann = { x:xYears, y:yAbove, xref:'x', yref:'y', text, showarrow:true, arrowhead:2, ax:0, ay:-20,
                bgcolor:'rgba(255,255,255,0.95)', bordercolor:'#94a3b8', font:{size:12}};
  Plotly.relayout(plotDiv, {annotations:[ann]});
  updatePanel(P, xYears);
});

// Hover drives panel when unlocked
plotDiv.on('plotly_hover', ev=>{
  if(!(ev.points && ev.points.length)) return;
  if (locked) return;
  updatePanel(PRECOMP[denomSel.value], ev.points[0].x);
});

// Date buttons
btnSet.onclick = ()=>{ if(!datePick.value) return; locked=true; lockedX=yearsFromISO(datePick.value); updatePanel(PRECOMP[denomSel.value], lockedX); };
btnToday.onclick = ()=>{ const P = PRECOMP[denomSel.value]; locked=true; lockedX=P.x_main[P.x_main.length-1]; updatePanel(P, lockedX); };

// Copy current view
btnCopy.onclick = async ()=>{
  try{
    const url = await Plotly.toImage(plotDiv, {format:'png', scale:2});
    try{
      if(navigator.clipboard && window.ClipboardItem){
        const blob = await (await fetch(url)).blob();
        await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
        return;
      }
    }catch(e){}
    const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png'; document.body.appendChild(a); a.click(); a.remove();
  }catch(e){ console.error(e); }
};

// Denominator change
denomSel.onchange = ()=>{
  const key=denomSel.value, P=PRECOMP[key];
  // main, click-catcher, cursor (cursor always full series to keep hover working across timeline)
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label]}, [IDX_MAIN]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]},                [IDX_CLICK]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]},                [IDX_CURSR]);
  Plotly.relayout(plotDiv, {annotations: []});
  renderRails(P);
  applyIndicatorMask(P);
  const x = locked ? lockedX : P.x_main[P.x_main.length-1];
  updatePanel(P, x);
};

// Indicator filter change
indicatorSel.onchange = ()=>{
  const P = PRECOMP[denomSel.value];
  applyIndicatorMask(P);
};

// Sync
function syncAll(){
  sortRails(); rebuildEditor();
  const P = PRECOMP[denomSel.value];
  renderRails(P);
  applyIndicatorMask(P);
  const x = locked ? lockedX : P.x_main[P.x_main.length-1];
  updatePanel(P, x);
}

// Init
denomSel.value='USD';
rebuildReadoutRows(); rebuildEditor();
renderRails(PRECOMP['USD']);
applyIndicatorMask(PRECOMP['USD']);
updatePanel(PRECOMP['USD'], PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);
</script>
</body></html>
"""

# Fill placeholders safely
HTML = (HTML
    .replace("__PLOT_HTML__", plot_html)
    .replace("__PRECOMP__", json.dumps(PRECOMP))
    .replace("__GENESIS__", GENESIS_DATE.strftime("%Y-%m-%d"))
    .replace("__MAX_RAIL_SLOTS__", str(MAX_RAIL_SLOTS))
    .replace("__IDX_MAIN__",  str(IDX_MAIN))
    .replace("__IDX_CLICK__", str(IDX_CLICK))
    .replace("__IDX_CURSR__", str(IDX_CURSR))
    .replace("__EPS_LOG_SPACING__", str(EPS_LOG_SPACING))
)

# Write
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print("Wrote", OUTPUT_HTML)
