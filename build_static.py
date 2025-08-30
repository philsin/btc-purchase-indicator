#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — Power-law rails with a robust median midline,
winsorized & symmetric percentile offsets, and a responsive UI with a Rails Dashboard.

Generates docs/index.html from data in ./data:
- data/btc_usd.csv (optional; otherwise fetched from Blockchain.info)
- data/denominator_*.csv (optional denominators; columns: date,price)
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
END_PROJ     = datetime(2040, 12, 31)      # fixed horizon

# Force chart x-axis to start here
X_START_DATE = datetime(2011, 1, 1)

# Rails behaviour
RESID_WINSOR     = 0.02   # clip 2% tails in residuals (robust ceiling/floor)
EPS_LOG_SPACING  = 0.010  # keep extreme rails from touching a bit

# Colors for BTC & panel labels
COL_BTC     = "#000000"
COL_FLOOR   = "#D32F2F"   # used only for "Floor" label in panel (20%)
COL_CEILING = "#2E7D32"   # used only for "Ceiling" label in panel (97.5%)

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def years_since_genesis(dates):
    d = pd.to_datetime(dates)
    delta_days = (d - GENESIS_DATE) / np.timedelta64(1, "D")
    # +1 day so log(x) > 0 at start
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)

def fetch_btc_csv() -> pd.DataFrame:
    """Load BTC (date, price). Use local file if present, else fetch."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(BTC_FILE):
        df = pd.read_csv(BTC_FILE, parse_dates=["date"])
        return df.sort_values("date").dropna()

    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = r.text.strip()
    # header can be present or absent; handle both
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

def load_denominators():
    """Return { KEY: DataFrame(date, price) } for ./data/denominator_*.csv"""
    out={}
    for p in glob.glob(os.path.join(DATA_DIR, "denominator_*.csv")):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            # tolerate arbitrary second column name
            if len(df.columns) < 2:
                continue
            price_col = [c for c in df.columns if c.lower()!="date"][0]
            df = df.rename(columns={price_col:"price"})[["date","price"]]
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.sort_values("date").dropna()
            out[key]=df
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Fitting + compact support for dynamic rails (sent to JS)
# ──────────────────────────────────────────────────────────────────────────────
def quantile_fit_loglog(x_years, y_vals, q=0.5):
    """Quantile regression of log10(y) on log10(x_years)."""
    x_years = np.asarray(x_years); y_vals = np.asarray(y_vals)
    mask = np.isfinite(x_years) & np.isfinite(y_vals) & (x_years>0) & (y_vals>0)
    xlog = np.log10(x_years[mask]); ylog = np.log10(y_vals[mask])
    X = pd.DataFrame({"const":1.0,"logx":xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = ylog - (a0 + b*xlog)
    return a0, b, resid

def build_support_for_dynamic_rails(x_years, y_vals):
    """
    Compute a compact lookup so JS can draw ANY percentile rail smoothly:
    - a0, b: midline params for log10(y) = a0 + b*log10(xyears)
    - med:   median of residuals (may be ~0)
    - q_grid:   0.50..0.995 (100 points)
    - d_grid:   symmetric offset d at each q (max of hi/lo tails)
               such that rails are med ± d
      Then, for a desired percentile p in [0,1], set q = 1 - p,
      offset = med + (p>=0.5 ? +d(q) : -d(q)).
    """
    a0, b, resid = quantile_fit_loglog(x_years, y_vals, q=0.5)
    r = np.copy(resid)
    if RESID_WINSOR:
        lo, hi = np.nanquantile(r, RESID_WINSOR), np.nanquantile(r, 1-RESID_WINSOR)
        r = np.clip(r, lo, hi)

    med = float(np.nanmedian(r))
    pos = r - med; pos = pos[pos>0.0]
    neg = med - r; neg = neg[neg>0.0]
    pos.sort(); neg.sort()

    q_grid = np.linspace(0.50, 0.995, 100)
    d_grid = []
    for q in q_grid:
        d_hi = float(np.quantile(pos, q)) if pos.size else 0.0
        d_lo = float(np.quantile(neg, q)) if neg.size else 0.0
        d = max(d_hi, d_lo)
        d_grid.append(d)

    # Nudge the very top end so ceiling cannot touch midline when extrapolated
    # (applied client-side too, but this keeps intent clear)
    d_grid[-1] += EPS_LOG_SPACING

    return {"a0":a0, "b":b, "med":float(med),
            "q_grid": [float(q) for q in q_grid],
            "d_grid": [float(d) for d in d_grid]}

# ──────────────────────────────────────────────────────────────────────────────
# Axis ticks
# ──────────────────────────────────────────────────────────────────────────────
def year_ticks_log(first_dt, last_dt):
    """Whole-year ticks on log-x (years since genesis). Hide odd labels after 2026."""
    vals, labs = [], []
    y0, y1 = first_dt.year, last_dt.year
    for y in range(y0, y1+1):
        d = datetime(y,1,1)
        if d < first_dt or d > last_dt: continue
        vy = float(years_since_genesis(pd.Series([d])).iloc[0])
        if vy <= 0: continue
        if y > 2026 and (y % 2 == 1):
            continue
        vals.append(vy); labs.append(str(y))
    return vals, labs

def y_ticks():
    vals = [10**e for e in range(0,9)]
    labs = [f"{int(10**e):,}" for e in range(0,9)]
    return vals, labs

# ──────────────────────────────────────────────────────────────────────────────
# Build data model
# ──────────────────────────────────────────────────────────────────────────────
btc = fetch_btc_csv().rename(columns={"price":"btc"})
denoms = load_denominators()

# Merge denominators into base
base = btc.sort_values("date").reset_index(drop=True)
for key, df in denoms.items():
    base = base.merge(df.rename(columns={"price": key.lower()}), on="date", how="left")

base["x_years"]   = years_since_genesis(base["date"])
base["date_iso"]  = base["date"].dt.strftime("%Y-%m-%d")

# Horizon: fixed to END_PROJ (12/31/2040)
first_dt = max(base["date"].iloc[0], X_START_DATE)
max_dt   = END_PROJ

# x_grid for rails (log-spaced in x-years) between first_dt .. max_dt
x_start = float(years_since_genesis(pd.Series([first_dt])).iloc[0])
x_end   = float(years_since_genesis(pd.Series([max_dt])).iloc[0])
x_grid  = np.logspace(np.log10(max(1e-6, x_start)), np.log10(x_end), 700)

# x/y ticks (to horizon)
xtickvals, xticktext = year_ticks_log(first_dt, max_dt)
ytickvals, yticktext = y_ticks()

def series_for_denom(df, key):
    """Return (series, label)."""
    if not key or key.lower() in ("usd", "none"):
        return df["btc"], "BTC / USD"
    k = key.lower()
    if k in df.columns:
        return df["btc"]/df[k], f"BTC / {key.upper()}"
    return df["btc"], "BTC / USD"

def build_payload(df, denom_key=None):
    y, label = series_for_denom(df, denom_key)
    support = build_support_for_dynamic_rails(df["x_years"], y)
    return {
        "label": label,
        "x_main": df["x_years"].tolist(),
        "y_main": y.tolist(),
        "date_iso_main": df["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "support": support
    }

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)

P0 = PRECOMP["USD"]

# ──────────────────────────────────────────────────────────────────────────────
# Base figure (we pre-allocate rail stubs; content filled by JS)
# ──────────────────────────────────────────────────────────────────────────────
MAX_RAIL_SLOTS = 12  # plenty of headroom

def add_stub(idx):
    # start invisible; JS will fill name/color/line and set visible
    return go.Scatter(
        x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
        name=f"Rail {idx+1}", line=dict(width=1.6, color="#999"), visible=False, hoverinfo="skip"
    )

traces = [add_stub(i) for i in range(MAX_RAIL_SLOTS)]
traces += [
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    # transparent cursor trace to keep x-hover alive
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
]

fig = go.Figure(traces)

# Set the default visible range to [first_dt .. 2040-12-31]
x_min = float(years_since_genesis(pd.Series([first_dt])).iloc[0])
x_max = float(years_since_genesis(pd.Series([max_dt])).iloc[0])

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",   # unified hover label (no persistent spike)
    showlegend=True,
    title="BTC Purchase Indicator — Rails",
    xaxis=dict(type="log", title=None, tickmode="array",
               tickvals=xtickvals, ticktext=xticktext,
               range=[np.log10(x_min), np.log10(x_max)], showspikes=False),
    yaxis=dict(type="log", title=P0["label"],
               tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),
)

plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive":True,"displayModeBar":True,"modeBarButtonsToRemove":["toImage"]})

# ──────────────────────────────────────────────────────────────────────────────
# HTML + JS (pixel width + dynamic Rails Dashboard)
# ──────────────────────────────────────────────────────────────────────────────
HTML = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
:root{{--panelW:420px;}}
html,body{{height:100%}} body{{margin:0;font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}}
.layout{{display:flex;min-height:100vh;width:100vw}}
.left{{flex:0 0 auto;width:1100px;min-width:280px;padding:8px 0 8px 8px}}
.left .js-plotly-plot,.left .plotly-graph-div{{width:100%!important}}
.right{{flex:0 0 var(--panelW);border-left:1px solid #e5e7eb;padding:12px;display:flex;flex-direction:column;gap:12px;overflow:auto}}
#controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
select,button,input[type=date],input[type=number],input[type=text]{{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}}
#readout{{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fafafa;font-size:14px}}
#readout .date{{font-weight:700;margin-bottom:6px}}
#readout .row{{display:grid;grid-template-columns:auto 1fr auto;column-gap:8px;align-items:baseline}}
#readout .num{{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;text-align:right;min-width:12ch;white-space:pre}}
.hoverlayer{{opacity:0!important;pointer-events:none}}
fieldset{{border:1px solid #e5e7eb;border-radius:8px;padding:8px 10px}}
legend{{padding:0 6px;color:#374151;font-weight:600;font-size:13px}}
.rail-row{{display:flex;align-items:center;gap:8px;margin:2px 0}}
.rail-row input[type=number]{{width:90px}}
.rail-row button{{padding:4px 8px;font-size:12px}}
.smallnote{{font-size:12px;color:#6b7280}}
@media (max-width:900px){{
  .layout{{flex-direction:column}}
  .right{{flex:0 0 auto;border-left:none;border-top:1px solid #e5e7eb}}
  .left{{flex:0 0 auto;width:100%;padding:8px}}
}}
#chartWidthBox{{display:flex;align-items:center;gap:8px}}
#chartWpx{{width:120px}}
.hidden{{display:none}}
</style>
</head><body>
<div id="capture" class="layout">
  <div class="left" id="leftCol">
    {plot_html}
  </div>
  <div class="right">
    <div id="controls">
      <label for="denomSel"><b>Denominator:</b></label>
      <select id="denomSel"></select>
      <input type="date" id="datePick"/>
      <button id="setDateBtn">Set Date</button>
      <button id="liveBtn">Live Hover</button>
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
        <span class="smallnote">Toggle edit mode to add/remove/change percent lines. Sorted automatically (high→low).</span>
      </div>
      <div id="railsView" class="smallnote">Current: <span id="railsListText"></span></div>

      <div id="railsEditor" class="hidden">
        <div id="railItems"></div>
        <div class="rail-row" style="margin-top:6px;">
          <input type="number" id="addPct" placeholder="Add % (e.g. 92.5)" step="0.1" min="0.1" max="99.9"/>
          <button id="addBtn">Add</button>
          <span class="smallnote">Valid range: 0.1–99.9. Use 20 for Floor, 97.5 for Ceiling, 50 for mid.</span>
        </div>
      </div>
    </fieldset>

    <div style="font-size:12px;color:#6b7280;">Detected denominators: <span id="denomsDetected"></span></div>

    <div id="readout">
      <div class="date">—</div>
      <div id="readoutRows"></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span id="mainVal" class="num">$0.00</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP = {json.dumps(PRECOMP)};
const GENESIS = new Date('{GENESIS_DATE.strftime("%Y-%m-%d")}T00:00:00Z');

function fmtUSD(v){{ return (isFinite(v)? '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}}) : '$—'); }}
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){{ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }}
function shortDateFromYears(y){{ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${{MONTHS[d.getUTCMonth()]}}-${{String(d.getUTCDate()).padStart(2,'0')}}-${{String(d.getUTCFullYear()).slice(-2)}}`; }}
function interp(xs, ys, x){{ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){{ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }}
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }}
function pctWithinLog(y,f,c){{ const ly=Math.log10(y), lf=Math.log10(f), lc=Math.log10(c); return Math.max(0,Math.min(100,100*(ly-lf)/Math.max(1e-12,lc-lf))); }}

// Smooth red→yellow→green color scale based on percentile p (0..100)
function colorForPercent(p){{ 
  const t = Math.max(0, Math.min(1, p/100)); // 0..1
  // piecewise: red (#D32F2F)→yellow (#FBC02D)→green (#2E7D32)
  function hex(c){{ return Math.max(0, Math.min(255, Math.round(c))); }}
  function toHex(r,g,b){{ return '#' + [r,g,b].map(v=>hex(v).toString(16).padStart(2,'0')).join(''); }}
  // first half 0..0.5 red→yellow
  if (t<=0.5) {{
    const u=t/0.5;
    const r=0xD3 + (0xFB-0xD3)*u;
    const g=0x2F + (0xC0-0x2F)*u;
    const b=0x2F + (0x2D-0x2F)*u;
    return toHex(r,g,b);
  }} else {{
    const u=(t-0.5)/0.5;
    const r=0xFB + (0x2E-0xFB)*u;
    const g=0xC0 + (0x7D-0xC0)*u;
    const b=0x2D + (0x32-0x2D)*u;
    return toHex(r,g,b);
  }}
}}

const leftCol=document.getElementById('leftCol');
const plotDiv=document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div');
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const setBtn=document.getElementById('setDateBtn');
const liveBtn=document.getElementById('liveBtn');
const copyBtn=document.getElementById('copyBtn');
const fitBtn=document.getElementById('fitBtn');
const elDenoms=document.getElementById('denomsDetected');
const elDate=document.querySelector('#readout .date');
const elRows=document.getElementById('readoutRows');
const elMain=document.getElementById('mainVal');
const elP=document.getElementById('pPct');
const chartWpx=document.getElementById('chartWpx');

const editBtn=document.getElementById('editBtn');
const railsView=document.getElementById('railsView');
const railsListText=document.getElementById('railsListText');
const railsEditor=document.getElementById('railsEditor');
const railItems=document.getElementById('railItems');
const addPct=document.getElementById('addPct');
const addBtn=document.getElementById('addBtn');

function applyChartWidthPx(px){{ 
  const v=Math.max(400, Math.min(2400, Number(px)||1100));
  leftCol.style.flex='0 0 auto';
  leftCol.style.width=v+'px';
  if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv);
}}
chartWpx.addEventListener('change',()=>applyChartWidthPx(chartWpx.value));
fitBtn.addEventListener('click',()=>{{ 
  const total=document.documentElement.clientWidth || window.innerWidth;
  const panel=420; const pad=32; 
  const target=Math.max(400, total - panel - pad);
  chartWpx.value=target; applyChartWidthPx(target);
}});
if(window.ResizeObserver) new ResizeObserver(()=>{{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }}).observe(leftCol);

// Build denom list
const denomKeys = Object.keys(PRECOMP);
const extra = denomKeys.filter(k=>k!=='USD');
elDenoms.textContent = extra.length ? extra.join(', ') : '(none)';
['USD', ...extra].forEach(k=>{{ const o=document.createElement('option'); o.value=k; o.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(o); }});

// Rails state (sorted high→low)
let rails = [97.5, 90, 75, 50, 25, 20];  // initial set
const MAX_SLOTS = {MAX_RAIL_SLOTS};

function sortRails(){{
  rails = rails
    .filter(p=>isFinite(p))
    .map(p=>Math.max(0.1, Math.min(99.9, Number(p))))
    .filter((p,i,arr)=>arr.indexOf(p)===i) // dedupe
    .sort((a,b)=>b-a); // high→low
}}

function railsText(){{
  return rails.map(p=>String(p).replace(/\.0$/, '')+'%').join(', ');
}}

// Build readout rows dynamically (to match rails selection)
function rebuildReadoutRows(){{
  elRows.innerHTML='';
  rails.forEach(p=>{{
    const row=document.createElement('div');
    row.className='row';
    const lab=document.createElement('div');
    const val=document.createElement('div'); val.className='num'; val.id='v'+p;
    const color = colorForPercent(p);
    let labelText = (Math.abs(p-20)<1e-9)?'Floor': (Math.abs(p-97.5)<1e-9?'Ceiling': p+'%');
    // make 50% bold
    const bOpen = (Math.abs(p-50)<1e-9)?'<b>':'';
    const bClose= (Math.abs(p-50)<1e-9)?'</b>':'';
    lab.innerHTML = `<span style="color:${{color}};">${{bOpen}}${{labelText}}${{bClose}}</span>`;
    row.appendChild(lab); row.appendChild(val); row.appendChild(document.createElement('div'));
    elRows.appendChild(row);
  }});
}}

// Editor UI
function rebuildEditor(){{
  railItems.innerHTML='';
  rails.forEach((p,idx)=>{{
    const row=document.createElement('div'); row.className='rail-row';
    const color=colorForPercent(p);
    const lab=document.createElement('span'); lab.style.minWidth='38px'; lab.style.color=color; lab.textContent=(Math.abs(p-20)<1e-9?'Floor':Math.abs(p-97.5)<1e-9?'Ceiling':p+'%');
    const inp=document.createElement('input'); inp.type='number'; inp.step='0.1'; inp.min='0.1'; inp.max='99.9'; inp.value=String(p);
    const rm=document.createElement('button'); rm.textContent='Remove';
    inp.addEventListener('change',()=>{{ 
      const v=Number(inp.value);
      rails[idx]=isFinite(v)?v:p;
      sortRails(); syncAll();
    }});
    rm.addEventListener('click',()=>{{ rails.splice(idx,1); syncAll(); }});
    row.appendChild(lab); row.appendChild(inp); row.appendChild(rm);
    railItems.appendChild(row);
  }});
  railsListText.textContent = railsText();
}}

addBtn.addEventListener('click',()=>{{
  const v=Number(addPct.value);
  if (!isFinite(v)) return;
  rails.push(v);
  addPct.value='';
  sortRails(); syncAll();
}});

editBtn.addEventListener('click',()=>{{
  railsEditor.classList.toggle('hidden');
  railsView.classList.toggle('hidden');
  editBtn.textContent = railsEditor.classList.contains('hidden') ? 'Edit Rails' : 'Done';
}});

// Compute logM(x) across grid (midline); helper to get offset d for percentile
function logMidline(P){{ 
  const d=P.support; 
  return P.x_grid.map(x=> (d.a0 + d.b*Math.log10(x)) );
}}
function offsetForPercent(P, percent){{
  const d=P.support;
  const qUpper = 1 - (percent/100);
  const q = Math.max(d.q_grid[0], Math.min(d.q_grid[d.q_grid.length-1], qUpper));
  const off = interp(d.q_grid, d.d_grid, q);
  const med = d.med;
  // symmetric: med ± d
  return (percent>=50 ? med + off : med - off);
}}

function seriesForPercent(P, percent){{
  const logM = logMidline(P); const off = offsetForPercent(P, percent);
  // small epsilon at the very top to avoid touching midline in far extrapolation
  const eps = (percent>97.4)? {EPS_LOG_SPACING} : 0.0;
  return logM.map(v=> Math.pow(10, v + off + (percent>=50? eps: -eps)) );
}}

// Render rails into pre-allocated slots (0..MAX_SLOTS-1)
function renderRails(P){{
  // Ensure we don't exceed slot count
  const n = Math.min(rails.length, MAX_SLOTS);
  for (let i=0;i<MAX_SLOTS;i++){{ 
    const visible = (i<n);
    let restyle = {{visible: visible}};
    if (visible) {{
      const p = rails[i];
      const name = (Math.abs(p-20)<1e-9)?'Floor' : (Math.abs(p-97.5)<1e-9?'Ceiling': (p+'%'));
      const color = colorForPercent(p);
      const dash = (Math.abs(p-50)<1e-9)? undefined : 'dot';
      const width = (Math.abs(p-50)<1e-9)? 2.6 : 1.6;
      restyle = Object.assign(restyle, {{
        x: [P.x_grid],
        y: [seriesForPercent(P, p)],
        name: name,
        line: {{color: color, width: width, dash: dash}}
      }});
    }}
    Plotly.restyle(plotDiv, restyle, [i]);
  }}
  railsListText.textContent = railsText();
  rebuildReadoutRows();
}}

let locked=false, lockedX=null;

function updatePanel(P,xYears){{
  elDate.textContent=shortDateFromYears(xYears);

  // Determine "Floor" and "Ceiling" chosen (if present), else use min/max rails
  const pFloor = rails.find(p=>Math.abs(p-20)<1e-9) ?? Math.min(...rails);
  const pCeil  = rails.find(p=>Math.abs(p-97.5)<1e-9) ?? Math.max(...rails);

  const floor = interp(P.x_grid, seriesForPercent(P,pFloor), xYears);
  const ceil  = interp(P.x_grid, seriesForPercent(P,pCeil),  xYears);

  // Fill each readout row in current order
  rails.forEach(p=>{{
    const v = interp(P.x_grid, seriesForPercent(P,p), xYears);
    const el = document.getElementById('v'+p);
    if (el) el.textContent = fmtUSD(v);
  }});

  // Main series value near xYears (snap to nearest x_main)
  let idx=0,best=1e99; 
  for(let i=0;i<P.x_main.length;i++){{ 
    const d=Math.abs(P.x_main[i]-xYears); if(d<best){{best=d; idx=i;}} 
  }}
  const y=P.y_main[idx]; elMain.textContent=fmtUSD(y);
  elP.textContent = `(p≈${{pctWithinLog(y, floor, ceil).toFixed(1)}}%)`;
  Plotly.relayout(plotDiv, {{"yaxis.title.text": P.label}});
}}

// Rails visibility sync is implicit because rails are always visible in slots that are used

// Hover & controls
plotDiv.on('plotly_hover', ev=>{{ if(ev.points && ev.points.length && !locked) updatePanel(PRECOMP[denomSel.value], ev.points[0].x); }});
setBtn.onclick = ()=>{{ if(!datePick.value) return; locked=true; lockedX=yearsFromISO(datePick.value); updatePanel(PRECOMP[denomSel.value], lockedX); }};
liveBtn.onclick = ()=>{{ locked=false; lockedX=null; }};
copyBtn.onclick = async ()=>{{ 
  const node=document.getElementById('capture');
  try{{ 
    const url=await htmlToImage.toPng(node,{{pixelRatio:2}});
    try{{ if(navigator.clipboard && window.ClipboardItem){{ const blob=await (await fetch(url)).blob(); await navigator.clipboard.write([new ClipboardItem({{'image/png':blob}})]); return; }} }}catch(e){{}}
    const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png'; document.body.appendChild(a); a.click(); a.remove();
  }}catch(e){{ console.error(e); }}
}};

// Denominator change
denomSel.onchange = ()=>{{ 
  const key=denomSel.value, P=PRECOMP[key];
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main], name:[P.label]}}, [{MAX_RAIL_SLOTS}]);     // BTC line
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main]}},              [{MAX_RAIL_SLOTS}+1]);      // cursor
  renderRails(P);
  updatePanel(P,(typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
}};

// Sync everything (rails list, editor, figure, readout)
function syncAll(){{
  sortRails();
  rebuildEditor();
  const P = PRECOMP[denomSel.value];
  renderRails(P);
  updatePanel(P,(typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
}}

// Init
denomSel.value='USD';
applyChartWidthPx(document.getElementById('chartWpx').value);
rebuildReadoutRows();
rebuildEditor();
renderRails(PRECOMP['USD']);
updatePanel(PRECOMP['USD'], PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);

</script>
</body></html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Write site
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"Wrote {OUTPUT_HTML}")