#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — dynamic percentile rails with an editable dashboard.

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
END_PROJ     = datetime(2040, 12, 31)
X_START_DATE = datetime(2011, 1, 1)

RESID_WINSOR     = 0.02   # clip 2% tails
EPS_LOG_SPACING  = 0.010  # tiny spacing for extreme rails
COL_BTC          = "#000000"

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def years_since_genesis(dates):
    d = pd.to_datetime(dates)
    delta_days = (d - GENESIS_DATE) / np.timedelta64(1, "D")
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)  # +1 day => log(x)>0

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

def load_denominators():
    out={}
    for p in glob.glob(os.path.join(DATA_DIR, "denominator_*.csv")):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            if len(df.columns) < 2: continue
            price_col = [c for c in df.columns if c.lower()!="date"][0]
            df = df.rename(columns={price_col:"price"})[["date","price"]]
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.sort_values("date").dropna()
            out[key]=df
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Fit support for dynamic rails
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
    """
    Provide everything JS needs to draw any percentile rail via interpolation:
      - a0, b: midline parameters for log10(y) = a0 + b*log10(xyears)
      - q_grid: dense percentile grid in [0.001..0.999]
      - off_grid: quantile(resid, q_grid) - median(resid)
    offset(50%) == 0, <50% negative, >50% positive.
    """
    a0, b, resid = quantile_fit_loglog(x_years, y_vals, q=0.5)
    r = np.copy(resid)
    if RESID_WINSOR:
        lo, hi = np.nanquantile(r, RESID_WINSOR), np.nanquantile(r, 1-RESID_WINSOR)
        r = np.clip(r, lo, hi)

    med = float(np.nanmedian(r))
    q_grid = np.linspace(0.001, 0.999, 999)
    rq = np.quantile(r, q_grid)
    off_grid = rq - med
    # keep extremes from touching midline on far extrapolation
    off_grid[0]  -= EPS_LOG_SPACING
    off_grid[-1] += EPS_LOG_SPACING

    return {"a0":a0, "b":b,
            "q_grid":[float(q) for q in q_grid],
            "off_grid":[float(v) for v in off_grid]}

# ──────────────────────────────────────────────────────────────────────────────
# Ticks
# ──────────────────────────────────────────────────────────────────────────────
def year_ticks_log(first_dt, last_dt):
    vals, labs = [], []
    y0, y1 = first_dt.year, last_dt.year
    for y in range(y0, y1+1):
        d = datetime(y,1,1)
        if d < first_dt or d > last_dt: continue
        vy = float(years_since_genesis(pd.Series([d])).iloc[0])
        if vy <= 0: continue
        if y > 2026 and (y % 2 == 1):  # hide odd labels after 2026
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
# Figure with pre-allocated rail slots
# ──────────────────────────────────────────────────────────────────────────────
MAX_RAIL_SLOTS = 12

def add_stub(idx):
    return go.Scatter(
        x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
        name=f"Rail {idx+1}", line=dict(width=1.6, color="#999"),
        visible=False, hoverinfo="skip"
    )

traces = [add_stub(i) for i in range(MAX_RAIL_SLOTS)]
traces += [
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
]

fig = go.Figure(traces)

x_min = float(years_since_genesis(pd.Series([first_dt])).iloc[0])
x_max = float(years_since_genesis(pd.Series([max_dt])).iloc[0])

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
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
# HTML + JS
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
        <span class="smallnote">Add/remove/change percents. Sorted automatically (high→low).</span>
      </div>
      <div id="railsView" class="smallnote">Current: <span id="railsListText"></span></div>

      <div id="railsEditor" class="hidden">
        <div id="railItems"></div>
        <div class="rail-row" style="margin-top:6px;">
          <input type="number" id="addPct" placeholder="Add % (e.g. 92.5)" step="0.1" min="0.1" max="99.9"/>
          <button id="addBtn">Add</button>
          <span class="smallnote">Valid range: 0.1–99.9. 50 is midline.</span>
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
const MAX_SLOTS = {MAX_RAIL_SLOTS};
const EPS_LOG_SPACING = {EPS_LOG_SPACING};

function fmtUSD(v){{ return (isFinite(v)? '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}}) : '$—'); }}
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function yearsFromISO(iso){{ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }}
function shortDateFromYears(y){{ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${{MONTHS[d.getUTCMonth()]}}-${{String(d.getUTCDate()).padStart(2,'0')}}-${{String(d.getUTCFullYear()).slice(-2)}}`; }}
function interp(xs, ys, x){{ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){{ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }}
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }}

// Smooth red→yellow→green by percentile (0..100)
function colorForPercent(p){{ 
  const t = Math.max(0, Math.min(1, p/100));
  function hex(c){{ return Math.max(0, Math.min(255, Math.round(c))); }}
  function toHex(r,g,b){{ return '#' + [r,g,b].map(v=>hex(v).toString(16).padStart(2,'0')).join(''); }}
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

// DOM handles
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

// Denominators
const denomKeys = Object.keys(PRECOMP);
const extra = denomKeys.filter(k=>k!=='USD');
elDenoms.textContent = extra.length ? extra.join(', ') : '(none)';
['USD', ...extra].forEach(k=>{{ const o=document.createElement('option'); o.value=k; o.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(o); }});

// Rails state (sorted high→low). Lowest default is 2.5%.
let rails = [97.5, 90, 75, 50, 25, 2.5];

function sortRails(){{
  rails = rails
    .filter(p=>isFinite(p))
    .map(p=>Math.max(0.1, Math.min(99.9, Number(p))))
    .filter((p,i,arr)=>arr.indexOf(p)===i)
    .sort((a,b)=>b-a);
}}
function railsText(){{
  return rails.map(p=>String(p).replace(/\.0$/, '')+'%').join(', ');
}}
// Safe id for readout rows (avoid '.' in ids)
function idFor(p){{ return 'v'+String(p).replace('.', '_'); }}

// Readout rows to match rails (Floor label for 2.5, Ceiling for 97.5)
function rebuildReadoutRows(){{
  elRows.innerHTML='';
  rails.forEach(p=>{{
    const row=document.createElement('div'); row.className='row';
    const lab=document.createElement('div');
    const val=document.createElement('div'); val.className='num'; val.id=idFor(p);
    const color = colorForPercent(p);
    const isMid = Math.abs(p-50)<1e-9;
    const name  = Math.abs(p-2.5)<1e-9 ? 'Floor'
                 : Math.abs(p-97.5)<1e-9 ? 'Ceiling'
                 : (p+'%');
    lab.innerHTML = `<span style="color:${{color}};">${{isMid?'<b>':''}}${{name}}${{isMid?'</b>':''}}</span>`;
    row.appendChild(lab); row.appendChild(val); row.appendChild(document.createElement('div'));
    elRows.appendChild(row);
  }});
}}

// Editor labels
function rebuildEditor(){{
  railItems.innerHTML='';
  rails.forEach((p,idx)=>{{
    const row=document.createElement('div'); row.className='rail-row';
    const color=colorForPercent(p);
    const labelTxt = Math.abs(p-2.5)<1e-9 ? 'Floor'
                    : Math.abs(p-97.5)<1e-9 ? 'Ceiling'
                    : (p+'%');
    const lab=document.createElement('span'); lab.style.minWidth='48px'; lab.style.color=color; lab.textContent=labelTxt;
    const inp=document.createElement('input'); inp.type='number'; inp.step='0.1'; inp.min='0.1'; inp.max='99.9'; inp.value=String(p);
    const rm=document.createElement('button'); rm.textContent='Remove';
    inp.addEventListener('change',()=>{{ const v=Number(inp.value); rails[idx]=isFinite(v)?v:p; sortRails(); syncAll(); }});
    rm.addEventListener('click',()=>{{ rails.splice(idx,1); syncAll(); }});
    row.appendChild(lab); row.appendChild(inp); row.appendChild(rm);
    railItems.appendChild(row);
  }});
  railsListText.textContent = railsText();
}}
addBtn.addEventListener('click',()=>{{ const v=Number(addPct.value); if(!isFinite(v)) return; rails.push(v); addPct.value=''; sortRails(); syncAll(); }});
editBtn.addEventListener('click',()=>{{ railsEditor.classList.toggle('hidden'); railsView.classList.toggle('hidden'); editBtn.textContent = railsEditor.classList.contains('hidden') ? 'Edit Rails' : 'Done'; }});

// Layout sizing
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

// Midline + offset helpers (from residual quantiles)
function logMidline(P){{ const d=P.support; return P.x_grid.map(x=> (d.a0 + d.b*Math.log10(x)) ); }}
function offsetForPercent(P, percent){{
  const d=P.support;
  const p01 = Math.max(d.q_grid[0], Math.min(d.q_grid[d.q_grid.length-1], percent/100));
  return interp(d.q_grid, d.off_grid, p01);
}}
function seriesForPercent(P, percent){{
  const logM = logMidline(P); const off = offsetForPercent(P, percent);
  const eps = (percent>=97.5 || percent<=2.5) ? EPS_LOG_SPACING : 0.0;
  return logM.map(v=> Math.pow(10, v + off + (percent>=50? eps : -eps)) );
}}

// Invert residual CDF: given offset -> percentile q (0..1)
function percentFromOffset(P, off){{
  const d=P.support;
  const q = Math.max(d.q_grid[0], Math.min(d.q_grid[d.q_grid.length-1], interp(d.off_grid, d.q_grid, off)));
  return q; // 0..1
}}

// Render into pre-allocated slots
function renderRails(P){{
  const n = Math.min(rails.length, MAX_SLOTS);
  for (let i=0;i<MAX_SLOTS;i++){{ 
    const visible = (i<n);
    let restyle = {{visible: visible}};
    if (visible) {{
      const p = rails[i];
      const color = colorForPercent(p);
      const dash  = (Math.abs(p-50)<1e-9)? 'solid' : 'dot';
      const width = (Math.abs(p-50)<1e-9)? 2.6 : 1.6;
      restyle = Object.assign(restyle, {{
        x: [P.x_grid],
        y: [seriesForPercent(P, p)],
        name: (Math.abs(p-2.5)<1e-9?'Floor':(Math.abs(p-97.5)<1e-9?'Ceiling':(p+'%'))),
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

  // 2.5% .. 97.5% envelope values (still used for context if you want)
  const floor = interp(P.x_grid, seriesForPercent(P, 2.5),  xYears);
  const ceil  = interp(P.x_grid, seriesForPercent(P, 97.5), xYears);

  // Fill each selected rail’s readout
  rails.forEach(p=>{{
    const v = interp(P.x_grid, seriesForPercent(P,p), xYears);
    const el = document.getElementById(idFor(p));
    if (el) el.textContent = fmtUSD(v);
  }});

  // Current price at nearest sample
  let idx=0,best=1e99; 
  for(let i=0;i<P.x_main.length;i++){{ const d=Math.abs(P.x_main[i]-xYears); if(d<best){{best=d; idx=i;}} }}
  const y=P.y_main[idx]; elMain.textContent=fmtUSD(y);

  // Compute empirical percentile of current price via residual CDF
  const d=P.support;
  const logx = Math.log10(xYears);
  const ly   = Math.log10(y);
  const mid  = d.a0 + d.b*logx;
  const off  = ly - mid;             // residual offset from midline
  const q    = percentFromOffset(P, off);  // 0..1
  const pVal = Math.max(0, Math.min(100, 100*q));
  elP.textContent = `(p≈${{pVal.toFixed(1)}}%)`;

  Plotly.relayout(plotDiv, {{"yaxis.title.text": P.label}});
}}

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
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main], name:[P.label]}}, [{MAX_RAIL_SLOTS}]);
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main]}},              [{MAX_RAIL_SLOTS}+1]);
  renderRails(P);
  updatePanel(P,(typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
}};

// Sync everything
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