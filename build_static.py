#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Purchase Indicator — Power-law rails with a robust median midline,
winsorized & symmetric percentile offsets, and a responsive UI.

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

# Default future horizon (years beyond last data) to show rails without wasting half the plot
FUTURE_YEARS = 3   # change to taste

# Rails behaviour
RESID_WINSOR     = 0.02   # clip 2% tails in residuals (robust ceiling/floor)
SYMMETRIC_RAILS  = True   # mirror rails about median residual
EPS_LOG_SPACING  = 0.010  # keep rails from touching (~2.3% in linear space)

# Colors (floor→ceiling red→green)
COL_FLOOR   = "#D32F2F"
COL_20      = "#F57C00"
COL_50      = "#FBC02D"
COL_80      = "#66BB6A"
COL_CEILING = "#2E7D32"
COL_BTC     = "#000000"

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
# Power-law fit & rails
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

def winsorize(arr, p):
    lo, hi = np.nanquantile(arr, p), np.nanquantile(arr, 1-p)
    return np.clip(arr, lo, hi)

def symmetric_offsets(resid, q_upper):
    """Return (low, high) symmetric about median residual."""
    med = float(np.nanmedian(resid))
    d_hi = float(np.nanquantile(resid - med, q_upper))
    d_lo = float(np.nanquantile(med - resid, q_upper))
    d = max(d_hi, d_lo)
    return med - d, med + d

def compute_defaults(x_years, y_vals):
    """Find midline (q50) and rail offsets at 2.5/97.5 and 20/80."""
    a0, b, resid = quantile_fit_loglog(x_years, y_vals, q=0.5)
    r = winsorize(resid, RESID_WINSOR) if RESID_WINSOR else resid
    if SYMMETRIC_RAILS:
        c025, c975 = symmetric_offsets(r, 0.975)
        c200, c800 = symmetric_offsets(r, 0.800)
    else:
        c025 = float(np.nanquantile(r, 0.025))
        c975 = float(np.nanquantile(r, 0.975))
        c200 = float(np.nanquantile(r, 0.200))
        c800 = float(np.nanquantile(r, 0.800))
    # keep floor/ceiling from touching on future extrapolation
    c025 -= EPS_LOG_SPACING
    c975 += EPS_LOG_SPACING
    return {"a0":a0,"b":b,"c025":c025,"c200":c200,"c800":c800,"c975":c975}

# ──────────────────────────────────────────────────────────────────────────────
# Axis ticks
# ──────────────────────────────────────────────────────────────────────────────
def year_ticks_log(first_dt, last_dt):
    """Whole-year ticks on log-x (years since genesis). Hide odd years after 2026."""
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
    # True log scale: no fake "0" tick
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

# Horizon limited to FUTURE_YEARS past the last data date (but never beyond END_PROJ)
last_dt = base["date"].iloc[-1]
max_dt  = datetime(min(END_PROJ.year, last_dt.year + FUTURE_YEARS), 12, 31)

# x_grid for rails (log-spaced in x-years) to the chosen horizon
x_start = float(base["x_years"].iloc[0])
x_end   = float(years_since_genesis(pd.Series([max_dt])).iloc[0])
x_grid  = np.logspace(np.log10(max(1e-6, x_start)), np.log10(x_end), 700)

# x/y ticks (to horizon)
xtickvals, xticktext = year_ticks_log(base["date"].iloc[0], max_dt)
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
    dft = compute_defaults(df["x_years"], y)
    return {
        "label": label,
        "x_main": df["x_years"].tolist(),
        "y_main": y.tolist(),
        "date_iso_main": df["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "defaults": dft
    }

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)

P0 = PRECOMP["USD"]

# ──────────────────────────────────────────────────────────────────────────────
# Base figure (rails are filled at runtime via JS)
# ──────────────────────────────────────────────────────────────────────────────
def add_stub(name, color, width=1.6, dash=None, bold=False):
    line = dict(width=2.6 if bold else width, color=color)
    if dash: line["dash"] = dash
    return go.Scatter(x=P0["x_grid"], y=[None]*len(P0["x_grid"]), mode="lines",
                      name=name, line=line, hoverinfo="skip")

fig = go.Figure([
    add_stub("Floor",   COL_FLOOR),
    add_stub("20%",     COL_20, dash="dot"),
    add_stub("50%",     COL_50, bold=True),
    add_stub("80%",     COL_80, dash="dot"),
    add_stub("Ceiling", COL_CEILING),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    # transparent cursor trace to keep x-hover alive
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
])

# Set the default visible range to [first data .. horizon]
x_min = float(base["x_years"].iloc[0])
x_max = float(years_since_genesis(pd.Series([max_dt])).iloc[0])

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",               # unified hover label (no persistent line)
    showlegend=True,
    title="BTC Purchase Indicator — Rails",
    xaxis=dict(
        type="log",
        title=None,
        tickmode="array",
        tickvals=xtickvals,
        ticktext=xticktext,
        range=[np.log10(x_min), np.log10(x_max)],   # avoid big empty right half
        showspikes=False
    ),
    yaxis=dict(
        type="log",
        title=P0["label"],
        tickmode="array",
        tickvals=ytickvals,
        ticktext=yticktext
    ),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),
)

plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive":True,"displayModeBar":True,"modeBarButtonsToRemove":["toImage"]})

# ──────────────────────────────────────────────────────────────────────────────
# HTML + JS (responsive layout + pixel width control)
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
select,button,input[type=date],input[type=number]{{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}}
#readout{{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fafafa;font-size:14px}}
#readout .date{{font-weight:700;margin-bottom:6px}}
#readout .row{{display:grid;grid-template-columns:auto 1fr auto;column-gap:8px;align-items:baseline}}
#readout .num{{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;text-align:right;min-width:12ch;white-space:pre}}
.hoverlayer{{opacity:0!important;pointer-events:none}}
@media (max-width:900px){{
  .layout{{flex-direction:column}}
  .right{{flex:0 0 auto;border-left:none;border-top:1px solid #e5e7eb}}
  .left{{flex:0 0 auto;width:100%;padding:8px}}
}}
#chartWidthBox{{display:flex;align-items:center;gap:8px}}
#chartWpx{{width:120px}}
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

    <div style="font-size:12px;color:#6b7280;">Detected denominators: <span id="denomsDetected"></span></div>

    <div id="readout">
      <div class="date">—</div>
      <div class="row"><div><span style="color:{COL_FLOOR};">Floor</span></div><div id="vF"  class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:{COL_20};">20%</span></div>  <div id="v20" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:{COL_50};font-weight:700;">50%</span></div><div id="v50" class="num" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:{COL_80};">80%</span></div>  <div id="v80" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:{COL_CEILING};">Ceiling</span></div><div id="vC"  class="num">$0.00</div><div></div></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span id="mainVal" class="num">$0.00</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP = {json.dumps(PRECOMP)};

function fmtUSD(v){{ return (isFinite(v)? '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}}) : '$—'); }}
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const GENESIS = new Date('{GENESIS_DATE.strftime("%Y-%m-%d")}T00:00:00Z');

function yearsFromISO(iso){{ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }}
function shortDateFromYears(y){{ const ms=(y-(1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${{MONTHS[d.getUTCMonth()]}}-${{String(d.getUTCDate()).padStart(2,'0')}}-${{String(d.getUTCFullYear()).slice(-2)}}`; }}
function interp(xs, ys, x){{ let lo=0,hi=xs.length-1; if(x<=xs[0]) return ys[0]; if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){{ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }}
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]); }}
function pctWithinLog(y,f,c){{ const ly=Math.log10(y), lf=Math.log10(f), lc=Math.log10(c); return Math.max(0,Math.min(100,100*(ly-lf)/Math.max(1e-12,lc-lf))); }}

function railsFromPercentiles(P){{
  const d=P.defaults, lx=P.x_grid.map(v=>Math.log10(v));
  const logM=lx.map(v=>d.a0 + d.b*v);
  const exp=a=>a.map(v=>Math.pow(10,v));
  return {{
    FLOOR:   exp(logM.map(v=>v+d.c025)),
    P20:     exp(logM.map(v=>v+d.c200)),
    P50:     exp(logM),
    P80:     exp(logM.map(v=>v+d.c800)),
    CEILING: exp(logM.map(v=>v+d.c975))
  }};
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
const elF=document.getElementById('vF'), el20=document.getElementById('v20'), el50=document.getElementById('v50'),
      el80=document.getElementById('v80'), elC=document.getElementById('vC'), elMain=document.getElementById('mainVal'),
      elP=document.getElementById('pPct');
const chartWpx=document.getElementById('chartWpx');

function applyChartWidthPx(px){{ 
  const v=Math.max(400, Math.min(2400, Number(px)||1100));
  leftCol.style.flex='0 0 auto';
  leftCol.style.width=v+'px';
  if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv);
}}
chartWpx.addEventListener('change',()=>applyChartWidthPx(chartWpx.value));
fitBtn.addEventListener('click',()=>{{ 
  // Fill remaining space minus panel width & paddings
  const total=document.documentElement.clientWidth || window.innerWidth;
  const panel=420; const pad=32; 
  const target=Math.max(400, total - panel - pad);
  chartWpx.value=target; applyChartWidthPx(target);
}});
if(window.ResizeObserver) new ResizeObserver(()=>{{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }}).observe(leftCol);

const denomKeys = Object.keys(PRECOMP);
const extra = denomKeys.filter(k=>k!=='USD');
elDenoms.textContent = extra.length ? extra.join(', ') : '(none)';
['USD', ...extra].forEach(k=>{{ const o=document.createElement('option'); o.value=k; o.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(o); }});

let CURRENT_RAILS=null, locked=false, lockedX=null;

function applyRails(P){{ 
  CURRENT_RAILS = railsFromPercentiles(P);
  Plotly.restyle(plotDiv, {{x:[P.x_grid], y:[CURRENT_RAILS.FLOOR]}},   [0]);
  Plotly.restyle(plotDiv, {{x:[P.x_grid], y:[CURRENT_RAILS.P20]}},     [1]);
  Plotly.restyle(plotDiv, {{x:[P.x_grid], y:[CURRENT_RAILS.P50]}},     [2]);
  Plotly.restyle(plotDiv, {{x:[P.x_grid], y:[CURRENT_RAILS.P80]}},     [3]);
  Plotly.restyle(plotDiv, {{x:[P.x_grid], y:[CURRENT_RAILS.CEILING]}}, [4]);
}}

function updatePanel(P,xYears){{ 
  elDate.textContent=shortDateFromYears(xYears);
  const F=interp(P.x_grid,CURRENT_RAILS.FLOOR,xYears);
  const v20=interp(P.x_grid,CURRENT_RAILS.P20,xYears);
  const v50=interp(P.x_grid,CURRENT_RAILS.P50,xYears);
  const v80=interp(P.x_grid,CURRENT_RAILS.P80,xYears);
  const C=interp(P.x_grid,CURRENT_RAILS.CEILING,xYears);
  elF.textContent=fmtUSD(F); el20.textContent=fmtUSD(v20); el50.textContent=fmtUSD(v50); el80.textContent=fmtUSD(v80); elC.textContent=fmtUSD(C);

  let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){{ const d=Math.abs(P.x_main[i]-xYears); if(d<best){{best=d; idx=i;}} }}
  const y=P.y_main[idx]; elMain.textContent=fmtUSD(y);
  elP.textContent = `(p≈${{pctWithinLog(y,F,C).toFixed(1)}}%)`;
  Plotly.relayout(plotDiv, {{"yaxis.title.text": P.label}});
}}

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

denomSel.onchange = ()=>{{ 
  const key=denomSel.value, P=PRECOMP[key];
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main], name:[P.label]}}, [5]);
  Plotly.restyle(plotDiv, {{x:[P.x_main], y:[P.y_main]}}, [6]);
  applyRails(P);
  updatePanel(P,(typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
}};

// Init
denomSel.value='USD';
applyChartWidthPx(document.getElementById('chartWpx').value);
applyRails(PRECOMP['USD']);
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