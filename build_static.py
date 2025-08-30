#!/usr/bin/env python3
"""
BTC Purchase Indicator — Midline Percentiles (responsive with adjustable width)

Main changes vs previous version:
- Midline is q50 (median) quantile regression on log(price) vs log(years).
- Rails are constant log offsets at 2.5 %, 20 %, 80 %, and 97.5 %, derived from
  winsorised, symmetric residuals around the midline. This reduces early-cycle
  blow-offs, aligns better with charts such as Porkopolis, and ensures rails
  are parallel.
- Responsive flex layout: a CSS variable --chartPct controls the chart width.
  A range slider updates this variable and resizes the Plotly graph via a
  ResizeObserver. Works across desktop and mobile.
"""

import os, io, glob, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# File paths
DATA_DIR = "data"
BTC_FILE = os.path.join(DATA_DIR, "btc_usd.csv")
OUTPUT_HTML = "docs/index.html"

GENESIS_DATE = datetime(2009, 1, 3)
END_PROJ = datetime(2040, 12, 31)

# Rails parameters
EPS_LOG = 0.010        # spacing to avoid overlapping rails (≈2.3 %)
RESID_WINSOR = 0.02    # clip tails at 2 % to curb early spikes
SYMMETRIC_RAILS = True # mirror residual quantiles around median

# Colour scheme (floor→ceiling red→green)
COL_FLOOR   = "#D32F2F"
COL_20      = "#F57C00"
COL_50      = "#FBC02D"
COL_80      = "#66BB6A"
COL_CEILING = "#2E7D32"
COL_BTC     = "#000000"

def years_since_genesis(dates):
    delta_days = (pd.to_datetime(dates) - GENESIS_DATE) / np.timedelta64(1, "D")
    # add a day so log(x) > 0
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)

def fetch_btc():
    """
    Returns a DataFrame with columns ['date','price'].
    If data file exists locally, reads from it; otherwise fetches from
    Blockchain.com and writes to file.
    """
    if os.path.exists(BTC_FILE):
        return pd.read_csv(BTC_FILE, parse_dates=["date"])
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    data = requests.get(url, timeout=30).text.strip().splitlines()
    # first line may be header or not, detect
    if data[0].lower().startswith("timestamp"):
        df = pd.read_csv(io.StringIO("\n".join(data)))
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "price"})
    else:
        df = pd.read_csv(io.StringIO("\n".join(data)), header=None, names=["date","price"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"])
    df = df.dropna().sort_values("date")
    df.to_csv(BTC_FILE, index=False)
    return df

def collect_denominators():
    """
    Returns a dict mapping uppercase denom key to DataFrame with columns ['date','price'].
    Looks for files named denominator_*.csv in DATA_DIR.
    """
    out={}
    for path in glob.glob(os.path.join(DATA_DIR,"denominator_*.csv")):
        key = os.path.splitext(os.path.basename(path))[0].replace("denominator_","").upper()
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.rename(columns={"date":"date", df.columns[1]:"price"})
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.dropna().sort_values("date")
            out[key]=df
        except Exception as e:
            print(f"[warn] Could not load {path}: {e}")
    return out

def quantile_fit(x_years, y_vals, q=0.5):
    """
    Fits Quantile regression (q) to log(price) vs log(x_years).
    Returns (a0,b,resid,mask).
    """
    m = (x_years>0)&(y_vals>0)&np.isfinite(x_years)&np.isfinite(y_vals)
    xlog = np.log10(x_years[m]); ylog = np.log10(y_vals[m])
    X = pd.DataFrame({"const":1.0,"logx":xlog})
    res = QuantReg(ylog, X).fit(q=q)
    a0, b = float(res.params["const"]), float(res.params["logx"])
    resid = ylog - (a0 + b*xlog)
    return a0, b, resid, m

def winsorize(arr, p):
    lo, hi = np.nanquantile(arr, p), np.nanquantile(arr, 1-p)
    return np.clip(arr, lo, hi)

def symmetric_offsets(resid, q):
    """For symmetric rails: returns (low, high) such that median±d captures q."""
    median = float(np.nanmedian(resid))
    d_hi = float(np.nanquantile(resid - median, q))
    d_lo = float(np.nanquantile(median - resid, q))
    d = max(d_hi, d_lo)
    return median - d, median + d

def defaults_for_series(x_years, y_vals):
    """
    Computes the midline (q50) and four rail offsets (2.5,20,80,97.5) in log space
    using winsorised, symmetric residuals if configured.
    """
    a0, b, resid, _ = quantile_fit(x_years, y_vals, q=0.5)
    r = resid.copy()
    if RESID_WINSOR and RESID_WINSOR>0:
        r = winsorize(r, RESID_WINSOR)
    if SYMMETRIC_RAILS:
        c025, c975 = symmetric_offsets(r, 0.975)
        c200, c800 = symmetric_offsets(r, 0.800)
    else:
        c025 = float(np.nanquantile(r, 0.025))
        c975 = float(np.nanquantile(r, 0.975))
        c200 = float(np.nanquantile(r, 0.200))
        c800 = float(np.nanquantile(r, 0.800))
    # buffer so rails never touch
    c025 -= EPS_LOG
    c975 += EPS_LOG
    return {"a0":a0, "b":b, "c025":c025, "c200":c200, "c800":c800, "c975":c975}

def build_payload(base, denom_key=None):
    y = base["btc"] if denom_key is None else base["btc"]/base[denom_key.lower()]
    label = "BTC / USD" if denom_key is None else f"BTC / {denom_key.upper()}"
    defaults = defaults_for_series(base["x_years"], y)
    return {
        "label": label,
        "x_main": base["x_years"].tolist(),
        "y_main": y.tolist(),
        "date_iso_main": base["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "defaults": defaults
    }

# prepare data
btc = fetch_btc()
base = btc.rename(columns={"price":"btc"}).sort_values("date").reset_index(drop=True)
denoms = collect_denominators()
for k,df in denoms.items():
    base = base.merge(df.rename(columns={"price": k.lower()}), on="date", how="left")
base["x_years"] = years_since_genesis(base["date"])
base["date_iso"] = base["date"].dt.strftime("%Y-%m-%d")

PRECOMP = {"USD": build_payload(base, None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(base, k)

# y-ticks with comma labels
def y_ticks():
    vals=[1e-8]+[10**e for e in range(0,9)]
    labs=["0"]+[f"{int(10**e):,}" for e in range(0,9)]
    return vals,labs
ytickvals, yticktext = y_ticks()

# Build base figure with placeholders for rails
fig = go.Figure([
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="Floor", line=dict(color=COL_FLOOR)),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="20%", line=dict(color=COL_20, dash="dot")),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="50%", line=dict(color=COL_50, width=3)),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="80%", line=dict(color=COL_80, dash="dot")),
    go.Scatter(x=x_grid, y=[None]*len(x_grid), mode="lines", name="Ceiling", line=dict(color=COL_CEILING)),
    go.Scatter(x=PRECOMP["USD"]["x_main"], y=PRECOMP["USD"]["y_main"],
               name="BTC / USD", mode="lines", line=dict(color=COL_BTC)),
    # transparent hover line
    go.Scatter(x=PRECOMP["USD"]["x_main"], y=PRECOMP["USD"]["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
])

fig.update_layout(
    template="plotly_white",
    hovermode="x",
    showlegend=True,
    title="BTC Purchase Indicator — Rails",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=PRECOMP["USD"]["label"], tickmode="array",
               tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),
)

# Write HTML
html_template = Template(r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
:root{--panelW:420px; --chartPct:72%}
html,body{height:100%; margin:0; font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
.layout{display:flex;min-height:100vh;width:100vw}
.left{flex:1 1 var(--chartPct);min-width:280px;padding:8px 0 8px 8px}
.left .js-plotly-plot,.left .plotly-graph-div{width:100%!important}
.right{flex:0 0 var(--panelW); border-left:1px solid #e5e7eb;padding:12px; display:flex; flex-direction:column; gap:12px; overflow:auto}
#controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,button,input[type=date],input[type=range]{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}
#readout{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fafafa;font-size:14px}
#readout .date{font-weight:700;margin-bottom:6px}
#readout .row{display:grid;grid-template-columns:auto 1fr auto;column-gap:8px;align-items:baseline}
#readout .num{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;text-align:right;min-width:12ch;white-space:pre}
.hoverlayer{opacity:0!important;pointer-events:none}
@media (max-width:900px){
  .layout{flex-direction:column}
  .right{flex:0 0 auto; border-left:none; border-top:1px solid #e5e7eb}
  .left{flex:0 0 auto;width:100%;padding:8px}
}
#chartWidthBox{display:flex;align-items:center;gap:8px}
#chartW{width:220px}
#chartWVal{min-width:3ch;text-align:right}
</style>
</head><body>
<div id="capture" class="layout">
  <div class="left">$plot_html</div>
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
      <b>Chart Width:</b>
      <input type="range" id="chartW" min="55" max="90" value="72"/>
      <span id="chartWVal">72%</span>
      <span style="color:#6b7280;font-size:12px;">(plot / panel ratio)</span>
    </div>

    <div style="font-size:12px;color:#6b7280;">Detected denominators: <span id="denomsDetected"></span></div>

    <div id="readout">
      <div class="date">—</div>
      <div class="row"><div><span style="color:${COL_FLOOR};">Floor</span></div><div id="vF" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:${COL_20};">20%</span></div><div id="v20" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:${COL_50};font-weight:700;">50%</span></div><div id="v50" class="num" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:${COL_80};">80%</span></div><div id="v80" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:${COL_CEILING};">Ceiling</span></div><div id="vC" class="num">$0.00</div><div></div></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span id="mainVal" class="num">$0.00</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>
<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP = $precomp_json;
function fmtUSD(v){ return (isFinite(v) ? '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '$—'); }
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const GENESIS = new Date('$genesis_iso'+'T00:00:00Z');

function yearsFromISO(iso){ const d=new Date(iso+'T00:00:00Z'); return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25); }
function shortDateFromYears(y){ const ms = (y - (1.0/365.25))*365.25*86400000; const d=new Date(GENESIS.getTime()+ms); return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`; }
function interp(xs, ys, x){
  let lo=0,hi=xs.length-1;
  if(x<=xs[0]) return ys[0];
  if(x>=xs[hi]) return ys[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xs[m]<=x) lo=m; else hi=m; }
  const t=(x-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]);
}
function pctWithinLog(y,f,c){
  const ly=Math.log10(y), lf=Math.log10(f), lc=Math.log10(c);
  return Math.max(0,Math.min(100,100*(ly-lf)/Math.max(1e-12,lc-lf)));
}

function railsFromPercentiles(P){
  const {a0,b,c025,c200,c800,c975}=P.defaults;
  const lx=P.x_grid.map(v=>Math.log10(v));
  const logM=lx.map(v=>a0+b*v);
  const exp=a=>a.map(v=>Math.pow(10,v));
  return {
    FLOOR:   exp(logM.map(v=>v+c025)),
    P20:     exp(logM.map(v=>v+c200)),
    P50:     exp(logM),
    P80:     exp(logM.map(v=>v+c800)),
    CEILING: exp(logM.map(v=>v+c975))
  };
}

const plotDiv=document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div');
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const setBtn=document.getElementById('setDateBtn');
const liveBtn=document.getElementById('liveBtn');
const copyBtn=document.getElementById('copyBtn');
const elDenoms=document.getElementById('denomsDetected');
const elDate=document.querySelector('#readout .date');
const elF=document.getElementById('vF'), el20=document.getElementById('v20'),
      el50=document.getElementById('v50'), el80=document.getElementById('v80'), elC=document.getElementById('vC'),
      elMain=document.getElementById('mainVal'), elP=document.getElementById('pPct');
const chartW=document.getElementById('chartW');
const chartWVal=document.getElementById('chartWVal');
const layoutRoot=document.querySelector('.layout');

function applyChartWidth(pct){
  document.documentElement.style.setProperty('--chartPct', pct+'%');
  chartWVal.textContent=pct+'%';
  if(window.Plotly&&plotDiv){ Plotly.Plots.resize(plotDiv); }
}
chartW.addEventListener('input', ()=>applyChartWidth(chartW.value));

if(window.ResizeObserver){
  new ResizeObserver(()=>{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); }).observe(document.querySelector('.left'));
}

const extraDenoms = Object.keys(PRECOMP).filter(k=>k!=='USD');
elDenoms.textContent = extraDenoms.length ? extraDenoms.join(', ') : '(none)';
['USD', ...extraDenoms].forEach(k => {
  const opt=document.createElement('option'); opt.value=k; opt.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(opt);
});

let CURRENT_RAILS=null, locked=false, lockedX=null;
function applyRails(P){
  CURRENT_RAILS = railsFromPercentiles(P);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.FLOOR]},   [0]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P20]},     [1]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P50]},     [2]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P80]},     [3]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.CEILING]}, [4]);
}

function updatePanel(P,xYears){
  elDate.textContent=shortDateFromYears(xYears);
  const F=interp(P.x_grid,CURRENT_RAILS.FLOOR,xYears);
  const v20=interp(P.x_grid,CURRENT_RAILS.P20,xYears);
  const v50=interp(P.x_grid,CURRENT_RAILS.P50,xYears);
  const v80=interp(P.x_grid,CURRENT_RAILS.P80,xYears);
  const C=interp(P.x_grid,CURRENT_RAILS.CEILING,xYears);
  elF.textContent=fmtUSD(F); el20.textContent=fmtUSD(v20); el50.textContent=fmtUSD(v50); el80.textContent=fmtUSD(v80); elC.textContent=fmtUSD(C);
  let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx]; elMain.textContent=fmtUSD(y);
  elP.textContent=`(p≈${pctWithinLog(y,F,C).toFixed(1)}%)`;
  Plotly.relayout(plotDiv,{"yaxis.title.text":P.label});
}

plotDiv.on('plotly_hover', ev=>{
  if(ev.points&&ev.points.length&& !locked){
    const P=PRECOMP[denomSel.value];
    updatePanel(P, ev.points[0].x);
  }
});
setBtn.onclick = ()=>{
  if(!datePick.value) return;
  locked=true; lockedX=yearsFromISO(datePick.value);
  updatePanel(PRECOMP[denomSel.value],lockedX);
};
liveBtn.onclick = ()=>{ locked=false; lockedX=null; };
copyBtn.onclick = async ()=>{
  const node=document.getElementById('capture');
  try{
    const url=await htmlToImage.toPng(node,{pixelRatio:2});
    try{
      if(navigator.clipboard && window.ClipboardItem){
        const blob=await (await fetch(url)).blob();
        await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
        return;
      }
    }catch(e){}
    const a=document.createElement('a'); a.href=url; a.download='btc-indicator.png'; document.body.appendChild(a); a.click(); a.remove();
  }catch(e){ console.error(e); }
};

denomSel.onchange = ()=>{
  const key=denomSel.value, P=PRECOMP[key];
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label]}, [5]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main]}, [6]);
  applyRails(P);
  updatePanel(P,(typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
};

// init with USD
denomSel.value='USD';
applyRails(PRECOMP['USD']);
updatePanel(PRECOMP['USD'],PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);
</script>
</body></html>
""")

# write to disk
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_template.safe_substitute(
        plot_html=plot_html,
        precomp_json=json.dumps(PRECOMP),
        genesis_iso=GENESIS_DATE.strftime("%Y-%m-%d"),
        COL_FLOOR=COL_FLOOR, COL_20=COL_20, COL_50=COL_50, COL_80=COL_80, COL_CEILING=COL_CEILING
    ))
print(f"Wrote {OUTPUT_HTML}")