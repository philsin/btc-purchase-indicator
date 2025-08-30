#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# BTC Purchase Indicator — Midline Percentiles (flex layout + width slider +
# symmetric/winsorized rails so ceiling/floor don't explode)
# ─────────────────────────────────────────────────────────────────────────────
import os, io, glob, time, json
from datetime import datetime, timezone
from string import Template
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ---------- Config ----------
OUTPUT_HTML   = "docs/index.html"
DATA_DIR      = "data"
BTC_FILE      = os.path.join(DATA_DIR, "btc_usd.csv")

GENESIS_DATE  = datetime(2009, 1, 3)
END_PROJ      = datetime(2040, 12, 31)

EPS_LOG         = 0.010       # tiny spacing so rails never touch (≈2.3%)
RESID_WINSOR    = 0.02        # clip extreme residuals at 2% tails before percentiles
SYMMETRIC_RAILS = True        # mirror upper/lower rails about median residual

# Colors (ramp red→green)
COL_FLOOR   = "#D32F2F"
COL_20      = "#F57C00"
COL_50      = "#FBC02D"
COL_80      = "#66BB6A"
COL_CEILING = "#2E7D32"
COL_BTC     = "#000000"

# ---------- Helpers ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def years_since_genesis(dates, genesis=GENESIS_DATE):
    d = pd.to_datetime(dates)
    s = pd.Series(d)
    delta_days = (s - pd.Timestamp(genesis)) / np.timedelta64(1, "D")
    return (delta_days.astype(float) / 365.25) + (1.0/365.25)  # avoid log(0)

def _retry(fn, tries=3, base=1.0, factor=2.0):
    last=None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(base*(factor**i))
    raise last

def _fetch_btc_from_coingecko() -> pd.DataFrame:
    key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY")
    if not key: raise RuntimeError("COINGECKO_API_KEY not set")
    start = int(datetime(2010,7,17,tzinfo=timezone.utc).timestamp())
    end   = int(datetime.now(timezone.utc).timestamp())
    url = ("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
           f"?vs_currency=usd&from={start}&to={end}")
    def call():
        r = requests.get(url, headers={"x-cg-pro-api-key": key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = [(datetime.utcfromtimestamp(ms/1000.0), float(p)) for ms,p in data.get("prices",[])]
        df = pd.DataFrame(rows, columns=["date","price"]).dropna().sort_values("date")
        return df
    return _retry(call)

def _fetch_btc_from_blockchain() -> pd.DataFrame:
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    def call():
        r = requests.get(url, timeout=30); r.raise_for_status()
        raw = r.text.strip()
        if raw.splitlines()[0].lower().startswith("timestamp"):
            df = pd.read_csv(io.StringIO(raw))
            ts = [c for c in df.columns if c.lower().startswith("timestamp")][0]
            val= [c for c in df.columns if c.lower().startswith("value")][0]
            df = df.rename(columns={ts:"date", val:"price"})
        else:
            df = pd.read_csv(io.StringIO(raw), header=None, names=["date","price"])
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df["price"]= pd.to_numeric(df["price"], errors="coerce")
        return df.dropna().sort_values("date")
    return _retry(call)

def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dc, pc = cols.get("date"), cols.get("price")
    if not dc or not pc: raise ValueError(f"{path} must have date,price columns")
    df = df[[dc,pc]].rename(columns={dc:"date", pc:"price"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date")
    df = df[df["price"]>0]
    return df.reset_index(drop=True)

def get_btc_df() -> pd.DataFrame:
    if os.path.exists(BTC_FILE): return load_series_csv(BTC_FILE)
    ensure_dir(DATA_DIR)
    df=None
    try:
        if os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY"):
            df=_fetch_btc_from_coingecko()
    except Exception as e:
        print("[warn] CoinGecko failed:", e)
    if df is None:
        df=_fetch_btc_from_blockchain()
    df.to_csv(BTC_FILE, index=False)
    return load_series_csv(BTC_FILE)

def collect_denominators():
    out={}
    for p in glob.glob(os.path.join(DATA_DIR,"denominator_*.csv")):
        key=os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try:
            out[key]=load_series_csv(p)
        except Exception as e:
            print(f"[warn] bad denom {p}: {e}")
    return out

# ---------- Fitting ----------
def quantile_fit_loglog(x_years: np.ndarray, y: np.ndarray, q=0.5):
    m = (x_years>0)&(y>0)&np.isfinite(x_years)&np.isfinite(y)
    x = np.log10(x_years[m]); z = np.log10(y[m])
    X = pd.DataFrame({"const":1.0,"logx":x})
    res = QuantReg(z,X).fit(q=q)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = z - (a0 + b*x)  # log residuals about the q50 midline
    return a0,b,resid,m

def winsorize(arr, p):
    lo, hi = np.nanquantile(arr, p), np.nanquantile(arr, 1-p)
    return np.clip(arr, lo, hi)

def symmetric_quantiles(resid, q_low, q_high):
    """Return (low, high) as symmetric offsets about the median residual."""
    med = float(np.nanmedian(resid))
    # distances from median
    d_hi = float(np.nanquantile(resid - med, q_high))
    d_lo = float(np.nanquantile(med - resid, q_high))
    d = max(d_hi, d_lo)
    return med - d, med + d

def defaults_for_series(dates, x_years, y_series):
    a0,b,resid,_ = quantile_fit_loglog(x_years.values, y_series.values, q=0.5)

    r = resid.copy()
    if RESID_WINSOR and RESID_WINSOR>0:
        r = winsorize(r, RESID_WINSOR)

    if SYMMETRIC_RAILS:
        # Symmetric 2.5/97.5 and 20/80 around median residual
        c025, c975 = symmetric_quantiles(r, 0.025, 0.975)
        c200, c800 = symmetric_quantiles(r, 0.200, 0.800)
    else:
        c025 = float(np.nanquantile(r, 0.025))
        c975 = float(np.nanquantile(r, 0.975))
        c200 = float(np.nanquantile(r, 0.200))
        c800 = float(np.nanquantile(r, 0.800))

    # keep rails from touching
    c025 -= EPS_LOG
    c975 += EPS_LOG

    return {"a0":a0,"b":b,"c025":c025,"c200":c200,"c800":c800,"c975":c975}

# ---------- Data prep ----------
btc = get_btc_df().rename(columns={"price":"btc"})
denoms = collect_denominators()

base = btc.sort_values("date").reset_index(drop=True)
for name,df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

base["x_years"]  = years_since_genesis(base["date"])
base["date_iso"] = base["date"].dt.strftime("%Y-%m-%d")

# grid for rails
x_start = float(base["x_years"].iloc[0])
x_end   = float(years_since_genesis(pd.Series([END_PROJ]), GENESIS_DATE).iloc[0])
x_grid  = np.logspace(np.log10(max(1e-6,x_start)), np.log10(x_end), 700)

def year_ticks_log(first_dt: datetime, last_dt: datetime):
    vals,labs = [],[]
    y0,y1 = first_dt.year, last_dt.year
    for y in range(y0,y1+1):
        d = datetime(y,1,1)
        if d<first_dt or d>last_dt: continue
        vy = float(years_since_genesis(pd.Series([d]), GENESIS_DATE).iloc[0])
        if vy<=0: continue
        if y>2026 and y%2==1:  # hide odd years after 2026
            continue
        vals.append(vy); labs.append(str(y))
    return vals,labs
first_dt = base["date"].iloc[0].to_pydatetime()
xtickvals, xticktext = year_ticks_log(first_dt, END_PROJ)

def series_for_denom(df, key):
    if not key or key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=key.lower()
    if k in df.columns:
        return df["btc"]/df[k], f"BTC / {key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def build_payload(key=None):
    y,label,_ = series_for_denom(base, key)
    d = defaults_for_series(base["date"], base["x_years"], y)
    return {
        "label": label,
        "x_main": base["x_years"].tolist(),
        "y_main": y.tolist(),
        "date_iso_main": base["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "defaults": d
    }

PRECOMP={"USD":build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k]=build_payload(k)
P0 = PRECOMP["USD"]

# ---------- Plot skeleton ----------
def add_line_stub(name, color, width=1.6, dash=None, bold=False):
    line=dict(width=2.6 if bold else width, color=color)
    if dash: line["dash"]=dash
    return go.Scatter(x=P0["x_grid"], y=[None]*len(P0["x_grid"]),
                      mode="lines", name=name, line=line, hoverinfo="skip")

traces = [
    add_line_stub("Floor",   COL_FLOOR),
    add_line_stub("20%",     COL_20, dash="dot"),
    add_line_stub("50%",     COL_50, bold=True),
    add_line_stub("80%",     COL_80, dash="dot"),
    add_line_stub("Ceiling", COL_CEILING),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               name="BTC / USD", line=dict(color=COL_BTC,width=2.0), hoverinfo="skip"),
    go.Scatter(x=P0["x_main"], y=P0["y_main"], mode="lines",
               line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor")
]

# y tick labels with commas (include 0 label)
def y_ticks():
    vals=[1e-8]+[10**e for e in range(0,9)]
    labs=["0"]+[f"{int(10**e):,}" for e in range(0,9)]
    return vals,labs
ytickvals, yticktext = y_ticks()

fig = go.Figure(traces)
fig.update_layout(
    template="plotly_white",
    hovermode="x",
    showlegend=True,
    title="BTC Purchase Indicator — Rails",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=P0["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    margin=dict(l=70, r=420, t=70, b=70),  # generous plotting space
)

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive":True,"displayModeBar":True,"modeBarButtonsToRemove":["toImage"]})

# ---------- HTML + JS (flex + ResizeObserver + working width slider) ----------
html_tpl = Template(r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
:root{--panelW:420px; --chartPct:72%} /* default chart width percentage */
html,body{height:100%}body{margin:0;font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
.layout{display:flex;min-height:100vh;width:100vw}
.left{flex: 1 1 var(--chartPct); min-width: 280px; padding:8px 0 8px 8px}
.left .js-plotly-plot,.left .plotly-graph-div{width:100%!important}
.right{flex: 0 0 var(--panelW); border-left:1px solid #e5e7eb; padding:12px; display:flex; flex-direction:column; gap:12px; overflow:auto}
#controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,button,input[type=date],input[type=range]{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}
#readout{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fafafa;font-size:14px}
#readout .date{font-weight:700;margin-bottom:6px}
#readout .row{display:grid;grid-template-columns:auto 1fr auto;column-gap:8px;align-items:baseline}
#readout .num{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;text-align:right;min-width:12ch;white-space:pre}
.hoverlayer{opacity:0!important;pointer-events:none}
@media (max-width: 900px){
  .layout{flex-direction:column}
  .right{flex: 0 0 auto; border-left:none; border-top:1px solid #e5e7eb}
  .left{flex: 0 0 auto; width:100%; padding:8px}
}
#chartWidthBox{display:flex;align-items:center;gap:8px}
#chartW{width:220px}
#chartWVal{min-width:3ch;text-align:right}
</style>
</head><body>
<div id="capture" class="layout">
  <div class="left">$PLOT</div>
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
      <div class="row"><div><span style="color:$COL_F;">Floor</span></div><div id="vF" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_20;">20%</span></div><div id="v20" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_50;font-weight:700;">50%</span></div><div id="v50" class="num" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_80;">80%</span></div><div id="v80" class="num">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_C;">Ceiling</span></div><div id="vC" class="num">$0.00</div><div></div></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span id="mainVal" class="num">$0.00</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>
<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP = $PRECOMP_JSON;
function fmtUSD(v){ if(!isFinite(v)) return '$—'; return '$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const GENESIS = new Date('$GENESIS_ISO'+'T00:00:00Z');

function yearsFromISO(iso){
  const d=new Date(iso+'T00:00:00Z');
  return ((d-GENESIS)/86400000)/365.25 + (1.0/365.25);
}
function shortDateFromYears(y){
  const ms = (y - (1.0/365.25))*365.25*86400000;
  const d=new Date(GENESIS.getTime()+ms);
  return `${MONTHS[d.getUTCMonth()]}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`;
}
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

// Rails from midline + residual offsets (parallel)
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

// DOM
const plotDiv=document.querySelector('.left .js-plotly-plot') || document.querySelector('.left .plotly-graph-div');
const denomSel=document.getElementById('denomSel');
const datePick=document.getElementById('datePick');
const setBtn=document.getElementById('setDateBtn');
const liveBtn=document.getElementById('liveBtn');
const copyBtn=document.getElementById('copyBtn');
const elDenoms=document.getElementById('denomsDetected');
const elDate=document.querySelector('#readout .date');
const elF=document.getElementById('vF'), el20=document.getElementById('v20'), el50=document.getElementById('v50'),
      el80=document.getElementById('v80'), elC=document.getElementById('vC'), elMain=document.getElementById('mainVal'),
      elP=document.getElementById('pPct');
const chartW=document.getElementById('chartW');
const chartWVal=document.getElementById('chartWVal');
const layoutRoot=document.querySelector('.layout');
const leftPane=document.querySelector('.left');

// Make slider actually change the chart column and trigger resize
function applyChartWidth(pct){
  document.documentElement.style.setProperty('--chartPct', pct+'%');
  chartWVal.textContent=pct+'%';
  if(window.Plotly && plotDiv){ Plotly.Plots.resize(plotDiv); }
}
chartW.addEventListener('input', ()=>applyChartWidth(chartW.value));
applyChartWidth(chartW.value);

// Also observe width changes (mobile orientation, panel open/close, etc.)
if(window.ResizeObserver){
  const ro=new ResizeObserver(()=>{ if(window.Plotly&&plotDiv) Plotly.Plots.resize(plotDiv); });
  ro.observe(leftPane);
}

const extraDenoms = Object.keys(PRECOMP).filter(k=>k!=='USD');
elDenoms.textContent = extraDenoms.length ? extraDenoms.join(', ') : '(none)';
['USD', ...extraDenoms].forEach(k => {
  const o=document.createElement('option'); o.value=k; o.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(o);
});

let CURRENT_RAILS=null, locked=false, lockedX=null;

function applyRails(P){
  CURRENT_RAILS = railsFromPercentiles(P);
  // traces: 0..4 rails, 5 main, 6 cursor
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.FLOOR]},   [0]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P20]},     [1]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P50]},     [2]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P80]},     [3]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.CEILING]}, [4]);
}

function updatePanel(P, xYears){
  elDate.textContent = shortDateFromYears(xYears);
  const F = interp(P.x_grid, CURRENT_RAILS.FLOOR, xYears);
  const v20=interp(P.x_grid, CURRENT_RAILS.P20,   xYears);
  const v50=interp(P.x_grid, CURRENT_RAILS.P50,   xYears);
  const v80=interp(P.x_grid, CURRENT_RAILS.P80,   xYears);
  const C = interp(P.x_grid, CURRENT_RAILS.CEILING, xYears);
  elF.textContent=fmtUSD(F); el20.textContent=fmtUSD(v20); el50.textContent=fmtUSD(v50);
  el80.textContent=fmtUSD(v80); elC.textContent=fmtUSD(C);

  let idx=0,best=1e99; for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx]; elMain.textContent=fmtUSD(y);
  elP.textContent = `(p≈${pctWithinLog(y,F,C).toFixed(1)}%)`;
  Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});
}

plotDiv.on('plotly_hover', ev=>{
  if(!ev.points||!ev.points.length) return;
  if(locked) return;
  const P=PRECOMP[denomSel.value];
  updatePanel(P, ev.points[0].x);
});

setBtn.onclick = ()=>{
  if(!datePick.value) return;
  locked=true; lockedX=yearsFromISO(datePick.value);
  updatePanel(PRECOMP[denomSel.value], lockedX);
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
  Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [5]);
  Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main] }, [6]); // cursor
  applyRails(P);
  updatePanel(P, (typeof lockedX==='number')?lockedX:P.x_main[P.x_main.length-1]);
};

// init
denomSel.value='USD';
applyRails(PRECOMP['USD']);
updatePanel(PRECOMP['USD'], PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);
</script>
</body></html>
""")

html = html_tpl.safe_substitute(
    PLOT=plot_html,
    PRECOMP_JSON=json.dumps(PRECOMP),
    GENESIS_ISO=GENESIS_DATE.strftime("%Y-%m-%d"),
    COL_F=COL_FLOOR, COL_20=COL_20, COL_50=COL_50, COL_80=COL_80, COL_C=COL_CEILING
)

ensure_dir(os.path.dirname(OUTPUT_HTML))
with open(OUTPUT_HTML,"w",encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {OUTPUT_HTML}")