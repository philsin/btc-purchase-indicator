#!/usr/bin/env python3
"""
BTC Purchase Indicator — Anchor-based Parallel Rails
Ceiling: line through 2011 high & 2017 high (log–log).
Floor:   parallel to ceiling; intercept fit to 2015 low & 2022 low (log–log).
Midlines: 25%, 50% (bold), 75% between floor & ceiling in log-space.

UI:
- Right panel follows cursor unless locked by date.
- p% is BTC's log-space position between floor & ceiling.
- Denominator dropdown; y-axis title updates; odd years after 2026 hidden; no x-axis title.
"""

import os, io, glob, time, math, json
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# ---------------------------- CONFIG ----------------------------

OUTPUT_HTML   = "docs/index.html"
DATA_DIR      = "data"
BTC_FILE      = os.path.join(DATA_DIR, "btc_usd.csv")

GENESIS_DATE  = datetime(2009, 1, 3)
END_PROJ      = datetime(2040, 12, 31)

# Anchor years
CEIL_YEARS = (2011, 2017)  # highs
FLOOR_YEARS = (2015, 2022) # lows

# Colors
LINE_FLOOR   = "#D32F2F"
LINE_25      = "#F57C00"
LINE_50      = "#111111"  # bold/black
LINE_75      = "#2E7D32"
LINE_CEILING = "#6A1B9A"
BTC_COLOR    = "#000000"

# ---------------------------- UTILS -----------------------------

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def _retry(fn, tries=3, base_delay=1.0, factor=2.0):
    last=None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(base_delay*(factor**i))
    raise last

def days_since_genesis(dates, genesis):
    """Vectorized: works for Series/arrays/DatetimeIndex."""
    d = pd.to_datetime(dates)
    s = pd.Series(d)
    delta = s - pd.Timestamp(genesis)
    return (delta / np.timedelta64(1, "D")).astype(float) + 1.0  # +1 avoids log(0)

# ------------------------- DATA LOADERS -------------------------

def _fetch_btc_from_coingecko() -> pd.DataFrame:
    api_key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY")
    if not api_key: raise RuntimeError("COINGECKO_API_KEY not set")
    start = int(datetime(2010,7,17,tzinfo=timezone.utc).timestamp())
    end   = int(datetime.now(timezone.utc).timestamp())
    url = ("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
           f"?vs_currency=usd&from={start}&to={end}")
    def _call():
        r = requests.get(url, headers={"x-cg-pro-api-key": api_key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = [(datetime.utcfromtimestamp(ms/1000.0).date().isoformat(), float(p))
                for ms,p in data.get("prices",[])]
        df = pd.DataFrame(rows, columns=["date","price"]).dropna().sort_values("date")
        if df.empty: raise RuntimeError("CoinGecko returned empty dataset")
        return df
    return _retry(_call)

def _fetch_btc_from_blockchain() -> pd.DataFrame:
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"
    def _call():
        r = requests.get(url, timeout=30); r.raise_for_status()
        raw = r.text.strip()
        if not raw: raise RuntimeError("Blockchain.com empty")
        if raw.splitlines()[0].lower().startswith("timestamp"):
            df = pd.read_csv(io.StringIO(raw))
            ts = [c for c in df.columns if c.lower().startswith("timestamp")][0]
            val= [c for c in df.columns if c.lower().startswith("value")][0]
            df = df.rename(columns={ts:"date", val:"price"})
        else:
            df = pd.read_csv(io.StringIO(raw), header=None, names=["date","price"])
        df["date"]  = pd.to_datetime(df["date"], utc=True).dt.date.astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna().sort_values("date")
        if df.empty: raise RuntimeError("Blockchain.com empty after parse")
        return df
    return _retry(_call)

def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dc, pc = cols.get("date"), cols.get("price")
    if not dc or not pc: raise ValueError(f"{path} needs columns date,price")
    df = df[[dc,pc]].rename(columns={dc:"date", pc:"price"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date"); df = df[df["price"]>0]
    return df.reset_index(drop=True)

def get_btc_df() -> pd.DataFrame:
    if os.path.exists(BTC_FILE): return load_series_csv(BTC_FILE)
    ensure_dir(DATA_DIR)
    df=None
    try:
        if os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY"):
            df=_fetch_btc_from_coingecko()
    except Exception as e:
        print(f"[warn] CG fetch failed: {e}")
    if df is None:
        try: df=_fetch_btc_from_blockchain()
        except Exception as e:
            raise RuntimeError("No BTC data; add data/btc_usd.csv or set COINGECKO_API_KEY") from e
    df.to_csv(BTC_FILE, index=False)
    return load_series_csv(BTC_FILE)

def collect_denominators() -> Dict[str, pd.DataFrame]:
    opts={}
    for p in sorted(glob.glob(os.path.join(DATA_DIR,"denominator_*.csv"))):
        key=os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try: opts[key]=load_series_csv(p)
        except Exception as e: print(f"[warn] bad denom {p}: {e}")
    return opts

# --------------------------- ANCHORS ----------------------------

def pick_anchor_for_year(dates: pd.Series, values: pd.Series, x_days: pd.Series, year: int, mode: str) -> Tuple[float,float]:
    """Return (x_days, value) for max/min within a calendar year."""
    mask = dates.dt.year == year
    sub_idx = np.where(mask)[0]
    if len(sub_idx) == 0:
        raise RuntimeError(f"No data for year {year}; supply data/btc_usd.csv or fetch API.")
    if mode == "max":
        i = sub_idx[np.argmax(values.iloc[sub_idx].values)]
    else:
        i = sub_idx[np.argmin(values.iloc[sub_idx].values)]
    return float(x_days.iloc[i]), float(values.iloc[i])

def line_from_two_points_parallelized(x1, y1, x2, y2) -> Tuple[float,float]:
    """Return intercept a and slope b of a line in log space through two points (x,y)."""
    lx1, ly1 = math.log10(x1), math.log10(y1)
    lx2, ly2 = math.log10(x2), math.log10(y2)
    b = (ly2 - ly1) / (lx2 - lx1)
    a = ly1 - b*lx1
    return a, b

# --------------------------- MODEL ------------------------------

def build_parallel_rails_from_anchors(dates: pd.Series, x_days: pd.Series, y_series: pd.Series,
                                      ceil_years=(2011,2017), floor_years=(2015,2022),
                                      x_grid: np.ndarray=None) -> Dict[str, list]:
    """
    Build rails using anchor highs and lows.
    Ceiling slope b from ceiling anchors; floor intercept fitted (same b) to two lows.
    """
    # Ceiling anchors
    xC1, yC1 = pick_anchor_for_year(dates, y_series, x_days, ceil_years[0], "max")
    xC2, yC2 = pick_anchor_for_year(dates, y_series, x_days, ceil_years[1], "max")
    aC, b = line_from_two_points_parallelized(xC1, yC1, xC2, yC2)

    # Floor intercept (same slope b) fit to lows (least squares in log-space with fixed slope)
    xF1, yF1 = pick_anchor_for_year(dates, y_series, x_days, floor_years[0], "min")
    xF2, yF2 = pick_anchor_for_year(dates, y_series, x_days, floor_years[1], "min")
    lf1 = math.log10(yF1) - b*math.log10(xF1)
    lf2 = math.log10(yF2) - b*math.log10(xF2)
    aF = (lf1 + lf2) / 2.0

    # Rails on grid
    lx = np.log10(x_grid)
    logF = aF + b*lx
    logC = aC + b*lx
    def to_price(logv): return (10**logv).tolist()

    rails = {
        "FLOOR":  to_price(logF),
        "P25":    to_price(logF + 0.25*(logC - logF)),
        "P50":    to_price(logF + 0.50*(logC - logF)),
        "P75":    to_price(logF + 0.75*(logC - logF)),
        "CEILING":to_price(logC),
        "params": {"aF":aF, "aC":aC, "b":b,
                   "anchors":{"ceil":[[xC1,yC1],[xC2,yC2]], "floor":[[xF1,yF1],[xF2,yF2]]}}
    }
    return rails

def rebase_to_one(s: pd.Series) -> pd.Series:
    s=pd.Series(s).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if s.empty or s.iloc[0]<=0: return pd.Series(s)*np.nan
    return pd.Series(s)/s.iloc[0]

# -------------------------- LOAD DATA ---------------------------

btc = get_btc_df().rename(columns={"price":"btc"})
denoms = collect_denominators()

base = btc.sort_values("date").reset_index(drop=True)
if base.empty: raise RuntimeError("No BTC data found")
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

base["x_days"]   = days_since_genesis(base["date"], GENESIS_DATE)
base["date_iso"] = base["date"].dt.strftime("%Y-%m-%d")
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# X grid (log-spaced) from first data point to END_PROJ
x_start = float(base["x_days"].iloc[0])
x_end   = float(days_since_genesis(pd.Series([END_PROJ]), GENESIS_DATE).iloc[0])
x_grid  = np.logspace(np.log10(max(1.0, x_start)), np.log10(x_end), 600)

# Year ticks mapped to log(days); hide odd years > 2026
def year_ticks_log(first_date: datetime, last_date: datetime):
    y0, y1 = first_date.year, last_date.year
    vals, labs = [], []
    for y in range(y0, y1+1):
        d = datetime(y, 1, 1)
        if d < first_date or d > last_date: continue
        dv = float(days_since_genesis(pd.Series([d]), GENESIS_DATE).iloc[0])
        if dv <= 0: continue
        if y > 2026 and (y % 2 == 1):  # hide odd years after 2026
            continue
        vals.append(dv); labs.append(str(y))
    return vals, labs
first_date = base["date"].iloc[0].to_pydatetime()
xtickvals, xticktext = year_ticks_log(first_date, END_PROJ)

# -------------------- PRECOMPUTE PER DENOM ---------------------

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=denom_key.lower()
    if k in df.columns:
        return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def build_payload(denom_key: Optional[str]):
    y_main, y_label, denom_series = series_for_denom(base, denom_key)

    # Build rails from anchors on current series (auto highs/lows for the specified years)
    rails = build_parallel_rails_from_anchors(
        dates=base["date"], x_days=base["x_days"], y_series=y_main,
        ceil_years=CEIL_YEARS, floor_years=FLOOR_YEARS, x_grid=x_grid
    )

    # p% at latest observed date
    last_x = float(base["x_days"].iloc[-1])
    # interpolate floor/ceiling at last_x from grid
    def interp(xa, ya, x):
        xa = np.asarray(xa); ya=np.asarray(ya)
        if x<=xa[0]: return float(ya[0])
        if x>=xa[-1]: return float(ya[-1])
        i = np.searchsorted(xa, x) - 1
        t = (x - xa[i])/(xa[i+1]-xa[i])
        return float(ya[i] + t*(ya[i+1]-ya[i]))
    F_last = interp(x_grid, rails["FLOOR"], last_x)
    C_last = interp(x_grid, rails["CEILING"], last_x)
    y_last = float(y_main.iloc[-1])

    def pos_pct_log(y,f,c):
        ly, lf, lc = math.log10(y), math.log10(f), math.log10(c)
        return max(0.0, min(100.0, 100.0*(ly-lf)/max(1e-12, lc-lf)))
    p_init = pos_pct_log(y_last, F_last, C_last)

    return {
        "label": y_label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "rails": {k: (v if isinstance(v,list) else v) for k,v in rails.items() if k!="params"},
        "params": rails["params"],
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "p_init": p_init,
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

# -------------------------- FIGURE ------------------------------

traces=[]; vis_norm=[]; vis_cmp=[]

def add_line_on_grid(y_list, name, color, width=1.2, dash=None, bold=False):
    line=dict(width=2.2 if bold else width, color=color)
    if dash: line["dash"]=dash
    traces.append(go.Scatter(
        x=init["x_grid"], y=y_list, mode="lines", name=name,
        line=line, hoverinfo="skip", showlegend=True
    ))
    vis_norm.append(True); vis_cmp.append(False)

# Draw rails in order: Floor, 25%, 50% (bold), 75%, Ceiling
add_line_on_grid(init["rails"]["FLOOR"], "Floor",   LINE_FLOOR,   width=1.2)
add_line_on_grid(init["rails"]["P25"],   "25%",     LINE_25,      width=1.2, dash="dot")
add_line_on_grid(init["rails"]["P50"],   "50%",     LINE_50,      bold=True)
add_line_on_grid(init["rails"]["P75"],   "75%",     LINE_75,      width=1.2, dash="dot")
add_line_on_grid(init["rails"]["CEILING"], "Ceiling", LINE_CEILING, width=1.2)

# BTC price (black)
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="BTC / USD", line=dict(width=1.9, color=BTC_COLOR),
                         hoverinfo="skip"))
vis_norm.append(True); vis_cmp.append(False)

# Transparent cursor trace to guarantee hover events
traces.append(go.Scatter(
    x=init["x_main"], y=init["y_main"], mode="lines",
    line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor"
))
vis_norm.append(True); vis_cmp.append(False)

# Compare-mode lines (rebased)
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False, line=dict(color=BTC_COLOR)))
traces.append(go.Scatter(x=init["x_main"], y=init["denom_rebased"], name="Denominator (rebased)",
                         mode="lines", line=dict(dash="dash"), hoverinfo="skip", visible=False))
vis_norm.extend([False, False]); vis_cmp.extend([True, True])

fig = go.Figure(data=traces)

# Y ticks: faux "0" then powers of 10 with commas
def make_y_ticks(max_y: float):
    exp_max = int(math.ceil(math.log10(max(1.0, max_y))))
    vals=[1e-8] + [10**e for e in range(0, exp_max+1)]
    texts=["0"] + [f"{int(10**e):,}" for e in range(0, exp_max+1)]
    return vals, texts

y_candidates = [np.nanmax(init["y_main"])] + [
    np.nanmax(init["rails"][k]) for k in ["FLOOR","P25","P50","P75","CEILING"]
]
y_max = max(y_candidates)
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x",  # keep events for live hover
    hoverdistance=30, spikedistance=30,
    title=f"BTC Purchase Indicator — Anchor Rails (p≈{init['p_init']:.1f}%)",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=init["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=400, t=70, b=70),
)

# ---------------------- HTML (panel + copy + date lock) ----------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)  # JSON-safe
genesis_iso    = GENESIS_DATE.strftime("%Y-%m-%d")

html_tpl = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root { --panelW: 400px; }
  body { font-family: Inter, Roboto, -apple-system, Segoe UI, Arial, sans-serif; margin: 0; }
  .layout { display: grid; grid-template-columns: 1fr var(--panelW); height: 100vh; }
  .left { padding: 8px 0 8px 8px; }
  .right { border-left: 1px solid #e5e7eb; padding: 12px; display: flex; flex-direction: column; gap: 12px; }

  #controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #controls label { white-space: nowrap; }
  #controls select, #controls button, #controls input[type="date"] {
    font-size:14px; padding:8px 10px; border-radius:8px; border:1px solid #d1d5db; background:white;
  }
  #copyBtn { cursor:pointer; }

  #readout { border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; font-size:14px; }
  #readout .date { font-weight:700; margin-bottom:6px; }
  #readout .row { display:grid; grid-template-columns: auto 1fr auto; column-gap:8px; align-items:baseline; }
  #readout .num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
                  font-variant-numeric: tabular-nums; text-align:right; min-width: 12ch; white-space:pre; }

  /* Keep hover events alive but hide the tooltip */
  .hoverlayer { opacity: 0 !important; pointer-events: none; }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; height: auto; }
    .right { border-left:none; border-top:1px solid #e5e7eb; }
    .left .js-plotly-plot { max-width: 100vw; }
  }
</style>
</head>
<body>
<div id="capture" class="layout">
  <div class="left">$PLOT_HTML</div>
  <div class="right">
    <div id="controls">
      <label for="denomSel"><b>Denominator:</b></label>
      <select id="denomSel"></select>

      <input type="date" id="datePick"/>
      <button id="setDateBtn" title="Lock panel to selected date">Set Date</button>
      <button id="liveBtn" title="Follow cursor">Live Hover</button>

      <button id="copyBtn">Copy Chart</button>
    </div>

    <div style="font-size:12px;color:#6b7280;">
      Detected denominators: <span id="denomsDetected"></span>
    </div>

    <div id="readout">
      <div class="date">—</div>
      <div class="row"><div><span style="color:$COL_F;">Floor</span></div><div class="num" id="vF">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_25;">25%</span></div><div class="num" id="v25">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_50;font-weight:700;">50%</span></div><div class="num" id="v50" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_75;">75%</span></div><div class="num" id="v75">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_C;">Ceiling</span></div><div class="num" id="vC">$0.00</div><div></div></div>

      <div style="margin-top:10px;"><b>BTC Price:</b> <span class="num" id="mainVal">$0.00</span></div>
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈$P_INIT%)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = $PRECOMP_JSON;
const GENESIS_ISO  = "$GENESIS_ISO";

const denomSel = document.getElementById('denomSel');
const detected = Object.keys(PRECOMP).filter(k => k !== 'USD');
document.getElementById('denomsDetected').textContent = detected.length ? detected.join(', ') : '(none)';
['USD', ...detected].forEach(function(k){
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = (k === 'USD') ? 'USD/None' : k;
  denomSel.appendChild(opt);
});

function fmtUSD(v){
  if(!isFinite(v)) return '$—';
  return '$' + Number(v).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
}
function shortMonthName(m){ return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m]; }
function dateFromDaysShort(x){
  const d0=new Date(GENESIS_ISO+'T00:00:00Z');
  const d=new Date(d0.getTime() + (x-1)*86400000);
  return `${shortMonthName(d.getUTCMonth())}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`;
}
function interp(xArr,yArr,x){
  let lo=0,hi=xArr.length-1;
  if(x<=xArr[0]) return yArr[0];
  if(x>=xArr[hi]) return yArr[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xArr[m]<=x) lo=m; else hi=m; }
  const t=(x-xArr[lo])/(xArr[hi]-xArr[lo]); return yArr[lo]+t*(yArr[hi]-yArr[lo]);
}
function positionPctLog(y, f, c){ // y price, f floor, c ceiling -> 0..100 in log space
  const ly=Math.log10(y), lf=Math.log10(f), lc=Math.log10(c);
  return Math.max(0, Math.min(100, 100*(ly-lf)/Math.max(1e-12, lc-lf)));
}

// Panel elements
const elDate=document.querySelector('#readout .date');
const elF=document.getElementById('vF'), el25=document.getElementById('v25'),
      el50=document.getElementById('v50'), el75=document.getElementById('v75'), elC=document.getElementById('vC');
const elMain=document.getElementById('mainVal'), elPPct=document.getElementById('pPct');

let locked=false, lockedX=null;

function updatePanel(den,xDays){
  const P=PRECOMP[den];
  elDate.textContent = dateFromDaysShort(xDays);

  const F=interp(P.x_grid,P.rails.FLOOR,xDays);
  const v25=interp(P.x_grid,P.rails.P25,xDays);
  const v50=interp(P.x_grid,P.rails.P50,xDays);
  const v75=interp(P.x_grid,P.rails.P75,xDays);
  const C=interp(P.x_grid,P.rails.CEILING,xDays);

  elF.textContent=fmtUSD(F); el25.textContent=fmtUSD(v25); el50.textContent=fmtUSD(v50);
  el75.textContent=fmtUSD(v75); elC.textContent=fmtUSD(C);

  // nearest observed price
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-xDays); if(d<best){best=d; idx=i;} }
  const y=ya[idx]; elMain.textContent = fmtUSD(y);

  const p = positionPctLog(y,F,C);
  elPPct.textContent = `(p≈${p.toFixed(1)}%)`;

  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — Anchor Rails (p≈${p.toFixed(1)}%)`});
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

  // Live hover ON (tooltip visually hidden via CSS)
  plotDiv.on('plotly_hover', function(ev){
    if(!ev.points||!ev.points.length) return;
    if(locked) return;
    updatePanel(denomSel.value, ev.points[0].x); // x is days (log axis)
  });

  document.getElementById('setDateBtn').addEventListener('click', function(){
    const val=document.getElementById('datePick').value;
    if(!val) return;
    const d=new Date(val+'T00:00:00Z');
    const d0=new Date(GENESIS_ISO+'T00:00:00Z');
    const xDays=((d.getTime()-d0.getTime())/86400000)+1.0;
    locked=true; lockedX=xDays;
    updatePanel(denomSel.value, xDays);
  });

  document.getElementById('liveBtn').addEventListener('click', function(){
    locked=false; lockedX=null;
  });

  // Copy Chart: try clipboard; otherwise download silently
  document.getElementById('copyBtn').addEventListener('click', async function(){
    const node=document.getElementById('capture');
    try{
      const dataUrl=await htmlToImage.toPng(node,{pixelRatio:2});
      try{
        if(navigator.clipboard && window.ClipboardItem){
          const blob=await (await fetch(dataUrl)).blob();
          await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
          return;
        }
      }catch(e){}
      const a=document.createElement('a'); a.href=dataUrl; a.download='btc-indicator.png';
      document.body.appendChild(a); a.click(); a.remove();
    }catch(e){ console.error(e); }
  });

  // Denominator change: restyle + relabel axis + reset panel
  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];

    function restyleLine(traceIdx, arr){
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[arr] }, [traceIdx]);
    }
    // Order: Floor(0), 25%(1), 50%(2), 75%(3), Ceiling(4), BTC(5), cursor(6), rebased(7,8)
    restyleLine(0, P.rails.FLOOR);
    restyleLine(1, P.rails.P25);
    restyleLine(2, P.rails.P50);
    restyleLine(3, P.rails.P75);
    restyleLine(4, P.rails.CEILING);

    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [5]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main] }, [6]); // cursor
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.main_rebased] }, [7]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.denom_rebased] }, [8]);

    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});

    const xTarget = locked ? lockedX : P.x_main[P.x_main.length-1];
    updatePanel(key, xTarget);
  });

  // Initialize panel at latest point
  updatePanel('USD', PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);
});
</script>
</body>
</html>
""")

html = html_tpl.safe_substitute(
    PLOT_HTML=plot_html,
    PRECOMP_JSON=precomp_json,
    GENESIS_ISO=genesis_iso,
    COL_F=LINE_FLOOR, COL_25=LINE_25, COL_50=LINE_50, COL_75=LINE_75, COL_C=LINE_CEILING,
    P_INIT=f"{init['p_init']:.1f}",
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
