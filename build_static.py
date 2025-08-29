#!/usr/bin/env python3
"""
BTC Purchase Indicator — Anchor Chooser (client-side rails)

Default rails:
- Ceiling: auto highs in 2011 & 2017 (log–log).
- Floor: earliest 2013 price & 2015-10-01 price, parallel to ceiling (log–log).
Midlines: 25%, 50% (bold), 75% between floor & ceiling in log-space.

New:
- Right-panel "Anchors" UI to pick ceiling & floor anchors interactively:
  * Ceiling: two YEARS (Highs).
  * Floor: choose one of:
      (A) 2013-start + specific date (default 2015-10-01),
      (B) Two YEARS (Lows),
      (C) Two custom DATES.
  * Apply Anchors recomputes rails instantly in JS (no rebuild).
- Denominator changes reapply current anchor settings to that series.

Other behavior:
- p% = BTC’s log-space position between floor & ceiling at cursor (or locked date).
- BTC line black; y-axis title updates with denominator; odd years after 2026 hidden; no x-axis title.
- Copy Chart: tries clipboard, else downloads PNG.
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

# Default ceiling years (you can change in UI later)
CEIL_YEARS_DEFAULT = (2011, 2017)

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

# -------------------------- BUILD BASE --------------------------

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

# -------------------- PRECOMPUTE PER DENOM (data only) ---------

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=denom_key.lower()
    if k in df.columns:
        return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def build_payload(denom_key: Optional[str]):
    y_main, y_label, denom_series = series_for_denom(base, denom_key)
    return {
        "label": y_label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "date_iso_main": base["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "main_rebased": (y_main / y_main.iloc[0]).tolist(),
        "denom_rebased": (denom_series/denom_series.iloc[0]).tolist() if denom_series is not None else [math.nan]*len(base),
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

# -------------------------- FIGURE (initial rails = defaults) ---

# We’ll draw empty rails now; JS will compute and restyle immediately based on defaults.
traces=[]; vis_norm=[]; vis_cmp=[]

def add_line_on_grid(name, color, width=1.2, dash=None, bold=False):
    line=dict(width=2.2 if bold else width, color=color)
    if dash: line["dash"]=dash
    traces.append(go.Scatter(
        x=init["x_grid"], y=[None]*len(init["x_grid"]), mode="lines", name=name,
        line=line, hoverinfo="skip", showlegend=True
    ))
    vis_norm.append(True); vis_cmp.append(False)

# Order: Floor, 25%, 50%(bold), 75%, Ceiling
add_line_on_grid("Floor",   LINE_FLOOR)
add_line_on_grid("25%",     LINE_25, dash="dot")
add_line_on_grid("50%",     LINE_50, bold=True)
add_line_on_grid("75%",     LINE_75, dash="dot")
add_line_on_grid("Ceiling", LINE_CEILING)

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
    exp_max = 8  # just to seed; JS will adjust panel values dynamically
    vals=[1e-8] + [10**e for e in range(0, exp_max+1)]
    texts=["0"] + [f"{int(10**e):,}" for e in range(0, exp_max+1)]
    return vals, texts

ytickvals, yticktext = make_y_ticks(1)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x",
    hoverdistance=30, spikedistance=30,
    title=f"BTC Purchase Indicator — Anchor Rails",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=init["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=420, t=70, b=70),
)

# ---------------------- HTML (+ Anchors UI) ---------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json = json.dumps(PRECOMP)
genesis_iso  = GENESIS_DATE.strftime("%Y-%m-%d")

html_tpl = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root { --panelW: 420px; }
  body { font-family: Inter, Roboto, -apple-system, Segoe UI, Arial, sans-serif; margin: 0; }
  .layout { display: grid; grid-template-columns: 1fr var(--panelW); height: 100vh; }
  .left { padding: 8px 0 8px 8px; }
  .right { border-left: 1px solid #e5e7eb; padding: 12px; display: flex; flex-direction: column; gap: 12px; overflow:auto; }

  #controls, #anchors { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #controls label, #anchors label { white-space: nowrap; }
  select, button, input[type="date"], input[type="number"] {
    font-size:14px; padding:8px 10px; border-radius:8px; border:1px solid #d1d5db; background:white;
  }
  fieldset { border:1px solid #e5e7eb; border-radius:10px; padding:10px; }
  legend { font-weight:700; font-size:13px; color:#374151; }

  #readout { border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; font-size:14px; }
  #readout .date { font-weight:700; margin-bottom:6px; }
  #readout .row { display:grid; grid-template-columns: auto 1fr auto; column-gap:8px; align-items:baseline; }
  #readout .num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
                  font-variant-numeric: tabular-nums; text-align:right; min-width: 12ch; white-space:pre; }

  .hoverlayer { opacity: 0 !important; pointer-events: none; } /* keep live hover but hide tooltip */

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

    <fieldset id="anchors">
      <legend>Anchors</legend>
      <div style="display:flex; flex-direction:column; gap:8px; width:100%;">
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <b>Ceiling (Highs):</b>
          <label>Year 1 <input type="number" id="ceilY1" value="2011" min="2010" max="2100" style="width:90px"/></label>
          <label>Year 2 <input type="number" id="ceilY2" value="2017" min="2010" max="2100" style="width:90px"/></label>
        </div>

        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <b>Floor:</b>
          <label><input type="radio" name="floorMode" value="A" checked/> 2013-start + Date</label>
          <label><input type="radio" name="floorMode" value="B"/> Lows in two Years</label>
          <label><input type="radio" name="floorMode" value="C"/> Two Dates</label>
        </div>

        <div id="floorA" style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <span>2013 earliest + </span>
          <label>Date <input type="date" id="floorA_date" value="2015-10-01"/></label>
        </div>

        <div id="floorB" style="display:none; gap:10px; align-items:center; flex-wrap:wrap;">
          <label>Year 1 <input type="number" id="floorB_y1" value="2015" min="2010" max="2100" style="width:90px"/></label>
          <label>Year 2 <input type="number" id="floorB_y2" value="2022" min="2010" max="2100" style="width:90px"/></label>
          <span>(use <i>lows</i>)</span>
        </div>

        <div id="floorC" style="display:none; gap:10px; align-items:center; flex-wrap:wrap;">
          <label>Date 1 <input type="date" id="floorC_d1" value="2013-01-01"/></label>
          <label>Date 2 <input type="date" id="floorC_d2" value="2015-10-01"/></label>
        </div>

        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          <button id="applyAnchors">Apply Anchors</button>
          <span style="font-size:12px;color:#6b7280;">Rails are straight & parallel; midlines at 25/50/75.</span>
        </div>
      </div>
    </fieldset>

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
      <div><b>Position:</b> <span id="pPct" style="font-weight:600;">(p≈—)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = $PRECOMP_JSON;
const GENESIS_ISO  = "$GENESIS_ISO";

// ---------- helpers ----------
function fmtUSD(v){ if(!isFinite(v)) return '$—'; return '$' + Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
function shortMonthName(m){ return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m]; }
function dateFromDaysShort(x){
  const d0=new Date(GENESIS_ISO+'T00:00:00Z');
  const d=new Date(d0.getTime() + (x-1)*86400000);
  return `${shortMonthName(d.getUTCMonth())}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`;
}
function parseISO(s){ return new Date(s+'T00:00:00Z'); }
function daysFromISO(s){
  const d0=new Date(GENESIS_ISO+'T00:00:00Z');
  const d=parseISO(s);
  return ((d.getTime()-d0.getTime())/86400000)+1.0;
}
function interp(xa, ya, x){
  let lo=0,hi=xa.length-1;
  if(x<=xa[0]) return ya[0];
  if(x>=xa[hi]) return ya[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xa[m]<=x) lo=m; else hi=m; }
  const t=(x-xa[lo])/(xa[hi]-xa[lo]); return ya[lo]+t*(ya[hi]-ya[lo]);
}
function positionPctLog(y,f,c){
  const ly=Math.log10(y), lf=Math.log10(f), lc=Math.log10(c);
  return Math.max(0, Math.min(100, 100*(ly-lf)/Math.max(1e-12, lc-lf)));
}

// find index by nearest date
function nearestIndexByDate(isoArr, targetISO){
  const t=parseISO(targetISO).getTime();
  let bestI=0, best=1e99;
  for(let i=0;i<isoArr.length;i++){
    const di=Math.abs(parseISO(isoArr[i]).getTime()-t);
    if(di<best){ best=di; bestI=i; }
  }
  return bestI;
}
function extremeInYear(isoArr, yArr, xArr, year, mode){
  let idxs=[];
  for(let i=0;i<isoArr.length;i++){
    if(isoArr[i].slice(0,4)==String(year)) idxs.push(i);
  }
  if(!idxs.length) throw new Error("No data for year "+year);
  let bestI=idxs[0];
  for(const i of idxs){
    if(mode==='max'){ if(yArr[i]>yArr[bestI]) bestI=i; }
    else { if(yArr[i]<yArr[bestI]) bestI=i; }
  }
  return {x:xArr[bestI], y:yArr[bestI], dateISO: isoArr[bestI]};
}

function lineFromTwoPointsLog(x1,y1,x2,y2){
  const lx1=Math.log10(x1), ly1=Math.log10(y1);
  const lx2=Math.log10(x2), ly2=Math.log10(y2);
  const b=(ly2-ly1)/(lx2-lx1);
  const a=ly1 - b*lx1;
  return {a,b};
}
function railsFromAnchors(P, ceilY1, ceilY2, floorMode, floorArgs){
  // Ceiling anchors (highs by year)
  const c1 = extremeInYear(P.date_iso_main, P.y_main, P.x_main, ceilY1, 'max');
  const c2 = extremeInYear(P.date_iso_main, P.y_main, P.x_main, ceilY2, 'max');
  const {a:aC, b} = lineFromTwoPointsLog(c1.x, c1.y, c2.x, c2.y);

  // Floor anchors
  let f1x,f1y, f2x,f2y;
  if(floorMode==='A'){ // earliest 2013 + date
    // earliest 2013
    let first2013 = -1;
    for(let i=0;i<P.date_iso_main.length;i++){ if(P.date_iso_main[i].slice(0,4)==='2013'){ first2013=i; break; } }
    if(first2013<0) throw new Error("No 2013 data to anchor floor");
    f1x=P.x_main[first2013]; f1y=P.y_main[first2013];
    // specific date (nearest)
    const idx=nearestIndexByDate(P.date_iso_main, floorArgs.dateISO);
    f2x=P.x_main[idx]; f2y=P.y_main[idx];
  } else if(floorMode==='B'){ // two YEARS lows
    const f1=extremeInYear(P.date_iso_main, P.y_main, P.x_main, floorArgs.y1, 'min');
    const f2=extremeInYear(P.date_iso_main, P.y_main, P.x_main, floorArgs.y2, 'min');
    f1x=f1.x; f1y=f1.y; f2x=f2.x; f2y=f2.y;
  } else { // 'C' two DATES
    const i1=nearestIndexByDate(P.date_iso_main, floorArgs.d1ISO);
    const i2=nearestIndexByDate(P.date_iso_main, floorArgs.d2ISO);
    f1x=P.x_main[i1]; f1y=P.y_main[i1];
    f2x=P.x_main[i2]; f2y=P.y_main[i2];
  }

  // Intercept aF consistent with slope b (average of two floor anchors in log-space)
  const aF1 = Math.log10(f1y) - b*Math.log10(f1x);
  const aF2 = Math.log10(f2y) - b*Math.log10(f2x);
  const aF = (aF1 + aF2)/2;

  const lx = P.x_grid.map(v => Math.log10(v));
  const logF = lx.map(v => aF + b*v);
  const logC = lx.map(v => aC + b*v);

  function exp10(arr){ return arr.map(v => Math.pow(10, v)); }
  const FLOOR   = exp10(logF);
  const CEILING = exp10(logC);
  const P25     = logF.map((lf,i)=> Math.pow(10, lf + 0.25*(logC[i]-lf)));
  const P50     = logF.map((lf,i)=> Math.pow(10, lf + 0.50*(logC[i]-lf)));
  const P75     = logF.map((lf,i)=> Math.pow(10, lf + 0.75*(logC[i]-lf)));

  return {FLOOR, P25, P50, P75, CEILING};
}

// ---------- DOM ----------
const denomSel  = document.getElementById('denomSel');
const denoms = Object.keys(PRECOMP).filter(k => k!=='USD');
document.getElementById('denomsDetected').textContent = denoms.length ? denoms.join(', ') : '(none)';
['USD', ...denoms].forEach(k => {
  const opt=document.createElement('option');
  opt.value=k; opt.textContent=(k==='USD')?'USD/None':k;
  denomSel.appendChild(opt);
});

const elDate=document.querySelector('#readout .date');
const elF=document.getElementById('vF'), el25=document.getElementById('v25'),
      el50=document.getElementById('v50'), el75=document.getElementById('v75'), elC=document.getElementById('vC');
const elMain=document.getElementById('mainVal'), elPPct=document.getElementById('pPct');

// anchors UI
const ceilY1 = document.getElementById('ceilY1');
const ceilY2 = document.getElementById('ceilY2');
const floorModeRadios = [...document.querySelectorAll('input[name="floorMode"]')];
const floorA = document.getElementById('floorA'), floorA_date=document.getElementById('floorA_date');
const floorB = document.getElementById('floorB'), floorB_y1=document.getElementById('floorB_y1'), floorB_y2=document.getElementById('floorB_y2');
const floorC = document.getElementById('floorC'), floorC_d1=document.getElementById('floorC_d1'), floorC_d2=document.getElementById('floorC_d2');
const applyBtn = document.getElementById('applyAnchors');

function currentFloorMode(){ return floorModeRadios.find(r=>r.checked).value; }
function showFloorUI(){
  const mode=currentFloorMode();
  floorA.style.display = (mode==='A')?'flex':'none';
  floorB.style.display = (mode==='B')?'flex':'none';
  floorC.style.display = (mode==='C')?'flex':'none';
}
floorModeRadios.forEach(r => r.addEventListener('change', showFloorUI));
showFloorUI();

// Keep current rails in memory
let CURRENT_RAILS = null;
let locked=false, lockedX=null;

function updatePanel(P, xDays){
  elDate.textContent = dateFromDaysShort(xDays);
  const F = interp(P.x_grid, CURRENT_RAILS.FLOOR,   xDays);
  const v25=interp(P.x_grid, CURRENT_RAILS.P25,     xDays);
  const v50=interp(P.x_grid, CURRENT_RAILS.P50,     xDays);
  const v75=interp(P.x_grid, CURRENT_RAILS.P75,     xDays);
  const C = interp(P.x_grid, CURRENT_RAILS.CEILING, xDays);
  elF.textContent=fmtUSD(F); el25.textContent=fmtUSD(v25); el50.textContent=fmtUSD(v50); el75.textContent=fmtUSD(v75); elC.textContent=fmtUSD(C);

  // nearest observed price
  let idx=0,best=1e99;
  for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xDays); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx]; elMain.textContent = fmtUSD(y);
  const p = positionPctLog(y,F,C);
  elPPct.textContent = `(p≈${p.toFixed(1)}%)`;

  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — Anchor Rails (p≈${p.toFixed(1)}%)`});
}

function restyleRails(P){
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  // traces: Floor(0),25(1),50(2),75(3),Ceil(4)
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.FLOOR]},   [0]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P25]},     [1]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P50]},     [2]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P75]},     [3]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.CEILING]}, [4]);
}

// compute rails from current UI for selected denom
function computeAndApplyRails(denKey){
  const P = PRECOMP[denKey];
  const y1 = parseInt(ceilY1.value,10);
  const y2 = parseInt(ceilY2.value,10);
  const mode = currentFloorMode();
  let rails=null;
  try{
    if(mode==='A'){
      rails = railsFromAnchors(P, y1, y2, 'A', {dateISO: floorA_date.value || '2015-10-01'});
    }else if(mode==='B'){
      rails = railsFromAnchors(P, y1, y2, 'B', {y1: parseInt(floorB_y1.value,10), y2: parseInt(floorB_y2.value,10)});
    }else{
      rails = railsFromAnchors(P, y1, y2, 'C', {d1ISO: floorC_d1.value, d2ISO: floorC_d2.value});
    }
  }catch(e){
    console.error(e);
    alert("Anchor computation error: "+e.message);
    return;
  }
  CURRENT_RAILS = rails;
  restyleRails(P);
  const xTarget = locked ? lockedX : P.x_main[P.x_main.length-1];
  updatePanel(P, xTarget);
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

  // Live hover
  plotDiv.on('plotly_hover', function(ev){
    if(!ev.points||!ev.points.length) return;
    if(locked) return;
    const P=PRECOMP[denomSel.value];
    updatePanel(P, ev.points[0].x);
  });

  document.getElementById('setDateBtn').addEventListener('click', function(){
    const val=document.getElementById('datePick').value;
    if(!val) return;
    const xDays=daysFromISO(val);
    locked=true; lockedX=xDays;
    updatePanel(PRECOMP[denomSel.value], xDays);
  });
  document.getElementById('liveBtn').addEventListener('click', function(){ locked=false; lockedX=null; });

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

  // Denominator change → recompute rails and relabel axis
  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [5]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main] }, [6]); // cursor
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.main_rebased] }, [7]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.denom_rebased] }, [8]);
    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});
    computeAndApplyRails(key);
  });

  // Apply Anchors
  applyBtn.addEventListener('click', function(){ computeAndApplyRails(denomSel.value); });

  // Initial rails (defaults 2011/2017 highs; 2013-start + 2015-10-01 floor)
  computeAndApplyRails('USD');
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
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote %s" % OUTPUT_HTML)