#!/usr/bin/env python3
# -- snip: header comment omitted for brevity (same as before) --
import os, io, glob, time, json
from typing import Optional, Dict
from datetime import datetime, timezone
from string import Template
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

OUTPUT_HTML   = "docs/index.html"
DATA_DIR      = "data"
BTC_FILE      = os.path.join(DATA_DIR, "btc_usd.csv")

GENESIS_DATE  = datetime(2009, 1, 3)
END_PROJ      = datetime(2040, 12, 31)

CEIL_Q  = 0.975
FLOOR_Q = 0.025
EPS_LOG = 0.015

COL_FLOOR   = "#D32F2F"  # red
COL_20      = "#F57C00"  # orange
COL_50      = "#FBC02D"  # gold
COL_80      = "#66BB6A"  # green
COL_CEILING = "#2E7D32"  # dark green
COL_BTC     = "#000000"  # black

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def _retry(fn, tries=3, base_delay=1.0, factor=2.0):
    last=None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(base_delay*(factor**i))
    raise last

def years_since_genesis(dates, genesis):
    d = pd.to_datetime(dates)
    s = pd.Series(d)
    delta_days = (s - pd.Timestamp(genesis)) / np.timedelta64(1, "D")
    years = (delta_days.astype(float) / 365.25) + (1.0/365.25)
    return years

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

def robust_q50_fit(x_years: np.ndarray, y_series: np.ndarray):
    m = (x_years > 0) & (y_series > 0) & np.isfinite(x_years) & np.isfinite(y_series)
    x = np.log10(x_years[m]); y = np.log10(y_series[m])
    X = pd.DataFrame({"const": 1.0, "logx": x})
    model = QuantReg(y, X)
    res = model.fit(q=0.5)
    a0 = float(res.params["const"]); b = float(res.params["logx"])
    resid = y - (a0 + b * x)
    return a0, b, resid, m

def suggest_defaults_for_series(dates, x_years, y_series,
                                ceil_q=CEIL_Q, floor_q=FLOOR_Q, eps=EPS_LOG, half_window_days=90):
    a0, b, resid, mask = robust_q50_fit(x_years.values, y_series.values)
    cC = float(np.nanquantile(resid, ceil_q)) + eps
    cF = float(np.nanquantile(resid, floor_q)) - eps

    idx_all = np.where(mask)[0]
    if len(idx_all) < 4:
        d0 = dates.iloc[idx_all[0]].date(); d1 = dates.iloc[idx_all[-1]].date()
        return {"a0":a0,"b":b,"cF":cF,"cC":cC,
                "ranges":{"ceil1":[str(d0),str(d1)],"ceil2":[str(d0),str(d1)],"floor":[str(d0),str(d1)]}}

    compact = np.where(mask)[0]; comp_to_resid = {orig:i for i,orig in enumerate(compact)}
    half = len(idx_all)//2; early = idx_all[:max(1, half)]; late = idx_all[max(1, half):]

    def closest_idx(subidx, target):
        r = np.array([resid[comp_to_resid[i]] for i in subidx])
        j = int(np.argmin(np.abs(r - target))); return int(subidx[j])

    iC1 = closest_idx(early, cC); iC2 = closest_idx(late, cC); iF = closest_idx(idx_all, cF)

    def mk_range(i):
        d = dates.iloc[i]
        lo = (d - pd.Timedelta(days=half_window_days)).date()
        hi = (d + pd.Timedelta(days=half_window_days)).date()
        return [str(lo), str(hi)]

    return {"a0":a0,"b":b,"cF":cF,"cC":cC,
            "ranges":{"ceil1":mk_range(iC1),"ceil2":mk_range(iC2),"floor":mk_range(iF)}}

btc = get_btc_df().rename(columns={"price":"btc"})
denoms = collect_denominators()
base = btc.sort_values("date").reset_index(drop=True)
if base.empty: raise RuntimeError("No BTC data found")
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

base["x_years"]  = years_since_genesis(base["date"], GENESIS_DATE)
base["date_iso"] = base["date"].dt.strftime("%Y-%m-%d")

x_start = float(base["x_years"].iloc[0])
x_end   = float(years_since_genesis(pd.Series([END_PROJ]), GENESIS_DATE).iloc[0])
x_grid  = np.logspace(np.log10(max(1e-6, x_start)), np.log10(x_end), 600)

def year_ticks_log(first_date: datetime, last_date: datetime):
    y0, y1 = first_date.year, last_date.year
    vals, labs = [], []
    for y in range(y0, y1+1):
        d = datetime(y, 1, 1)
        if d < first_date or d > last_date: continue
        vy = float(years_since_genesis(pd.Series([d]), GENESIS_DATE).iloc[0])
        if vy <= 0: continue
        if y > 2026 and (y % 2 == 1): continue
        vals.append(vy); labs.append(str(y))
    return vals, labs
first_date = base["date"].iloc[0].to_pydatetime()
xtickvals, xticktext = year_ticks_log(first_date, END_PROJ)

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=denom_key.lower()
    if k in df.columns: return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def build_payload(denom_key: Optional[str]):
    y_main, y_label, denom_series = series_for_denom(base, denom_key)
    defaults = suggest_defaults_for_series(base["date"], base["x_years"], y_main)
    return {
        "label": y_label,
        "x_main": base["x_years"].tolist(),
        "y_main": y_main.tolist(),
        "date_iso_main": base["date_iso"].tolist(),
        "x_grid": x_grid.tolist(),
        "main_rebased": (y_main / y_main.iloc[0]).tolist(),
        "denom_rebased": (denom_series/denom_series.iloc[0]).tolist() if denom_series is not None else [float("nan")]*len(base),
        "defaults": defaults,
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()): PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

traces=[]
def add_line_on_grid(name, color, width=1.3, dash=None, bold=False):
    line=dict(width=2.4 if bold else width, color=color)
    if dash: line["dash"]=dash
    traces.append(go.Scatter(x=init["x_grid"], y=[None]*len(init["x_grid"]),
                             mode="lines", name=name, line=line, hoverinfo="skip", showlegend=True))
add_line_on_grid("Floor",   COL_FLOOR)
add_line_on_grid("20%",     COL_20, dash="dot")
add_line_on_grid("50%",     COL_50, bold=True)
add_line_on_grid("80%",     COL_80, dash="dot")
add_line_on_grid("Ceiling", COL_CEILING)
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="BTC / USD", line=dict(width=1.9, color=COL_BTC), hoverinfo="skip"))
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         line=dict(width=0), opacity=0.003, hoverinfo="x", showlegend=False, name="_cursor"))
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False, line=dict(color=COL_BTC)))
traces.append(go.Scatter(x=init["x_main"], y=init["denom_rebased"], name="Denominator (rebased)",
                         mode="lines", line=dict(dash="dash"), hoverinfo="skip", visible=False))

def make_y_ticks():
    vals=[1e-8] + [10**e for e in range(0, 9)]
    texts=["0"] + [f"{int(10**e):,}" for e in range(0, 9)]
    return vals, texts
ytickvals, yticktext = make_y_ticks()

fig = go.Figure(data=traces)
fig.update_layout(
    template="plotly_white", showlegend=True, hovermode="x",
    hoverdistance=30, spikedistance=30,
    title="BTC Purchase Indicator — Rails (Years on X)",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=init["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=520, t=70, b=70),
)

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"responsive": True, "displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json = json.dumps(PRECOMP)
genesis_iso  = GENESIS_DATE.strftime("%Y-%m-%d")

html_tpl = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root { --panelW: 520px; }
  html, body { height: 100%; margin:0; padding:0; }
  body { font-family: Inter, Roboto, -apple-system, Segoe UI, Arial, sans-serif; }
  .layout { display: grid; grid-template-columns: 1fr var(--panelW); min-height: 100vh; width: 100vw; }
  .left { padding: 8px 0 8px 8px; }
  .left .js-plotly-plot, .left .plotly-graph-div { width: 100% !important; height: 100% !important; }
  .right { border-left: 1px solid #e5e7eb; padding: 12px; display: flex; flex-direction: column; gap: 12px; overflow:auto; }

  #controls, #anchors { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  select, button, input[type="date"] { font-size:14px; padding:8px 10px; border-radius:8px; border:1px solid #d1d5db; background:white; }
  fieldset { border:1px solid #e5e7eb; border-radius:10px; padding:10px; }
  legend { font-weight:700; font-size:13px; color:#374151; }

  #readout { border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; font-size:14px; }
  #readout .date { font-weight:700; margin-bottom:6px; }
  #readout .row { display:grid; grid-template-columns: auto 1fr auto; column-gap:8px; align-items:baseline; }
  #readout .num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-variant-numeric: tabular-nums; text-align:right; min-width:12ch; white-space:pre; }

  .hoverlayer { opacity: 0 !important; pointer-events: none; } /* hide default tooltip */

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .right { border-left:none; border-top:1px solid #e5e7eb; }
    .left { padding: 8px; }
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
      <button id="setDateBtn">Set Date</button>
      <button id="liveBtn">Live Hover</button>
      <button id="copyBtn">Copy Chart</button>
    </div>

    <fieldset id="anchors">
      <legend>Rails Mode</legend>
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        <label><input type="radio" name="railsMode" value="AUTO" checked/> Auto (Percentiles)</label>
        <label><input type="radio" name="railsMode" value="RANGES"/> Date Ranges</label>
      </div>

      <div id="rangeUI" style="margin-top:10px; display:none;">
        <div style="font-weight:700; margin-bottom:4px;">Ceiling (TWO ranges; uses HIGH within each):</div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <label>Range 1: <input type="date" id="c1s"/> — <input type="date" id="c1e"/></label>
          <label>Range 2: <input type="date" id="c2s"/> — <input type="date" id="c2e"/></label>
        </div>
        <div style="font-weight:700; margin:10px 0 4px;">Floor (ONE range; uses LOW):</div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <label>Range: <input type="date" id="fs"/> — <input type="date" id="fe"/></label>
          <span style="font-size:12px;color:#6b7280;">(One range = start & end)</span>
        </div>
        <div style="display:flex; gap:10px; margin-top:10px;">
          <button id="applyRanges">Apply Date Ranges</button>
          <span style="font-size:12px;color:#6b7280;">20%/80% are 60% of distance from midline to outer rails.</span>
        </div>
      </div>
    </fieldset>

    <div style="font-size:12px;color:#6b7280;">Detected denominators: <span id="denomsDetected"></span></div>

    <div id="readout">
      <div class="date">—</div>
      <div class="row"><div><span style="color:$COL_F;">Floor</span></div><div class="num" id="vF">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_20;">20%</span></div><div class="num" id="v20">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_50;font-weight:700;">50%</span></div><div class="num" id="v50" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_80;">80%</span></div><div class="num" id="v80">$0.00</div><div></div></div>
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

function fmtUSD(v){ if(!isFinite(v)) return '$—'; return '$' + Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
function shortMonthName(m){ return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m]; }
function dateFromYearsShort(y){
  const d0=new Date(GENESIS_ISO+'T00:00:00Z');
  const ms = (y - (1.0/365.25)) * 365.25 * 86400000;
  const d=new Date(d0.getTime() + ms);
  return `${shortMonthName(d.getUTCMonth())}-${String(d.getUTCDate()).padStart(2,'0')}-${String(d.getUTCFullYear()).slice(-2)}`;
}
function parseISO(s){ return new Date(s+'T00:00:00Z'); }
function yearsFromISO(s){
  const d0=new Date(GENESIS_ISO+'T00:00:00Z');
  const d=parseISO(s);
  return ( (d.getTime()-d0.getTime())/86400000 ) / 365.25 + (1.0/365.25);
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

// --------- FIXED JS helper (previously pasted in Python) ----------
function railsFromMidAndEnvelopeJS(logx, aMid, bMid, aFloor, aCeil){
  // Midline
  const logM = logx.map(v => aMid + bMid*v);
  // Outer rails (parallel)
  const logF = logx.map(v => aFloor + bMid*v);
  const logC = logx.map(v => aCeil  + bMid*v);
  // Inner rails at 60% of distance between midline and outer rails
  const logP20 = logM.map((m,i)=> m - 0.6*(m - logF[i]));
  const logP80 = logM.map((m,i)=> m + 0.6*(logC[i] - m));
  const exp10 = arr => arr.map(v => Math.pow(10, v));
  return {FLOOR:exp10(logF), P20:exp10(logP20), P50:exp10(logM), P80:exp10(logP80), CEILING:exp10(logC)};
}

function railsFromQuantiles(P){
  const aMid = P.defaults.a0, bMid = P.defaults.b;
  const cF   = P.defaults.cF, cC   = P.defaults.cC;
  const lx = P.x_grid.map(v => Math.log10(v));
  const aFloor = aMid + cF;
  const aCeil  = aMid + cC;
  return railsFromMidAndEnvelopeJS(lx, aMid, bMid, aFloor, aCeil);
}

function capToRange(iso, startISO, endISO){
  const t=new Date(iso+'T00:00:00Z').getTime();
  return (t >= new Date(startISO+'T00:00:00Z').getTime() && t <= new Date(endISO+'T00:00:00Z').getTime());
}
function maxInRange(isoArr,yArr,xArr,startISO,endISO){
  let bestI=-1, best=-Infinity;
  for(let i=0;i<isoArr.length;i++){
    if(capToRange(isoArr[i], startISO, endISO) && yArr[i]>0){
      if(yArr[i]>best){ best=yArr[i]; bestI=i; }
    }
  }
  if(bestI<0) throw new Error(`No data in ceiling range ${startISO}..${endISO}`);
  return {x:xArr[bestI], y:yArr[bestI], iso: isoArr[bestI]};
}
function minInRange(isoArr,yArr,xArr,startISO,endISO){
  let bestI=-1, best= Infinity;
  for(let i=0;i<isoArr.length;i++){
    if(capToRange(isoArr[i], startISO, endISO) && yArr[i]>0){
      if(yArr[i]<best){ best=yArr[i]; bestI=i; }
    }
  }
  if(bestI<0) throw new Error(`No data in floor range ${startISO}..${endISO}`);
  return {x:xArr[bestI], y:yArr[bestI], iso: isoArr[bestI]};
}
function lineFromTwoPointsLog(x1,y1,x2,y2){
  const lx1=Math.log10(x1), ly1=Math.log10(y1);
  const lx2=Math.log10(x2), ly2=Math.log10(y2);
  const b=(ly2-ly1)/(lx2-lx1);
  const a=ly1 - b*lx1;
  return {a,b};
}
function railsFromRanges(P, c1s, c1e, c2s, c2e, fs, fe){
  const C1 = maxInRange(P.date_iso_main, P.y_main, P.x_main, c1s, c1e);
  const C2 = maxInRange(P.date_iso_main, P.y_main, P.x_main, c2s, c2e);
  const {a:aC, b:bC} = lineFromTwoPointsLog(C1.x, C1.y, C2.x, C2.y);
  const F0 = minInRange(P.date_iso_main, P.y_main, P.x_main, fs, fe);
  const aF = Math.log10(F0.y) - bC*Math.log10(F0.x);
  const aMid = P.defaults.a0, bMid = P.defaults.b;
  const lx = P.x_grid.map(v => Math.log10(v));
  // Floor/Ceiling parallel to bC, midline uses bMid (independent)
  const logM = lx.map(v => aMid + bMid*v);
  const logF = lx.map(v => aF   + bC *v);
  const logC = lx.map(v => aC   + bC *v);
  const logP20 = logM.map((m,i)=> m - 0.6*(m - logF[i]));
  const logP80 = logM.map((m,i)=> m + 0.6*(logC[i] - m));
  const exp10 = a => a.map(v => Math.pow(10,v));
  return {FLOOR:exp10(logF), P20:exp10(logP20), P50:exp10(logM), P80:exp10(logP80), CEILING:exp10(logC)};
}

const denomSel  = document.getElementById('denomSel');
const denoms = Object.keys(PRECOMP).filter(k => k!=='USD');
document.getElementById('denomsDetected').textContent = denoms.length ? denoms.join(', ') : '(none)';
['USD', ...denoms].forEach(k => {
  const opt=document.createElement('option'); opt.value=k; opt.textContent=(k==='USD')?'USD/None':k; denomSel.appendChild(opt);
});

const railsModeRadios = [...document.querySelectorAll('input[name="railsMode"]')];
const rangeUI = document.getElementById('rangeUI');
function currentRailsMode(){ return railsModeRadios.find(r=>r.checked).value; }
railsModeRadios.forEach(r => r.addEventListener('change', () => {
  rangeUI.style.display = currentRailsMode()==='RANGES' ? 'block' : 'none';
  computeAndApplyRails(denomSel.value);
}));

const c1s=document.getElementById('c1s'), c1e=document.getElementById('c1e');
const c2s=document.getElementById('c2s'), c2e=document.getElementById('c2e');
const fs =document.getElementById('fs'),  fe =document.getElementById('fe');
document.getElementById('applyRanges').addEventListener('click', () => computeAndApplyRails(denomSel.value));

const elDate=document.querySelector('#readout .date');
const elF=document.getElementById('vF'), el20=document.getElementById('v20'),
      el50=document.getElementById('v50'), el80=document.getElementById('v80'), elC=document.getElementById('vC');
const elMain=document.getElementById('mainVal'), elPPct=document.getElementById('pPct');

let CURRENT_RAILS = null;
let locked=false, lockedX=null;

function updatePanel(P, xYears){
  elDate.textContent = dateFromYearsShort(xYears);
  const F = interp(P.x_grid, CURRENT_RAILS.FLOOR,   xYears);
  const v20=interp(P.x_grid, CURRENT_RAILS.P20,     xYears);
  const v50=interp(P.x_grid, CURRENT_RAILS.P50,     xYears);
  const v80=interp(P.x_grid, CURRENT_RAILS.P80,     xYears);
  const C = interp(P.x_grid, CURRENT_RAILS.CEILING, xYears);
  elF.textContent=fmtUSD(F); el20.textContent=fmtUSD(v20); el50.textContent=fmtUSD(v50); el80.textContent=fmtUSD(v80); elC.textContent=fmtUSD(C);

  let idx=0,best=1e99;
  for(let i=0;i<P.x_main.length;i++){ const d=Math.abs(P.x_main[i]-xYears); if(d<best){best=d; idx=i;} }
  const y=P.y_main[idx]; elMain.textContent = fmtUSD(y);
  const p = positionPctLog(y,F,C);
  elPPct.textContent = `(p≈${p.toFixed(1)}%)`;

  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — Rails (p≈${p.toFixed(1)}%)`});
}

function restyleRails(P){
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.FLOOR]},   [0]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P20]},     [1]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P50]},     [2]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.P80]},     [3]);
  Plotly.restyle(plotDiv, {x:[P.x_grid], y:[CURRENT_RAILS.CEILING]}, [4]);
}

function prefillRangesForDenom(P){
  const R = P.defaults.ranges;
  const [d1s,d1e] = R.ceil1;  const [d2s,d2e] = R.ceil2;  const [dfs,dfe] = R.floor;
  document.getElementById('c1s').value=d1s; document.getElementById('c1e').value=d1e;
  document.getElementById('c2s').value=d2s; document.getElementById('c2e').value=d2e;
  document.getElementById('fs').value=dfs;  document.getElementById('fe').value=dfe;
}

function computeAndApplyRails(denKey){
  const P = PRECOMP[denKey];
  try{
    CURRENT_RAILS = (document.querySelector('input[name="railsMode"]:checked').value==='AUTO')
      ? railsFromQuantiles(P)
      : railsFromRanges(P,
          document.getElementById('c1s').value, document.getElementById('c1e').value,
          document.getElementById('c2s').value, document.getElementById('c2e').value,
          document.getElementById('fs').value,  document.getElementById('fe').value
        );
  }catch(e){
    console.error(e); alert("Rails error: "+e.message); return;
  }
  restyleRails(P);
  const xTarget = (typeof lockedX==='number') ? lockedX : P.x_main[P.x_main.length-1];
  updatePanel(P, xTarget);
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

  plotDiv.on('plotly_hover', function(ev){
    if(!ev.points||!ev.points.length) return;
    if(locked) return;
    const P=PRECOMP[denomSel.value];
    updatePanel(P, ev.points[0].x);
  });

  document.getElementById('setDateBtn').addEventListener('click', function(){
    const val=document.getElementById('datePick').value;
    if(!val) return;
    locked=true; lockedX=yearsFromISO(val);
    updatePanel(PRECOMP[denomSel.value], lockedX);
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

  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];
    prefillRangesForDenom(P);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [5]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main] }, [6]);
    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});
    computeAndApplyRails(key);
  });

  prefillRangesForDenom(PRECOMP['USD']);
  computeAndApplyRails('USD');
  document.getElementById('rangeUI').style.display =
    document.querySelector('input[name="railsMode"]:checked').value==='RANGES' ? 'block' : 'none';
});
</script>
</body>
</html>
""")

html = html_tpl.safe_substitute(
    PLOT_HTML=plot_html,
    PRECOMP_JSON=precomp_json,
    GENESIS_ISO=genesis_iso,
    COL_F=COL_FLOOR, COL_20=COL_20, COL_50=COL_50, COL_80=COL_80, COL_C=COL_CEILING,
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {OUTPUT_HTML}")