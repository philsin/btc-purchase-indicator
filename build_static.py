#!/usr/bin/env python3
"""
BTC Purchase Indicator — Envelope Mode (Floor/Ceiling + log-midlines 10%, 50%, 90%)

Model:
- Trend: log10(price) = a + b * log10(days since Genesis)
- Residuals: e(t) = log10(price) - (a + b*log10(days))
- Floor/Ceiling: exponentially weighted lower/upper envelopes of e(t)
- Midlines at log-space fractions between Floor and Ceiling: 10%, 50% (bold), 90%

UI:
- One view (Envelope Mode). Compare toggle retained (rebased lines; envelopes hidden in Compare).
- Right panel tracks cursor unless a date is locked. Values show $ with two decimals, right-aligned.
- p% is the log-space position between Floor and Ceiling at the chosen date.
- Y-axis title updates with denominator; x-axis title removed. Odd years hidden after 2026.

"""

import os, io, glob, time, math, json
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ---------------------------- CONFIG ----------------------------

OUTPUT_HTML   = "docs/index.html"
DATA_DIR      = "data"
BTC_FILE      = os.path.join(DATA_DIR, "btc_usd.csv")

GENESIS_DATE  = datetime(2009, 1, 3)
END_PROJ      = datetime(2040, 12, 31)

# Envelope smoothing
ENVELOPE_HALFLIFE_YEARS = 2.0  # adjust if you want faster/slower adaptation

# Colors
LINE_FLOOR   = "#D32F2F"
LINE_10      = "#F57C00"
LINE_50      = "#111111"  # bold/black
LINE_90      = "#2E7D32"
LINE_CEILING = "#6A1B9A"
BTC_COLOR    = "#000000"

# ---------------------------- UTILS -----------------------------

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#"); r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

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

# --------------------------- MODEL ------------------------------

def fit_trend_median(x_days: np.ndarray, y_price: np.ndarray) -> Tuple[float,float,pd.Series]:
    """
    Robust central trend via median (q=0.5) quantile regression in log–log space.
    Returns (a,b,residuals), where residuals are y - (a+b*x) in log10 space.
    """
    m=(x_days>0)&(y_price>0)&np.isfinite(x_days)&np.isfinite(y_price)
    x=x_days[m]; y=np.log10(y_price[m]); X=pd.DataFrame({"logx":np.log10(x)})
    if len(x)<10: raise RuntimeError("Not enough data to fit trend")
    model=QuantReg(y, pd.concat([pd.Series(1.0,index=X.index,name="const"), X],axis=1))
    res=model.fit(q=0.5)
    a=float(res.params["const"]); b=float(res.params["logx"])
    resid = y - (a + b*np.log10(x))
    return a,b,resid

def ew_envelopes(resid: np.ndarray, dates: pd.Series, halflife_years: float) -> Tuple[np.ndarray,np.ndarray]:
    """
    Exponentially weighted floor (lower envelope) and ceiling (upper envelope) on log residuals.
    The smoothing uses time deltas (days) so unequal spacing is ok.
    """
    # convert date spacing to days
    d = pd.to_datetime(dates).to_numpy()
    d_days = np.r_[0.0, (d[1:] - d[:-1]) / np.timedelta64(1, "D")]
    hl_days = halflife_years * 365.25
    # per-step decay lambda = 0.5^(delta/hl)
    lambdas = np.power(0.5, d_days/hl_days)
    L = np.empty_like(resid); U = np.empty_like(resid)
    L[0] = resid[0]; U[0] = resid[0]
    for i in range(1, len(resid)):
        lam = float(lambdas[i])
        emaL = lam*L[i-1] + (1.0-lam)*resid[i]
        emaU = lam*U[i-1] + (1.0-lam)*resid[i]
        L[i] = min(resid[i], emaL)  # lower envelope
        U[i] = max(resid[i], emaU)  # upper envelope
    return L, U

def interp_env_to_grid(x_src: np.ndarray, v_src: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation in x (days)."""
    return np.interp(x_grid, x_src, v_src, left=v_src[0], right=v_src[-1])

def predict_env_lines(a: float, b: float,
                      x_main: np.ndarray, L_resid: np.ndarray, U_resid: np.ndarray,
                      x_grid: np.ndarray) -> Dict[str, list]:
    """
    Build Floor/Ceiling and midlines (10%,50%,90%) in price space on x_grid.
    All interpolation in log residual space; convert back with trend.
    """
    Lg = interp_env_to_grid(x_main, L_resid, x_grid)
    Ug = interp_env_to_grid(x_main, U_resid, x_grid)
    lx = np.log10(x_grid)
    trend = a + b*lx  # log10(price)
    def to_price(log_offset): return (10**(trend + log_offset)).tolist()
    # fractions in log residual space:
    frac10 = Lg + 0.10*(Ug - Lg)
    frac50 = Lg + 0.50*(Ug - Lg)
    frac90 = Lg + 0.90*(Ug - Lg)
    return {
        "FLOOR":   to_price(Lg),
        "P10":     to_price(frac10),
        "P50":     to_price(frac50),
        "P90":     to_price(frac90),
        "CEILING": to_price(Ug),
    }

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
    x_vals = base["x_days"].values.astype(float)

    # 1) central trend (median regression)
    a,b,resid = fit_trend_median(x_vals, y_main.values)

    # 2) envelopes in log-residual space on main dates
    L_resid, U_resid = ew_envelopes(resid.values, base["date"], ENVELOPE_HALFLIFE_YEARS)

    # 3) build Floor/Ceiling + midlines on x_grid
    env_lines = predict_env_lines(a,b,x_vals,L_resid,U_resid,x_grid)

    # p% at the latest data point (used only as initial title; panel recomputes dynamically)
    r_last = resid.values[-1]
    lo, hi = L_resid[-1], U_resid[-1]
    p_init = float(np.clip((r_last - lo) / max(1e-12, (hi - lo)), 0, 1)) * 100.0

    return {
        "label": y_label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "env": env_lines,                 # dict of lists: FLOOR, P10, P50, P90, CEILING
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "p_init": p_init,
        "trend_params": {"a":a, "b":b},   # for debugging/future
        "envelope_resid": {"L": L_resid.tolist(), "U": U_resid.tolist()},
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

# -------------------------- FIGURE ------------------------------

traces=[]; vis_norm=[]; vis_cmp=[]

# Envelope lines on x_grid
def add_line(y_list, name, color, width=1.2, dash=None, bold=False, showlegend=True):
    line=dict(width=width, color=color)
    if dash: line["dash"]=dash
    if bold: line["width"]=2.2
    traces.append(go.Scatter(
        x=init["x_grid"], y=y_list, mode="lines", name=name,
        line=line, hoverinfo="skip", showlegend=showlegend
    ))
    vis_norm.append(True); vis_cmp.append(False)

add_line(init["env"]["FLOOR"],   "Floor",   LINE_FLOOR,   width=1.2)
add_line(init["env"]["P10"],     "10%",     LINE_10,      width=1.2, dash="dot")
add_line(init["env"]["P50"],     "50%",     LINE_50,      bold=True)
add_line(init["env"]["P90"],     "90%",     LINE_90,      width=1.2, dash="dot")
add_line(init["env"]["CEILING"], "Ceiling", LINE_CEILING, width=1.2)

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
    np.nanmax(init["env"][k]) for k in ["FLOOR","P10","P50","P90","CEILING"]
]
y_max = max(y_candidates)
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x",  # keep events for live hover
    hoverdistance=30, spikedistance=30,
    title=f"BTC Purchase Indicator — Envelope Mode (p≈{init['p_init']:.1f}%)",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=init["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=400, t=70, b=70),
)

# ---------------------- HTML (panel + copy + date lock) ----------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)  # all lists now JSON-safe
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
      <div class="row"><div><span style="color:$COL_10;">10%</span></div><div class="num" id="v10">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_50;font-weight:700;">50%</span></div><div class="num" id="v50" style="font-weight:700;">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_90;">90%</span></div><div class="num" id="v90">$0.00</div><div></div></div>
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
const elF=document.getElementById('vF'), el10=document.getElementById('v10'),
      el50=document.getElementById('v50'), el90=document.getElementById('v90'), elC=document.getElementById('vC');
const elMain=document.getElementById('mainVal'), elPPct=document.getElementById('pPct');

let locked=false, lockedX=null;

function updatePanel(den,xDays){
  const P=PRECOMP[den];
  elDate.textContent = dateFromDaysShort(xDays);

  const F=interp(P.x_grid,P.env.FLOOR,xDays);
  const v10=interp(P.x_grid,P.env.P10,xDays);
  const v50=interp(P.x_grid,P.env.P50,xDays);
  const v90=interp(P.x_grid,P.env.P90,xDays);
  const C=interp(P.x_grid,P.env.CEILING,xDays);

  elF.textContent=fmtUSD(F); el10.textContent=fmtUSD(v10); el50.textContent=fmtUSD(v50);
  el90.textContent=fmtUSD(v90); elC.textContent=fmtUSD(C);

  // nearest observed price
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-xDays); if(d<best){best=d; idx=i;} }
  const y=ya[idx]; elMain.textContent = fmtUSD(y);

  const p = positionPctLog(y,F,C);
  elPPct.textContent = `(p≈${p.toFixed(1)}%)`;

  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — Envelope Mode (p≈${p.toFixed(1)}%)`});
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
    // Order: Floor(0), 10%(1), 50%(2), 90%(3), Ceiling(4), BTC(5), cursor(6), rebased(7,8)
    restyleLine(0, P.env.FLOOR);
    restyleLine(1, P.env.P10);
    restyleLine(2, P.env.P50);
    restyleLine(3, P.env.P90);
    restyleLine(4, P.env.CEILING);

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
    COL_F=LINE_FLOOR, COL_10=LINE_10, COL_50=LINE_50, COL_90=LINE_90, COL_C=LINE_CEILING,
    P_INIT=f"{init['p_init']:.1f}",
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
