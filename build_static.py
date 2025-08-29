#!/usr/bin/env python3
"""
BTC Purchase Indicator — Envelope Mode (Floor/Ceiling + 10%/50%/90% midlines)
- Trend: log10(price) = a + b*log10(days since Genesis)  [robust median regression]
- Residuals: e = log10(price) - (a + b log10(days))
- Envelopes on residuals (log space):
    Floor_t   = min(e_t, λ*Floor_{t-1}   + (1-λ)*e_t)
    Ceiling_t = max(e_t, λ*Ceiling_{t-1} + (1-λ)*e_t)
  with λ = exp(-ln(2)/half_life_days)  (default half-life = 2 years)
- Lines on grid (price space):
    Floor, Ceiling, and log-midlines at 10%, 50% (bold), 90% between them
- X: log(days since Genesis); ticks show whole years (odd years hidden after 2026)
- Y: log, axis title updates to denominator (e.g., BTC / USD)
- Right panel: shows $ values for Floor / 10% / 50% / 90% / Ceiling and BTC Price
- p% = position of BTC between Floor and Ceiling in log space at the selected date
- Live hover: panel follows cursor; can lock a date with the picker
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

# Envelope smoothing half-life (days) — tweak to taste (2y is a good start)
ENVELOPE_HALFLIFE_DAYS = 365 * 2

# Colors
COL_FLOOR   = "#D32F2F"   # red
COL_CEIL    = "#6A1B9A"   # purple
COL_10      = "#F9A825"   # amber
COL_50      = "#2E7D32"   # bold green (mid)
COL_90      = "#388E3C"   # green
COL_PRICE   = "#000000"   # black

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
    """Median quantile regression in log–log space. Returns (a,b,residuals) with residuals in log10."""
    m=(x_days>0)&(y_price>0)&np.isfinite(x_days)&np.isfinite(y_price)
    x=x_days[m]; y=np.log10(y_price[m]); X=pd.DataFrame({"logx":np.log10(x)})
    if len(x)<10: raise RuntimeError("Not enough data to fit trend")
    model=QuantReg(y, pd.concat([pd.Series(1.0,index=X.index,name="const"), X],axis=1))
    res=model.fit(q=0.5)
    a=float(res.params["const"]); b=float(res.params["logx"])
    resid = y - (a + b*np.log10(x))
    # Re-expand residuals to original index (align by mask)
    full_resid = pd.Series(np.nan, index=np.arange(len(x_days), dtype=int))
    full_resid[m] = resid
    return a,b,full_resid

def compute_envelopes(resid: pd.Series, x_days: np.ndarray,
                      half_life_days: float = ENVELOPE_HALFLIFE_DAYS) -> Tuple[np.ndarray,np.ndarray]:
    """
    Exponentially-smoothed running min/max on residuals (log space).
    Returns arrays Floor_t, Ceiling_t aligned to x_days (NaNs before first valid residual).
    For periods after the last data point, we keep the last envelope value (flat).
    """
    lam = float(math.exp(-math.log(2.0) / max(1.0, half_life_days)))
    e = resid.to_numpy(copy=False)
    n = len(e)
    L = np.full(n, np.nan, dtype=float)
    U = np.full(n, np.nan, dtype=float)

    # find first non-nan
    idxs = np.where(np.isfinite(e))[0]
    if len(idxs)==0:
        return L, U
    i0 = idxs[0]
    L[i0] = e[i0]
    U[i0] = e[i0]
    for i in range(i0+1, n):
        if not np.isfinite(e[i]):
            # carry forward with decay but no new info (keeps values steady)
            L[i] = L[i-1]
            U[i] = U[i-1]
            continue
        # EW decayed proposal
        Lp = lam*L[i-1] + (1.0-lam)*e[i]
        Up = lam*U[i-1] + (1.0-lam)*e[i]
        # Hard min/max to keep envelopes hugging extremes
        L[i] = min(e[i], Lp)
        U[i] = max(e[i], Up)

    return L, U

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

    # 1) Trend fit (median)
    a,b,resid_full = fit_trend_median(x_vals, y_main.values)

    # 2) Envelopes on residuals (aligned to x_main)
    L_series, U_series = compute_envelopes(resid_full, x_vals, ENVELOPE_HALFLIFE_DAYS)

    # 3) Interpolate envelopes to x_grid (constant beyond last point)
    x_main = base["x_days"].to_numpy()
    def interp_const(xm, ys, xg):
        # linear interp inside, clamp edges
        y = np.interp(xg, xm, ys, left=ys[0], right=ys[-1])
        return y
    L_grid = interp_const(x_main, L_series, x_grid)
    U_grid = interp_const(x_main, U_series, x_grid)

    # 4) Map envelopes + midlines to PRICE space on grid
    lx = np.log10(x_grid)
    trend_log_grid = a + b*lx
    def price_from_log_resid(resid_log):
        return 10**(trend_log_grid + resid_log)

    floor_grid = price_from_log_resid(L_grid)
    ceil_grid  = price_from_log_resid(U_grid)
    # midlines in log space: L + α*(U-L)
    def mid(alpha): return price_from_log_resid(L_grid + alpha*(U_grid - L_grid))
    line10_grid = mid(0.10)
    line50_grid = mid(0.50)
    line90_grid = mid(0.90)

    # For panel p%: compute position for observed dates, interpolate later
    trend_log_main = a + b*np.log10(x_main)
    resid_obs = np.log10(y_main.values) - trend_log_main
    # ensure no divide-by-zero
    span = np.maximum(1e-12, (U_series - L_series))
    p_main = 100.0 * np.clip((resid_obs - L_series) / span, 0.0, 1.0)

    return {
        "label": y_label,
        "x_main": x_main.tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "floor":  floor_grid.tolist(),
        "ceil":   ceil_grid .tolist(),
        "line10": line10_grid.tolist(),
        "line50": line50_grid.tolist(),
        "line90": line90_grid.tolist(),
        "p_main": p_main.tolist(),                   # % per observed date
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "env_params": {"a":a, "b":b, "lambda": float(math.exp(-math.log(2.0)/ENVELOPE_HALFLIFE_DAYS))},
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

# -------------------------- FIGURE ------------------------------

traces=[]; vis_norm=[]; vis_cmp=[]

# Envelope and midlines on grid
def add_line(y, name, color, width=1.2, dash=None, legend=True):
    traces.append(go.Scatter(
        x=init["x_grid"], y=y, mode="lines", name=name,
        line=dict(width=width, color=color, dash=(dash or "solid")),
        hoverinfo="skip", showlegend=legend
    ))
    vis_norm.append(True); vis_cmp.append(False)

add_line(init["floor"],  "Floor",     COL_FLOOR, width=1.5)
add_line(init["line10"], "10%",       COL_10,   width=1.2, dash="dot")
add_line(init["line50"], "50%",       COL_50,   width=2.2)          # bold
add_line(init["line90"], "90%",       COL_90,   width=1.2, dash="dot")
add_line(init["ceil"],   "Ceiling",   COL_CEIL, width=1.5)

# BTC price (black)
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="BTC / USD", line=dict(width=1.9, color=COL_PRICE),
                         hoverinfo="skip"))
vis_norm.append(True); vis_cmp.append(False)

# Transparent cursor line to guarantee hover events
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="_cursor", line=dict(width=0), opacity=0.003,
                         hoverinfo="x", showlegend=False))
vis_norm.append(True); vis_cmp.append(False)

# Compare-mode (kept for future; hidden by default)
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False, line=dict(color=COL_PRICE)))
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

y_max = max(
    np.nanmax(init["y_main"]),
    np.nanmax(init["floor"]),
    np.nanmax(init["ceil"])
)
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x",
    hoverdistance=30, spikedistance=30,
    title=f"BTC Purchase Indicator",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title=init["label"], tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=420, t=70, b=70),
)

# ---------------------- HTML (panel + copy + date lock) ----------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)  # all lists -> JSON-safe
genesis_iso    = GENESIS_DATE.strftime("%Y-%m-%d")

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
                  font-variant-numeric: tabular-nums; text-align:right; min-width: 14ch; white-space:pre; }

  /* Hide Plotly tooltip visually but keep events */
  .hoverlayer { opacity: 0 !important; pointer-events: none; }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; height: auto; }
    .right { border-left:none; border-top:1px solid #e5e7eb; }
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
      <div class="row"><div><span style="color:$COL_F;">Floor</span></div><div class="num" id="floor">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_10;">10%</span></div><div class="num" id="p10">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_50;"><b>50%</b></span></div><div class="num" id="p50">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_90;">90%</span></div><div class="num" id="p90">$0.00</div><div></div></div>
      <div class="row"><div><span style="color:$COL_C;">Ceiling</span></div><div class="num" id="ceil">$0.00</div><div></div></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span class="num" id="mainVal">$0.00</span></div>
      <div><b>Position:</b> <span id="bandLbl" style="font-weight:600;color:$COL_50;">—</span>
           <span id="pPct" style="color:#6b7280;">(p≈—%)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = $PRECOMP_JSON;
const GENESIS_ISO  = "$GENESIS_ISO";

const COLORS = {
  floor: "$COL_F", ceil: "$COL_C",
  p10: "$COL_10", p50: "$COL_50", p90: "$COL_90"
};

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

function updatePanel(den,xDays){
  const P=PRECOMP[den];
  const label=P.label || "BTC / USD";
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"yaxis.title.text": label});

  document.querySelector('#readout .date').textContent = dateFromDaysShort(xDays);

  const fl=interp(P.x_grid,P.floor,xDays);
  const ce=interp(P.x_grid,P.ceil,xDays);
  const v10=interp(P.x_grid,P.line10,xDays);
  const v50=interp(P.x_grid,P.line50,xDays);
  const v90=interp(P.x_grid,P.line90,xDays);

  document.getElementById('floor').textContent=fmtUSD(fl);
  document.getElementById('p10').textContent  =fmtUSD(v10);
  document.getElementById('p50').textContent  =fmtUSD(v50);
  document.getElementById('p90').textContent  =fmtUSD(v90);
  document.getElementById('ceil').textContent =fmtUSD(ce);

  // nearest observed price for BTC Price
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-xDays); if(d<best){best=d; idx=i;} }
  const y=ya[idx]; document.getElementById('mainVal').textContent = fmtUSD(y);

  // Position p% in log space between floor and ceiling
  const ly=Math.log10(y), lF=Math.log10(fl), lC=Math.log10(ce);
  const p = Math.max(0, Math.min(1, (ly - lF)/Math.max(1e-12, lC - lF))) * 100.0;
  document.getElementById('pPct').textContent = `(p≈${p.toFixed(1)}%)`;
  // Text label around nearest midline
  let name="~50%"; let color=COLORS.p50;
  const d10=Math.abs(ly - Math.log10(v10));
  const d50=Math.abs(ly - Math.log10(v50));
  const d90=Math.abs(ly - Math.log10(v90));
  if (d10 <= d50 && d10 <= d90) { name="~10%"; color=COLORS.p10; }
  else if (d90 <= d10 && d90 <= d50) { name="~90%"; color=COLORS.p90; }
  document.getElementById('bandLbl').textContent = name;
  document.getElementById('bandLbl').style.color = color;

  // Title reflect current position
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — ${name} (p≈${p.toFixed(1)}%)`});
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

  // Live hover
  plotDiv.on('plotly_hover', function(ev){
    if(!ev.points||!ev.points.length) return;
    if(window._locked) return;
    updatePanel(denomSel.value, ev.points[0].x);
  });

  document.getElementById('setDateBtn').addEventListener('click', function(){
    const val=document.getElementById('datePick').value;
    if(!val) return;
    const d=new Date(val+'T00:00:00Z');
    const d0=new Date(GENESIS_ISO+'T00:00:00Z');
    const xDays=((d.getTime()-d0.getTime())/86400000)+1.0;
    window._locked=true; window._lockedX=xDays;
    updatePanel(denomSel.value, xDays);
  });

  document.getElementById('liveBtn').addEventListener('click', function(){
    window._locked=false; window._lockedX=null;
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

  // Denominator change: restyle lines + axis title + panel
  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];

    // Lines on grid
    const updates = [
      [{x:[P.x_grid], y:[P.floor]}, 0],
      [{x:[P.x_grid], y:[P.line10]}, 1],
      [{x:[P.x_grid], y:[P.line50]}, 2],
      [{x:[P.x_grid], y:[P.line90]}, 3],
      [{x:[P.x_grid], y:[P.ceil]}, 4],
      [{x:[P.x_main], y:[P.y_main], name:[P.label]}, 5],
      [{x:[P.x_main], y:[P.y_main]}, 6],  // cursor
      [{x:[P.x_main], y:[P.main_rebased]}, 7],
      [{x:[P.x_main], y:[P.denom_rebased]}, 8],
    ];
    updates.forEach(u => Plotly.restyle(plotDiv, u[0], [u[1]]));

    // Update axis title
    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label});

    // Update panel (use locked or latest)
    const xTarget = (window._locked && window._lockedX) ?
