#!/usr/bin/env python3
"""
BTC Purchase Indicator — Bitbo-style plot with right-side panel & copy-to-clipboard

- Data autoload (BTC): CoinGecko Pro (if COINGECKO_API_KEY) else Blockchain.com Charts.
- Optional denominators from CSV: data/denominator_*.csv with columns: date,price
- Start date: 2011-01-01; extend quantile bands to 2040-12-31
- Axes: x = log(days since start) with YEAR ticks only (angled), y = log value
  * y shows a faux bottom tick label "0", then 1, 10, 100... with comma separators
  * y-axis title: "BTC / <DENOMINATOR>"
- Right panel: bold date on top, color-coded values per band, main value with commas
- Legend fixed on right; compare-mode preserved (rebased)
- Denominator selector (HTML <select>) updates figure + panel
- Copy Snapshot copies chart + panel; iOS/Safari fallback opens PNG
- Bands enforced non-crossing + slight gap to avoid touching
"""

import os, io, glob, time, math, json
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ---------------------------- CONFIG ----------------------------

OUTPUT_HTML = "docs/index.html"
DATA_DIR    = "data"
BTC_FILE    = os.path.join(DATA_DIR, "btc_usd.csv")

START_DATE  = datetime(2011, 1, 1)
END_PROJ    = datetime(2040, 12, 31)

QUANTILES   = (0.1, 0.3, 0.5, 0.7, 0.9)
Q_ORDER     = [0.1, 0.3, 0.5, 0.7, 0.9]

BAND_COLORS = {
    0.1: "#D32F2F",  # red-ish
    0.3: "#F57C00",  # orange
    0.5: "#FBC02D",  # yellow
    0.7: "#7CB342",  # yellow-green
    0.9: "#2E7D32",  # green
}

BAND_NAMES = {
    0.1: "Bottom 10%",
    0.3: "10–30%",
    0.5: "30–50%",
    0.7: "50–70%",
    0.9: "70–90%",
    1.0: "Top 10%",
}

# band fill order (inner → outer)
BAND_PAIRS = [(0.3, 0.5), (0.5, 0.7), (0.1, 0.3), (0.7, 0.9)]

# minimum *log-space* separation between adjacent quantile curves (soft, to avoid visual touching)
MIN_GAP_LOG10 = 1e-3  # ~0.23% spacing

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

# --------------------------- MATH -------------------------------

def days_since_start(dates: pd.Series, start: datetime) -> pd.Series:
    return (dates - start).dt.days.astype(float) + 1.0  # avoid log(0)

def make_year_ticks(start: datetime, end: datetime):
    # whole-year ticks (Jan 1) -> x_days + label = year
    vals=[]; labs=[]
    year = start.year
    if start.month>1 or start.day>1: year += 1
    while True:
        d = datetime(year,1,1)
        if d > end: break
        vals.append((d - start).days + 1.0)
        labs.append(str(year))
        year += 1
    return vals, labs

def fit_quantile_params(x: np.ndarray, y: np.ndarray, qs=QUANTILES) -> Dict[float, Tuple[float,float]]:
    m=(x>0)&(y>0)&np.isfinite(x)&np.isfinite(y)
    x_use,y_use=x[m],y[m]
    if len(x_use)<10: return {}
    X=pd.DataFrame({"logx":np.log10(x_use)})
    z=np.log10(y_use)
    params={}
    for q in qs:
        try:
            model=QuantReg(z, pd.concat([pd.Series(1.0,index=X.index,name="const"), X],axis=1))
            res=model.fit(q=q)
            params[q]=(float(res.params["const"]), float(res.params["logx"]))
        except Exception: pass
    return params

def predict_grid(params: Dict[float,Tuple[float,float]], x_grid: np.ndarray,
                 enforce=True, min_gap_log10=MIN_GAP_LOG10) -> Dict[float,np.ndarray]:
    # predict each quantile on x_grid
    preds={}
    lx=np.log10(x_grid)
    for q,(a,b) in params.items():
        z=a+b*lx
        preds[q]=10**z

    if not enforce: return preds

    # enforce non-crossing + soft spacing in log-space
    # at each x index, make q10<=q30<=... and separate by min_gap_log10
    qs=sorted(preds.keys())
    if not qs: return preds
    mat=np.vstack([np.log10(preds[q]) for q in qs])  # shape (nq, n)
    for j in range(mat.shape[1]):
        # isotonic-like pass
        for i in range(1, mat.shape[0]):
            mat[i,j] = max(mat[i,j], mat[i-1,j] + min_gap_log10)
    # back to linear
    for i,q in enumerate(qs):
        preds[q]=10**mat[i,:]
    return preds

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=denom_key.lower()
    if k in df.columns:
        return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def rebase_to_one(s: pd.Series) -> pd.Series:
    s=pd.Series(s).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if s.empty or s.iloc[0]<=0: return pd.Series(s)*np.nan
    return pd.Series(s)/s.iloc[0]

def classify_band(y_last: float, x_last: float, params: Dict[float,Tuple[float,float]]) -> Tuple[str,float,str]:
    qs_sorted=sorted(params.keys())
    if len(qs_sorted)<5: return "N/A",0.0,"#222"
    def pred(q): a,b=params[q]; return 10**(a+b*math.log10(x_last))
    q10,q30,q50,q70,q90 = (pred(0.1), pred(0.3), pred(0.5), pred(0.7), pred(0.9))
    if y_last < q10:   return BAND_NAMES[0.1], 0.05, BAND_COLORS[0.1]
    if y_last < q30:   t=(y_last-q10)/max(1e-12,(q30-q10)); return BAND_NAMES[0.3], 0.10+0.20*t, BAND_COLORS[0.3]
    if y_last < q50:   t=(y_last-q30)/max(1e-12,(q50-q30)); return BAND_NAMES[0.5], 0.30+0.20*t, BAND_COLORS[0.5]
    if y_last < q70:   t=(y_last-q50)/max(1e-12,(q70-q50)); return BAND_NAMES[0.7], 0.50+0.20*t, BAND_COLORS[0.7]
    if y_last < q90:   t=(y_last-q70)/max(1e-12,(q90-q70)); return BAND_NAMES[0.9], 0.70+0.20*t, BAND_COLORS[0.9]
    return BAND_NAMES[1.0], 0.95, BAND_COLORS[0.9]

# -------------------------- LOAD DATA ---------------------------

btc = get_btc_df().rename(columns={"price":"btc"})
denoms = collect_denominators()

# merge denoms
base = btc.copy()
for name,df in denoms.items():
    base = base.merge(df.rename(columns={"price":name.lower()}), on="date", how="left")

# trim to start
base = base[base["date"] >= START_DATE].reset_index(drop=True)
if base.empty: raise RuntimeError("No BTC data on/after 2011-01-01")

start_date = base["date"].iloc[0]
base["x_days"] = days_since_start(base["date"], start_date)
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# x-grid to 2040
x_future_end = days_since_start(pd.Series([END_PROJ]), start_date).iloc[0]
x_grid = np.logspace(math.log10(1.0), math.log10(float(x_future_end)), 600)

# -------------------- PRECOMPUTE PER DENOM ---------------------

def build_payload(denom_key: Optional[str]):
    y_main, y_label, denom_series = series_for_denom(base, denom_key)
    x_vals = base["x_days"].values.astype(float)
    params = fit_quantile_params(x_vals, y_main.values, QUANTILES)
    preds  = predict_grid(params, x_grid, enforce=True, min_gap_log10=MIN_GAP_LOG10)

    # band classification at last point
    valid_idx = np.where(np.isfinite(y_main.values))[0]
    last_idx  = int(valid_idx[-1])
    x_last    = float(base["x_days"].iloc[last_idx])
    y_last    = float(y_main.iloc[last_idx])
    band_txt, p_est, color = classify_band(y_last, x_last, params)

    payload = {
        "label": y_label,          # e.g., "BTC / USD"
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "q_lines": {str(q): preds[q].tolist() if q in preds else [] for q in QUANTILES},
        "bands": {
            "0.3-0.5": {"upper": preds.get(0.5, []).tolist() if 0.5 in preds else [],
                        "lower": preds.get(0.3, []).tolist() if 0.3 in preds else []},
            "0.5-0.7": {"upper": preds.get(0.7, []).tolist() if 0.7 in preds else [],
                        "lower": preds.get(0.5, []).tolist() if 0.5 in preds else []},
            "0.1-0.3": {"upper": preds.get(0.3, []).tolist() if 0.3 in preds else [],
                        "lower": preds.get(0.1, []).tolist() if 0.1 in preds else []},
            "0.7-0.9": {"upper": preds.get(0.9, []).tolist() if 0.9 in preds else [],
                        "lower": preds.get(0.7, []).tolist() if 0.7 in preds else []},
        },
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "band_label": band_txt, "percentile": p_est, "line_color": color,
    }
    return payload

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)

# -------------------------- FIGURE ------------------------------

init = PRECOMP["USD"]

# traces (fixed order: 0..15)
traces=[]; vis_norm=[]; vis_cmp=[]

# 0..4 quantile lines on extended grid (no hover; right panel handles it)
for q in Q_ORDER:
    traces.append(go.Scatter(
        x=init["x_grid"], y=init["q_lines"][str(q)], mode="lines",
        name=f"q{int(q*100)}", line=dict(width=0.8, dash="dot", color=BAND_COLORS[q]),
        hoverinfo="skip", showlegend=False
    ))
    vis_norm.append(True); vis_cmp.append(False)

# 5..12 filled bands (inner→outer)
def add_band(pair_key, ql, qh):
    U = init["bands"][pair_key]["upper"]; L = init["bands"][pair_key]["lower"]
    traces.append(go.Scatter(x=init["x_grid"], y=U, mode="lines",
                             line=dict(width=0.5, color=BAND_COLORS[qh]),
                             hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter(x=init["x_grid"], y=L, mode="lines",
                             line=dict(width=0.5, color=BAND_COLORS[ql]),
                             hoverinfo="skip", showlegend=False,
                             fill="tonexty", fillcolor=rgba(BAND_COLORS[qh], 0.18)))
    vis_norm += [True, True]; vis_cmp += [False, False]

add_band("0.3-0.5",0.3,0.5); add_band("0.5-0.7",0.5,0.7); add_band("0.1-0.3",0.1,0.3); add_band("0.7-0.9",0.7,0.9)

# 13 main
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name=init["label"], line=dict(width=1.6, color=init["line_color"]),
                         hoverinfo="skip"))
vis_norm.append(True); vis_cmp.append(False)

# 14 main rebased; 15 denom rebased
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False))
traces.append(go.Scatter(x=init["x_main"], y=init["denom_rebased"], name="Denominator (rebased)",
                         mode="lines", line=dict(dash="dash"), hoverinfo="skip", visible=False))
vis_norm += [False, False]; vis_cmp += [True, True]

fig = go.Figure(data=traces)

# X ticks: whole years (angled)
xtickvals, xticktext = make_year_ticks(start_date, END_PROJ)
# Y ticks: faux "0" then powers of 10 with comma separators
def make_y_ticks(max_y: float):
    exp_max = int(math.ceil(math.log10(max(1.0, max_y))))
    vals=[1e-8] + [10**e for e in range(0, exp_max+1)]
    texts=["0"] + [f"{int(v):,}" for v in [10**e for e in range(0, exp_max+1)]]
    return vals, texts

y_max = max(np.nanmax(init["y_main"]), *(np.nanmax(init["q_lines"][str(q)]) if len(init["q_lines"][str(q)]) else 1.0 for q in Q_ORDER))
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode=False,
    title=f"BTC Purchase Indicator — {init['band_label']} (p≈{init['percentile']:.2f})",
    xaxis=dict(
        type="log", title="Time (log scale)",
        tickmode="array", tickvals=xtickvals, ticktext=xticktext, tickangle=-30,
        range=[math.log10(xtickvals[0]), math.log10(xtickvals[-1])]
    ),
    yaxis=dict(
        type="log",
        title=init["label"],  # already "BTC / <DENOM>"
        tickmode="array", tickvals=ytickvals, ticktext=yticktext
    ),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=360, t=70, b=70),
)

# Mode buttons
vis_cmp_mask=[False]*16; vis_cmp_mask[14]=True; vis_cmp_mask[15]=True
mode_menu=dict(
    buttons=[
        dict(label="Normal", method="update",
             args=[{"visible": vis_norm}, {"yaxis.title.text": init["label"]}]),
        dict(label="Compare (Rebased)", method="update",
             args=[{"visible": vis_cmp_mask}, {"yaxis.title.text": "Rebased to 1.0 (log scale)"}]),
    ],
    direction="down", showactive=True, x=0.0, y=1.18, xanchor="left", yanchor="top"
)
fig.update_layout(updatemenus=[mode_menu])

# ---------------------- HTML (panel + copy) --------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)
band_colors_js = json.dumps({str(k): v for k,v in BAND_COLORS.items()})
band_names_js  = json.dumps(BAND_NAMES)
xgrid_json     = json.dumps(init["x_grid"])
start_iso      = start_date.strftime("%Y-%m-%d")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root {{ --panelW: 320px; }}
  body {{ font-family: Inter, Roboto, -apple-system, Segoe UI, Arial, sans-serif; margin: 0; }}
  .layout {{
    display: grid;
    grid-template-columns: 1fr var(--panelW);
    height: 100vh;
  }}
  .left {{ padding: 8px 0 8px 8px; }}
  .right {{
    border-left: 1px solid #e5e7eb; padding: 12px; display: flex; flex-direction: column; gap: 12px;
  }}
  #readout {{ border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; font-size:14px; }}
  #readout .date {{ font-weight:700; margin-bottom:6px; }}
  #controls select, #controls button {{
    font-size:14px; padding:8px 10px; border-radius:8px; border:1px solid #d1d5db; background:white;
  }}
  #copyBtn {{ cursor:pointer; }}
  /* Hide Plotly hover popups (we use the side panel) */
  .hoverlayer {{ display:none !important; }}

  /* Mobile (Safari-friendly) */
  @media (max-width: 900px) {{
    .layout {{ grid-template-columns: 1fr; height: auto; }}
    .right {{ border-left:none; border-top:1px solid #e5e7eb; }}
    .left .js-plotly-plot {{ max-width: 100vw; }}
  }}
</style>
</head>
<body>
<div id="capture" class="layout">
  <div class="left">{plot_html}</div>
  <div class="right">
    <div id="controls" style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <label for="denomSel"><b>Denominator:</b></label>
      <select id="denomSel"></select>
      <button id="copyBtn">Copy Snapshot</button>
    </div>
    <div style="font-size:12px;color:#6b7280;">
      Detected denominators: <span id="denomsDetected"></span>
    </div>
    <div id="readout">
      <div class="date">—</div>
      <div id="bands">
        <div><span style="color:{BAND_COLORS[0.1]};">q10</span>: <span id="q10">—</span> <em>({BAND_NAMES[0.1]})</em></div>
        <div><span style="color:{BAND_COLORS[0.3]};">q30</span>: <span id="q30">—</span> <em>({BAND_NAMES[0.3]})</em></div>
        <div><span style="color:{BAND_COLORS[0.5]};">q50</span>: <span id="q50">—</span> <em>({BAND_NAMES[0.5]})</em></div>
        <div><span style="color:{BAND_COLORS[0.7]};">q70</span>: <span id="q70">—</span> <em>({BAND_NAMES[0.7]})</em></div>
        <div><span style="color:{BAND_COLORS[0.9]};">q90</span>: <span id="q90">—</span> <em>({BAND_NAMES[0.9]})</em></div>
      </div>
      <div style="margin-top:8px;"><b>Main:</b> <span id="mainVal">—</span></div>
      <div><b>Band:</b> <span id="bandLbl">{init['band_label']}</span> <span style="color:#6b7280;">(p≈{init['percentile']:.2f})</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = {json.dumps(PRECOMP)};
const BAND_COLORS  = {band_colors_js};
const BAND_NAMES   = {band_names_js};
const XGRID        = {xgrid_json};
const START_ISO    = "{start_iso}";

// Build denominator selector
const denomSel = document.getElementById('denomSel');
const detected = Object.keys(PRECOMP).filter(k => k !== 'USD');
document.getElementById('denomsDetected').textContent = detected.length ? detected.join(', ') : '(none)';
['USD', ...detected].forEach(k => {
  const opt=document.createElement('option'); opt.value=k; opt.textContent=(k==='USD'?'USD/None':k);
  denomSel.appendChild(opt);
});

// helpers
function numFmt(v){
  if(!isFinite(v)) return '—';
  // Comma formatting for USD-like magnitudes
  if (v >= 1000) return Number(v).toLocaleString(undefined, {maximumFractionDigits: 2});
  if (v >= 1) return Number(v).toLocaleString(undefined, {maximumFractionDigits: 6});
  // small values
  return Number(v).toExponential(2);
}
function dateFromX(x){
  const d0=new Date(START_ISO+'T00:00:00Z');
  const d=new Date(d0.getTime() + (x-1)*86400000);
  return `${String(d.getUTCMonth()+1).padStart(2,'0')}/${String(d.getUTCDate()).padStart(2,'0')}/${String(d.getUTCFullYear()).slice(-2)}`;
}
function interp(xArr,yArr,x){
  let lo=0,hi=xArr.length-1;
  if(x<=xArr[0]) return yArr[0];
  if(x>=xArr[hi]) return yArr[hi];
  while(hi-lo>1){ const m=(hi+lo)>>1; if(xArr[m]<=x) lo=m; else hi=m; }
  const t=(x-xArr[lo])/(xArr[hi]-xArr[lo]); return yArr[lo]+t*(yArr[hi]-yArr[lo]);
}

// right panel hooks
const elDate=document.querySelector('#readout .date');
const elQ10=document.getElementById('q10'), elQ30=document.getElementById('q30'),
      elQ50=document.getElementById('q50'), elQ70=document.getElementById('q70'), elQ90=document.getElementById('q90');
const elMain=document.getElementById('mainVal'), elBand=document.getElementById('bandLbl');

function updatePanel(den,x){
  const P=PRECOMP[den];
  elDate.textContent = dateFromX(x);
  const q10=P.q_lines["0.1"].length?interp(P.x_grid,P.q_lines["0.1"],x):NaN;
  const q30=P.q_lines["0.3"].length?interp(P.x_grid,P.q_lines["0.3"],x):NaN;
  const q50=P.q_lines["0.5"].length?interp(P.x_grid,P.q_lines["0.5"],x):NaN;
  const q70=P.q_lines["0.7"].length?interp(P.x_grid,P.q_lines["0.7"],x):NaN;
  const q90=P.q_lines["0.9"].length?interp(P.x_grid,P.q_lines["0.9"],x):NaN;
  elQ10.textContent=numFmt(q10); elQ30.textContent=numFmt(q30);
  elQ50.textContent=numFmt(q50); elQ70.textContent=numFmt(q70); elQ90.textContent=numFmt(q90);
  // main ~ nearest x
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-x); if(d<best){best=d; idx=i;} }
  elMain.textContent = numFmt(ya[idx]);
  elBand.textContent = P.band_label;
}

// connect hover to panel
document.addEventListener('DOMContentLoaded', ()=>{
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  plotDiv.on('plotly_hover', ev=>{
    if(!ev.points||!ev.points.length) return;
    updatePanel(denomSel.value, ev.points[0].x);
  });
});

// denom change → update traces & labels
denomSel.addEventListener('change', ()=>{
  const key=denomSel.value, P=PRECOMP[key];
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  // 0..4 q-lines
  const qs=['0.1','0.3','0.5','0.7','0.9'];
  for(let i=0;i<qs.length;i++){
    Plotly.restyle(plotDiv, {x:[P.x_grid], y:[P.q_lines[qs[i]]], name:[`q${parseInt(parseFloat(qs[i])*100)}`]}, [i]);
  }
  // 5..12 bands
  function setBand(slot,key){ const U=P.bands[key]?.upper||[], L=P.bands[key]?.lower||[];
    Plotly.restyle(plotDiv, {x:[P.x_grid], y:[U]}, [slot]);
    Plotly.restyle(plotDiv, {x:[P.x_grid], y:[L]}, [slot+1]);
  }
  setBand(5,"0.3-0.5"); setBand(7,"0.5-0.7"); setBand(9,"0.1-0.3"); setBand(11,"0.7-0.9");
  // 13 main, 14 main rebased, 15 denom rebased
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.y_main], name:[P.label], "line.color":[P.line_color]}, [13]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.main_rebased]}, [14]);
  Plotly.restyle(plotDiv, {x:[P.x_main], y:[P.denom_rebased]}, [15]);
  Plotly.relayout(plotDiv, {"yaxis.title.text": P.label, "title.text": `BTC Purchase Indicator — ${P.band_label} (p≈${P.percentile.toFixed(2)})`});
  updatePanel(key, P.x_main[P.x_main.length-1]);
});

// initial panel state
updatePanel('USD', PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);

// copy snapshot (chart + panel). iOS Safari fallback opens image.
document.getElementById('copyBtn').addEventListener('click', async ()=>{
  const node=document.getElementById('capture');
  try{
    const canvas=await htmlToImage.toCanvas(node,{pixelRatio:2});
    const blob=await new Promise(res=>canvas.toBlob(res,'image/png'));
    if(navigator.clipboard && window.ClipboardItem){
      await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
    }else{
      const url=URL.createObjectURL(blob); const win=window.open(); win.document.write('<img src="'+url+'"/>');
    }
  }catch(e){ console.error(e); alert('Copy failed; your browser may block clipboard images.'); }
});
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
