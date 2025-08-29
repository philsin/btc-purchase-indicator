#!/usr/bin/env python3
"""
BTC Purchase Indicator — Log–Log power-law with YEAR labels, cursor+date panel, non-crossing bands

- Model: log10(price) ~ a + b * log10(days since Genesis).
- X axis: log(days since 2009-01-03), tick labels are calendar YEARS mapped to that scale.
- Y axis: log; faux "0" tick then 1,10,100,... with comma separators.
- Start at the first available datapoint; extend projections to 2040-12-31.
- Bands: q10/q30/q50/q70/q90; non-crossing via rearrangement + small log-gap; filled correctly (lower→upper).
- BTC price line: black.
- Right panel tracks cursor unless a date is locked (date picker + buttons).
- Copy Chart: tries clipboard, else silently downloads PNG (no alert).
- Compare mode: rebased lines, hides bands+annotations.
"""

import os, io, glob, time, math, json
from typing import Optional, Dict, Tuple
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

QUANTILES     = (0.1, 0.3, 0.5, 0.7, 0.9)
Q_ORDER       = [0.1, 0.3, 0.5, 0.7, 0.9]

BAND_COLORS = {
    0.1: "#D32F2F",
    0.3: "#F57C00",
    0.5: "#FBC02D",
    0.7: "#7CB342",
    0.9: "#2E7D32",
}
BAND_NAMES = {
    0.1: "Bottom 10%",
    0.3: "10–30%",
    0.5: "30–50%",
    0.7: "50–70%",
    0.9: "70–90%",
    1.0: "Top 10%",
}

MIN_GAP_LOG10 = 1e-3  # ~0.23% in linear space; prevents bands from visually touching

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
    """Vectorized days-since-Genesis that works for Series, arrays, or DatetimeIndex."""
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

def fit_quantile_params(x_days: np.ndarray, y: np.ndarray, qs=QUANTILES) -> Dict[float, Tuple[float,float]]:
    """
    Power-law in time: log10(y) = a + b * log10(days).
    Fit per-quantile, then monotone-in-quantile (cumulative max) on parameters to reduce crossings.
    """
    m=(x_days>0)&(y>0)&np.isfinite(x_days)&np.isfinite(y)
    x=x_days[m]; z=np.log10(y[m])
    if len(x)<10: return {}
    X=pd.DataFrame({"logx":np.log10(x)})

    qs_sorted=sorted(qs)
    A,B=[],[]
    for q in qs_sorted:
        try:
            model=QuantReg(z, pd.concat([pd.Series(1.0,index=X.index,name="const"), X],axis=1))
            res=model.fit(q=q)
            A.append(float(res.params["const"]))
            B.append(float(res.params["logx"]))
        except Exception:
            A.append(np.nan); B.append(np.nan)

    qq=np.array(qs_sorted, dtype=float)
    A=np.array(A, dtype=float); B=np.array(B, dtype=float)
    if np.any(~np.isfinite(A)):
        mask=np.isfinite(A); A[~mask]=np.interp(qq[~mask], qq[mask], A[mask])
    if np.any(~np.isfinite(B)):
        mask=np.isfinite(B); B[~mask]=np.interp(qq[~mask], qq[mask], B[mask])

    # enforce monotonic parameters
    A=np.maximum.accumulate(A)
    B=np.maximum.accumulate(B)

    return {float(q):(float(a), float(b)) for q,a,b in zip(qs_sorted, A, B)}

def predict_from_params(params: Dict[float,Tuple[float,float]],
                        x_days_grid: np.ndarray,
                        enforce: bool=True,
                        min_gap_log10: float=MIN_GAP_LOG10) -> Dict[float,np.ndarray]:
    """
    Predict curves; then apply 'rearrangement' per x (sort across quantiles) + small log-gap.
    Guarantees non-crossing & avoids touching fills.
    """
    if not params: return {}
    qs=sorted(params.keys())
    lx=np.log10(x_days_grid)
    mat_log=[]
    for q in qs:
        a,b=params[q]
        mat_log.append(a + b*lx)
    mat_log=np.vstack(mat_log)  # (nq, n)

    if enforce:
        mat_log=np.sort(mat_log, axis=0)  # rearrangement ensures order
        for i in range(1, mat_log.shape[0]):  # add tiny separation
            mat_log[i,:]=np.maximum(mat_log[i,:], mat_log[i-1,:] + min_gap_log10)

    return {q:10**mat_log[i,:] for i,q in enumerate(qs)}

def rebase_to_one(s: pd.Series) -> pd.Series:
    s=pd.Series(s).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if s.empty or s.iloc[0]<=0: return pd.Series(s)*np.nan
    return pd.Series(s)/s.iloc[0]

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC / USD", None
    k=denom_key.lower()
    if k in df.columns:
        return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC / USD", None

def classify_band(y_last: float, x_last_days: float, params: Dict[float,Tuple[float,float]]):
    if len(params)<5: return "N/A",0.0,"#000"
    def pred(q): a,b=params[q]; return 10**(a+b*math.log10(x_last_days))
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

# Start at first datapoint; merge denominators if present
base = btc.sort_values("date").reset_index(drop=True)
if base.empty: raise RuntimeError("No BTC data found")
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

base["x_days"]   = days_since_genesis(base["date"], GENESIS_DATE)
base["date_iso"] = base["date"].dt.strftime("%Y-%m-%d")
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# X grid (log-spaced) from first data day to END_PROJ
x_start = float(base["x_days"].iloc[0])
x_end   = float(days_since_genesis(pd.Series([END_PROJ]), GENESIS_DATE).iloc[0])
x_grid  = np.logspace(np.log10(max(1.0, x_start)), np.log10(x_end), 600)

# Year ticks mapped to log(days), hiding odd years after 2030
def year_ticks_log(first_date: datetime, last_date: datetime):
    y0, y1 = first_date.year, last_date.year
    vals, labs = [], []
    for y in range(y0, y1+1):
        d = datetime(y, 1, 1)
        if d < first_date or d > last_date: continue
        dv = float(days_since_genesis(pd.Series([d]), GENESIS_DATE).iloc[0])
        if dv <= 0: continue
        if y > 2030 and (y % 2 == 1):  # hide odd years after 2030
            continue
        vals.append(dv); labs.append(str(y))
    return vals, labs

first_date = base["date"].iloc[0].to_pydatetime()
xtickvals, xticktext = year_ticks_log(first_date, END_PROJ)

# -------------------- PRECOMPUTE PER DENOM ---------------------

def build_payload(denom_key: Optional[str]):
    y_main, y_label, denom_series = series_for_denom(base, denom_key)
    x_vals = base["x_days"].values.astype(float)

    params = fit_quantile_params(x_vals, y_main.values, QUANTILES)
    preds  = predict_from_params(params, x_grid, enforce=True, min_gap_log10=MIN_GAP_LOG10)

    valid_idx = np.where(np.isfinite(y_main.values))[0]
    last_idx  = int(valid_idx[-1])
    x_last    = float(base["x_days"].iloc[last_idx])
    y_last    = float(y_main.iloc[last_idx])
    band_txt, p_est, _ = classify_band(y_last, x_last, params)

    return {
        "label": y_label,
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
        "band_label": band_txt, "percentile": p_est,
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)

init = PRECOMP["USD"]

# -------------------------- FIGURE ------------------------------

traces=[]; vis_norm=[]; vis_cmp=[]

# 0..4 dotted quantile lines
for q in Q_ORDER:
    traces.append(go.Scatter(
        x=init["x_grid"], y=init["q_lines"][str(q)], mode="lines",
        name=f"q{int(q*100)}", line=dict(width=0.8, dash="dot", color=BAND_COLORS[q]),
        hoverinfo="skip", showlegend=False
    ))
    vis_norm.append(True); vis_cmp.append(False)

# 5..12 filled bands — draw LOWER first, then UPPER with fill='tonexty'
def add_band(pair_key, ql, qh):
    L = init["bands"][pair_key]["lower"]; U = init["bands"][pair_key]["upper"]
    traces.append(go.Scatter(x=init["x_grid"], y=L, mode="lines",
                             line=dict(width=0.6, color=BAND_COLORS[ql]),
                             hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter(x=init["x_grid"], y=U, mode="lines",
                             line=dict(width=0.6, color=BAND_COLORS[qh]),
                             hoverinfo="skip", showlegend=False,
                             fill="tonexty", fillcolor=rgba(BAND_COLORS[qh], 0.18)))
    vis_norm.extend([True, True]); vis_cmp.extend([False, False])

add_band("0.3-0.5",0.3,0.5); add_band("0.5-0.7",0.5,0.7); add_band("0.1-0.3",0.1,0.3); add_band("0.7-0.9",0.7,0.9)

# 13 main BTC (black)
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="BTC / USD", line=dict(width=1.9, color="#000000"),
                         hoverinfo="skip"))
vis_norm.append(True); vis_cmp.append(False)

# 14 main rebased; 15 denom rebased (Compare mode only)
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False, line=dict(color="#000000")))
traces.append(go.Scatter(x=init["x_main"], y=init["denom_rebased"], name="Denominator (rebased)",
                         mode="lines", line=dict(dash="dash"), hoverinfo="skip", visible=False))
vis_norm.extend([False, False]); vis_cmp.extend([True, True])

fig = go.Figure(data=traces)

# Y ticks: faux "0" then powers of 10
def make_y_ticks(max_y: float):
    exp_max = int(math.ceil(math.log10(max(1.0, max_y))))
    vals=[1e-8] + [10**e for e in range(0, exp_max+1)]
    texts=["0"] + [f"{int(10**e):,}" for e in range(0, exp_max+1)]
    return vals, texts

y_candidates = [np.nanmax(init["y_main"])]
for q in Q_ORDER:
    arr = init["q_lines"][str(q)]
    if len(arr): y_candidates.append(np.nanmax(arr))
y_max = max(y_candidates)
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode=False,
    title=f"BTC Purchase Indicator — {init['band_label']} (p≈{init['percentile']:.2f})",
    xaxis=dict(
        type="log",
        title=None,                               # no x-axis title
        tickmode="array", tickvals=xtickvals, ticktext=xticktext
    ),
    yaxis=dict(
        type="log",
        title="BTC / <DENOMINATOR>",
        tickmode="array", tickvals=ytickvals, ticktext=yticktext
    ),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=380, t=70, b=70),
)

# Band label annotations (Normal mode)
def band_annotations_for(preds: Dict[str, list], x_vals: list):
    if not x_vals: return []
    idx = int(0.82 * (len(x_vals)-1))  # near right edge
    def gm(a,b): return math.sqrt(a*b)
    ann=[]
    q10=preds.get("0.1",[]); q30=preds.get("0.3",[]); q50=preds.get("0.5",[])
    q70=preds.get("0.7",[]); q90=preds.get("0.9",[])
    if all(len(arr)>idx for arr in [q10,q30,q50,q70,q90]):
        x=x_vals[idx]
        pts=[
            (x, q10[idx]*0.80, BAND_NAMES[0.1]),
            (x, gm(q10[idx], q30[idx]), BAND_NAMES[0.3]),
            (x, gm(q30[idx], q50[idx]), BAND_NAMES[0.5]),
            (x, gm(q50[idx], q70[idx]), BAND_NAMES[0.7]),
            (x, gm(q70[idx], q90[idx]), BAND_NAMES[0.9]),
            (x, q90[idx]*1.25, BAND_NAMES[1.0]),
        ]
        for xx,yy,txt in pts:
            ann.append(dict(x=xx,y=yy,xref="x",yref="y",text=txt,showarrow=False,
                            font=dict(size=12,color="#111"), bgcolor="rgba(255,255,255,0.6)",
                            bordercolor="rgba(0,0,0,0.1)", borderwidth=1, borderpad=3))
    return ann

initial_annotations = band_annotations_for(
    {k: init["q_lines"][k] for k in ["0.1","0.3","0.5","0.7","0.9"]}, init["x_grid"]
)
fig.update_layout(annotations=initial_annotations)

# Mode buttons (hide bands+annotations in Compare)
vis_cmp_mask=[False]*len(traces); vis_cmp_mask[14]=True; vis_cmp_mask[15]=True
mode_menu=dict(
    buttons=[
        dict(label="Normal", method="update",
             args=[{"visible": vis_norm},
                   {"yaxis.title.text": "BTC / <DENOMINATOR>", "annotations": initial_annotations}]),
        dict(label="Compare (Rebased)", method="update",
             args=[{"visible": vis_cmp_mask},
                   {"yaxis.title.text": "Rebased to 1.0 (log scale)", "annotations": []}]),
    ],
    direction="down", showactive=True, x=0.0, y=1.18, xanchor="left", yanchor="top"
)
fig.update_layout(updatemenus=[mode_menu])

# ---------------------- HTML (panel + copy + date lock) ----------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)
band_colors_js = json.dumps({str(k): v for k,v in BAND_COLORS.items()})
band_names_js  = json.dumps(BAND_NAMES)
genesis_iso    = GENESIS_DATE.strftime("%Y-%m-%d")

html_tpl = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  :root { --panelW: 380px; }
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
  #readout .row { display:flex; justify-content:space-between; gap:8px; }
  #readout .row div { white-space: nowrap; }

  .hoverlayer { display:none !important; } /* drive panel ourselves */

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
      <div id="bands">
        <div class="row"><div><span style="color:$COL_10;">q10</span></div><div id="q10">—</div><div><em>(Bottom 10%)</em></div></div>
        <div class="row"><div><span style="color:$COL_30;">q30</span></div><div id="q30">—</div><div><em>(10–30%)</em></div></div>
        <div class="row"><div><span style="color:$COL_50;">q50</span></div><div id="q50">—</div><div><em>(30–50%)</em></div></div>
        <div class="row"><div><span style="color:$COL_70;">q70</span></div><div id="q70">—</div><div><em>(50–70%)</em></div></div>
        <div class="row"><div><span style="color:$COL_90;">q90</span></div><div id="q90">—</div><div><em>(70–90%)</em></div></div>
      </div>
      <div style="margin-top:8px;"><b>Main:</b> <span id="mainVal">—</span></div>
      <div><b>Band:</b> <span id="bandLbl">$INIT_BAND</span> <span style="color:#6b7280;">(p≈$INIT_PCT)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = $PRECOMP_JSON;
const BAND_COLORS  = $BAND_COLORS_JS;
const BAND_NAMES   = $BAND_NAMES_JS;
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

function numFmt(v){
  if(!isFinite(v)) return '—';
  if (v >= 1000) return Number(v).toLocaleString(undefined, {maximumFractionDigits: 2});
  if (v >= 1)    return Number(v).toLocaleString(undefined, {maximumFractionDigits: 6});
  return Number(v).toExponential(2);
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

// Panel elements
const elDate=document.querySelector('#readout .date');
const elQ10=document.getElementById('q10'), elQ30=document.getElementById('q30'),
      elQ50=document.getElementById('q50'), elQ70=document.getElementById('q70'), elQ90=document.getElementById('q90');
const elMain=document.getElementById('mainVal'), elBand=document.getElementById('bandLbl');

// Color the numeric values to match bands
elQ10.style.color = "$COL_10"; elQ30.style.color = "$COL_30"; elQ50.style.color = "$COL_50";
elQ70.style.color = "$COL_70"; elQ90.style.color = "$COL_90";

let locked=false, lockedX=null;

function updatePanel(den,xDays){
  const P=PRECOMP[den];
  elDate.textContent = dateFromDaysShort(xDays);
  const q10=P.q_lines["0.1"].length?interp(P.x_grid,P.q_lines["0.1"],xDays):NaN;
  const q30=P.q_lines["0.3"].length?interp(P.x_grid,P.q_lines["0.3"],xDays):NaN;
  const q50=P.q_lines["0.5"].length?interp(P.x_grid,P.q_lines["0.5"],xDays):NaN;
  const q70=P.q_lines["0.7"].length?interp(P.x_grid,P.q_lines["0.7"],xDays):NaN;
  const q90=P.q_lines["0.9"].length?interp(P.x_grid,P.q_lines["0.9"],xDays):NaN;
  elQ10.textContent=numFmt(q10); elQ30.textContent=numFmt(q30);
  elQ50.textContent=numFmt(q50); elQ70.textContent=numFmt(q70); elQ90.textContent=numFmt(q90);
  // nearest observed price
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-xDays); if(d<best){best=d; idx=i;} }
  elMain.textContent = numFmt(ya[idx]);
  // simple band name by where price sits vs q's
  let band = "Top 10%";
  if (ya[idx] < q10) band = "Bottom 10%";
  else if (ya[idx] < q30) band = "10–30%";
  else if (ya[idx] < q50) band = "30–50%";
  else if (ya[idx] < q70) band = "50–70%";
  else if (ya[idx] < q90) band = "70–90%";
  elBand.textContent = band;
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

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

  // Copy Chart: try clipboard; if blocked, download PNG silently
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
      // fallback: download
      const a=document.createElement('a');
      a.href=dataUrl; a.download='btc-indicator.png';
      document.body.appendChild(a); a.click(); a.remove();
    }catch(e){ console.error(e); }
  });

  // Denominator changes: restyle traces + annotations + panel
  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];
    const qs=['0.1','0.3','0.5','0.7','0.9'];
    for(let i=0;i<qs.length;i++){
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[P.q_lines[qs[i]]] }, [i]);
    }
    function setBand(slot,keyBand){ const U=P.bands[keyBand]?.upper||[], L=P.bands[keyBand]?.lower||[];
      // lower then upper with fill
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[L] }, [slot]);
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[U] }, [slot+1]);
    }
    setBand(5,"0.3-0.5"); setBand(7,"0.5-0.7"); setBand(9,"0.1-0.3"); setBand(11,"0.7-0.9");

    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [13]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.main_rebased] }, [14]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.denom_rebased] }, [15]);

    // recompute annotations for this denom (Normal mode)
    function bandAnnotations(P){
      const xs=P.x_grid; const idx=Math.max(0, Math.floor(0.82*(xs.length-1)));
      const q10=P.q_lines["0.1"], q30=P.q_lines["0.3"], q50=P.q_lines["0.5"], q70=P.q_lines["0.7"], q90=P.q_lines["0.9"];
      if(!q10.length||!q30.length||!q50.length||!q70.length||!q90.length) return [];
      const gm=(a,b)=>Math.sqrt(a*b), x=xs[idx];
      return [
        {x, y:q10[idx]*0.80, text:"Bottom 10%"},
        {x, y:gm(q10[idx],q30[idx]), text:"10–30%"},
        {x, y:gm(q30[idx],q50[idx]), text:"30–50%"},
        {x, y:gm(q50[idx],q70[idx]), text:"50–70%"},
        {x, y:gm(q70[idx],q90[idx]), text:"70–90%"},
        {x, y:q90[idx]*1.25, text:"Top 10%"},
      ].map(p => ({...p, xref:"x", yref:"y", showarrow:false, font:{size:12,color:"#111"},
                   bgcolor:"rgba(255,255,255,0.6)", bordercolor:"rgba(0,0,0,0.1)", borderwidth:1, borderpad:3}));
    }
    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label, "annotations": bandAnnotations(P)});

    // set panel at latest or locked
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
    BAND_COLORS_JS=band_colors_js,
    BAND_NAMES_JS=band_names_js,
    GENESIS_ISO=genesis_iso,
    COL_10=BAND_COLORS[0.1],
    COL_30=BAND_COLORS[0.3],
    COL_50=BAND_COLORS[0.5],
    COL_70=BAND_COLORS[0.7],
    COL_90=BAND_COLORS[0.9],
    INIT_BAND=init["band_label"],
    INIT_PCT=f"{init['percentile']:.2f}",
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
