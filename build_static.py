#!/usr/bin/env python3
"""
BTC Purchase Indicator — power-law (log–log) with constant log-offset bands

- Trend: log10(price) = a + b*log10(days since Genesis)
- Bands: parallel lines via residual quantiles c_p so P_p(t) = 10^c_p * A * t^b
- Non-crossing by construction + tiny log-gap so fills don't touch
- BTC line = black
- Live hover (panel follows cursor); lockable date picker
- Panel: $ currency, right aligned (monospace), "BTC Price" label, Band label colored, p as %
- X-axis title removed; hide odd years after 2026
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

QUANTILES     = (0.1, 0.3, 0.5, 0.7, 0.9)
Q_ORDER       = [0.1, 0.3, 0.5, 0.7, 0.9]

BAND_COLORS = {
    0.1: "#D32F2F",  # Bottom
    0.3: "#F57C00",
    0.5: "#FBC02D",
    0.7: "#7CB342",
    0.9: "#2E7D32",  # Top- band color used below q90, Top 10% uses same family
}
BAND_NAMES = {
    0.1: "Bottom 10%",
    0.3: "10–30%",
    0.5: "30–50%",
    0.7: "50–70%",
    0.9: "70–90%",
    1.0: "Top 10%",
}

MIN_GAP_LOG10 = 1e-3   # ~0.23% in linear; keeps bands from visually touching

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

def residual_quantiles(resid: pd.Series, qs: Tuple[float,...]) -> Dict[float,float]:
    """Quantiles in log space; enforce strict monotonicity with tiny gap."""
    cq = {float(q): float(np.nanquantile(resid, q)) for q in qs}
    order = sorted(cq.keys())
    for i in range(1,len(order)):
        prev,cur = cq[order[i-1]], cq[order[i]]
        if cur <= prev + MIN_GAP_LOG10:
            cq[order[i]] = prev + MIN_GAP_LOG10
    return cq

def predict_parallel_bands(a: float, b: float, cq: Dict[float,float], x_days_grid: np.ndarray) -> Dict[str, list]:
    """Return dict of q-lines (price) for xgrid using constant log-offset bands, as plain Python lists."""
    lx = np.log10(x_days_grid)
    trend_log = a + b*lx
    out={}
    for q,val in cq.items():
        ylog = trend_log + val
        out[str(q)] = (10**ylog).tolist()
    return out

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

    # 1) trend (median)
    a,b,resid = fit_trend_median(x_vals, y_main.values)
    # 2) residual quantiles for constant parallel bands
    cq = residual_quantiles(resid, QUANTILES)
    # 3) predictions on x_grid -> lists
    q_lines = predict_parallel_bands(a,b,cq,x_grid)

    # classify latest
    valid_idx = np.where(np.isfinite(y_main.values))[0]
    last_idx  = int(valid_idx[-1])
    x_last    = float(base["x_days"].iloc[last_idx])
    y_last    = float(y_main.iloc[last_idx])

    r_last = math.log10(y_last) - (a + b*math.log10(x_last))
    edges=[0.1,0.3,0.5,0.7,0.9]
    band_lbl="Top 10%"; band_col=BAND_COLORS[0.9]; p=95.0
    if r_last < cq[0.1]:
        band_lbl="Bottom 10%"; band_col=BAND_COLORS[0.1]; p=5.0
    else:
        for i in range(len(edges)-1):
            ql, qh = edges[i], edges[i+1]
            el, eh = cq[ql], cq[qh]
            if r_last < eh:
                t = (r_last - el) / max(1e-12, (eh - el))
                p = (ql + (qh-ql)*float(np.clip(t,0,1))) * 100.0
                if   ql==0.1: band_lbl="10–30%"; band_col=BAND_COLORS[0.3]
                elif ql==0.3: band_lbl="30–50%"; band_col=BAND_COLORS[0.5]
                elif ql==0.5: band_lbl="50–70%"; band_col=BAND_COLORS[0.7]
                elif ql==0.7: band_lbl="70–90%"; band_col=BAND_COLORS[0.9]
                break

    # build bands dict using lists (no .tolist())
    bands = {
        "0.1-0.3": {"lower": q_lines["0.1"], "upper": q_lines["0.3"]},
        "0.3-0.5": {"lower": q_lines["0.3"], "upper": q_lines["0.5"]},
        "0.5-0.7": {"lower": q_lines["0.5"], "upper": q_lines["0.7"]},
        "0.7-0.9": {"lower": q_lines["0.7"], "upper": q_lines["0.9"]},
    }

    return {
        "label": y_label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "q_lines": q_lines,              # lists
        "bands": bands,                  # lists
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "band_label": band_lbl, "percentile": p, "band_color": band_col,
        "trend_params": {"a":a, "b":b, "cq": cq},
    }

PRECOMP = {"USD": build_payload(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = build_payload(k)
init = PRECOMP["USD"]

# -------------------------- FIGURE ------------------------------

traces=[]; vis_norm=[]; vis_cmp=[]

# Quantile guide lines (dotted)
for q in Q_ORDER:
    traces.append(go.Scatter(
        x=init["x_grid"], y=init["q_lines"][str(q)], mode="lines",
        name=f"q{int(q*100)}", line=dict(width=0.8, dash="dot", color=BAND_COLORS[q]),
        hoverinfo="skip", showlegend=False
    ))
    vis_norm.append(True); vis_cmp.append(False)

# Filled bands — draw LOWER first, then UPPER with fill='tonexty'
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

add_band("0.1-0.3",0.1,0.3); add_band("0.3-0.5",0.3,0.5); add_band("0.5-0.7",0.5,0.7); add_band("0.7-0.9",0.7,0.9)

# BTC price (black)
traces.append(go.Scatter(x=init["x_main"], y=init["y_main"], mode="lines",
                         name="BTC / USD", line=dict(width=1.9, color="#000000"),
                         hoverinfo="skip"))
vis_norm.append(True); vis_cmp.append(False)

# Compare-mode lines
traces.append(go.Scatter(x=init["x_main"], y=init["main_rebased"], name="Main (rebased)",
                         mode="lines", hoverinfo="skip", visible=False, line=dict(color="#000000")))
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

y_candidates = [np.nanmax(init["y_main"])] + [np.nanmax(init["q_lines"][str(q)]) for q in Q_ORDER if len(init["q_lines"][str(q)])]
y_max = max(y_candidates)
ytickvals, yticktext = make_y_ticks(y_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x",  # keep events for live hover
    hoverdistance=30, spikedistance=30,
    title=f"BTC Purchase Indicator — {init['band_label']} (p≈{init['percentile']:.1f}%)",
    xaxis=dict(type="log", title=None, tickmode="array", tickvals=xtickvals, ticktext=xticktext),
    yaxis=dict(type="log", title="BTC / <DENOMINATOR>", tickmode="array", tickvals=ytickvals, ticktext=yticktext),
    legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.0)"),
    margin=dict(l=70, r=400, t=70, b=70),
)

# Band annotations (Normal mode)
def band_annotations_for(q_lines: Dict[str, List[float]], xs: List[float]):
    if not xs: return []
    idx = max(0, int(0.82*(len(xs)-1)))
    def gm(a,b): return math.sqrt(a*b)
    q10=q_lines["0.1"]; q30=q_lines["0.3"]; q50=q_lines["0.5"]; q70=q_lines["0.7"]; q90=q_lines["0.9"]
    if not all(len(v)>idx for v in [q10,q30,q50,q70,q90]): return []
    x=xs[idx]
    pts=[
        (x, q10[idx]*0.80, "Bottom 10%"),
        (x, gm(q10[idx],q30[idx]), "10–30%"),
        (x, gm(q30[idx],q50[idx]), "30–50%"),
        (x, gm(q50[idx],q70[idx]), "50–70%"),
        (x, gm(q70[idx],q90[idx]), "70–90%"),
        (x, q90[idx]*1.25, "Top 10%"),
    ]
    return [dict(x=xx,y=yy,xref="x",yref="y",text=txt,showarrow=False,
                 font=dict(size=12,color="#111"), bgcolor="rgba(255,255,255,0.6)",
                 bordercolor="rgba(0,0,0,0.1)", borderwidth=1, borderpad=3) for (xx,yy,txt) in pts]

initial_annotations = band_annotations_for(init["q_lines"], init["x_grid"])
fig.update_layout(annotations=initial_annotations)

# Mode buttons
vis_cmp_mask=[False]*len(traces); vis_cmp_mask[-2]=True; vis_cmp_mask[-1]=True
mode_menu=dict(
    buttons=[
        dict(label="Normal", method="update",
             args=[{"visible": vis_norm}, {"yaxis.title.text": "BTC / <DENOMINATOR>", "annotations": initial_annotations}]),
        dict(label="Compare (Rebased)", method="update",
             args=[{"visible": vis_cmp_mask}, {"yaxis.title.text": "Rebased to 1.0 (log scale)", "annotations": []}]),
    ],
    direction="down", showactive=True, x=0.0, y=1.18, xanchor="left", yanchor="top"
)
fig.update_layout(updatemenus=[mode_menu])

# ---------------------- HTML (panel + copy + date lock) ----------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

precomp_json   = json.dumps(PRECOMP)  # all lists now JSON-safe
band_colors_js = json.dumps({str(k): v for k,v in BAND_COLORS.items()})
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

  /* Hide Plotly tooltip; keep events alive for hover */
  .hoverlayer { display:none !important; }

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
      <div class="row"><div><span style="color:$COL_10;">q10</span></div><div class="num" id="q10">$0.00</div><div><em>(Bottom 10%)</em></div></div>
      <div class="row"><div><span style="color:$COL_30;">q30</span></div><div class="num" id="q30">$0.00</div><div><em>(10–30%)</em></div></div>
      <div class="row"><div><span style="color:$COL_50;">q50</span></div><div class="num" id="q50">$0.00</div><div><em>(30–50%)</em></div></div>
      <div class="row"><div><span style="color:$COL_70;">q70</span></div><div class="num" id="q70">$0.00</div><div><em>(50–70%)</em></div></div>
      <div class="row"><div><span style="color:$COL_90;">q90</span></div><div class="num" id="q90">$0.00</div><div><em>(70–90%)</em></div></div>
      <div style="margin-top:10px;"><b>BTC Price:</b> <span class="num" id="mainVal">$0.00</span></div>
      <div><b>Band:</b> <span id="bandLbl" style="font-weight:600;">$INIT_BAND</span>
           <span style="color:#6b7280;">(p≈$INIT_PCT%)</span></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>
<script>
const PRECOMP      = $PRECOMP_JSON;
const BAND_COLORS  = $BAND_COLORS_JS;
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
function percFromY(y, q10,q30,q50,q70,q90){
  const ly=Math.log10(y), l10=Math.log10(q10), l30=Math.log10(q30), l50=Math.log10(q50), l70=Math.log10(q70), l90=Math.log10(q90);
  if (ly < l10) return 5;
  if (ly < l30) return 10 + 20*( (ly-l10)/Math.max(1e-12, (l30-l10)) );
  if (ly < l50) return 30 + 20*( (ly-l30)/Math.max(1e-12, (l50-l30)) );
  if (ly < l70) return 50 + 20*( (ly-l50)/Math.max(1e-12, (l70-l50)) );
  if (ly < l90) return 70 + 20*( (ly-l70)/Math.max(1e-12, (l90-l70)) );
  return 95;
}
function bandNameAndColor(y,q10,q30,q50,q70,q90){
  if (y < q10) return ["Bottom 10%", BAND_COLORS["0.1"]];
  if (y < q30) return ["10–30%",    BAND_COLORS["0.3"]];
  if (y < q50) return ["30–50%",    BAND_COLORS["0.5"]];
  if (y < q70) return ["50–70%",    BAND_COLORS["0.7"]];
  if (y < q90) return ["70–90%",    BAND_COLORS["0.9"]];
  return ["Top 10%", BAND_COLORS["0.9"]];
}

// Panel elements
const elDate=document.querySelector('#readout .date');
const elQ10=document.getElementById('q10'), elQ30=document.getElementById('q30'),
      elQ50=document.getElementById('q50'), elQ70=document.getElementById('q70'), elQ90=document.getElementById('q90');
const elMain=document.getElementById('mainVal'), elBand=document.getElementById('bandLbl');

let locked=false, lockedX=null;

function updatePanel(den,xDays){
  const P=PRECOMP[den];
  elDate.textContent = dateFromDaysShort(xDays);
  const q10=P.q_lines["0.1"].length?interp(P.x_grid,P.q_lines["0.1"],xDays):NaN;
  const q30=P.q_lines["0.3"].length?interp(P.x_grid,P.q_lines["0.3"],xDays):NaN;
  const q50=P.q_lines["0.5"].length?interp(P.x_grid,P.q_lines["0.5"],xDays):NaN;
  const q70=P.q_lines["0.7"].length?interp(P.x_grid,P.q_lines["0.7"],xDays):NaN;
  const q90=P.q_lines["0.9"].length?interp(P.x_grid,P.q_lines["0.9"],xDays):NaN;
  elQ10.textContent=fmtUSD(q10); elQ30.textContent=fmtUSD(q30);
  elQ50.textContent=fmtUSD(q50); elQ70.textContent=fmtUSD(q70); elQ90.textContent=fmtUSD(q90);

  // nearest observed price
  const xa=P.x_main, ya=P.y_main; let idx=0,best=1e99;
  for(let i=0;i<xa.length;i++){ const d=Math.abs(xa[i]-xDays); if(d<best){best=d; idx=i;} }
  const y=ya[idx]; elMain.textContent = fmtUSD(y);

  const [name,color] = bandNameAndColor(y,q10,q30,q50,q70,q90);
  elBand.textContent = name; elBand.style.color = color;

  const p = percFromY(y,q10,q30,q50,q70,q90);
  const plotDiv=document.querySelector('.left .js-plotly-plot');
  Plotly.relayout(plotDiv, {"title.text": `BTC Purchase Indicator — ${name} (p≈${p.toFixed(1)}%)`});
}

document.addEventListener('DOMContentLoaded', function(){
  const plotDiv=document.querySelector('.left .js-plotly-plot');

  // Live hover ON
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

  // Denominator change: restyle + recompute annotations and panel
  denomSel.addEventListener('change', function(){
    const key=denomSel.value, P=PRECOMP[key];
    const qs=['0.1','0.3','0.5','0.7','0.9'];
    for(let i=0;i<qs.length;i++){
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[P.q_lines[qs[i]]] }, [i]);
    }
    function setBand(slot,keyBand){ const L=P.bands[keyBand]?.lower||[], U=P.bands[keyBand]?.upper||[];
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[L] }, [slot]);
      Plotly.restyle(plotDiv, { x:[P.x_grid], y:[U] }, [slot+1]);
    }
    setBand(5,"0.1-0.3"); setBand(7,"0.3-0.5"); setBand(9,"0.5-0.7"); setBand(11,"0.7-0.9");

    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.y_main], name:[P.label] }, [13]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.main_rebased] }, [14]);
    Plotly.restyle(plotDiv, { x:[P.x_main], y:[P.denom_rebased] }, [15]);

    function bandAnnotations(P){
      const xs=P.x_grid; const idx=Math.max(0, Math.floor(0.82*(xs.length-1)));
      const q10=P.q_lines["0.1"], q30=P.q_lines["0.3"], q50=P.q_lines["0.5"], q70=P.q_lines["0.7"], q90=P.q_lines["0.9"];
      if(!q10.length||!q30.length||!q50.length||!q70.length||!q90.length) return [];
      const gm=(a,b)=>Math.sqrt(a*b), x=xs[idx];
      const pts=[
        {x, y:q10[idx]*0.80, text:"Bottom 10%"},
        {x, y:gm(q10[idx],q30[idx]), text:"10–30%"},
        {x, y:gm(q30[idx],q50[idx]), text:"30–50%"},
        {x, y:gm(q50[idx],q70[idx]), text:"50–70%"},
        {x, y:gm(q70[idx],q90[idx]), text:"70–90%"},
        {x, y:q90[idx]*1.25, text:"Top 10%"},
      ];
      return pts.map(p => ({...p, xref:"x", yref:"y", showarrow:false,
                            font:{size:12,color:"#111"}, bgcolor:"rgba(255,255,255,0.6)",
                            bordercolor:"rgba(0,0,0,0.1)", borderwidth:1, borderpad:3}));
    }
    Plotly.relayout(plotDiv, {"yaxis.title.text": P.label, "annotations": bandAnnotations(P)});

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
    GENESIS_ISO=genesis_iso,
    COL_10=BAND_COLORS[0.1],
    COL_30=BAND_COLORS[0.3],
    COL_50=BAND_COLORS[0.5],
    COL_70=BAND_COLORS[0.7],
    COL_90=BAND_COLORS[0.9],
    INIT_BAND=init["band_label"],
    INIT_PCT=f"{init['percentile']:.1f}",
)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
