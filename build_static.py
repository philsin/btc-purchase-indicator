#!/usr/bin/env python3
"""
BTC Purchase Indicator — Plotly + right-side hover panel + copy-to-clipboard

What this script does
---------------------
- Loads BTC USD daily prices (auto-creates data/btc_usd.csv if missing).
  * Tries CoinGecko Pro if COINGECKO_API_KEY is set, otherwise falls back to Blockchain.com Charts.
- Optionally loads denominators from CSVs: data/denominator_*.csv (columns: date,price).
- Trims data to start at 2011-01-01. X axis is log(days since start).
- Fits log-log quantile regressions at q = 0.1,0.3,0.5,0.7,0.9 and
  projects them to 2040-12-31 (bands maintain trajectory).
- Builds a Plotly figure with legend pinned to the right.
- Renders a static right-side panel that shows a unified hover readout
  (bold date at top; color-coded values by band color).
- Adds a "Copy Snapshot" button that copies the chart + side panel to clipboard.
- Replaces the old Plotly denominator dropdown with an HTML <select>.
  Selecting a denominator updates traces & the panel (no page reload).

Files
-----
- Input:  data/btc_usd.csv  (auto-created if absent)
          data/denominator_*.csv (optional extras)
- Output: docs/index.html
"""

import os, io, glob, time, math, json
from typing import Optional, Dict, Tuple
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

START_DATE  = datetime(2011, 1, 1)              # chart begins here
END_PROJ    = datetime(2040, 12, 31)            # extend bands to here
QUANTILES   = (0.1, 0.3, 0.5, 0.7, 0.9)
Q_ORDER     = [0.1, 0.3, 0.5, 0.7, 0.9]

BAND_COLORS = {
    0.1: "#D32F2F",  # red-ish
    0.3: "#F57C00",  # orange
    0.5: "#FBC02D",  # yellow
    0.7: "#7CB342",  # yellow-green
    0.9: "#2E7D32",  # green
}

# order of filled band pairs (draw inner to outer): (low, high)
BAND_PAIRS = [(0.3, 0.5), (0.5, 0.7), (0.1, 0.3), (0.7, 0.9)]

# -------------------------- UTILITIES ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

def _retry(fn, tries=3, base_delay=1.0, factor=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(base_delay * (factor ** i))
    raise last

# ----------------------- DATA LOADERS ---------------------------

def _fetch_btc_from_coingecko() -> pd.DataFrame:
    api_key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY")
    if not api_key:
        raise RuntimeError("COINGECKO_API_KEY not set")
    start = int(datetime(2010, 7, 17, tzinfo=timezone.utc).timestamp())
    end   = int(datetime.now(timezone.utc).timestamp())
    url = ("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
           f"?vs_currency=usd&from={start}&to={end}")

    def _call():
        r = requests.get(url, headers={"x-cg-pro-api-key": api_key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = [(datetime.utcfromtimestamp(ms/1000.0).date().isoformat(), float(price))
                for ms, price in data.get("prices", [])]
        df = pd.DataFrame(rows, columns=["date", "price"]).dropna().sort_values("date")
        if df.empty:
            raise RuntimeError("CoinGecko returned empty dataset")
        return df

    return _retry(_call)

def _fetch_btc_from_blockchain() -> pd.DataFrame:
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv&sampled=false"

    def _call():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        raw = r.text.strip()
        if not raw:
            raise RuntimeError("Blockchain.com returned empty response")
        if raw.splitlines()[0].lower().startswith("timestamp"):
            df = pd.read_csv(io.StringIO(raw))
            ts_col = [c for c in df.columns if c.lower().startswith("timestamp")][0]
            val_col = [c for c in df.columns if c.lower().startswith("value")][0]
            df = df.rename(columns={ts_col: "date", val_col: "price"})
        else:
            df = pd.read_csv(io.StringIO(raw), header=None, names=["date", "price"])
        df["date"]  = pd.to_datetime(df["date"], utc=True).dt.date.astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna().sort_values("date")
        if df.empty:
            raise RuntimeError("Blockchain.com dataset is empty after parsing")
        return df

    return _retry(_call)

def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col  = cols.get("date")
    price_col = cols.get("price")
    if not date_col or not price_col:
        raise ValueError(f"{path} must have 'date' and 'price' columns")
    df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"})
    df["date"]  = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").dropna()
    df = df[df["price"] > 0]
    return df.reset_index(drop=True)

def get_btc_df() -> pd.DataFrame:
    """Try local CSV; if missing, fetch (CG Pro if key, else Blockchain.com), save, and load."""
    if os.path.exists(BTC_FILE):
        return load_series_csv(BTC_FILE)
    ensure_dir(DATA_DIR)
    df = None
    try:
        if os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY"):
            df = _fetch_btc_from_coingecko()
    except Exception as e:
        print(f"[warn] CoinGecko fetch failed: {e}")
    if df is None:
        try:
            df = _fetch_btc_from_blockchain()
        except Exception as e:
            raise RuntimeError(
                "Could not fetch BTC data from CoinGecko (needs API key) or Blockchain.com. "
                "Provide data/btc_usd.csv (date,price) or set COINGECKO_API_KEY."
            ) from e
    df.to_csv(BTC_FILE, index=False)
    return load_series_csv(BTC_FILE)

def collect_denominators() -> Dict[str, pd.DataFrame]:
    """Find files like data/denominator_*.csv (must have date,price)."""
    opts = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "denominator_*.csv"))):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_", "").upper()
        try:
            opts[key] = load_series_csv(p)
        except Exception as e:
            print(f"[warn] Bad denominator file {p}: {e}")
    return opts

# ---------------------- MATH / FITS -----------------------------

def days_since_start(dates: pd.Series, start: datetime) -> pd.Series:
    return (dates - start).dt.days.astype(float) + 1.0  # +1 to avoid log(0)

def make_log_time_ticks(start_date: datetime, x_min: float, x_max: float):
    ticks, ticktexts = [], []
    if x_min <= 0:
        x_min = 1e-6
    exp_min = math.floor(math.log10(x_min))
    exp_max = math.ceil(math.log10(x_max))
    for e in range(exp_min, exp_max + 1):
        for m in (1, 2, 5):
            v = m * (10 ** e)
            if x_min <= v <= x_max:
                d = start_date + timedelta(days=float(v))
                ticks.append(v)
                ticktexts.append(d.strftime("%m/%d/%y"))
    if 1.0 >= x_min and 1.0 <= x_max and 1.0 not in ticks:
        ticks.append(1.0)
        d = start_date + timedelta(days=1.0)
        ticktexts.append(d.strftime("%m/%d/%y"))
    idx = np.argsort(ticks)
    return [ticks[i] for i in idx], [ticktexts[i] for i in idx]

def fit_quantile_params(x: np.ndarray, y: np.ndarray, qs=QUANTILES) -> Dict[float, Tuple[float, float]]:
    """Return {q: (a, b)} for log10(y) = a + b*log10(x)."""
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_use, y_use = x[m], y[m]
    if len(x_use) < 10:
        return {}
    X = pd.DataFrame({"logx": np.log10(x_use)})
    z = np.log10(y_use)
    params = {}
    for q in qs:
        try:
            model = QuantReg(z, pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1))
            res = model.fit(q=q)
            a = float(res.params["const"])
            b = float(res.params["logx"])
            params[q] = (a, b)
        except Exception:
            pass
    return params

def predict_from_params(params: Dict[float, Tuple[float, float]], x_grid: np.ndarray) -> Dict[float, np.ndarray]:
    preds = {}
    lx = np.log10(x_grid)
    for q, (a, b) in params.items():
        zhat = a + b * lx
        preds[q] = 10 ** zhat
    return preds

def percentile_band(y_last: float, x_last: float, preds_at_x: Dict[float, float]) -> Tuple[str, float, str]:
    """Return (label, approx_percentile, color) for the latest point vs quantiles."""
    qv = preds_at_x
    need = all(q in qv and np.isfinite(qv[q]) for q in QUANTILES)
    if not need:
        return "N/A", 0.0, "#333333"
    if y_last < qv[0.1]:
        return "<10%", 0.05, BAND_COLORS[0.1]
    if y_last < qv[0.3]:
        t = (y_last - qv[0.1]) / max(1e-12, (qv[0.3]-qv[0.1])); return "10–30%", 0.10 + 0.20*t, BAND_COLORS[0.3]
    if y_last < qv[0.5]:
        t = (y_last - qv[0.3]) / max(1e-12, (qv[0.5]-qv[0.3])); return "30–50%", 0.30 + 0.20*t, BAND_COLORS[0.5]
    if y_last < qv[0.7]:
        t = (y_last - qv[0.5]) / max(1e-12, (qv[0.7]-qv[0.5])); return "50–70%", 0.50 + 0.20*t, BAND_COLORS[0.7]
    if y_last < qv[0.9]:
        t = (y_last - qv[0.7]) / max(1e-12, (qv[0.9]-qv[0.7])); return "70–90%", 0.70 + 0.20*t, BAND_COLORS[0.9]
    return ">90%", 0.95, BAND_COLORS[0.9]

def rebase_to_one(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.iloc[0] <= 0:
        return pd.Series(series) * np.nan
    return pd.Series(series) / s.iloc[0]

# ----------------------- LOAD DATA ------------------------------

btc = get_btc_df().rename(columns={"price": "btc"})
denoms = collect_denominators()  # {"SPX": df, "GOLD": df, ...}

# Merge denominators
base = btc.copy()
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

# Trim to START_DATE
base = base[base["date"] >= START_DATE].reset_index(drop=True)
if base.empty:
    raise RuntimeError("No BTC data on/after 2011-01-01.")

# X axis (days since START_DATE)
start_date = base["date"].iloc[0]  # should be >= START_DATE
base["x_days"]  = days_since_start(base["date"], start_date)
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# Prediction grid to END_PROJ
x_min        = float(max(1.0, base["x_days"].min()))
x_future_end = float(days_since_start(pd.Series([END_PROJ]), start_date).iloc[0])
x_grid       = np.logspace(math.log10(x_min), math.log10(x_future_end), 500)

# -------------------- PRECOMPUTE ALL DENOMS --------------------

def series_for_denom(df: pd.DataFrame, denom_key: Optional[str]):
    """Return (series, label). denom_key can be None/'USD' or one of the available denom columns."""
    if not denom_key or denom_key.lower() in ("usd", "none"):
        return df["btc"], "BTC (USD)", None  # no separate denom series
    k = denom_key.lower()
    if k in df.columns:
        return (df["btc"] / df[k]), f"BTC / {denom_key.upper()}", df[k]
    return df["btc"], "BTC (USD)", None

def pack_precomp(denom_key: Optional[str]):
    y_main, label, denom_series = series_for_denom(base, denom_key)
    x_vals = base["x_days"].values.astype(float)

    # Fit params on available data & predict on extended grid
    params  = fit_quantile_params(x_vals, y_main.values, QUANTILES)
    qpred   = predict_from_params(params, x_grid)

    # classify latest point
    valid_idx = np.where(np.isfinite(y_main.values))[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No valid data points")
    last_idx  = int(valid_idx[-1])
    x_last    = float(base["x_days"].iloc[last_idx])
    y_last    = float(y_main.iloc[last_idx])
    preds_at_xlast = {q: float(10 ** (params[q][0] + params[q][1] * math.log10(x_last))) for q in params}
    band_label, approx_p, line_color = percentile_band(y_last, x_last, preds_at_xlast)

    # Build payload
    payload = {
        "label": label,
        "x_main": base["x_days"].tolist(),
        "y_main": y_main.tolist(),
        "x_grid": x_grid.tolist(),
        "q_lines": {str(q): qpred[q].tolist() if q in qpred else [] for q in QUANTILES},
        "bands": {
            "0.3-0.5": {"upper": qpred[0.5].tolist() if 0.5 in qpred else [],
                        "lower": qpred[0.3].tolist() if 0.3 in qpred else []},
            "0.5-0.7": {"upper": qpred[0.7].tolist() if 0.7 in qpred else [],
                        "lower": qpred[0.5].tolist() if 0.5 in qpred else []},
            "0.1-0.3": {"upper": qpred[0.3].tolist() if 0.3 in qpred else [],
                        "lower": qpred[0.1].tolist() if 0.1 in qpred else []},
            "0.7-0.9": {"upper": qpred[0.9].tolist() if 0.9 in qpred else [],
                        "lower": qpred[0.7].tolist() if 0.7 in qpred else []},
        },
        "main_rebased": rebase_to_one(y_main).tolist(),
        "denom_rebased": rebase_to_one(denom_series).tolist() if denom_series is not None else [math.nan]*len(base),
        "band_label": band_label,
        "percentile": approx_p,
        "line_color": line_color,
    }
    return payload

PRECOMP = {"USD": pack_precomp(None)}
for k in sorted(denoms.keys()):
    PRECOMP[k] = pack_precomp(k)

# -------------------- BUILD INITIAL FIGURE ---------------------

# Use USD view for initial render
init = PRECOMP["USD"]

# traces index map:
# 0..4  : q-lines q10,q30,q50,q70,q90  (x = x_grid)
# 5..12 : band pairs upper/lower: (0.3-0.5), (0.5-0.7), (0.1-0.3), (0.7-0.9)
# 13    : main series (x = x_main)
# 14    : main rebased
# 15    : denom rebased

traces = []
visibility_normal = []
visibility_compare = []

# q-lines
for q in Q_ORDER:
    yq = init["q_lines"][str(q)]
    traces.append(go.Scatter(
        x=init["x_grid"], y=yq, mode="lines",
        name=f"q{int(q*100)}",
        line=dict(width=0.8, dash="dot", color=BAND_COLORS[q]),
        hoverinfo="skip",  # we'll use our static panel
        showlegend=False
    ))
    visibility_normal.append(True); visibility_compare.append(False)

# bands
def add_band(pair_key, ql, qh):
    upper = init["bands"][pair_key]["upper"]
    lower = init["bands"][pair_key]["lower"]
    traces.append(go.Scatter(
        x=init["x_grid"], y=upper, mode="lines",
        line=dict(width=0.5, color=BAND_COLORS[qh]),
        hoverinfo="skip", showlegend=False
    ))
    visibility_normal.append(True); visibility_compare.append(False)
    traces.append(go.Scatter(
        x=init["x_grid"], y=lower, mode="lines",
        line=dict(width=0.5, color=BAND_COLORS[ql]),
        hoverinfo="skip", showlegend=False,
        fill="tonexty", fillcolor=rgba(BAND_COLORS[qh], 0.18)
    ))
    visibility_normal.append(True); visibility_compare.append(False)

add_band("0.3-0.5", 0.3, 0.5)
add_band("0.5-0.7", 0.5, 0.7)
add_band("0.1-0.3", 0.1, 0.3)
add_band("0.7-0.9", 0.7, 0.9)

# main
traces.append(go.Scatter(
    x=init["x_main"], y=init["y_main"], mode="lines",
    name=init["label"], line=dict(width=1.5, color=init["line_color"]),
    hoverinfo="skip"
))
visibility_normal.append(True); visibility_compare.append(False)

# compare (rebased)
traces.append(go.Scatter(
    x=init["x_main"], y=init["main_rebased"], mode="lines",
    name="Main (rebased)", line=dict(width=1.5),
    hoverinfo="skip", visible=False
))
visibility_normal.append(False); visibility_compare.append(True)

traces.append(go.Scatter(
    x=init["x_main"], y=init["denom_rebased"], mode="lines",
    name="Denominator (rebased)", line=dict(width=1.5, dash="dash"),
    hoverinfo="skip", visible=False
))
visibility_normal.append(False); visibility_compare.append(True)

fig = go.Figure(data=traces)

# axis ticks & layout
xticks, xticktext = make_log_time_ticks(start_date, float(init["x_grid"][0]), float(init["x_grid"][-1]))

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode=False,  # we use our own right-side hover panel
    title=f"BTC Purchase Indicator — {init['band_label']} (p≈{init['percentile']:.2f})",
    xaxis=dict(
        type="log",
        title="Time (log scale)",
        tickvals=xticks, ticktext=xticktext,
        range=[math.log10(max(1e-6, xticks[0])), math.log10(max(1.0, xticks[-1]))]
    ),
    yaxis=dict(
        type="log",
        title=init["label"] + " (log scale)"
    ),
    legend=dict(
        x=1.02, xanchor="left", y=1.0, yanchor="top", orientation="v",
        bgcolor="rgba(255,255,255,0.0)"
    ),
    margin=dict(l=60, r=360, t=70, b=60),  # room on right for legend + panel
)

# Buttons: just the mode toggle (Normal / Compare). Denominator is handled via HTML select.
visible_compare = [False]*16; visible_compare[14] = True; visible_compare[15] = True

mode_menu = dict(
    buttons=[
        dict(label="Normal",
             method="update",
             args=[{"visible": visibility_normal},
                   {"yaxis.title.text": init["label"] + " (log scale)"}]),
        dict(label="Compare (Rebased)",
             method="update",
             args=[{"visible": visible_compare},
                   {"yaxis.title.text": "Rebased to 1.0 (log scale)"}]),
    ],
    direction="down", showactive=True, x=0.0, y=1.18, xanchor="left", yanchor="top"
)

fig.update_layout(updatemenus=[mode_menu])

# ----------------------- HTML SHELL -----------------------------

ensure_dir(os.path.dirname(OUTPUT_HTML))

# Build the figure HTML (no full page; we'll wrap it ourselves)
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]})

# JSON payloads for front-end updates
precomp_json  = json.dumps(PRECOMP)          # denom → data pack
band_colors_js = json.dumps({str(k): v for k,v in BAND_COLORS.items()})
xgrid_json    = json.dumps(init["x_grid"])
xmin_js       = float(init["x_grid"][0]); xmax_js = float(init["x_grid"][-1])
start_iso     = start_date.strftime("%Y-%m-%d")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<style>
  body {{ font-family: Inter, Roboto, -apple-system, Segoe UI, Arial, sans-serif; margin: 0; }}
  .wrap {{
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 0;
    height: 100vh;
  }}
  .left {{ padding: 10px 0 10px 10px; }}
  .right {{
    border-left: 1px solid #e5e7eb;
    padding: 12px 12px 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  #readout {{
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    background: #fafafa;
    font-size: 14px;
    line-height: 1.4;
  }}
  #readout .date {{ font-weight: 700; margin-bottom: 6px; }}
  #controls select, #controls button {{
    font-size: 14px; padding: 8px 10px; border-radius: 8px; border: 1px solid #d1d5db;
    background: white;
  }}
  #copyBtn {{ cursor: pointer; }}
  /* Hide default plotly hover popups */
  .hoverlayer, .g-gtitle:hover {{ display: none !important; }}
</style>
</head>
<body>
<div id="capture" class="wrap">
  <div class="left">
    {plot_html}
  </div>
  <div class="right">
    <div id="controls">
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <label for="denomSel"><b>Denominator:</b></label>
        <select id="denomSel"></select>
        <button id="copyBtn" title="Copy chart + panel to clipboard">Copy Snapshot</button>
      </div>
      <div style="font-size:12px;color:#6b7280;">
        Detected denominators: <span id="denomsDetected"></span>
      </div>
    </div>
    <div id="readout">
      <div class="date">—</div>
      <div id="rows">
        <div><span style="color:{BAND_COLORS[0.1]};">q10:</span> —</div>
        <div><span style="color:{BAND_COLORS[0.3]};">q30:</span> —</div>
        <div><span style="color:{BAND_COLORS[0.5]};">q50:</span> —</div>
        <div><span style="color:{BAND_COLORS[0.7]};">q70:</span> —</div>
        <div><span style="color:{BAND_COLORS[0.9]};">q90:</span> —</div>
        <div style="margin-top:6px;"><b>Main:</b> <span id="mainVal">—</span></div>
        <div><b>Band:</b> <span id="bandLbl">{init['band_label']}</span> <span style="color:#6b7280;">(p≈{init['percentile']:.2f})</span></div>
      </div>
    </div>
  </div>
</div>

<!-- html-to-image for clipboard snapshot -->
<script src="https://unpkg.com/html-to-image@1.11.11/dist/html-to-image.umd.js"></script>

<script>
// Data prepared by Python:
const PRECOMP = {precomp_json};
const BAND_COLORS = {band_colors_js};
const XGRID = {xgrid_json};
const START_DATE_ISO = "{start_iso}";

// Build denominator selector
const denomSel = document.getElementById('denomSel');
const detected = Object.keys(PRECOMP).filter(k => k !== 'USD');
document.getElementById('denomsDetected').textContent = detected.length ? detected.join(', ') : '(none)';
['USD', ...detected].forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = (k === 'USD') ? 'USD/None' : k;
  denomSel.appendChild(opt);
}});

// Helpers
function numFmt(v) {{
  if (!isFinite(v)) return '—';
  const mag = Math.log10(v);
  if (mag >= 6 || mag <= -3) return v.toExponential(2);
  if (v >= 1000) return v.toLocaleString(undefined, {{maximumFractionDigits: 2}});
  return v.toLocaleString(undefined, {{maximumFractionDigits: 6}});
}}
function dateFromXDays(x) {{
  const d0 = new Date(START_DATE_ISO + 'T00:00:00Z');
  const d  = new Date(d0.getTime() + (x-1)*24*3600*1000);
  const mm = String(d.getUTCMonth()+1).padStart(2,'0');
  const dd = String(d.getUTCDate()).padStart(2,'0');
  const yy = String(d.getUTCFullYear()).slice(-2);
  return mm + '/' + dd + '/' + yy;
}}
function interp(xArr, yArr, x) {{
  // assumes xArr is increasing
  let lo=0, hi=xArr.length-1;
  if (x <= xArr[0]) return yArr[0];
  if (x >= xArr[hi]) return yArr[hi];
  while (hi - lo > 1) {{
    const mid = (hi+lo)>>1;
    if (xArr[mid] <= x) lo = mid; else hi = mid;
  }}
  const t = (x - xArr[lo]) / (xArr[hi] - xArr[lo]);
  return yArr[lo] + t*(yArr[hi]-yArr[lo]);
}}

// Right-side unified readout
const readout = {{
  dateEl:  document.querySelector('#readout .date'),
  q10:     document.querySelectorAll('#readout #rows div')[0],
  q30:     document.querySelectorAll('#readout #rows div')[1],
  q50:     document.querySelectorAll('#readout #rows div')[2],
  q70:     document.querySelectorAll('#readout #rows div')[3],
  q90:     document.querySelectorAll('#readout #rows div')[4],
  mainVal: document.getElementById('mainVal'),
  bandLbl: document.getElementById('bandLbl')
}};
function updatePanel(denKey, x) {{
  const P = PRECOMP[denKey];
  readout.dateEl.textContent = dateFromXDays(x);
  // quantiles on grid
  const q10 = P.q_lines["0.1"].length ? interp(P.x_grid, P.q_lines["0.1"], x) : NaN;
  const q30 = P.q_lines["0.3"].length ? interp(P.x_grid, P.q_lines["0.3"], x) : NaN;
  const q50 = P.q_lines["0.5"].length ? interp(P.x_grid, P.q_lines["0.5"], x) : NaN;
  const q70 = P.q_lines["0.7"].length ? interp(P.x_grid, P.q_lines["0.7"], x) : NaN;
  const q90 = P.q_lines["0.9"].length ? interp(P.x_grid, P.q_lines["0.9"], x) : NaN;
  readout.q10.lastChild.textContent = ' ' + numFmt(q10);
  readout.q30.lastChild.textContent = ' ' + numFmt(q30);
  readout.q50.lastChild.textContent = ' ' + numFmt(q50);
  readout.q70.lastChild.textContent = ' ' + numFmt(q70);
  readout.q90.lastChild.textContent = ' ' + numFmt(q90);
  // main value ~ nearest in x_main
  const xa = P.x_main, ya = P.y_main;
  let idx = 0; let bestD = Infinity;
  for (let i=0;i<xa.length;i++) {{ const d = Math.abs(xa[i]-x); if (d<bestD){{bestD=d; idx=i;}} }}
  readout.mainVal.textContent = numFmt(ya[idx]);
  readout.bandLbl.textContent = P.band_label;
}}

// Copy snapshot (chart + side panel)
document.getElementById('copyBtn').addEventListener('click', async () => {{
  const node = document.getElementById('capture');
  try {{
    const canvas = await htmlToImage.toCanvas(node, {{ pixelRatio: 2 }});
    if (navigator.clipboard && window.ClipboardItem) {{
      const blob = await new Promise(res => canvas.toBlob(res));
      await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
      document.getElementById('copyBtn').textContent = 'Copied!';
      setTimeout(()=>document.getElementById('copyBtn').textContent='Copy Snapshot', 1400);
    }} else {{
      // fallback: open in new tab
      const url = canvas.toDataURL('image/png');
      const win = window.open(); win.document.write('<img src="'+url+'"/>');
    }}
  }} catch(err) {{
    console.error(err);
    alert('Copy failed. Browser may block clipboard images.');
  }}
}});

// Populate panel initially at the last x of the main series
updatePanel('USD', PRECOMP['USD'].x_main[PRECOMP['USD'].x_main.length-1]);

// Hook Plotly hover to update the panel
document.addEventListener('DOMContentLoaded', () => {{
  const plotDiv = document.querySelector('.left .js-plotly-plot');
  plotDiv.on('plotly_hover', (ev) => {{
    if (!ev.points || !ev.points.length) return;
    // take x from the first point
    const x = ev.points[0].x;
    updatePanel(denomSel.value, x);
  }});
}});

// Denominator change → restyle all traces + labels
denomSel.addEventListener('change', () => {{
  const key = denomSel.value;
  const P = PRECOMP[key];

  const updates = {{}};
  const indices = [...Array(16).keys()];

  // 0..4 q-lines
  const qs = ['0.1','0.3','0.5','0.7','0.9'];
  for (let i=0;i<qs.length;i++) {{
    updates[i] = {{
      x: [P.x_grid],
      y: [P.q_lines[qs[i]]],
      name: ['q' + String(parseFloat(qs[i])*100).split('.')[0]]
    }};
  }}

  // 5..12 band pairs (upper, lower) using x_grid
  function setBand(slot, pairKey) {{
    const U = P.bands[pairKey]?.upper || [];
    const L = P.bands[pairKey]?.lower || [];
    updates[slot]   = {{ x: [P.x_grid], y: [U] }};
    updates[slot+1] = {{ x: [P.x_grid], y: [L] }};
  }}
  setBand(5, "0.3-0.5");
  setBand(7, "0.5-0.7");
  setBand(9, "0.1-0.3");
  setBand(11,"0.7-0.9");

  // 13 main
  updates[13] = {{ x: [P.x_main], y: [P.y_main], name: [P.label], 'line.color': [P.line_color] }};

  // 14 main rebased
  updates[14] = {{ x: [P.x_main], y: [P.main_rebased], name: ['Main (rebased)'] }};

  // 15 denom rebased
  updates[15] = {{ x: [P.x_main], y: [P.denom_rebased], name: ['Denominator (rebased)'] }};

  const plotDiv = document.querySelector('.left .js-plotly-plot');
  // apply per-trace restyle
  for (const idx in updates) {{
    Plotly.restyle(plotDiv, updates[idx], [parseInt(idx)]);
  }}
  // update layout titles
  Plotly.relayout(plotDiv, {{
    'yaxis.title.text': P.label + ' (log scale)',
    'title.text': 'BTC Purchase Indicator — ' + P.band_label + ' (p≈' + P.percentile.toFixed(2) + ')'
  }});

  // refresh side panel to last x of new main
  updatePanel(key, P.x_main[P.x_main.length-1]);
}});
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML}")
