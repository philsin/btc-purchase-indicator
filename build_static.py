#!/usr/bin/env python3
"""
BTC Purchase Indicator builder (Plotly HTML)

Features
--------
- Data: auto-creates data/btc_usd.csv if missing
    * Tries CoinGecko Pro (needs COINGECKO_API_KEY env)
    * Falls back to Blockchain.com Charts (no key)
    * Retries with backoff on network hiccups
- Axes: log time (numeric days-since-start with date tick labels) + log y
- Quantile regressions (log-log) at 10/30/50/70/90 + filled bands
- Hover order: q10, q30, q50, q70, q90 (ascending)
- Title & main-line color reflect the current band position
- Compare Mode: hides bands, shows rebased main + rebased denominator
- Denominator menu: auto-detects data/denominator_*.csv (date,price)
- Short date (MM/DD/YY) on ticks and hover

Requirements
-----------
pandas
numpy
plotly
statsmodels
requests
"""

import os
import io
import glob
import time
import math
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

# ---------------- Configuration ----------------

OUTPUT_HTML = "docs/index.html"
DATA_DIR = "data"
BTC_FILE = os.path.join(DATA_DIR, "btc_usd.csv")
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)

BAND_COLORS = {  # base colors for lines & band fills
    0.1: "#D32F2F",  # red-ish
    0.3: "#F57C00",  # orange
    0.5: "#FBC02D",  # yellow
    0.7: "#7CB342",  # yellow-green
    0.9: "#2E7D32",  # green
}

# ---------------- Utilities ----------------

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

# ---------------- Data fetchers ----------------

def _fetch_btc_from_coingecko() -> pd.DataFrame:
    """CoinGecko Pro (requires COINGECKO_API_KEY or X_CG_PRO_API_KEY)."""
    api_key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY")
    if not api_key:
        raise RuntimeError("COINGECKO_API_KEY not set")
    start = int(datetime(2010, 7, 17, tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp())
    url = ("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
           f"?vs_currency=usd&from={start}&to={end}")

    def _call():
        r = requests.get(url, headers={"x-cg-pro-api-key": api_key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = [(datetime.utcfromtimestamp(ms / 1000.0).date().isoformat(), float(price))
                for ms, price in data.get("prices", [])]
        df = pd.DataFrame(rows, columns=["date", "price"]).dropna().sort_values("date")
        if df.empty:
            raise RuntimeError("CoinGecko returned empty dataset")
        return df

    return _retry(_call)

def _fetch_btc_from_blockchain() -> pd.DataFrame:
    """Blockchain.com Charts API (public)."""
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
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.date.astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna().sort_values("date")
        if df.empty:
            raise RuntimeError("Blockchain.com dataset is empty after parsing")
        return df

    return _retry(_call)

def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    price_col = cols.get("price")
    if not date_col or not price_col:
        raise ValueError(f"{path} must have 'date' and 'price' columns")
    df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").dropna()
    df = df[df["price"] > 0]
    return df.reset_index(drop=True)

def get_btc_df() -> pd.DataFrame:
    """
    Try local CSV; if missing, fetch (CG Pro if key, else Blockchain.com),
    write to BTC_FILE, and return the DataFrame.
    """
    if os.path.exists(BTC_FILE):
        return load_series_csv(BTC_FILE)

    ensure_dir(DATA_DIR)
    df: Optional[pd.DataFrame] = None
    # Try CoinGecko Pro
    try:
        if os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_PRO_API_KEY"):
            df = _fetch_btc_from_coingecko()
    except Exception as e:
        print(f"[warn] CoinGecko fetch failed: {e}")
    # Fallback to Blockchain.com
    if df is None:
        try:
            df = _fetch_btc_from_blockchain()
        except Exception as e:
            raise RuntimeError(
                "Could not fetch BTC data from CoinGecko (needs API key) or Blockchain.com. "
                "Provide a data/btc_usd.csv (date,price) or set COINGECKO_API_KEY."
            ) from e

    df.to_csv(BTC_FILE, index=False)
    return load_series_csv(BTC_FILE)

def collect_denominators() -> Dict[str, pd.DataFrame]:
    """Find files like data/denominator_*.csv (must have date,price)."""
    paths = glob.glob(os.path.join(DATA_DIR, "denominator_*.csv"))
    opts = {}
    for p in sorted(paths):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_", "").upper()
        try:
            opts[key] = load_series_csv(p)
        except Exception:
            pass
    return opts

# ---------------- Transforms & modeling ----------------

def days_since_start(dates: pd.Series, start: datetime) -> pd.Series:
    # +1 to avoid log(0)
    return (dates - start).dt.days.astype(float) + 1.0

def make_log_time_ticks(start_date: datetime, x_min: float, x_max: float):
    """Powers-of-ten ticks with 1-2-5 mantissas, labeled as dates (MM/DD/YY)."""
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
    # ensure day 1 tick is included if in range
    if 1.0 >= x_min and 1.0 <= x_max and 1.0 not in ticks:
        ticks.append(1.0)
        d = start_date + timedelta(days=1.0)
        ticktexts.append(d.strftime("%m/%d/%y"))
    idx = np.argsort(ticks)
    return [ticks[i] for i in idx], [ticktexts[i] for i in idx]

def fit_quantiles(x: np.ndarray, y: np.ndarray,
                  quantiles=QUANTILES) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Fit log-log quantile regressions:
        log10(y) ~ a + b * log10(x)
    Return dict q -> (x_sorted, yhat).
    """
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_use, y_use = x[m], y[m]
    if len(x_use) < 10:
        return {}
    X = pd.DataFrame({"logx": np.log10(x_use)})
    z = np.log10(y_use)
    preds = {}
    for q in quantiles:
        try:
            model = QuantReg(z, pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1))
            res = model.fit(q=q)
            x_sorted = np.sort(np.unique(x_use))
            Xp = pd.DataFrame({"const": 1.0, "logx": np.log10(x_sorted)})
            zhat = res.predict(Xp)
            yhat = (10 ** zhat)
            preds[q] = (x_sorted, yhat)
        except Exception:
            pass
    return preds

def series_for_denominator(df: pd.DataFrame, denom_key: Optional[str]):
    """Return (series, label). denom_key can be None/'USD'/'NONE'/or a key in df columns."""
    if not denom_key or denom_key.lower() in ("usd", "none"):
        return df["btc"], "BTC (USD)"
    k = denom_key.lower()
    if k in df.columns:
        return (df["btc"] / df[k]), f"BTC / {denom_key.upper()}"
    return df["btc"], "BTC (USD)"

def rebase_to_one(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.iloc[0] <= 0:
        return pd.Series(series) * np.nan
    return pd.Series(series) / s.iloc[0]

def interpolate_at(qpred, q: float, xval: float) -> float:
    if q not in qpred:
        return np.nan
    xq, yq = qpred[q]
    if len(xq) == 0:
        return np.nan
    if xval < xq[0] or xval > xq[-1]:
        return np.nan
    return float(np.interp(xval, xq, yq))

def current_band_and_percent(y_last: float, x_last: float, qpred) -> Tuple[str, float, str]:
    """
    Return (band_label, approx_percentile, color) given the latest y and x, and a qpred dict.
    """
    qv = {q: interpolate_at(qpred, q, x_last) for q in QUANTILES}
    if any(np.isnan(list(qv.values()))):
        return "N/A", 0.0, "#333333"

    # piecewise percentile within bands
    if y_last < qv[0.1]:
        return "<10%", 0.05, BAND_COLORS[0.1]
    elif y_last < qv[0.3]:
        t = (y_last - qv[0.1]) / max(1e-12, (qv[0.3] - qv[0.1]))
        return "10–30%", 0.10 + 0.20 * t, BAND_COLORS[0.3]
    elif y_last < qv[0.5]:
        t = (y_last - qv[0.3]) / max(1e-12, (qv[0.5] - qv[0.3]))
        return "30–50%", 0.30 + 0.20 * t, BAND_COLORS[0.5]
    elif y_last < qv[0.7]:
        t = (y_last - qv[0.5]) / max(1e-12, (qv[0.7] - qv[0.5]))
        return "50–70%", 0.50 + 0.20 * t, BAND_COLORS[0.7]
    elif y_last < qv[0.9]:
        t = (y_last - qv[0.7]) / max(1e-12, (qv[0.9] - qv[0.7]))
        return "70–90%", 0.70 + 0.20 * t, BAND_COLORS[0.9]
    else:
        return ">90%", 0.95, BAND_COLORS[0.9]

# ---------------- Load data ----------------

# BTC (auto-fetch if needed)
btc = get_btc_df().rename(columns={"price": "btc"})

# Denominators (optional)
denoms = collect_denominators()

# Merge denominators onto base df
base = btc.copy()
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

# X axis as "days since start"
start_date = base["date"].iloc[0]
base["x_days"] = days_since_start(base["date"], start_date)
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# Initial series (USD)
y_main, y_label = series_for_denominator(base, None)
x_vals = base["x_days"].values.astype(float)

# Quantile fits
qpred = fit_quantiles(x_vals, y_main.values, QUANTILES)

# Determine latest band/color
valid_idx = np.where(np.isfinite(y_main.values))[0]
if len(valid_idx) == 0:
    raise RuntimeError("No valid BTC data points found.")
last_idx = int(valid_idx[-1])
x_last = float(base["x_days"].iloc[last_idx])
y_last = float(y_main.iloc[last_idx])
band_label, approx_p, band_color = current_band_and_percent(y_last, x_last, qpred)

# ---------------- Build traces with stable indices ----------------
# We will create a fixed set of 16 traces in a known order.

traces = []
visibility_normal = []
visibility_compare = []

# 0..4: q-lines q10, q30, q50, q70, q90 (always present, NaN if missing)
q_order = [0.1, 0.3, 0.5, 0.7, 0.9]
for q in q_order:
    if q in qpred:
        xq, yq = qpred[q]
        x_arr = xq
        y_arr = yq
        custom = [(start_date + timedelta(days=float(d))).strftime("%m/%d/%y") for d in x_arr]
    else:
        x_arr = base["x_days"].values
        y_arr = np.full_like(x_arr, np.nan, dtype=float)
        custom = base["date_str"].tolist()

    traces.append(go.Scatter(
        x=x_arr, y=y_arr, mode="lines",
        name=f"q{int(q*100)}",
        line=dict(width=0.8, dash="dot", color=BAND_COLORS[q]),
        hovertemplate="Date: %{customdata}<br>%{fullData.name}: %{y:.6g}<extra></extra>",
        customdata=custom,
        showlegend=False
    ))
    visibility_normal.append(True)   # visible in Normal
    visibility_compare.append(False) # hidden in Compare

# 5..12: band pairs in order: (0.3–0.5), (0.5–0.7), (0.1–0.3), (0.7–0.9)
band_pairs = [(0.3, 0.5), (0.5, 0.7), (0.1, 0.3), (0.7, 0.9)]
for (ql, qh) in band_pairs:
    if ql in qpred and qh in qpred:
        xl, yl = qpred[ql]
        xh, yh = qpred[qh]
        x_common = np.intersect1d(xl, xh)
        if len(x_common):
            yl_i = np.interp(x_common, xl, yl)
            yh_i = np.interp(x_common, xh, yh)
            # upper boundary
            traces.append(go.Scatter(
                x=x_common, y=yh_i, mode="lines",
                line=dict(width=0.5, color=BAND_COLORS[qh]),
                hoverinfo="skip", showlegend=False
            ))
            visibility_normal.append(True); visibility_compare.append(False)
            # lower boundary (fill down to this)
            traces.append(go.Scatter(
                x=x_common, y=yl_i, mode="lines",
                line=dict(width=0.5, color=BAND_COLORS[ql]),
                hoverinfo="skip", showlegend=False,
                fill="tonexty", fillcolor=rgba(BAND_COLORS[qh], 0.18)
            ))
            visibility_normal.append(True); visibility_compare.append(False)
        else:
            # placeholders (NaN)
            nan_arr = np.full_like(base["x_days"].values, np.nan, dtype=float)
            traces.append(go.Scatter(x=base["x_days"], y=nan_arr, mode="lines",
                                     line=dict(width=0.5), hoverinfo="skip", showlegend=False))
            traces.append(go.Scatter(x=base["x_days"], y=nan_arr, mode="lines",
                                     line=dict(width=0.5), hoverinfo="skip", showlegend=False,
                                     fill="tonexty"))
            visibility_normal += [True, True]; visibility_compare += [False, False]
    else:
        nan_arr = np.full_like(base["x_days"].values, np.nan, dtype=float)
        traces.append(go.Scatter(x=base["x_days"], y=nan_arr, mode="lines",
                                 line=dict(width=0.5), hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter(x=base["x_days"], y=nan_arr, mode="lines",
                                 line=dict(width=0.5), hoverinfo="skip", showlegend=False,
                                 fill="tonexty"))
        visibility_normal += [True, True]; visibility_compare += [False, False]

# 13: main BTC (colored by current band)
traces.append(go.Scatter(
    x=base["x_days"], y=y_main, mode="lines",
    name=y_label,
    line=dict(width=1.5, color=band_color),
    hovertemplate="Date: %{customdata}<br>Value: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"]
))
visibility_normal.append(True)
visibility_compare.append(False)  # hide original main in Compare mode

# 14: Compare main (rebased)
traces.append(go.Scatter(
    x=base["x_days"], y=rebase_to_one(y_main),
    name="Main (rebased)", mode="lines",
    line=dict(width=1.5),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"],
    visible=False
))
visibility_normal.append(False)
visibility_compare.append(True)

# 15: Compare denominator (rebased) – starts empty
traces.append(go.Scatter(
    x=base["x_days"], y=[np.nan]*len(base),
    name="Denominator (rebased)", mode="lines",
    line=dict(width=1.5, dash="dash"),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"],
    visible=False
))
visibility_normal.append(False)
visibility_compare.append(True)

# ---------------- Figure & layout ----------------

fig = go.Figure(data=traces)

x_min = float(base["x_days"].min())
x_max = float(base["x_days"].max())
xticks, xticktext = make_log_time_ticks(start_date, x_min, x_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x unified",
    title=f"BTC Purchase Indicator — {band_label} (p≈{approx_p:.2f})",
    xaxis=dict(
        type="log",
        title="Time (log scale)",
        tickvals=xticks,
        ticktext=xticktext,
        range=[math.log10(max(x_min, 1e-6)), math.log10(max(x_max, 1.0))]
    ),
    yaxis=dict(
        type="log",
        title=y_label + " (log scale)"
    ),
    margin=dict(l=60, r=20, t=70, b=50),
)

# ---------------- Menus ----------------

# Hover + legend menu (use explicit show/hide to avoid "toggle" ambiguity)
hover_menu = dict(
    buttons=[
        dict(label="Hover: Unified",   method="relayout", args=[{"hovermode": "x unified"}]),
        dict(label="Hover: Per Trace", method="relayout", args=[{"hovermode": "x"}]),
        dict(label="Show Legend",      method="relayout", args=[{"showlegend": True}]),
        dict(label="Hide Legend",      method="relayout", args=[{"showlegend": False}]),
    ],
    direction="down", showactive=False,
    x=0.00, y=1.14, xanchor="left", yanchor="top",
)

# Mode menu: Normal vs Compare (bands hidden in Compare; only rebased lines)
visible_normal = visibility_normal
visible_compare = [False]*16
visible_compare[14] = True  # main rebased
visible_compare[15] = True  # denom rebased

mode_menu = dict(
    buttons=[
        dict(
            label="Normal",
            method="update",
            args=[{"visible": visible_normal},
                  {"yaxis.title.text": y_label + " (log scale)", "hovermode": "x unified"}],
        ),
        dict(
            label="Compare (Rebased)",
            method="update",
            args=[{"visible": visible_compare},
                  {"yaxis.title.text": "Rebased to 1.0 (log scale)", "hovermode": "x unified"}],
        ),
    ],
    direction="down", showactive=True,
    x=0.22, y=1.14, xanchor="left", yanchor="top",
)

# Denominator menu
denom_labels = ["USD/None"]
denom_values = ["USD"]
for k in sorted(denoms.keys()):
    denom_labels.append(k)
    denom_values.append(k)

def make_denom_update(denom_key: Optional[str]):
    # Recompute series & quantiles
    y_d, label_d = series_for_denominator(base, denom_key)
    qpred_d = fit_quantiles(base["x_days"].values.astype(float), y_d.values, QUANTILES)

    # Re-eval band & color at the latest point
    y_last_d = float(y_d.iloc[last_idx])
    band_txt, p_est, band_col = current_band_and_percent(y_last_d, x_last, qpred_d)

    updates = []

    # 0..4 q-lines (q10, q30, q50, q70, q90)
    for q in q_order:
        if q in qpred_d:
            xq, yq = qpred_d[q]
            updates.append({"x": xq, "y": yq, "name": f"q{int(q*100)}"})
        else:
            updates.append({"y": [np.nan]*len(base)})

    # 5..12 band pairs
    def add_band_updates(ql, qh):
        if ql in qpred_d and qh in qpred_d:
            xl, yl = qpred_d[ql]
            xh, yh = qpred_d[qh]
            x_common = np.intersect1d(xl, xh)
            if len(x_common):
                yl_i = np.interp(x_common, xl, yl)
                yh_i = np.interp(x_common, xh, yh)
                updates.append({"x": x_common, "y": yh_i})  # upper
                updates.append({"x": x_common, "y": yl_i})  # lower
                return
        updates.append({"y": [np.nan]*len(base)})
        updates.append({"y": [np.nan]*len(base)})

    add_band_updates(0.3, 0.5)  # 5,6
    add_band_updates(0.5, 0.7)  # 7,8
    add_band_updates(0.1, 0.3)  # 9,10
    add_band_updates(0.7, 0.9)  # 11,12

    # 13 main line
    updates.append({"y": y_d, "name": label_d, "line.color": band_col})

    # 14 main rebased
    updates.append({"y": rebase_to_one(y_d), "name": "Main (rebased)"})

    # 15 denom rebased
    if denom_key and denom_key.upper() in denoms:
        dfden = denoms[denom_key.upper()]
        dfm = base.merge(dfden.rename(columns={"price": "den"}), on="date", how="left")
        updates.append({"y": rebase_to_one(dfm["den"]), "name": f"{denom_key.upper()} (rebased)"})
    else:
        updates.append({"y": [np.nan]*len(base), "name": "Denominator (rebased)"})

    layout_updates = {
        "yaxis.title.text": label_d + " (log scale)",
        "title.text": f"BTC Purchase Indicator — {band_txt} (p≈{p_est:.2f})"
    }
    return updates, layout_updates

denom_buttons = []
for lab, val in zip(denom_labels, denom_values):
    tr_updates, ly_updates = make_denom_update(val if val != "USD" else None)
    # Make sure we pass a list of dicts matching the number of traces (16)
    if len(tr_updates) != 16:
        # pad with empty dicts if needed (defensive)
        tr_updates = tr_updates + [{}] * (16 - len(tr_updates))
    denom_buttons.append(dict(
        label=lab,
        method="update",
        args=[{"transforms": []} , ly_updates],  # dummy (we'll use 'restyle' below)
    ))
# Plotly quirk: to restyle multiple traces at once with per-trace dicts,
# we use method='restyle' with a list of updates & trace indices.
# We'll attach a second updatemenu that actually restyles (hooked to the same labels).

# Build a parallel set of 'restyle' buttons using the prepared updates.
denom_buttons_restyle = []
for lab, val in zip(denom_labels, denom_values):
    tr_updates, _ = make_denom_update(val if val != "USD" else None)
    if len(tr_updates) != 16:
        tr_updates = tr_updates + [{}] * (16 - len(tr_updates))
    # Apply each dict to its trace index
    args0 = []
    idxs = list(range(16))
    # Plotly expects a single dict of property: [values per trace], but it also
    # accepts a list of dicts with indices. We'll push one-by-one for reliability.
    # To minimize UI clutter, we do one compound call with "args2": indices.
    denom_buttons_restyle.append(dict(
        label=lab,
        method="restyle",
        args=[tr_updates, idxs]
    ))

denom_menu_layout = dict(
    buttons=denom_buttons,
    direction="down", showactive=True,
    x=0.48, y=1.14, xanchor="left", yanchor="top"
)
denom_menu_restyle = dict(
    buttons=denom_buttons_restyle,
    direction="down", showactive=False,  # hidden action layer
    x=0.48, y=1.08, xanchor="left", yanchor="top"
)

fig.update_layout(updatemenus=[hover_menu, mode_menu, denom_menu_layout, denom_menu_restyle])

# ---------------- Write output ----------------

ensure_dir(os.path.dirname(OUTPUT_HTML))
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
print(f"Wrote {OUTPUT_HTML}")
