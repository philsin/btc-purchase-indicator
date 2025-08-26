#!/usr/bin/env python3
"""
BTC Purchase Indicator builder (Plotly HTML)

- Auto-fetches BTC daily history to data/btc_usd.csv if missing (CoinGecko).
- Log-scaled time (numeric "days since start") with human-readable MM/DD/YY ticks.
- Log-scaled y axis.
- Quantile regressions (log-log) at 10/30/50/70/90 with filled bands.
- Hover panel shows quantile lines in ascending order: q10, q30, q50, q70, q90.
- Title annotation & main-line color reflect the CURRENT band position.
- Compare Mode (rebased to 1.0) hides bands.
- Denominator menu auto-detects data/denominator_*.csv (date,price).

Requirements: pandas, numpy, plotly, statsmodels, requests
"""

import os, glob, math, json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from statsmodels.regression.quantile_regression import QuantReg

OUTPUT_HTML = "docs/index.html"
DATA_DIR = "data"
BTC_FILE = os.path.join(DATA_DIR, "btc_usd.csv")

# ------------------------------ Utilities

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_btc_if_missing():
    """Create data/btc_usd.csv using CoinGecko daily prices if file missing."""
    if os.path.exists(BTC_FILE):
        return
    ensure_dir(DATA_DIR)
    # CoinGecko market chart range: from 2010-07-17 to now, daily resolution
    vs = "usd"
    # 2010-07-17 epoch (approx BTC start)
    start = int(datetime(2010, 7, 17, tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency={vs}&from={start}&to={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # data["prices"] = [[ms, price], ...]
    rows = []
    for ms, price in data.get("prices", []):
        d = datetime.utcfromtimestamp(ms/1000.0).date()
        rows.append((d.isoformat(), float(price)))
    df = pd.DataFrame(rows, columns=["date","price"]).dropna()
    df = df.sort_values("date")
    df.to_csv(BTC_FILE, index=False)

def load_series_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    price_col = cols.get("price")
    if not date_col or not price_col:
        raise ValueError(f"{path} must have 'date' and 'price' columns")
    df = df[[date_col, price_col]].rename(columns={date_col:"date", price_col:"price"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").dropna()
    df = df[df["price"] > 0]
    return df.reset_index(drop=True)

def days_since_start(dates, start):
    return (dates - start).dt.days.astype(float) + 1.0  # +1 to avoid log(0)

def make_log_time_ticks(start_date, x_min, x_max):
    ticks, ticktexts = [], []
    if x_min <= 0: x_min = 1e-6
    exp_min = math.floor(math.log10(x_min))
    exp_max = math.ceil(math.log10(x_max))
    for e in range(exp_min, exp_max+1):
        for m in (1,2,5):
            v = m*(10**e)
            if x_min <= v <= x_max:
                d = start_date + timedelta(days=float(v))
                ticks.append(v)
                ticktexts.append(d.strftime("%m/%d/%y"))
    if 1.0 >= x_min and 1.0 <= x_max and 1.0 not in ticks:
        ticks.append(1.0)
        d = start_date + timedelta(days=1.0)
        ticktexts.append(d.strftime("%m/%d/%y"))
    # De-dup, keep order by sorting
    idx = np.argsort(ticks)
    return [ticks[i] for i in idx], [ticktexts[i] for i in idx]

def fit_quantiles(x, y, quantiles=(0.1,0.3,0.5,0.7,0.9)):
    m = (x>0) & (y>0)
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
            Xp = pd.DataFrame({"const":1.0, "logx":np.log10(x_sorted)})
            zhat = res.predict(Xp)
            yhat = (10**zhat)
            preds[q] = (x_sorted, yhat)
        except Exception:
            pass
    return preds

def rebase_to_one(series):
    s = pd.Series(series).astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty or s.iloc[0] <= 0:
        return pd.Series(series)*np.nan
    return pd.Series(series)/s.iloc[0]

def collect_denominators():
    paths = glob.glob(os.path.join(DATA_DIR, "denominator_*.csv"))
    opts = {}
    for p in sorted(paths):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_","").upper()
        try:
            opts[key] = load_series_csv(p)
        except Exception:
            pass
    return opts

def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

# ------------------------------ Load / Prepare

fetch_btc_if_missing()  # <-- new: auto-create BTC CSV if missing

btc = load_series_csv(BTC_FILE).rename(columns={"price":"btc"})
denoms = collect_denominators()

base = btc.copy()
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

start_date = base["date"].iloc[0]
base["x_days"] = days_since_start(base["date"], start_date)
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

def series_for_denom(df, denom_key=None):
    if not denom_key or denom_key.lower() in ("usd","none"):
        return df["btc"], "BTC (USD)"
    k = denom_key.lower()
    if k in df.columns:
        return (df["btc"]/df[k]), f"BTC / {denom_key.upper()}"
    return df["btc"], "BTC (USD)"

y_main, y_label = series_for_denom(base, None)
quantiles = (0.1,0.3,0.5,0.7,0.9)
qpred = fit_quantiles(base["x_days"].values, y_main.values, quantiles)

# ------------------------------ Current band status

colors = {
    0.1:"#D32F2F", 0.3:"#F57C00", 0.5:"#FBC02D", 0.7:"#7CB342", 0.9:"#2E7D32"
}

def interpolate_quantile_at_x(qpred, q, xval):
    if q not in qpred: return np.nan
    xq, yq = qpred[q]
    if xval < xq[0] or xval > xq[-1]:
        return np.nan
    return float(np.interp(xval, xq, yq))

# Use the last valid point in y_main/x_days
last_idx = int(np.where(np.isfinite(y_main.values))[0][-1])
x_last = float(base["x_days"].iloc[last_idx])
y_last = float(y_main.iloc[last_idx])

q_vals = {q: interpolate_quantile_at_x(qpred, q, x_last) for q in quantiles}
# Determine band + approx percentile (linear between nearest quantiles)
band_label = "below 10%"
band_color = colors[0.1]
approx_p = 0.0

if not any(np.isnan(list(q_vals.values()))):
    if y_last < q_vals[0.1]:
        band_label, approx_p, band_color = "<10%", 0.05, colors[0.1]
    elif y_last < q_vals[0.3]:
        # between 10 and 30
        t = (y_last - q_vals[0.1]) / max(1e-12, (q_vals[0.3] - q_vals[0.1]))
        approx_p = 0.10 + 0.20*t
        band_label, band_color = "10–30%", colors[0.3]
    elif y_last < q_vals[0.5]:
        t = (y_last - q_vals[0.3]) / max(1e-12, (q_vals[0.5] - q_vals[0.3]))
        approx_p = 0.30 + 0.20*t
        band_label, band_color = "30–50%", colors[0.5]
    elif y_last < q_vals[0.7]:
        t = (y_last - q_vals[0.5]) / max(1e-12, (q_vals[0.7] - q_vals[0.5]))
        approx_p = 0.50 + 0.20*t
        band_label, band_color = "50–70%", colors[0.7]
    elif y_last < q_vals[0.9]:
        t = (y_last - q_vals[0.7]) / max(1e-12, (q_vals[0.9] - q_vals[0.7]))
        approx_p = 0.70 + 0.20*t
        band_label, band_color = "70–90%", colors[0.9]
    else:
        band_label, approx_p, band_color = ">90%", 0.95, colors[0.9]

# ------------------------------ Traces

traces = []
visibility_normal, visibility_compare = [], []

# Quantile LINES for hover (ascending order): q10, q30, q50, q70, q90
q_lines_order = [0.1,0.3,0.5,0.7,0.9]
for q in q_lines_order:
    if q in qpred:
        xq, yq = qpred[q]
        traces.append(go.Scatter(
            x=xq, y=yq, mode="lines",
            name=f"q{int(q*100)}",
            line=dict(width=0.8, dash="dot", color=colors[q]),
            hovertemplate="Date: %{customdata}<br>%{fullData.name}: %{y:.6g}<extra></extra>",
            customdata=[(start_date + timedelta(days=float(d))).strftime("%m/%d/%y") for d in xq],
            showlegend=False
        ))
        visibility_normal.append(True)
        visibility_compare.append(False)

# Filled bands (hover suppressed; legend off)
def add_band(q_low, q_high):
    if q_low not in qpred or q_high not in qpred: return
    xl, yl = qpred[q_low]
    xh, yh = qpred[q_high]
    x_common = np.intersect1d(xl, xh)
    if len(x_common)==0: return
    yl_i = np.interp(x_common, xl, yl)
    yh_i = np.interp(x_common, xh, yh)
    # Upper boundary (thin)
    traces.append(go.Scatter(
        x=x_common, y=yh_i, mode="lines",
        line=dict(width=0.5, color=colors[q_high]),
        hoverinfo="skip", showlegend=False
    ))
    visibility_normal.append(True); visibility_compare.append(False)
    # Lower boundary (fill to previous)
    traces.append(go.Scatter(
        x=x_common, y=yl_i, mode="lines",
        line=dict(width=0.5, color=colors[q_low]),
        hoverinfo="skip", showlegend=False,
        fill="tonexty", fillcolor=rgba(colors[q_high], 0.18)
    ))
    visibility_normal.append(True); visibility_compare.append(False)

# Add bands from inner to outer (so fill stacks nicely)
add_band(0.3,0.5)
add_band(0.5,0.7)
add_band(0.1,0.3)
add_band(0.7,0.9)

# Main BTC (or BTC/USD) line — colored by current band
traces.append(go.Scatter(
    x=base["x_days"], y=y_main, mode="lines",
    name=y_label,
    line=dict(width=1.5, color=band_color),
    hovertemplate="Date: %{customdata}<br>Value: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"]
))
visibility_normal.append(True); visibility_compare.append(True)  # visible in both modes

# Compare mode traces (rebased main + (optional) rebased denominator)
traces.append(go.Scatter(
    x=base["x_days"], y=rebase_to_one(y_main),
    name="Main (rebased)", mode="lines",
    line=dict(width=1.5),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"], visible=False
))
visibility_normal.append(False); visibility_compare.append(True)

traces.append(go.Scatter(
    x=base["x_days"], y=[np.nan]*len(base),
    name="Denominator (rebased)", mode="lines",
    line=dict(width=1.5, dash="dash"),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"], visible=False
))
visibility_normal.append(False); visibility_compare.append(True)

fig = go.Figure(data=traces)

# Axes + layout
x_min, x_max = float(base["x_days"].min()), float(base["x_days"].max())
xticks, xticktext = make_log_time_ticks(start_date, x_min, x_max)

title_txt = f"BTC Purchase Indicator — {band_label} (p≈{approx_p:.2f})"
fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x unified",
    title=title_txt,
    xaxis=dict(
        type="log",
        title="Time (log scale)",
        tickvals=xticks, ticktext=xticktext,
        range=[math.log10(max(x_min,1e-6)), math.log10(max(x_max,1.0))]
    ),
    yaxis=dict(type="log", title=y_label + " (log scale)"),
    margin=dict(l=60,r=20,t=70,b=50)
)

# Menus
hover_menu = dict(
    buttons=[
        dict(label="Hover: Unified", method="relayout", args=[{"hovermode":"x unified"}]),
        dict(label="Hover: Per Trace", method="relayout", args=[{"hovermode":"x"}]),
        dict(label="Toggle Legend", method="relayout", args=[{"showlegend": not fig.layout.showlegend}]),
    ],
    direction="down", showactive=False, x=0.0, y=1.16, xanchor="left", yanchor="top",
)

mode_menu = dict(
    buttons=[
        dict(label="Normal", method="update",
             args=[{"visible": visibility_normal},
                   {"yaxis.title.text": y_label + " (log scale)", "hovermode":"x unified"}]),
        dict(label="Compare (Rebased)", method="update",
             args=[{"visible": visibility_compare},
                   {"yaxis.title.text": "Rebased to 1.0 (log scale)", "hovermode":"x unified"}]),
    ],
    direction="down", showactive=True, x=0.22, y=1.16, xanchor="left", yanchor="top",
)

denom_labels = ["USD/None"]; denom_values = ["USD"]
for k in sorted(denoms.keys()):
    denom_labels.append(k); denom_values.append(k)

def make_denom_update(denom_key):
    # Main series & label
    y_d, label_d = series_for_denom(base, denom_key)
    qpred_d = fit_quantiles(base["x_days"].values, y_d.values, quantiles)

    # Re-evaluate current band & color
    y_last_d = float(y_d.iloc[last_idx])
    q_vals_d = {q: interpolate_quantile_at_x(qpred_d, q, x_last) for q in quantiles}
    band_txt, band_col, p_est = "N/A", "#222", 0.0
    if all(not np.isnan(v) for v in q_vals_d.values()):
        if y_last_d < q_vals_d[0.1]:
            band_txt, band_col, p_est = "<10%", colors[0.1], 0.05
        elif y_last_d < q_vals_d[0.3]:
            t = (y_last_d - q_vals_d[0.1]) / max(1e-12, (q_vals_d[0.3]-q_vals_d[0.1]))
            band_txt, band_col, p_est = "10–30%", colors[0.3], 0.10+0.20*t
        elif y_last_d < q_vals_d[0.5]:
            t = (y_last_d - q_vals_d[0.3]) / max(1e-12, (q_vals_d[0.5]-q_vals_d[0.3]))
            band_txt, band_col, p_est = "30–50%", colors[0.5], 0.30+0.20*t
        elif y_last_d < q_vals_d[0.7]:
            t = (y_last_d - q_vals_d[0.5]) / max(1e-12, (q_vals_d[0.7]-q_vals_d[0.5]))
            band_txt, band_col, p_est = "50–70%", colors[0.7], 0.50+0.20*t
        elif y_last_d < q_vals_d[0.9]:
            t = (y_last_d - q_vals_d[0.7]) / max(1e-12, (q_vals_d[0.9]-q_vals_d[0.7]))
            band_txt, band_col, p_est = "70–90%", colors[0.9], 0.70+0.20*t
        else:
            band_txt, band_col, p_est = ">90%", colors[0.9], 0.95

    # Prepare per-trace updates IN ORDER:
    updates = []

    # q-lines (q10, q30, q50, q70, q90) occupy indices 0..(nq-1)
    nq = sum(1 for q in q_lines_order if q in qpred)  # original counts
    # Rebuild lines for new denom
    for q in q_lines_order:
        if q in qpred_d:
            xq, yq = qpred_d[q]
            updates.append({"x": xq, "y": yq, "name": f"q{int(q*100)}"})
        else:
            updates.append({"y":[np.nan]})  # keep slot

    # Band pairs (added next): keep geometry but values change
    def band_pair(q_low,q_high):
        if q_low in qpred_d and q_high in qpred_d:
            xl, yl = qpred_d[q_low]; xh, yh = qpred_d[q_high]
            x_common = np.intersect1d(xl, xh)
            if len(x_common):
                yl_i = np.interp(x_common, xl, yl)
                yh_i = np.interp(x_common, xh, yh)
                updates.append({"x": x_common, "y": yh_i})
                updates.append({"x": x_common, "y": yl_i})
                return
        updates.append({"y":[np.nan]}); updates.append({"y":[np.nan]})
    band_pair(0.3,0.5)
    band_pair(0.5,0.7)
    band_pair(0.1,0.3)
    band_pair(0.7,0.9)

    # Main line (colored by band)
    updates.append({"y": y_d, "name": label_d, "line.color": band_col})

    # Compare traces (rebased main + denom)
    updates.append({"y": rebase_to_one(y_d), "name": "Main (rebased)"})
    if denom_key and denom_key.upper() in denoms:
        dfden = denoms[denom_key.upper()]
        dfm = base.merge(dfden.rename(columns={"price":"den"}), on="date", how="left")
        updates.append({"y": rebase_to_one(dfm["den"]), "name": f"{denom_key.upper()} (rebased)"})
    else:
        updates.append({"y": [np.nan]*len(base), "name": "Denominator (rebased)"})

    layout_updates = {
        "yaxis.title.text": label_d + " (log scale)",
        "title.text": f"BTC Purchase Indicator — {band_txt} (p≈{p_est:.2f})"
    }
    return updates, layout_updates

denom_buttons = []
for lab, val in zip(["USD/None"]+[k for k in sorted(denoms.keys())],
                    ["USD"]+[k for k in sorted(denoms.keys())]):
    tr_updates, ly_updates = make_denom_update(val)
    denom_buttons.append(dict(
        label=lab, method="update",
        args=[[tr_updates,], ly_updates]
    ))

denom_menu = dict(
    buttons=denom_buttons, direction="down", showactive=True,
    x=0.48, y=1.16, xanchor="left", yanchor="top"
)

fig.update_layout(updatemenus=[hover_menu, mode_menu, denom_menu])

# Final write
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
print(f"Wrote {OUTPUT_HTML}")
