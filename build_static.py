#!/usr/bin/env python3
"""
Builds a self-contained Plotly HTML with:
- Log-scaled time (via numeric "days since start") + human-readable date ticks
- Log-scaled y-axis
- Quantile regression bands (10/30/50/70/90) around the 50% median curve
- Legend + hover panel toggle
- Compare Mode (rebased to 1.0) that hides bands
- Denominator menu (auto-detects denominator_*.csv)
- Short date formatting on ticks and hover (MM/DD/YY)

Requirements:
    pandas, numpy, plotly, statsmodels
"""

import os
import glob
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.regression.quantile_regression import QuantReg

OUTPUT_HTML = "docs/index.html"  # adjust if your GH Pages serves from another path
DATA_DIR = "data"
BTC_FILE = os.path.join(DATA_DIR, "btc_usd.csv")  # columns: date, price

# ---------- Helpers

def load_series_csv(path):
    df = pd.read_csv(path)
    # Flexible column casing
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

def days_since_start(dates, start):
    # +1 to avoid log(0)
    return (dates - start).dt.days.astype(float) + 1.0

def make_log_time_ticks(start_date, x_min, x_max):
    """
    Build nice log ticks on x (in 'days since start') and map them back to dates.
    We use powers of 10 * {1,2,5}.
    """
    ticks = []
    ticktexts = []

    if x_min <= 0:
        x_min = 1e-6

    exp_min = math.floor(math.log10(x_min))
    exp_max = math.ceil(math.log10(x_max))

    mantissas = [1, 2, 5]
    for e in range(exp_min, exp_max + 1):
        for m in mantissas:
            val = m * (10 ** e)
            if x_min <= val <= x_max:
                d = start_date + timedelta(days=float(val))
                ticks.append(val)
                ticktexts.append(d.strftime("%m/%d/%y"))
    # Ensure we always include first & last dates
    if 1.0 >= x_min and 1.0 <= x_max:
        ticks = sorted(set(ticks + [1.0]))
        ticktexts = []
        for v in ticks:
            d = start_date + timedelta(days=float(v))
            ticktexts.append(d.strftime("%m/%d/%y"))
    return ticks, ticktexts

def fit_quantiles(x, y, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    Fit log-log quantile regressions:
        log10(y) ~ a + b * log10(x)
    Returns dict of q -> predicted y for sorted x.
    """
    # Use only positive x,y
    m = (x > 0) & (y > 0)
    x_use = x[m]
    y_use = y[m]
    if len(x_use) < 10:
        return {}

    X = pd.DataFrame({"logx": np.log10(x_use)})
    z = np.log10(y_use)

    preds = {}
    for q in quantiles:
        try:
            model = QuantReg(z, pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1))
            res = model.fit(q=q)
            # Predict over the *full* x range (sorted)
            x_sorted = np.sort(x.unique())
            x_sorted = x_sorted[x_sorted > 0]
            Xp = pd.DataFrame({"const": 1.0, "logx": np.log10(x_sorted)})
            zhat = res.predict(Xp)
            yhat = (10 ** zhat)
            preds[q] = (x_sorted, yhat)
        except Exception:
            # If a quantile fails (rare), skip it
            continue
    return preds

def rebase_to_one(series):
    """Rebase a positive series to 1.0 at the first valid point."""
    s = series.copy().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return series * np.nan
    base = s.iloc[0]
    if base <= 0:
        return series * np.nan
    return series / base

def collect_denominators():
    """Find files like data/denominator_*.csv."""
    paths = glob.glob(os.path.join(DATA_DIR, "denominator_*.csv"))
    opts = {}
    for p in sorted(paths):
        key = os.path.splitext(os.path.basename(p))[0].replace("denominator_", "").upper()
        try:
            opts[key] = load_series_csv(p)
        except Exception:
            # skip invalid files
            pass
    return opts

# ---------- Load data

btc = load_series_csv(BTC_FILE).rename(columns={"price": "btc"})
denoms = collect_denominators()  # e.g., {"SPX": df, "GOLD": df}
has_denoms = len(denoms) > 0

# Merge denominators onto btc date index for clean ratios
base = btc.copy()
for name, df in denoms.items():
    base = base.merge(df.rename(columns={"price": name.lower()}), on="date", how="left")

# Compute “days since start” for log-time x
start_date = base["date"].iloc[0]
base["x_days"] = days_since_start(base["date"], start_date)

# ---------- Core series & denominators

def series_for_denominator(df, denom_key=None):
    """Return (y_main, label) where y_main is either BTC/USD or BTC/denominator."""
    if not denom_key or denom_key.lower() == "usd" or denom_key.lower() == "none":
        return df["btc"], "BTC (USD)"
    key = denom_key.lower()
    if key in df.columns:
        return (df["btc"] / df[key]), f"BTC / {denom_key.upper()}"
    # Fallback if missing
    return df["btc"], "BTC (USD)"

# ---------- Quantile bands on log-log (time-log, price-log)

y_main, y_label = series_for_denominator(base, denom_key=None)
quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)
qpred = fit_quantiles(base["x_days"].values, y_main.values, quantiles=quantiles)

# For hover with date formatting while x is numeric:
base["date_str"] = base["date"].dt.strftime("%m/%d/%y")

# ---------- Build traces

traces_all = []
visibility_bands = []   # bands visible in normal mode, hidden in compare
visibility_normal = []  # base state
visibility_compare = [] # compare state

# Main BTC (or BTC/USD) line
trace_btc = go.Scatter(
    x=base["x_days"],
    y=y_main,
    name=y_label,
    mode="lines",
    line=dict(width=1.5),
    hovertemplate="Date: %{customdata}<br>Value: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"],
)
traces_all.append(trace_btc)

# Quantile median curve + bands (fill between 10-30, 30-50, 50-70, 70-90)
# Colors: red(10) → yellow(50) → green(90) via semi-transparent fills
# We'll draw inner to outer for clean stacking.
def rgba(hex_color, alpha):
    # hex_color like "#FF0000"
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

colors = {
    0.1: "#D32F2F", # red-ish
    0.3: "#F57C00", # orange
    0.5: "#FBC02D", # yellow
    0.7: "#7CB342", # yellow-green
    0.9: "#2E7D32", # green
}

# Draw median line (thin)
if 0.5 in qpred:
    x50, y50 = qpred[0.5]
    traces_all.append(go.Scatter(
        x=x50, y=y50,
        name="Median (50%)",
        mode="lines",
        line=dict(width=1.2, dash="dot", color=colors[0.5]),
        hovertemplate="Date: %{customdata}<br>Median: %{y:.6g}<extra></extra>",
        customdata=[(start_date + timedelta(days=float(d))).strftime("%m/%d/%y") for d in x50],
    ))
else:
    x50, y50 = (None, None)

# Helper to plot filled band between q_low and q_high
def add_band(q_low, q_high):
    if q_low not in qpred or q_high not in qpred:
        return
    xl, yl = qpred[q_low]
    xh, yh = qpred[q_high]
    # Ensure same x grid (both are sorted)
    # We assume xl == xh; if not, align by intersection
    x_common = np.intersect1d(xl, xh)
    if len(x_common) == 0:
        return
    yl_i = np.interp(x_common, xl, yl)
    yh_i = np.interp(x_common, xh, yh)
    # Upper boundary
    traces_all.append(go.Scatter(
        x=x_common, y=yh_i,
        name=f"{int(q_low*100)}–{int(q_high*100)}% band",
        mode="lines",
        line=dict(width=0.5, color=colors[q_high]),
        hoverinfo="skip",
        showlegend=False,
    ))
    # Lower boundary (fill down to this)
    traces_all.append(go.Scatter(
        x=x_common, y=yl_i,
        mode="lines",
        line=dict(width=0.5, color=colors[q_low]),
        hoverinfo="skip",
        fill="tonexty",
        fillcolor=rgba(colors[q_high], 0.18),
        name=f"{int(q_low*100)}–{int(q_high*100)}% band",
        showlegend=False,
    ))

# Add bands outward (50±20, 50±40)
add_band(0.3, 0.5)
add_band(0.5, 0.7)
add_band(0.1, 0.3)
add_band(0.7, 0.9)

# Track visibilities:
# Index 0 → main line
# Next traces are median + bands (variable count)
band_start_index = 1
band_end_index = len(traces_all) - 1

for i, _ in enumerate(traces_all):
    if i == 0:
        visibility_normal.append(True)   # main line visible
        visibility_compare.append(True)  # also visible (rebased version is different trace)
    else:
        visibility_normal.append(True)   # bands + median visible in normal mode
        visibility_compare.append(False) # hidden in compare mode

# ---------- Compare Mode traces (rebased BTC and rebased Denominator)

# We will add:
# - Rebased main (BTC or BTC/denom) line (thin solid)
# - Optional rebased denominator line (if denominator selected), to compare directly
# These are visible only in compare mode.

# Placeholder traces (we update data via buttons)
trace_compare_main = go.Scatter(
    x=base["x_days"], y=rebase_to_one(y_main),
    name="Main (rebased)",
    mode="lines",
    line=dict(width=1.5),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"],
    visible=False,
)
traces_all.append(trace_compare_main)
visibility_normal.append(False)
visibility_compare.append(True)

# “Denominator (rebased)” starts as NaN (no denom in default 'USD/none' view)
trace_compare_denom = go.Scatter(
    x=base["x_days"],
    y=[np.nan]*len(base),
    name="Denominator (rebased)",
    mode="lines",
    line=dict(width=1.5, dash="dash"),
    hovertemplate="Date: %{customdata}<br>Rebased: %{y:.6g}<extra>%{fullData.name}</extra>",
    customdata=base["date_str"],
    visible=False,
)
traces_all.append(trace_compare_denom)
visibility_normal.append(False)
visibility_compare.append(True)

# ---------- Figure

fig = go.Figure(data=traces_all)

# Axes: time log on x (numeric), log y
x_min = float(base["x_days"].min())
x_max = float(base["x_days"].max())
xticks, xticktext = make_log_time_ticks(start_date, x_min, x_max)

fig.update_layout(
    template="plotly_white",
    showlegend=True,
    hovermode="x unified",  # default; toggleable
    xaxis=dict(
        type="log",
        title="Time (log scale)",
        tickvals=xticks,
        ticktext=xticktext,
        range=[math.log10(max(x_min, 1e-6)), math.log10(max(x_max, 1.0))],
    ),
    yaxis=dict(
        type="log",
        title=y_label + " (log scale)",
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    title="BTC Purchase Indicator (Log Time & Log Value)",
)

# ---------- Menus

# 1) Hover panel toggle (unified vs. closest), and legend toggle
hover_menu = dict(
    buttons=[
        dict(
            label="Hover: Unified",
            method="relayout",
            args=[{"hovermode": "x unified"}],
        ),
        dict(
            label="Hover: Per Trace",
            method="relayout",
            args=[{"hovermode": "x"}],
        ),
        dict(
            label="Toggle Legend",
            method="relayout",
            args=[{"showlegend": not fig.layout.showlegend}],
        ),
    ],
    direction="down",
    showactive=False,
    x=0.0, y=1.16,
    xanchor="left", yanchor="top",
)

# 2) Mode toggle: Normal vs Compare (hide bands in Compare)
mode_menu = dict(
    buttons=[
        dict(
            label="Normal",
            method="update",
            args=[
                {"visible": visibility_normal},
                {
                    "yaxis.title.text": y_label + " (log scale)",
                    "hovermode": "x unified",
                },
            ],
        ),
        dict(
            label="Compare (Rebased)",
            method="update",
            args=[
                {"visible": visibility_compare},
                {
                    "yaxis.title.text": "Rebased to 1.0 (log scale)",
                    "hovermode": "x unified",
                },
            ],
        ),
    ],
    direction="down",
    showactive=True,
    x=0.22, y=1.16,
    xanchor="left", yanchor="top",
)

# 3) Denominator menu (including NONE/USD)
denom_labels = ["USD/None"]
denom_values = ["USD"]
for k in sorted(denoms.keys()):
    denom_labels.append(k)
    denom_values.append(k)

def make_update_for_denom(denom_key):
    # Update main series + label
    y_main_d, label_d = series_for_denominator(base, denom_key=denom_key)
    rebased_main = rebase_to_one(y_main_d)

    # Recompute quantile preds for normal mode
    qpred_d = fit_quantiles(base["x_days"].values, y_main_d.values, quantiles=quantiles)

    # Prepare new data arrays for:
    # index 0: main line
    new_traces = {
        0: dict(y=y_main_d, name=label_d, hovertemplate="Date: %{customdata}<br>Value: %{y:.6g}<extra>%{fullData.name}</extra>"),
    }

    # Median + bands are between indices [1..band_end_index]
    # We’ll rebuild them “in place” (if available). If missing, set to NaN to hide shape but keep visibility logic.
    # Start by defaulting to NaNs:
    for i in range(1, band_end_index + 1):
        new_traces[i] = dict(y=[np.nan]*len(traces_all[i]["x"]))

    if 0.5 in qpred_d:
        x50_d, y50_d = qpred_d[0.5]
        # Find the median trace index (first after main)
        new_traces[1] = dict(x=x50_d, y=y50_d, name="Median (50%)")

    # Rebuild each band pair in the same append order used earlier:
    # Order added earlier:
    #   median (1)
    #   30–50 upper (2), 30–50 lower (3)
    #   50–70 upper (4), 50–70 lower (5)
    #   10–30 upper (6), 10–30 lower (7)
    #   70–90 upper (8), 70–90 lower (9)
    # We must align to match existing indices.
    band_pairs = [(0.3,0.5),(0.5,0.7),(0.1,0.3),(0.7,0.9)]
    idx = 2
    for (ql, qh) in band_pairs:
        if ql in qpred_d and qh in qpred_d:
            xl, yl = qpred_d[ql]
            xh, yh = qpred_d[qh]
            x_common = np.intersect1d(xl, xh)
            if len(x_common) > 0:
                yl_i = np.interp(x_common, xl, yl)
                yh_i = np.interp(x_common, xh, yh)
                # upper
                new_traces[idx]   = dict(x=x_common, y=yh_i)
                # lower
                new_traces[idx+1] = dict(x=x_common, y=yl_i)
        idx += 2

    # Compare traces (last two) — main rebased + denominator rebased
    # index: len(traces_all)-2 → main rebased
    # index: len(traces_all)-1 → denom rebased (only if denom selected and exists)
    i_main_cmp = len(traces_all) - 2
    i_denom_cmp = len(traces_all) - 1

    new_traces[i_main_cmp] = dict(y=rebased_main, name="Main (rebased)")

    if denom_key and denom_key.upper() in denoms:
        denom_df = denoms[denom_key.upper()]
        dfm = base.merge(denom_df.rename(columns={"price": "den"}), on="date", how="left")
        rebased_den = rebase_to_one(dfm["den"])
        new_traces[i_denom_cmp] = dict(y=rebased_den, name=f"{denom_key.upper()} (rebased)")
    else:
        new_traces[i_denom_cmp] = dict(y=[np.nan]*len(base), name="Denominator (rebased)")

    # Title update
    new_layout = {"yaxis.title.text": (label_d + " (log scale)")}

    # Build args for "restyle" with a list in trace order
    restyle_args = []
    for i in range(len(traces_all)):
        restyle_args.append(new_traces.get(i, {}))

    return [{"yaxis.type": "log"}, restyle_args, new_layout]

denom_buttons = []
for lab, val in zip(denom_labels, denom_values):
    args = make_update_for_denom(val)
    # Plotly "updatemenus" supports a combination of "relayout" + "restyle" + "update",
    # but here we’ll use method='update' with args=[trace_updates, layout_updates]
    # To pass multiple instruction types, we can wrap them with a custom key ordering by leveraging
    # Plotly's 'args' semantics. Simpler approach: use method='update' and include {"yaxis.type":...} in layout.
    denom_buttons.append(dict(
        label=lab,
        method="update",
        args=[
            # traces (in order)
            [{"y": a.get("y", None), "x": a.get("x", None), "name": a.get("name", None), "hovertemplate": a.get("hovertemplate", None)} for a in args[1]],
            args[2]  # layout
        ],
    ))

denom_menu = dict(
    buttons=denom_buttons,
    direction="down",
    showactive=True,
    x=0.48, y=1.16,
    xanchor="left", yanchor="top",
)

fig.update_layout(
    updatemenus=[hover_menu, mode_menu, denom_menu]
)

# Ensure visibility states match initial "Normal" mode
fig.update_traces(visible=None)  # keep as configured above

# ---------- Write HTML

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
print(f"Wrote {OUTPUT_HTML}")
