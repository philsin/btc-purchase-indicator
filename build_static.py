# build_static.py · BTC Purchase Indicator (static build)
# --------------------------------------------------------
# Outputs: ./dist/index.html (uses template.html)
# Data sources: Stooq BTC & Gold (public daily CSVs)
#
# How to run:
#   python build_static.py
#
# Key design:
# - X axis is log10(days since Genesis), matching your power-law view.
# - Bands are fit in USD space, then converted to Gold oz/BTC by dividing by gold price.
# - "Gold" denomination = Gold oz / BTC = (BTCUSD / GoldUSD).
# - Hover panel hides BTC row when the selected date is > 6 months past last historical day.
#
# Edit points you'll most likely tweak in the future:
#  (A) Sigma levels (Support/Bear/Mid/Frothy/Top)
#  (B) How far projections go (PROJ_END)
#  (C) Styling: names, colors, dash, widths
#
# --------------------------------------------------------

import os, json, math, io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------ Parameters you can safely tweak ------------------

GENESIS = pd.Timestamp("2009-01-03", tz="UTC")
PROJ_END = pd.Timestamp("2040-12-31", tz="UTC")

# (A) Sigma levels (legend names are applied in the template)
LEVELS = {
    "Support":    -1.5,
    "Bear":       -0.75,   # << updated as requested
    "PL Best Fit": 0.0,
    "Frothy":     +1.0,
    "Top":        +2.0,    # << Top at +2σ
}

COLORS = {
    "Support":    "red",
    "Bear":       "rgba(255,100,100,1)",
    "PL Best Fit":"white",
    "Frothy":     "rgba(100,255,100,1)",
    "Top":        "green",
    "BTC":        "gold",
}

LINE_DASH = {
    "Support":    "dash",
    "Bear":       "dash",
    "PL Best Fit":"dash",
    "Frothy":     "dash",
    "Top":        "dash",
    "BTC":        "solid",
}

# Fallback sigma floor so bands never collapse visually
SIGMA_FLOOR = 0.25

# --------------------------------------------------------------------


def _read_csv(url: str, date_col_guess=("Date", "date")) -> pd.DataFrame:
    df = pd.read_csv(url)
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    # find date column
    date_col = None
    for dc in date_col_guess:
        if dc.lower() in df.columns:
            date_col = dc.lower()
            break
    if date_col is None:
        # try best guess
        for c in df.columns:
            if "date" in c:
                date_col = c
                break
    if date_col is None:
        raise RuntimeError("No date column found for CSV at " + url)
    # standardize
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def fetch_btc_daily() -> pd.DataFrame:
    # Stooq BTC/USD daily
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = _read_csv(url)
    # price column (stooq uses 'close')
    price_col = None
    for c in df.columns:
        if "close" in c or "price" in c:
            price_col = c
            break
    if price_col is None:
        raise RuntimeError("No price/close column found in BTC CSV")
    df = df.rename(columns={price_col: "btc"})
    df["btc"] = pd.to_numeric(df["btc"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["btc"])[["date", "btc"]]


def fetch_gold_daily() -> pd.DataFrame:
    # Stooq XAU/USD daily
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = _read_csv(url)
    price_col = None
    for c in df.columns:
        if "close" in c or "price" in c:
            price_col = c
            break
    if price_col is None:
        raise RuntimeError("No price/close column found in Gold CSV")
    df = df.rename(columns={price_col: "gold"})
    df["gold"] = pd.to_numeric(df["gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["gold"])[["date", "gold"]]


def daily_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")


def log_time_days(dts: pd.Series) -> np.ndarray:
    # days since GENESIS (minimum 1 day), then log10
    days = (dts - GENESIS).dt.days.clip(lower=1).astype(float)
    return np.log10(days.to_numpy())


def fit_power_usd(dates: pd.Series, price_usd: pd.Series):
    X = log_time_days(dates)
    y = np.log10(price_usd.to_numpy())
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    resid = y - mid_log
    sigma = float(np.std(resid))
    return slope, intercept, sigma, mid_log


def build_projection_frame(hist: pd.DataFrame) -> pd.DataFrame:
    # extend to monthly steps through PROJ_END (bands only)
    last_hist = hist["date"].max()
    if last_hist.tz is None:
        last_hist = last_hist.tz_localize("UTC")
    if PROJ_END <= last_hist:
        future = pd.DatetimeIndex([], tz="UTC")
    else:
        future = pd.date_range(last_hist + pd.offsets.MonthBegin(1), PROJ_END, freq="MS", tz="UTC")
    full = pd.concat([hist[["date", "btc", "gold"]], pd.DataFrame({"date": future})], ignore_index=True)
    return full


def compute_series():
    # 1) Fetch
    btc = fetch_btc_daily()
    gold = fetch_gold_daily()

    # 2) Align onto a daily calendar (inner join on date then reindex on full daily union)
    start = max(btc["date"].min(), gold["date"].min())
    end = min(btc["date"].max(), gold["date"].max())
    cal = daily_calendar(start, end)

    df = pd.merge_asof(
        btc.sort_values("date"),
        gold.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta("1D"),
    ).dropna()

    # Reindex to strict daily calendar with ffill to fill any small gaps
    df = df.set_index("date").reindex(cal).ffill().reset_index().rename(columns={"index": "date"})

    # 3) Gold denomination: Gold oz / BTC = BTCUSD / GoldUSD
    df["oz_per_btc"] = df["btc"] / df["gold"]

    # 4) Fit USD power-law on historical range
    slope, intercept, sigma, mid_log_hist = fit_power_usd(df["date"], df["btc"])
    sigma_vis = max(sigma, SIGMA_FLOOR)

    # 5) Build projection frame (for bands out to 2040)
    full = build_projection_frame(df)
    X_full = log_time_days(full["date"])
    mid_log_full = slope * X_full + intercept
    mid_full_usd = 10 ** mid_log_full

    # 6) USD bands
    bands_usd = {name: 10 ** (mid_log_full + sigma_vis * k) for name, k in LEVELS.items()}

    # 7) Convert USD bands to Gold oz/BTC by dividing by gold price (forward-fill for future months)
    full["gold"] = full["gold"].ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        bands_gld = {name: np.where(full["gold"].to_numpy() > 0,
                                    bands_usd[name] / full["gold"].to_numpy(),
                                    np.nan) for name in LEVELS.keys()}

    # 8) Arrays for template
    arrays = {
        "dates": [d.strftime("%Y-%m-%d") for d in full["date"]],
        "usd": {name: bands_usd[name].tolist() for name in LEVELS.keys()},
        "gld": {name: bands_gld[name].tolist() for name in LEVELS.keys()},
        "hist": {
            "dates": [d.strftime("%Y-%m-%d") for d in df["date"]],
            "usd": df["btc"].round(0).tolist(),
            "gld": df["oz_per_btc"].tolist(),  # keep decimals; template formats
        },
    }

    # 9) Build figure
    fig = build_fig(arrays)

    return arrays, fig


def build_fig(arrays: dict) -> go.Figure:
    """Construct Plotly figure with both denominations present, controlled by legendgroup."""
    dates_iso = arrays["dates"]
    dates_dt = pd.to_datetime(dates_iso, utc=True)
    x_log = np.log10(((dates_dt - GENESIS).days.clip(lower=1)).astype(float))

    # helper
    def add_line(fig, y, name, color, dash, group, visible):
        fig.add_trace(go.Scatter(
            x=x_log, y=y,
            name=name, legendgroup=group, showlegend=True,
            visible=visible,
            mode="lines",
            line=dict(color=color, dash=dash, width=2),
            hoverinfo="skip",
        ))

    fig = go.Figure()

    # --- USD bands & BTC
    usd_group = "USD"
    # order: Top, Frothy, Mid, Bear, Support to match your desired legend order
    for nm in ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]:
        add_line(
            fig, arrays["usd"][nm],
            f"{nm}", COLORS[nm], LINE_DASH[nm],
            usd_group, True  # USD visible by default
        )
    # BTC USD history
    hist_dt = pd.to_datetime(arrays["hist"]["dates"], utc=True)
    hist_x = np.log10(((hist_dt - GENESIS).days.clip(lower=1)).astype(float))
    fig.add_trace(go.Scatter(
        x=hist_x, y=arrays["hist"]["usd"],
        name="BTC", legendgroup=usd_group, showlegend=True, visible=True,
        mode="lines", line=dict(color=COLORS["BTC"], width=3),
        hoverinfo="skip",
    ))
    # yellow marker (USD)
    usd_mark_idx = len(fig.data)
    fig.add_trace(go.Scatter(
        x=[hist_x[-1]], y=[arrays["hist"]["usd"][-1]],
        name="", legendgroup=usd_group, showlegend=False, visible=True,
        mode="markers", marker=dict(color="#facc15", size=8),
        hoverinfo="skip",
    ))

    # --- GOLD bands & BTC (oz/BTC)
    gld_group = "GLD"
    for nm in ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]:
        add_line(
            fig, arrays["gld"][nm],
            f"{nm}", COLORS[nm], LINE_DASH[nm],
            gld_group, False  # hidden by default; toggled via dropdown
        )
    # BTC in oz/BTC
    fig.add_trace(go.Scatter(
        x=hist_x, y=arrays["hist"]["gld"],
        name="BTC", legendgroup=gld_group, showlegend=True, visible=False,
        mode="lines", line=dict(color=COLORS["BTC"], width=3),
        hoverinfo="skip",
    ))
    # yellow marker (GLD)
    gld_mark_idx = len(fig.data)
    fig.add_trace(go.Scatter(
        x=[hist_x[-1]], y=[arrays["hist"]["gld"][-1]],
        name="", legendgroup=gld_group, showlegend=False, visible=False,
        mode="markers", marker=dict(color="#facc15", size=8),
        hoverinfo="skip",
    ))

    # Axis & layout (log-time on x, log on y)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=40, r=20, t=10, b=40),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
        xaxis=dict(
            title="Year",
            type="linear",  # our x is already log10(days)
            showgrid=True, gridwidth=0.5, zeroline=False,
        ),
        yaxis=dict(
            title="USD / BTC",
            type="log",
            showgrid=True, gridwidth=0.5, zeroline=False,
            tickformat="$,d",
        ),
        # meta: indices of the two marker traces for JS access
        meta=dict(USD_MARK_IDX=usd_mark_idx, GLD_MARK_IDX=gld_mark_idx)
    )

    return fig


def write_dist(arrays: dict, fig: go.Figure):
    os.makedirs("dist", exist_ok=True)
    with open("template.html", "r", encoding="utf-8") as f:
        tpl = f.read()
    figjson = fig.to_json()  # safe JSON for Plotly
    arrays_json = json.dumps(arrays, separators=(",", ":"))  # compact
    html = (
        tpl.replace("__TITLE__", "BTC Purchase Indicator")
           .replace("__FIGJSON__", figjson)
           .replace("__ARRAYS__", arrays_json)
    )
    with open("dist/index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    arrays, fig = compute_series()
    write_dist(arrays, fig)
    print(f"[build] rows: {len(arrays['hist']['dates']):,}  (BTC & Gold)")
    print("[build] wrote dist/index.html")


if __name__ == "__main__":
    main()
