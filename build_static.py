#!/usr/bin/env python3 
# ─────────────────────────────────────────────────────────────
# build_static.py  ·  BTC Purchase Indicator (static Plotly)
#  - Power-law bands on log-time (days since 2009-01-03)
#  - Denomination: USD/BTC or Gold oz/BTC
#  - Unified hover: shows all lines; header = Month Year
#  - Weekly (Monday) slider, snaps to nearest historical point
#  - BTC marker follows slider
#  - Readout shows only date + BTC price
#  - Tap outside chart hides hover box
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import io, json, requests, numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

GENESIS  = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
UA       = {"User-Agent": "btc-pl-pages/1.2"}

LEVELS = {
    "Top":         +2.0,    
    "Frothy":      +1.0,
    "PL Best Fit":  0.0,
    "Bear":        -0.75,
    "Support":     -1.5,
}

def _fmt_sigma_for_legend(k: float) -> str:
    """Format kσ for legend with only significant figures:
       +2, +1, 0, -0.5, -1.5 …"""
    if abs(k) < 1e-12:
        return "0"
    # +g keeps only significant digits and adds the sign
    return f"{k:+g}"
    
COLORS = {
    "Top":         "#16a34a",
    "Frothy":      "#86efac",
    "PL Best Fit": "#ffffff",
    "Bear":        "#fda4af",
    "Support":     "#ef4444",
    "BTC":         "#ffd166",
    "BTC_MARK":    "#ffd166",
}
DASHES = {k: "dash" for k in LEVELS}
DASHES["PL Best Fit"] = "dash"

# --------------------- loaders
def _read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _btc_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    ccol = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={dcol: "Date", ccol: "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github() -> pd.DataFrame:
    df = _read_csv("https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv")
    df = df.rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _gold_stooq() -> pd.DataFrame:
    df = _read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    ccol = [c for c in df.columns if ("close" in c) or ("price" in c)][-1]
    df = df.rename(columns={dcol: "Date", ccol: "Gold"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def _gold_lbma() -> pd.DataFrame:
    df = _read_csv("https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    col = "USD (PM)" if "USD (PM)" in df.columns else ("USD (AM)" if "USD (AM)" in df.columns else [c for c in df.columns if "USD" in c.upper()][0])
    df = df.rename(columns={col: "Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().query("Gold>0").sort_values("Date")[["Date","Gold"]]

def load_btc_gold() -> pd.DataFrame:
    try:
        btc = _btc_stooq()
        if len(btc) < 1000: raise ValueError
    except Exception:
        btc = _btc_github()
    try:
        gold = _gold_stooq()
        if len(gold) < 1000: raise ValueError
    except Exception:
        gold = _gold_lbma()
    gold_ff = gold.set_index("Date").reindex(btc["Date"]).ffill().reset_index().rename(columns={"index":"Date"})
    return btc.merge(gold_ff, on="Date", how="left").dropna()

# --------------------- math
def log_days(dates) -> np.ndarray:
    td = pd.to_datetime(dates) - GENESIS
    if isinstance(td, pd.Series):
        days = td.dt.days.to_numpy()
    elif isinstance(td, pd.TimedeltaIndex):
        days = td.days.astype(float)
    else:
        days = (np.asarray(td) / np.timedelta64(1, "D")).astype(float)
    days = np.where(days <= 0, np.nan, days)
    return np.log10(days)

def fit_power(dates: pd.Series, values: pd.Series):
    X = log_days(dates)
    y = np.log10(values.to_numpy(dtype="float64"))
    m, b = np.polyfit(X[np.isfinite(X)], y[np.isfinite(X)], 1)
    sigma = float(np.std(y[np.isfinite(X)] - (m*X[np.isfinite(X)] + b)))
    return m, b, sigma

def build_bands(dates, m, b, sigma) -> dict:
    X = log_days(dates)
    mid = 10 ** (m*X + b)
    out = {"mid": mid}
    for name, k in LEVELS.items():
        if name == "PL Best Fit": continue
        out[name] = 10 ** (m*X + b + sigma*k)
    return out

def year_ticks(start=2012, dense_until=2020, end=2040):
    years = list(range(start, dense_until+1)) + list(range(dense_until+2, end+1, 2))
    vals  = log_days([pd.Timestamp(f"{y}-01-01") for y in years]).tolist()
    text  = [str(y) for y in years]
    return vals, text

# --------------------- figure
def make_powerlaw_fig(df: pd.DataFrame):
    """
    Builds the power-law chart on log-time (x) with:
      - USD and Gold (oz/BTC) denominations (Gold hidden initially)
      - Lines: Top, Frothy, PL Best Fit, Bear, Support, BTC
      - One composite hover per denomination (Month Year + all 6 values)
      - Colored hover rows matching line colors
      - Slider-driven BTC marker (indices exposed via layout.meta)
      - Projection of bands monthly to 2040-12
      - Legend names include σ with significant figures, e.g. Top (+2σ), Frothy (+1σ) …
    Returns: fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist
    """
    # ----- series -----
    usd = df["BTC"].astype(float)                  # USD / BTC (price)
    gld = (df["BTC"] / df["Gold"]).astype(float)   # Gold oz / BTC

    # ----- power-law fits (log-time) -----
    m_u, b_u, s_u = fit_power(df["Date"], usd)
    m_g, b_g, s_g = fit_power(df["Date"], gld)

    # ----- extend dates to 2040 (monthly) -----
    last = df["Date"].iloc[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    full_dates = pd.Index(df["Date"]).append(future)

    # ----- bands over full span -----
    bands_usd = build_bands(full_dates, m_u, b_u, s_u)   # keys: mid, Top, Frothy, Bear, Support
    bands_gld = build_bands(full_dates, m_g, b_g, s_g)

    # ----- x arrays (log10 days since GENESIS) -----
    x_hist = log_days(df["Date"])
    x_full = log_days(full_dates)

    # Month-Year strings aligned to x_full (for composite hover headers)
    monyr_full = np.array([pd.Timestamp(d).strftime("%b %Y") for d in full_dates])

    # Align historical BTC series to full_dates for hover readout
    usd_full = (
        pd.Series(usd.values, index=df["Date"])
          .reindex(full_dates).ffill().to_numpy()
    )
    gld_full = (
        pd.Series(gld.values, index=df["Date"])
          .reindex(full_dates).ffill().to_numpy()
    )

    fig = go.Figure()
    order = ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]

    # ================= USD group (visible) =================
    # Lines (suppress their own hover; we’ll use one composite)
    for name in order:
        y = bands_usd["mid"] if name == "PL Best Fit" else bands_usd[name]
        # Legend name with σ in significant figures
        sig = _fmt_sigma_for_legend(LEVELS[name])
        legend_name = f"{name} ({sig}σ)"
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=legend_name, legendgroup="USD",
            hoverinfo="skip",  # <- only composite hover shows
            visible=True
        ))

    # BTC (USD)
    fig.add_trace(go.Scatter(
        x=x_hist, y=usd, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC", legendgroup="USD",
        hoverinfo="skip",
        visible=True
    ))
    USD_BTC_IDX = len(fig.data) - 1

    # Slider-driven BTC marker (USD)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC_MARK"], size=8, line=dict(color="#000", width=.5)),
        name=" ", legendgroup="USD", showlegend=False, hoverinfo="skip", visible=True
    ))
    USD_MARK_IDX = len(fig.data) - 1

    # Composite hover (USD): Month Year + color-coded rows
    hover_usd = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:" + COLORS["Top"] + "'>Top</span> | "
        "<span style='color:" + COLORS["Top"] + "'>%{customdata[1]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Frothy"] + "'>Frothy</span> | "
        "<span style='color:" + COLORS["Frothy"] + "'>%{customdata[2]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["PL Best Fit"] + "'>PL Best Fit</span> | "
        "<span style='color:" + COLORS["PL Best Fit"] + "'>%{customdata[3]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Bear"] + "'>Bear</span> | "
        "<span style='color:" + COLORS["Bear"] + "'>%{customdata[4]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["Support"] + "'>Support</span> | "
        "<span style='color:" + COLORS["Support"] + "'>%{customdata[5]:$,.0f}</span><br>"
        "<span style='color:" + COLORS["BTC"] + "'>BTC</span> | "
        "<span style='color:" + COLORS["BTC"] + "'>%{customdata[6]:$,.0f}</span>"
        "<extra></extra>"
    )
    cd_usd = np.column_stack([
        monyr_full,
        bands_usd["Top"], bands_usd["Frothy"], bands_usd["mid"],
        bands_usd["Bear"], bands_usd["Support"],
        usd_full
    ])
    fig.add_trace(go.Scatter(
        x=x_full, y=bands_usd["mid"],
        mode="markers",
        marker=dict(size=1, opacity=0),
        name="", showlegend=False, legendgroup="USD",
        hovertemplate=hover_usd,
        customdata=cd_usd,
        visible=True
    ))

    # ================= GOLD group (hidden) =================
    for name in order:
        y = bands_gld["mid"] if name == "PL Best Fit" else bands_gld[name]
        sig = _fmt_sigma_for_legend(LEVELS[name])
        legend_name = f"{name} ({sig}σ)"
        fig.add_trace(go.Scatter(
            x=x_full, y=y, mode="lines",
            line=dict(color=COLORS[name], width=2, dash=DASHES[name]),
            name=legend_name, legendgroup="GLD",
            hoverinfo="skip",
            visible=False
        ))

    # BTC (Gold oz/BTC)
    fig.add_trace(go.Scatter(
        x=x_hist, y=gld, mode="lines",
        line=dict(color=COLORS["BTC"], width=2.5),
        name="BTC", legendgroup="GLD",
        hoverinfo="skip",
        visible=False
    ))
    GLD_BTC_IDX = len(fig.data) - 1

    # Slider-driven BTC marker (Gold)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=COLORS["BTC_MARK"], size=8, line=dict(color="#000", width=.5)),
        name=" ", legendgroup="GLD", showlegend=False, hoverinfo="skip", visible=False
    ))
    GLD_MARK_IDX = len(fig.data) - 1

    # Composite hover (Gold)
    hover_gld = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:" + COLORS["Top"] + "'>Top</span> | "
        "<span style='color:" + COLORS["Top"] + "'>%{customdata[1]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Frothy"] + "'>Frothy</span> | "
        "<span style='color:" + COLORS["Frothy"] + "'>%{customdata[2]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["PL Best Fit"] + "'>PL Best Fit</span> | "
        "<span style='color:" + COLORS["PL Best Fit"] + "'>%{customdata[3]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Bear"] + "'>Bear</span> | "
        "<span style='color:" + COLORS["Bear"] + "'>%{customdata[4]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["Support"] + "'>Support</span> | "
        "<span style='color:" + COLORS["Support"] + "'>%{customdata[5]:,.2f} oz/BTC</span><br>"
        "<span style='color:" + COLORS["BTC"] + "'>BTC</span> | "
        "<span style='color:" + COLORS["BTC"] + "'>%{customdata[6]:,.2f} oz/BTC</span>"
        "<extra></extra>"
    )
    cd_gld = np.column_stack([
        monyr_full,
        bands_gld["Top"], bands_gld["Frothy"], bands_gld["mid"],
        bands_gld["Bear"], bands_gld["Support"],
        gld_full
    ])
    fig.add_trace(go.Scatter(
        x=x_full, y=bands_gld["mid"],
        mode="markers",
        marker=dict(size=1, opacity=0),
        name="", showlegend=False, legendgroup="GLD",
        hovertemplate=hover_gld,
        customdata=cd_gld,
        visible=False
    ))

    # ----- axes (year ticks on log-time) -----
    tickvals, ticktext = year_ticks(2012, 2020, 2040)

    fig.update_layout(
        template="plotly_dark",
        hovermode="x",  # only our composite hover appears
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="#263041", zeroline=False
        ),
        yaxis=dict(
            title="USD / BTC", type="log", tickformat="$,d",
            showgrid=True, gridcolor="#263041", zeroline=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(
            bgcolor="rgba(20,24,32,0.85)",  # subtle, translucent
            bordercolor="#3b4455",
            font=dict(color="#e5e7eb"),
            align="left"
        ),
        margin=dict(l=60, r=24, t=18, b=64),
        paper_bgcolor="#0f1116", plot_bgcolor="#151821",
        meta=dict(
            USD_MARK_IDX=USD_MARK_IDX,
            GLD_MARK_IDX=GLD_MARK_IDX
        ),
        showlegend=True  # ensure legend is visible initially
    )

    return fig, full_dates, bands_usd, bands_gld, usd, gld

# --------------------- HTML writer
def write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist,
                     out_path="dist/index.html", page_title="BTC Purchase Indicator",
                     template_path="template.html"):
    from pathlib import Path
    import json, numpy as np, pandas as pd
    import plotly.io as pio

    # ensure output folder exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) serialize plotly figure
    fig_json = pio.to_json(fig, pretty=False)

    # 2) pack arrays payload
    dates_iso = pd.to_datetime(full_dates).strftime("%Y-%m-%d").tolist()
    hist_len  = len(usd_hist)
    hist_iso  = pd.to_datetime(full_dates[:hist_len]).strftime("%Y-%m-%d").tolist()

    payload = {
        "dates": dates_iso,
        "hist": {
            "dates": hist_iso,
            "usd": pd.Series(usd_hist, dtype="float64").round(6).tolist(),
            "gld": pd.Series(gld_hist, dtype="float64").round(6).tolist(),
        },
        "usd": {k: np.asarray(v, dtype="float64").round(6).tolist() for k, v in {
            "Top": bands_usd["Top"], "Frothy": bands_usd["Frothy"],
            "Mid": bands_usd["mid"], "Bear": bands_usd["Bear"], "Support": bands_usd["Support"]
        }.items()},
        "gld": {k: np.asarray(v, dtype="float64").round(6).tolist() for k, v in {
            "Top": bands_gld["Top"], "Frothy": bands_gld["Frothy"],
            "Mid": bands_gld["mid"], "Bear": bands_gld["Bear"], "Support": bands_gld["Support"]
        }.items()}
    }
    arrays_json = json.dumps(payload, separators=(",", ":"))

    # 3) read template and replace placeholders
    tpl_path = Path(template_path)
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template not found: {tpl_path.resolve()}")

    tpl = tpl_path.read_text(encoding="utf-8")
    html = (tpl
            .replace("__TITLE__", page_title)
            .replace("__FIGJSON__", fig_json)
            .replace("__ARRAYS__", arrays_json))

    # 4) write to dist/index.html
    Path(out_path).write_text(html, encoding="utf-8")
    print("[build] wrote", out_path)

# --------------------- main
def main():
    print("[build] loading BTC & Gold …")
    df = load_btc_gold()
    print(f"[build] rows: {len(df):,}  (BTC & Gold)")
    fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist = make_powerlaw_fig(df)
    write_index_html(fig, full_dates, bands_usd, bands_gld, usd_hist, gld_hist)

if __name__ == "__main__":
    main()