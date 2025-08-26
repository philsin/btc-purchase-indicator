# build_static.py  — minimal static page with quantile power-law bands
# Writes build/index.html (no external template file required)

from pathlib import Path
import datetime as dt
import io
import json
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import statsmodels.api as sm

# ─────────────────────────────────────────────────────────────
# Constants
GENESIS = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
GRID_D = "M24"  # vertical grid every 2 years

# ─────────────────────────────────────────────────────────────
# Utilities

def to_days_since_genesis(dates_like) -> np.ndarray:
    """
    Robustly compute whole days since GENESIS for Series/Index/array.
    Avoid .dt on TimedeltaIndex; use timedelta division instead.
    """
    delta = pd.to_datetime(dates_like) - GENESIS
    try:
        # Works for Series
        days = delta.dt.days.to_numpy()
    except Exception:
        # Works for DatetimeIndex / TimedeltaIndex / ndarray
        days = (delta / np.timedelta64(1, "D")).astype(int)
    # Clamp to >= 1 to avoid log10(0)
    return np.maximum(days, 1)

def fmt_utc_now():
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ─────────────────────────────────────────────────────────────
# Data loaders

def load_btc_stooq() -> pd.DataFrame:
    """
    Load daily BTC/USD from Stooq. Columns: Date, Price
    """
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    # Find date/close columns robustly
    date_col = next(c for c in df.columns if "date" in c)
    close_col = next(c for c in df.columns if ("close" in c) or ("price" in c))
    out = df.rename(columns={date_col: "Date", close_col: "Price"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Price"] = pd.to_numeric(
        out["Price"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["Date", "Price"])
    out = out.query("Price > 0").sort_values("Date")
    return out[["Date", "Price"]]

def load_btc_github_mirror() -> pd.DataFrame:
    """
    Fallback: simple open dataset (may lag, but fine for static build)
    """
    url = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df = pd.read_csv(url)
    # Expect columns: Date, Closing Price (USD)
    if "Closing Price (USD)" in df.columns:
        df = df.rename(columns={"Closing Price (USD)": "Price"})
    elif "Close" in df.columns:
        df = df.rename(columns={"Close": "Price"})
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Price column missing in fallback dataset; got: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).query("Price > 0").sort_values("Date")
    return df[["Date", "Price"]]

def load_history() -> pd.DataFrame:
    """
    Try Stooq first, then fallback to GH mirror.
    """
    try:
        df = load_btc_stooq()
        if len(df) >= 500:  # sanity floor
            return df
    except Exception:
        pass
    return load_btc_github_mirror()

# ─────────────────────────────────────────────────────────────
# Quantile power-law fit

def fit_power_quantile_params(df: pd.DataFrame, price_col: str = "Price",
                              quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    Fit log10(price) ~ log10(days since genesis) using Quantile Regression
    for each requested quantile q. Return dict {q: (slope, intercept)}.
    """
    days = to_days_since_genesis(df["Date"])
    X = np.log10(days).astype(float)
    y = np.log10(df[price_col].to_numpy()).astype(float)

    X_const = sm.add_constant(X)  # [1, log10(days)]

    params = {}
    for q in quantiles:
        model = sm.QuantReg(y, X_const)
        res = model.fit(q=q)
        intercept, slope = res.params  # order: const, x
        params[q] = (slope, intercept)
    return params

def eval_quantile_lines(params: dict, dates: pd.Series) -> pd.DataFrame:
    """
    Evaluate y=10**(slope*log10(days)+intercept) for each quantile on the given dates.
    Returns a DataFrame with columns Q10..Q90.
    """
    days = to_days_since_genesis(dates)
    Xf = np.log10(days).astype(float)

    out = {}
    for q, (slope, intercept) in params.items():
        y_log = slope * Xf + intercept
        out[f"Q{int(q*100)}"] = np.power(10.0, y_log)
    return pd.DataFrame(out, index=pd.RangeIndex(len(dates)))

# ─────────────────────────────────────────────────────────────
# Build chart

def make_powerlaw_fig(hist: pd.DataFrame, full: pd.DataFrame, bands: pd.DataFrame) -> go.Figure:
    """
    Plot:
      - Quantile bands: Q10 (red) → Q90 (green)
      - BTC price (gold)
    """
    # Colors smooth red → yellow → green
    band_colors = {
        "Q10": "#ef4444",   # red
        "Q30": "#f59e0b",   # amber
        "Q50": "#fde047",   # yellow
        "Q70": "#86efac",   # light-green
        "Q90": "#22c55e",   # green
    }

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13),
        xaxis=dict(type="date", title="Year", dtick=GRID_D, showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="USD / BTC", tickformat="$,d", showgrid=True, gridwidth=0.5),
        plot_bgcolor="#0f1116", paper_bgcolor="#0f1116",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    ))

    # Bands
    for col in ["Q10", "Q30", "Q50", "Q70", "Q90"]:
        fig.add_trace(go.Scatter(
            x=full["Date"], y=bands[col],
            name=f"{col} band",
            line=dict(color=band_colors[col], dash="dash"),
            opacity=0.95
        ))

    # BTC price (historical)
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Price"],
        name="BTC",
        line=dict(color="gold", width=2.5)
    ))

    return fig

# ─────────────────────────────────────────────────────────────
# Zone label (based on latest historical day)

def compute_zone_label(hist: pd.DataFrame, full: pd.DataFrame, bands: pd.DataFrame) -> str:
    """
    Compare latest historical price to bands at the same date.
    """
    last_date = hist["Date"].iloc[-1]
    price = hist["Price"].iloc[-1]

    # Find index in full closest to last_date
    ix = full["Date"].searchsorted(last_date)
    ix = max(0, min(ix, len(full) - 1))

    q10 = float(bands.iloc[ix]["Q10"])
    q30 = float(bands.iloc[ix]["Q30"])
    q70 = float(bands.iloc[ix]["Q70"])
    q90 = float(bands.iloc[ix]["Q90"])

    if price < q10:
        return "SELL THE HOUSE!!"
    elif price < q30:
        return "Buy"
    elif price < q70:
        return "DCA"
    elif price < q90:
        return "Relax"
    else:
        return "Frothy"

# ─────────────────────────────────────────────────────────────
# HTML writer

def write_html(fig: go.Figure, zone: str, out_path: Path):
    """
    Write a minimal, styled HTML containing a zone badge and the Plotly chart.
    """
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Simple badge color
    zone_colors = {
        "SELL THE HOUSE!!": "#ffffff",
        "Buy": "#f59e0b",
        "DCA": "#ffffff",
        "Relax": "#22c55e",
        "Frothy": "#ef4444",
    }
    dot = zone_colors.get(zone, "#e5e7eb")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC Purchase Indicator · Quantile Bands</title>
<style>
  :root {{ color-scheme: dark; --bg:#0f1116; --fg:#e7e9ee; --card:#151821; --muted:#8e95a5; }}
  html,body {{ margin:0; background:var(--bg); color:var(--fg);
               font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width: 1100px; margin: 26px auto; padding: 0 16px; }}
  h1 {{ font-size: clamp(28px, 4.5vw, 42px); margin: 0 0 8px 0; }}
  .chip {{ display:inline-flex; align-items:center; gap:10px; padding:8px 12px;
           border-radius: 999px; background: var(--card); border: 1px solid #263041; }}
  .dot {{ width:12px; height:12px; border-radius:50%; background: {dot}; }}
  .meta {{ color: var(--muted); font-size: 0.92rem; margin-top: 6px; }}
  #fig {{ height: 74vh; min-height: 430px; margin-top: 14px; background: var(--card); border-radius: 14px; padding: 8px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>
    <div class="chip"><span class="dot"></span><b>Price Zone:</b> <span>{zone}</span></div>
    <div class="meta">Updated: {fmt_utc_now()}</div>
    <div id="fig">{fig_html}</div>
  </div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[build] wrote {out_path}")

# ─────────────────────────────────────────────────────────────
# Main

def main():
    # 1) load history
    hist = load_history()

    # 2) concatenate a monthly future to 2040 for band visualization
    future = pd.date_range(hist["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                           PROJ_END, freq="MS")
    full = pd.concat([hist[["Date"]], pd.DataFrame({"Date": future})], ignore_index=True)

    # 3) quantile params from historical prices only
    params = fit_power_quantile_params(hist, price_col="Price",
                                       quantiles=(0.1, 0.3, 0.5, 0.7, 0.9))

    # 4) evaluate bands on full timeline
    bands = eval_quantile_lines(params, full["Date"])

    # 5) make figure & zone label
    fig = make_powerlaw_fig(hist, full, bands)
    zone = compute_zone_label(hist, full, bands)

    # 6) write HTML
    out = Path("build/index.html")
    write_html(fig, zone, out)

if __name__ == "__main__":
    main()
