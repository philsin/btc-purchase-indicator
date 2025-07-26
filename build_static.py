# build_static.py
# Static builder for BTC Purchase Indicator (GitHub Pages)
# - Power-law (log-time X, out to 2040)
# - DMA (USD or Gold oz/BTC)
# - Legend toggle, denomination dropdown, back buttons
# Python 3.11+

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ────────────────────────────── constants
OUTDIR    = Path("dist")
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
ANCHOR_DT = pd.Timestamp("2030-01-01")
ANCHOR_P  = 491_776.0  # USD (target mid-line at 2030-01-01)

# fixed sigma bands (in stdev units)
LEVELS = {
    "Top": 1.75,
    "Frothy": 1.00,
    "PL Best Fit": 0.00,
    "Bear": -0.50,
    "Support": -1.50,
}
COLORS = {
    "Top": "rgba(0,191,99,1)",
    "Frothy": "rgba(144,238,144,1)",
    "PL Best Fit": "white",
    "Bear": "rgba(255,140,140,1)",
    "Support": "rgba(255,59,59,1)",
    "BTC": "gold",
    # DMA palette
    "DMA_50_USD": "mediumseagreen",
    "DMA_200_USD": "forestgreen",
    "DMA_50_GOLD": "khaki",
    "DMA_200_GOLD": "goldenrod",
}

# ────────────────────────────── helpers

def days_since_genesis(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    """Return float days since GENESIS for pandas Series/Index/list."""
    td = pd.to_datetime(dates) - GENESIS
    return (td / pd.Timedelta(days=1)).to_numpy()

def log_days(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    d = days_since_genesis(dates)
    # avoid log10(0)
    d = np.clip(d, 1e-6, None)
    return np.log10(d)

def year_ticks(start: int, stop: int) -> Tuple[list[float], list[str]]:
    """Ticks on log-time axis at Jan-01 for each year. Returns (vals, labels)."""
    years = list(range(start, stop + 1))
    dts = [pd.Timestamp(f"{y}-01-01") for y in years]
    vals = log_days(dts).tolist()
    txt  = [str(y) for y in years]
    return vals, txt

def fetch_btc() -> pd.DataFrame:
    # 1) Stooq
    try:
        df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        date_col = [c for c in cols if "date" in c][0]
        price_col = [c for c in cols if ("close" in c) or ("price" in c)][0]
        df = df.rename(columns={date_col: "Date", price_col: "BTC"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna().query("BTC>0").sort_values("Date")
        if len(df) > 1000:
            print(f"[build] BTC from Stooq: {len(df)} rows")
            return df[["Date","BTC"]]
    except Exception as e:
        print("[build] Stooq BTC failed:", e)

    # 2) GitHub dataset fallback
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    rdf = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "BTC"})
    rdf["Date"] = pd.to_datetime(rdf["Date"])
    rdf = rdf[["Date", "BTC"]].sort_values("Date")
    print(f"[build] BTC from GitHub dataset: {len(rdf)} rows")
    return rdf

def fetch_gold() -> pd.DataFrame:
    # 1) Stooq XAUUSD
    try:
        df = pd.read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        date_col = [c for c in cols if "date" in c][0]
        price_col = [c for c in cols if ("close" in c) or ("price" in c)][0]
        df = df.rename(columns={date_col: "Date", price_col: "Gold"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna().sort_values("Date")
        if len(df) > 1000:
            print(f"[build] Gold from Stooq: {len(df)} rows")
            return df[["Date","Gold"]]
    except Exception as e:
        print("[build] Stooq Gold failed:", e)

    # 2) LBMA CSV fallback
    raw = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    rdf = pd.read_csv(raw)
    rdf["Date"] = pd.to_datetime(rdf["Date"])
    rdf = rdf.rename(columns={"USD (PM)": "Gold"})
    rdf = rdf[["Date","Gold"]].dropna().sort_values("Date")
    print(f"[build] Gold from LBMA mirror: {len(rdf)} rows")
    return rdf

def fit_power(btc_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return slope, intercept, sigma for log10(price) vs log10(days since genesis)."""
    X = log_days(btc_df["Date"])
    y = np.log10(btc_df["BTC"].to_numpy())
    m, b = np.polyfit(X, y, 1)
    mid = m * X + b
    sigma = float(np.std(y - mid))
    # Anchor at 2030-01-01 = ANCHOR_P
    x_a = log_days([ANCHOR_DT])[0]
    b = np.log10(ANCHOR_P) - m * x_a
    return float(m), float(b), sigma

def add_sigma_bands(df: pd.DataFrame, m: float, b: float, sigma: float) -> pd.DataFrame:
    x = log_days(df["Date"])
    mid_log = m * x + b
    out = df.copy()
    out["PL Best Fit"] = 10 ** mid_log
    for name, k in LEVELS.items():
        if name == "PL Best Fit":
            continue
        out[name] = 10 ** (mid_log + sigma * k)
    return out

def make_badge(zone: str) -> str:
    return f"""
    <div class="badge"><span class="dot"></span><span>Current zone:&nbsp;</span><strong>{zone}</strong></div>
    """

def zone_for(p: float, bands: dict[str,float]) -> str:
    if p < bands["Support"]:
        return "SELL THE HOUSE!!"
    if p < bands["Bear"]:
        return "Buy"
    if p < bands["Frothy"]:
        return "DCA"
    if p < bands["Top"]:
        return "Relax"
    return "Frothy"

def ensure_dir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── figures

def fig_powerlaw(df_all: pd.DataFrame, denom_default="USD") -> go.Figure:
    """
    Build a dual-denomination power-law fig. We plot in log-time (x is log10 days),
    but display year labels as ticktext.
    """
    # Compute bands (USD)
    m, b, sigma = fit_power(df_all)
    bands_usd = add_sigma_bands(df_all, m, b, max(sigma, 0.25))

    # Gold oz/BTC
    df_gold = bands_usd.copy()
    df_gold["BTC_gold"] = df_gold["BTC"] / df_gold["Gold"]
    for nm in LEVELS:
        df_gold[f"{nm} (Gold)"] = df_gold[nm] / df_gold["Gold"]

    # Build traces (two legendgroups, toggle via JS)
    traces = []

    # Order: Top, Frothy, PL, Bear, Support, BTC
    order = ["Top", "Frothy", "PL Best Fit", "Bear", "Support"]

    # USD group (visible depending on default)
    vis_usd = True if denom_default == "USD" else False
    for nm in order:
        traces.append(go.Scatter(
            x=log_days(bands_usd["Date"]),
            y=bands_usd[nm],
            mode="lines",
            name=f"{nm} (USD)" if nm != "PL Best Fit" else "PL Best Fit (USD)",
            line=dict(color=COLORS[nm], width=2, dash="dash" if nm != "PL Best Fit" else "dash"),
            legendgroup="USD",
            visible=vis_usd
        ))
    traces.append(go.Scatter(
        x=log_days(bands_usd["Date"]), y=bands_usd["BTC"],
        mode="lines", name="BTC (USD)", line=dict(color=COLORS["BTC"], width=2.5),
        legendgroup="USD", visible=vis_usd
    ))

    # GOLD group (visible if denom_default == "Gold")
    vis_gold = not vis_usd
    for nm in order:
        traces.append(go.Scatter(
            x=log_days(df_gold["Date"]),
            y=df_gold[f"{nm} (Gold)"],
            mode="lines",
            name=f"{nm} (Gold)",
            line=dict(color=COLORS[nm], width=2, dash="dash" if nm != "PL Best Fit" else "dash"),
            legendgroup="Gold",
            visible=vis_gold
        ))
    traces.append(go.Scatter(
        x=log_days(df_gold["Date"]), y=df_gold["BTC_gold"],
        mode="lines", name="BTC (Gold)", line=dict(color=COLORS["BTC"], width=2.5, dash="solid"),
        legendgroup="Gold", visible=vis_gold
    ))

    # Axis ticks for years 2012..2040 (dense to sparse)
    tickvals, ticktext = year_ticks(2012, 2040)

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,  # hidden initially; button will toggle
        margin=dict(l=70, r=20, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridwidth=0.5
        ),
        yaxis=dict(
            type="log",
            title="USD / BTC" if denom_default == "USD" else "Gold oz / BTC",
            tickformat="$,d" if denom_default == "USD" else ",d",
            showgrid=True, gridwidth=0.5
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    # we’ll pass data arrays for JS (zone slider etc.)
    return fig

def fig_dma(df_all: pd.DataFrame, denom_default="USD") -> go.Figure:
    ser = df_all.copy()
    # USD DMA
    ser["DMA50_USD"] = ser["BTC"].rolling(50).mean()
    ser["DMA200_USD"] = ser["BTC"].rolling(200).mean()

    # Gold oz/BTC DMA
    ser["BTC_gold"] = ser["BTC"] / ser["Gold"]
    ser["DMA50_GOLD"] = ser["BTC_gold"].rolling(50).mean()
    ser["DMA200_GOLD"] = ser["BTC_gold"].rolling(200).mean()

    # x on log-time
    X = log_days(ser["Date"])

    traces = []
    vis_usd = True if denom_default == "USD" else False
    vis_gold = not vis_usd

    # USD
    traces += [
        go.Scatter(x=X, y=ser["DMA200_USD"], name="200-DMA (USD)",
                   line=dict(color=COLORS["DMA_200_USD"], width=2), legendgroup="USD", visible=vis_usd),
        go.Scatter(x=X, y=ser["DMA50_USD"], name="50-DMA (USD)",
                   line=dict(color=COLORS["DMA_50_USD"], width=2), legendgroup="USD", visible=vis_usd),
        go.Scatter(x=X, y=ser["BTC"], name="BTC (USD)",
                   line=dict(color=COLORS["BTC"], width=2.5), legendgroup="USD", visible=vis_usd),
    ]

    # GOLD
    traces += [
        go.Scatter(x=X, y=ser["DMA200_GOLD"], name="200-DMA (Gold)",
                   line=dict(color=COLORS["DMA_200_GOLD"], width=2), legendgroup="Gold", visible=vis_gold),
        go.Scatter(x=X, y=ser["DMA50_GOLD"], name="50-DMA (Gold)",
                   line=dict(color=COLORS["DMA_50_GOLD"], width=2), legendgroup="Gold", visible=vis_gold),
        go.Scatter(x=X, y=ser["BTC_gold"], name="BTC (Gold)",
                   line=dict(color=COLORS["BTC"], width=2.5, dash="solid"), legendgroup="Gold", visible=vis_gold),
    ]

    tickvals, ticktext = year_ticks(2012, 2040)

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=70, r=20, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridwidth=0.5
        ),
        yaxis=dict(
            type="log",
            title="USD / BTC" if denom_default == "USD" else "Gold oz / BTC",
            tickformat="$,d" if denom_default == "USD" else ",d",
            showgrid=True, gridwidth=0.5
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ────────────────────────────── HTML wrappers

CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, Helvetica, Arial, sans-serif;
       margin:0; background:#0f1116; color:#e9e9ea; }
.container { max-width: 980px; margin: 22px auto 60px; padding: 0 14px; }
h1 { font-size: 40px; line-height: 1.05; margin: 0 0 10px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:16px 0; }
.btn { background:#1b1e27; border:1px solid #2b2f3a; color:#cfd3dc; border-radius:12px; padding:10px 14px; cursor:pointer; }
.btn:hover { background:#262a36; }
.badge { display:inline-flex; align-items:center; gap:8px; background:#1b1e27; border:1px solid #2b2f3a;
         padding:10px 14px; border-radius:18px; }
.badge .dot { width:10px; height:10px; background:#cfd3dc; border-radius:50%; display:inline-block; }
.subtle { color:#a9b0bd; }
.footer { color:#768199; font-size:12px; margin-top:10px; }
select { background:#1b1e27; color:#e9e9ea; border:1px solid #2b2f3a; border-radius:12px; padding:8px 10px; }
.sliderrow { margin:10px 0 6px; color:#cfd3dc; }
.card { background:#0f1116; border:1px solid #2b2f3a; border-radius:18px; padding:12px; }
a { color:#7fb0ff; text-decoration:none; }
a:hover { text-decoration:underline; }
</style>
"""

# small JS helpers — remember to escape braces in f-strings (use doubled braces)
JS = """
<script>
function toggleLegend(figid) {{
  const gd = document.getElementById(figid);
  const showing = gd.layout.showlegend === true;
  Plotly.relayout(gd, {{showlegend: !showing}});
}}
function setDenom(figid, denom) {{
  // Toggle visibility by legendgroup: 'USD' vs 'Gold'
  const gd = document.getElementById(figid);
  const visUSD  = (denom === 'USD');
  const visGold = !visUSD;
  const upd = {{visible: []}};
  const groups = gd.data.map(tr => tr.legendgroup);
  for (let i=0; i<groups.length; i++) {{
    const g = groups[i];
    upd.visible.push( (g === 'USD') ? visUSD : (g === 'Gold' ? visGold : true) );
  }}
  Plotly.update(gd, upd, {{
    yaxis: {{
      title: (denom === 'USD') ? 'USD / BTC' : 'Gold oz / BTC',
      tickformat: (denom === 'USD') ? '$,d' : ',d'
    }}
  }});
}}
</script>
"""

def write_html(filename: str, title: str, fig: go.Figure, extra_controls: str = "", badge_html: str = ""):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
{CSS}
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
{JS}
<body>
<div class="container">
  <h1>BTC Purchase Indicator</h1>
  {badge_html}
  <div class="controls">
    <button class="btn" onclick="history.back()">&larr; Back</button>
    <span class="subtle">Denomination:&nbsp;</span>
    <select id="denom" onchange="setDenom('plfig', this.value)">
      <option value="USD">USD</option>
      <option value="Gold">Gold</option>
    </select>
    <button class="btn" onclick="toggleLegend('plfig')">Legend</button>
    {extra_controls}
  </div>
  <div id="plfig" class="card"></div>
  <div class="footer">philsin.github.io</div>
</div>
<script>
  const FIGDATA = {fig.to_json()};
  Plotly.newPlot('plfig', FIGDATA.data, FIGDATA.layout, {{displayModeBar: true, responsive: true}});
  // Hide legend by default (button toggles)
  Plotly.relayout('plfig', {{showlegend:false}});
  // Set default denomination
  setDenom('plfig', 'USD');
</script>
</body>
</html>
"""
    (OUTDIR / filename).write_text(html, encoding="utf-8")
    print(f"[build] wrote {OUTDIR/filename}")

# ────────────────────────────── main build

def main():
    ensure_dir()
    btc = fetch_btc()
    gold = fetch_gold()
    df = pd.merge(btc, gold, on="Date", how="inner").sort_values("Date")
    df = df[df["Date"] >= pd.Timestamp("2012-01-01")].reset_index(drop=True)

    # Power-law
    pl_fig = fig_powerlaw(df, denom_default="USD")
    # Zone badge uses most-recent USD bands
    m, b, sigma = fit_power(df)
    usd_bands = add_sigma_bands(df, m, b, max(sigma, 0.25))
    last = usd_bands.dropna().iloc[-1]
    zone = zone_for(float(last["BTC"]), {
        "Support": float(last["Support"]),
        "Bear": float(last["Bear"]),
        "Frothy": float(last["Frothy"]),
        "Top": float(last["Top"]),
    })
    badge = make_badge(zone)
    write_html("index.html", "BTC Purchase Indicator · Power-law", pl_fig, badge_html=badge)

    # DMA page
    dma_fig = fig_dma(df, denom_default="USD")
    write_html("dma.html", "BTC USD & Gold DMA", dma_fig,
               extra_controls='<a class="btn" href="index.html">Power-law chart &rarr;</a>')

if __name__ == "__main__":
    main()