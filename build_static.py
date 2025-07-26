# build_static.py
# Static builder for BTC Purchase Indicator (GitHub Pages)
# - Power-law (log-time X to 2040), no anchoring (fit = historical only)
# - DMA (USD or Gold oz/BTC)
# - Legend toggle, compact denomination dropdown, cross-page buttons
# Python 3.11+

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# constants
OUTDIR    = Path("dist")
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")

# fixed sigma bands (z-values) and colors
LEVELS = {
    "Top": 1.75,
    "Frothy": 1.00,
    "PL Best Fit": 0.00,
    "Bear": -0.50,
    "Support": -1.50,
}
COLORS = {
    "Top":       "#16a34a",   # green-600
    "Frothy":    "#86efac",   # green-300
    "PL Best Fit": "white",
    "Bear":      "#fca5a5",   # red-300
    "Support":   "#ef4444",   # red-500
    "BTC":       "gold",
    # DMA palette
    "DMA_50_USD":  "#60a5fa",   # blue-400
    "DMA_200_USD": "#7c3aed",   # violet-600
    "DMA_50_GOLD": "#fde047",   # yellow-300
    "DMA_200_GOLD":"#f59e0b",   # amber-500
}

# Zone → dot color (opposite of band tone as requested)
ZONE_DOT = {
    "SELL THE HOUSE!!": "#22c55e",  # green
    "Buy":              "#86efac",  # light-green
    "DCA":              "#ffffff",  # white
    "Relax":            "#fdba74",  # light orange
    "TO THE MOON":      "#ef4444",  # red
}

# ─────────────────────────────────────────────────────────────
# data loaders

def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def fetch_btc() -> pd.DataFrame:
    # 1) Stooq
    try:
        df = _read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
        df.columns = [c.lower() for c in df.columns]
        date_col  = [c for c in df.columns if "date"  in c][0]
        price_col = [c for c in df.columns if ("close" in c) or ("price" in c)][0]
        df = df.rename(columns={date_col: "Date", price_col: "BTC"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
        out = df.dropna().query("BTC>0").sort_values("Date")
        if len(out) > 1000:
            print(f"[build] BTC from Stooq: {len(out)} rows")
            return out[["Date","BTC"]]
    except Exception as e:
        print("[build] Stooq BTC failed:", e)
    # 2) GitHub dataset
    try:
        raw = _read_csv("https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv")
        raw = raw.rename(columns={"Closing Price (USD)": "BTC"})
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw["BTC"]  = pd.to_numeric(raw["BTC"], errors="coerce")
        out = raw.dropna().sort_values("Date")[["Date","BTC"]]
        print(f"[build] BTC from GitHub dataset: {len(out)} rows")
        return out
    except Exception as e:
        raise RuntimeError(f"Failed to load BTC: {e}")

def fetch_gold() -> pd.DataFrame:
    # 1) Stooq XAUUSD
    try:
        df = _read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")
        df.columns = [c.lower() for c in df.columns]
        date_col  = [c for c in df.columns if "date"  in c][0]
        price_col = [c for c in df.columns if ("close" in c) or ("price" in c)][0]
        df = df.rename(columns={date_col: "Date", price_col: "Gold"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
        out = df.dropna().sort_values("Date")[["Date","Gold"]]
        if len(out) > 1000:
            print(f"[build] Gold from Stooq: {len(out)} rows")
            return out
    except Exception as e:
        print("[build] Stooq Gold failed:", e)
    # 2) LBMA mirror
    try:
        alt = _read_csv("https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv")
        alt["Date"] = pd.to_datetime(alt["Date"], errors="coerce")
        alt = alt.rename(columns={"USD (PM)": "Gold"})
        out = alt[["Date","Gold"]].dropna().sort_values("Date")
        print(f"[build] Gold from LBMA mirror: {len(out)} rows")
        return out
    except Exception as e:
        raise RuntimeError(f"Failed to load Gold: {e}")

def load_merged() -> pd.DataFrame:
    btc  = fetch_btc()
    gold = fetch_gold()
    df = pd.merge(btc, gold, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────
# transforms & model

def days_since_genesis(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    td = pd.to_datetime(dates) - GENESIS
    return (td / pd.Timedelta(days=1)).to_numpy()

def log_days(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    d = np.clip(days_since_genesis(dates), 1.0, None)
    return np.log10(d)

def year_ticks_split(start=2012, split=2020, end=2040):
    yrs = list(range(start, split + 1)) + list(range(split + 2, end + 1, 2))
    vals = log_days([pd.Timestamp(f"{y}-01-01") for y in yrs]).tolist()
    return vals, [str(y) for y in yrs]

def extend_monthly_to_2040(df: pd.DataFrame) -> pd.DataFrame:
    if df["Date"].max() >= PROJ_END:
        return df.copy()
    future = pd.date_range(df["Date"].max() + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    tail = pd.DataFrame({"Date": future})
    out = pd.concat([df, tail], ignore_index=True)
    # For Gold we need values to divide into bands; ffill Gold only
    if "Gold" in out:
        out["Gold"] = out["Gold"].ffill()
    return out

def fit_power(df_btc: pd.DataFrame) -> tuple[float,float,float]:
    """Fit log10(price) ~ log10(days) from historical BTC (no anchoring)."""
    X = log_days(df_btc["Date"])
    y = np.log10(df_btc["BTC"].to_numpy())
    m, b = np.polyfit(X, y, 1)
    sigma = float(np.std(y - (m * X + b)))
    return float(m), float(b), sigma

def add_sigma_bands_over(dates: pd.Series, m: float, b: float, sigma: float) -> pd.DataFrame:
    """Compute PL mid & bands over an arbitrary date index."""
    d = pd.to_datetime(dates)              # ensure pandas datetime
    x = log_days(d)                        # log10(days since genesis)
    mid_log = m * x + b

    out = pd.DataFrame({"Date": d})        # keep as pandas Timestamps
    out["PL Best Fit"] = 10 ** mid_log

    sig = max(0.25, sigma)
    for name, k in LEVELS.items():
        if name == "PL Best Fit":
            continue
        out[name] = 10 ** (mid_log + sig * k)
    return out

def zone_for(price: float, bands: dict[str,float]) -> str:
    if price < bands["Support"]: return "SELL THE HOUSE!!"
    if price < bands["Bear"]:    return "Buy"
    if price < bands["Frothy"]:  return "DCA"
    if price < bands["Top"]:     return "Relax"
    return "TO THE MOON"

# ─────────────────────────────────────────────────────────────
# figures

def fig_powerlaw(df_hist: pd.DataFrame) -> go.Figure:
    # Fit only on history
    m, b, sigma = fit_power(df_hist)

    # Build a timeline to 2040 for bands (price line remains historical)
    base = df_hist[["Date","BTC","Gold"]].copy()
    full = extend_monthly_to_2040(base)

    # Bands in USD on the full timeline
    bands_usd = add_sigma_bands_over(full["Date"], m, b, sigma)
    bands_usd = bands_usd.merge(full[["Date","Gold"]], on="Date", how="left")

    # USD traces
    order = ["Top","Frothy","PL Best Fit","Bear","Support"]
    traces = []
    for nm in order:
        traces.append(go.Scatter(
            x=log_days(bands_usd["Date"]),
            y=bands_usd[nm],
            name=f"{nm} (USD)",
            mode="lines",
            line=dict(color=COLORS[nm], dash="dash", width=2 if nm!="PL Best Fit" else 2),
            legendgroup="USD",
            visible=True
        ))
    # Historical BTC price (USD) — no forward-fill into future
    traces.append(go.Scatter(
        x=log_days(df_hist["Date"]), y=df_hist["BTC"],
        name="BTC (USD)", mode="lines",
        line=dict(color=COLORS["BTC"], width=2.6),
        legendgroup="USD", visible=True
    ))

    # GOLD denomination (oz/BTC = BTC(USD)/Gold(USD/oz))
    bands_gold = bands_usd.copy()
    for nm in order:
        bands_gold[f"{nm} (Gold)"] = bands_gold[nm] / bands_gold["Gold"]
    hist_gold = df_hist.copy()
    hist_gold["BTC_gold"] = hist_gold["BTC"] / hist_gold["Gold"]

    for nm in order:
        traces.append(go.Scatter(
            x=log_days(bands_gold["Date"]),
            y=bands_gold[f"{nm} (Gold)"],
            name=f"{nm} (Gold)",
            mode="lines",
            line=dict(color=COLORS[nm], dash="dash", width=2 if nm!="PL Best Fit" else 2),
            legendgroup="Gold", visible=False
        ))
    traces.append(go.Scatter(
        x=log_days(hist_gold["Date"]), y=hist_gold["BTC_gold"],
        name="BTC (Gold)", mode="lines",
        line=dict(color=COLORS["BTC"], width=2.6, dash="solid"),
        legendgroup="Gold", visible=False
    ))

    tickvals, ticktext = year_ticks_split(2012, 2020, 2040)

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=70, r=24, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridwidth=0.5
        ),
        yaxis=dict(
            type="log", title="USD / BTC", tickformat="$,d",
            showgrid=True, gridwidth=0.5
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def fig_dma(df_hist: pd.DataFrame) -> go.Figure:
    ser = df_hist.copy()
    ser = ser[ser["Date"] >= pd.Timestamp("2012-04-01")].reset_index(drop=True)

    # USD DMA
    ser["DMA50_USD"]  = ser["BTC"].rolling(50).mean()
    ser["DMA200_USD"] = ser["BTC"].rolling(200).mean()

    # Gold oz/BTC DMA
    ser["BTC_gold"] = ser["BTC"] / ser["Gold"]
    ser["DMA50_GOLD"]  = ser["BTC_gold"].rolling(50).mean()
    ser["DMA200_GOLD"] = ser["BTC_gold"].rolling(200).mean()

    X = log_days(ser["Date"])
    tickvals, ticktext = year_ticks_split(2012, 2020, 2040)

    traces = [
        # USD group (visible by default)
        go.Scatter(x=X, y=ser["DMA200_USD"], name="200-DMA (USD)",
                   line=dict(color=COLORS["DMA_200_USD"], width=2), legendgroup="USD", visible=True),
        go.Scatter(x=X, y=ser["DMA50_USD"],  name="50-DMA (USD)",
                   line=dict(color=COLORS["DMA_50_USD"], width=2), legendgroup="USD", visible=True),
        go.Scatter(x=X, y=ser["BTC"],        name="BTC (USD)",
                   line=dict(color=COLORS["BTC"], width=2.6), legendgroup="USD", visible=True),

        # Gold group (start hidden)
        go.Scatter(x=X, y=ser["DMA200_GOLD"], name="200-DMA (Gold)",
                   line=dict(color=COLORS["DMA_200_GOLD"], width=2), legendgroup="Gold", visible=False),
        go.Scatter(x=X, y=ser["DMA50_GOLD"],  name="50-DMA (Gold)",
                   line=dict(color=COLORS["DMA_50_GOLD"], width=2), legendgroup="Gold", visible=False),
        go.Scatter(x=X, y=ser["BTC_gold"],    name="BTC (Gold)",
                   line=dict(color=COLORS["BTC"], width=2.6, dash="solid"), legendgroup="Gold", visible=False),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=70, r=24, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridwidth=0.5
        ),
        yaxis=dict(
            type="log", title="USD / BTC", tickformat="$,d",
            showgrid=True, gridwidth=0.5
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ─────────────────────────────────────────────────────────────
# HTML wrappers

CSS = """
<style>
:root { color-scheme: dark; }
body { margin:0; background:#0f1116; color:#eceff4;
       font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.container { max-width: 1024px; margin: 22px auto 64px; padding: 0 14px; }
h1 { font-size: clamp(28px,4.2vw,44px); line-height:1.05; margin: 0 0 8px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:12px 0 10px; }
.btn { background:#161a22; border:1px solid #2a2f3a; color:#dbe2ec; border-radius:12px; padding:.45rem .65rem; }
.btn:hover { background:#1b2030; cursor:pointer; }
.compact-label { color:#a8b3c7; font-weight:600; }
.select-compact { background:#161a22; color:#dbe2ec; border:1px solid #2a2f3a;
                  border-radius:10px; padding:.15rem .35rem; font-size:14px; height:auto; }
.badge { display:inline-flex; align-items:center; gap:8px; background:#161a22; border:1px solid #2a2f3a;
         padding:.45rem .65rem; border-radius:18px; }
.badge .dot { width:10px; height:10px; border-radius:50%; background:#cfd3dc; display:inline-block; }
.card { background:#0f1116; border:1px solid #262b36; border-radius:16px; padding:10px; }
a { color:#7fb0ff; text-decoration:none; }
a:hover { text-decoration:underline; }
</style>
"""

# JS helpers (double braces so f-strings don’t break)
JS = """
<script>
function toggleLegend(figid) {{
  const gd = document.getElementById(figid);
  const showing = gd.layout.showlegend === true;
  Plotly.relayout(gd, {{showlegend: !showing}});
}}
function setDenom(figid, denom) {{
  const gd = document.getElementById(figid);
  const groups = gd.data.map(tr => tr.legendgroup);
  const visUSD  = (denom === 'USD');
  const visGold = !visUSD;
  const vis = [];
  for (let i=0;i<groups.length;i++) {{
    const g = groups[i];
    vis.push(g === 'USD' ? visUSD : (g === 'Gold' ? visGold : true));
  }}
  Plotly.update(gd, {{visible: vis}}, {{
    yaxis: {{
      title: (denom === 'USD') ? 'USD / BTC' : 'Gold oz / BTC',
      tickformat: (denom === 'USD') ? '$,d' : ',d'
    }}
  }});
}}
</script>
"""

def write_powerlaw_page(fig: go.Figure, zone: str):
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Purchase Indicator · Power-law</title>
<link rel="preconnect" href="https://cdn.plot.ly" />
{CSS}
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
{JS}
<body>
<div class="container">
  <h1>BTC Purchase Indicator</h1>
  <div class="badge">
    <span class="dot" style="background:{ZONE_DOT.get(zone,'#cfd3dc')}"></span>
    <span>Current zone:&nbsp;</span><strong>{zone}</strong>
  </div>
  <div class="controls">
    <a class="btn" href="./dma.html">Open DMA chart →</a>
    <span class="compact-label">Denomination</span>
    <select id="denom" class="select-compact" onchange="setDenom('pl-fig', this.value)">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
    <button class="btn" onclick="toggleLegend('pl-fig')">Legend</button>
  </div>
  <div id="pl-fig" class="card"></div>
</div>
<script>
  const FIG = {fig.to_json()};
  Plotly.newPlot('pl-fig', FIG.data, FIG.layout, {{displayModeBar:true,responsive:true}});
  Plotly.relayout('pl-fig', {{showlegend:false}});
  setDenom('pl-fig','USD');
</script>
</body>
</html>
"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "index.html").write_text(html, encoding="utf-8")
    print("[build] wrote dist/index.html")

def write_dma_page(fig: go.Figure):
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Purchase Indicator · DMA</title>
<link rel="preconnect" href="https://cdn.plot.ly" />
{CSS}
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
{JS}
<body>
<div class="container">
  <h1>BTC USD & Gold DMA</h1>
  <div class="controls">
    <a class="btn" href="./index.html">← Open Power-law chart</a>
    <span class="compact-label">Denomination</span>
    <select id="denom" class="select-compact" onchange="setDenom('dma-fig', this.value)">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
    <button class="btn" onclick="toggleLegend('dma-fig')">Legend</button>
  </div>
  <div id="dma-fig" class="card"></div>
</div>
<script>
  const FIG = {fig.to_json()};
  Plotly.newPlot('dma-fig', FIG.data, FIG.layout, {{displayModeBar:true,responsive:true}});
  Plotly.relayout('dma-fig', {{showlegend:false}});
  setDenom('dma-fig','USD');
</script>
</body>
</html>
"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "dma.html").write_text(html, encoding="utf-8")
    print("[build] wrote dist/dma.html")

# ─────────────────────────────────────────────────────────────
# main

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_merged()
    # Zone from latest real price vs today’s bands (USD) — compute at last row
    m, b, sigma = fit_power(df)
    last_date = df["Date"].iloc[-1]
    bands_today = add_sigma_bands_over(pd.Index([last_date]), m, b, sigma).iloc[0]
    last_price  = float(df["BTC"].iloc[-1])
    zone = zone_for(last_price, {
        "Support": float(bands_today["Support"]),
        "Bear":    float(bands_today["Bear"]),
        "Frothy":  float(bands_today["Frothy"]),
        "Top":     float(bands_today["Top"]),
    })

    pl_fig  = fig_powerlaw(df)
    dma_fig = fig_dma(df)

    write_powerlaw_page(pl_fig, zone)
    write_dma_page(dma_fig)

if __name__ == "__main__":
    main()