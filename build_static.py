# build_static.py — creates dist/index.html and dist/dma.html
# Run locally:  python build_static.py
# In Actions:   the workflow calls this script

import os, io, math, textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio

# -------------------- constants --------------------
UA = {"User-Agent": "btc-pl-pages/1.0"}
GENESIS = pd.Timestamp("2009-01-03")
PROJ_END = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")
ANCHOR_DATE = pd.Timestamp("2030-01-01")
ANCHOR_USD = 491_776.0                      # power-law mid @ 2030-01-01 (USD/BTC)
FD_SUPPLY = 21_000_000

# power-law bands (z-values)
BANDS = [
    ("Top",        +1.75,  "rgba( 30,180, 60,1)"),
    ("Frothy",     +1.00,  "rgba(120,220,120,1)"),
    ("PL Best Fit", 0.00,  "white"),
    ("Bear",       -0.50,  "rgba(255,120,120,1)"),
    ("Support",    -1.50,  "rgba(220, 60, 60,1)"),
]
# draw order for power-law chart legend (Top..BTC)
PL_ORDER = ["Top", "Frothy", "PL Best Fit", "Bear", "Support", "BTC"]

# -------------------- helpers ----------------------
def read_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def load_btc() -> pd.DataFrame:
    # 1) Stooq
    try:
        df = read_csv_url("https://stooq.com/q/d/l/?s=btcusd&i=d")
        cols = {c.lower(): c for c in df.columns}
        # normalize
        df.columns = [c.lower() for c in df.columns]
        dcol = [c for c in df.columns if "date" in c][0]
        pcol = [c for c in df.columns if ("close" in c or "price" in c)][0]
        out = df.rename(columns={dcol: "Date", pcol: "BTC"})
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["BTC"] = pd.to_numeric(
            out["BTC"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        out = out.dropna().query("BTC>0").sort_values("Date").reset_index(drop=True)
        if len(out) > 1000:
            return out
    except Exception:
        pass
    # 2) GitHub dataset mirror
    raw = read_csv_url(
        "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    ).rename(columns={"Date": "Date", "Closing Price (USD)": "BTC"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["BTC"] = pd.to_numeric(raw["BTC"], errors="coerce")
    return raw[["Date", "BTC"]].dropna().reset_index(drop=True)

def load_gold() -> pd.DataFrame:
    # 1) Stooq XAUUSD
    try:
        df = read_csv_url("https://stooq.com/q/d/l/?s=xauusd&i=d")
        df.columns = [c.lower() for c in df.columns]
        dcol = [c for c in df.columns if "date" in c][0]
        pcol = [c for c in df.columns if ("close" in c or "price" in c)][0]
        out = df.rename(columns={dcol: "Date", pcol: "Gold"})
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Gold"] = pd.to_numeric(
            out["Gold"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        out = out.dropna().sort_values("Date").reset_index(drop=True)
        if len(out) > 1000:
            return out
    except Exception:
        pass
    # 2) LBMA CSV mirror
    alt = read_csv_url(
        "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    )
    alt["Date"] = pd.to_datetime(alt["Date"])
    # choose PM price when available
    g = alt[["Date", "USD (PM)"]].rename(columns={"USD (PM)": "Gold"}).dropna()
    g["Gold"] = pd.to_numeric(g["Gold"], errors="coerce")
    return g.dropna().reset_index(drop=True)

def merge_prices() -> pd.DataFrame:
    b = load_btc()
    g = load_gold()
    df = pd.merge(b, g, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    # Construct Gold oz / BTC (oz per bitcoin) and USD / BTC (redundant naming clarity)
    df["USD_per_BTC"] = df["BTC"]
    df["OZ_per_BTC"] = df["BTC"] / df["Gold"]  # oz/BTC
    return df

def log_days(dates: pd.Series) -> np.ndarray:
    """log10(days since genesis), safely ignoring nonpositive days."""
    days = (pd.to_datetime(dates) - GENESIS).dt.days.to_numpy()
    days = np.where(days < 1, 1, days)
    return np.log10(days)

def power_fit(df: pd.DataFrame, price_col: str) -> tuple[float, float, float]:
    """Return slope, intercept, sigma for log10(days) -> log10(price_col)."""
    x = log_days(df["Date"])
    y = np.log10(df[price_col].to_numpy())
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    slope, intercept = np.polyfit(x, y, 1)
    sigma = float(np.std(y - (slope * x + intercept)))
    return slope, intercept, sigma

def anchor_intercept(slope: float, anchor_date: pd.Timestamp, anchor_value: float) -> float:
    return np.log10(anchor_value) - slope * np.log10(max((anchor_date - GENESIS).days, 1))

def band_series(dates: pd.Series, slope: float, intercept: float, sigma: float, z: float) -> np.ndarray:
    x = log_days(dates)
    mid = slope * x + intercept
    return 10 ** (mid + sigma * z)

def year_ticks(start_year=2012, turn_year=2020, end_year=2040):
    """Tick every year to turn_year, then every 2 years to end_year."""
    years = list(range(start_year, turn_year + 1)) + list(range(turn_year + 2, end_year + 1, 2))
    tickvals = log_days(pd.to_datetime([f"{y}-01-01" for y in years]))
    ticktext = [str(y) for y in years]
    return tickvals.tolist(), ticktext

def zone_label(price: float, sup: float, bear: float, frothy: float, top: float) -> str:
    if price < sup:
        return "SELL THE HOUSE!!"
    if price < bear:
        return "Buy"
    if price < frothy:
        return "DCA"
    if price < top:
        return "Relax"
    return "Frothy"

# -------------------- figure builders --------------------
def make_powerlaw_fig(df: pd.DataFrame):
    """Return (fig, arrays_for_js) for power-law with USD & Gold traces."""
    # Fit on USD/BTC and anchor at ANCHOR_USD on ANCHOR_DATE
    slope_usd, intercept_usd, sigma = power_fit(df, "USD_per_BTC")
    intercept_usd = anchor_intercept(slope_usd, ANCHOR_DATE, ANCHOR_USD)

    # Convert the anchored USD mid/bands into oz/BTC using gold prices on each date
    # For projection we create future monthly dates then forward-fill gold
    last_hist = df["Date"].iloc[-1]
    future = pd.date_range(last_hist + pd.offsets.MonthBegin(1), PROJ_END, freq="MS")
    full = pd.concat([df[["Date", "USD_per_BTC", "OZ_per_BTC", "Gold"]], 
                      pd.DataFrame({"Date": future})], ignore_index=True)
    full = full.sort_values("Date").reset_index(drop=True)
    full[["USD_per_BTC", "OZ_per_BTC", "Gold"]] = full[["USD_per_BTC", "OZ_per_BTC", "Gold"]].ffill()

    # compute USD bands
    bands_usd = {}
    for name, z, color in BANDS:
        bands_usd[name] = band_series(full["Date"], slope_usd, intercept_usd, sigma, z)

    # derive gold bands by dividing USD bands by Gold (oz/BTC)
    bands_gold = {name: (bands_usd[name] / full["Gold"].to_numpy()) for name, _, _ in BANDS}

    # Build figure
    tickvals, ticktext = year_ticks(2012, 2020, 2040)
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        showlegend=False,
        font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, Arial", size=14),
        xaxis=dict(
            title="Year (log-time)",
            type="linear",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True, gridwidth=0.5, gridcolor="rgba(150,150,180,0.25)",
        ),
    ))

    # USD traces (hidden if denomination=Gold)
    colors = {name: color for name, _, color in BANDS}
    for name in PL_ORDER:
        if name == "BTC":
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["USD_per_BTC"], name="BTC (USD)",
                line=dict(color="gold", width=2), visible=True, yaxis="y1"
            ))
        elif name in colors:
            fig.add_trace(go.Scatter(
                x=full["Date"], y=bands_usd[name], name=f"{name} (USD)",
                line=dict(color=colors[name], width=2, dash="dash"), visible=True, yaxis="y1"
            ))

    # Gold traces (start hidden)
    for name in PL_ORDER:
        if name == "BTC":
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["OZ_per_BTC"], name="BTC (Gold)",
                line=dict(color="gold", width=2, dash="dashdot"),
                visible=False, yaxis="y1"
            ))
        elif name in colors:
            fig.add_trace(go.Scatter(
                x=full["Date"], y=bands_gold[name], name=f"{name} (Gold)",
                line=dict(color=colors[name], width=2, dash="dash"),
                visible=False, yaxis="y1"
            ))

    # Y-axis (we switch title via JS)
    fig.update_yaxes(type="log", title="USD / BTC", tickformat="$,d",
                     showgrid=True, gridwidth=0.5, gridcolor="rgba(150,150,180,0.25)")

    # Pack arrays for JS zone calculator (latest on full timeline)
    arrays = dict(
        dates=[d.strftime("%Y-%m-%d") for d in full["Date"]],
        price_usd=full["USD_per_BTC"].astype(float).tolist(),
        price_gold=full["OZ_per_BTC"].astype(float).tolist(),
        sup_usd=bands_usd["Support"].astype(float).tolist(),
        bear_usd=bands_usd["Bear"].astype(float).tolist(),
        froth_usd=bands_usd["Frothy"].astype(float).tolist(),
        top_usd=bands_usd["Top"].astype(float).tolist(),
        sup_g=bands_gold["Support"].astype(float).tolist(),
        bear_g=bands_gold["Bear"].astype(float).tolist(),
        froth_g=bands_gold["Frothy"].astype(float).tolist(),
        top_g=bands_gold["Top"].astype(float).tolist(),
    )
    return fig, arrays

def make_dma_fig(df: pd.DataFrame):
    # Prepare USD series
    dma = df[df["Date"] >= DMA_START].copy()
    dma["BTC_50"] = dma["USD_per_BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["USD_per_BTC"].rolling(200).mean()
    dma["OZ_50"] = (df["OZ_per_BTC"][df["Date"] >= DMA_START]).rolling(50).mean().to_numpy()
    dma["OZ_200"] = (df["OZ_per_BTC"][df["Date"] >= DMA_START]).rolling(200).mean().to_numpy()

    dma = dma.dropna().reset_index(drop=True)

    # Crossovers (USD): 200DMA crosses down through 50DMA after 100+ days above
    diff = dma["BTC_200"] - dma["BTC_50"]
    above_100 = diff.shift(1).rolling(100).apply(lambda a: float((a > 0).all()), raw=True)
    cross_usd = (diff.shift(1) > 0) & (diff < 0) & (above_100 == 1.0)

    # Crossovers (Gold)
    diffg = dma["OZ_200"] - dma["OZ_50"]
    above_100g = diffg.shift(1).rolling(100).apply(lambda a: float((a > 0).all()), raw=True)
    cross_gold = (diffg.shift(1) > 0) & (diffg < 0) & (above_100g == 1.0)

    tickvals, ticktext = year_ticks(2012, 2020, 2040)
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        showlegend=False,
        font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, Arial", size=14),
        xaxis=dict(
            title="Year (log-time)",
            type="linear",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True, gridwidth=0.5, gridcolor="rgba(150,150,180,0.25)",
        ),
    ))

    # USD traces (visible by default)
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["USD_per_BTC"], name="BTC (USD)",
                             line=dict(color="gold", width=2), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"], name="50-DMA (USD)",
                             line=dict(color="rgb(50,150,90)", width=2), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"], name="200-DMA (USD)",
                             line=dict(color="rgb(20,100,50)", width=2), visible=True))
    fig.add_trace(go.Scatter(x=dma.loc[cross_usd, "Date"], y=dma.loc[cross_usd, "USD_per_BTC"],
                             name="Top Marker (USD)", mode="markers",
                             marker=dict(symbol="diamond", size=10,
                                         color="rgb(0,200,120)", line=dict(width=1,color="#0a0")),
                             visible=True))

    # Gold traces (start hidden)
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["OZ_per_BTC"], name="BTC (Gold)",
                             line=dict(color="gold", width=2, dash="dashdot"), visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["OZ_50"], name="50-DMA (Gold)",
                             line=dict(color="rgb(210,180,70)", width=2), visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["OZ_200"], name="200-DMA (Gold)",
                             line=dict(color="rgb(180,150,40)", width=2), visible=False))
    fig.add_trace(go.Scatter(x=dma.loc[cross_gold, "Date"], y=dma.loc[cross_gold, "OZ_per_BTC"],
                             name="Top Marker (Gold)", mode="markers",
                             marker=dict(symbol="diamond", size=10,
                                         color="rgb(230,190,40)", line=dict(width=1,color="#aa8")),
                             visible=False))

    # Y axis titles are toggled by JS
    fig.update_yaxes(type="log", title="USD / BTC", tickformat="$,d",
                     showgrid=True, gridwidth=0.5, gridcolor="rgba(150,150,180,0.25)")
    arrays = dict(
        dates=[d.strftime("%Y-%m-%d") for d in dma["Date"]],
        price_usd=dma["USD_per_BTC"].astype(float).tolist(),
        price_gold=dma["OZ_per_BTC"].astype(float).tolist(),
    )
    return fig, arrays

# -------------------- HTML writers --------------------
STYLE = """
<style>
:root { --bg:#0e0f12; --panel:#14161a; --text:#e8eaed; --sub:#9aa0a6; --accent:#8ab4f8; }
body{margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial;}
.wrap{max-width:980px; margin:22px auto; padding:0 16px;}
h1{font-size:44px; line-height:1.05; margin:0 0 14px;}
.badge{display:inline-flex; align-items:center; gap:.6rem; background:#1f2228; border-radius:28px; padding:8px 14px; font-weight:600;}
.badge .dot{width:10px;height:10px;border-radius:50%;background:#cfd8dc;}
.row{display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:10px 0 8px;}
.btn{background:var(--panel); color:var(--text); border:1px solid #2a2f36; border-radius:14px; padding:10px 14px; cursor:pointer;}
.btn:active{transform:translateY(1px);}
select, input[type=range]{accent-color:var(--accent);}
.controls{display:flex; align-items:center; gap:14px; flex-wrap:wrap;}
.legend-panel{display:none; background:var(--panel); border:1px solid #2a2f36; border-radius:12px; padding:10px 12px; margin:8px 0;}
.chart{background:#0c0d10; border:1px solid #23262d; border-radius:14px; padding:8px; }
.note{color:var(--sub); font-size:14px; margin:6px 0 12px;}
hr{border:none;border-top:1px solid #2a2f36;margin:8px 0 14px;}
</style>
"""

# JS shared: denomination toggle, legend toggle, zone-by-date slider
COMMON_JS = """
<script>
function toggleLegend(figId){
  const gd = document.getElementById(figId);
  const showing = gd.layout.showlegend === true;
  Plotly.relayout(gd, {showlegend: !showing});
}
function setDenom(figId, denom){  // 'USD' or 'Gold'
  const gd = document.getElementById(figId);
  if(!gd || !gd.data) return;
  const isGold = denom === 'Gold';
  const v = [];
  for (let i=0;i<gd.data.length;i++){
    const name = gd.data[i].name || '';
    v.push(name.includes('(Gold)') ? isGold : (!name.includes('(Gold)')));
  }
  Plotly.restyle(gd, 'visible', v);
  const ytitle = isGold ? 'Gold oz / BTC' : 'USD / BTC';
  const fmt    = isGold ? ',d' : '$,d';
  Plotly.relayout(gd, {'yaxis.title.text': ytitle, 'yaxis.tickformat': fmt});
}
function zoneFor(p, sup, bear, froth, top){
  if (p < sup) return 'SELL THE HOUSE!!';
  if (p < bear) return 'Buy';
  if (p < froth) return 'DCA';
  if (p < top)  return 'Relax';
  return 'Frothy';
}
function initSlider(figId, arrays, denom){
  const isGold = denom === 'Gold';
  const dates = arrays.dates;
  const prices = isGold ? arrays.price_gold : arrays.price_usd;
  const sup = isGold ? arrays.sup_g : arrays.sup_usd;
  const bear = isGold ? arrays.bear_g : arrays.bear_usd;
  const froth = isGold ? arrays.froth_g : arrays.froth_usd;
  const top = isGold ? arrays.top_g : arrays.top_usd;

  const slider = document.getElementById('dateSlider');
  const label  = document.getElementById('dateLabel');
  const badge  = document.getElementById('zoneText');

  function update(idx){
    const i = Math.max(0, Math.min(dates.length-1, idx));
    const p = prices[i];
    const z = zoneFor(p, sup[i], bear[i], froth[i], top[i]);
    badge.textContent = z;
    const unit = isGold ? ' oz/BTC' : ' $/BTC';
    label.textContent = dates[i] + ' · ' + (isGold ? Math.round(p) : p.toLocaleString('en-US',{maximumFractionDigits:0})) + unit;
  }
  slider.setAttribute('min', 0); slider.setAttribute('max', dates.length-1); slider.value = dates.length-1;
  slider.oninput = (e)=> update(parseInt(e.target.value,10));
  update(dates.length-1);
}
</script>
"""

def index_html(fig1_div: str, arrays_js: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Purchase Indicator</title>
{STYLE}
{COMMON_JS}
</head>
<body>
<div class="wrap">
  <h1>BTC Purchase<br/>Indicator</h1>
  <div class="badge"><span class="dot"></span> <span>Current zone:</span> <strong id="zoneText">…</strong></div>

  <div class="row controls" style="margin-top:14px;">
    <a class="btn" href="./dma.html">Open BTC<br/>USD &amp; Gold<br/>DMA chart →</a>
    <div class="row" style="align-items:center;">
      <div>Denomination:</div>
      <select id="denom" class="btn" onchange="setDenom('plfig', this.value); initSlider('plfig', PL_ARRAYS, this.value);">
        <option>USD</option>
        <option>Gold</option>
      </select>
    </div>
    <button class="btn" onclick="toggleLegend('plfig')">Legend</button>
  </div>

  <div class="note">View at date:</div>
  <input id="dateSlider" type="range" min="0" max="10" value="10" style="width:100%"/>
  <div id="dateLabel" class="btn" style="margin-top:8px; display:inline-block;"></div>

  <div class="chart" style="margin-top:12px;">
    {fig1_div}
  </div>
</div>

<script>
const PL_ARRAYS = {arrays_js};
document.addEventListener('DOMContentLoaded', function(){ setDenom('plfig','USD'); initSlider('plfig', PL_ARRAYS, 'USD'); });
</script>
</body></html>"""

def dma_html(fig2_div: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC USD & Gold DMA</title>
{STYLE}
{COMMON_JS}
</head>
<body>
<div class="wrap">
  <div class="row">
    <a class="btn" href="./index.html">← Back</a>
    <div class="row" style="align-items:center;">
      <div>Denomination:</div>
      <select id="denom" class="btn" onchange="setDenom('dmafig', this.value);">
        <option>USD</option>
        <option>Gold</option>
      </select>
    </div>
    <button class="btn" onclick="toggleLegend('dmafig')">Legend</button>
  </div>

  <div class="chart" style="margin-top:12px;">
    {fig2_div}
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function(){ setDenom('dmafig','USD'); });
</script>
</body></html>"""

# -------------------- main build --------------------
def main():
    out = Path("dist"); out.mkdir(parents=True, exist_ok=True)

    df = merge_prices()

    # -------- power-law page
    fig1, arrays = make_powerlaw_fig(df)
    fig1_div = pio.to_html(
        fig1, include_plotlyjs="cdn", full_html=False,
        config=dict(displaylogo=False), div_id="plfig"
    )
    # arrays for client-side slider/zone
    arrays_js = pd.Series(arrays).to_json()  # safe simple JSON

    html = index_html(fig1_div, arrays_js)
    (out / "index.html").write_text(html, encoding="utf-8")

    # -------- DMA page
    fig2, _ = make_dma_fig(df)
    fig2_div = pio.to_html(
        fig2, include_plotlyjs="cdn", full_html=False,
        config=dict(displaylogo=False), div_id="dmafig"
    )
    (out / "dma.html").write_text(dma_html(fig2_div), encoding="utf-8")

    print("[build] wrote dist/index.html and dist/dma.html")

if __name__ == "__main__":
    main()