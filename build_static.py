# build_static.py
# Outputs a static site to ./dist with:
#  - index.html (Power-law bands, USD/Gold toggle, log-time x-axis, projection to 2040)
#  - dma.html    (50/200 DMA, USD/Gold toggle, Top Markers)
import os, json, math, io, requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

# -------------------- constants --------------------
UA        = {"User-Agent": "btc-pl-pages/1.0"}
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")

# fixed band multipliers
LEVELS = {
    "Support": -1.5,
    "Bear":    -0.5,
    "PL Best Fit": 0.0,
    "Frothy":  +1.0,
    "Top":     +1.75,
}

# -------------------- helpers ----------------------
def ensure_dist():
    os.makedirs("dist", exist_ok=True)

def _btc_stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df  = pd.read_csv(url)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # map date / close
    date_col  = next(c for c in cols if "date"  in c)
    close_col = next(c for c in cols if "close" in c or "price" in c)
    df = df.rename(columns={date_col: "Date", close_col: "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw).rename(columns={"Date": "Date", "Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().sort_values("Date")

def _gold_stooq():
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"  # Gold (oz) in USD
    df  = pd.read_csv(url)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    date_col  = next(c for c in cols if "date"  in c)
    close_col = next(c for c in cols if "close" in c or "price" in c)
    df = df.rename(columns={date_col: "Date", close_col: "Gold"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")

def _gold_lbma():
    # simple public mirror (Date, USD (PM))
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df[["Date","Gold"]].dropna().sort_values("Date")

def load_data():
    # BTC
    try:
        b = _btc_stooq()
        if len(b) < 1000:
            raise ValueError("short")
    except Exception:
        b = _btc_github()
    # Gold
    try:
        g = _gold_stooq()
        if len(g) < 1000:
            raise ValueError("short gold")
    except Exception:
        g = _gold_lbma()

    # align on Date
    df = b.merge(g, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    # oz/BTC = BTC[USD] / Gold[USD/oz]
    df["OZ_per_BTC"] = df["BTC"] / df["Gold"]
    return df

def log_days(dates):
    """Return log10(days since GENESIS) robust to Series/Index/array."""
    d = pd.to_datetime(dates)
    # convert to int days safely across pandas versions
    days = (d - GENESIS).values.astype("timedelta64[D]").astype("int64")
    days = np.where(days <= 0, 1, days)  # guard
    return np.log10(days.astype(float))

def year_ticks(end_year):
    """Year ticks: every year ≤2020, every 2 years afterward."""
    years = list(range(2012, int(end_year) + 1))
    tickvals = log_days(pd.to_datetime([f"{y}-01-01" for y in years]))
    ticktext = [str(y) if (y <= 2020 or y % 2 == 0) else "" for y in years]
    return tickvals.tolist(), ticktext

def fit_power(df, ycol):
    x = log_days(df["Date"])
    y = np.log10(df[ycol].astype(float))
    slope, intercept = np.polyfit(x, y, 1)
    sigma = np.std(y - (slope * x + intercept))
    return slope, intercept, sigma

def anchor_intercept(slope, anchor_date, target_price):
    adays = (pd.to_datetime(anchor_date) - GENESIS).days
    adays = max(1, adays)
    return np.log10(target_price) - slope * np.log10(adays)

def project_monthly(last_date, end_date):
    start = (pd.to_datetime(last_date) + pd.offsets.MonthBegin(1)).normalize()
    return pd.date_range(start=start, end=pd.to_datetime(end_date), freq="MS")

def zone_label(p, row_bands):
    if p < row_bands["Support"]:
        return "SELL THE HOUSE!!"
    elif p < row_bands["Bear"]:
        return "Buy"
    elif p < row_bands["Frothy"]:
        return "DCA"
    elif p < row_bands["Top"]:
        return "Relax"
    else:
        return "Frothy"

# ---------------- figure factories ----------------
def make_powerlaw_fig(full, data, bands_usd_full, bands_gold_full):
    x_full = log_days(full["Date"])
    x_data = log_days(data["Date"])
    end_year = int(min(2040, full["Date"].dt.year.max()))
    tickvals, ticktext = year_ticks(end_year)

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="linear", title="Year (log-time)",
                   tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridwidth=0.6),
        yaxis=dict(type="log", title="USD / BTC",
                   tickformat="$,d", showgrid=True, gridwidth=0.6),
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=64, r=40, t=16, b=56),
        showlegend=False
    ))

    # USD traces first (0..5)
    color_usd = {"Top":"green","Frothy":"rgba(100,255,100,1)","PL Best Fit":"white",
                 "Bear":"rgba(255,100,100,1)","Support":"red"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_full, y=bands_usd_full[nm],
                                 name=f"{nm} (USD)", line=dict(color=color_usd[nm], dash="dash"),
                                 visible=True))
    fig.add_trace(go.Scatter(x=x_data, y=data["BTC"],
                             name="BTC (USD)", line=dict(color="gold", width=2), visible=True))

    # GOLD traces (6..11) – fit directly in oz/BTC space
    color_gold = {"Top":"#ffd54f","Frothy":"#ffeb3b","PL Best Fit":"#fff8e1",
                  "Bear":"#ffcc80","Support":"#ffb74d"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_full, y=bands_gold_full[nm],
                                 name=f"{nm} (Gold oz/BTC)",
                                 line=dict(color=color_gold[nm], dash="dash"),
                                 visible=False))
    fig.add_trace(go.Scatter(x=x_data, y=data["OZ_per_BTC"],
                             name="BTC (Gold oz/BTC)",
                             line=dict(color="#ffc107", width=2), visible=False))
    return fig

def make_dma_fig(data):
    """DMA chart with USD and Gold denomination groups; x is log-time."""
    dma = data[data["Date"] >= DMA_START].copy()
    # USD DMAs
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    # Gold DMAs (oz/BTC)
    dma["G_50"]  = dma["OZ_per_BTC"].rolling(50).mean()
    dma["G_200"] = dma["OZ_per_BTC"].rolling(200).mean()
    dma = dma.dropna().reset_index(drop=True)

    # Top Markers (USD): 200 crosses down under 50 after ≥100 days above
    diff = dma["BTC_200"] - dma["BTC_50"]
    above_100 = diff.shift(1).rolling(100).apply(lambda x: float((x > 0).all()), raw=True) == 1.0
    cross_usd = (diff.shift(1) > 0) & (diff < 0) & above_100

    # Top Markers (Gold):
    diff_g = dma["G_200"] - dma["G_50"]
    above_100_g = diff_g.shift(1).rolling(100).apply(lambda x: float((x > 0).all()), raw=True) == 1.0
    cross_gold = (diff_g.shift(1) > 0) & (diff_g < 0) & above_100_g

    x = log_days(dma["Date"])
    end_year = int(min(2040, data["Date"].dt.year.max()))
    tickvals, ticktext = year_ticks(end_year)

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="linear", title="Year (log-time)",
                   tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridwidth=0.6),
        yaxis=dict(type="log", title="USD / BTC", tickformat="$,d",
                   showgrid=True, gridwidth=0.6),
        plot_bgcolor="#111", paper_bgcolor="#111",
        margin=dict(l=64, r=40, t=16, b=56),
        showlegend=False
    ))

    # USD (0..3)
    fig.add_trace(go.Scatter(x=x, y=dma["BTC_200"], name="200-DMA (USD)",
                             line=dict(color="#2e7d32", width=1.6), visible=True))
    fig.add_trace(go.Scatter(x=x, y=dma["BTC_50"],  name="50-DMA (USD)",
                             line=dict(color="#66bb6a", width=1.6), visible=True))
    fig.add_trace(go.Scatter(x=x, y=dma["BTC"],     name="BTC (USD)",
                             line=dict(color="gold", width=2.2), visible=True))
    fig.add_trace(go.Scatter(x=x[cross_usd], y=dma.loc[cross_usd,"BTC"],
                             name="Top Marker (USD)", mode="markers",
                             marker=dict(symbol="diamond", size=9, color="#00e676"),
                             visible=True))

    # Gold (4..7)  oz/BTC
    fig.add_trace(go.Scatter(x=x, y=dma["G_200"], name="200-DMA (Gold oz/BTC)",
                             line=dict(color="#ffb300", width=1.6), visible=False))
    fig.add_trace(go.Scatter(x=x, y=dma["G_50"],  name="50-DMA (Gold oz/BTC)",
                             line=dict(color="#ffd54f", width=1.6), visible=False))
    fig.add_trace(go.Scatter(x=x, y=dma["OZ_per_BTC"], name="BTC (Gold oz/BTC)",
                             line=dict(color="#ffc107", width=2.2), visible=False))
    fig.add_trace(go.Scatter(x=x[cross_gold], y=dma.loc[cross_gold,"OZ_per_BTC"],
                             name="Top Marker (Gold)", mode="markers",
                             marker=dict(symbol="diamond", size=9, color="#ff8f00"),
                             visible=False))
    return fig

# -------------------- build pages --------------------
def build():
    ensure_dist()
    data = load_data()

    # -------- USD power-law (fit & project) ----------
    slope_u, intercept_u, sigma_u = fit_power(data, "BTC")
    # anchor to ~491,776 on 2030-01-01
    intercept_u = anchor_intercept(slope_u, pd.Timestamp("2030-01-01"), 491_776)
    sigma_u = max(sigma_u, 0.25)

    future = project_monthly(data["Date"].iloc[-1], PROJ_END)
    full = pd.concat([data[["Date","BTC","Gold","OZ_per_BTC"]],
                      pd.DataFrame({"Date": future})], ignore_index=True)

    x_full = log_days(full["Date"])
    mid_u  = slope_u * x_full + intercept_u
    bands_usd_full = {nm: 10 ** (mid_u + sigma_u * k) for nm, k in LEVELS.items()}

    # -------- Gold power-law (fit in oz/BTC & project) ----------
    slope_g, intercept_g, sigma_g = fit_power(data, "OZ_per_BTC")
    sigma_g = max(sigma_g, 0.25)
    mid_g   = slope_g * x_full + intercept_g
    bands_gold_full = {nm: 10 ** (mid_g + sigma_g * k) for nm, k in LEVELS.items()}

    # -------- Zone badge (USD) ----------
    latest = data.iloc[-1]
    # Compute current USD bands at latest date
    x_latest = log_days([latest["Date"]])[0]
    mid_latest = slope_u * x_latest + intercept_u
    row_bands = {nm: 10 ** (mid_latest + sigma_u * k) for nm, k in LEVELS.items()}
    zone = zone_label(latest["BTC"], row_bands)

    # -------- figures ----------
    fig1 = make_powerlaw_fig(full, data, bands_usd_full, bands_gold_full)
    fig1_div = pio.to_html(fig1, include_plotlyjs=False, full_html=False,
                           config=dict(displaylogo=False), div_id="plfig")

    fig2 = make_dma_fig(data)
    fig2_div = pio.to_html(fig2, include_plotlyjs=False, full_html=False,
                           config=dict(displaylogo=False), div_id="dmafig")

    # -------- shared JS/CSS ----------
    css = """
    body{background:#0f0f10;color:#e7e7ea;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Inter,Helvetica,Arial,sans-serif;margin:0}
    .wrap{max-width:1100px;margin:0 auto;padding:16px}
    h1{font-weight:800;letter-spacing:.4px;margin:8px 0 12px}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    .badge{display:inline-flex;align-items:center;gap:8px;background:#202125;border-radius:20px;padding:8px 12px}
    .dot{width:10px;height:10px;border-radius:50%;background:#aaa}
    .btn{background:#1b1c1f;border:1px solid #333;border-radius:10px;color:#dcdde3;padding:8px 12px}
    .btn:active{transform:translateY(1px)}
    select{background:#1b1c1f;border:1px solid #333;border-radius:10px;color:#dcdde3;padding:8px 10px}
    .card{background:#121316;border:1px solid #2a2c32;border-radius:18px;padding:10px}
    .note{opacity:.85;font-size:.95rem;margin:10px 0 4px}
    .slider{width:100%}
    .toolbar{display:flex;gap:10px;align-items:center;margin:6px 0 10px}
    a{color:#7db3ff;text-decoration:none}
    """

    js = """
    function toggleLegend(figId){
      const gd = document.getElementById(figId);
      const cur = (gd && gd.layout && typeof gd.layout.showlegend !== 'undefined')
                    ? gd.layout.showlegend : false;
      Plotly.relayout(gd, {'showlegend': !cur});
    }

    function setDenom(figId, denom){
      // USD traces 0..5 visible for USD; Gold traces 6..11 visible for Gold
      const usdVis  = (denom === 'USD');
      const goldVis = !usdVis;
      const vis = [];
      for(let i=0;i<12;i++){
        if(i<=5){ vis.push(usdVis); } else { vis.push(goldVis); }
      }
      Plotly.restyle(figId, 'visible', vis);
      // y-axis title
      const ytitle = usdVis ? 'USD / BTC' : 'Gold oz / BTC';
      Plotly.relayout(figId, {'yaxis.title.text': ytitle});
    }

    function setDenomDMA(figId, denom){
      // USD traces 0..3; Gold traces 4..7
      const usdVis  = (denom === 'USD');
      const goldVis = !usdVis;
      const vis = [];
      for(let i=0;i<8;i++){
        if(i<=3){ vis.push(usdVis); } else { vis.push(goldVis); }
      }
      Plotly.restyle(figId, 'visible', vis);
      const ytitle = usdVis ? 'USD / BTC' : 'Gold oz / BTC';
      Plotly.relayout(figId, {'yaxis.title.text': ytitle});
    }

    function goto(href){ window.location.href = href; }

    // Slider → recompute badge at that date (USD bands)
    function initSlider(){
      const store = JSON.parse(document.getElementById('plstore').textContent);
      const slider = document.getElementById('viewsld');
      const readout = document.getElementById('viewlbl');
      slider.min = 0; slider.max = store.dates.length-1; slider.value = store.dates.length-1;
      function fmt(d){ const dt=new Date(d); return dt.toISOString().slice(0,10); }
      function zoneAt(i){
        const p = store.usd_price[i];
        const s = store.usd_bands.Support[i];
        const b = store.usd_bands.Bear[i];
        const f = store.usd_bands.Frothy[i];
        const t = store.usd_bands.Top[i];
        let z = '';
        if(p < s) z='SELL THE HOUSE!!';
        else if(p < b) z='Buy';
        else if(p < f) z='DCA';
        else if(p < t) z='Relax';
        else z='Frothy';
        return z;
      }
      function update(){
        const i = parseInt(slider.value);
        readout.textContent = fmt(store.dates[i]) + ' · ' + zoneAt(i);
      }
      slider.addEventListener('input', update);
      update();
    }
    document.addEventListener('DOMContentLoaded', initSlider);
    """

    # ------ store for slider (dates + USD prices + bands) ------
    store = {
        "dates": full["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "usd_price": data["BTC"].reindex(range(len(full)), method="ffill").tolist(),
        "usd_bands": {k: bands_usd_full[k].tolist() for k in LEVELS.keys()}
    }

    # ---------------- index.html (power-law) -------------------
    denom_select = """
      <div class="toolbar">
        <button class="btn" onclick="goto('./dma.html')">Open BTC USD & Gold DMA chart →</button>
        <div style="flex:1"></div>
        <label>Denomination:</label>
        <select id="denom" onchange="setDenom('plfig', this.value)">
          <option value="USD" selected>USD</option>
          <option value="Gold">Gold</option>
        </select>
        <button class="btn" onclick="toggleLegend('plfig')">Legend</button>
      </div>
    """
    zone_color = {"SELL THE HOUSE!!":"#ff5252","Buy":"#4caf50","DCA":"#cccccc",
                  "Relax":"#64b5f6","Frothy":"#ffb300"}[zone]
    index_html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC Purchase Indicator</title>
<link rel="preconnect" href="https://cdn.plot.ly"/>
<style>{css}</style>
</head><body>
<div class="wrap">
  <h1>BTC Purchase Indicator</h1>
  <div class="row">
    <div class="badge"><span class="dot" style="background:{zone_color}"></span><b>Current zone:</b>&nbsp;{zone}</div>
  </div>
  {denom_select}
  <div class="note">View at date:</div>
  <input id="viewsld" class="slider" type="range" min="0" max="1" value="1"/>
  <div id="viewlbl" class="note"></div>
  <div class="card">{fig1_div}</div>
</div>

<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>{js}</script>
<script id="plstore" type="application/json">{json.dumps(store)}</script>
<script>
// default denomination
setDenom('plfig','USD');
</script>
</body></html>
"""
    with open("dist/index.html","w",encoding="utf-8") as f:
        f.write(index_html)

    # ---------------- dma.html -------------------
    dma_html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC DMA (USD & Gold)</title>
<link rel="preconnect" href="https://cdn.plot.ly"/>
<style>{css}</style>
</head><body>
<div class="wrap">
  <div class="toolbar">
    <button class="btn" onclick="goto('./')">← Back</button>
    <div style="flex:1"></div>
    <label>Denomination:</label>
    <select id="denom2" onchange="setDenomDMA('dmafig', this.value)">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
    <button class="btn" onclick="toggleLegend('dmafig')">Legend</button>
  </div>
  <div class="card">{fig2_div}</div>
</div>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>{js}</script>
<script>setDenomDMA('dmafig','USD');</script>
</body></html>
"""
    with open("dist/dma.html","w",encoding="utf-8") as f:
        f.write(dma_html)

    print("[build] Wrote dist/index.html and dist/dma.html")

if __name__ == "__main__":
    build()