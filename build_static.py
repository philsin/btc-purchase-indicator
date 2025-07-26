# build_static.py
# Build a static 2-page site (index + DMA) into ./dist for GitHub Pages
# - Power-law chart uses log(time) on x axis (days since 2009-01-03)
# - USD / Gold denomination dropdown for BOTH charts
# - Per-line checkboxes kept for manual visibility control

import io
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ───────────────── constants ─────────────────
UA        = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS   = pd.Timestamp("2009-01-03")
FD_SUPPLY = 21_000_000
PROJ_END  = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")

DIST = Path("dist")
DIST.mkdir(parents=True, exist_ok=True)

LEVELS = {  # fixed sigma bands
    "Support":     -1.5,
    "Bear":        -0.5,
    "PL Best Fit":  0.0,
    "Frothy":      +1.0,
    "Top":         +1.75,
}

# ───────────────── utils ─────────────────
def log_days(dates) -> np.ndarray:
    """log10 of days since GENESIS (for log-time x-axis)."""
    d = pd.to_datetime(dates)
    days = (d - GENESIS).dt.days   # <-- use .dt.days (not .days)
    return np.log10(np.maximum(days, 1))  # guard against 0/neg

def year_ticks(end_year: int):
    """Year labels: every year 2012→2020, then every other year afterward."""
    yrs = list(range(2012, 2021)) + list(range(2022, end_year + 1, 2))
    dv  = pd.to_datetime([f"{y}-01-01" for y in yrs])
    tv  = log_days(dv)
    return tv.tolist(), [str(y) for y in yrs]

# ───────────────── data loaders ─────────────────
def _btc_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "BTC"  for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC > 0").sort_values("Date")[["Date", "BTC"]]

def _btc_github() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date", "BTC"]]

def load_btc() -> pd.DataFrame:
    try:
        df = _btc_stooq()
        if len(df) > 0:
            print(f"[build] BTC from Stooq: {len(df)} rows"); return df
    except Exception as e:
        print(f"[build] Stooq BTC failed: {e}")
    df = _btc_github()
    print(f"[build] BTC from GitHub CSV: {len(df)} rows"); return df

def _gold_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().sort_values("Date")[["Date", "Gold"]]

def _gold_lbma() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date", "Gold"]]

def load_gold() -> pd.DataFrame:
    try:
        g = _gold_stooq()
        if len(g) > 1000:
            print(f"[build] Gold from Stooq: {len(g)} rows"); return g
        print("[build] Stooq XAUUSD too short → LBMA mirror")
    except Exception as e:
        print(f"[build] Stooq Gold failed: {e}")
    g = _gold_lbma()
    print(f"[build] Gold from LBMA: {len(g)} rows"); return g

# ───────────────── power-law & projections ─────────────────
def fit_power(df: pd.DataFrame):
    X = log_days(df["Date"])
    y = np.log10(df["BTC"])
    slope, intercept = np.polyfit(X, y, 1)
    sigma = np.std(y - (slope * X + intercept))
    return slope, intercept, sigma

def anchor_intercept(slope: float, date: pd.Timestamp, target_price: float) -> float:
    x = log_days([date])[0]
    return np.log10(target_price) - slope * x

def project_monthly(last_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    start = (last_date + pd.offsets.MonthBegin(1)).normalize()
    if start > end_date:
        return pd.DatetimeIndex([])
    return pd.date_range(start, end_date, freq="MS")

# ───────────────── figures ─────────────────
def make_powerlaw_fig(full: pd.DataFrame,
                      data: pd.DataFrame,
                      full_usd_bands: dict,
                      data_gold_bands: dict,
                      denom_default="USD") -> go.Figure:
    # x is log-time for both USD and Gold traces
    x_full  = log_days(full["Date"])
    x_data  = log_days(data["Date"])
    end_year = min(2040, full["Date"].dt.year.max())
    tickvals, ticktext = year_ticks(end_year)

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="linear", title="Year",
                   tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   showgrid=True, gridwidth=0.6),
        yaxis=dict(type="log", title="Price (USD / oz Gold)",
                   tickformat="$,d", showgrid=True, gridwidth=0.6),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    # Trace index plan (for dropdown wiring):
    # USD:  Top(0) Frothy(1) PL(2) Bear(3) Support(4) BTC(5)
    # Gold: Top(6) Frothy(7) PL(8) Bear(9) Support(10) BTC(11)

    # USD bands (full timeline incl. projection)
    color_usd = {"Top":"green","Frothy":"rgba(100,255,100,1)","PL Best Fit":"white",
                 "Bear":"rgba(255,100,100,1)","Support":"red"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_full, y=full_usd_bands[nm],
                                 name=f"{nm} (USD)", line=dict(color=color_usd[nm], dash="dash"),
                                 visible=True))
    fig.add_trace(go.Scatter(x=x_data, y=data["BTC"],
                             name="BTC (USD)", line=dict(color="gold", width=2),
                             visible=True))

    # Gold bands (historical only; no future projection → NaN masked naturally)
    color_gld = {"Top":"#ffd54f","Frothy":"#ffeb3b","PL Best Fit":"#fff8e1",
                 "Bear":"#ffcc80","Support":"#ffb74d"}
    for nm in ["Top","Frothy","PL Best Fit","Bear","Support"]:
        fig.add_trace(go.Scatter(x=x_data, y=data_gold_bands[nm],
                                 name=f"{nm} (Gold)", line=dict(color=color_gld[nm], dash="dash"),
                                 visible=False))
    fig.add_trace(go.Scatter(x=x_data, y=data["BTCG"],
                             name="BTC (Gold)", line=dict(color="#ffc107", width=2),
                             visible=False))

    # Adjust default group visibility
    if denom_default.upper() == "GOLD":
        for i in range(0, 6):
            fig.data[i].visible = "legendonly"
        for i in range(6, 12):
            fig.data[i].visible = True

    return fig

def rolling_markers(d: pd.DataFrame):
    diff = d["BTC_200"] - d["BTC_50"]
    was_above = diff.shift(1) > 0
    was_above_100 = was_above.rolling(100).apply(lambda x: (x > 0).all(), raw=False).astype(bool)
    cross_down = (diff.shift(1) > 0) & (diff < 0) & was_above_100
    return d.loc[cross_down, "Date"], d.loc[cross_down, "BTC"], d.loc[cross_down, "BTCG"]

def make_dma_fig(dma: pd.DataFrame, denom_default="USD") -> go.Figure:
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="date", title="Year", showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="BTC Price (USD)", tickformat="$,d", showgrid=True, gridwidth=0.6),
        yaxis2=dict(type="log", title="BTC Price (oz Gold)", tickformat=",d",
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))
    # USD group (0..3)
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"], name="200-DMA USD",
                             line=dict(color="#2e7d32", width=1.5), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],  name="50-DMA USD",
                             line=dict(color="#43a047", width=1.5), visible=True))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],     name="BTC USD",
                             line=dict(color="#66bb6a", width=2), visible=True))
    m_dates, m_usd, m_gld = rolling_markers(dma)
    fig.add_trace(go.Scatter(x=m_dates, y=m_usd, name="Top Marker (USD)", mode="markers",
                             marker=dict(symbol="diamond", color="#00c853", size=9), visible=True))
    # Gold group (4..7)
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"], name="200-DMA Gold",
                             line=dict(color="#ffd54f", width=1.5), yaxis="y2", visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],  name="50-DMA Gold",
                             line=dict(color="#ffeb3b", width=1.5), yaxis="y2", visible=False))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"], name="BTC Gold",
                             line=dict(color="#ffc107", width=2), yaxis="y2", visible=False))
    fig.add_trace(go.Scatter(x=m_dates, y=m_gld, name="Top Marker (Gold)", mode="markers",
                             marker=dict(symbol="diamond", color="#ffd600", size=9),
                             yaxis="y2", visible=False))

    if denom_default.upper() == "GOLD":
        for i in range(0, 4):  fig.data[i].visible = "legendonly"
        for i in range(4, 8):  fig.data[i].visible = True
    return fig

# ───────────────── small HTML helpers ─────────────────
CSS = """
:root{--bg:#0e0e10;--panel:#121216;--text:#e6e6e6;--muted:#9aa0a6;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:16px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:clamp(28px,4vw,40px);margin:8px 0 18px}
a{color:#8ab4ff;text-decoration:none} a:hover{text-decoration:underline}
.badge{display:inline-flex;align-items:center;gap:10px;background:#22242a;border-radius:28px;padding:10px 14px;margin:8px 0 20px}
.dot{width:12px;height:12px;border-radius:50%;background:#8ab4ff;display:inline-block}
.panel{background:var(--panel);border-radius:12px;padding:12px 12px}
.btn{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:8px 12px;border:1px solid #2a2d34;color:#e6e6e6}
.btn:hover{background:#1b1e23}
.back{margin:6px 0 10px;display:inline-block}
.controls{display:flex;gap:18px;align-items:center;margin:10px 0 12px;flex-wrap:wrap}
fieldset{border:1px solid #2b2f36;border-radius:10px;padding:8px 12px} legend{color:#9aa0a6;padding:0 6px;font-size:13px}
select{background:#1b1e23;border:1px solid #2b2f36;border-radius:8px;color:#e6e6e6;padding:6px 10px}
"""

JS_TOGGLE = """
function bindDenomSelect(figId, selectId, usdIdxs, goldIdxs){
  const sel=document.getElementById(selectId);
  function setVis(){
    const usdVis = sel.value==='USD' ? true : 'legendonly';
    const gVis   = sel.value==='Gold'? true : 'legendonly';
    Plotly.restyle(figId, {'visible': usdVis}, usdIdxs);
    Plotly.restyle(figId, {'visible': gVis  }, goldIdxs);
  }
  sel.addEventListener('change', setVis);
  setVis();
}
function bindLineToggle(figId){
  document.querySelectorAll('input[data-trace]').forEach(cb=>{
    function apply(){ Plotly.restyle(figId, {'visible': cb.checked? true:'legendonly'}, [parseInt(cb.value)]) }
    cb.addEventListener('change', apply); apply();
  });
}
function back(){ history.length ? history.back() : location.href = './index.html'; }
"""

def wrap_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head><body><div class="wrap">
{body}
</div>
<script>{JS_TOGGLE}</script>
</body></html>"""

def zone_badge_html(zone: str) -> str:
    color = {"SELL THE HOUSE!!":"#ff5252","Undervalued":"#29b6f6","Fair Value":"#e6e6e6",
             "Overvalued":"#ffb74d","TO THE MOON":"#ffd54f","DCA":"#e6e6e6"}.get(zone,"#e6e6e6")
    return f'<div class="badge"><span class="dot" style="background:{color}"></span><b>Current zone:</b> {zone}</div>'

# ───────────────── page builders ─────────────────
def build():
    # Load / merge
    btc  = load_btc()
    gold = load_gold()
    data = pd.merge(btc, gold, on="Date", how="inner")

    # Power-law fit & anchor (~$500k on 2030-01-01)
    slope, intercept, sigma = fit_power(data)
    intercept = anchor_intercept(slope, pd.Timestamp("2030-01-01"), 491_776)
    sigma_vis = max(sigma, 0.25)

    # Projection (USD only)
    future = project_monthly(data["Date"].iloc[-1], PROJ_END)
    full = pd.concat([data[["Date","BTC"]], pd.DataFrame({"Date": future})], ignore_index=True)

    # USD bands on full timeline (including projection)
    x_full = log_days(full["Date"])
    mid_log_full = slope * x_full + intercept
    bands_usd_full = {nm: 10 ** (mid_log_full + sigma_vis * k) for nm, k in LEVELS.items()}

    # Gold-denominated bands (historical only) by dividing USD bands by gold
    x_data = log_days(data["Date"])
    mid_log_data = slope * x_data + intercept
    bands_usd_data = {nm: 10 ** (mid_log_data + sigma_vis * k) for nm, k in LEVELS.items()}
    data = data.copy()
    data["BTCG"] = data["BTC"] / data["Gold"]
    bands_gold_data = {nm: bands_usd_data[nm] / data["Gold"].to_numpy() for nm in LEVELS}

    # Zone using latest historical USD point
    latest = data.iloc[-1]
    # sample bands on that date
    x_latest = log_days([latest["Date"]])[0]
    pl_latest = slope * x_latest + intercept
    ref = {nm: 10 ** (pl_latest + sigma_vis * k) for nm, k in LEVELS.items()}
    p = latest["BTC"]
    if p < ref["Support"]: zone = "SELL THE HOUSE!!"
    elif p < ref["Bear"]:  zone = "Undervalued"
    elif p < ref["Frothy"]:zone = "Fair Value"
    elif p < ref["Top"]:   zone = "Overvalued"
    else:                  zone = "TO THE MOON"

    # Figures
    fig1 = make_powerlaw_fig(full, data, bands_usd_full, bands_gold_data, denom_default="USD")
    fig1_html = pio.to_html(fig1, include_plotlyjs=False, full_html=False, config={"displaylogo": False}, div_id="plfig")

    dma = data[data["Date"] >= DMA_START].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma["G50"]     = dma["BTCG"].rolling(50).mean()
    dma["G200"]    = dma["BTCG"].rolling(200).mean()
    dma = dma.dropna()
    fig2 = make_dma_fig(dma, denom_default="USD")
    fig2_html = pio.to_html(fig2, include_plotlyjs=False, full_html=False, config={"displaylogo": False}, div_id="dmafig")

    # ----- index.html (power-law) -----
    index_controls = """
<div class="controls">
  <a class="back btn" href="javascript:void(0)" onclick="back()">← Back</a>
  <label>Denomination:
    <select id="denom1">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
  </label>
  <fieldset><legend>Lines</legend>
    <!-- USD traces 0..5 -->
    <label><input type="checkbox" data-trace value="0" checked> Top (USD)</label>
    <label><input type="checkbox" data-trace value="1" checked> Frothy (USD)</label>
    <label><input type="checkbox" data-trace value="2" checked> PL Best Fit (USD)</label>
    <label><input type="checkbox" data-trace value="3" checked> Bear (USD)</label>
    <label><input type="checkbox" data-trace value="4" checked> Support (USD)</label>
    <label><input type="checkbox" data-trace value="5" checked> BTC (USD)</label>
    <span style="margin:0 6px">|</span>
    <!-- Gold traces 6..11 -->
    <label><input type="checkbox" data-trace value="6"> Top (Gold)</label>
    <label><input type="checkbox" data-trace value="7"> Frothy (Gold)</label>
    <label><input type="checkbox" data-trace value="8"> PL Best Fit (Gold)</label>
    <label><input type="checkbox" data-trace value="9"> Bear (Gold)</label>
    <label><input type="checkbox" data-trace value="10"> Support (Gold)</label>
    <label><input type="checkbox" data-trace value="11"> BTC (Gold)</label>
  </fieldset>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  bindDenomSelect('plfig','denom1',[0,1,2,3,4,5],[6,7,8,9,10,11]);
  bindLineToggle('plfig');
});
</script>
"""
    index_body = f"""
<h1>BTC Purchase Indicator</h1>
{zone_badge_html(zone)}
<p><a href="./dma.html">Open BTC USD &amp; Gold DMA chart →</a></p>
{index_controls}
<div class="panel">{fig1_html}</div>
"""
    (DIST / "index.html").write_text(wrap_html("BTC Purchase Indicator", index_body), encoding="utf-8")

    # ----- dma.html -----
    dma_controls = """
<div class="controls">
  <a class="back btn" href="javascript:void(0)" onclick="back()">← Back</a>
  <label>Denomination:
    <select id="denom2">
      <option value="USD" selected>USD</option>
      <option value="Gold">Gold</option>
    </select>
  </label>
  <fieldset><legend>Lines</legend>
    <!-- USD traces 0..3 -->
    <label><input type="checkbox" data-trace value="0" checked> 200-DMA USD</label>
    <label><input type="checkbox" data-trace value="1" checked> 50-DMA USD</label>
    <label><input type="checkbox" data-trace value="2" checked> BTC USD</label>
    <label><input type="checkbox" data-trace value="3" checked> Top Marker (USD)</label>
    <span style="margin:0 6px">|</span>
    <!-- Gold traces 4..7 -->
    <label><input type="checkbox" data-trace value="4"> 200-DMA Gold</label>
    <label><input type="checkbox" data-trace value="5"> 50-DMA Gold</label>
    <label><input type="checkbox" data-trace value="6"> BTC Gold</label>
    <label><input type="checkbox" data-trace value="7"> Top Marker (Gold)</label>
  </fieldset>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  bindDenomSelect('dmafig','denom2',[0,1,2,3],[4,5,6,7]);
  bindLineToggle('dmafig');
});
</script>
"""
    dma_body = f"""
<h1>BTC USD &amp; Gold — 50/200 DMA</h1>
{dma_controls}
<div class="panel">{pio.to_html(fig2, include_plotlyjs=False, full_html=False, config={{"displaylogo": False}}, div_id="dmafig")}</div>
"""
    (DIST / "dma.html").write_text(wrap_html("BTC — USD & Gold DMA", dma_body), encoding="utf-8")

    # robots
    (DIST / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    print("[build] wrote: dist/index.html, dist/dma.html")

# ───────────────── entry ─────────────────
if __name__ == "__main__":
    build()