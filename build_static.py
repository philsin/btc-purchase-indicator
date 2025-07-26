# build_static.py
# Static builder for BTC Purchase Indicator (GitHub Pages)
# - Power-law chart (log-time x-axis to 2040) with USD/Gold dropdown
# - DMA chart (50/200) with USD/Gold dropdown
# - “Legend” toggle, Current-zone badge, date slider
# - Outputs written to ./dist

import os, io, json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# constants
GENESIS   = pd.Timestamp("2009-01-03")
UA        = {"User-Agent": "btc-pl-pages/1.0"}
FD_SUPPLY = 21_000_000
PROJ_END  = pd.Timestamp("2040-12-31")

OUT = Path("dist")
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# helpers: robust date → log10(days) & year ticks

def log_days(dates) -> np.ndarray:
    """Return log10(days since GENESIS) for array-like of dates (Series, list, DatetimeIndex)."""
    dt = pd.to_datetime(dates)
    td = dt - GENESIS
    # Convert to days without using .dt (works on Series and DatetimeIndex)
    days = (td / np.timedelta64(1, "D")).astype(float)
    # guard against log10(0)
    days = np.clip(days, 1.0, None)
    return np.log10(days)

def year_ticks(start: int, mid: int, end: int):
    """Years every 1 from start..mid, then every 2 to end. Returns (tickvals, ticktext) for log-time."""
    left  = np.arange(start,  mid + 1, 1, dtype=int)
    right = np.arange(mid + 2, end + 1, 2, dtype=int)
    years = np.r_[left, right]
    tickdates = pd.to_datetime([f"{y}-01-01" for y in years])
    tickvals  = log_days(tickdates).tolist()
    ticktext  = [str(y) for y in years]
    return tickvals, ticktext

# ─────────────────────────────────────────────────────────────
# data loaders (no API keys)

def _btc_stooq():
    r = requests.get("https://stooq.com/q/d/l/?s=btcusd&i=d", timeout=30, headers=UA)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]
    df = df.rename(columns={date_col: "Date", price_col: "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    r = requests.get(raw, timeout=30, headers=UA); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)).rename(columns={"Closing Price (USD)":"BTC"})
    df["Date"] = pd.to_datetime(df["Date"]); df["BTC"]=pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().sort_values("Date")

def _gold_stooq():
    r = requests.get("https://stooq.com/q/d/l/?s=xauusd&i=d", timeout=30, headers=UA)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]
    df = df.rename(columns={date_col:"Date", price_col:"Gold"})
    df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"]=pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")

def load_merged():
    # BTC first source then fallback
    try:
        b = _btc_stooq()
        if len(b) < 1000:
            raise ValueError("short BTC from Stooq")
    except Exception:
        b = _btc_github()
    # Gold only from Stooq
    g = _gold_stooq()
    df = b.merge(g, on="Date", how="inner")
    return df

# ─────────────────────────────────────────────────────────────
# model & transforms

def fit_power(df_btc):
    """Return slope, intercept, sigma on log10(time) → log10(price)."""
    X = log_days(df_btc["Date"])
    y = np.log10(df_btc["BTC"].values)
    slope, intercept = np.polyfit(X, y, 1)
    sigma = np.std(y - (slope * X + intercept))
    return slope, intercept, sigma

def anchor_intercept(slope, target_price=491_776, target_date="2030-01-01"):
    x = log_days([pd.Timestamp(target_date)])[0]
    return np.log10(target_price) - slope * x

def extend_to_2040(df):
    last = df["Date"].iloc[-1]
    future = pd.date_range((last + pd.offsets.MonthBegin(1)), PROJ_END, freq="MS")
    if len(future):
        tail = pd.DataFrame({"Date": future})
        return pd.concat([df, tail], ignore_index=True)
    return df.copy()

def bands_df(df, slope, intercept, sigma):
    """Compute PL mid & fixed bands into a copy of df (Date, BTC, Gold)."""
    ylog_mid = slope * log_days(df["Date"]) + intercept
    dfc = df.copy()
    dfc["PL Best Fit"] = 10 ** ylog_mid
    # fixed sigma multipliers
    levels = {
        "Top"     : +1.75,
        "Frothy"  : +1.00,
        "Bear"    : -0.50,
        "Support" : -1.50,
    }
    sig = max(0.25, sigma)   # keep visible
    for name, k in levels.items():
        dfc[name] = 10 ** (ylog_mid + sig * k)
    return dfc

def zone_for(price, row):
    if price < row["Support"]: return "SELL THE HOUSE!!"
    if price < row["Bear"]:    return "Buy"
    if price < row["Frothy"]:  return "DCA"
    if price < row["Top"]:     return "Relax"
    return "Frothy"

# ─────────────────────────────────────────────────────────────
# figure builders (Plotly JSON for embedding)

COLORS = dict(
    price="gold",
    mid="white",
    top="#16a34a",       # green-600
    frothy="#86efac",    # green-300
    bear="#fca5a5",      # red-300
    support="#ef4444",   # red-500
)

def fig_powerlaw(df_all, denom="USD"):
    """Return Plotly figure JSON for power-law chart on log-time x-axis."""
    if denom == "USD":
        yseries = dict(
            price=("BTC", COLORS["price"], 2.0),
            bands=[("Top", COLORS["top"]), ("Frothy", COLORS["frothy"]),
                   ("PL Best Fit", COLORS["mid"]), ("Bear", COLORS["bear"]),
                   ("Support", COLORS["support"])],
            ytitle="USD / BTC", tickformat="$,d",
        )
        yvals = df_all
    else:
        # Gold oz per BTC: oz/BTC = Gold_USD_per_oz / BTC_USD
        df = df_all.copy()
        df["BTC_gold"] = df["Gold"] / df["BTC"]
        for c in ["PL Best Fit", "Top", "Frothy", "Bear", "Support"]:
            df[f"{c} (Gold)"] = df["Gold"] / df[c]
        yseries = dict(
            price=("BTC_gold", COLORS["price"], 2.0),
            bands=[("Top (Gold)", COLORS["top"]), ("Frothy (Gold)", COLORS["frothy"]),
                   ("PL Best Fit (Gold)", COLORS["mid"]), ("Bear (Gold)", COLORS["bear"]),
                   ("Support (Gold)", COLORS["support"])],
            ytitle="Gold oz / BTC", tickformat=",d",
        )
        yvals = df

    # x in log-time
    x = log_days(yvals["Date"])
    tickvals, ticktext = year_ticks(2012, 2020, 2040)

    fig = go.Figure()
    for name, col in yseries["bands"]:
        fig.add_trace(go.Scatter(
            x=x, y=yvals[name], name=name,
            mode="lines", line=dict(color=col, dash="dash", width=1.8),
            hovertemplate="%{y:.0f}<extra>"+name+"</extra>"
        ))
    pcol, pw = yseries["price"][1], yseries["price"][2]
    fig.add_trace(go.Scatter(
        x=x, y=yvals[yseries["price"][0]], name=yseries["price"][0],
        mode="lines", line=dict(color=pcol, width=pw),
        hovertemplate="%{y:.0f}<extra>"+yseries["price"][0]+"</extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=20, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False,
        ),
        yaxis=dict(
            title=yseries["ytitle"], tickformat=yseries["tickformat"],
            type="log", showgrid=True, gridcolor="rgba(255,255,255,0.12)",
        ),
        plot_bgcolor="#111", paper_bgcolor="#111",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=14),
    )
    return fig.to_json()

def fig_dma(df_all, denom="USD"):
    """DMA (50/200) figure with USD or Gold oz/BTC denomination."""
    x = log_days(df_all["Date"])
    tickvals, ticktext = year_ticks(2012, 2020, 2040)

    if denom == "USD":
        ser = df_all
        title = "USD / BTC"
        tformat = "$,d"
        traces = [
            ("BTC_200",   "200-DMA USD", "#7c3aed", 1.8),
            ("BTC_50",    "50-DMA USD",  "#60a5fa", 1.8),
            ("BTC",       "BTC USD",     COLORS["price"], 2.2),
        ]
    else:
        ser = df_all.copy()
        ser["BTC_gold"] = ser["Gold"] / ser["BTC"]
        ser["G50"]      = ser["BTC_gold"].rolling(50).mean()
        ser["G200"]     = ser["BTC_gold"].rolling(200).mean()
        ser = ser.dropna(subset=["G50","G200"])
        x = log_days(ser["Date"])
        title = "Gold oz / BTC"
        tformat = ",d"
        traces = [
            ("G200", "200-DMA Gold", "#f59e0b", 1.8),
            ("G50",  "50-DMA Gold",  "#fde047", 1.8),
            ("BTC_gold","BTC Gold",  COLORS["price"], 2.2),
        ]

    fig = go.Figure()
    for col, name, color, w in traces:
        fig.add_trace(go.Scatter(
            x=x, y=ser[col], name=name, mode="lines",
            line=dict(color=color, width=w),
            hovertemplate="%{y:.0f}<extra>"+name+"</extra>"
        ))

    fig.update_layout(
        template="plotly_dark", showlegend=False,
        margin=dict(l=60, r=20, t=10, b=60),
        xaxis=dict(
            title="Year (log-time)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False,
        ),
        yaxis=dict(
            title=title, tickformat=tformat, type="log",
            showgrid=True, gridcolor="rgba(255,255,255,0.12)",
        ),
        plot_bgcolor="#111", paper_bgcolor="#111",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=14),
    )
    return fig.to_json()

# ─────────────────────────────────────────────────────────────
# HTML template (shared)

HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://cdn.plot.ly">
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0 auto; padding: 20px 14px; max-width: 1200px;
    background:#0b0b0c; color:#f4f4f5; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  h1 {{ font-size: clamp(28px,4.5vw,46px); line-height:1.05; margin: 6px 0 18px; }}
  .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  .badge {{ display:inline-flex; align-items:center; gap:.6rem; background:#1f2937; border-radius:999px; padding:.5rem .9rem; }}
  .dot {{ width:.7rem; height:.7rem; background:#9ca3af; border-radius:999px; display:inline-block; }}
  .btn, select {{ background:#111827; color:#e5e7eb; border:1px solid #374151; border-radius:12px; padding:.55rem .9rem; }}
  .btn:hover {{ background:#0f172a; cursor:pointer; }}
  .spacer {{ height: 8px; }}
  .card {{ background:#111; border:1px solid #222; border-radius:16px; padding:8px; }}
  .full {{ width: 100%; }}
  .topbar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }}
  .subtle {{ color:#d1d5db; font-weight:600; }}
  .legend-btn {{ margin-left:auto; }}
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>

<h1>BTC Purchase Indicator</h1>

<div class="topbar">
  <div class="badge"><span class="dot"></span><span class="subtle">Current zone:</span><span id="zone">{zone}</span></div>
  {nav_html}
  <label class="subtle" for="denom" style="margin-left:8px;">Denomination:</label>
  <select id="denom">
    <option value="USD" {usd_sel}>USD</option>
    <option value="Gold" {gld_sel}>Gold</option>
  </select>
  <button id="legend" class="btn legend-btn" type="button">Legend</button>
</div>

<div class="row subtle" style="gap:10px;">
  <label for="datepick">View at date:</label>
  <input id="datepick" type="range" min="{dmin}" max="{dmax}" value="{dmax}" class="full">
</div>
<div class="spacer"></div>
<div id="date_label" class="badge">{date_label}</div>
<div class="spacer"></div>

<div id="fig" class="card full" style="min-height:520px;"></div>

<script>
const FIGS = {{
  "USD":  {fig_usd},
  "Gold": {fig_gold}
}};

let current = "{curr}";
let figDiv = document.getElementById('fig');

function renderFig() {{
  Plotly.newPlot(figDiv, FIGS[current].data, FIGS[current].layout, {{displayModeBar:true,responsive:true}});
}}

function setDenom(v) {{
  current = v;
  renderFig();
}}

function toggleLegend() {{
  let s = FIGS[current];
  let cur = s.layout.showlegend === undefined ? false : s.layout.showlegend;
  s.layout.showlegend = !cur;
  Plotly.react(figDiv, s.data, s.layout);
}}

function setDate(ts) {{
  // ts is YYYYMMDD as integer
  const y = Math.floor(ts/10000);
  const m = Math.floor((ts%10000)/100);
  const d = ts%100;
  const ds = new Date(y, m-1, d);
  const out = ds.toISOString().slice(0,10);
  document.getElementById('date_label').innerText = out;
}}

document.getElementById('denom').addEventListener('change', (e)=> setDenom(e.target.value));
document.getElementById('legend').addEventListener('click', toggleLegend);
document.getElementById('datepick').addEventListener('input', (e)=> setDate(parseInt(e.target.value)));

renderFig();
</script>

</body></html>
"""

NAV_MAIN = """<a class="btn" href="./dma.html">Open BTC USD & Gold DMA chart →</a>"""
NAV_DMA  = """<a class="btn" href="./index.html">← Back</a>"""

# ─────────────────────────────────────────────────────────────
# build all pages

def main():
    print("[build] loading prices…")
    df = load_merged()
    print(f"[build] Merged rows: {len(df):,}")

    # DMA prep
    dma = df.copy()
    dma = dma[dma["Date"] >= pd.Timestamp("2012-04-01")].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma = dma.dropna(subset=["BTC_50","BTC_200"])

    # Power-law prep
    slope, intercept0, sigma = fit_power(df)
    intercept = anchor_intercept(slope)
    df_ext = extend_to_2040(df[["Date","BTC","Gold"]])
    df_pl  = bands_df(df_ext, slope, intercept, sigma)

    # Zone for latest (USD)
    last_row = df_pl.dropna(subset=["BTC"]).iloc[-1]
    zone = zone_for(last_row["BTC"], last_row)

    # figures (as JSON)
    pl_usd  = fig_powerlaw(df_pl, denom="USD")
    pl_gold = fig_powerlaw(df_pl, denom="Gold")
    dma_usd = fig_dma(dma, denom="USD")
    dma_gold= fig_dma(dma, denom="Gold")

    # date slider bounds (YYYYMMDD)
    dmin = int(df["Date"].iloc[0].strftime("%Y%m%d"))
    dmax = int(min(PROJ_END, df_pl["Date"].iloc[-1]).strftime("%Y%m%d"))
    date_label = f"{min(PROJ_END, df_pl['Date'].iloc[-1]).date()}"

    # write index.html (power-law)
    html_pl = HTML_SHELL.format(
        title="BTC Purchase Indicator · Power-law",
        zone=zone,
        nav_html=NAV_MAIN,
        usd_sel="selected",
        gld_sel="",
        fig_usd=pl_usd,
        fig_gold=pl_gold,
        curr="USD",
        dmin=dmin, dmax=dmax, date_label=date_label,
    )
    (OUT/"index.html").write_text(html_pl, encoding="utf-8")

    # write dma.html
    html_dma = HTML_SHELL.format(
        title="BTC Purchase Indicator · DMA",
        zone=zone,
        nav_html=NAV_DMA,
        usd_sel="selected",
        gld_sel="",
        fig_usd=dma_usd,
        fig_gold=dma_gold,
        curr="USD",
        dmin=int(dma["Date"].iloc[0].strftime("%Y%m%d")),
        dmax=int(dma["Date"].iloc[-1].strftime("%Y%m%d")),
        date_label=str(dma["Date"].iloc[-1].date()),
    )
    (OUT/"dma.html").write_text(html_dma, encoding="utf-8")

    print("[build] wrote:", OUT/"index.html")
    print("[build] wrote:", OUT/"dma.html")

if __name__ == "__main__":
    main()