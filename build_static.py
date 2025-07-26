# build_static.py
# BTC Purchase Indicator — static site generator for GitHub Pages
# Pages:
#   - index.html (launcher with favorites)
#   - powerlaw.html (power-law bands; USD vs Gold toggle + legend control)
#   - dma.html (BTC/USD & BTC/Gold with 50/200 DMA; USD vs Gold toggle + legend control)
#
# Run locally:   python build_static.py
# GitHub Action: uses this file and publishes ./dist to Pages.

import os, io, math, json, pathlib, textwrap
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.io import to_html

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
UA          = {"User-Agent": "btc-pl-pages/1.0"}
GENESIS     = pd.Timestamp("2009-01-03")
FD_SUPPLY   = 21_000_000
PROJ_END    = pd.Timestamp("2040-12-31")
DMA_START   = pd.Timestamp("2012-04-01")
GRID_D      = "M24"   # 2-year visual cadence (we’ll override with custom ticks)

DIST = pathlib.Path("dist")
DIST.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def year_ticks(start_year: int, end_year: int):
    """Return tickvals/ticktext so labels are yearly up to 2020 then every other year."""
    yrs = list(range(start_year, min(2020, end_year) + 1))
    if end_year >= 2022:
        yrs += list(range(2022, end_year + 1, 2))
    tickvals = pd.to_datetime([f"{y}-01-01" for y in yrs])
    ticktext = [str(y) for y in yrs]
    return tickvals, ticktext

def write_page(filename: str, html: str):
    out = DIST / filename
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")

def shell(title: str, body_html: str, *, back_href: str | None = None,
          fav_path: str | None = None) -> str:
    """
    Basic dark layout, optional back link, and favorites star using localStorage.
    If fav_path is provided, the star toggles this page as a favorite.
    """
    back = ""
    if back_href:
        back = f"""
        <a class="back" href="{back_href}" aria-label="Back">←</a>
        """

    fav = ""
    fav_js = ""
    if fav_path:
        fav = f"""
        <button id="favBtn" class="fav" title="Toggle favorite" aria-label="Favorite">☆</button>
        """
        fav_js = f"""
        <script>
          const favKey = 'btcpi.favorite';
          const favBtn = document.getElementById('favBtn');
          function setStar(on) {{
            favBtn.textContent = on ? '★' : '☆';
            favBtn.classList.toggle('on', on);
          }}
          function isFav() {{
            try {{ return localStorage.getItem(favKey) === '{fav_path}'; }}
            catch(e) {{ return false; }}
          }}
          setStar(isFav());
          favBtn.onclick = () => {{
            const on = !isFav();
            try {{
              if (on) localStorage.setItem(favKey, '{fav_path}');
              else localStorage.removeItem(favKey);
            }} catch(e) {{}}
            setStar(on);
          }};
        </script>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://cdn.plot.ly">
<style>
  :root {{
    --bg:#0f0f12; --panel:#17171b; --text:#f1f3f5; --muted:#a1a1aa; --accent:#74c0fc;
    --pill:#1f2430; --green:#20c997; --red:#ff6b6b;
  }}
  html,body {{ background:var(--bg); color:var(--text); margin:0; font:400 16px/1.6 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; }}
  .wrap {{ max-width:1000px; margin:18px auto 64px; padding:0 14px; }}
  .hdr {{ display:flex; align-items:center; gap:.5rem; }}
  .back {{
    text-decoration:none; color:var(--text); background:var(--panel); border-radius:10px; padding:.2rem .55rem; font-size:1.35rem;
    display:inline-flex; align-items:center; justify-content:center;
  }}
  h1 {{ margin:.2rem 0 1rem; font-weight:700; letter-spacing:.2px; }}
  .fav {{
    background:var(--panel); border:none; color:#ffd43b; font-size:1.35rem; line-height:1;
    padding:.35rem .55rem; border-radius:10px; cursor:pointer; margin-left:auto;
  }}
  .fav.on {{ filter: drop-shadow(0 0 6px #ffd43b88); }}
  .pill {{
    display:inline-flex; align-items:center; gap:.5rem; background:var(--pill); color:var(--text);
    border-radius:999px; padding:.45rem .9rem; margin:.4rem 0 1rem; font-weight:600;
  }}
  .dot {{ width:.8rem; height:.8rem; border-radius:50%; background:#e9ecef; display:inline-block; }}
  .card {{ background:var(--panel); border-radius:16px; padding:12px; }}
  .toolbar {{ display:flex; gap:.75rem; align-items:center; margin:.25rem 0 .75rem; flex-wrap:wrap; font-size:.95rem; }}
  .chk input {{ transform: translateY(1px); }}
  .chk span {{ margin-left:.25rem; }}
  .hint {{ opacity:.65; }}
  a.ln {{ color:var(--accent); text-decoration:none; }}
  a.ln:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    {back}
    <h1>{title}</h1>
    {fav}
  </div>
  {body_html}
</div>
{fav_js}
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def _btc_stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d", headers=UA)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "BTC" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw, headers=UA).rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date","BTC"]]

def load_btc():
    try:
        df = _btc_stooq()
        if len(df) > 1000:
            return df
    except Exception:
        pass
    return _btc_github()

def _gold_stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d", headers=UA)  # USD/oz
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().sort_values("Date")

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url, headers=UA)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df[["Date","Gold"]].dropna()

def load_gold():
    try:
        g = _gold_stooq()
        if len(g) >= 1000:
            return g
        raise ValueError("short gold series")
    except Exception:
        return _gold_lbma()

def load_joined():
    b = load_btc()
    g = load_gold()
    df = b.merge(g, on="Date", how="inner")
    return df

# ─────────────────────────────────────────────────────────────
# Power-law fit & bands
# ─────────────────────────────────────────────────────────────
def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days)
    y = np.log10(df["BTC"])
    slope, intercept = np.polyfit(X, y, 1)
    sigma = np.std(y - (slope*X + intercept))
    return slope, intercept, sigma

def compute_powerlaw(df):
    slope, intercept, sigma = fit_power(df)

    # Anchor: ~$500k on 2030-01-01
    anchor_date = pd.Timestamp("2030-01-01")
    intercept = np.log10(491_776) - slope * np.log10((anchor_date - GENESIS).days)

    # Extend timeline to 2040 month-start
    future = pd.date_range(df["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                           PROJ_END, freq="MS", inclusive="both")
    full = pd.concat([df, pd.DataFrame({"Date": future})], ignore_index=True)

    days    = (full["Date"] - GENESIS).dt.days
    mid_log = slope * np.log10(days) + intercept
    σ_vis   = max(sigma, 0.25)

    levels = {
        "Support":     -1.5,
        "Bear":        -0.5,
        "PL Best Fit":  0.0,
        "Frothy":      +1.0,
        "Top":         +1.75,
    }
    for name, k in levels.items():
        full[name] = 10 ** (mid_log + σ_vis * k)

    # Zone (USD)
    last = full.dropna(subset=["BTC"]).iloc[-1]
    p = last["BTC"]
    if p < last["Support"]:
        zone = "SELL THE HOUSE!!"
    elif p < last["Bear"]:
        zone = "Buy"
    elif p < last["Frothy"]:
        zone = "DCA"
    elif p < last["Top"]:
        zone = "Relax"
    else:
        zone = "Frothy"

    return full, levels, zone

# ─────────────────────────────────────────────────────────────
# Figures (with USD/Gold toggle logic)
# ─────────────────────────────────────────────────────────────
def usd_gold_toggle_script(div_id: str, usd_idx: list[int], gold_idx: list[int], usd_on=True, gold_on=False) -> str:
    """Return JS+controls to toggle legend groups."""
    return f"""
<div class="toolbar">
  <label class="chk"><input id="{div_id}-usd" type="checkbox" {'checked' if usd_on else ''}> <span>USD</span></label>
  <label class="chk"><input id="{div_id}-gld" type="checkbox" {'checked' if gold_on else ''}> <span>Gold</span></label>
  <span class="hint">Tip: you can still click legend items to hide/show individual lines.</span>
</div>
<script>
  const gd_{div_id} = document.getElementById('{div_id}');
  const usdIdx_{div_id}  = {usd_idx};
  const gldIdx_{div_id}  = {gold_idx};
  function setGroup_{div_id}(idxs, on) {{
    const vis = on ? true : 'legendonly';
    for (const i of idxs) {{ Plotly.restyle(gd_{div_id}, {{visible: vis}}, [i]); }}
  }}
  // initialize
  setGroup_{div_id}(usdIdx_{div_id}, {'true' if usd_on else 'false'});
  setGroup_{div_id}(gldIdx_{div_id}, {'true' if gold_on else 'false'});
  document.getElementById('{div_id}-usd').onchange = (e)=> setGroup_{div_id}(usdIdx_{div_id}, e.target.checked);
  document.getElementById('{div_id}-gld').onchange = (e)=> setGroup_{div_id}(gldIdx_{div_id}, e.target.checked);
</script>
"""

def powerlaw_page_html(full: pd.DataFrame, levels: dict, zone: str) -> str:
    # Build figure
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="date", title="Year", showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        yaxis2=dict(type="log", title="BTC Price (oz Gold)", tickformat=",d",
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    start_y = max(2012, int(full["Date"].min().year))
    end_y   = int(min(PROJ_END.year, full["Date"].max().year))
    tv, tt  = year_ticks(start_y, end_y)
    fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)

    # USD lines (legendgroup='USD')
    colors = {
        "Support": "#ff6b6b",
        "Bear": "rgba(255,100,100,1)",
        "PL Best Fit": "#ffffff",
        "Frothy": "#8df08d",
        "Top": "#1eb31e",
    }
    for name in ["Top", "Frothy"]:
        fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                 name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                                 line=dict(color=colors[name], dash="dash"),
                                 legendgroup="USD"))
    fig.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                             name="PL Best Fit",
                             line=dict(color=colors["PL Best Fit"], dash="dash"),
                             legendgroup="USD"))
    for name in ["Bear", "Support"]:
        fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                 name=f"{name} ({levels[name]:+.2f}σ)",
                                 line=dict(color=colors[name], dash="dash"),
                                 legendgroup="USD"))
    fig.add_trace(go.Scatter(x=full.dropna(subset=["BTC"])["Date"],
                             y=full.dropna(subset=["BTC"])["BTC"],
                             name="BTC USD", line=dict(color="#ffd54a", width=2.2),
                             legendgroup="USD"))

    # GOLD line (legendgroup='GOLD') — ratio on y2
    if "Gold" in full.columns:
        btcg = (full["BTC"] / full["Gold"]).replace([np.inf, -np.inf], np.nan).dropna()
        dates = full.loc[btcg.index, "Date"]
        fig.add_trace(go.Scatter(x=dates, y=btcg, name="BTC Gold",
                                 line=dict(color="#ffd700", width=2),
                                 legendgroup="GOLD", yaxis="y2"))

    # Convert figure
    div_id = "plChart"
    fig_html = to_html(fig, include_plotlyjs="cdn", full_html=False,
                       config={"responsive": True, "displaylogo": False}, div_id=div_id)

    # Determine trace indices for groups
    usd_idx  = [i for i, t in enumerate(fig.data) if getattr(t, "legendgroup", "") == "USD"]
    gold_idx = [i for i, t in enumerate(fig.data) if getattr(t, "legendgroup", "") == "GOLD"]

    badge = f"""
    <div class="pill"><span class="dot"></span> <span><strong>Current zone:</strong> {zone}</span></div>
    """

    return f"""
    {badge}
    {usd_gold_toggle_script(div_id, usd_idx, gold_idx, usd_on=True, gold_on=False)}
    <div class="card">{fig_html}</div>
    """

def dma_dataframe(joined: pd.DataFrame) -> pd.DataFrame:
    dma = joined[joined["Date"] >= DMA_START].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma["BTCG"]    = dma["BTC"] / dma["Gold"]
    dma["G50"]     = dma["BTCG"].rolling(50).mean()
    dma["G200"]    = dma["BTCG"].rolling(200).mean()
    dma = dma.dropna()

    # USD Top Marker: 200DMA crosses down below 50DMA, preceded by >100 days with 200>50
    diff_u = dma["BTC_200"] - dma["BTC_50"]
    prev_pos = (diff_u.shift(1) > 0)
    streak100 = prev_pos.rolling(100).apply(lambda x: float((x > 0.5).all()), raw=False).astype(bool)
    dma["_usd_cross"] = (prev_pos & (diff_u < 0) & streak100)

    # GOLD Top Marker: same logic on gold DMAs
    diff_g = dma["G200"] - dma["G50"]
    prev_pos_g = (diff_g.shift(1) > 0)
    streak100g = prev_pos_g.rolling(100).apply(lambda x: float((x > 0.5).all()), raw=False).astype(bool)
    dma["_gold_cross"] = (prev_pos_g & (diff_g < 0) & streak100g)

    return dma

def dma_page_html(dma: pd.DataFrame) -> str:
    fig2 = go.Figure(layout=dict(
        template="plotly_dark",
        font=dict(family="Currency, monospace", size=12),
        xaxis=dict(type="date", title="Year", showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="BTC Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        yaxis2=dict(type="log", title="BTC Price (oz Gold)", tickformat=",d",
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    start_y = 2012
    end_y   = int(min(PROJ_END.year, dma["Date"].max().year))
    tv, tt  = year_ticks(start_y, end_y)
    fig2.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)

    # USD group
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],
                              name="BTC USD", line=dict(color="#00b050", width=2.2),
                              legendgroup="USD"))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],
                              name="50-DMA USD", line=dict(color="#7bdc8e", width=1.6),
                              legendgroup="USD"))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"],
                              name="200-DMA USD", line=dict(color="#2a7f43", width=1.6),
                              legendgroup="USD"))
    fig2.add_trace(go.Scatter(x=dma.loc[dma["_usd_cross"],"Date"],
                              y=dma.loc[dma["_usd_cross"],"BTC"],
                              name="Top Marker (USD)", mode="markers",
                              marker=dict(symbol="diamond", color="#20c997", size=9,
                                          line=dict(color="#094e3b", width=1)),
                              legendgroup="USD"))

    # GOLD group on y2
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"],
                              name="BTC Gold", line=dict(color="#ffd700", width=2.2),
                              legendgroup="GOLD", yaxis="y2"))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],
                              name="50-DMA Gold",  line=dict(color="#ffe680", width=1.6),
                              legendgroup="GOLD", yaxis="y2"))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"],
                              name="200-DMA Gold", line=dict(color="#bfa100", width=1.6),
                              legendgroup="GOLD", yaxis="y2"))
    fig2.add_trace(go.Scatter(x=dma.loc[dma["_gold_cross"],"Date"],
                              y=dma.loc[dma["_gold_cross"],"BTCG"],
                              name="Top Marker (Gold)", mode="markers",
                              marker=dict(symbol="diamond", color="#ff9900", size=9,
                                          line=dict(color="#8a5a00", width=1)),
                              legendgroup="GOLD", yaxis="y2"))

    # Render
    div_id = "dmaChart"
    fig_html = to_html(fig2, include_plotlyjs="cdn", full_html=False,
                       config={"responsive": True, "displaylogo": False}, div_id=div_id)

    usd_idx  = [i for i, t in enumerate(fig2.data) if getattr(t, "legendgroup", "") == "USD"]
    gold_idx = [i for i, t in enumerate(fig2.data) if getattr(t, "legendgroup", "") == "GOLD"]

    controls = usd_gold_toggle_script(div_id, usd_idx, gold_idx, usd_on=True, gold_on=False)
    return f"""{controls}<div class="card">{fig_html}</div>"""

# ─────────────────────────────────────────────────────────────
# Index page (launcher with favorites + autoroute to favorite)
# ─────────────────────────────────────────────────────────────
def index_html(zone: str) -> str:
    return f"""
<div class="pill"><span class="dot"></span> <span><strong>Current zone:</strong> {zone}</span></div>

<ul style="list-style:none; padding:0; margin:0; display:grid; gap:.75rem;">
  <li class="card" style="padding:14px;">
    <div style="display:flex; align-items:center; gap:.5rem;">
      <a class="ln" href="powerlaw.html">Open Power-law bands →</a>
      <button id="fav_pl" class="fav" title="Favorite Power-law">☆</button>
    </div>
  </li>
  <li class="card" style="padding:14px;">
    <div style="display:flex; align-items:center; gap:.5rem;">
      <a class="ln" href="dma.html">Open BTC USD & Gold DMA chart →</a>
      <button id="fav_dma" class="fav" title="Favorite DMA">☆</button>
    </div>
  </li>
</ul>

<script>
  const favKey = 'btcpi.favorite';
  function setBtn(btn, on) {{ btn.textContent = on ? '★' : '☆'; btn.classList.toggle('on', on); }}
  function bindStar(btnId, pagePath) {{
    const btn = document.getElementById(btnId);
    const on = (localStorage.getItem(favKey) === pagePath);
    setBtn(btn, on);
    btn.onclick = () => {{
      const nowOn = !(localStorage.getItem(favKey) === pagePath);
      if (nowOn) localStorage.setItem(favKey, pagePath);
      else localStorage.removeItem(favKey);
      setBtn(btn, nowOn);
    }};
  }}
  bindStar('fav_pl',  'powerlaw.html');
  bindStar('fav_dma', 'dma.html');

  // If a favorite is set and we're on index, redirect immediately.
  const fav = localStorage.getItem(favKey);
  if (fav) {{
    // Comment out the next line if you prefer to always land on index.
    // location.href = fav;
  }}
</script>
"""

# ─────────────────────────────────────────────────────────────
# Build all pages
# ─────────────────────────────────────────────────────────────
def main():
    joined = load_joined()
    full, levels, zone = compute_powerlaw(joined)
    dma   = dma_dataframe(joined)

    # Index
    write_page("index.html", shell("BTC Purchase Indicator",
                                   index_html(zone),
                                   back_href=None, fav_path=None))

    # Power-law page (with USD/Gold toggle)
    pl_html = powerlaw_page_html(full, levels, zone)
    write_page("powerlaw.html",
               shell("BTC Power-law Bands",
                     pl_html, back_href="index.html", fav_path="powerlaw.html"))

    # DMA page (with USD/Gold toggle)
    dma_html = dma_page_html(dma)
    write_page("dma.html",
               shell("BTC USD & Gold DMA",
                     dma_html, back_href="index.html", fav_path="dma.html"))

if __name__ == "__main__":
    main()