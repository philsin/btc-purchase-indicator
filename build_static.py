# build_static.py  —  build static site into ./dist for GitHub Pages
# Pages:
#   /index.html  : main menu (auto-redirect to favorite)
#   /power.html  : power-law bands + badge
#   /dma.html    : BTC/USD & BTC/Gold 50/200 DMA (dual log) + top markers
#
# IMPORTANT: set BASE for a project page.
#   If your site is https://philsin.github.io/btc-purchase-indicator/
#   leave BASE = "/btc-purchase-indicator"
#   If you publish to a *user/organization* site root (https://philsin.github.io/)
#   set BASE = "".

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------- settings ----------------
BASE      = "/btc-purchase-indicator"   # <— adjust only if you publish at site root
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")

OUT = Path("dist")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- data loaders ----------------
def _btc_stooq():
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "BTC"  for c in df.columns if ("close" in c) or ("price" in c)})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).replace({",": ""}, regex=True), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _btc_github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna()[["Date","BTC"]].sort_values("Date")

def _gold_stooq():
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"   # USD/oz
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if ("close" in c) or ("price" in c)})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).replace({",": ""}, regex=True), errors="coerce")
    return df.dropna()[["Date","Gold"]].sort_values("Date")

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df[["Date","Gold"]].dropna().sort_values("Date")

def load_prices():
    try:
        btc = _btc_stooq()
        if len(btc) < 365:
            raise ValueError
    except Exception:
        btc = _btc_github()

    try:
        gold = _gold_stooq()
        if len(gold) < 365:
            raise ValueError
    except Exception:
        gold = _gold_lbma()

    return btc.merge(gold, on="Date", how="inner")

# ---------------- math helpers ----------------
def log_days(dates):
    dd = pd.to_datetime(dates)
    days = (dd - GENESIS) / np.timedelta64(1, "D")
    return np.log10(days.to_numpy())

def fit_power(df):
    X = log_days(df["Date"])
    y = np.log10(df["BTC"].values)
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    sigma   = np.std(y - mid_log)
    return slope, intercept, sigma

def anchor_intercept(slope):
    anchor_date = pd.Timestamp("2030-01-01")
    target_log  = np.log10(491_776)  # ≈$500k
    return target_log - slope * log_days([anchor_date])[0]

def year_ticks(final_year: int):
    yrs = list(range(2012, min(2020, final_year)+1)) + \
          list(range(max(2022, min(2022, final_year)), final_year+1, 2))
    tickvals = [pd.Timestamp(f"{y}-01-01") for y in yrs]
    ticktext = [str(y) for y in yrs]
    return tickvals, ticktext

def zone_label(px, row_levels):
    if px < row_levels["Support"]:
        return "SELL THE HOUSE!!", "#d9534f"
    if px < row_levels["Bear"]:
        return "Buy", "#ffb3b3"
    if px < row_levels["Frothy"]:
        return "DCA", "#ffffff"
    if px < row_levels["Top"]:
        return "Relax", "#b6f5b6"
    return "Frothy", "#2ecc71"

# ---------------- HTML shell & widgets ----------------
def css():
    return f"""
<style>
  :root {{ --bg:#111; --fg:#eee; --muted:#9aa0a6; --card:#171717; --border:#262626; }}
  * {{ box-sizing: border-box; }}
  html,body{{ background:var(--bg); color:var(--fg); margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,Apple Color Emoji,Segoe UI Emoji; }}
  .wrap{{ max-width:1200px; margin:24px auto; padding:0 12px }}
  h1{{ font-size:40px; margin:0 0 6px; font-weight:700 }}
  .card{{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px 14px 6px; margin:18px 0 }}
  .nav a{{ color:#9fd3ff; text-decoration:none; margin-right:16px }}
  .badge{{ display:inline-block; padding:8px 12px; border-radius:999px; background:#202020; border:1px solid #333; margin:8px 0 16px }}
  .badge .dot{{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle }}
  .back{{ position:fixed; top:10px; left:10px; color:#9fd3ff; text-decoration:none; padding:6px 10px; border-radius:8px; background:#161b22; border:1px solid #2a2f3a }}
  .fav{{ position:fixed; top:10px; right:10px; cursor:pointer; user-select:none; font-size:18px; padding:6px 10px; border-radius:8px; background:#161b22; border:1px solid #2a2f3a }}
  .fav.on{{ color:#ffd54d }}
  .menu a.card{{ display:block; color:inherit; text-decoration:none }}
  .row{{ display:grid; grid-template-columns:1fr; gap:12px }}
  @media (min-width:720px){{ .row{{ grid-template-columns:1fr 1fr }} }}
</style>
"""

def shell(title: str, body: str, back_href: str | None, fav_path: str | None):
    back_html = f'<a class="back" href="{back_href}">← Back</a>' if back_href else ""
    fav_html = ""
    if fav_path:
        # Use doubled braces to emit literal { } in the JS
        fav_html = """
        <div id="fav" class="fav">☆ Favorite</div>
        <script>
          const key='btcpi_favorite';
          const favEl=document.getElementById('fav');
          function update(){{
            const val=localStorage.getItem(key);
            if(val==='{{FAV}}'){{ favEl.classList.add('on'); favEl.textContent='★ Favorite'; }}
            else{{ favEl.classList.remove('on'); favEl.textContent='☆ Favorite'; }}
          }}
          favEl.onclick=()=>{{
            const val=localStorage.getItem(key);
            if(val==='{{FAV}}') localStorage.removeItem(key);
            else localStorage.setItem(key,'{{FAV}}');
            update();
          }};
          update();
        </script>
        """.replace("{{FAV}}", fav_path)

    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
{css()}
<body>
  {back_html}{fav_html}
  <div class="wrap">
    {body}
  </div>
</body></html>"""

def write_html(name: str, html: str):
    p = OUT / name
    p.write_text(html, encoding="utf-8")
    print("wrote", p)

# ---------------- figure builders ----------------
def build_power_page(data: pd.DataFrame):
    slope, intercept, sigma = fit_power(data)
    intercept = anchor_intercept(slope)
    sigma_vis = max(sigma, 0.25)

    # extend to 2040 (calendar months)
    future = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                           PROJ_END, freq="MS")
    full = pd.concat([data, pd.DataFrame({"Date": future})], ignore_index=True)

    levels = {
        "Support": -1.5, "Bear": -0.5, "PL Best Fit": 0.0, "Frothy": 1.0, "Top": 1.75,
    }
    colors = {
        "Support": "red",
        "Bear": "rgba(255,100,100,1)",
        "PL Best Fit": "white",
        "Frothy": "rgba(100,255,100,1)",
        "Top": "green",
    }

    mid_log = slope * log_days(full["Date"]) + intercept
    for name, k in levels.items():
        full[name] = 10 ** (mid_log + sigma_vis * k)

    tickvals, ticktext = year_ticks(PROJ_END.year)

    last = full.dropna(subset=["BTC"]).iloc[-1]
    zone, zcol = zone_label(last["BTC"], {
        "Support": last["Support"], "Bear": last["Bear"],
        "Frothy": last["Frothy"], "Top": last["Top"],
    })

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        xaxis=dict(type="date", title="Year",
                   tickvals=tickvals, ticktext=ticktext, showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    ))

    for name in ["Top", "Frothy"]:
        fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                 name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                                 line=dict(color=colors[name], dash="dash")))
    fig.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                             name="PL Best Fit", line=dict(color="white", dash="dash")))
    for name in ["Bear", "Support"]:
        fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                 name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                                 line=dict(color=colors[name], dash="dash")))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["BTC"],
                             name="BTC", line=dict(color="gold", width=2)))

    badge = f'<span class="badge"><span class="dot" style="background:{zcol}"></span>Current zone: <b>{zone}</b></span>'
    body = (
        f"<h1>BTC Purchase Indicator</h1>"
        f"{badge}"
        f'<div class="card">{fig.to_html(include_plotlyjs="cdn", full_html=False)}</div>'
        f'<div class="nav"><a href="{BASE}/dma.html">Open BTC USD & Gold DMA chart →</a></div>'
    )
    html = shell("BTC Purchase Indicator — Power law",
                 body, back_href=f"{BASE}/", fav_path=f"{BASE}/power.html")
    write_html("power.html", html)

def build_dma_page(data: pd.DataFrame):
    dma = data[data["Date"] >= DMA_START].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma["BTCG"]    = dma["BTC"] / dma["Gold"]
    dma["G50"]     = dma["BTCG"].rolling(50).mean()
    dma["G200"]    = dma["BTCG"].rolling(200).mean()
    dma = dma.dropna()

    # top markers (USD) requiring 100-day 200>50 beforehand
    diff_u = dma["BTC_200"] - dma["BTC_50"]
    pos_u  = diff_u > 0
    grp_u  = (pos_u != pos_u.shift()).cumsum()
    streak_u = pos_u.groupby(grp_u).cumcount() + 1
    cond_u = pos_u.shift(1).fillna(False) & (streak_u.shift(1).fillna(0) >= 100) & (diff_u < 0)
    x_u, y_u = dma.loc[cond_u, "Date"], dma.loc[cond_u, "BTC"]

    # top markers (Gold) with same rule
    diff_g = dma["G200"] - dma["G50"]
    pos_g  = diff_g > 0
    grp_g  = (pos_g != pos_g.shift()).cumsum()
    streak_g = pos_g.groupby(grp_g).cumcount() + 1
    cond_g = pos_g.shift(1).fillna(False) & (streak_g.shift(1).fillna(0) >= 100) & (diff_g < 0)
    x_g, y_g = dma.loc[cond_g, "Date"], dma.loc[cond_g, "BTCG"]

    # right-axis (gold) log ticks
    def log_ticks(series: pd.Series):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        lo = int(np.floor(np.log10(s.min())))
        hi = int(np.ceil (np.log10(s.max())))
        vals = [10 ** k for k in range(lo, hi + 1)]
        return vals, [f"{v:g}" for v in vals]

    tickvals, ticktext = year_ticks(PROJ_END.year)
    y2_vals, y2_text = log_ticks(dma["BTCG"])

    fig = go.Figure(layout=dict(
        template="plotly_dark",
        xaxis=dict(type="date", title="Year",
                   tickvals=tickvals, ticktext=ticktext, showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="BTC Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        yaxis2=dict(type="log", title="BTC Price (oz Gold)",
                    tickvals=y2_vals, ticktext=y2_text,
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    ))

    # USD: soft→bright green
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"],
                             name="200-DMA USD", line=dict(color="seagreen", width=1.5)))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],
                             name="50-DMA USD",  line=dict(color="palegreen", width=1.5)))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],
                             name="BTC USD",     line=dict(color="limegreen", width=2)))
    fig.add_trace(go.Scatter(x=x_u, y=y_u, mode="markers",
                             name="Top Marker (USD)",
                             marker=dict(symbol="diamond", color="darkgreen", size=9)))

    # Gold: soft→bright gold (right axis)
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"], yaxis="y2",
                             name="200-DMA Gold", line=dict(color="goldenrod", width=1.5)))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],  yaxis="y2",
                             name="50-DMA Gold",  line=dict(color="khaki", width=1.5)))
    fig.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"], yaxis="y2",
                             name="BTC Gold",     line=dict(color="gold", width=2)))
    fig.add_trace(go.Scatter(x=x_g, y=y_g, yaxis="y2", mode="markers",
                             name="Top Marker (Gold)",
                             marker=dict(symbol="diamond", color="darkgoldenrod", size=9)))

    body = (
        f"<h1>BTC USD & Gold — 50/200 DMA</h1>"
        f'<div class="card">{fig.to_html(include_plotlyjs="cdn", full_html=False)}</div>'
    )
    html = shell("BTC Purchase Indicator — DMA",
                 body, back_href=f"{BASE}/", fav_path=f"{BASE}/dma.html")
    write_html("dma.html", html)

def build_menu_page():
    # A small index that auto-redirects to favorite if set
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Purchase Indicator</title>
{css()}
<body>
  <div class="wrap">
    <h1>BTC Purchase Indicator</h1>
    <p class="nav">Choose a chart below. Click ★ to set your favorite.</p>
    <div class="row menu">
      <a class="card" href="{BASE}/power.html">
        <b>Power-law Bands</b>
        <div style="color:var(--muted)">Long-term power-law fit with support/resistance bands.</div>
      </a>
      <a class="card" href="{BASE}/dma.html">
        <b>BTC USD & Gold — 50/200 DMA</b>
        <div style="color:var(--muted)">Dual-axis log: USD (left) and BTC/Gold (right), with top markers.</div>
      </a>
    </div>
  </div>
  <script>
    const key='btcpi_favorite';
    const fav = localStorage.getItem(key);
    if(fav) window.location.replace(fav);
  </script>
</body></html>"""
    write_html("index.html", html)

# ---------------- main ----------------
if __name__ == "__main__":
    data = load_prices()
    build_menu_page()
    build_power_page(data)
    build_dma_page(data)
    print("Site written to ./dist")