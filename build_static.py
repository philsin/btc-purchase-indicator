# build_static.py  —  make static pages into ./dist for GitHub Pages
# ---------------------------------------------------------------
# Pages:
#  • index.html  : Power-law bands + badge + link to DMA page
#  • dma.html    : BTC/USD + BTC/Gold 50/200 DMA (log), with top markers
#
# Data:
#  • BTC/USD daily   from Stooq (fallback: datasets mirror)
#  • Gold USD/oz     from Stooq (fallback: LBMA mirror)
#
# Notes:
#  • X axis is calendar time (date). Y axes are log.
#  • Power-law midline is anchored to ≈$491,776 on 2030-01-01.

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------- constants ----------------------------
GENESIS   = pd.Timestamp("2009-01-03")
PROJ_END  = pd.Timestamp("2040-12-31")
DMA_START = pd.Timestamp("2012-04-01")
FD_SUPPLY = 21_000_000  # for MCAP mode (not used here, but left for ref)

OUT = Path("dist")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------- data loaders -------------------------
def _btc_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date"  for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "BTC"   for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"].astype(str).replace({",": ""}, regex=True), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date", "BTC"]]

def _btc_github() -> pd.DataFrame:
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "BTC"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna()[["Date", "BTC"]].sort_values("Date")

def _gold_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"   # USD per troy ounce
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"].astype(str).replace({",": ""}, regex=True), errors="coerce")
    return df.dropna()[["Date", "Gold"]].sort_values("Date")

def _gold_lbma() -> pd.DataFrame:
    # community mirror of LBMA PM USD fix
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df[["Date", "Gold"]].dropna().sort_values("Date")

def load_prices() -> pd.DataFrame:
    # BTC
    try:
        btc = _btc_stooq()
        if len(btc) < 365:
            raise ValueError("too few btc rows")
    except Exception:
        btc = _btc_github()

    # Gold
    try:
        gold = _gold_stooq()
        if len(gold) < 365:
            raise ValueError("too few gold rows")
    except Exception:
        gold = _gold_lbma()

    df = btc.merge(gold, on="Date", how="inner")
    return df

# ---------------------------- helpers ------------------------------
def log_days(dates: pd.Series | list) -> np.ndarray:
    """log10 of days since genesis; robust to arrays/strings."""
    dd = pd.to_datetime(dates)
    # convert to float days to avoid TimedeltaIndex .dt issues in CI
    days = (dd - GENESIS) / np.timedelta64(1, "D")
    return np.log10(days.to_numpy())

def fit_power(df: pd.DataFrame):
    X = log_days(df["Date"])
    y = np.log10(df["BTC"].values)
    slope, intercept = np.polyfit(X, y, 1)
    mid_log = slope * X + intercept
    sigma   = np.std(y - mid_log)
    return slope, intercept, sigma

def anchor_intercept(slope: float) -> float:
    # force ≈ $491,776 on 2030-01-01
    anchor_date  = pd.Timestamp("2030-01-01")
    target_log   = np.log10(491_776)
    return target_log - slope * log_days([anchor_date])[0]

def year_ticks(final_year: int):
    """Return x tickvals & ticktext:
       every year 2012..2020, then every 2 years to final_year."""
    yrs = list(range(2012, min(2020, final_year) + 1)) + \
          list(range(max(2022, min(2022, final_year)), final_year + 1, 2))
    tickvals = [pd.Timestamp(f"{y}-01-01") for y in yrs]
    ticktext = [str(y) for y in yrs]
    return tickvals, ticktext

def zone_label(px, row_levels: dict) -> tuple[str, str]:
    """Return (label, color_hex) for the zone badge."""
    if px < row_levels["Support"]:
        return "SELL THE HOUSE!!", "#d9534f"
    if px < row_levels["Bear"]:
        return "Buy", "#ffb3b3"
    if px < row_levels["Frothy"]:
        return "DCA", "#ffffff"
    if px < row_levels["Top"]:
        return "Relax", "#b6f5b6"
    return "Frothy", "#2ecc71"

def html_shell(title: str, body_html: str, back_link: bool = False) -> str:
    back_html = (
        '<a class="back" href="/"><span class="arr">←</span> Back</a>'
        if back_link else ""
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg:#111; --fg:#eee; --muted:#aaa; --card:#171717; --accent:#2ecc71;
  }}
  html,body{{background:var(--bg);color:var(--fg);margin:0}}
  .wrap{{max-width:1200px;margin:24px auto;padding:0 12px}}
  h1,h2{{font-weight:600;margin:0 0 8px}}
  .badge{{display:inline-block;margin:6px 0 18px;padding:8px 12px;
          border-radius:999px;background:#222;border:1px solid #333;}}
  .badge .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;
               margin-right:8px;vertical-align:middle;background:var(--accent)}}
  .nav{{margin:14px 0 24px}}
  .nav a{{color:#9fd3ff;text-decoration:none;margin-right:16px}}
  .back{{position:fixed;top:10px;left:10px;color:#9fd3ff;text-decoration:none;
         padding:6px 10px;border-radius:8px;background:#161b22;border:1px solid #2a2f3a}}
  .arr{{font-size:14px;margin-right:6px}}
  .card{{background:var(--card);border:1px solid #262626;border-radius:14px;
         padding:14px 14px 6px;margin:18px 0}}
</style>
<body>
  {back_html}
  <div class="wrap">
    {body_html}
  </div>
</body>
</html>"""

# ---------------------------- build figs ----------------------------
def build_pages():
    data = load_prices()

    # power-law fit (anchor)
    slope, intercept, sigma = fit_power(data)
    intercept = anchor_intercept(slope)
    sigma_vis = max(sigma, 0.25)

    # extend calendar to 2040 (first day next month → PROJ_END)
    future = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                           PROJ_END, freq="MS")
    full = pd.concat([data, pd.DataFrame({"Date": future})], ignore_index=True)

    # power-law levels
    levels = {
        "Support":     -1.5,
        "Bear":        -0.5,
        "PL Best Fit":  0.0,
        "Frothy":      +1.0,
        "Top":         +1.75,
    }
    colors = {
        "Support":     "red",
        "Bear":        "rgba(255,100,100,1)",
        "PL Best Fit": "white",
        "Frothy":      "rgba(100,255,100,1)",
        "Top":         "green",
    }

    # compute lines
    mid_log = slope * log_days(full["Date"]) + intercept
    for name, k in levels.items():
        full[name] = 10 ** (mid_log + sigma_vis * k)

    # ---------------- Power-law page (index.html) ----------------
    tickvals, ticktext = year_ticks(PROJ_END.year)

    # zone
    last = full.dropna(subset=["BTC"]).iloc[-1]
    zone, zcolor = zone_label(last["BTC"], {
        "Support": last["Support"],
        "Bear":    last["Bear"],
        "Frothy":  last["Frothy"],
        "Top":     last["Top"],
    })

    fig1 = go.Figure(layout=dict(
        template="plotly_dark",
        xaxis=dict(type="date", title="Year",
                   tickvals=tickvals, ticktext=ticktext, tickangle=0,
                   showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    ))

    for name in ["Top", "Frothy"]:
        fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                  name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                                  line=dict(color=colors[name], dash="dash")))
    fig1.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                              name="PL Best Fit", line=dict(color="white", dash="dash")))
    for name in ["Bear", "Support"]:
        fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                                  name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                                  line=dict(color=colors[name], dash="dash")))
    fig1.add_trace(go.Scatter(x=data["Date"], y=data["BTC"],
                              name="BTC", line=dict(color="gold", width=2)))

    badge = f'<span class="badge"><span class="dot" style="background:{zcolor}"></span>Current zone: <b>{zone}</b></span>'
    nav   = '<div class="nav"><a href="/dma.html">Open BTC USD & Gold DMA chart →</a></div>'

    html1 = html_shell(
        "BTC Purchase Indicator — Power law",
        f'<h1>BTC Purchase Indicator</h1>{badge}{nav}'
        f'<div class="card">{fig1.to_html(include_plotlyjs="cdn", full_html=False)}</div>'
    )
    (OUT / "index.html").write_text(html1, encoding="utf-8")

    # ---------------- DMA page (dma.html) ----------------
    dma = data[data["Date"] >= DMA_START].copy()
    dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
    dma["BTC_200"] = dma["BTC"].rolling(200).mean()
    dma["BTCG"]    = dma["BTC"] / dma["Gold"]
    dma["G50"]     = dma["BTCG"].rolling(50).mean()
    dma["G200"]    = dma["BTCG"].rolling(200).mean()
    dma = dma.dropna()

    # top markers with 100-day rule (USD)
    diff_u = dma["BTC_200"] - dma["BTC_50"]
    pos_u  = diff_u > 0
    # streak of consecutive True
    grp_u  = (pos_u != pos_u.shift()).cumsum()
    streak_u = pos_u.groupby(grp_u).cumcount() + 1
    cond_u = pos_u.shift(1).fillna(False) & (streak_u.shift(1).fillna(0) >= 100) & (diff_u < 0)
    cross_u_dates  = dma.loc[cond_u, "Date"]
    cross_u_vals   = dma.loc[cond_u, "BTC"]

    # top markers with 100-day rule (Gold)
    diff_g = dma["G200"] - dma["G50"]
    pos_g  = diff_g > 0
    grp_g  = (pos_g != pos_g.shift()).cumsum()
    streak_g = pos_g.groupby(grp_g).cumcount() + 1
    cond_g = pos_g.shift(1).fillna(False) & (streak_g.shift(1).fillna(0) >= 100) & (diff_g < 0)
    cross_g_dates = dma.loc[cond_g, "Date"]
    cross_g_vals  = dma.loc[cond_g, "BTCG"]

    # y2 (gold) ticks — log, readable
    def log_ticks(series: pd.Series):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        lo = int(np.floor(np.log10(s.min())))
        hi = int(np.ceil (np.log10(s.max())))
        vals = [10 ** k for k in range(lo, hi + 1)]
        return vals, [f"{v:g}" for v in vals]

    y2_vals, y2_text = log_ticks(dma["BTCG"])

    fig2 = go.Figure(layout=dict(
        template="plotly_dark",
        xaxis=dict(type="date", title="Year",
                   tickvals=tickvals, ticktext=ticktext, tickangle=0,
                   showgrid=True, gridwidth=0.5),
        yaxis=dict(type="log", title="BTC Price (USD)", tickformat="$,d",
                   showgrid=True, gridwidth=0.5),
        yaxis2=dict(type="log", title="BTC Price (oz Gold)",
                    tickvals=y2_vals, ticktext=y2_text,
                    overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    ))

    # USD color scheme (soft→bright green)
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"],
                              name="200-DMA USD", line=dict(color="seagreen", width=1.5)))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],
                              name="50-DMA USD",  line=dict(color="palegreen", width=1.5)))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],
                              name="BTC USD",     line=dict(color="limegreen", width=2)))
    fig2.add_trace(go.Scatter(x=cross_u_dates, y=cross_u_vals, mode="markers",
                              name="Top Marker (USD)",
                              marker=dict(symbol="diamond", color="darkgreen", size=9)))

    # Gold color scheme (soft→bright gold)
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"], yaxis="y2",
                              name="200-DMA Gold", line=dict(color="goldenrod", width=1.5)))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],  yaxis="y2",
                              name="50-DMA Gold",  line=dict(color="khaki", width=1.5)))
    fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"], yaxis="y2",
                              name="BTC Gold",     line=dict(color="gold", width=2)))
    fig2.add_trace(go.Scatter(x=cross_g_dates, y=cross_g_vals, mode="markers", yaxis="y2",
                              name="Top Marker (Gold)",
                              marker=dict(symbol="diamond", color="darkgoldenrod", size=9)))

    html2 = html_shell(
        "BTC Purchase Indicator — DMA",
        "<h1>BTC USD & Gold — 50/200 DMA</h1>"
        '<p class="muted">Dual log axes; right axis shows BTC priced in troy ounces of gold.</p>'
        f'<div class="card">{fig2.to_html(include_plotlyjs="cdn", full_html=False)}</div>',
        back_link=True
    )
    (OUT / "dma.html").write_text(html2, encoding="utf-8")

# ---------------------------- main ----------------------------
if __name__ == "__main__":
    build_pages()
    print(f"Wrote: {(OUT / 'index.html').resolve()}")
    print(f"Wrote: {(OUT / 'dma.html').resolve()}")