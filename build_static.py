import os, pandas as pd, numpy as np, plotly.graph_objects as go

# ───────── Settings ─────────
GENESIS        = pd.Timestamp("2009-01-03")
PROJ_END       = pd.Timestamp("2040-12-31")
DMA_START      = pd.Timestamp("2012-04-01")
ANCHOR_DATE    = pd.Timestamp("2030-01-01")
ANCHOR_PRICE_USD = 491_776         # mid-line target on ANCHOR_DATE (USD)
OUTDIR         = "public"
os.makedirs(OUTDIR, exist_ok=True)

# ───────── Data loaders (BTC & Gold via Stooq; Gold fallback LBMA) ─────────
def _btc():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "BTC"  for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]  = pd.to_numeric(df["BTC"], errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")[["Date","BTC"]]

def _gold_stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")  # USD/oz
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date" in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    return df.dropna().sort_values("Date")[["Date","Gold"]]

def _gold_lbma_fallback():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df.dropna()[["Date","Gold"]].sort_values("Date")

def load_joined():
    btc = _btc()
    try:
        gold = _gold_stooq()
        if len(gold) < 400:
            raise ValueError("gold too short")
    except Exception:
        gold = _gold_lbma_fallback()
    return btc.merge(gold, on="Date", how="inner")

# ───────── Power-law helpers ─────────
def fit_power_usd(df):
    X = np.log10((df["Date"] - GENESIS).dt.days)
    y = np.log10(df["BTC"])
    m, b = np.polyfit(X, y, 1)
    sigma = np.std(y - (m*X + b))
    return m, b, sigma

def log_days(dates):  # for log-time charts
    return np.log10((pd.to_datetime(dates) - GENESIS).dt.days)

# ───────── Load data & fit USD power-law (anchored to 2030 target) ─────────
data = load_joined()
m, b, s = fit_power_usd(data)
b = np.log10(ANCHOR_PRICE_USD) - m * np.log10((ANCHOR_DATE - GENESIS).days)
sigma = max(s, 0.25)

# project monthly through 2040
future = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                       PROJ_END, freq="MS")
full = pd.concat([data, pd.DataFrame({"Date": future})], ignore_index=True)
days = (full["Date"] - GENESIS).dt.days
mid_usd = 10 ** (m * np.log10(days) + b)

levels = {
    "Support": -1.5,
    "Bear":    -0.5,
    "PL Best Fit": 0.0,
    "Frothy": +1.0,
    "Top":    +1.75,
}
for name, k in levels.items():
    full[name] = 10 ** (np.log10(mid_usd) + sigma * k)

# ───────── Chart 1: Power-law bands (USD) ─────────
fig1 = go.Figure()
for name,color,lev in [("Top","green","+1.75"),("Frothy","rgba(100,255,100,1)","+1.00")]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name], name=f"{name} ({lev}σ)",
                              line=dict(color=color, dash="dash")))
fig1.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"], name="PL Best Fit",
                          line=dict(color="white", dash="dash")))
for name,color,lev in [("Bear","rgba(255,100,100,1)","-0.50"),("Support","red","-1.50")]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name], name=f"{name} ({lev}σ)",
                              line=dict(color=color, dash="dash")))
fig1.add_trace(go.Scatter(x=data["Date"], y=data["BTC"], name="BTC",
                          line=dict(color="gold", width=2.5)))
fig1.update_layout(template="plotly_dark",
                   xaxis=dict(type="date", title="Year"),
                   yaxis=dict(type="log", title="Price (USD)", tickformat="$,d"),
                   plot_bgcolor="#111", paper_bgcolor="#111")
fig1.write_html(f"{OUTDIR}/bands_usd.html", include_plotlyjs="cdn", full_html=True)

# ───────── Chart 2: Dual-axis DMA (USD + BTC/Gold) with markers ─────────
dma = data[data["Date"] >= DMA_START].copy()
dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
dma["BTC_200"] = dma["BTC"].rolling(200).mean()
dma["BTCG"]    = dma["BTC"] / dma["Gold"]
dma["G50"]     = dma["BTCG"].rolling(50).mean()
dma["G200"]    = dma["BTCG"].rolling(200).mean()
dma = dma.dropna()

def cross_mask(series_long, series_short, lookback=100):
    diff = series_long - series_short
    return ((diff.shift(1) > 0) & (diff < 0) &
            (diff.shift(1).rolling(lookback).apply(lambda x: (x>0).all()).astype(bool)))

usd_cross = cross_mask(dma["BTC_200"], dma["BTC_50"], 100)
gld_cross = cross_mask(dma["G200"],    dma["G50"],    100)

fig2 = go.Figure()
# right axis (gold)
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"], name="200-DMA Gold",
                          line=dict(color="khaki", width=1.5), yaxis="y2"))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],  name="50-DMA Gold",
                          line=dict(color="goldenrod", width=1.5), yaxis="y2"))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"], name="BTC Gold",
                          line=dict(color="gold", width=2), yaxis="y2"))
fig2.add_trace(go.Scatter(x=dma.loc[gld_cross,"Date"], y=dma.loc[gld_cross,"BTCG"],
                          name="Top Marker (Gold)", mode="markers", yaxis="y2",
                          marker=dict(symbol="diamond", color="darkorange", size=9)))
# left axis (USD)
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"], name="200-DMA USD",
                          line=dict(color="lightgreen", width=1.5)))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],  name="50-DMA USD",
                          line=dict(color="mediumseagreen", width=1.5)))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],     name="BTC USD",
                          line=dict(color="limegreen", width=2)))
fig2.add_trace(go.Scatter(x=dma.loc[usd_cross,"Date"], y=dma.loc[usd_cross,"BTC"],
                          name="Top Marker (USD)", mode="markers",
                          marker=dict(symbol="diamond", color="forestgreen", size=9)))
fig2.update_layout(template="plotly_dark",
    xaxis =dict(type="date", title="Year", showgrid=True, gridwidth=0.5),
    yaxis =dict(type="log", title="USD",  tickformat="$,d", showgrid=True, gridwidth=0.5),
    yaxis2=dict(type="log", title="oz Gold", overlaying="y", side="right",
                tickvals=[0.01,0.1,1,10], ticktext=["0.01","0.1","1","10"],
                showgrid=True, gridwidth=0.5, gridcolor="rgba(255,255,255,0.2)"),
    plot_bgcolor="#111", paper_bgcolor="#111")
fig2.write_html(f"{OUTDIR}/dma_dual_axis.html", include_plotlyjs="cdn", full_html=True)

# ───────── Chart 3: Long-term power law (log-time, USD) ─────────
def year_ticks(end_year=2040):
    yrs = list(range(2012, min(2020, end_year)+1)) + list(range(2022, end_year+1, 2))
    if 2010 not in yrs: yrs = [2010] + yrs
    tv = log_days(pd.to_datetime([f"{y}-01-01" for y in yrs]))
    return tv, [str(y) for y in yrs]

x_hist = log_days(data["Date"])
x_full = log_days(full["Date"])
tickvals, ticktext = year_ticks(min(2040, PROJ_END.year))

fig3 = go.Figure()
for name,color,lev in [("Top","green","+1.75"),("Frothy","rgba(100,255,100,1)","+1.00")]:
    fig3.add_trace(go.Scatter(x=x_full, y=full[name], name=f"{name} ({lev}σ)",
                              line=dict(color=color, dash="dash")))
fig3.add_trace(go.Scatter(x=x_full, y=full["PL Best Fit"], name="PL Best Fit",
                          line=dict(color="white", dash="dash")))
for name,color,lev in [("Bear","rgba(255,100,100,1)","-0.50"),("Support","red","-1.50")]:
    fig3.add_trace(go.Scatter(x=x_full, y=full[name], name=f"{name} ({lev}σ)",
                              line=dict(color=color, dash="dash")))
fig3.add_trace(go.Scatter(x=x_hist, y=data["BTC"], name="BTC",
                          line=dict(color="gold", width=2.5)))
fig3.update_layout(template="plotly_dark",
    xaxis=dict(type="linear", title="Year (log time since 2009-01-03)",
               tickvals=tickvals, ticktext=ticktext, showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title="Price (USD, log)", tickformat="$,d",
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111")
fig3.write_html(f"{OUTDIR}/ltpl_usd.html", include_plotlyjs="cdn", full_html=True)

# ───────── Chart 4: Long-term power law (log-time, in oz Gold) ─────────
gold_hist = data.dropna(subset=["BTC","Gold"]).copy()
gold_hist["BTCG"] = gold_hist["BTC"]/gold_hist["Gold"]
Xg = np.log10((gold_hist["Date"] - GENESIS).dt.days); yg = np.log10(gold_hist["BTCG"])
mg, bg = np.polyfit(Xg, yg, 1); sg = np.std(yg - (mg*Xg + bg)); sg = max(sg, 0.25)
days_full = (full["Date"] - GENESIS).dt.days
mid_g = 10 ** (mg * np.log10(days_full) + bg)
g_levels = {f"{n} (Gold)": v for n,v in levels.items()}
for gname, v in g_levels.items():
    full[gname] = 10 ** (np.log10(mid_g) + sg * v)
x_hist_g = log_days(gold_hist["Date"]); x_full_g = log_days(full["Date"])
tickvals, ticktext = year_ticks(min(2040, PROJ_END.year))

fig4 = go.Figure()
for name,color,lev in [("Top","green","+1.75"),("Frothy","rgba(100,255,100,1)","+1.00")]:
    gname=f"{name} (Gold)"
    fig4.add_trace(go.Scatter(x=x_full_g, y=full[gname], name=f"{gname} ({lev}σ)",
                              line=dict(color=color, dash="dash")))
fig4.add_trace(go.Scatter(x=x_full_g, y=full["PL Best Fit (Gold)"], name="PL Best Fit (Gold)",
                          line=dict(color="white", dash="dash")))
for name,color,lev in [("Bear","rgba(255,100,100,1)","-0.50"),("Support","red","-1.50")]:
    gname=f"{name} (Gold)"
    fig4.add_trace(go.Scatter(x=x_full_g, y=full[gname], name=f"{gname} ({lev}σ)",
                              line=dict(color=colors[name] if (colors:={
                                  "Support":"red","Bear":"rgba(255,100,100,1)","Frothy":"rgba(100,255,100,1)","Top":"green"
                              }) else "red", dash="dash")))
fig4.add_trace(go.Scatter(x=x_hist_g, y=gold_hist["BTCG"], name="BTC (oz Gold)",
                          line=dict(color="darkorange", width=2.5)))
fig4.update_layout(template="plotly_dark",
    xaxis=dict(type="linear", title="Year (log time since 2009-01-03)",
               tickvals=tickvals, ticktext=ticktext, showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title="Price (oz Gold, log)", tickformat=",g",
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111")
fig4.write_html(f"{OUTDIR}/ltpl_gold.html", include_plotlyjs="cdn", full_html=True)

# ───────── Index page (links) ─────────
open(f"{OUTDIR}/index.html","w").write("""
<!doctype html><meta charset="utf-8">
<title>BTC Purchase Indicator</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{background:#111;color:#eee;font:16px/1.5 system-ui;margin:2rem}
 a{color:#8cf;text-decoration:none} a:hover{text-decoration:underline}
 .grid{display:grid;gap:1rem}
</style>
<h1>BTC Purchase Indicator</h1>
<div class="grid">
  <a href="bands_usd.html">Power-law bands (USD)</a>
  <a href="dma_dual_axis.html">DMA – USD &amp; BTC/Gold</a>
  <a href="ltpl_usd.html">Long-term power law (log-time, USD)</a>
  <a href="ltpl_gold.html">Long-term power law (log-time, in oz Gold)</a>
</div>
<p>Auto-built hourly by GitHub Actions.</p>
""")
print("Wrote static site to /public")
