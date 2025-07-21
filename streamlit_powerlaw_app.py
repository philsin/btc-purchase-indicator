# ─────────────────────────────────────────────────────────────
# streamlit_powerlaw_app.py  ·  BTC Purchase Indicator
#  ▸ power‑law bands  ▸ DMA chart with BTC/USD & BTC/Gold
# ─────────────────────────────────────────────────────────────
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA         = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS    = pd.Timestamp("2009-01-03")
FD_SUPPLY  = 21_000_000
GRID_D     = "M24"
PROJ_END   = pd.Timestamp("2040-12-31")
DMA_START  = pd.Timestamp("2012-04-01")

# ─── data loaders ────────────────────────────────────────────
def _btc_stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date"  for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "BTC"   for c in df.columns if "close" in c or "price" in c})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["BTC"]   = pd.to_numeric(df["BTC"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("BTC>0").sort_values("Date")

def _gold_stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=xauusd&i=d")  # LBMA USD/oz
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date" for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "Gold" for c in df.columns if "close" in c or "price" in c})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold"]  = pd.to_numeric(df["Gold"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().sort_values("Date")

def _gold_lbma():
    url = "https://raw.githubusercontent.com/koindata/gold-prices/master/data/gold.csv"
    df  = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"USD (PM)": "Gold"})
    return df[["Date", "Gold"]].dropna()

def load_prices():
    btc = _btc_stooq()
    try:
        gold = _gold_stooq()
        if len(gold) < 1000:
            raise ValueError
    except Exception:
        gold = _gold_lbma()
    return btc.merge(gold, on="Date", how="inner")

# ─── power‑law fit ───────────────────────────────────────────
def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days)
    y = np.log10(df["BTC"])
    slope, intercept = np.polyfit(X, y, 1)
    sigma            = np.std(y - (slope * X + intercept))
    return slope, intercept, sigma

# ─── Streamlit layout ────────────────────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

data = load_prices()
slope, intercept, σ = fit_power(data)

# anchor mid‑line
anchor_date  = pd.Timestamp("2030-01-01")
intercept    = np.log10(491_776) - slope * np.log10((anchor_date - GENESIS).days)

# extend timeline
future = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                       PROJ_END, freq="MS", inclusive="both")
full = pd.concat([data, pd.DataFrame({"Date": future})], ignore_index=True)
days = (full["Date"] - GENESIS).dt.days
mid_log = slope * np.log10(days) + intercept
σ_vis   = max(σ, 0.25)

levels = {
    "Support":     -1.5,
    "Bear":        -0.5,
    "PL Best Fit":  0.0,
    "Frothy":      +1.0,
    "Top":         +1.75,
}
colors = {
    "Support":     "red",
    "Bear":        "rgba(255,100,100,1)",
    "PL Best Fit": "white",
    "Frothy":      "rgba(100,255,100,1)",
    "Top":         "green",
}
for name, k in levels.items():
    full[name] = 10 ** (mid_log + σ_vis * k)

# Market‑Cap toggle
as_cap  = st.sidebar.toggle("Market‑Cap")
y_title = "Price (USD)"
if as_cap:
    cols = ["BTC", *levels.keys()]
    full[cols] = full[cols].fillna(method="ffill") * FD_SUPPLY
    data["BTC"] *= FD_SUPPLY
    y_title = "Market‑Cap (USD)"

# zone badge
row = full.dropna(subset=["BTC"]).iloc[-1]
p   = row["BTC"]
if p < row["Support"]:
    zone = "SELL THE HOUSE!!"
elif p < row["Bear"]:
    zone = "Undervalued"
elif p < row["Frothy"]:
    zone = "Fair Value"
elif p < row["Top"]:
    zone = "Overvalued"
else:
    zone = "TO THE MOON"
st.markdown(f"### **Current zone:** {zone}")

# ─── Chart 1: power‑law bands ───────────────────────────────
fig1 = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D, showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title=y_title, tickformat="$,d", showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

for name in ["Top", "Frothy"]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                              name=f"{name} ({levels[name]:+.2f}σ)".replace("+-","-"),
                              line=dict(color=colors[name], dash="dash")))
fig1.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                          name="PL Best Fit", line=dict(color="white", dash="dash")))
for name in ["Bear", "Support"]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                              name=f"{name} ({levels[name]:+.2f}σ)",
                              line=dict(color=colors[name], dash="dash")))
fig1.add_trace(go.Scatter(x=data["Date"], y=data["BTC"],
                          name="BTC", line=dict(color="gold", width=2)))

# keep zoom
if "xrange" in st.session_state:
    fig1.update_xaxes(range=st.session_state["xrange"])
st.plotly_chart(fig1, use_container_width=True)

ev = plotly_events(fig1, select_event=False, click_event=False, key="zoom")
if ev and "xaxis.range[0]" in ev[0]:
    st.session_state["xrange"] = [ev[0]["xaxis.range[0]"], ev[0]["xaxis.range[1]"]]

# ─── Chart 2: BTC/USD & BTC/Gold DMA ─────────────────────────
dma = data[data["Date"] >= DMA_START].copy()
dma["BTC_50"]  = dma["BTC"].rolling(50).mean()
dma["BTC_200"] = dma["BTC"].rolling(200).mean()
dma["BTCG"]    = dma["BTC"] / dma["Gold"]
dma["G50"]     = dma["BTCG"].rolling(50).mean()
dma["G200"]    = dma["BTCG"].rolling(200).mean()
dma = dma.dropna()

# crossover markers (USD)
diff = dma["BTC_200"] - dma["BTC_50"]
cross = (diff.shift(1) > 0) & (diff < 0) & (diff.shift(1).rolling(100).apply(lambda x:(x>0).all()).astype(bool))
cross_dates  = dma.loc[cross, "Date"]
cross_prices = dma.loc[cross, "BTC"]

fig2 = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),

    xaxis=dict(type="date", title="Year", dtick=GRID_D,
               showgrid=True, gridwidth=0.5),

    # left axis – USD
    yaxis=dict(type="log", title="BTC Price (USD)",
               tickformat="$,d",
               showgrid=True, gridwidth=0.5),

    # right axis – ounces of gold
    yaxis2=dict(type="log", title="BTC Price (oz Gold)",
                tickvals=[0.01, 0.1, 1, 10],         # choose the span you need
                ticktext=["0.01", "0.1", "1", "10"],
                overlaying="y", side="right",

                # draw its own grid so every label gets a line
                showgrid=True, gridwidth=0.5,
                gridcolor="rgba(255,255,255,0.2)"),

    plot_bgcolor="#111", paper_bgcolor="#111",
))

# Gold‑denominated traces
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G200"],
                          name="200‑DMA Gold", line=dict(color="lightsalmon", width=1.5), yaxis="y2"))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["G50"],
                          name="50‑DMA Gold",  line=dict(color="darkorange", width=1.5),  yaxis="y2"))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTCG"],
                          name="BTC Gold",     line=dict(color="gold", width=2),          yaxis="y2"))

# USD traces
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_200"],
                          name="200‑DMA USD", line=dict(color="mediumvioletred", width=1.5)))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC_50"],
                          name="50‑DMA USD",  line=dict(color="mediumorchid", width=1.5)))
fig2.add_trace(go.Scatter(x=dma["Date"], y=dma["BTC"],
                          name="BTC USD",     line=dict(color="purple", width=2)))
fig2.add_trace(go.Scatter(x=cross_dates, y=cross_prices,
                          name="Top Marker", mode="markers",
                          marker=dict(symbol="diamond", color="red", size=9)))


st.plotly_chart(fig2, use_container_width=True)
# ─────────────────────────────────────────────────────────────
