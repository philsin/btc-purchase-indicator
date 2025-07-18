# ─────────────────────────────────────────────────────────────
# streamlit_powerlaw_app.py  ·  BTC Purchase Indicator
#  ▸ fixed bands  ▸ projection to 2040  ▸ 50‑DMA / 200‑DMA chart
# ─────────────────────────────────────────────────────────────
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA         = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS    = pd.Timestamp("2009-01-03")
FD_SUPPLY  = 21_000_000
GRID_D     = "M24"
PROJ_END   = pd.Timestamp("2040-12-31")

# ─── data loaders ────────────────────────────────────────────
def _stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "Date"  for c in df.columns if "date"  in c})
    df = df.rename(columns={c: "Price" for c in df.columns if "close" in c or "price" in c})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna().query("Price>0").sort_values("Date")

def _github():
    raw = "https://raw.githubusercontent.com/datasets/bitcoin-price/master/data/bitcoin_price.csv"
    df  = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

def get_price_history():
    try:
        df = _stooq()
        if len(df) > 1000:
            return df
    except Exception:
        pass
    return _github()

# ─── power‑law fit ───────────────────────────────────────────
def fit_power(df):
    X = np.log10((df["Date"] - GENESIS).dt.days)
    y = np.log10(df["Price"])
    slope, intercept = np.polyfit(X, y, 1)
    sigma            = np.std(y - (slope * X + intercept))
    return slope, intercept, sigma

# ─── Streamlit layout ────────────────────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

hist = get_price_history()
slope, intercept, σ = fit_power(hist)

# anchor mid‑line
anchor_date  = pd.Timestamp("2030-01-01")
intercept    = np.log10(491_776) - slope * np.log10((anchor_date - GENESIS).days)

# projection to 2040
future = pd.date_range(hist["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                       PROJ_END, freq="MS", inclusive="both")
full = pd.concat([hist, pd.DataFrame({"Date": future})], ignore_index=True)
days = (full["Date"] - GENESIS).dt.days
mid_log = slope * np.log10(days) + intercept
σ_vis = max(σ, 0.25)

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
    cols = ["Price", *levels.keys()]
    full[cols] = full[cols].fillna(method="ffill") * FD_SUPPLY
    y_title = "Market‑Cap (USD)"

# zone badge
row = full.dropna(subset=["Price"]).iloc[-1]
p   = row["Price"]
if p < row["Support"]:
    zone = "SELL THE HOUSE!!"
elif p < row["Bear"]:
    zone = "Undervalued"
elif p < row["Frothy"]:
    zone = "Fair"
elif p < row["Top"]:
    zone = "Overvalued"
else:
    zone = "TO THE MOON"
st.markdown(f"### **Current zone:** {zone}")

# ─── Chart 1: Power‑law bands ────────────────────────────────
fig1 = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D, showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title=y_title, tickformat="$,d", showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

for name in ["Top", "Frothy"]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                              name=f"{name} (+{levels[name]:.2f}σ)".replace("+-","-"),
                              line=dict(color=colors[name], dash="dash")))
fig1.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                          name="PL Best Fit", line=dict(color="white", dash="dash")))
for name in ["Bear", "Support"]:
    fig1.add_trace(go.Scatter(x=full["Date"], y=full[name],
                              name=f"{name} ({levels[name]:+.2f}σ)",
                              line=dict(color=colors[name], dash="dash")))
fig1.add_trace(go.Scatter(x=hist["Date"], y=hist["Price"],
                          name="BTC", line=dict(color="gold", width=2)))

if "xrange" in st.session_state:
    fig1.update_xaxes(range=st.session_state["xrange"])
st.plotly_chart(fig1, use_container_width=True)

# capture zoom for persistence
ev = plotly_events(fig1, select_event=False, click_event=False, key="zoom")
if ev and "xaxis.range[0]" in ev[0]:
    st.session_state["xrange"] = [ev[0]["xaxis.range[0]"], ev[0]["xaxis.range[1]"]]

# ─── Chart 2: BTC price + 50‑DMA & 200‑DMA ───────────────────
hist_ma = hist.copy()
hist_ma["50DMA"]  = hist_ma["Price"].rolling(window=50).mean()
hist_ma["200DMA"] = hist_ma["Price"].rolling(window=200).mean()

fig2 = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D, showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title="Price (USD)", tickformat="$,d",
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

fig2.add_trace(go.Scatter(x=hist_ma["Date"], y=hist_ma["Price"],
                          name="BTC", line=dict(color="gold", width=2.5)))
fig2.add_trace(go.Scatter(x=hist_ma["Date"], y=hist_ma["50DMA"],
                          name="50‑DMA", line=dict(color="royalblue", width=2)))
fig2.add_trace(go.Scatter(x=hist_ma["Date"], y=hist_ma["200DMA"],
                          name="200‑DMA", line=dict(color="purple", width=2)))

st.plotly_chart(fig2, use_container_width=True)
# ─────────────────────────────────────────────────────────────
