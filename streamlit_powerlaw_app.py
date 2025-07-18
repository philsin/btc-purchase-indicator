# ─────────────────────────────────────────────────────────────
# streamlit_powerlaw_app.py  ·  BTC Purchase Indicator
#  ▸ fixed bands  ▸ projection to 2040  ▸ pandas‑1.3 compatible
# ─────────────────────────────────────────────────────────────
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA         = {"User-Agent": "btc-pl-tool/1.0"}
GENESIS    = pd.Timestamp("2009-01-03")
FD_SUPPLY  = 21_000_000
GRID_D     = "M24"                      # vertical grid every 2 years
PROJ_END   = pd.Timestamp("2040-12-31")

# ─── data loaders (Stooq → GitHub fallback) ─────────────────
def _stooq():
    df = pd.read_csv("https://stooq.com/q/d/l/?s=btcusd&i=d")
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]
    df = df.rename(columns={date_col: "Date", price_col: "Price"})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str)
                                          .str.replace(",", ""), errors="coerce")
    return df.dropna().query("Price>0").sort_values("Date")

def _github():
    raw = ("https://raw.githubusercontent.com/datasets/bitcoin-price/"
           "master/data/bitcoin_price.csv")
    df = pd.read_csv(raw).rename(columns={"Closing Price (USD)": "Price"})
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

# anchor mid‑line at ≈ $492 k on 1‑Jan‑2030
anchor_date  = pd.Timestamp("2030-01-01")
anchor_days  = (anchor_date - GENESIS).days
intercept    = np.log10(491_776) - slope * np.log10(anchor_days)

# timeline until 2040
future = pd.date_range(
    hist["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
    PROJ_END,
    freq="MS",
    inclusive="both"          # pandas 1.3+ / 1.4+ compatible
)
full = pd.concat([hist, pd.DataFrame({"Date": future})], ignore_index=True)

days    = (full["Date"] - GENESIS).dt.days
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

# ─── figure ──────────────────────────────────────────────────
fig = go.Figure(layout=dict(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(type="date", title="Year", dtick=GRID_D,
               showgrid=True, gridwidth=0.5),
    yaxis=dict(type="log", title=y_title, tickformat="$,d",
               showgrid=True, gridwidth=0.5),
    plot_bgcolor="#111", paper_bgcolor="#111",
))

# legend / drawing order: Top ▸ Frothy ▸ PL ▸ Bear ▸ Support ▸ BTC
for name in ["Top  (+1.75σ)", "Frothy (+1.0σ)"]:
    fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                             name=name, line=dict(color=colors[name], dash="dash")))
fig.add_trace(go.Scatter(x=full["Date"], y=full["PL Best Fit"],
                         name="PL Best Fit", line=dict(color="white", dash="dash")))
for name in ["Bear (-0.5σ)", "Support (-1.5σ)"]:
    fig.add_trace(go.Scatter(x=full["Date"], y=full[name],
                             name=name, line=dict(color=colors[name], dash="dash")))
fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Price"],
                         name="BTC", line=dict(color="gold", width=3)))

# persistent zoom
if "xrange" in st.session_state:
    fig.update_xaxes(range=st.session_state["xrange"])

st.plotly_chart(fig, use_container_width=True)

ev = plotly_events(fig, select_event=False, click_event=False, key="zoom")
if ev and "xaxis.range[0]" in ev[0]:
    st.session_state["xrange"] = [ev[0]["xaxis.range[0]"],
                                  ev[0]["xaxis.range[1]"]]
# ─────────────────────────────────────────────────────────────
