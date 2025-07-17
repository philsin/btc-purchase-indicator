# streamlit_powerlaw_app.py — BTC Purchase Indicator
# --------------------------------------------------
import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

UA = {"User-Agent": "btc-pl-tool/1.0"}  # helps bypass basic API bot-filters
GENESIS = pd.Timestamp("2009‑01‑03")
SUPPLY = 21_000_000  # fully‑diluted

# ─────────────────────────────  DATA  ──────────────────────────────
def _coinmetrics() -> pd.DataFrame:
    url = (
        "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
        "?assets=btc&metrics=PriceUSD&frequency=1d&start_time=2010-01-01"
    )
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"time": "Date", "PriceUSD": "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Price"]]

def _coingecko() -> pd.DataFrame:
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        "?vs_currency=usd&days=max&interval=daily"
    )
    r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    data = r.json()["prices"]
    df = pd.DataFrame(data, columns=["ts", "Price"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["Date", "Price"]]

def _stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    date_col  = [c for c in df.columns if "date"  in c][0]
    price_col = [c for c in df.columns if "close" in c or "price" in c][0]
    df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").dropna()

@st.cache_data(ttl=3600, show_spinner="Downloading price history…")
def load_prices() -> pd.DataFrame:
    for fn in (_coinmetrics, _coingecko, _stooq):
        try:
            return fn()
        except Exception:
            continue
    st.error("All price sources failed — please try again later."); st.stop()

# ───────────────────────  POWER‑LAW FIT  ───────────────────────────
def days_since_genesis(ts): return (ts - GENESIS).days

def fit_power_law(df: pd.DataFrame):
    X = np.log10(df["days"].values)
    y = np.log10(df["Price"].values)
    B, A = np.polyfit(X, y, 1)        # y = A + B·log10(days)
    mid_log = A + B * X
    sigma   = np.std(y - mid_log)
    mid     = 10 ** mid_log
    return mid, mid_log, sigma

# ────────────────────────  STREAMLIT UI  ───────────────────────────
st.set_page_config(page_title="BTC Purchase Indicator", layout="wide")

df = load_prices()
df["days"] = df["Date"].apply(days_since_genesis)
mid, mid_log, sigma = fit_power_law(df)
df["mid"]      = mid
df["support1"] = 10 ** (mid_log - sigma)   # 1 σ below
df["resist1"]  = 10 ** (mid_log + sigma)   # 1 σ above

# ─── Sidebar controls ──────────────────────────────────────────────
sigma_mult = st.sidebar.slider("σ band width", 0.5, 2.5, 1.0, 0.25)
use_mcap   = st.sidebar.toggle("Show Market‑Cap")

# ─── Data selection (Price vs Market‑Cap) ──────────────────────────
if use_mcap:
    scale = SUPPLY
    y_title = "Market‑Cap (USD)"
    df_plot = df.assign(
        Price=df["Price"] * scale,
        mid   =df["mid"]   * scale,
        support=df["support1"]*scale,
        resist =df["resist1"] *scale,
    )
else:
    scale = 1.0
    y_title = "Price (USD)"
    df_plot = df.rename(columns={"support1": "support", "resist1": "resist"})

# ─── Evaluate current zone ─────────────────────────────────────────
support_now = df_plot["support"].iloc[-1] * sigma_mult
resist_now  = df_plot["resist"].iloc[-1]  * sigma_mult
price_now   = df_plot["Price"].iloc[-1]

zone = ("🟢 Value"   if price_now < support_now else
        "⚪ Neutral" if price_now <= resist_now else
        "🔴 Frothy")

st.markdown(f"### **Current zone:** {zone}")

# ─── Build Plotly figure ───────────────────────────────────────────
fig = go.Figure()
fig.update_layout(
    template="plotly_dark",
    font=dict(family="Currency, monospace", size=12),
    xaxis=dict(
        title="Year",
        type="date",
        showgrid=True,
        gridwidth=0.5,
        tick0="2010-01-01",
        dtick="M24",                     # grid every 24 months
    ),
    yaxis=dict(
        title=y_title, type="log", showgrid=True, gridwidth=0.5
    ),
    plot_bgcolor="#111", paper_bgcolor="#111",
)

fig.add_trace(go.Scatter(
    x=df_plot["Date"], y=df_plot["Price"],
    name="BTC", line=dict(color="gold", width=1.8),
))
fig.add_trace(go.Scatter(
    x=df_plot["Date"], y=df_plot["mid"],
    name="Mid‑line", line=dict(color="white", dash="dash"),
))
fig.add_trace(go.Scatter(
    x=df_plot["Date"], y=df_plot["support"]*sigma_mult,
    name="-σ", line=dict(color="green", dash="dash"),
))
fig.add_trace(go.Scatter(
    x=df_plot["Date"], y=df_plot["resist"]*sigma_mult,
    name="+σ", line=dict(color="red", dash="dash"),
))

# ─── Preserve zoom after slider change ─────────────────────────────
if "xrange" not in st.session_state:
    st.session_state["xrange"] = None

# Display the chart and capture events
sel = plotly_events(fig, override_height=600, key="pl_events")

# If user zooms or pans, store the new range
if sel and "xaxis.range[0]" in sel[0]:
    st.session_state["xrange"] = [
        sel[0]["xaxis.range[0]"], sel[0]["xaxis.range[1]"]
    ]

# After each rerun (e.g., slider move) re‑apply stored range
if st.session_state["xrange"]:
    fig.update_xaxes(range=st.session_state["xrange"])

st.plotly_chart(fig, use_container_width=True)
